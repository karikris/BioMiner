"""Cosine evidence over the whole vocabulary, with independent visual axes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.embeddinggemma.common import (
    atomic_json,
    atomic_parquet,
    exclusive_output,
    file_sha256,
    now,
    producer_sha,
)
from biominer.embeddinggemma.labels import evidence_family, label_fingerprint
from biominer.embeddinggemma.runtime import normalize_vectors

RANKING_SCHEMA = {
    **{
        key: pl.String
        for key in (
            "item_id",
            "image_hash",
            "status",
            "human_label",
            "human_group",
            "review_status",
            "review_content_binding",
            "cache_group",
            "local_path",
            "top1_label",
            "leading_evidence_group",
        )
    },
    **{
        key: pl.Float32
        for key in (
            "top1_score",
            "top1_top2_margin",
            "real_butterfly_score",
            "depiction_score",
            "other_biological_score",
            "non_biological_score",
            "real_vs_other_margin",
            "real_vs_depiction_margin",
            "larval_score",
            "pupal_score",
            "egg_score",
            "specimen_score",
            "detached_wing_score",
            "adult_live_score",
        )
    },
    "top5_labels": pl.List(pl.String),
    "top5_scores": pl.List(pl.Float32),
    "group_maxima": pl.List(
        pl.Struct({"group": pl.String, "label_id": pl.String, "score": pl.Float32})
    ),
    "flower_outranks_butterfly_on_flower": pl.Boolean,
}


@exclusive_output()
def score_configuration(
    items: pl.DataFrame,
    media: pl.DataFrame,
    embeddings: pl.DataFrame,
    labels: list[dict],
    text_vectors,
    output_dir: Path,
    *,
    budget: int,
    dimension: int,
    provenance: dict,
) -> dict:
    if budget not in {280, 1120}:
        raise ValueError("Unsupported vision token budget")
    text = normalize_vectors(text_vectors, dimension)
    if len(labels) != len(text) or not labels:
        raise ValueError("Label/vector count mismatch")
    output_dir = Path(output_dir)
    allowed = (
        set(media.filter(pl.col("status") == "ok")["image_hash"].to_list())
        if "image_hash" in media.columns
        else set()
    )
    embeddings = embeddings.filter(pl.col("image_hash").is_in(allowed)).sort(
        "image_hash"
    )
    if embeddings["image_hash"].n_unique() != embeddings.height:
        raise ValueError("Duplicate image embedding identities")
    image_vectors = (
        normalize_vectors(embeddings["embedding"].to_list(), dimension)
        if embeddings.height
        else np.empty((0, dimension), dtype=np.float32)
    )
    fingerprint = canonical_semantic_fingerprint(
        {
            "provenance": {
                key: value
                for key, value in provenance.items()
                if key not in {"metrics", "completed_batches"}
            },
            "dimension": dimension,
            "budget": budget,
            "labels": label_fingerprint(labels),
            "encoded_image_hashes": embeddings["image_hash"].to_list(),
            "image_vectors_sha256": hashlib.sha256(
                image_vectors.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "text_vectors_sha256": hashlib.sha256(
                text.astype("<f4", copy=False).tobytes()
            ).hexdigest(),
            "items": items.sort("item_id").to_dicts(),
            "media": media.select(
                [
                    column
                    for column in ("item_id", "status", "image_hash")
                    if column in media.columns
                ]
            )
            .sort("item_id")
            .to_dicts(),
            "scoring_policy": "overlapping-cosine-axes-v1",
        }
    )
    previous_manifest = output_dir / "manifest.json"
    if (
        previous_manifest.exists()
        and json.loads(previous_manifest.read_text())["fingerprint"] != fingerprint
    ):
        raise ValueError("Scoring fingerprint changed; select a new output directory")
    families = [evidence_family(label) for label in labels]
    content_indices = [i for i, family in enumerate(families) if family is not None]
    if not content_indices:
        raise ValueError("No content labels")
    groups = sorted({label["group"] for label in labels})
    group_indices = {
        group: [i for i, label in enumerate(labels) if label["group"] == group]
        for group in groups
    }
    family_indices = {
        family: [i for i, value in enumerate(families) if value == family]
        for family in ("butterfly", "other_biological", "non_biological")
    }
    depiction_indices = [
        i
        for i, label in enumerate(labels)
        if label["group"] == "Butterfly Depictions"
        or label["label_id"]
        in {
            "butterfly_on_screen_only",
            "butterfly_on_paper_only",
            "butterfly_on_product_only",
        }
    ]
    ids = [label["label_id"] for label in labels]
    evidence = []
    ranking_by_hash = {}
    for start in range(0, len(image_vectors), 256):
        block = image_vectors[start : start + 256] @ text.T
        for offset, scores in enumerate(block):
            image_hash = embeddings["image_hash"][start + offset]
            evidence.append({"image_hash": image_hash, "scores": scores.tolist()})
            order = sorted(content_indices, key=lambda i: (-float(scores[i]), ids[i]))
            best = order[0]
            family_scores = {
                family: max((float(scores[i]) for i in indices), default=None)
                for family, indices in family_indices.items()
            }
            real = family_scores["butterfly"]
            other = max(
                (
                    float(scores[i])
                    for i in content_indices
                    if families[i] != "butterfly"
                ),
                default=None,
            )
            depiction = max((float(scores[i]) for i in depiction_indices), default=None)
            group_maxima = [
                {
                    "group": group,
                    "label_id": ids[max(indices, key=lambda i: float(scores[i]))],
                    "score": max(float(scores[i]) for i in indices),
                }
                for group, indices in group_indices.items()
            ]
            flower = ids.index("flower_closeup") if "flower_closeup" in ids else None
            butterfly_flower = (
                ids.index("real_butterfly_on_flower")
                if "real_butterfly_on_flower" in ids
                else None
            )
            ranking_by_hash[image_hash] = {
                "top1_label": ids[best],
                "top1_score": float(scores[best]),
                "top5_labels": [ids[i] for i in order[:5]],
                "top5_scores": [float(scores[i]) for i in order[:5]],
                "top1_top2_margin": float(scores[best] - scores[order[1]])
                if len(order) > 1
                else None,
                "group_maxima": group_maxima,
                "leading_evidence_group": families[best],
                "real_butterfly_score": real,
                "depiction_score": depiction,
                "other_biological_score": family_scores["other_biological"],
                "adult_live_score": max(
                    (
                        float(scores[i])
                        for i in family_indices["butterfly"]
                        if ids[i] != "real_butterfly_specimen"
                    ),
                    default=None,
                ),
                "non_biological_score": family_scores["non_biological"],
                "real_vs_other_margin": real - other
                if real is not None and other is not None
                else None,
                "real_vs_depiction_margin": real - depiction
                if real is not None and depiction is not None
                else None,
                "flower_outranks_butterfly_on_flower": bool(
                    scores[flower] > scores[butterfly_flower]
                )
                if flower is not None and butterfly_flower is not None
                else None,
                **{
                    name: float(scores[ids.index(key)]) if key in ids else None
                    for name, key in (
                        ("larval_score", "real_butterfly_larva"),
                        ("pupal_score", "real_butterfly_pupa"),
                        ("egg_score", "real_butterfly_egg"),
                        ("specimen_score", "real_butterfly_specimen"),
                        ("detached_wing_score", "real_butterfly_wing"),
                    )
                },
            }
    media_by_id = {row["item_id"]: row for row in media.iter_rows(named=True)}
    rows = []
    for item in items.iter_rows(named=True):
        cached = media_by_id.get(item["item_id"], {})
        image_hash = cached.get("image_hash")
        ranked = ranking_by_hash.get(image_hash)
        status = "ok" if ranked else cached.get("status", "not_cached")
        if not ranked and status == "ok":
            status = "not_encoded"
        rows.append(
            {
                "item_id": item["item_id"],
                "image_hash": image_hash,
                "status": status,
                "human_label": item.get("human_label"),
                "human_group": item.get("human_group"),
                "review_status": item.get("review_status"),
                "review_content_binding": item.get(
                    "review_content_binding", "historical_content_unverified"
                ),
                "cache_group": item.get("cache_group"),
                "local_path": cached.get("local_path"),
                **(ranked or {}),
            }
        )
    rankings = (
        pl.from_dicts(rows, schema=RANKING_SCHEMA)
        if rows
        else pl.DataFrame(schema=RANKING_SCHEMA)
    )
    if "top1_score" in rankings.columns:
        # Ordinal review order, not an uncertainty probability or reject threshold.
        rankings = rankings.with_columns(
            (
                pl.col("top1_score").rank(method="average")
                + pl.col("top1_top2_margin").rank(method="average")
            ).alias("review_order_score")
        )
    scores_frame = pl.DataFrame(
        evidence, schema={"image_hash": pl.String, "scores": pl.List(pl.Float32)}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_parquet(output_dir / "scores.parquet", scores_frame)
    atomic_parquet(output_dir / "rankings.parquet", rankings)
    atomic_parquet(
        output_dir / "labels.parquet",
        pl.DataFrame(labels).with_row_index("score_index"),
    )
    atomic_parquet(
        output_dir / "image_embeddings.parquet",
        pl.DataFrame(
            {
                "image_hash": embeddings["image_hash"].to_list(),
                "embedding": image_vectors.tolist(),
            },
            schema={"image_hash": pl.String, "embedding": pl.List(pl.Float32)},
        ),
    )
    atomic_parquet(
        output_dir / "text_embeddings.parquet",
        pl.DataFrame(
            {"label_id": ids, "embedding": text.tolist()},
            schema={"label_id": pl.String, "embedding": pl.List(pl.Float32)},
        ),
    )
    manifest = {
        "schema_version": "embeddinggemma-scores-v1",
        "configuration": f"{budget}-{dimension}",
        "budget": budget,
        "dimension": dimension,
        "dtype": "float32",
        "created_at": now(),
        "evidence_maturity": "experimental_model_evidence",
        "producer_sha": producer_sha(),
        "fingerprint": fingerprint,
        "provenance": provenance,
        "scored": sum(row["status"] == "ok" for row in rows),
        "unscored": sum(row["status"] != "ok" for row in rows),
        "items": len(rows),
        "unique_embeddings": scores_frame.height,
        "labels": len(labels),
        "fine_label_accuracy": None,
        "fine_label_accuracy_reason": "No corresponding fine-grained human annotations",
        "label_bank_fingerprint": label_fingerprint(labels),
        "metrics": {
            **provenance.get("metrics", {}),
            "raw_image_embedding_bytes": embeddings.height * dimension * 4,
            "raw_text_embedding_bytes": len(labels) * dimension * 4,
            "image_embedding_parquet_bytes": (output_dir / "image_embeddings.parquet")
            .stat()
            .st_size,
            "text_embedding_parquet_bytes": (output_dir / "text_embeddings.parquet")
            .stat()
            .st_size,
        },
        "outputs": {
            name: file_sha256(output_dir / name)
            for name in (
                "scores.parquet",
                "rankings.parquet",
                "labels.parquet",
                "image_embeddings.parquet",
                "text_embeddings.parquet",
            )
        },
    }
    atomic_json(output_dir / "manifest.json", manifest)
    return manifest
