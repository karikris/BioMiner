"""Local, source-bound comparisons of experimental cosine evidence.

Large item joins and review queues remain inspectable Parquet artifacts. JSON
contains aggregate denominators only, and HTML uses local thumbnails with no
source URLs or external resources.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from html import escape
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

import polars as pl

from biominer.embeddinggemma.common import (
    atomic_json,
    atomic_parquet,
    exclusive_output,
    file_sha256,
    now,
)
from biominer.embeddinggemma.labels import label_fingerprint

_GROUPS = ("butterfly", "other_biological", "non_biological")
_OPTIONAL = {
    "image_hash": pl.String,
    "human_group": pl.String,
    "human_label": pl.String,
    "review_status": pl.String,
    "review_content_binding": pl.String,
    "cache_group": pl.String,
    "local_path": pl.String,
    "top1_label": pl.String,
    "leading_evidence_group": pl.String,
    "top1_score": pl.Float64,
    "top1_top2_margin": pl.Float64,
    "real_vs_other_margin": pl.Float64,
    "real_vs_depiction_margin": pl.Float64,
    "real_butterfly_score": pl.Float64,
    "depiction_score": pl.Float64,
    "flower_outranks_butterfly_on_flower": pl.Boolean,
}
_CANDIDATE_SCHEMA = {
    "configuration": pl.String,
    "item_id": pl.String,
    "image_hash": pl.String,
    "local_path": pl.String,
    "top1_label": pl.String,
    "top1_score": pl.Float64,
    "top1_top2_margin": pl.Float64,
    "real_vs_depiction_margin": pl.Float64,
    "tiny_subject_score": pl.Float64,
    "configuration_disagreement": pl.Boolean,
    "weak_content_rank": pl.Float64,
    "small_margin_rank": pl.Float64,
    "depiction_tie_rank": pl.Float64,
    "review_priority_rank": pl.Float64,
}


@exclusive_output()
def compare_configurations(configuration_dirs: list[Path], output_dir: Path) -> dict:
    """Compare configurations only after proving compatible source/runtime/labels.

    Different token budgets and output dimensions are intentional comparison
    axes. Different source snapshots, label banks, runtime identities, or named
    cohorts are rejected. Subsets within the same cohort retain unmatched counts.
    """
    if not configuration_dirs:
        raise ValueError("at least one scored configuration is required")
    output_dir = Path(output_dir).resolve()
    loaded = [_load(Path(path).resolve()) for path in configuration_dirs]
    names = [entry["manifest"]["configuration"] for entry in loaded]
    if len(names) != len(set(names)):
        raise ValueError("configuration names must be unique")
    if len(loaded) > 1:
        for left, right in combinations(loaded, 2):
            _compatible(left, right)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries, confusion_rows, candidate_rows = [], [], []
    all_predictions = defaultdict(set)
    for entry in loaded:
        for row in (
            entry["valid"]
            .select("item_id", "image_hash", "top1_label")
            .iter_rows(named=True)
        ):
            all_predictions[(row["item_id"], row["image_hash"])].add(row["top1_label"])
    for entry in loaded:
        summary, confusion = _summary(entry)
        summaries.append(summary)
        confusion_rows.extend(confusion)
        candidate_rows.extend(_candidates(entry, all_predictions))
    confusion = pl.from_dicts(
        confusion_rows,
        schema={
            "configuration": pl.String,
            "human_group": pl.String,
            "leading_evidence_group": pl.String,
            "count": pl.Int64,
        },
    ).sort(["configuration", "human_group", "leading_evidence_group"])
    candidates = pl.from_dicts(candidate_rows, schema=_CANDIDATE_SCHEMA)
    unresolved = candidates.sort(
        ["review_priority_rank", "configuration", "item_id"], nulls_last=True
    )
    tiny = candidates.filter(pl.col("tiny_subject_score").is_not_null()).sort(
        ["tiny_subject_score", "configuration", "item_id"],
        descending=[True, False, False],
    )
    artifacts = {
        "group_confusion": "group-confusion.parquet",
        "review_candidates": "review-candidates.parquet",
        "small_subject_candidates": "small-subject-candidates.parquet",
    }
    for name, frame in (
        ("group_confusion", confusion),
        ("review_candidates", unresolved),
        ("small_subject_candidates", tiny),
    ):
        atomic_parquet(output_dir / artifacts[name], frame)
    pairs = []
    for index, (left, right) in enumerate(combinations(loaded, 2)):
        pair, paired = _pair(left, right)
        pair["artifact"] = f"paired-{index:03d}.parquet"
        atomic_parquet(output_dir / pair["artifact"], paired)
        pairs.append(pair)
    result = {
        "schema_version": "embeddinggemma-comparison-v2",
        "created_at": now(),
        "configurations": summaries,
        "pairs": pairs,
        "artifacts": artifacts,
        "gallery_counts": {
            "small_subject_candidates": tiny.height,
            "unresolved_candidates": unresolved.height,
        },
        "fine_label_accuracy": None,
        "thresholds": None,
        "historical_content_binding": "unverified",
        "population_claim": "Balanced cache and previously reviewed subsets are not population estimates",
        "evidence_maturity": "experimental_model_evidence_not_species_or_occurrence_truth",
    }
    html = _html(result, loaded, confusion, tiny, unresolved, output_dir)
    temporary = output_dir / ".index.html.tmp"
    temporary.write_text(html)
    temporary.replace(output_dir / "index.html")
    atomic_json(output_dir / "comparison.json", result)
    return result


def _load(path: Path) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    for name, digest in manifest.get("outputs", {}).items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("scoring artifact path must remain inside configuration")
        if file_sha256(path / name) != digest:
            raise ValueError("Scoring artifact checksum mismatch")
    frame = pl.read_parquet(path / "rankings.parquet")
    if not {"item_id", "status"}.issubset(frame.columns):
        raise ValueError("rankings lack item identity or scoring status")
    if frame["item_id"].null_count() or frame["item_id"].n_unique() != frame.height:
        raise ValueError("rankings contain duplicate or missing item identities")
    scored_count = frame.filter(pl.col("status") == "ok").height
    if scored_count and "image_hash" not in frame.columns:
        raise ValueError("scored rankings require source image_hash for comparison")
    for column, dtype in _OPTIONAL.items():
        if column not in frame.columns:
            frame = frame.with_columns(pl.lit(None, dtype=dtype).alias(column))
        elif frame.schema[column] == pl.Null:
            frame = frame.with_columns(pl.col(column).cast(dtype))
    valid = frame.filter(pl.col("status") == "ok")
    if valid["image_hash"].null_count():
        raise ValueError("scored rankings require nonnull source image_hash")
    labels_path = path / "labels.parquet"
    labels = (
        pl.read_parquet(labels_path).sort("score_index")
        if labels_path.exists()
        else None
    )
    label_identity = (
        label_fingerprint(labels.select("label_id", "group", "text").to_dicts())
        if labels is not None
        else None
    )
    claimed_label_identity = manifest.get("label_bank_fingerprint")
    if (
        claimed_label_identity
        and label_identity
        and claimed_label_identity != label_identity
    ):
        raise ValueError("label bank fingerprint does not match artifact")
    return {
        "path": path,
        "manifest": manifest,
        "frame": frame,
        "valid": valid,
        "labels": labels,
        "label_identity": label_identity or claimed_label_identity,
    }


def _compatible(left: dict, right: dict) -> None:
    for entry in (left, right):
        provenance = entry["manifest"].get("provenance", {})
        if (
            not entry["label_identity"]
            or not provenance.get("comparison_runtime_fingerprint")
            or not provenance.get("input_manifest_sha256")
        ):
            raise ValueError(
                "compatible label-bank, runtime, and source fingerprints are required for comparisons"
            )
    a, b = (
        left["manifest"].get("provenance", {}),
        right["manifest"].get("provenance", {}),
    )
    if a.get("cohort") != b.get("cohort"):
        raise ValueError("different corpus cohorts cannot be compared together")
    if (
        left["label_identity"] != right["label_identity"]
        or a["comparison_runtime_fingerprint"] != b["comparison_runtime_fingerprint"]
        or a["input_manifest_sha256"] != b["input_manifest_sha256"]
    ):
        raise ValueError("configuration provenance is not compatible")
    if left["manifest"].get("dtype", "float32") != right["manifest"].get(
        "dtype", "float32"
    ):
        raise ValueError("configuration precision is not compatible")


def _counts(frame: pl.DataFrame, column: str) -> dict:
    return dict(
        sorted(
            Counter(
                str(value) if value is not None else "missing"
                for value in frame[column]
            ).items()
        )
    )


def _summary(entry: dict) -> tuple[dict, list[dict]]:
    frame, valid, manifest, path = (
        entry["frame"],
        entry["valid"],
        entry["manifest"],
        entry["path"],
    )
    decisive = valid.filter(
        (pl.col("review_status") == "decisive") & pl.col("human_group").is_not_null()
    )
    groups, confusion = [], []
    for group in _GROUPS:
        subset = decisive.filter(pl.col("human_group") == group)
        groups.append(
            {
                "human_group": group,
                "denominator": subset.height,
                "leading_group_matches": subset.filter(
                    pl.col("leading_evidence_group") == group
                ).height,
            }
        )
    for row in (
        decisive.group_by("human_group", "leading_evidence_group")
        .len(name="count")
        .iter_rows(named=True)
    ):
        confusion.append({"configuration": manifest["configuration"], **row})
    available = valid.filter(pl.col("real_vs_depiction_margin").is_not_null())
    axes = {
        "denominator": available.height,
        "real_outranks_depiction": available.filter(
            pl.col("real_vs_depiction_margin") > 0
        ).height,
        "depiction_outranks_real": available.filter(
            pl.col("real_vs_depiction_margin") < 0
        ).height,
        "equal_scores": available.filter(
            pl.col("real_vs_depiction_margin") == 0
        ).height,
        "mean_margin": _mean(available, "real_vs_depiction_margin"),
    }
    metrics = {
        **manifest.get("provenance", {}).get("metrics", {}),
        **manifest.get("metrics", {}),
    }
    storage = {}
    for kind in ("image", "text"):
        artifact = path / f"{kind}_embeddings.parquet"
        raw_bytes = metrics.get(f"raw_{kind}_embedding_bytes")
        if artifact.exists():
            vectors = pl.read_parquet(artifact, columns=["embedding"])
            # Persisted normalized float32 vectors, independent of compression.
            raw_bytes = (
                int(vectors.select(pl.col("embedding").list.len().sum()).item() or 0)
                * 4
            )
        storage[f"normalized_{kind}_vector_bytes"] = raw_bytes
        storage[f"{kind}_vector_parquet_bytes"] = (
            artifact.stat().st_size if artifact.exists() else None
        )
    storage["ranking_parquet_bytes"] = (path / "rankings.parquet").stat().st_size
    butterfly = decisive.filter(pl.col("human_group") == "butterfly")
    return {
        "configuration": manifest["configuration"],
        "budget": manifest["budget"],
        "dimension": manifest["dimension"],
        "score_fingerprint": manifest["fingerprint"],
        "cohort": manifest.get("provenance", {}).get("cohort"),
        "scored": valid.height,
        "unscored": frame.height - valid.height,
        "coverage": {
            "total": frame.height,
            "status_counts": _counts(frame, "status"),
            "review_status_counts": _counts(frame, "review_status"),
            "scored_decisive_review_denominator": decisive.height,
        },
        "groups": groups,
        "metrics": metrics,
        "storage": storage,
        "real_vs_depiction": axes,
        "flower_competition": butterfly.filter(
            pl.col("flower_outranks_butterfly_on_flower") == True
        ).height,
        "butterfly_denominator": butterfly.height,
    }, confusion


def _pair(left: dict, right: dict) -> tuple[dict, pl.DataFrame]:
    a, b = left["valid"], right["valid"]
    paired = a.join(b, on=["item_id", "image_hash"], suffix="_right")
    identities = a.select("item_id", "image_hash").join(
        b.select("item_id", "image_hash"), on="item_id", suffix="_right"
    )
    left_ids, right_ids = set(left["frame"]["item_id"]), set(right["frame"]["item_id"])
    return {
        "left": left["manifest"]["configuration"],
        "right": right["manifest"]["configuration"],
        "paired_scored": paired.height,
        "same_item_cohort": left_ids == right_ids,
        "left_only_items": len(left_ids - right_ids),
        "right_only_items": len(right_ids - left_ids),
        "common_items": len(left_ids & right_ids),
        "left_unmatched_scored": a.height - paired.height,
        "right_unmatched_scored": b.height - paired.height,
        "content_hash_mismatches": identities.filter(
            pl.col("image_hash") != pl.col("image_hash_right")
        ).height,
        "top1_changed": paired.filter(
            pl.col("top1_label") != pl.col("top1_label_right")
        ).height,
        "leading_group_changed": paired.filter(
            pl.col("leading_evidence_group") != pl.col("leading_evidence_group_right")
        ).height,
        "mean_butterfly_margin_delta": paired.select(
            (
                pl.col("real_vs_other_margin_right") - pl.col("real_vs_other_margin")
            ).mean()
        ).item()
        if paired.height
        else None,
    }, paired


def _candidates(entry: dict, predictions: dict) -> list[dict]:
    valid, labels = entry["valid"], entry["labels"]
    tiny_scores = {}
    if labels is not None:
        indices = labels.filter(pl.col("label_id") == "real_butterfly_tiny")[
            "score_index"
        ].to_list()
        if indices and (entry["path"] / "scores.parquet").exists():
            for row in pl.read_parquet(entry["path"] / "scores.parquet").iter_rows(
                named=True
            ):
                if len(row["scores"]) != labels.height:
                    raise ValueError("raw score width does not match label bank")
                tiny_scores[row["image_hash"]] = row["scores"][indices[0]]
    rows = valid.to_dicts()
    weak = _ordinal(rows, "top1_score")
    margin = _ordinal(rows, "top1_top2_margin")
    ties = _ordinal(rows, "real_vs_depiction_margin", absolute=True)
    result = []
    for row in rows:
        item_id = row["item_id"]
        disagreement = len(predictions[(item_id, row["image_hash"])]) > 1
        result.append(
            {
                "configuration": entry["manifest"]["configuration"],
                **{
                    key: row.get(key)
                    for key in (
                        "item_id",
                        "image_hash",
                        "local_path",
                        "top1_label",
                        "top1_score",
                        "top1_top2_margin",
                        "real_vs_depiction_margin",
                    )
                },
                "tiny_subject_score": tiny_scores.get(row["image_hash"]),
                "configuration_disagreement": disagreement,
                "weak_content_rank": weak[item_id],
                "small_margin_rank": margin[item_id],
                "depiction_tie_rank": ties[item_id],
                "review_priority_rank": weak[item_id]
                + margin[item_id]
                + ties[item_id]
                - int(disagreement),
            }
        )
    return result


def _ordinal(rows: list[dict], key: str, *, absolute: bool = False) -> dict:
    values = [
        (abs(row[key]) if absolute else row[key], row["item_id"])
        for row in rows
        if isinstance(row.get(key), (float, int)) and math.isfinite(row[key])
    ]
    ordered_values = sorted({value for value, _ in values})
    ranks = {
        value: index / max(1, len(ordered_values) - 1)
        for index, value in enumerate(ordered_values)
    }
    result = {row["item_id"]: 0.5 for row in rows}
    result.update({item_id: ranks[value] for value, item_id in values})
    return result


def _mean(frame: pl.DataFrame, column: str):
    return frame[column].mean() if frame.height else None


def _table(headers, rows) -> str:
    return (
        "<table><thead><tr>"
        + "".join("<th>" + escape(str(c)) + "</th>" for c in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(
                "<td>" + escape("unavailable" if c is None else str(c)) + "</td>"
                for c in row
            )
            + "</tr>"
            for row in rows
        )
        + "</tbody></table>"
    )


def _gallery(
    frame: pl.DataFrame, output_dir: Path, sources: dict, *, tiny: bool = False
) -> str:
    rendered = []
    for row in frame.head(24).iter_rows(named=True):
        image = ""
        local = row.get("local_path")
        if isinstance(local, str) and local and not urlsplit(local).scheme:
            path = Path(local)
            if not path.is_absolute():
                path = sources[row["configuration"]] / path
            if path.is_file():
                relative = escape(
                    os.path.relpath(path.resolve(), output_dir), quote=True
                )
                image = f'<img loading="lazy" src="{relative}" alt="Local review candidate">'
        score = row.get("tiny_subject_score") if tiny else row.get("top1_score")
        value = f"{score:.3f}" if isinstance(score, (float, int)) else "unavailable"
        rendered.append(
            "<figure>"
            + image
            + "<figcaption>"
            + escape(str(row["configuration"]))
            + " · "
            + escape(str(row["item_id"])[:22])
            + "<br>"
            + escape(str(row.get("top1_label") or "unavailable"))
            + "<br>cosine "
            + escape(value)
            + (
                " · configuration disagreement"
                if row.get("configuration_disagreement")
                else ""
            )
            + "</figcaption></figure>"
        )
    return "".join(rendered) or "<p>No scored candidates are available.</p>"


def _html(result, loaded, confusion, tiny, unresolved, output_dir) -> str:
    summaries = result["configurations"]
    coverage = _table(
        (
            "Configuration",
            "Cohort",
            "Total",
            "Scored",
            "Excluded/unscored",
            "Status counts",
        ),
        (
            (
                r["configuration"],
                r["cohort"],
                r["coverage"]["total"],
                r["scored"],
                r["unscored"],
                json.dumps(r["coverage"]["status_counts"], sort_keys=True),
            )
            for r in summaries
        ),
    )
    metrics = _table(
        (
            "Configuration",
            "Token budget",
            "Dimensions",
            "Images/s",
            "Inference seconds",
            "Peak RSS bytes",
            "MPS driver bytes",
        ),
        (
            (
                r["configuration"],
                r["budget"],
                r["dimension"],
                r["metrics"].get("images_per_second"),
                r["metrics"].get("inference_seconds"),
                r["metrics"].get("peak_rss_bytes"),
                r["metrics"].get(
                    "mps_driver_allocated_bytes", r["metrics"].get("mps_driver_bytes")
                ),
            )
            for r in summaries
        ),
    )
    storage = _table(
        (
            "Configuration",
            "Normalized image bytes",
            "Normalized text bytes",
            "Image Parquet bytes",
            "Text Parquet bytes",
        ),
        (
            (
                r["configuration"],
                r["storage"]["normalized_image_vector_bytes"],
                r["storage"]["normalized_text_vector_bytes"],
                r["storage"]["image_vector_parquet_bytes"],
                r["storage"]["text_vector_parquet_bytes"],
            )
            for r in summaries
        ),
    )
    agreement = _table(
        ("Configuration", "Human group", "Leading evidence group", "Count"),
        (
            (
                r["configuration"],
                r["human_group"],
                r["leading_evidence_group"],
                r["count"],
            )
            for r in confusion.iter_rows(named=True)
        ),
    )
    axes = _table(
        (
            "Configuration",
            "Axis denominator",
            "Real exceeds depiction",
            "Depiction exceeds real",
            "Equal",
            "Mean cosine margin",
        ),
        (
            (
                r["configuration"],
                r["real_vs_depiction"]["denominator"],
                r["real_vs_depiction"]["real_outranks_depiction"],
                r["real_vs_depiction"]["depiction_outranks_real"],
                r["real_vs_depiction"]["equal_scores"],
                r["real_vs_depiction"]["mean_margin"],
            )
            for r in summaries
        ),
    )
    pairs = _table(
        (
            "Left",
            "Right",
            "Same-content pairs",
            "Hash mismatches",
            "Left unmatched",
            "Right unmatched",
            "Top label changed",
        ),
        (
            (
                p["left"],
                p["right"],
                p["paired_scored"],
                p["content_hash_mismatches"],
                p["left_unmatched_scored"],
                p["right_unmatched_scored"],
                p["top1_changed"],
            )
            for p in result["pairs"]
        ),
    )
    sources = {e["manifest"]["configuration"]: e["path"] for e in loaded}
    return (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>EmbeddingGemma 2 comparison</title><style>body{font:16px system-ui;max-width:1200px;margin:32px auto;padding:16px;color:#193c32}"
        "table{border-collapse:collapse;display:block;overflow:auto}th,td{padding:9px;border:1px solid #ccd5d0;text-align:left}"
        ".gallery{display:flex;flex-wrap:wrap}figure{width:230px;margin:10px}img{width:230px;height:170px;object-fit:contain}figcaption{font-size:13px}</style>"
        "<h1>EmbeddingGemma 2 comparison</h1><p>Experimental cosine evidence: scores are not probabilities. "
        "Labels overlap. There are no rejection thresholds or automatic decisions.</p><p>Historical human reviews are not bound to the downloaded image bytes. "
        "Broad agreement is descriptive; fine-label accuracy and annotated tiny-subject performance are unavailable. "
        "Balanced cache, reviewed, and full-corpus cohorts require separate reports and do not establish population performance.</p>"
        "<h2>Coverage and exclusions</h2>"
        + coverage
        + "<h2>Budget, throughput, and memory</h2><p>Measurements describe the recorded runtime; missing measurements remain unavailable. Token budget and vector dimension comparisons are paired only on identical item and content hashes.</p>"
        + metrics
        + "<h2>Normalized vector storage</h2><p>Float32 vector payload and compressed Parquet sizes are reported separately.</p>"
        + storage
        + "<h2>Broad human group versus leading evidence group</h2><p>Decisive historical reviews only; these are overlapping model evidence groups.</p>"
        + agreement
        + "<h2>Real butterfly versus depiction evidence</h2><p>Positive margins indicate stronger real-butterfly cosine evidence; they do not prove presence or absence.</p>"
        + axes
        + "<h2>Paired configuration changes</h2>"
        + pairs
        + '<h2>Small-subject review candidates</h2><p>Ranked by the raw real_butterfly_tiny label cosine: candidate evidence, not annotated tiny-subject performance.</p><div class="gallery">'
        + _gallery(tiny, output_dir, sources, tiny=True)
        + "</div><h2>Independent unresolved review candidates</h2>"
        "<p>Ordinal review order combines weaker content evidence, smaller top-label margins, configuration disagreement, and nearer real-versus-depiction ties. "
        'These review cues define no automatic rejection rule.</p><div class="gallery">'
        + _gallery(unresolved, output_dir, sources)
        + '</div><p><a href="comparison.json">Comparison receipt</a>; complete paired joins, confusion counts, and candidate queues are adjacent Parquet artifacts.</p></html>'
    )
