"""Freeze a local species-ranking cohort without promoting model evidence to truth."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

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
from biominer.embeddinggemma.media import read_cached_items
from biominer.embeddinggemma.workflow import read_prepared

CONFIGS = ("280-256", "1120-256", "280-768", "1120-768")


def species_labels(
    names: list[str], *, comparison_target: str | None = None
) -> tuple[dict, dict]:
    positions, duplicates, unique = {}, [], []
    for index, name in enumerate(names, 1):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Scientific-name descriptions must be nonempty strings")
        name = name.strip()
        if name in positions:
            duplicates.append(
                {"name": name, "position": index, "first_position": positions[name]}
            )
        else:
            positions[name] = index
            unique.append(name)
    if len(unique) < 3:
        raise ValueError("At least three unique custom labels are required")
    config = {
        "version": "butterfly-species-literal-v1",
        "mode": "species",
        "templates": [
            "a photo of {description}",
            "a close-up photo of {description}",
            "a photo of the butterfly {description}",
        ],
        "heads": {"species": {name: name for name in unique}},
    }
    if comparison_target is not None:
        if comparison_target not in unique:
            raise ValueError(
                "Historical comparison target must be one of the supplied names"
            )
        config["historical_comparison_target"] = comparison_target
    return config, {
        "requested_count": len(names),
        "unique_count": len(unique),
        "duplicates": duplicates,
        "policy": "first occurrence; supplied spellings preserved; no synonym reconciliation",
    }


def select_candidates(
    items: pl.DataFrame,
    predictions: dict[str, pl.DataFrame],
    associations: pl.DataFrame,
    *,
    target_scientific_name: str,
):
    if not target_scientific_name or not target_scientific_name.strip():
        raise ValueError("Target scientific name is required")
    adult_labels = {
        "adult_butterfly",
        "target_scientific_name:" + target_scientific_name,
    }
    if set(predictions) != set(CONFIGS):
        raise ValueError("All four configurations are required")
    if items["item_id"].n_unique() != items.height:
        raise ValueError("Duplicate input identities")
    reference = None
    consensus = None
    for config in CONFIGS:
        frame = predictions[config]
        if frame["item_id"].n_unique() != frame.height or set(frame["item_id"]) != set(
            items["item_id"]
        ):
            raise ValueError("Configuration item identities differ")
        identity = frame.select("item_id", "image_hash", "status").sort("item_id")
        if reference is not None and not reference.equals(identity):
            raise ValueError("Configuration content or status differs")
        reference = identity
        positive = set(
            frame.filter(
                (pl.col("status") == "ok")
                & (pl.col("leading_evidence_group") == "butterfly")
            )["item_id"]
        )
        consensus = positive if consensus is None else consensus & positive
    human = set(
        items.filter(
            (pl.col("review_status") == "decisive")
            & pl.col("human_label").is_in(adult_labels)
        )["item_id"]
    )
    unseen = (
        set(items.filter(pl.col("review_status") == "unreviewed")["item_id"])
        & consensus
    )
    chosen = items.filter(pl.col("item_id").is_in(human | unseen)).sort("item_id")
    if chosen.filter(pl.col("source_provider") != "flickr").height:
        raise ValueError("This local runner requires Flickr identities")
    links = associations.filter(pl.col("item_id").is_in(human | unseen))
    collections = defaultdict(set)
    for row in links.select("item_id", "input_path").iter_rows(named=True):
        collections[row["item_id"]].add(
            "geotagged"
            if Path(row["input_path"]).parent.name == "geotagged"
            else "original"
        )
    chosen = chosen.with_columns(
        pl.Series(
            "selection_reason",
            [
                "human_adult" if key in human else "unreviewed_four_way"
                for key in chosen["item_id"]
            ],
        ),
        pl.Series(
            "source_collections",
            [json.dumps(sorted(collections[key])) for key in chosen["item_id"]],
        ),
        pl.col("source_record_id").alias("flickr_photo_id"),
    )
    if (
        chosen["flickr_photo_id"].null_count()
        or chosen["flickr_photo_id"].n_unique() != chosen.height
    ):
        raise ValueError("Selected Flickr IDs must be nonnull and unique")
    return (
        chosen,
        links,
        {
            "human_adult": len(human),
            "unreviewed_four_way": len(unseen),
            "selected": chosen.height,
            "human_labels": dict(
                Counter(
                    chosen.filter(pl.col("selection_reason") == "human_adult")[
                        "human_label"
                    ]
                )
            ),
        },
    )


@exclusive_output()
def prepare_selection(
    benchmark_dir: Path,
    cache_dir: Path,
    names: list[str],
    output_dir: Path,
    *,
    target_scientific_name: str,
) -> dict:
    benchmark_dir, output_dir = Path(benchmark_dir), Path(output_dir)
    if json.loads((benchmark_dir / "run.json").read_text()).get("status") != "complete":
        raise ValueError("EmbeddingGemma benchmark is not complete")
    items = read_prepared(benchmark_dir / "prepared")
    associations = pl.read_parquet(benchmark_dir / "prepared/associations.parquet")
    frames, source_hashes = {}, {}
    for config in CONFIGS:
        directory = benchmark_dir / "scores/all" / config
        manifest = json.loads((directory / "manifest.json").read_text())
        path = directory / "rankings.parquet"
        if (
            manifest["configuration"] != config
            or file_sha256(path) != manifest["outputs"]["rankings.parquet"]
        ):
            raise ValueError("Configuration ranking checksum or identity differs")
        if manifest["provenance"]["input_manifest_sha256"] != file_sha256(
            benchmark_dir / "prepared/manifest.json"
        ):
            raise ValueError("Ranking source snapshot differs")
        frames[config] = pl.read_parquet(path)
        source_hashes[config] = file_sha256(directory / "manifest.json")
    selected, links, counts = select_candidates(
        items, frames, associations, target_scientific_name=target_scientific_name
    )
    config, label_audit = species_labels(
        names, comparison_target=target_scientific_name
    )
    identity = canonical_semantic_fingerprint(
        {
            "source": source_hashes,
            "labels": config,
            "requested_names": names,
            "selection": "human_adult_or_unreviewed_four_way_v1",
            "prepared": file_sha256(benchmark_dir / "prepared/manifest.json"),
        }
    )
    receipt_path = output_dir / "selection_manifest.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt["fingerprint"] != identity:
            raise ValueError(
                "Selection fingerprint changed; use a new output directory"
            )
        for name, digest in receipt["artifacts"].items():
            if file_sha256(output_dir / name) != digest:
                raise ValueError("Selection artifact checksum changed")
        return receipt
    if any(p.name != ".worker.lock" for p in output_dir.iterdir()):
        raise ValueError("Uncommitted selection output exists")
    media = read_cached_items(selected, cache_dir)
    if media.height != selected.height or media.filter(pl.col("status") != "ok").height:
        raise ValueError("Every selected image must have verified local cached bytes")
    binding = frames[CONFIGS[0]].select(
        "item_id", pl.col("image_hash").alias("eg_image_hash")
    )
    media = media.join(binding, on="item_id", validate="1:1")
    if media.filter(pl.col("image_hash") != pl.col("eg_image_hash")).height:
        raise ValueError("Cached bytes differ from the EmbeddingGemma evidence")
    selected = selected.join(
        media.select(
            "item_id",
            pl.col("local_path").alias("cached_image_path"),
            pl.col("image_hash").alias("cached_image_sha256"),
        ),
        on="item_id",
        validate="1:1",
    ).sort("item_id")
    atomic_parquet(output_dir / "selected_images.parquet", selected)
    atomic_parquet(output_dir / "associations.parquet", links)
    # Species descriptions retain the user's first-occurrence order on disk.
    temporary = output_dir / ".labels.json.tmp"
    with temporary.open("w") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(output_dir / "labels.json")
    atomic_json(output_dir / "label_audit.json", label_audit)
    receipt = {
        "schema_version": "biominer-species-selection-v1",
        "created_at": now(),
        "producer_sha": producer_sha(),
        "fingerprint": identity,
        "counts": counts,
        "labels": label_audit,
        "source_manifests": source_hashes,
        "unique_cached_content": media["image_hash"].n_unique(),
        "human_review_content_binding": "historical_content_unverified",
        "evidence_maturity": "experimental_species_suggestions_only",
        "source_snapshot": "saved benchmark; no live review refresh",
        "artifacts": {
            name: file_sha256(output_dir / name)
            for name in (
                "selected_images.parquet",
                "associations.parquet",
                "labels.json",
                "label_audit.json",
            )
        },
    }
    atomic_json(receipt_path, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-scientific-name", required=True)
    args = parser.parse_args()
    names = [
        part.strip()
        for part in args.names.read_text().replace("\n", ",").split(",")
        if part.strip()
    ]
    receipt = prepare_selection(
        args.benchmark_dir,
        args.cache_dir,
        names,
        args.output_dir,
        target_scientific_name=args.target_scientific_name,
    )
    print(
        json.dumps(
            {
                "counts": receipt["counts"],
                "labels": receipt["labels"],
                "fingerprint": receipt["fingerprint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
