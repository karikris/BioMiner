"""Prepare immutable, private benchmark inputs from historical review exports.

The projection intentionally retains discovery associations and sanitized review
outcomes, without carrying reviewer identity, free text, or unrelated source
metadata into benchmark artifacts. Historical labels are not bound to bytes
retrieved by this experiment and never constitute occurrence/species truth.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.embeddinggemma.common import exclusive_output
from biominer.storage.content_address import sha256_file

SCHEMA_VERSION = "biominer-embeddinggemma-prepare-v1"
_GROUPS = ("non_biological", "other_biological", "butterfly")
_BIOLOGICAL = {
    "caterpillar",
    "moth",
    "other_insect",
    "arachnid",
    "other_arthropod",
    "plant",
    "mammal_or_person",
    "bird",
    "other_animal",
    "fungus",
}
_NONBIOLOGICAL = {"artifact_or_illustration", "no_biological_subject"}
_POLICY = {
    "review_selection": "strict_majority_latest_round_with_reviews_no_fallback",
    "duplicate_review_evidence": "conflicting_latest_round_evidence_is_disputed",
    "target_mapping": "explicit_papilio_target_provenance_only_no_species_truth",
    "review_content_binding": "historical_content_unverified",
    "source_projection": "one_association_per_source_row_with_sanitized_review_counts",
    "sample": "seeded_semantic_rank_decisive_only_known_content_deduplicated",
    "unknown_content_duplicates": "must_be_audited_after_download",
    "storage_mode": "explicit_local_filesystem_benchmark",
}
_ITEM_SCHEMA = {
    "item_id": pl.String,
    "source_provider": pl.String,
    "source_record_id": pl.String,
    "image_url": pl.String,
    "fallback_url": pl.String,
    "human_label": pl.String,
    "review_status": pl.String,
    "human_group": pl.String,
    "review_content_binding": pl.String,
    "cache_group": pl.String,
    "review_round": pl.Int64,
    "review_count": pl.Int64,
    "review_conflict": pl.Boolean,
    "known_image_hash": pl.String,
    "association_count": pl.Int64,
}
_ASSOCIATION_SCHEMA = {
    "item_id": pl.String,
    "input_path": pl.String,
    "input_sha256": pl.String,
    "input_row_index": pl.Int64,
    "source_provider": pl.String,
    "source_record_id": pl.String,
    "source_record_hash": pl.String,
    "veritaxa_image_id": pl.String,
    "image_url": pl.String,
    "fallback_url": pl.String,
    "target_scientific_name": pl.String,
    "export_snapshot_at_utc": pl.String,
    "human_label": pl.String,
    "review_status": pl.String,
    "human_group": pl.String,
    "review_round": pl.Int64,
    "review_count": pl.Int64,
    "review_rounds_json": pl.String,
    "known_image_hash": pl.String,
}


@exclusive_output()
def prepare(
    inputs: list[Path],
    output_dir: Path,
    *,
    seed: str = "embeddinggemma-v1",
    sample_per_group: int = 500,
    sample_selection: Path | None = None,
) -> dict:
    """Create associations/items Parquet and publish ``manifest.json`` last.

    Inputs are Parquet exports, not a production source lineage. The same
    fingerprint reuses verified output; changed inputs or policies require a
    different output directory.
    """
    if not inputs:
        raise ValueError("at least one Parquet input is required")
    if not isinstance(seed, str) or not seed:
        raise ValueError("seed must be a nonempty string")
    if (
        isinstance(sample_per_group, bool)
        or not isinstance(sample_per_group, int)
        or sample_per_group < 0
    ):
        raise ValueError("sample_per_group must be a nonnegative integer")
    paths = sorted({Path(p).resolve() for p in inputs}, key=str)
    if len(paths) != len(inputs):
        raise ValueError("input paths must be distinct")
    for path in paths:
        if path.suffix.casefold() != ".parquet":
            raise ValueError("benchmark preparation accepts only Parquet inputs")
        if not path.is_file():
            raise ValueError("benchmark input does not exist")
    input_records = [
        {"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        for p in paths
    ]
    selection = None
    selection_record = None
    if sample_selection is not None:
        sample_selection = Path(sample_selection).resolve()
        selection_record = {
            "path": str(sample_selection),
            "sha256": sha256_file(sample_selection),
        }
        selection = pl.read_parquet(sample_selection).select(
            "item_id", "benchmark_group"
        )
    identity = {
        "schema_version": SCHEMA_VERSION,
        "inputs": input_records,
        "seed": seed,
        "sample_per_group": sample_per_group,
        "policy": _POLICY,
    }
    if selection_record is not None:
        identity["sample_selection"] = selection_record
    fingerprint = canonical_semantic_fingerprint(identity)
    output_dir = Path(output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return _reuse(manifest_path, fingerprint)
    if output_dir.exists() and any(
        path.name != ".worker.lock" for path in output_dir.iterdir()
    ):
        raise ValueError("uncommitted preparation output exists without a manifest")

    associations = []
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path, input_record in zip(paths, input_records, strict=True):
        frame = pl.read_parquet(path)
        input_record["rows"] = frame.height
        for row_index, row in enumerate(frame.iter_rows(named=True)):
            association = _association(row, path, input_record["sha256"], row_index)
            associations.append(association)
            grouped[association["item_id"]].append(association)
    items = [_item(item_id, rows) for item_id, rows in sorted(grouped.items())]
    sampling = (
        _planned_sample(items, selection)
        if selection is not None
        else _sample(items, seed, sample_per_group)
    )
    item_frame = pl.from_dicts(items, schema=_ITEM_SCHEMA)
    association_frame = pl.from_dicts(associations, schema=_ASSOCIATION_SCHEMA).sort(
        ["item_id", "input_path", "input_row_index"]
    )
    # Verify the physical inputs did not change while the tables were read.
    if any(
        sha256_file(path) != record["sha256"]
        for path, record in zip(paths, input_records, strict=True)
    ):
        raise ValueError("input fingerprint changed during preparation")
    if (
        selection_record is not None
        and sha256_file(sample_selection) != selection_record["sha256"]
    ):
        raise ValueError("sample selection changed during preparation")
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_parquet(output_dir / "associations.parquet", association_frame)
    _atomic_parquet(output_dir / "items.parquet", item_frame)
    artifact_frames = {
        "associations.parquet": association_frame,
        "items.parquet": item_frame,
    }
    if selection is not None:
        artifact_frames["benchmark-selection.parquet"] = selection.sort("item_id")
        _atomic_parquet(
            output_dir / "benchmark-selection.parquet",
            artifact_frames["benchmark-selection.parquet"],
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "fingerprint": fingerprint,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "producer_git_sha": _producer_sha(),
        "evidence_maturity": "experimental_historical_review_not_occurrence_truth",
        "inputs": input_records,
        "seed": seed,
        "sample_per_group": sample_per_group,
        "policy": _POLICY,
        "counts": {
            "associations": len(associations),
            "items": len(items),
            "duplicate_associations": len(associations) - len(items),
            "review_status": dict(
                sorted(Counter(r["review_status"] for r in items).items())
            ),
            "human_group": dict(
                sorted(
                    Counter(r["human_group"] for r in items if r["human_group"]).items()
                )
            ),
            "review_conflicts": sum(r["review_conflict"] for r in items),
        },
        "cache_sampling": sampling,
        "artifacts": {
            name: {
                "path": name,
                "sha256": sha256_file(output_dir / name),
                "bytes": (output_dir / name).stat().st_size,
                "rows": frame.height,
            }
            for name, frame in artifact_frames.items()
        },
    }
    if selection_record is not None:
        receipt["sample_selection"] = selection_record
    _atomic_json(manifest_path, receipt)
    return receipt


def _association(row: dict, path: Path, source_hash: str, row_index: int) -> dict:
    provider = (
        _text(row.get("source_provider")) or _text(row.get("source")) or "unknown"
    ).casefold()
    source_id = _text(row.get("source_record_id"))
    if provider == "flickr":
        source_id = _text(row.get("flickr_photo_id")) or source_id
    source_id = (
        source_id or _text(row.get("image_id")) or _text(row.get("veritaxa_image_id"))
    )
    url = _canonical_url(
        _text(row.get("image_url")) or _text(row.get("display_url")) or ""
    )
    fallback = _text(row.get("fallback_url")) or _text(row.get("display_url"))
    fallback = _canonical_url(fallback) if fallback else None
    if fallback == url:
        fallback = None
    if not source_id and not url:
        raise ValueError("source row has neither a stable identity nor an image URL")
    identity = (
        {"provider": provider, "source_record_id": source_id}
        if source_id
        else {"canonical_image_url": url}
    )
    item_id = "image:" + canonical_semantic_fingerprint(identity)
    rounds = [_round(row, "first_round", 1), _round(row, "second_round", 2)]
    latest = next((r for r in reversed(rounds) if r["review_count"] > 0), None)
    target = _text(row.get("target_scientific_name"))
    review = _decision(latest, target)
    known_hash = next(
        (
            value
            for key in (
                "image_hash",
                "image_sha256",
                "content_hash",
                "content_sha256",
                "media_sha256",
                "source_image_sha256",
            )
            if (value := _content_hash(row.get(key))) is not None
        ),
        None,
    )
    return {
        "item_id": item_id,
        "input_path": str(path),
        "input_sha256": source_hash,
        "input_row_index": row_index,
        "source_provider": provider,
        "source_record_id": source_id,
        "source_record_hash": _text(row.get("source_record_hash")),
        "veritaxa_image_id": _text(row.get("veritaxa_image_id")),
        "image_url": url,
        "fallback_url": fallback,
        "target_scientific_name": target,
        "export_snapshot_at_utc": _text(row.get("export_snapshot_at_utc")),
        **review,
        "review_rounds_json": json.dumps(rounds, sort_keys=True, separators=(",", ":")),
        "known_image_hash": known_hash,
    }


def _round(row: dict, prefix: str, number: int) -> dict:
    raw_reviews = _json(row.get(prefix + "_reviews_json"), [])
    raw_counts = _json(row.get(prefix + "_result_counts_json"), {})
    if not isinstance(raw_reviews, list) or not isinstance(raw_counts, dict):
        raise ValueError("review evidence has an incompatible JSON shape")  # noqa: TRY004
    counts: Counter[str] = Counter()
    for review in raw_reviews:
        if not isinstance(review, dict):
            raise ValueError("review evidence must contain review objects")  # noqa: TRY004
        result = (
            _text(review.get("result"))
            or _text(review.get("human_label"))
            or _text(review.get("label"))
        )
        if result == "target_scientific_name":
            scientific_name = _text(review.get("scientificName")) or _text(
                review.get("target_scientific_name")
            )
            if scientific_name:
                result += ":" + scientific_name
        if result:
            counts[result] += 1
    declared = row.get(prefix + "_review_count") or 0
    try:
        declared = int(declared)
    except TypeError, ValueError:
        raise ValueError("review count must be a nonnegative integer") from None
    if declared < 0:
        raise ValueError("review count must be a nonnegative integer")
    summary_counts = {}
    for label, count in raw_counts.items():
        if (
            not isinstance(label, str)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 0
        ):
            raise ValueError("review result counts must be nonnegative integers")
        if count:
            summary_counts[label] = count
    inconsistent = bool(counts and summary_counts and dict(counts) != summary_counts)
    if not raw_reviews:
        counts.update(summary_counts)
    total = max(declared, len(raw_reviews), sum(counts.values()))
    return {
        "round": number,
        "review_count": total,
        "result_counts": dict(sorted(counts.items())),
        "inconsistent": inconsistent,
    }


def _decision(evidence: dict | None, target: str | None) -> dict:
    if evidence is None:
        return {
            "human_label": None,
            "review_status": "unreviewed",
            "human_group": None,
            "review_round": 0,
            "review_count": 0,
        }
    counts = evidence["result_counts"]
    total = evidence["review_count"]
    winners = [label for label, count in counts.items() if count * 2 > total]
    label = winners[0] if len(winners) == 1 and not evidence["inconsistent"] else None
    if label is None:
        status = "disputed" if counts else "uncertain"
    elif label == "uncertain":
        status = "uncertain"
    elif label == "image_unavailable":
        status = "unavailable"
    else:
        status = "decisive"
    group = _group(label, target) if status == "decisive" else None
    return {
        "human_label": label,
        "review_status": status,
        "human_group": group,
        "review_round": evidence["round"],
        "review_count": total,
    }


def _group(label: str | None, target: str | None) -> str | None:
    if label in _NONBIOLOGICAL:
        return "non_biological"
    if label in _BIOLOGICAL:
        return "other_biological"
    if label == "adult_butterfly":
        return "butterfly"
    if label and label.split(":", 1)[0] == "target_scientific_name":
        qualified_target = label.partition(":")[2].strip() or target
        if qualified_target and re.match(
            r"^Papilio\s+[a-z][a-z-]*(?:\s|$)", qualified_target
        ):
            if (
                target
                and label.partition(":")[2]
                and qualified_target.casefold() != target.casefold()
            ):
                return None
            return "butterfly"
    return None


def _item(item_id: str, rows: list[dict]) -> dict:
    latest_number = max(r["review_round"] for r in rows)
    latest = [r for r in rows if r["review_round"] == latest_number]
    evidence = {
        canonical_semantic_fingerprint(
            {
                "decision": {
                    k: r[k] for k in ("human_label", "review_status", "human_group")
                },
                "round_evidence": json.loads(r["review_rounds_json"])[latest_number - 1]
                if latest_number
                else None,
            }
        )
        for r in latest
    }
    conflict = len(evidence) > 1
    chosen = min(latest, key=lambda r: (r["input_path"], r["input_row_index"]))
    urls = sorted({r["image_url"] for r in rows if r["image_url"]})
    fallbacks = sorted(
        {r["fallback_url"] for r in rows if r["fallback_url"]} | set(urls[1:])
    )
    hashes = {r["known_image_hash"] for r in rows if r["known_image_hash"]}
    # Different claimed byte identities for one source identity also fail closed.
    conflict = conflict or len(hashes) > 1
    return {
        "item_id": item_id,
        "source_provider": chosen["source_provider"],
        "source_record_id": chosen["source_record_id"],
        "image_url": urls[0] if urls else "",
        "fallback_url": fallbacks[0] if fallbacks else None,
        "human_label": None if conflict else chosen["human_label"],
        "review_status": "disputed" if conflict else chosen["review_status"],
        "human_group": None if conflict else chosen["human_group"],
        "review_content_binding": "historical_content_unverified",
        "cache_group": None,
        "review_round": latest_number,
        "review_count": max(r["review_count"] for r in latest),
        "review_conflict": conflict,
        "known_image_hash": next(iter(hashes)) if len(hashes) == 1 else None,
        "association_count": len(rows),
    }


def _sample(items: list[dict], seed: str, requested: int) -> dict:
    candidates: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        if item["review_status"] == "decisive" and item["human_group"] in _GROUPS:
            candidates[item["human_group"]].append(item)
    by_content: dict[str, set[str]] = defaultdict(set)
    for group, rows in candidates.items():
        for item in rows:
            if item["known_image_hash"]:
                by_content[item["known_image_hash"]].add(group)
    conflicting_hashes = {key for key, groups in by_content.items() if len(groups) > 1}
    groups = {}
    for group in _GROUPS:
        ranked = sorted(
            candidates[group],
            key=lambda r: canonical_semantic_fingerprint(
                {"seed": seed, "group": group, "item_id": r["item_id"]}
            ),
        )
        seen_hashes = set()
        eligible = []
        for item in ranked:
            image_hash = item["known_image_hash"]
            if image_hash and (
                image_hash in seen_hashes or image_hash in conflicting_hashes
            ):
                continue
            if image_hash:
                seen_hashes.add(image_hash)
            eligible.append(item)
        selected = eligible[:requested]
        for item in selected:
            item["cache_group"] = group
        groups[group] = {
            "decisive_candidates": len(ranked),
            "unique_known_content_candidates": len(eligible),
            "selected": len(selected),
            "requested": requested,
            "shortfall": max(0, requested - len(selected)),
        }
    return {
        "groups": groups,
        "known_cross_group_content_conflicts": len(conflicting_hashes),
        "unknown_content_duplicates": "not_yet_audited",
    }


def _planned_sample(items: list[dict], selection: pl.DataFrame) -> dict:
    """Use an explicit, human-labelled, content-distinct benchmark selection."""
    permitted = {"recent_butterfly", "earlier_butterfly", "other_categories"}
    if (
        selection["item_id"].null_count()
        or selection["item_id"].n_unique() != selection.height
        or not set(selection["benchmark_group"]).issubset(permitted)
    ):
        raise ValueError("Invalid or duplicate benchmark sample identities/groups")
    by_id = {item["item_id"]: item for item in items}
    content_groups: dict[str, set[str]] = defaultdict(set)
    for item in items:
        if item["known_image_hash"] and item["human_group"]:
            content_groups[item["known_image_hash"]].add(item["human_group"])
    seen = set()
    counts: Counter[str] = Counter()
    for item_id, group in selection.iter_rows():
        item = by_id.get(item_id)
        if item is None:
            raise ValueError("Benchmark sample is outside the prepared image corpus")
        if item["review_status"] != "decisive" or item["human_group"] not in _GROUPS:
            raise ValueError(
                "Benchmark samples require decisive human content evidence"
            )
        if (group in {"recent_butterfly", "earlier_butterfly"}) != (
            item["human_group"] == "butterfly"
        ):
            raise ValueError("Benchmark subset contradicts its human content evidence")
        image_hash = item["known_image_hash"]
        if not image_hash or image_hash in seen or len(content_groups[image_hash]) > 1:
            raise ValueError("Benchmark samples require distinct known image content")
        seen.add(image_hash)
        item["cache_group"] = item["human_group"]
        counts[item["human_group"]] += 1
    return {
        "policy": "explicit_human_labelled_content_distinct_selection",
        "groups": {
            group: {
                "selected": counts[group],
                "requested": counts[group],
                "shortfall": 0,
            }
            for group in _GROUPS
        },
        "benchmark_groups": dict(sorted(Counter(selection["benchmark_group"]).items())),
        "known_cross_group_content_conflicts": 0,
        "unknown_content_duplicates": "none_in_selected_known_content",
    }


def _reuse(path: Path, fingerprint: str) -> dict:
    receipt = json.loads(path.read_text())
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("fingerprint") != fingerprint
    ):
        raise ValueError(
            "existing preparation fingerprint does not match inputs or policy"
        )
    names = ["items.parquet", "associations.parquet"]
    if receipt.get("sample_selection"):
        names.append("benchmark-selection.parquet")
    for name in names:
        artifact = receipt.get("artifacts", {}).get(name, {})
        artifact_path = path.parent / name
        if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get(
            "sha256"
        ):
            raise ValueError("prepared artifact checksum does not match manifest")
    return receipt


def _content_hash(value) -> str | None:
    value = _text(value)
    if value is None:
        return None
    value = value.casefold().removeprefix("sha256:")
    return "sha256:" + value if re.fullmatch(r"[0-9a-f]{64}", value) else None


def _text(value) -> str | None:
    if value is None:
        return None
    return str(value).strip() or None


def _json(value, default):
    if value is None or value == "":
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except ValueError, TypeError:
        raise ValueError("review evidence contains invalid JSON") from None


def _canonical_url(value: str) -> str:
    if not value:
        return ""
    try:
        parts = urlsplit(value)
        if (
            parts.scheme.casefold() not in ("https", "http")
            or not parts.netloc
            or parts.username
            or parts.password
        ):
            raise ValueError
        return urlunsplit(
            (
                parts.scheme.casefold(),
                parts.netloc.casefold(),
                parts.path,
                parts.query,
                "",
            )
        )
    except ValueError:
        raise ValueError(
            "source image URL must be an HTTP(S) URL without credentials"
        ) from None


def _atomic_parquet(path: Path, frame: pl.DataFrame) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.write_parquet(temporary, compression="zstd")
        with open(temporary, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _producer_sha() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"
