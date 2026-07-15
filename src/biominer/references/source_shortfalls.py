from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import polars as pl


REFERENCE_SOURCE_SHORTFALL_SCHEMA_VERSION = "reference-source-shortfalls-v1.0.0"
REFERENCE_SOURCE_SHORTFALL_FILE = "reference_source_shortfalls.json"
REFERENCE_SOURCE_SHORTFALL_MARKDOWN_FILE = "reference_source_shortfalls.md"

_ELIGIBLE_LICENCE_STATUSES = frozenset(
    {"allowed", "research_only", "unreviewed"}
)
_ELIGIBLE_DOWNLOAD_STATUSES = frozenset({"pending", "complete"})


def compile_reference_source_shortfalls(
    *,
    query_plan: Mapping[str, object],
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    created_at: str | datetime | None = None,
    query_plan_path: str | Path | None = None,
) -> dict[str, object]:
    quotas = query_plan.get("acquisition_quotas")
    if not isinstance(quotas, Mapping) or not quotas:
        raise ValueError("reference source query plan has no acquisition_quotas")
    query_rows = query_plan.get("queries")
    if not isinstance(query_rows, list) or not query_rows:
        raise ValueError("reference source query plan has no queries")
    query_keys = {
        _accepted_taxon_key(row.get("accepted_taxon_key"))
        for row in query_rows
        if isinstance(row, Mapping)
    }
    if len(query_keys) != len(query_rows):
        raise ValueError("reference source query plan contains duplicate or invalid taxa")
    candidate_counts, adult_counts, larval_counts = _eligible_media_counts(
        observations,
        media_candidates,
    )
    group_rows: list[dict[str, object]] = []
    for name, raw_quota in quotas.items():
        if not isinstance(raw_quota, Mapping):
            raise ValueError(f"acquisition quota {name!r} must be an object")
        species = tuple(
            _accepted_taxon_key(value)
            for value in _array(raw_quota.get("species"), field=f"{name}.species")
        )
        unknown = sorted(set(species) - query_keys)
        if unknown:
            raise ValueError(
                f"acquisition quota {name!r} references unqueried taxa: "
                + ", ".join(unknown)
            )
        life_stage = str(raw_quota.get("life_stage") or "").strip()
        if life_stage == "adult":
            counts = adult_counts
        elif life_stage == "larva":
            counts = larval_counts
        elif life_stage:
            raise ValueError(
                f"acquisition quota {name!r} has unsupported life_stage {life_stage!r}"
            )
        else:
            counts = candidate_counts
        reviewed: dict[str, int] = {}
        per_species = [
            {
                "accepted_taxon_key": key,
                "source_candidate_media_count": counts.get(key, 0),
                "human_verified_media_count": reviewed.get(key, 0),
            }
            for key in species
        ]
        minimum_per_species = _optional_nonnegative_integer(
            raw_quota.get("minimum_per_species"),
            field=f"{name}.minimum_per_species",
        )
        minimum_total = _optional_nonnegative_integer(
            raw_quota.get("minimum_total"),
            field=f"{name}.minimum_total",
        )
        if minimum_per_species is not None and minimum_total is not None:
            raise ValueError(
                f"acquisition quota {name!r} cannot define both minimum forms"
            )
        if minimum_per_species is not None:
            required = minimum_per_species * len(species)
            candidate_shortfall = sum(
                max(0, minimum_per_species - counts.get(key, 0))
                for key in species
            )
            verified_shortfall = sum(
                max(0, minimum_per_species - reviewed.get(key, 0))
                for key in species
            )
        else:
            required = minimum_total or 0
            candidate_shortfall = max(
                0,
                required - sum(counts.get(key, 0) for key in species),
            )
            verified_shortfall = max(
                0,
                required - sum(reviewed.get(key, 0) for key in species),
            )
        planned_status = str(raw_quota.get("status") or "").strip()
        if planned_status.startswith("unresolved"):
            status = planned_status
        elif candidate_shortfall:
            status = "source_candidate_shortfall"
        elif verified_shortfall:
            status = "awaiting_human_review"
        else:
            status = "minimum_human_verified_quota_met"
        group_rows.append(
            {
                "group": str(name),
                "status": status,
                "planned_status": planned_status or None,
                "species_count": len(species),
                "species": list(species),
                "classes": [
                    str(value)
                    for value in _array(
                        raw_quota.get("classes"),
                        field=f"{name}.classes",
                    )
                ],
                "life_stage": life_stage or None,
                "minimum_required": required,
                "source_candidate_media_count": sum(
                    counts.get(key, 0) for key in species
                ),
                "human_verified_media_count": sum(
                    reviewed.get(key, 0) for key in species
                ),
                "source_candidate_shortfall": candidate_shortfall,
                "human_verified_shortfall": verified_shortfall,
                "per_species": per_species,
                "separate_from_adult_bank": bool(
                    raw_quota.get("separate_from_adult_bank", False)
                ),
            }
        )
    created = _utc_datetime(created_at or datetime.now(UTC))
    return {
        "schema_version": REFERENCE_SOURCE_SHORTFALL_SCHEMA_VERSION,
        "status": "awaiting_human_review_or_additional_sources"
        if any(row["status"] != "minimum_human_verified_quota_met" for row in group_rows)
        else "complete",
        "candidate_semantics": str(query_plan.get("candidate_semantics") or ""),
        "created_at": created.isoformat().replace("+00:00", "Z"),
        "query_plan": str(query_plan_path) if query_plan_path is not None else None,
        "query_plan_sha256": (
            _sha256_file(Path(query_plan_path)) if query_plan_path is not None else None
        ),
        "query_count": len(query_rows),
        "queried_species_count": len(query_keys),
        "source_observation_count": observations.height,
        "source_media_candidate_count": media_candidates.height,
        "eligible_source_media_candidate_count": sum(candidate_counts.values()),
        "human_review_status": "not_available_at_metadata_stage",
        "human_verified_source_media_count": 0,
        "groups": group_rows,
        "summary": {
            "group_count": len(group_rows),
            "groups_awaiting_human_review": sum(
                row["status"] == "awaiting_human_review" for row in group_rows
            ),
            "groups_with_source_candidate_shortfall": sum(
                row["status"] == "source_candidate_shortfall" for row in group_rows
            ),
            "unresolved_group_count": sum(
                str(row["status"]).startswith("unresolved") for row in group_rows
            ),
            "source_candidate_shortfall": sum(
                int(row["source_candidate_shortfall"]) for row in group_rows
            ),
            "human_verified_shortfall": sum(
                int(row["human_verified_shortfall"]) for row in group_rows
            ),
        },
    }


def write_reference_source_shortfalls(
    report: Mapping[str, object],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    if report.get("schema_version") != REFERENCE_SOURCE_SHORTFALL_SCHEMA_VERSION:
        raise ValueError("unsupported reference source shortfall report schema")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / REFERENCE_SOURCE_SHORTFALL_FILE
    markdown_path = output / REFERENCE_SOURCE_SHORTFALL_MARKDOWN_FILE
    if not overwrite:
        for path in (json_path, markdown_path):
            if path.exists():
                raise FileExistsError(path)
    _write_json_atomic(report, json_path)
    _write_text_atomic(_shortfall_markdown(report), markdown_path)
    return {"source_shortfalls": json_path, "source_shortfalls_markdown": markdown_path}


def _eligible_media_counts(
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
) -> tuple[
    dict[str, int],
    dict[str, int],
    dict[str, int],
]:
    observation_fields = {
        "reference_observation_id",
        "accepted_taxon_key",
        "taxon_reconciliation_status",
        "uncertain_taxon_match",
        "fossil",
        "occurrence_absent",
        "basis_of_record_suitable",
        "preserved_specimen",
        "life_stage",
    }
    media_fields = {
        "reference_media_id",
        "reference_observation_id",
        "download_status",
        "verification_status",
        "licence_policy_status",
    }
    _require_columns(observations, observation_fields, artifact="reference observations")
    _require_columns(media_candidates, media_fields, artifact="reference media candidates")
    eligible_observations = observations.filter(
        pl.col("taxon_reconciliation_status").is_in(
            ["accepted_key_exact", "accepted_name_synonym"]
        )
        & ~pl.col("uncertain_taxon_match")
        & ~pl.col("fossil")
        & ~pl.col("occurrence_absent")
        & pl.col("basis_of_record_suitable")
        & ~pl.col("preserved_specimen")
    ).select("reference_observation_id", "accepted_taxon_key", "life_stage")
    eligible_media = (
        media_candidates.filter(
            pl.col("download_status").is_in(sorted(_ELIGIBLE_DOWNLOAD_STATUSES))
            & pl.col("licence_policy_status").is_in(
                sorted(_ELIGIBLE_LICENCE_STATUSES)
            )
        )
        .join(
            eligible_observations,
            on="reference_observation_id",
            how="inner",
            validate="m:1",
        )
        .unique("reference_media_id")
    )
    adult = eligible_media.filter(pl.col("life_stage") == "adult")
    larval = eligible_media.filter(pl.col("life_stage") == "larva")
    return (
        _counts(eligible_media),
        _counts(adult),
        _counts(larval),
    )


def _counts(frame: pl.DataFrame) -> dict[str, int]:
    return {
        str(row["accepted_taxon_key"]): int(row["count"])
        for row in frame.group_by("accepted_taxon_key")
        .agg(pl.len().alias("count"))
        .iter_rows(named=True)
    }


def _shortfall_markdown(report: Mapping[str, object]) -> str:
    rows = report.get("groups") or []
    lines = [
        "# Reference source shortfalls",
        "",
        f"Status: `{report.get('status')}`",
        "",
        "| Group | Status | Minimum | Source candidates | Candidate shortfall | Human-verified | Verified shortfall |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        lines.append(
            "| {group} | {status} | {minimum_required} | "
            "{source_candidate_media_count} | {source_candidate_shortfall} | "
            "{human_verified_media_count} | {human_verified_shortfall} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Source taxon matches are acquisition candidates, not human-verified image labels.",
            "",
        ]
    )
    return "\n".join(lines)


def _accepted_taxon_key(value: object) -> str:
    text = str(value or "").strip().removeprefix("gbif:")
    if not text.isdigit() or int(text) < 1:
        raise ValueError("accepted taxon key must contain a positive GBIF key")
    return f"gbif:{int(text)}"


def _array(value: object, *, field: str) -> list[object]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    return value


def _optional_nonnegative_integer(value: object, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must include a UTC offset")
    return parsed.astimezone(UTC)


def _require_columns(frame: pl.DataFrame, required: set[str], *, artifact: str) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(f"{artifact} must be a Polars DataFrame")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{artifact} is missing columns: " + ", ".join(missing))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    _write_text_atomic(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        path,
    )


def _write_text_atomic(content: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    temporary.replace(path)


__all__ = [
    "REFERENCE_SOURCE_SHORTFALL_FILE",
    "REFERENCE_SOURCE_SHORTFALL_MARKDOWN_FILE",
    "compile_reference_source_shortfalls",
    "write_reference_source_shortfalls",
]
