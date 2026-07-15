from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import time
from typing import Any

import polars as pl

from biominer.registry.gbif_production import RetryingHTTPGet
from biominer.storage.parquet import write_parquet


REGIONAL_COMPETITOR_EVIDENCE_SCHEMA_VERSION = (
    "regional-competitor-occurrence-evidence-v1.0.0"
)
REGIONAL_COMPETITOR_CHECKPOINT_SCHEMA_VERSION = (
    "regional-competitor-facet-checkpoint-v1.0.0"
)
REGIONAL_COMPETITOR_MANIFEST_SCHEMA_VERSION = (
    "regional-competitor-build-manifest-v1.0.0"
)
REGIONAL_COMPETITOR_EVIDENCE_FILE = "regional_competitor_evidence.parquet"
REGIONAL_COMPETITOR_MANIFEST_FILE = "regional_competitor_manifest.json"
REGIONAL_COMPETITOR_CHECKPOINT_FILE = "state.json"
GBIF_SPECIES_FACET_LIMIT = 1_000

HTTPGet = Callable[[str, dict[str, object]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SpeciesFacetCount:
    accepted_taxon_key: str
    occurrence_count: int


@dataclass(frozen=True, slots=True)
class RegionalCompetitorBuildResult:
    evidence: pl.DataFrame
    manifest: dict[str, object]
    resumed: bool


class GBIFRegionalCompetitorFacetSource:
    source = "GBIF"

    def __init__(
        self,
        *,
        candidate_genus_taxon_key: str,
        source_snapshot_version: str,
        facet_limit: int = GBIF_SPECIES_FACET_LIMIT,
        max_retries: int = 5,
        http_get: HTTPGet | None = None,
    ) -> None:
        self.candidate_genus_taxon_key = _accepted_taxon_key(
            candidate_genus_taxon_key
        )
        self.source_snapshot_version = _required_text(
            source_snapshot_version,
            field="source_snapshot_version",
        )
        if isinstance(facet_limit, bool) or not isinstance(facet_limit, int):
            raise TypeError("facet_limit must be an integer")
        if not 1 <= facet_limit <= GBIF_SPECIES_FACET_LIMIT:
            raise ValueError(
                f"facet_limit must be between 1 and {GBIF_SPECIES_FACET_LIMIT}"
            )
        self.facet_limit = facet_limit
        self._transport = None if http_get is not None else RetryingHTTPGet(
            max_retries=max_retries
        )
        self._http_get = http_get or self._transport

    @property
    def attempt_count(self) -> int:
        return int(getattr(self._transport, "attempt_count", 0))

    @property
    def retry_count(self) -> int:
        return int(getattr(self._transport, "retry_count", 0))

    @property
    def rate_limit_count(self) -> int:
        return int(getattr(self._transport, "rate_limit_count", 0))

    def country_species_counts(
        self,
        country_code: str,
    ) -> tuple[SpeciesFacetCount, ...]:
        country = _country_code(country_code)
        payload = self._http_get(
            "/occurrence/search",
            {
                "taxonKey": _bare_gbif_key(self.candidate_genus_taxon_key),
                "country": country,
                "hasCoordinate": "true",
                "hasGeospatialIssue": "false",
                "occurrenceStatus": "PRESENT",
                "limit": 0,
                "facet": "speciesKey",
                "facetLimit": self.facet_limit,
            },
        )
        if not isinstance(payload, dict):
            raise ValueError("GBIF occurrence facet response must be a JSON object")
        counts = _facet_counts(payload, "SPECIES_KEY")
        rows = [
            SpeciesFacetCount(
                accepted_taxon_key=_accepted_taxon_key(item.get("name")),
                occurrence_count=_positive_integer(
                    item.get("count"),
                    field="facet occurrence count",
                ),
            )
            for item in counts
        ]
        return tuple(
            sorted(rows, key=lambda item: item.accepted_taxon_key)
        )

    def close(self) -> None:
        if self._transport is not None:
            self._transport.close()

    def __enter__(self) -> GBIFRegionalCompetitorFacetSource:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self.close()


def regional_competitor_evidence_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "target_accepted_taxon_key": pl.String,
        "target_scientific_name": pl.String,
        "candidate_accepted_taxon_key": pl.String,
        "candidate_scientific_name": pl.String,
        "family": pl.String,
        "genus": pl.String,
        "candidate_reason": pl.String,
        "candidate_rank": pl.UInt32,
        "overlap_country_count": pl.UInt32,
        "target_country_count": pl.UInt32,
        "candidate_occurrence_count": pl.UInt64,
        "weighted_overlap_score": pl.Float64,
        "country_evidence": pl.List(
            pl.Struct(
                {
                    "country_code": pl.String,
                    "target_eligible_occurrence_count": pl.UInt64,
                    "candidate_occurrence_count": pl.UInt64,
                }
            )
        ),
        "source": pl.String,
        "source_taxon_key": pl.String,
        "source_snapshot_version": pl.String,
        "source_query_hash": pl.String,
        "registry_version": pl.String,
        "retrieved_at": pl.Datetime("us", "UTC"),
    }


def build_regional_competitor_evidence(
    *,
    target_occurrence_evidence: pl.DataFrame,
    taxa: pl.DataFrame,
    target_accepted_taxon_key: str,
    candidate_genus_taxon_key: str,
    registry_version: str,
    source_snapshot_version: str,
    source: GBIFRegionalCompetitorFacetSource,
    checkpoint_dir: str | Path,
    retrieved_at: str | datetime,
    minimum_target_country_occurrences: int = 5,
    maximum_candidates: int = 30,
) -> RegionalCompetitorBuildResult:
    started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    target_key = _accepted_taxon_key(target_accepted_taxon_key)
    genus_key = _accepted_taxon_key(candidate_genus_taxon_key)
    registry = _required_text(registry_version, field="registry_version")
    snapshot = _required_text(
        source_snapshot_version,
        field="source_snapshot_version",
    )
    if source.candidate_genus_taxon_key != genus_key:
        raise ValueError("facet source candidate genus key does not match the build")
    if source.source_snapshot_version != snapshot:
        raise ValueError("facet source snapshot version does not match the build")
    minimum = _positive_integer(
        minimum_target_country_occurrences,
        field="minimum_target_country_occurrences",
    )
    maximum = _positive_integer(maximum_candidates, field="maximum_candidates")
    retrieved = _utc_datetime(retrieved_at)
    taxonomy = _accepted_taxonomy(taxa)
    target = taxonomy.get(target_key)
    if target is None or target["rank"] != "SPECIES":
        raise ValueError(f"target is not an accepted registry species: {target_key}")
    genus = taxonomy.get(genus_key)
    if genus is None or genus["rank"] != "GENUS":
        raise ValueError(f"candidate genus is not accepted in the registry: {genus_key}")
    if genus["family"] != target["family"]:
        raise ValueError("candidate genus is not in the target species family")
    target_countries = _target_country_counts(
        target_occurrence_evidence,
        target_key=target_key,
        minimum=minimum,
    )
    if not target_countries:
        raise ValueError("no target countries meet the minimum occurrence threshold")

    query_hash = _source_query_hash(
        target_key=target_key,
        genus_key=genus_key,
        target_countries=target_countries,
        source_snapshot_version=snapshot,
        minimum=minimum,
    )
    identity: dict[str, object] = {
        "schema_version": REGIONAL_COMPETITOR_CHECKPOINT_SCHEMA_VERSION,
        "target_accepted_taxon_key": target_key,
        "candidate_genus_taxon_key": genus_key,
        "registry_version": registry,
        "source": source.source,
        "source_snapshot_version": snapshot,
        "source_query_hash": query_hash,
        "retrieved_at": retrieved.isoformat().replace("+00:00", "Z"),
        "minimum_target_country_occurrences": minimum,
        "target_countries": target_countries,
    }
    checkpoint = Path(checkpoint_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)
    state_path = checkpoint / REGIONAL_COMPETITOR_CHECKPOINT_FILE
    state, resumed = _load_or_initialize_state(state_path, identity=identity)
    completed = dict(state.get("completed_country_facets") or {})
    initial_completed_count = len(completed)
    for country_code in target_countries:
        if country_code in completed:
            continue
        counters_before = (
            source.attempt_count,
            source.retry_count,
            source.rate_limit_count,
        )
        counts = source.country_species_counts(country_code)
        counters_after = (
            source.attempt_count,
            source.retry_count,
            source.rate_limit_count,
        )
        completed[country_code] = [
            {
                "accepted_taxon_key": item.accepted_taxon_key,
                "occurrence_count": item.occurrence_count,
            }
            for item in counts
        ]
        state["completed_country_facets"] = {
            code: completed[code] for code in sorted(completed)
        }
        state["status"] = (
            "complete" if len(completed) == len(target_countries) else "running"
        )
        metrics = dict(state.get("api") or {})
        metrics["successful_query_count"] = int(
            metrics.get("successful_query_count") or 0
        ) + 1
        for field, before, after in zip(
            ("attempt_count", "retry_count", "rate_limit_count"),
            counters_before,
            counters_after,
            strict=True,
        ):
            metrics[field] = int(metrics.get(field) or 0) + max(0, after - before)
        state["api"] = metrics
        _write_json_atomic(state, state_path)
    if set(completed) != set(target_countries):
        raise ValueError("regional competitor facets ended with incomplete countries")

    evidence, unmatched_keys = _compile_evidence(
        completed,
        taxonomy=taxonomy,
        target_key=target_key,
        target=target,
        genus_key=genus_key,
        target_countries=target_countries,
        registry_version=registry,
        source_snapshot_version=snapshot,
        source_query_hash=query_hash,
        retrieved_at=retrieved,
        maximum_candidates=maximum,
        same_genus=target["genus_key"] == genus_key,
    )
    completed_at = datetime.now(UTC)
    api_metrics = dict(state.get("api") or {})
    manifest: dict[str, object] = {
        "schema_version": REGIONAL_COMPETITOR_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "target_accepted_taxon_key": target_key,
        "target_scientific_name": target["scientific_name"],
        "candidate_genus_taxon_key": genus_key,
        "candidate_genus_scientific_name": genus["scientific_name"],
        "registry_version": registry,
        "source": source.source,
        "source_snapshot_version": snapshot,
        "source_query_hash": query_hash,
        "retrieved_at": retrieved.isoformat().replace("+00:00", "Z"),
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "elapsed_seconds": round(time.monotonic() - started_monotonic, 6),
        "pid": os.getpid(),
        "git_sha": _current_git_sha(),
        "minimum_target_country_occurrences": minimum,
        "maximum_candidates": maximum,
        "target_country_count": len(target_countries),
        "target_countries": [
            {
                "country_code": code,
                "eligible_occurrence_count": target_countries[code],
            }
            for code in target_countries
        ],
        "candidate_count": evidence.height,
        "unmatched_or_out_of_scope_facet_key_count": len(unmatched_keys),
        "unmatched_or_out_of_scope_facet_keys": unmatched_keys,
        "checkpoint": {
            "resumed": resumed,
            "initial_completed_country_count": initial_completed_count,
            "completed_country_count": len(completed),
            "state_file": str(state_path),
        },
        "api": {
            "successful_query_count": int(
                api_metrics.get("successful_query_count") or 0
            ),
            "attempt_count": int(api_metrics.get("attempt_count") or 0),
            "retry_count": int(api_metrics.get("retry_count") or 0),
            "rate_limit_count": int(api_metrics.get("rate_limit_count") or 0),
        },
    }
    return RegionalCompetitorBuildResult(
        evidence=evidence,
        manifest=manifest,
        resumed=resumed,
    )


def write_regional_competitor_artifacts(
    result: RegionalCompetitorBuildResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    if not isinstance(result, RegionalCompetitorBuildResult):
        raise TypeError("result must be a RegionalCompetitorBuildResult")
    if result.evidence.schema != regional_competitor_evidence_schema():
        raise ValueError("regional competitor evidence has an invalid schema")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evidence_path = output / REGIONAL_COMPETITOR_EVIDENCE_FILE
    manifest_path = output / REGIONAL_COMPETITOR_MANIFEST_FILE
    if not overwrite:
        existing = [path for path in (evidence_path, manifest_path) if path.exists()]
        if existing:
            raise FileExistsError(existing[0])
    write_parquet(result.evidence, evidence_path, overwrite=overwrite)
    manifest = {
        **result.manifest,
        "files": {
            "regional_competitor_evidence": {
                "file": evidence_path.name,
                "row_count": result.evidence.height,
                "byte_count": evidence_path.stat().st_size,
                "sha256": _sha256_file(evidence_path),
            }
        },
    }
    _write_json_atomic(manifest, manifest_path)
    return {"evidence": evidence_path, "manifest": manifest_path}


def _target_country_counts(
    frame: pl.DataFrame,
    *,
    target_key: str,
    minimum: int,
) -> dict[str, int]:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("target_occurrence_evidence must be a Polars DataFrame")
    required = {
        "gbif_id",
        "accepted_taxon_key",
        "country_code",
        "range_inference_eligible",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "target occurrence evidence is missing columns: " + ", ".join(missing)
        )
    rows = (
        frame.filter(
            (pl.col("accepted_taxon_key") == target_key)
            & pl.col("range_inference_eligible")
            & pl.col("country_code").is_not_null()
            & (pl.col("country_code").str.len_chars() == 2)
        )
        .select("gbif_id", "country_code")
        .unique()
        .group_by("country_code")
        .agg(pl.len().alias("eligible_occurrence_count"))
        .filter(pl.col("eligible_occurrence_count") >= minimum)
        .sort(["eligible_occurrence_count", "country_code"], descending=[True, False])
        .to_dicts()
    )
    return {
        _country_code(row["country_code"]): int(row["eligible_occurrence_count"])
        for row in rows
    }


def _accepted_taxonomy(taxa: pl.DataFrame) -> dict[str, dict[str, str]]:
    if not isinstance(taxa, pl.DataFrame):
        raise TypeError("taxa must be a Polars DataFrame")
    required = {
        "accepted_taxon_key",
        "scientific_name",
        "rank",
        "taxonomic_status",
        "family",
        "genus",
        "genus_key",
    }
    missing = sorted(required - set(taxa.columns))
    if missing:
        raise ValueError("taxa is missing columns: " + ", ".join(missing))
    accepted = taxa.filter(pl.col("taxonomic_status") == "ACCEPTED")
    if accepted["accepted_taxon_key"].n_unique() != accepted.height:
        raise ValueError("accepted registry taxa contain duplicate keys")
    return {
        str(row["accepted_taxon_key"]): {
            "scientific_name": str(row["scientific_name"]),
            "rank": str(row["rank"]),
            "family": str(row["family"] or ""),
            "genus": str(row["genus"] or ""),
            "genus_key": str(row["genus_key"] or ""),
        }
        for row in accepted.iter_rows(named=True)
    }


def _compile_evidence(
    completed: Mapping[str, object],
    *,
    taxonomy: Mapping[str, Mapping[str, str]],
    target_key: str,
    target: Mapping[str, str],
    genus_key: str,
    target_countries: Mapping[str, int],
    registry_version: str,
    source_snapshot_version: str,
    source_query_hash: str,
    retrieved_at: datetime,
    maximum_candidates: int,
    same_genus: bool,
) -> tuple[pl.DataFrame, list[str]]:
    occurrences: defaultdict[str, dict[str, int]] = defaultdict(dict)
    unmatched: set[str] = set()
    for country_code in target_countries:
        raw_counts = completed.get(country_code)
        if not isinstance(raw_counts, list):
            raise ValueError(f"checkpoint country {country_code} has invalid facet counts")
        for item in raw_counts:
            if not isinstance(item, Mapping):
                raise ValueError("checkpoint facet count must be an object")
            key = _accepted_taxon_key(item.get("accepted_taxon_key"))
            count = _positive_integer(
                item.get("occurrence_count"),
                field="checkpoint facet occurrence count",
            )
            taxon = taxonomy.get(key)
            if (
                taxon is None
                or taxon["rank"] != "SPECIES"
                or taxon["genus_key"] != genus_key
            ):
                unmatched.add(key)
                continue
            if key != target_key:
                occurrences[key][country_code] = count

    provisional: list[dict[str, object]] = []
    for key, country_counts in occurrences.items():
        taxon = taxonomy[key]
        country_evidence = [
            {
                "country_code": country_code,
                "target_eligible_occurrence_count": target_countries[country_code],
                "candidate_occurrence_count": country_counts[country_code],
            }
            for country_code in target_countries
            if country_code in country_counts
        ]
        weighted_score = sum(
            math.log1p(
                min(
                    row["target_eligible_occurrence_count"],
                    row["candidate_occurrence_count"],
                )
            )
            for row in country_evidence
        )
        provisional.append(
            {
                "schema_version": REGIONAL_COMPETITOR_EVIDENCE_SCHEMA_VERSION,
                "target_accepted_taxon_key": target_key,
                "target_scientific_name": target["scientific_name"],
                "candidate_accepted_taxon_key": key,
                "candidate_scientific_name": taxon["scientific_name"],
                "family": taxon["family"],
                "genus": taxon["genus"],
                "candidate_reason": (
                    "regional_same_genus_occurrence_overlap"
                    if same_genus
                    else "regional_reviewed_genus_occurrence_overlap"
                ),
                "overlap_country_count": len(country_counts),
                "target_country_count": len(target_countries),
                "candidate_occurrence_count": sum(country_counts.values()),
                "weighted_overlap_score": weighted_score,
                "country_evidence": country_evidence,
                "source": "GBIF",
                "source_taxon_key": genus_key,
                "source_snapshot_version": source_snapshot_version,
                "source_query_hash": source_query_hash,
                "registry_version": registry_version,
                "retrieved_at": retrieved_at,
            }
        )
    provisional.sort(
        key=lambda row: (
            -int(row["overlap_country_count"]),
            -float(row["weighted_overlap_score"]),
            -int(row["candidate_occurrence_count"]),
            str(row["candidate_accepted_taxon_key"]),
        )
    )
    rows = [
        {**row, "candidate_rank": rank}
        for rank, row in enumerate(provisional[:maximum_candidates], start=1)
    ]
    return (
        pl.DataFrame(rows, schema=regional_competitor_evidence_schema())
        .sort("candidate_rank")
        if rows
        else pl.DataFrame(schema=regional_competitor_evidence_schema()),
        sorted(unmatched),
    )


def _load_or_initialize_state(
    path: Path,
    *,
    identity: Mapping[str, object],
) -> tuple[dict[str, object], bool]:
    if not path.exists():
        state = {
            **identity,
            "status": "running",
            "completed_country_facets": {},
            "api": {
                "successful_query_count": 0,
                "attempt_count": 0,
                "retry_count": 0,
                "rate_limit_count": 0,
            },
        }
        _write_json_atomic(state, path)
        return state, False
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("regional competitor checkpoint must be a JSON object")
    for key, expected in identity.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"regional competitor checkpoint identity mismatch for {key}"
            )
    completed = payload.get("completed_country_facets")
    if not isinstance(completed, dict):
        raise ValueError("regional competitor checkpoint facets must be an object")
    unexpected = sorted(set(completed) - set(identity["target_countries"]))
    if unexpected:
        raise ValueError(
            "regional competitor checkpoint contains unexpected countries: "
            + ", ".join(unexpected)
        )
    return payload, bool(completed)


def _facet_counts(payload: Mapping[str, object], field: str) -> list[Mapping[str, object]]:
    facets = payload.get("facets") or []
    if not isinstance(facets, list):
        raise ValueError("GBIF occurrence facets must be an array")
    for facet in facets:
        if not isinstance(facet, Mapping):
            continue
        if str(facet.get("field") or "").upper() != field:
            continue
        counts = facet.get("counts") or []
        if not isinstance(counts, list) or not all(
            isinstance(item, Mapping) for item in counts
        ):
            raise ValueError("GBIF occurrence facet counts must be an array of objects")
        return counts
    raise ValueError(f"GBIF occurrence response omitted required facet {field}")


def _source_query_hash(
    *,
    target_key: str,
    genus_key: str,
    target_countries: Mapping[str, int],
    source_snapshot_version: str,
    minimum: int,
) -> str:
    return _sha256_json(
        {
            "source": "GBIF",
            "target_accepted_taxon_key": target_key,
            "candidate_genus_taxon_key": genus_key,
            "target_countries": target_countries,
            "source_snapshot_version": source_snapshot_version,
            "minimum_target_country_occurrences": minimum,
            "filters": {
                "hasCoordinate": True,
                "hasGeospatialIssue": False,
                "occurrenceStatus": "PRESENT",
                "facet": "speciesKey",
            },
        }
    )


def _accepted_taxon_key(value: object) -> str:
    text = str(value or "").strip().removeprefix("gbif:")
    if not text.isdigit() or int(text) < 1:
        raise ValueError("accepted taxon key must contain a positive GBIF key")
    return f"gbif:{int(text)}"


def _bare_gbif_key(value: object) -> str:
    return _accepted_taxon_key(value).removeprefix("gbif:")


def _country_code(value: object) -> str:
    code = str(value or "").strip().upper()
    if len(code) != 2 or not code.isalpha():
        raise ValueError(f"invalid ISO alpha-2 country code: {value!r}")
    return code


def _positive_integer(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} must be an integer") from exc
    if parsed < 1:
        raise ValueError(f"{field} must be positive")
    return parsed


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must not be blank")
    return text


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
        str(value).replace("Z", "+00:00")
    )
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("retrieved_at must include a UTC offset")
    return parsed.astimezone(UTC)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _current_git_sha() -> str | None:
    from biominer.reports.flickr_fetch import current_git_sha

    return current_git_sha()


def _write_json_atomic(payload: Mapping[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


__all__ = [
    "GBIFRegionalCompetitorFacetSource",
    "REGIONAL_COMPETITOR_CHECKPOINT_FILE",
    "REGIONAL_COMPETITOR_EVIDENCE_FILE",
    "REGIONAL_COMPETITOR_MANIFEST_FILE",
    "RegionalCompetitorBuildResult",
    "SpeciesFacetCount",
    "build_regional_competitor_evidence",
    "regional_competitor_evidence_schema",
    "write_regional_competitor_artifacts",
]
