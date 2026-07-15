from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.references.licensing import ReferenceLicencePolicy
from biominer.references.prototype_planner import (
    PLANNER_SCORE_SEMANTICS,
    PROTOTYPE_REFERENCE_PLANNER_VERSION,
    PrototypeReferencePlanResult,
    PrototypeReferencePlannerConfig,
    PrototypeReferenceQuota,
    plan_trust_first_layered_references,
    write_prototype_reference_plan_result,
)
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
    make_acquisition_plan_id,
    reference_acquisition_plan_frame,
    validate_reference_media_candidates,
    validate_reference_observations,
    write_reference_acquisition_plan,
)


PROTOTYPE_ACQUISITION_VERSION = "prototype-acquisition-v1.0.0"
PROTOTYPE_SOURCE_SUMMARY_SCHEMA_VERSION = "prototype-reference-source-summary-v1.0.0"
PROTOTYPE_SHORTFALL_SCHEMA_VERSION = "prototype-reference-shortfalls-v1.0.0"
PROTOTYPE_ACQUISITION_REPORT_SCHEMA_VERSION = "prototype-acquisition-report-v1.0.0"
PROTOTYPE_SOURCE_SUMMARY_FILE = "prototype_reference_source_summary.parquet"
PROTOTYPE_SHORTFALL_FILE = "prototype_reference_shortfalls.parquet"
PROTOTYPE_ACQUISITION_REPORT_FILE = "prototype_acquisition_report.json"
PROTOTYPE_ACQUISITION_REPORT_MARKDOWN_FILE = "prototype_acquisition_report.md"


@dataclass(frozen=True, slots=True)
class PrototypeAcquisitionResult:
    plan: pl.DataFrame
    planner: PrototypeReferencePlanResult
    source_summary: pl.DataFrame
    shortfalls: pl.DataFrame
    report: dict[str, Any]
    markdown: str


@dataclass(frozen=True, slots=True)
class _Quota:
    reference_group: str
    accepted_taxon_key: str
    scientific_name: str
    requested_count: int
    route: str
    life_stage: str
    visual_domain: str


def prototype_reference_source_summary_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "acquisition_plan_id": pl.String,
        "reference_group": pl.String,
        "candidate_scope_type": pl.String,
        "candidate_scope_id": pl.String,
        "candidate_name": pl.String,
        "source": pl.String,
        "trust_level": pl.String,
        "geographic_layer": pl.String,
        "source_record_count": pl.UInt32,
        "media_record_count": pl.UInt32,
        "unique_media_identifier_count": pl.UInt32,
        "licence_eligible_count": pl.UInt32,
        "attribution_complete_count": pl.UInt32,
        "independent_observation_count": pl.UInt32,
        "eligible_candidate_count": pl.UInt32,
        "selected_for_download_count": pl.UInt32,
        "prototype_only": pl.Boolean,
    }


def prototype_reference_shortfall_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "acquisition_plan_id": pl.String,
        "reference_group": pl.String,
        "candidate_scope_type": pl.String,
        "candidate_scope_id": pl.String,
        "candidate_name": pl.String,
        "route": pl.String,
        "requested_count": pl.UInt32,
        "available_candidate_count": pl.UInt32,
        "selected_candidate_count": pl.UInt32,
        "shortfall_count": pl.UInt32,
        "status": pl.String,
        "shortfall_reason": pl.String,
        "prototype_only": pl.Boolean,
    }


def compile_prototype_acquisition(
    *,
    observations: Sequence[pl.DataFrame],
    media_candidates: Sequence[pl.DataFrame],
    query_plans: Sequence[Mapping[str, object]],
    visual_domain_manifest: Mapping[str, object],
    created_at: str | datetime,
    selection_seed: int = 20260715,
    licence_policy: ReferenceLicencePolicy | None = None,
) -> PrototypeAcquisitionResult:
    combined_observations = _concat_validated(
        observations,
        validator=validate_reference_observations,
        sort_by=["source", "source_observation_id"],
        unique_id="reference_observation_id",
        artifact="reference observations",
    )
    combined_media = _concat_validated(
        media_candidates,
        validator=validate_reference_media_candidates,
        sort_by=["source", "provider_media_id", "reference_observation_id"],
        unique_id="reference_media_id",
        artifact="reference media candidates",
    )
    policy = licence_policy or ReferenceLicencePolicy()
    qualified_media = _apply_licence_policy(combined_media, policy)
    quotas = _compile_quotas(query_plans)
    target_key = _single_target(query_plans, visual_domain_manifest)
    planner = plan_trust_first_layered_references(
        observations=combined_observations,
        media_candidates=qualified_media,
        quotas=[
            PrototypeReferenceQuota(
                reference_group=quota.reference_group,
                accepted_taxon_key=quota.accepted_taxon_key,
                scientific_name=quota.scientific_name,
                requested_count=quota.requested_count,
                route=quota.route,
                life_stage=quota.life_stage,
                visual_domain=quota.visual_domain,
            )
            for quota in quotas
        ],
        config=PrototypeReferencePlannerConfig(selection_seed=selection_seed),
    )
    visual_rows = _visual_domain_rows(visual_domain_manifest)
    configuration_fingerprint = _sha256_json(
        {
            "version": PROTOTYPE_ACQUISITION_VERSION,
            "target_accepted_taxon_key": target_key,
            "selection_seed": selection_seed,
            "licence_policy_fingerprint": policy.fingerprint,
            "planner_version": PROTOTYPE_REFERENCE_PLANNER_VERSION,
            "quotas": [asdict(quota) for quota in quotas],
            "visual_domain_manifest_version": visual_domain_manifest.get(
                "manifest_version"
            ),
        }
    )
    candidate_set_id = _sha256_json(
        {
            "target": target_key,
            "taxa": sorted({quota.accepted_taxon_key for quota in quotas}),
            "visual_domains": sorted(
                str(row["visual_domain_category"]) for row in visual_rows
            ),
        }
    )
    acquisition_plan_id = make_acquisition_plan_id(
        target_accepted_taxon_key=target_key,
        candidate_set_id=candidate_set_id,
        plan_configuration_fingerprint=configuration_fingerprint,
    )
    created = _utc_datetime(created_at)
    plan = _build_plan(
        planner=planner,
        quotas=quotas,
        target_key=target_key,
        candidate_set_id=candidate_set_id,
        acquisition_plan_id=acquisition_plan_id,
        configuration_fingerprint=configuration_fingerprint,
        licence_policy_version=policy.version,
        selection_seed=selection_seed,
        created_at=created,
    )
    source_summary = _build_source_summary(
        planner=planner,
        visual_rows=visual_rows,
        acquisition_plan_id=acquisition_plan_id,
    )
    shortfalls = _build_shortfalls(
        planner=planner,
        quotas=quotas,
        visual_rows=visual_rows,
        acquisition_plan_id=acquisition_plan_id,
    )
    report = _build_report(
        target_key=target_key,
        acquisition_plan_id=acquisition_plan_id,
        candidate_set_id=candidate_set_id,
        configuration_fingerprint=configuration_fingerprint,
        policy=policy,
        observations=combined_observations,
        media=qualified_media,
        planner=planner,
        visual_rows=visual_rows,
        source_summary=source_summary,
        shortfalls=shortfalls,
        created_at=created,
    )
    result = PrototypeAcquisitionResult(
        plan=plan,
        planner=planner,
        source_summary=source_summary,
        shortfalls=shortfalls,
        report=report,
        markdown=_report_markdown(report),
    )
    validate_prototype_acquisition_result(result)
    return result


def validate_prototype_acquisition_result(result: PrototypeAcquisitionResult) -> None:
    from biominer.references.schemas import validate_reference_acquisition_plan

    validate_reference_acquisition_plan(result.plan)
    _validate_frame(
        result.source_summary,
        schema=prototype_reference_source_summary_schema(),
        schema_version=PROTOTYPE_SOURCE_SUMMARY_SCHEMA_VERSION,
        sort_by=[
            "reference_group",
            "candidate_scope_type",
            "candidate_scope_id",
            "source",
            "trust_level",
            "geographic_layer",
        ],
    )
    _validate_frame(
        result.shortfalls,
        schema=prototype_reference_shortfall_schema(),
        schema_version=PROTOTYPE_SHORTFALL_SCHEMA_VERSION,
        sort_by=["reference_group", "candidate_scope_type", "candidate_scope_id"],
    )
    for row in result.shortfalls.iter_rows(named=True):
        if row["shortfall_count"] != row["requested_count"] - row["selected_candidate_count"]:
            raise ValueError("prototype reference shortfall count is inconsistent")
        if row["selected_candidate_count"] > row["available_candidate_count"]:
            raise ValueError("prototype selected count exceeds available candidates")
    summary = result.report.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("prototype acquisition report summary is missing")
    if summary.get("selected_for_download_count") != int(
        result.source_summary["selected_for_download_count"].sum()
    ):
        raise ValueError("prototype selected-for-download report count is inconsistent")


def write_prototype_acquisition_result(
    result: PrototypeAcquisitionResult,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> dict[str, Path]:
    validate_prototype_acquisition_result(result)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "plan": output / "reference_acquisition_plan.parquet",
        "source_summary": output / PROTOTYPE_SOURCE_SUMMARY_FILE,
        "shortfalls": output / PROTOTYPE_SHORTFALL_FILE,
        "report": output / PROTOTYPE_ACQUISITION_REPORT_FILE,
        "markdown": output / PROTOTYPE_ACQUISITION_REPORT_MARKDOWN_FILE,
    }
    if not overwrite:
        for path in paths.values():
            if path.exists():
                raise FileExistsError(path)
    write_reference_acquisition_plan(result.plan, output, overwrite=overwrite)
    _write_parquet(result.source_summary, paths["source_summary"])
    _write_parquet(result.shortfalls, paths["shortfalls"])
    _write_json(result.report, paths["report"])
    _write_text(result.markdown, paths["markdown"])
    planner_paths = write_prototype_reference_plan_result(result.planner, output)
    paths.update(
        {f"planner_{key}": path for key, path in planner_paths.items()}
    )
    return paths


def _concat_validated(
    frames: Sequence[pl.DataFrame],
    *,
    validator: Any,
    sort_by: list[str],
    unique_id: str,
    artifact: str,
) -> pl.DataFrame:
    if not frames:
        raise ValueError(f"{artifact} inputs must not be empty")
    for frame in frames:
        validator(frame)
    combined = pl.concat(list(frames), how="vertical").sort(sort_by)
    if combined[unique_id].n_unique() != combined.height:
        raise ValueError(f"{artifact} contain duplicate {unique_id} values")
    validator(combined)
    return combined


def _apply_licence_policy(
    media: pl.DataFrame,
    policy: ReferenceLicencePolicy,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for raw in media.iter_rows(named=True):
        row = dict(raw)
        decision = policy.evaluate(
            media_licence=row["licence"],
            licence_uri=row["licence_uri"],
            attribution=row["attribution"],
        )
        row["licence_policy_status"] = decision.status
        if decision.reason:
            reasons = {
                value
                for value in str(row.get("exclusion_reason") or "").split(";")
                if value
            }
            reasons.add(decision.reason)
            row["exclusion_reason"] = ";".join(sorted(reasons))
        rows.append(row)
    from biominer.references.schemas import reference_media_candidates_frame

    return reference_media_candidates_frame(rows)


def _compile_quotas(query_plans: Sequence[Mapping[str, object]]) -> tuple[_Quota, ...]:
    values: list[_Quota] = []
    seen_groups: set[str] = set()
    for plan in query_plans:
        queries = plan.get("queries")
        quotas = plan.get("acquisition_quotas")
        if not isinstance(queries, list) or not isinstance(quotas, Mapping):
            raise ValueError("prototype acquisition query plan is malformed")
        names = {
            str(row["accepted_taxon_key"]): str(row["scientific_name"])
            for row in queries
            if isinstance(row, Mapping)
        }
        for group, raw in quotas.items():
            if not isinstance(raw, Mapping) or str(raw.get("status") or "").startswith(
                "unresolved"
            ):
                continue
            species = [str(value) for value in raw.get("species", [])]
            if not species:
                continue
            requests = _requested_by_species(raw, species)
            life_stage = str(raw.get("life_stage") or "adult")
            route = "larval" if life_stage in {"larva", "caterpillar"} else "adult_field"
            for key in species:
                reference_group = f"{group}:{key}"
                if reference_group in seen_groups:
                    raise ValueError(f"duplicate prototype reference group: {reference_group}")
                seen_groups.add(reference_group)
                values.append(
                    _Quota(
                        reference_group=reference_group,
                        accepted_taxon_key=key,
                        scientific_name=names[key],
                        requested_count=requests[key],
                        route=route,
                        life_stage=life_stage,
                        visual_domain="live_field",
                    )
                )
    if not values:
        raise ValueError("prototype acquisition has no concrete quotas")
    return tuple(sorted(values, key=lambda value: value.reference_group))


def _requested_by_species(
    quota: Mapping[str, object],
    species: list[str],
) -> dict[str, int]:
    per_species = quota.get("minimum_per_species", quota.get("planned_per_species"))
    if per_species is not None:
        requested = int(per_species)
        return {key: requested for key in species}
    total = int(quota.get("minimum_total") or 0)
    base, remainder = divmod(total, len(species))
    return {
        key: base + (1 if index < remainder else 0)
        for index, key in enumerate(sorted(species))
    }


def _single_target(
    query_plans: Sequence[Mapping[str, object]],
    visual_domain_manifest: Mapping[str, object],
) -> str:
    values = {
        str(plan.get("target_accepted_taxon_key") or plan.get("target", ""))
        for plan in query_plans
        if plan.get("target_accepted_taxon_key") or plan.get("target")
    }
    visual_target = str(
        visual_domain_manifest.get("target_accepted_taxon_key") or ""
    ).strip()
    if visual_target:
        values.add(visual_target)
    if not values:
        raise ValueError("prototype acquisition target taxon is missing")
    if len(values) != 1:
        raise ValueError("prototype query plans disagree on target taxon")
    return next(iter(values))


def _build_plan(
    *,
    planner: PrototypeReferencePlanResult,
    quotas: Sequence[_Quota],
    target_key: str,
    candidate_set_id: str,
    acquisition_plan_id: str,
    configuration_fingerprint: str,
    licence_policy_version: str,
    selection_seed: int,
    created_at: datetime,
) -> pl.DataFrame:
    report_by_group = {
        str(row["reference_group"]): row for row in planner.report["quota_results"]
    }
    evidence_by_group = {
        group: frame
        for (group,), frame in planner.evidence.group_by("reference_group")
    }
    rows: list[dict[str, object]] = []
    for quota in quotas:
        report = report_by_group[quota.reference_group]
        evidence = evidence_by_group.get(quota.reference_group)
        selected = (
            evidence.filter(pl.col("selected"))
            if evidence is not None
            else planner.selected.head(0)
        )
        sources = sorted(set(selected["source"].to_list())) if selected.height else []
        rows.append(
            {
                "schema_version": REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
                "acquisition_plan_id": acquisition_plan_id,
                "target_accepted_taxon_key": target_key,
                "candidate_set_id": candidate_set_id,
                "candidate_accepted_taxon_key": quota.accepted_taxon_key,
                "scientific_name": quota.scientific_name,
                "geo_cluster_id": "trust_first_layered_mix",
                "life_stage": quota.life_stage,
                "visual_domain": quota.visual_domain,
                "source": sources[0] if len(sources) == 1 else "mixed" if sources else "none",
                "requested_count": quota.requested_count,
                "existing_support_count": 0,
                "available_candidate_count": int(report["eligible_count"]),
                "selected_candidate_count": int(report["selected_count"]),
                "shortfall_count": int(report["shortfall_count"]),
                "fallback_level": 3,
                "selection_strategy": PROTOTYPE_REFERENCE_PLANNER_VERSION,
                "selection_seed": selection_seed,
                "max_distance_km": None,
                "licence_policy_version": licence_policy_version,
                "source_snapshot_version": "mixed_source_snapshots",
                "plan_configuration_fingerprint": configuration_fingerprint,
                "created_at": created_at,
            }
        )
    return reference_acquisition_plan_frame(rows)


def _visual_domain_rows(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("candidates")
    if not isinstance(raw, list) or not raw:
        raise ValueError("visual-domain manifest has no candidates")
    rows = [dict(row) for row in raw if isinstance(row, Mapping)]
    if len(rows) != len(raw):
        raise ValueError("visual-domain candidate rows must be objects")
    return rows


def _build_source_summary(
    *,
    planner: PrototypeReferencePlanResult,
    visual_rows: Sequence[Mapping[str, object]],
    acquisition_plan_id: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    keys = [
        "reference_group",
        "accepted_taxon_key",
        "scientific_name",
        "source",
        "trust_level",
        "geographic_layer",
    ]
    for values, frame in planner.evidence.group_by(keys):
        group, scope, name, source, trust, layer = values
        rows.append(
            {
                "schema_version": PROTOTYPE_SOURCE_SUMMARY_SCHEMA_VERSION,
                "acquisition_plan_id": acquisition_plan_id,
                "reference_group": group,
                "candidate_scope_type": "accepted_taxon",
                "candidate_scope_id": scope,
                "candidate_name": name,
                "source": source,
                "trust_level": trust,
                "geographic_layer": layer,
                "source_record_count": frame["reference_observation_id"].n_unique(),
                "media_record_count": frame.height,
                "unique_media_identifier_count": frame["provider_media_id"].n_unique(),
                "licence_eligible_count": frame.filter(
                    pl.col("licence_policy_status").is_in(
                        ["allowed", "research_only"]
                    )
                ).height,
                "attribution_complete_count": frame.filter(
                    pl.col("attribution_complete")
                ).height,
                "independent_observation_count": frame[
                    "reference_observation_id"
                ].n_unique(),
                "eligible_candidate_count": frame.filter(pl.col("eligible")).height,
                "selected_for_download_count": frame.filter(pl.col("selected")).height,
                "prototype_only": True,
            }
        )
    for raw in visual_rows:
        eligible = bool(raw.get("prototype_eligible")) and str(
            raw.get("licence_check_status")
        ) == "allowed" and bool(str(raw.get("attribution") or "").strip())
        category = str(raw["visual_domain_category"])
        rows.append(
            {
                "schema_version": PROTOTYPE_SOURCE_SUMMARY_SCHEMA_VERSION,
                "acquisition_plan_id": acquisition_plan_id,
                "reference_group": "visual_domain_negatives",
                "candidate_scope_type": "visual_domain",
                "candidate_scope_id": category,
                "candidate_name": category.replace("_", " "),
                "source": str(raw["source"]),
                "trust_level": "R4",
                "geographic_layer": "E",
                "source_record_count": 1,
                "media_record_count": 1,
                "unique_media_identifier_count": 1,
                "licence_eligible_count": int(eligible),
                "attribution_complete_count": int(
                    bool(str(raw.get("attribution") or "").strip())
                ),
                "independent_observation_count": 1,
                "eligible_candidate_count": int(eligible),
                "selected_for_download_count": int(eligible),
                "prototype_only": True,
            }
        )
    schema = prototype_reference_source_summary_schema()
    return pl.DataFrame(rows, schema=schema, strict=True).sort(
        [
            "reference_group",
            "candidate_scope_type",
            "candidate_scope_id",
            "source",
            "trust_level",
            "geographic_layer",
        ]
    )


def _build_shortfalls(
    *,
    planner: PrototypeReferencePlanResult,
    quotas: Sequence[_Quota],
    visual_rows: Sequence[Mapping[str, object]],
    acquisition_plan_id: str,
) -> pl.DataFrame:
    reports = {
        str(row["reference_group"]): row for row in planner.report["quota_results"]
    }
    rows: list[dict[str, object]] = []
    for quota in quotas:
        report = reports[quota.reference_group]
        selected = int(report["selected_count"])
        requested = quota.requested_count
        shortfall = requested - selected
        rows.append(
            {
                "schema_version": PROTOTYPE_SHORTFALL_SCHEMA_VERSION,
                "acquisition_plan_id": acquisition_plan_id,
                "reference_group": quota.reference_group.split(":", 1)[0],
                "candidate_scope_type": "accepted_taxon",
                "candidate_scope_id": quota.accepted_taxon_key,
                "candidate_name": quota.scientific_name,
                "route": quota.route,
                "requested_count": requested,
                "available_candidate_count": int(report["eligible_count"]),
                "selected_candidate_count": selected,
                "shortfall_count": shortfall,
                "status": "met" if not shortfall else "documented_shortfall",
                "shortfall_reason": "" if not shortfall else "insufficient_metadata_qualified_media",
                "prototype_only": True,
            }
        )
    for raw in visual_rows:
        eligible = bool(raw.get("prototype_eligible")) and str(
            raw.get("licence_check_status")
        ) == "allowed" and bool(str(raw.get("attribution") or "").strip())
        category = str(raw["visual_domain_category"])
        rows.append(
            {
                "schema_version": PROTOTYPE_SHORTFALL_SCHEMA_VERSION,
                "acquisition_plan_id": acquisition_plan_id,
                "reference_group": "visual_domain_negatives",
                "candidate_scope_type": "visual_domain",
                "candidate_scope_id": category,
                "candidate_name": category.replace("_", " "),
                "route": "visual_artifact" if not raw.get("contains_biological_butterfly") else "ambiguous_visual_domain",
                "requested_count": 1,
                "available_candidate_count": int(eligible),
                "selected_candidate_count": int(eligible),
                "shortfall_count": int(not eligible),
                "status": "met" if eligible else "documented_shortfall",
                "shortfall_reason": "" if eligible else "curated_candidate_ineligible",
                "prototype_only": True,
            }
        )
    return pl.DataFrame(
        rows,
        schema=prototype_reference_shortfall_schema(),
        strict=True,
    ).sort(["reference_group", "candidate_scope_type", "candidate_scope_id"])


def _build_report(
    *,
    target_key: str,
    acquisition_plan_id: str,
    candidate_set_id: str,
    configuration_fingerprint: str,
    policy: ReferenceLicencePolicy,
    observations: pl.DataFrame,
    media: pl.DataFrame,
    planner: PrototypeReferencePlanResult,
    visual_rows: Sequence[Mapping[str, object]],
    source_summary: pl.DataFrame,
    shortfalls: pl.DataFrame,
    created_at: datetime,
) -> dict[str, Any]:
    selected = planner.selected
    visual_selected = sum(
        bool(row.get("prototype_eligible"))
        and str(row.get("licence_check_status")) == "allowed"
        and bool(str(row.get("attribution") or "").strip())
        for row in visual_rows
    )
    trust = Counter(str(value) for value in selected["trust_level"])
    trust["R4"] += visual_selected
    layers = Counter(str(value) for value in selected["geographic_layer"])
    sources = Counter(str(value) for value in selected["source"])
    sources.update(str(row["source"]) for row in visual_rows if row.get("prototype_eligible"))
    licences = Counter(
        str(value)
        for value in media.filter(
            pl.col("licence_policy_status").is_in(["allowed", "research_only"])
        )[
            "licence"
        ]
    )
    selected_layer_counts = {
        key: layers.get(key, 0) + (visual_selected if key == "E" else 0)
        for key in ("A", "B", "C", "D", "E")
    }
    selected_layer_buckets = {
        "AB": selected_layer_counts["A"] + selected_layer_counts["B"],
        "C": selected_layer_counts["C"],
        "D": selected_layer_counts["D"],
        "E": selected_layer_counts["E"],
    }
    total_selected = sum(selected_layer_buckets.values())
    return {
        "schema_version": PROTOTYPE_ACQUISITION_REPORT_SCHEMA_VERSION,
        "prototype_only": True,
        "target_accepted_taxon_key": target_key,
        "acquisition_plan_id": acquisition_plan_id,
        "candidate_set_id": candidate_set_id,
        "plan_configuration_fingerprint": configuration_fingerprint,
        "planner_version": PROTOTYPE_REFERENCE_PLANNER_VERSION,
        "licence_policy_version": policy.version,
        "licence_policy_fingerprint": policy.fingerprint,
        "created_at": created_at.isoformat().replace("+00:00", "Z"),
        "score_semantics": PLANNER_SCORE_SEMANTICS,
        "score_is_probability": False,
        "provider_supported_is_human_verified": False,
        "summary": {
            "source_record_count": observations.height + len(visual_rows),
            "media_record_count": media.height + len(visual_rows),
            "unique_media_identifier_count": len(
                set(str(value) for value in media["media_identifier"])
                | {str(row["media_uri"]) for row in visual_rows}
            ),
            "licence_eligible_count": media.filter(
                pl.col("licence_policy_status").is_in(
                    ["allowed", "research_only"]
                )
            ).height
            + visual_selected,
            "independent_observation_count": observations[
                "reference_observation_id"
            ].n_unique()
            + len(visual_rows),
            "selected_for_download_count": int(
                source_summary["selected_for_download_count"].sum()
            ),
            "biological_selected_for_download_count": selected.height,
            "visual_domain_selected_for_download_count": visual_selected,
            "reference_group_count": shortfalls["reference_group"].n_unique(),
            "shortfall_scope_count": shortfalls.filter(
                pl.col("shortfall_count") > 0
            ).height,
            "total_shortfall_count": int(shortfalls["shortfall_count"].sum()),
        },
        "selected_trust_distribution": {
            key: trust.get(key, 0) for key in ("R1", "R2", "R3", "R4", "R5")
        },
        "selected_geographic_layer_distribution": selected_layer_counts,
        "selected_layer_bucket_distribution": selected_layer_buckets,
        "actual_layer_bucket_proportions": {
            key: count / total_selected if total_selected else 0.0
            for key, count in selected_layer_buckets.items()
        },
        "selected_source_distribution": dict(sorted(sources.items())),
        "eligible_licence_distribution": dict(sorted(licences.items())),
        "licence_policy_status_distribution": {
            str(row["licence_policy_status"]): int(row["count"])
            for row in media.group_by("licence_policy_status")
            .agg(pl.len().alias("count"))
            .sort("licence_policy_status")
            .iter_rows(named=True)
        },
        "unresolved_risks": [
            "provider-supported reference labels have not all received independent human taxonomic verification",
            "perceptual duplicates and provider mirrors require downloaded bytes before final resolution",
            "visual-neighbour candidates remain morphology candidates until frozen-embedding confirmation",
            "bounded source queries may document shortfalls rather than exhaust provider inventories",
        ],
    }


def _report_markdown(report: Mapping[str, object]) -> str:
    summary = report["summary"]
    assert isinstance(summary, Mapping)
    return "\n".join(
        [
            "# Prototype reference acquisition plan",
            "",
            "This is prototype-only metadata-qualified support evidence. Provider support is not human verification.",
            "",
            "| Metric | Count |",
            "| --- | ---: |",
            f"| Source records | {summary['source_record_count']} |",
            f"| Media records | {summary['media_record_count']} |",
            f"| Unique media identifiers | {summary['unique_media_identifier_count']} |",
            f"| Licence eligible | {summary['licence_eligible_count']} |",
            f"| Independent observations | {summary['independent_observation_count']} |",
            f"| Selected for download | {summary['selected_for_download_count']} |",
            f"| Shortfall scopes | {summary['shortfall_scope_count']} |",
            "",
            "Planner scores are ordinal priorities, not probabilities.",
            "",
        ]
    )


def _validate_frame(
    frame: pl.DataFrame,
    *,
    schema: Mapping[str, pl.DataType],
    schema_version: str,
    sort_by: list[str],
) -> None:
    if frame.schema != dict(schema):
        raise ValueError("prototype acquisition artifact schema mismatch")
    if frame.height and set(frame["schema_version"].to_list()) != {schema_version}:
        raise ValueError("prototype acquisition artifact schema version mismatch")
    if not frame.equals(frame.sort(sort_by)):
        raise ValueError("prototype acquisition artifact is not deterministically sorted")


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("created_at must be timezone-aware")
    return parsed.astimezone(UTC)


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _write_parquet(frame: pl.DataFrame, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        frame.write_parquet(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(value: Mapping[str, object], path: Path) -> None:
    _write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", path)


def _write_text(value: str, path: Path) -> None:
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


__all__ = [
    "PROTOTYPE_ACQUISITION_REPORT_FILE",
    "PROTOTYPE_ACQUISITION_REPORT_MARKDOWN_FILE",
    "PROTOTYPE_ACQUISITION_VERSION",
    "PROTOTYPE_SHORTFALL_FILE",
    "PROTOTYPE_SHORTFALL_SCHEMA_VERSION",
    "PROTOTYPE_SOURCE_SUMMARY_FILE",
    "PROTOTYPE_SOURCE_SUMMARY_SCHEMA_VERSION",
    "PrototypeAcquisitionResult",
    "compile_prototype_acquisition",
    "prototype_reference_shortfall_schema",
    "prototype_reference_source_summary_schema",
    "validate_prototype_acquisition_result",
    "write_prototype_acquisition_result",
]
