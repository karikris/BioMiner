from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field as dataclass_field
from datetime import UTC, date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from biominer.candidates.regional_union import validate_regional_candidate_species
from biominer.references.schemas import (
    REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
    REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
    make_acquisition_plan_id,
    make_reference_selection_id,
    reference_acquisition_plan_frame,
    reference_acquisition_selections_frame,
    validate_reference_acquisition_plan,
    validate_reference_acquisition_selections,
    validate_reference_media_candidates,
    validate_reference_observations,
    write_reference_acquisition_plan,
    write_reference_acquisition_selections,
)


REFERENCE_PLANNER_POLICY_VERSION = "minimum-sqrt-diversity-v1.0.0"
REFERENCE_PLAN_REPORT_SCHEMA_VERSION = "reference-acquisition-report-v1.0.0"
REFERENCE_PLAN_METRICS_FILE = "reference_acquisition_plan.json"
REFERENCE_PLAN_SUMMARY_FILE = "reference_acquisition_plan.md"


@dataclass(frozen=True, slots=True)
class ReferenceStratumQuota:
    life_stage: str = "adult"
    visual_domain: str = "unreviewed"
    requested_per_species: int = 20

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "life_stage",
            _required_text(self.life_stage, field="life_stage").casefold(),
        )
        object.__setattr__(
            self,
            "visual_domain",
            _required_text(self.visual_domain, field="visual_domain").casefold(),
        )
        object.__setattr__(
            self,
            "requested_per_species",
            _positive_int(
                self.requested_per_species,
                field="requested_per_species",
            ),
        )


@dataclass(frozen=True, slots=True)
class ReferencePlannerConfig:
    strata: tuple[ReferenceStratumQuota, ...] = dataclass_field(
        default_factory=lambda: (ReferenceStratumQuota(),)
    )
    minimum_per_sufficient_cluster: int = 2
    sufficiently_populated_candidate_count: int = 10
    distance_balance_band_km: float = 50.0
    selection_seed: int = 42
    licence_policy_version: str = "reference-licences-v1"
    selection_strategy: str = REFERENCE_PLANNER_POLICY_VERSION
    eligible_download_statuses: tuple[str, ...] = ("pending", "complete")
    eligible_licence_policy_statuses: tuple[str, ...] = (
        "allowed",
        "research_only",
        "unreviewed",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.strata, tuple) or not self.strata:
            raise ValueError("strata must be a non-empty tuple")
        if not all(isinstance(value, ReferenceStratumQuota) for value in self.strata):
            raise TypeError("strata must contain ReferenceStratumQuota values")
        keys = [(value.life_stage, value.visual_domain) for value in self.strata]
        if len(set(keys)) != len(keys):
            raise ValueError("reference planning strata must be unique")
        object.__setattr__(
            self,
            "minimum_per_sufficient_cluster",
            _nonnegative_int(
                self.minimum_per_sufficient_cluster,
                field="minimum_per_sufficient_cluster",
            ),
        )
        object.__setattr__(
            self,
            "sufficiently_populated_candidate_count",
            _positive_int(
                self.sufficiently_populated_candidate_count,
                field="sufficiently_populated_candidate_count",
            ),
        )
        if (
            self.sufficiently_populated_candidate_count
            < self.minimum_per_sufficient_cluster
        ):
            raise ValueError(
                "sufficiently_populated_candidate_count must be at least "
                "minimum_per_sufficient_cluster"
            )
        distance_band = float(self.distance_balance_band_km)
        if not math.isfinite(distance_band) or distance_band <= 0.0:
            raise ValueError("distance_balance_band_km must be finite and positive")
        object.__setattr__(self, "distance_balance_band_km", distance_band)
        if isinstance(self.selection_seed, bool) or not isinstance(
            self.selection_seed, int
        ):
            raise TypeError("selection_seed must be an integer")
        if not 0 <= self.selection_seed <= 18_446_744_073_709_551_615:
            raise ValueError("selection_seed must fit an unsigned 64-bit integer")
        for field in ("licence_policy_version", "selection_strategy"):
            object.__setattr__(
                self,
                field,
                _required_text(getattr(self, field), field=field),
            )
        object.__setattr__(
            self,
            "eligible_download_statuses",
            _normalised_values(
                self.eligible_download_statuses,
                field="eligible_download_statuses",
            ),
        )
        object.__setattr__(
            self,
            "eligible_licence_policy_statuses",
            _normalised_values(
                self.eligible_licence_policy_statuses,
                field="eligible_licence_policy_statuses",
            ),
        )

    def payload(self) -> dict[str, object]:
        return {
            "policy_version": self.selection_strategy,
            "strata": [
                {
                    "life_stage": value.life_stage,
                    "visual_domain": value.visual_domain,
                    "requested_per_species": value.requested_per_species,
                }
                for value in self.strata
            ],
            "minimum_per_sufficient_cluster": self.minimum_per_sufficient_cluster,
            "sufficiently_populated_candidate_count": (
                self.sufficiently_populated_candidate_count
            ),
            "distance_balance_band_km": self.distance_balance_band_km,
            "selection_seed": self.selection_seed,
            "licence_policy_version": self.licence_policy_version,
            "eligible_download_statuses": self.eligible_download_statuses,
            "eligible_licence_policy_statuses": (self.eligible_licence_policy_statuses),
        }


@dataclass(frozen=True, slots=True)
class ReferencePlanResult:
    plan: pl.DataFrame
    selections: pl.DataFrame
    report: dict[str, Any]
    markdown: str


@dataclass(slots=True)
class _DiversityState:
    observations: set[str]
    observers: Counter[str]
    dates: Counter[str]
    localities: Counter[str]
    backgrounds: Counter[str]
    sources: Counter[str]

    @classmethod
    def empty(cls) -> _DiversityState:
        return cls(set(), Counter(), Counter(), Counter(), Counter(), Counter())


def plan_geographically_balanced_support_bank(
    *,
    candidate_species: pl.DataFrame,
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    config: ReferencePlannerConfig | None = None,
    review_metadata: pl.DataFrame | None = None,
    existing_selections: pl.DataFrame | None = None,
    created_at: str | datetime | None = None,
) -> ReferencePlanResult:
    effective_config = config or ReferencePlannerConfig()
    if not isinstance(effective_config, ReferencePlannerConfig):
        raise TypeError("config must be a ReferencePlannerConfig")
    candidate_context = _candidate_context(candidate_species)
    validate_reference_observations(observations)
    validate_reference_media_candidates(media_candidates)
    existing = _existing_support_context(
        existing_selections,
        candidate_context=candidate_context,
    )
    observation_rows = {
        str(row["reference_observation_id"]): row
        for row in observations.iter_rows(named=True)
    }
    missing_observations = sorted(
        set(media_candidates["reference_observation_id"].to_list())
        - set(observation_rows)
    )
    if missing_observations:
        raise ValueError(
            "reference planning media rows lack observation metadata: "
            f"{missing_observations}"
        )
    review_by_media = _review_metadata(review_metadata)
    built_at = _utc_datetime(created_at or datetime.now(UTC), field="created_at")
    candidate_set_id = make_reference_candidate_union_id(candidate_species)
    reference_input_fingerprint = _reference_input_fingerprint(
        observations,
        media_candidates,
        review_metadata,
        existing_selections,
    )
    plan_configuration_fingerprint = _sha256_json(
        {
            "config": effective_config.payload(),
            "candidate_set_id": candidate_set_id,
            "candidate_set_fingerprints": candidate_context["fingerprints"],
            "candidate_input_fingerprint": candidate_context["input_fingerprint"],
            "reference_input_fingerprint": reference_input_fingerprint,
        }
    )
    acquisition_plan_id = make_acquisition_plan_id(
        target_accepted_taxon_key=candidate_context["target_key"],
        candidate_set_id=candidate_set_id,
        plan_configuration_fingerprint=plan_configuration_fingerprint,
    )
    source_snapshot_version = _composite_source_snapshot_version(observations)
    pools = _planning_pools(
        candidate_context=candidate_context,
        observation_rows=observation_rows,
        media_candidates=media_candidates,
        review_by_media=review_by_media,
        config=effective_config,
    )

    plan_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    species = candidate_context["species"]
    cluster_sets = candidate_context["cluster_sets"]
    diversity_by_taxon = {taxon_key: _DiversityState.empty() for taxon_key in species}
    bank_diversity = _DiversityState.empty()
    selected_observation_ids = set(existing["observation_ids"])
    for stratum in effective_config.strata:
        for taxon_key in sorted(species):
            scientific_name = species[taxon_key]
            cluster_ids = sorted(cluster_sets[taxon_key])
            counts = {
                cluster_id: len(
                    {
                        str(row["reference_observation_id"])
                        for row in pools.get(
                            (
                                taxon_key,
                                cluster_id,
                                stratum.life_stage,
                                stratum.visual_domain,
                            ),
                            (),
                        )
                        if str(row["reference_observation_id"])
                        not in selected_observation_ids
                    }
                )
                for cluster_id in cluster_ids
            }
            existing_counts = {
                cluster_id: int(
                    existing["counts_by_cluster"].get(
                        (
                            taxon_key,
                            cluster_id,
                            stratum.life_stage,
                            stratum.visual_domain,
                        ),
                        0,
                    )
                )
                for cluster_id in cluster_ids
            }
            existing_total = int(
                existing["counts_by_stratum"].get(
                    (taxon_key, stratum.life_stage, stratum.visual_domain),
                    0,
                )
            )
            deficit = max(0, stratum.requested_per_species - existing_total)
            requests = _allocate_cluster_requests(
                cluster_ids,
                counts=counts,
                existing_counts=existing_counts,
                requested=deficit,
                config=effective_config,
            )
            diversity = diversity_by_taxon[taxon_key]
            selection_rank = 0
            for cluster_id in cluster_ids:
                pool = list(
                    pools.get(
                        (
                            taxon_key,
                            cluster_id,
                            stratum.life_stage,
                            stratum.visual_domain,
                        ),
                        (),
                    )
                )
                selected = _select_diverse_candidates(
                    pool,
                    requested=requests[cluster_id],
                    diversity=diversity,
                    bank_diversity=bank_diversity,
                    selected_observation_ids=selected_observation_ids,
                    config=effective_config,
                )
                cluster_selection_rows: list[dict[str, object]] = []
                for selected_row, selection_round in selected:
                    selection_rank += 1
                    row = _selection_row(
                        selected_row,
                        selection_round=selection_round,
                        selection_rank=selection_rank,
                        acquisition_plan_id=acquisition_plan_id,
                        target_key=candidate_context["target_key"],
                        candidate_set_id=candidate_set_id,
                        candidate_key=taxon_key,
                        scientific_name=scientific_name,
                        cluster_id=cluster_id,
                        stratum=stratum,
                        source_candidate_set_id=cluster_sets[taxon_key][cluster_id],
                        config=effective_config,
                        plan_configuration_fingerprint=plan_configuration_fingerprint,
                        selected_at=built_at,
                    )
                    selection_rows.append(row)
                    cluster_selection_rows.append(row)
                selected_sources = sorted(
                    {str(row["source"]) for row in cluster_selection_rows}
                )
                source = (
                    selected_sources[0]
                    if len(selected_sources) == 1
                    else "mixed"
                    if selected_sources
                    else "none"
                )
                fallback_level = max(
                    (int(row["fallback_level"]) for row in cluster_selection_rows),
                    default=0,
                )
                distances = [
                    float(row["distance_to_cluster_medoid_km"])
                    for row in cluster_selection_rows
                    if row["distance_to_cluster_medoid_km"] is not None
                ]
                plan_rows.append(
                    {
                        "schema_version": REFERENCE_ACQUISITION_PLAN_SCHEMA_VERSION,
                        "acquisition_plan_id": acquisition_plan_id,
                        "target_accepted_taxon_key": candidate_context["target_key"],
                        "candidate_set_id": candidate_set_id,
                        "candidate_accepted_taxon_key": taxon_key,
                        "scientific_name": scientific_name,
                        "geo_cluster_id": cluster_id,
                        "life_stage": stratum.life_stage,
                        "visual_domain": stratum.visual_domain,
                        "source": source,
                        "requested_count": requests[cluster_id],
                        "existing_support_count": existing_counts[cluster_id],
                        "available_candidate_count": counts[cluster_id],
                        "selected_candidate_count": len(cluster_selection_rows),
                        "shortfall_count": requests[cluster_id]
                        - len(cluster_selection_rows),
                        "fallback_level": fallback_level,
                        "selection_strategy": effective_config.selection_strategy,
                        "selection_seed": effective_config.selection_seed,
                        "max_distance_km": max(distances) if distances else None,
                        "licence_policy_version": (
                            effective_config.licence_policy_version
                        ),
                        "source_snapshot_version": source_snapshot_version,
                        "plan_configuration_fingerprint": (
                            plan_configuration_fingerprint
                        ),
                        "created_at": built_at,
                    }
                )

    plan = reference_acquisition_plan_frame(plan_rows)
    selections = reference_acquisition_selections_frame(selection_rows)
    report = _build_report(
        plan=plan,
        selections=selections,
        target_key=candidate_context["target_key"],
        candidate_set_id=candidate_set_id,
        acquisition_plan_id=acquisition_plan_id,
        plan_configuration_fingerprint=plan_configuration_fingerprint,
        reference_input_fingerprint=reference_input_fingerprint,
        source_snapshot_version=source_snapshot_version,
        existing=existing,
        config=effective_config,
        created_at=built_at,
    )
    result = ReferencePlanResult(
        plan=plan,
        selections=selections,
        report=report,
        markdown=reference_plan_markdown(report),
    )
    validate_reference_plan_result(result)
    return result


def validate_reference_plan_result(result: ReferencePlanResult) -> None:
    if not isinstance(result, ReferencePlanResult):
        raise TypeError("result must be a ReferencePlanResult")
    validate_reference_acquisition_plan(result.plan)
    validate_reference_acquisition_selections(result.selections)
    plan_rows = {
        (
            str(row["candidate_accepted_taxon_key"]),
            str(row["geo_cluster_id"]),
            str(row["life_stage"]),
            str(row["visual_domain"]),
        ): row
        for row in result.plan.iter_rows(named=True)
    }
    selected_counts: Counter[tuple[str, str, str, str]] = Counter()
    selected_rows: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = (
        defaultdict(list)
    )
    for row in result.selections.iter_rows(named=True):
        key = (
            str(row["candidate_accepted_taxon_key"]),
            str(row["geo_cluster_id"]),
            str(row["life_stage"]),
            str(row["visual_domain"]),
        )
        plan_row = plan_rows.get(key)
        if plan_row is None:
            raise ValueError("reference selection has no matching acquisition-plan row")
        for field in (
            "acquisition_plan_id",
            "target_accepted_taxon_key",
            "candidate_set_id",
            "scientific_name",
            "plan_configuration_fingerprint",
        ):
            if row[field] != plan_row[field]:
                raise ValueError(
                    f"reference selection {field} conflicts with acquisition plan"
                )
        selected_counts[key] += 1
        selected_rows[key].append(row)
    for key, row in plan_rows.items():
        if selected_counts[key] != int(row["selected_candidate_count"]):
            raise ValueError(
                "reference acquisition plan selected count conflicts with selection ledger"
            )
        rows = selected_rows[key]
        sources = sorted({str(value["source"]) for value in rows})
        expected_source = (
            sources[0] if len(sources) == 1 else "mixed" if sources else "none"
        )
        if row["source"] != expected_source:
            raise ValueError(
                "reference acquisition plan source conflicts with selection ledger"
            )
        expected_fallback = max(
            (int(value["fallback_level"]) for value in rows),
            default=0,
        )
        if int(row["fallback_level"]) != expected_fallback:
            raise ValueError(
                "reference acquisition plan fallback conflicts with selection ledger"
            )
        distances = [
            float(value["distance_to_cluster_medoid_km"])
            for value in rows
            if value["distance_to_cluster_medoid_km"] is not None
        ]
        expected_max_distance = max(distances) if distances else None
        if row["max_distance_km"] != expected_max_distance:
            raise ValueError(
                "reference acquisition plan distance conflicts with selection ledger"
            )
    ranks_by_class: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for row in result.selections.iter_rows(named=True):
        ranks_by_class[
            (
                str(row["candidate_accepted_taxon_key"]),
                str(row["life_stage"]),
                str(row["visual_domain"]),
            )
        ].append(int(row["selection_rank"]))
    for ranks in ranks_by_class.values():
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            raise ValueError(
                "reference selection ranks must be contiguous within a class"
            )
    summary = _mapping(result.report.get("summary"))
    if summary.get("selected") != result.selections.height:
        raise ValueError("reference acquisition report selected count is inconsistent")
    if summary.get("requested") != int(result.plan["requested_count"].sum()):
        raise ValueError("reference acquisition report requested count is inconsistent")
    if summary.get("shortfall") != int(result.plan["shortfall_count"].sum()):
        raise ValueError("reference acquisition report shortfall count is inconsistent")


def write_reference_plan_result(
    result: ReferencePlanResult,
    output: str | Path,
) -> dict[str, Path]:
    validate_reference_plan_result(result)
    directory = Path(output)
    plan_path = write_reference_acquisition_plan(result.plan, directory)
    selections_path = write_reference_acquisition_selections(
        result.selections,
        directory,
    )
    metrics_path = directory / REFERENCE_PLAN_METRICS_FILE
    summary_path = directory / REFERENCE_PLAN_SUMMARY_FILE
    _write_text_atomic(
        json.dumps(result.report, indent=2, sort_keys=True) + "\n",
        metrics_path,
    )
    _write_text_atomic(result.markdown, summary_path)
    return {
        "plan": plan_path,
        "selections": selections_path,
        "metrics": metrics_path,
        "summary": summary_path,
    }


def reference_plan_markdown(report: Mapping[str, object]) -> str:
    summary = _mapping(report.get("summary"))
    distance = _mapping(report.get("distance_distribution_km"))
    diversity = _mapping(report.get("diversity"))
    lines = [
        "# Reference Acquisition Plan",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Candidate species | {_display(summary.get('candidate_species_count'))} |",
        f"| Configured quota | {_display(summary.get('configured_quota'))} |",
        f"| Existing support | {_display(summary.get('existing_support'))} |",
        f"| Requested | {_display(summary.get('requested'))} |",
        f"| Available | {_display(summary.get('available'))} |",
        f"| Selected | {_display(summary.get('selected'))} |",
        f"| Shortfall | {_display(summary.get('shortfall'))} |",
        f"| Support after selection | {_display(summary.get('support_after_selection'))} |",
        f"| Independent observations | {_display(diversity.get('independent_observations'))} |",
        f"| Unique observers | {_display(diversity.get('unique_observers'))} |",
        f"| Unique localities | {_display(diversity.get('unique_localities'))} |",
        f"| Distance p50 km | {_display(distance.get('p50'))} |",
        f"| Distance p95 km | {_display(distance.get('p95'))} |",
        f"| Distance max km | {_display(distance.get('max'))} |",
        "",
        "## Fallbacks",
        "",
        "| Level | Selected |",
        "| ---: | ---: |",
    ]
    for level, count in sorted(_mapping(report.get("fallback_distribution")).items()):
        lines.append(f"| {_display(level)} | {_display(count)} |")
    lines.extend(
        [
            "",
            "## Sources",
            "",
            "| Source | Selected |",
            "| --- | ---: |",
        ]
    )
    for source, count in sorted(_mapping(report.get("source_distribution")).items()):
        lines.append(f"| {_markdown_text(source)} | {_display(count)} |")
    lines.extend(
        [
            "",
            "## Licences",
            "",
            "| Licence | Selected |",
            "| --- | ---: |",
        ]
    )
    for licence, count in sorted(_mapping(report.get("licence_distribution")).items()):
        lines.append(f"| {_markdown_text(licence)} | {_display(count)} |")
    return "\n".join(lines) + "\n"


def _candidate_context(candidate_species: pl.DataFrame) -> dict[str, Any]:
    if not isinstance(candidate_species, pl.DataFrame):
        raise TypeError("candidate_species must be a Polars DataFrame")
    validate_regional_candidate_species(candidate_species)
    if candidate_species.is_empty():
        raise ValueError("candidate_species must not be empty")
    target_keys = set(candidate_species["target_accepted_taxon_key"].to_list())
    if len(target_keys) != 1:
        raise ValueError(
            "candidate species input must contain exactly one target identity"
        )
    target_key = str(next(iter(target_keys)))
    species: dict[str, str] = {}
    cluster_sets: dict[str, dict[str, str]] = defaultdict(dict)
    fingerprints: set[str] = set()
    target_clusters: set[str] = set()
    for (candidate_set_id,), group in candidate_species.group_by(
        "candidate_set_id",
        maintain_order=True,
    ):
        if group["target_candidate"].sum() != 1:
            raise ValueError(
                f"candidate set {candidate_set_id} must contain one target"
            )
        if group["candidate_accepted_taxon_key"].n_unique() != group.height:
            raise ValueError(
                f"candidate set {candidate_set_id} contains duplicate species"
            )
        cluster_values = set(group["geo_cluster_id"].to_list())
        if len(cluster_values) != 1:
            raise ValueError(
                f"candidate set {candidate_set_id} spans multiple clusters"
            )
        cluster_id = str(next(iter(cluster_values)))
        target_clusters.add(cluster_id)
        for row in group.iter_rows(named=True):
            taxon_key = str(row["candidate_accepted_taxon_key"])
            name = str(row["scientific_name"])
            if taxon_key in species and species[taxon_key] != name:
                raise ValueError(f"candidate species {taxon_key} has conflicting names")
            species[taxon_key] = name
            cluster_sets[taxon_key][cluster_id] = str(candidate_set_id)
            fingerprints.add(str(row["candidate_set_fingerprint"]))
    if target_key not in species:
        raise ValueError("target species is absent from the candidate union")
    if set(cluster_sets[target_key]) != target_clusters:
        raise ValueError("target species must be present in every candidate cluster")
    return {
        "target_key": target_key,
        "species": species,
        "cluster_sets": cluster_sets,
        "fingerprints": sorted(fingerprints),
        "input_fingerprint": _sha256_json(candidate_species.to_dicts()),
    }


def _existing_support_context(
    frame: pl.DataFrame | None,
    *,
    candidate_context: Mapping[str, Any],
) -> dict[str, object]:
    counts_by_cluster: Counter[tuple[str, str, str, str]] = Counter()
    counts_by_stratum: Counter[tuple[str, str, str]] = Counter()
    if frame is None:
        return {
            "observation_ids": set(),
            "counts_by_cluster": counts_by_cluster,
            "counts_by_stratum": counts_by_stratum,
        }
    validate_reference_acquisition_selections(frame)
    target_key = str(candidate_context["target_key"])
    target_keys = set(frame["target_accepted_taxon_key"].to_list())
    if target_keys - {target_key}:
        raise ValueError(
            "existing reference selections contain another target identity"
        )
    observation_groups: dict[str, tuple[str, str, str, str]] = {}
    for row in frame.iter_rows(named=True):
        observation_id = str(row["reference_observation_id"])
        key = (
            str(row["candidate_accepted_taxon_key"]),
            str(row["geo_cluster_id"]),
            str(row["life_stage"]).casefold(),
            str(row["visual_domain"]).casefold(),
        )
        cluster_sets = candidate_context["cluster_sets"]
        if key[0] not in cluster_sets or key[1] not in cluster_sets[key[0]]:
            raise ValueError(
                "existing reference selection is outside the current candidate union"
            )
        previous = observation_groups.setdefault(observation_id, key)
        if previous != key:
            raise ValueError(
                "existing reference observation has conflicting class or stratum metadata"
            )
    for key in observation_groups.values():
        counts_by_cluster[key] += 1
        counts_by_stratum[(key[0], key[2], key[3])] += 1
    return {
        "observation_ids": set(observation_groups),
        "counts_by_cluster": counts_by_cluster,
        "counts_by_stratum": counts_by_stratum,
    }


def _planning_pools(
    *,
    candidate_context: Mapping[str, Any],
    observation_rows: Mapping[str, Mapping[str, object]],
    media_candidates: pl.DataFrame,
    review_by_media: Mapping[str, Mapping[str, object]],
    config: ReferencePlannerConfig,
) -> dict[tuple[str, str, str, str], tuple[dict[str, object], ...]]:
    cluster_sets = candidate_context["cluster_sets"]
    pools: dict[tuple[str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    for media in media_candidates.iter_rows(named=True):
        if str(media["download_status"]) not in config.eligible_download_statuses:
            continue
        if (
            str(media["licence_policy_status"])
            not in config.eligible_licence_policy_statuses
        ):
            continue
        if media["exclusion_reason"] is not None:
            continue
        observation = observation_rows[str(media["reference_observation_id"])]
        taxon_key = _optional_text(observation.get("accepted_taxon_key"))
        cluster_id = _optional_text(observation.get("geo_cluster_id"))
        if taxon_key is None or cluster_id is None:
            continue
        if taxon_key not in cluster_sets or cluster_id not in cluster_sets[taxon_key]:
            continue
        review = review_by_media.get(str(media["reference_media_id"]), {})
        life_stage = _optional_text(review.get("life_stage")) or _required_text(
            observation.get("life_stage"),
            field="life_stage",
        )
        visual_domain = _optional_text(review.get("visual_domain")) or "unreviewed"
        observed_at = observation.get("observed_at")
        observed_date = (
            observed_at.date() if isinstance(observed_at, datetime) else None
        )
        row = {
            **dict(media),
            "candidate_accepted_taxon_key": taxon_key,
            "geo_cluster_id": cluster_id,
            "life_stage": life_stage.casefold(),
            "visual_domain": visual_domain.casefold(),
            "fallback_level": int(observation["fallback_level"]),
            "distance_to_cluster_medoid_km": observation.get(
                "distance_to_cluster_medoid_km"
            ),
            "observer_id": observation.get("observer_id"),
            "observed_date": observed_date,
            "locality": observation.get("locality"),
            "background_group_id": review.get("background_group_id"),
            "source_snapshot_version": observation["source_snapshot_version"],
        }
        pools[(taxon_key, cluster_id, row["life_stage"], row["visual_domain"])].append(
            row
        )
    return {
        key: tuple(sorted(rows, key=lambda row: str(row["reference_media_id"])))
        for key, rows in pools.items()
    }


def _allocate_cluster_requests(
    cluster_ids: Sequence[str],
    *,
    counts: Mapping[str, int],
    existing_counts: Mapping[str, int],
    requested: int,
    config: ReferencePlannerConfig,
) -> dict[str, int]:
    allocation = {cluster_id: 0 for cluster_id in cluster_ids}
    if requested == 0 or not cluster_ids:
        return allocation
    sufficient = [
        cluster_id
        for cluster_id in cluster_ids
        if counts[cluster_id] >= config.sufficiently_populated_candidate_count
    ]
    remaining = requested
    minimum_needs = {
        cluster_id: max(
            0,
            config.minimum_per_sufficient_cluster - existing_counts[cluster_id],
        )
        for cluster_id in sufficient
    }
    for minimum_round in range(max(minimum_needs.values(), default=0)):
        for cluster_id in sorted(sufficient):
            if remaining == 0:
                break
            if minimum_needs[cluster_id] > minimum_round:
                allocation[cluster_id] += 1
                remaining -= 1

    eligible = [cluster_id for cluster_id in cluster_ids if counts[cluster_id] > 0]
    capacities = {
        cluster_id: max(0, counts[cluster_id] - allocation[cluster_id])
        for cluster_id in eligible
    }
    weights = {cluster_id: math.sqrt(counts[cluster_id]) for cluster_id in eligible}
    apportioned, remaining = _capped_hamilton(
        eligible,
        seats=remaining,
        weights=weights,
        capacities=capacities,
    )
    for cluster_id, value in apportioned.items():
        allocation[cluster_id] += value

    if remaining:
        shortage_clusters = eligible or list(cluster_ids)
        shortage_weights = {
            cluster_id: math.sqrt(max(1, counts[cluster_id]))
            for cluster_id in shortage_clusters
        }
        shortage = _hamilton(
            shortage_clusters,
            seats=remaining,
            weights=shortage_weights,
        )
        for cluster_id, value in shortage.items():
            allocation[cluster_id] += value
    if sum(allocation.values()) != requested:
        raise AssertionError(
            "reference cluster quota allocation did not conserve quota"
        )
    return allocation


def _capped_hamilton(
    cluster_ids: Sequence[str],
    *,
    seats: int,
    weights: Mapping[str, float],
    capacities: Mapping[str, int],
) -> tuple[dict[str, int], int]:
    allocation = {cluster_id: 0 for cluster_id in cluster_ids}
    remaining = seats
    while remaining:
        active = [
            cluster_id
            for cluster_id in cluster_ids
            if allocation[cluster_id] < capacities[cluster_id]
        ]
        if not active:
            break
        round_allocation = _hamilton(active, seats=remaining, weights=weights)
        used = 0
        for cluster_id in active:
            available = capacities[cluster_id] - allocation[cluster_id]
            value = min(available, round_allocation[cluster_id])
            allocation[cluster_id] += value
            used += value
        if used == 0:
            cluster_id = min(
                active,
                key=lambda value: (-weights[value], value),
            )
            allocation[cluster_id] += 1
            used = 1
        remaining -= used
    return allocation, remaining


def _hamilton(
    cluster_ids: Sequence[str],
    *,
    seats: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    allocation = {cluster_id: 0 for cluster_id in cluster_ids}
    if seats == 0 or not cluster_ids:
        return allocation
    weight_total = sum(weights[cluster_id] for cluster_id in cluster_ids)
    if weight_total <= 0.0:
        raise ValueError("Hamilton allocation requires a positive total weight")
    exact = {
        cluster_id: seats * weights[cluster_id] / weight_total
        for cluster_id in cluster_ids
    }
    floors = {cluster_id: math.floor(value) for cluster_id, value in exact.items()}
    for cluster_id, value in floors.items():
        allocation[cluster_id] = value
    remainder = seats - sum(floors.values())
    order = sorted(
        cluster_ids,
        key=lambda cluster_id: (
            -(exact[cluster_id] - floors[cluster_id]),
            -weights[cluster_id],
            cluster_id,
        ),
    )
    for cluster_id in order[:remainder]:
        allocation[cluster_id] += 1
    return allocation


def _select_diverse_candidates(
    pool: list[dict[str, object]],
    *,
    requested: int,
    diversity: _DiversityState,
    bank_diversity: _DiversityState,
    selected_observation_ids: set[str],
    config: ReferencePlannerConfig,
) -> list[tuple[dict[str, object], str]]:
    remaining = [
        row
        for row in pool
        if str(row["reference_observation_id"]) not in selected_observation_ids
    ]
    selected: list[tuple[dict[str, object], str]] = []
    while remaining and len(selected) < requested:
        candidates = [
            row
            for row in remaining
            if str(row["reference_observation_id"]) not in selected_observation_ids
        ]
        if not candidates:
            break
        chosen = min(
            candidates,
            key=lambda row: _selection_key(
                row,
                diversity=diversity,
                bank_diversity=bank_diversity,
                config=config,
            ),
        )
        observation_id = str(chosen["reference_observation_id"])
        selected_observation_ids.add(observation_id)
        remaining = [
            row
            for row in remaining
            if str(row["reference_observation_id"]) != observation_id
        ]
        selected.append((chosen, "independent_observation"))
        _update_diversity(diversity, chosen)
        _update_diversity(bank_diversity, chosen)
    return selected


def _selection_key(
    row: Mapping[str, object],
    *,
    diversity: _DiversityState,
    bank_diversity: _DiversityState,
    config: ReferencePlannerConfig,
) -> tuple[object, ...]:
    distance_value = row.get("distance_to_cluster_medoid_km")
    distance = float(distance_value) if distance_value is not None else math.inf
    distance_band = (
        math.floor(distance / config.distance_balance_band_km)
        if math.isfinite(distance)
        else 2**31
    )
    observer = _diversity_key(row.get("observer_id"))
    observed_date = _diversity_key(row.get("observed_date"))
    locality = _diversity_key(row.get("locality"))
    background = _diversity_key(row.get("background_group_id"))
    source = _required_text(row.get("source"), field="source")
    media_id = _required_text(row.get("reference_media_id"), field="reference_media_id")
    return (
        int(row["fallback_level"]),
        distance_band,
        _diversity_rank(diversity.observers, observer),
        _diversity_rank(diversity.dates, observed_date),
        _diversity_rank(diversity.localities, locality),
        _diversity_rank(diversity.backgrounds, background),
        diversity.sources[source],
        _diversity_rank(bank_diversity.observers, observer),
        _diversity_rank(bank_diversity.dates, observed_date),
        _diversity_rank(bank_diversity.localities, locality),
        _diversity_rank(bank_diversity.backgrounds, background),
        bank_diversity.sources[source],
        distance,
        _seeded_tiebreak(config.selection_seed, media_id),
        media_id,
    )


def _update_diversity(
    diversity: _DiversityState,
    row: Mapping[str, object],
) -> None:
    diversity.observations.add(str(row["reference_observation_id"]))
    for counter, value in (
        (diversity.observers, row.get("observer_id")),
        (diversity.dates, row.get("observed_date")),
        (diversity.localities, row.get("locality")),
        (diversity.backgrounds, row.get("background_group_id")),
        (diversity.sources, row.get("source")),
    ):
        key = _diversity_key(value)
        if key:
            counter[key] += 1


def _diversity_rank(counter: Counter[str], value: str | None) -> tuple[int, int]:
    if value is None:
        return (1, 0)
    return (0, counter[value])


def _selection_row(
    selected: Mapping[str, object],
    *,
    selection_round: str,
    selection_rank: int,
    acquisition_plan_id: str,
    target_key: str,
    candidate_set_id: str,
    candidate_key: str,
    scientific_name: str,
    cluster_id: str,
    stratum: ReferenceStratumQuota,
    source_candidate_set_id: str,
    config: ReferencePlannerConfig,
    plan_configuration_fingerprint: str,
    selected_at: datetime,
) -> dict[str, object]:
    selection_id = make_reference_selection_id(
        acquisition_plan_id=acquisition_plan_id,
        reference_media_id=str(selected["reference_media_id"]),
        candidate_accepted_taxon_key=candidate_key,
        geo_cluster_id=cluster_id,
        life_stage=stratum.life_stage,
        visual_domain=stratum.visual_domain,
    )
    return {
        "schema_version": REFERENCE_ACQUISITION_SELECTIONS_SCHEMA_VERSION,
        "reference_selection_id": selection_id,
        "acquisition_plan_id": acquisition_plan_id,
        "target_accepted_taxon_key": target_key,
        "candidate_set_id": candidate_set_id,
        "source_candidate_set_id": source_candidate_set_id,
        "candidate_accepted_taxon_key": candidate_key,
        "scientific_name": scientific_name,
        "geo_cluster_id": cluster_id,
        "life_stage": stratum.life_stage,
        "visual_domain": stratum.visual_domain,
        "reference_media_id": selected["reference_media_id"],
        "reference_observation_id": selected["reference_observation_id"],
        "source": selected["source"],
        "fallback_level": selected["fallback_level"],
        "selection_rank": selection_rank,
        "selection_round": selection_round,
        "distance_to_cluster_medoid_km": selected.get("distance_to_cluster_medoid_km"),
        "observer_id": selected.get("observer_id"),
        "observed_date": selected.get("observed_date"),
        "locality": selected.get("locality"),
        "background_group_id": selected.get("background_group_id"),
        "licence": selected.get("licence"),
        "source_snapshot_version": selected["source_snapshot_version"],
        "selection_strategy": config.selection_strategy,
        "selection_seed": config.selection_seed,
        "plan_configuration_fingerprint": plan_configuration_fingerprint,
        "selected_at": selected_at,
    }


def _build_report(
    *,
    plan: pl.DataFrame,
    selections: pl.DataFrame,
    target_key: str,
    candidate_set_id: str,
    acquisition_plan_id: str,
    plan_configuration_fingerprint: str,
    reference_input_fingerprint: str,
    source_snapshot_version: str,
    existing: Mapping[str, object],
    config: ReferencePlannerConfig,
    created_at: datetime,
) -> dict[str, Any]:
    plan_rows = plan.iter_rows(named=True)
    selection_rows = selections.to_dicts()
    requested = sum(int(row["requested_count"]) for row in plan_rows)
    available = sum(int(value) for value in plan["available_candidate_count"].to_list())
    selected = selections.height
    shortfall = sum(int(value) for value in plan["shortfall_count"].to_list())
    existing_support = len(set(existing["observation_ids"]))
    distances = sorted(
        float(row["distance_to_cluster_medoid_km"])
        for row in selection_rows
        if row["distance_to_cluster_medoid_km"] is not None
    )
    observers = [
        str(row["observer_id"]) for row in selection_rows if row["observer_id"]
    ]
    observations = [str(row["reference_observation_id"]) for row in selection_rows]
    localities = [str(row["locality"]) for row in selection_rows if row["locality"]]
    dates = [
        str(row["observed_date"]) for row in selection_rows if row["observed_date"]
    ]
    backgrounds = [
        str(row["background_group_id"])
        for row in selection_rows
        if row["background_group_id"]
    ]
    per_species_requested = {
        str(key): int(group["requested_count"].sum())
        for (key,), group in plan.group_by("candidate_accepted_taxon_key")
    }
    configured_quota = len(per_species_requested) * sum(
        stratum.requested_per_species for stratum in config.strata
    )
    return {
        "schema_version": REFERENCE_PLAN_REPORT_SCHEMA_VERSION,
        "acquisition_plan_id": acquisition_plan_id,
        "target_accepted_taxon_key": target_key,
        "candidate_set_id": candidate_set_id,
        "plan_configuration_fingerprint": plan_configuration_fingerprint,
        "reference_input_fingerprint": reference_input_fingerprint,
        "source_snapshot_version": source_snapshot_version,
        "created_at": _timestamp(created_at),
        "configuration": config.payload(),
        "summary": {
            "candidate_species_count": len(per_species_requested),
            "configured_quota": configured_quota,
            "existing_support": existing_support,
            "requested": requested,
            "available": available,
            "selected": selected,
            "shortfall": shortfall,
            "support_after_selection": existing_support + selected,
            "target_selected": int(
                selections.filter(
                    pl.col("candidate_accepted_taxon_key") == target_key
                ).height
            ),
            "balanced_requested_quota": len(set(per_species_requested.values())) <= 1,
            "balanced_configured_quota": True,
        },
        "distance_distribution_km": _distribution(distances),
        "licence_distribution": _counter_payload(
            row["licence"] or "missing" for row in selection_rows
        ),
        "source_distribution": _counter_payload(
            row["source"] for row in selection_rows
        ),
        "fallback_distribution": _counter_payload(
            str(row["fallback_level"]) for row in selection_rows
        ),
        "selection_round_distribution": _counter_payload(
            row["selection_round"] for row in selection_rows
        ),
        "diversity": {
            "independent_observations": len(set(observations)),
            "same_observation_extra_images": selected - len(set(observations)),
            "unique_observers": len(set(observers)),
            "missing_observer_count": selected - len(observers),
            "maximum_images_per_observer": max(Counter(observers).values(), default=0),
            "unique_observation_dates": len(set(dates)),
            "unique_localities": len(set(localities)),
            "unique_background_groups": len(set(backgrounds)),
        },
        "by_species": _plan_group_report(plan, "candidate_accepted_taxon_key"),
        "by_cluster": _plan_group_report(plan, "geo_cluster_id"),
        "by_life_stage": _plan_group_report(plan, "life_stage"),
        "by_visual_domain": _plan_group_report(plan, "visual_domain"),
        "by_species_stratum": _plan_group_report_multi(
            plan,
            ("candidate_accepted_taxon_key", "life_stage", "visual_domain"),
        ),
        "diversity_by_species": _selection_diversity_report(
            selections,
            ("candidate_accepted_taxon_key",),
        ),
        "diversity_by_species_stratum": _selection_diversity_report(
            selections,
            ("candidate_accepted_taxon_key", "life_stage", "visual_domain"),
        ),
        "unsupported_metrics": {
            "background_diversity": (None if backgrounds else "not_instrumented"),
        },
        "artifacts": {
            "plan": "reference_acquisition_plan.parquet",
            "selections": "reference_acquisition_selections.parquet",
            "metrics": REFERENCE_PLAN_METRICS_FILE,
            "summary": REFERENCE_PLAN_SUMMARY_FILE,
        },
    }


def _plan_group_report(plan: pl.DataFrame, field: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for (value,), group in plan.group_by(field):
        rows.append(
            {
                field: value,
                "existing_support": int(group["existing_support_count"].sum()),
                "requested": int(group["requested_count"].sum()),
                "available": int(group["available_candidate_count"].sum()),
                "selected": int(group["selected_candidate_count"].sum()),
                "shortfall": int(group["shortfall_count"].sum()),
            }
        )
    return sorted(rows, key=lambda row: str(row[field]))


def _plan_group_report_multi(
    plan: pl.DataFrame,
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in plan.group_by(list(fields)):
        rows.append(
            {
                **dict(zip(fields, keys, strict=True)),
                "existing_support": int(group["existing_support_count"].sum()),
                "requested": int(group["requested_count"].sum()),
                "available": int(group["available_candidate_count"].sum()),
                "selected": int(group["selected_candidate_count"].sum()),
                "shortfall": int(group["shortfall_count"].sum()),
            }
        )
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in fields))


def _selection_diversity_report(
    selections: pl.DataFrame,
    fields: tuple[str, ...],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for keys, group in selections.group_by(list(fields)):
        observers = [value for value in group["observer_id"].to_list() if value]
        localities = [value for value in group["locality"].to_list() if value]
        dates = [value for value in group["observed_date"].to_list() if value]
        backgrounds = [
            value for value in group["background_group_id"].to_list() if value
        ]
        rows.append(
            {
                **dict(zip(fields, keys, strict=True)),
                "selected": group.height,
                "independent_observations": group[
                    "reference_observation_id"
                ].n_unique(),
                "unique_observers": len(set(observers)),
                "missing_observer_count": group.height - len(observers),
                "maximum_images_per_observer": max(
                    Counter(observers).values(),
                    default=0,
                ),
                "unique_observation_dates": len(set(dates)),
                "unique_localities": len(set(localities)),
                "unique_background_groups": len(set(backgrounds)),
            }
        )
    return sorted(rows, key=lambda row: tuple(str(row[field]) for field in fields))


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "p50": _quantile(values, 0.50),
        "p95": _quantile(values, 0.95),
        "max": max(values) if values else None,
    }


def _quantile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    index = math.ceil(quantile * len(values)) - 1
    return float(values[max(0, min(index, len(values) - 1))])


def _counter_payload(values: Sequence[object] | Any) -> dict[str, int]:
    counter = Counter(str(value) for value in values)
    return dict(sorted(counter.items()))


def _review_metadata(
    frame: pl.DataFrame | None,
) -> dict[str, Mapping[str, object]]:
    if frame is None:
        return {}
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("review_metadata must be a Polars DataFrame")
    if "reference_media_id" not in frame.columns:
        raise ValueError("review_metadata is missing reference_media_id")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("review_metadata contains duplicate reference_media_id values")
    return {str(row["reference_media_id"]): row for row in frame.iter_rows(named=True)}


def make_reference_candidate_union_id(frame: pl.DataFrame) -> str:
    """Return the immutable planner identity for a regional candidate union."""

    validate_regional_candidate_species(frame)
    payload = [
        {
            "candidate_set_id": str(row["candidate_set_id"]),
            "candidate_set_fingerprint": str(row["candidate_set_fingerprint"]),
            "geo_cluster_id": str(row["geo_cluster_id"]),
        }
        for row in frame.select(
            ["candidate_set_id", "candidate_set_fingerprint", "geo_cluster_id"]
        )
        .unique()
        .sort(["candidate_set_id", "geo_cluster_id"])
        .iter_rows(named=True)
    ]
    digest = _sha256_json(payload).removeprefix("sha256:")
    return f"reference-candidate-union:{digest[:32]}"


def _reference_input_fingerprint(
    observations: pl.DataFrame,
    media_candidates: pl.DataFrame,
    review_metadata: pl.DataFrame | None,
    existing_selections: pl.DataFrame | None,
) -> str:
    review_fields = (
        "reference_media_id",
        "life_stage",
        "visual_domain",
        "background_group_id",
        "view",
    )
    payload = {
        "observations": observations.to_dicts(),
        "media": media_candidates.to_dicts(),
        "review": (
            [
                {
                    field: row.get(field)
                    for field in review_fields
                    if field in review_metadata.columns
                }
                for row in review_metadata.sort("reference_media_id").iter_rows(
                    named=True
                )
            ]
            if review_metadata is not None
            else []
        ),
        "existing_selections": (
            [
                {
                    "reference_observation_id": row["reference_observation_id"],
                    "reference_media_id": row["reference_media_id"],
                    "candidate_accepted_taxon_key": row["candidate_accepted_taxon_key"],
                    "geo_cluster_id": row["geo_cluster_id"],
                    "life_stage": row["life_stage"],
                    "visual_domain": row["visual_domain"],
                    "source_snapshot_version": row["source_snapshot_version"],
                    "plan_configuration_fingerprint": row[
                        "plan_configuration_fingerprint"
                    ],
                }
                for row in existing_selections.iter_rows(named=True)
            ]
            if existing_selections is not None
            else []
        ),
    }
    return _sha256_json(payload)


def _composite_source_snapshot_version(observations: pl.DataFrame) -> str:
    versions = sorted(set(observations["source_snapshot_version"].to_list()))
    digest = _sha256_json(versions).removeprefix("sha256:")
    return f"reference-sources:{digest[:32]}"


def _seeded_tiebreak(seed: int, media_id: str) -> str:
    return hashlib.sha256(f"{seed}:{media_id}".encode("utf-8")).hexdigest()


def _diversity_key(value: object) -> str | None:
    text = _optional_text(value)
    if text is None or text.casefold() in {"unknown", "unreviewed", "none"}:
        return None
    return text


def _normalised_values(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple):
        raise TypeError(f"{field} must be a tuple")
    normalised = tuple(sorted({str(value or "").strip() for value in values}))
    if not normalised or "" in normalised:
        raise ValueError(f"{field} must contain nonblank values")
    return normalised


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _utc_datetime(value: str | datetime, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"{field} must be an ISO datetime") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _write_text_atomic(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _markdown_text(value: object) -> str:
    return str(value or "unknown").replace("|", "\\|").replace("\n", " ")


def _display(value: object) -> str:
    return "not_instrumented" if value is None else str(value)


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _optional_text(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _positive_int(value: object, *, field: str) -> int:
    parsed = _nonnegative_int(value, field=field)
    if parsed == 0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < 0:
        raise ValueError(f"{field} must be nonnegative")
    return value


__all__ = [
    "REFERENCE_PLAN_METRICS_FILE",
    "REFERENCE_PLAN_REPORT_SCHEMA_VERSION",
    "REFERENCE_PLAN_SUMMARY_FILE",
    "REFERENCE_PLANNER_POLICY_VERSION",
    "ReferencePlanResult",
    "ReferencePlannerConfig",
    "ReferenceStratumQuota",
    "make_reference_candidate_union_id",
    "plan_geographically_balanced_support_bank",
    "reference_plan_markdown",
    "validate_reference_plan_result",
    "write_reference_plan_result",
]
