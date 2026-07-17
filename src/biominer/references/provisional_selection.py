"""Deterministic independent selection for provisional reference support."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import math

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.schemas import REFERENCE_ROUTES, REFERENCE_VIEWS
from biominer.storage.parquet import write_parquet


PROVISIONAL_SELECTION_POLICY_SCHEMA_VERSION = (
    "provisional-selection-policy-v1.0.0"
)
PROVISIONAL_SELECTION_SCHEMA_VERSION = "provisional-selection-decisions-v1.0.0"
PROVISIONAL_SELECTION_DECISIONS_FILE = (
    "reference_provisional_selection_decisions.parquet"
)
PROVISIONAL_SELECTIONS_FILE = "reference_provisional_selections.parquet"

DISTINCT_VIEW_EVIDENCE = frozenset(
    {"none", "provider_documented", "embedding_distinct"}
)

PROVISIONAL_SELECTION_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "reference_media_id": pl.String,
    "accepted_taxon_key": pl.String,
    "reference_observation_id": pl.String,
    "observer_id": pl.String,
    "duplicate_group_id": pl.String,
    "canonical_reference_media_id": pl.String,
    "route": pl.String,
    "documented_view": pl.String,
    "distinct_view_evidence": pl.String,
    "quality_score": pl.Float64,
    "admission_decision": pl.String,
}

PROVISIONAL_SELECTION_DECISION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    **PROVISIONAL_SELECTION_CANDIDATE_SCHEMA,
    "decision": pl.String,
    "decision_reason": pl.String,
    "selection_round": pl.String,
    "selection_rank": pl.UInt32,
    "observation_selection_ordinal": pl.UInt32,
    "observer_selection_ordinal": pl.UInt32,
    "observer_reuse_justified": pl.Boolean,
    "distinct_additional_view_justified": pl.Boolean,
    "species_quota": pl.UInt32,
    "species_selected_count": pl.UInt32,
    "selection_policy_fingerprint": pl.String,
    "decision_fingerprint": pl.String,
}


@dataclass(frozen=True, slots=True)
class ProvisionalSelectionPolicy:
    """Versioned fixed-quota and independence policy."""

    schema_version: str = PROVISIONAL_SELECTION_POLICY_SCHEMA_VERSION
    policy_version: str = "provisional-independent-support-v1"
    quota_per_species: int = 12
    maximum_images_per_observation: int = 1
    maximum_images_per_observer_before_reuse: int = 1
    allow_distinct_additional_views: bool = True
    maximum_distinct_views_per_observation: int = 2
    selection_seed: int = 0

    def __post_init__(self) -> None:
        if self.schema_version != PROVISIONAL_SELECTION_POLICY_SCHEMA_VERSION:
            raise ValueError("unsupported provisional selection policy schema")
        if not isinstance(self.policy_version, str) or not self.policy_version.strip():
            raise ValueError("policy_version must be nonblank text")
        object.__setattr__(self, "policy_version", self.policy_version.strip())
        for field in (
            "quota_per_species",
            "maximum_images_per_observation",
            "maximum_images_per_observer_before_reuse",
            "maximum_distinct_views_per_observation",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field} must be a positive integer")
        if self.maximum_images_per_observation != 1:
            raise ValueError(
                "provisional selection defaults to one image per observation"
            )
        if self.maximum_images_per_observer_before_reuse != 1:
            raise ValueError(
                "provisional selection requires one image per observer before reuse"
            )
        if not isinstance(self.allow_distinct_additional_views, bool):
            raise TypeError("allow_distinct_additional_views must be Boolean")
        if not self.allow_distinct_additional_views and (
            self.maximum_distinct_views_per_observation != 1
        ):
            raise ValueError(
                "disabled additional views require a one-view observation limit"
            )
        if self.allow_distinct_additional_views and (
            self.maximum_distinct_views_per_observation != 2
        ):
            raise ValueError(
                "enabled additional views permit one bounded second view"
            )
        if (
            isinstance(self.selection_seed, bool)
            or not isinstance(self.selection_seed, int)
            or self.selection_seed < 0
        ):
            raise ValueError("selection_seed must be a non-negative integer")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": self.schema_version,
                "policy_version": self.policy_version,
                "quota_per_species": self.quota_per_species,
                "maximum_images_per_observation": (
                    self.maximum_images_per_observation
                ),
                "maximum_images_per_observer_before_reuse": (
                    self.maximum_images_per_observer_before_reuse
                ),
                "allow_distinct_additional_views": (
                    self.allow_distinct_additional_views
                ),
                "maximum_distinct_views_per_observation": (
                    self.maximum_distinct_views_per_observation
                ),
                "selection_seed": self.selection_seed,
            }
        )


@dataclass(frozen=True, slots=True)
class ProvisionalSelectionResult:
    """Selected and skipped decisions with fixed-quota diagnostics."""

    decisions: pl.DataFrame
    selections: pl.DataFrame
    decisions_path: Path
    selections_path: Path
    selected_per_species: tuple[tuple[str, int], ...]
    shortfall_per_species: tuple[tuple[str, int], ...]
    policy_fingerprint: str


@dataclass(slots=True)
class _SelectionState:
    selected_ids: set[str]
    selected_rows: list[dict[str, object]]
    used_duplicate_groups: set[str]
    observation_views: dict[str, set[str]]
    observation_counts: Counter[str]
    observer_counts: Counter[str]
    species_counts: Counter[str]


def select_independent_provisional_support(
    candidates: pl.DataFrame,
    *,
    output_dir: str | Path,
    policy: ProvisionalSelectionPolicy | None = None,
) -> ProvisionalSelectionResult:
    """Select balanced class support and persist a decision for every candidate."""

    active = policy or ProvisionalSelectionPolicy()
    rows = _validated_candidates(candidates)
    prefiltered = _prefilter_reasons(rows)
    selectable = [
        row for row in rows if str(row["reference_media_id"]) not in prefiltered
    ]
    species = sorted({str(row["accepted_taxon_key"]) for row in rows})
    by_species: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in selectable:
        by_species[str(row["accepted_taxon_key"])].append(row)

    state = _SelectionState(
        selected_ids=set(),
        selected_rows=[],
        used_duplicate_groups=set(),
        observation_views=defaultdict(set),
        observation_counts=Counter(),
        observer_counts=Counter(),
        species_counts=Counter(),
    )
    _fill_round_robin(
        by_species,
        species=species,
        policy=active,
        state=state,
        allow_observer_reuse=False,
        allow_additional_views=False,
        selection_round="independent_unused_observer",
    )
    _fill_round_robin(
        by_species,
        species=species,
        policy=active,
        state=state,
        allow_observer_reuse=True,
        allow_additional_views=False,
        selection_round="independent_observer_reuse",
    )
    if active.allow_distinct_additional_views:
        _fill_round_robin(
            by_species,
            species=species,
            policy=active,
            state=state,
            allow_observer_reuse=True,
            allow_additional_views=True,
            selection_round="distinct_additional_view",
        )

    selected_by_id = {
        str(row["reference_media_id"]): row for row in state.selected_rows
    }
    decisions = [
        _decision_row(
            row,
            selected=selected_by_id.get(str(row["reference_media_id"])),
            prefilter_reason=prefiltered.get(str(row["reference_media_id"])),
            state=state,
            policy=active,
        )
        for row in rows
    ]
    decision_frame = pl.DataFrame(
        decisions, schema=PROVISIONAL_SELECTION_DECISION_SCHEMA
    ).sort("reference_media_id")
    selection_frame = decision_frame.filter(pl.col("decision") == "selected").sort(
        "selection_rank"
    )
    output = Path(output_dir)
    decisions_path = write_parquet(
        decision_frame, output / PROVISIONAL_SELECTION_DECISIONS_FILE
    )
    selections_path = write_parquet(
        selection_frame, output / PROVISIONAL_SELECTIONS_FILE
    )
    selected_counts = tuple(
        (taxon, state.species_counts[taxon]) for taxon in species
    )
    shortfalls = tuple(
        (taxon, max(0, active.quota_per_species - state.species_counts[taxon]))
        for taxon in species
    )
    return ProvisionalSelectionResult(
        decisions=decision_frame,
        selections=selection_frame,
        decisions_path=decisions_path,
        selections_path=selections_path,
        selected_per_species=selected_counts,
        shortfall_per_species=shortfalls,
        policy_fingerprint=active.fingerprint,
    )


def _validated_candidates(frame: pl.DataFrame) -> list[dict[str, object]]:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("candidates must be a Polars DataFrame")
    if frame.schema != PROVISIONAL_SELECTION_CANDIDATE_SCHEMA:
        raise ValueError("provisional selection candidate schema mismatch")
    if frame["reference_media_id"].n_unique() != frame.height:
        raise ValueError("reference_media_id must be unique")
    rows = frame.to_dicts()
    observation_taxa: dict[str, set[str]] = defaultdict(set)
    group_canonical_ids: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        for field in (
            "reference_media_id",
            "accepted_taxon_key",
            "reference_observation_id",
            "duplicate_group_id",
            "canonical_reference_media_id",
        ):
            _required_text(row[field], field)
        if row["observer_id"] is not None:
            _required_text(row["observer_id"], "observer_id")
        route = _required_text(row["route"], "route").casefold()
        if route not in REFERENCE_ROUTES:
            raise ValueError(f"unsupported route: {route}")
        row["route"] = route
        view = _required_text(row["documented_view"], "documented_view").casefold()
        if view not in REFERENCE_VIEWS:
            raise ValueError(f"unsupported documented_view: {view}")
        row["documented_view"] = view
        view_evidence = _required_text(
            row["distinct_view_evidence"], "distinct_view_evidence"
        ).casefold()
        if view_evidence not in DISTINCT_VIEW_EVIDENCE:
            raise ValueError("unsupported distinct_view_evidence")
        row["distinct_view_evidence"] = view_evidence
        score = row["quality_score"]
        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not math.isfinite(score)
            or not 0 <= score <= 1
        ):
            raise ValueError("quality_score must be finite and in [0, 1]")
        row["quality_score"] = float(score)
        decision = _required_text(
            row["admission_decision"], "admission_decision"
        ).casefold()
        if decision not in {"admitted", "excluded", "review_required"}:
            raise ValueError("unsupported admission_decision")
        row["admission_decision"] = decision
        observation_taxa[str(row["reference_observation_id"])].add(
            str(row["accepted_taxon_key"])
        )
        group_canonical_ids[str(row["duplicate_group_id"])].add(
            str(row["canonical_reference_media_id"])
        )
    if any(len(values) != 1 for values in observation_taxa.values()):
        raise ValueError("one observation cannot support multiple accepted taxa")
    if any(len(values) != 1 for values in group_canonical_ids.values()):
        raise ValueError("duplicate group has inconsistent canonical media IDs")
    return sorted(rows, key=lambda row: str(row["reference_media_id"]))


def _prefilter_reasons(
    rows: list[dict[str, object]],
) -> dict[str, str]:
    reasons: dict[str, str] = {}
    for row in rows:
        media_id = str(row["reference_media_id"])
        if row["admission_decision"] != "admitted":
            reasons[media_id] = f"admission_{row['admission_decision']}"
        elif row["canonical_reference_media_id"] != media_id:
            reasons[media_id] = "noncanonical_duplicate_member"
        elif row["observer_id"] is None:
            reasons[media_id] = "observer_identity_missing"
    return reasons


def _fill_round_robin(
    pools: Mapping[str, list[dict[str, object]]],
    *,
    species: list[str],
    policy: ProvisionalSelectionPolicy,
    state: _SelectionState,
    allow_observer_reuse: bool,
    allow_additional_views: bool,
    selection_round: str,
) -> None:
    progress = True
    while progress:
        progress = False
        for taxon in species:
            if state.species_counts[taxon] >= policy.quota_per_species:
                continue
            eligible = [
                row
                for row in pools.get(taxon, ())
                if _eligible(
                    row,
                    policy=policy,
                    state=state,
                    allow_observer_reuse=allow_observer_reuse,
                    allow_additional_views=allow_additional_views,
                )
            ]
            if not eligible:
                continue
            chosen = min(eligible, key=lambda row: _selection_key(row, policy, state))
            _select(
                chosen,
                state=state,
                selection_round=selection_round,
            )
            progress = True


def _eligible(
    row: Mapping[str, object],
    *,
    policy: ProvisionalSelectionPolicy,
    state: _SelectionState,
    allow_observer_reuse: bool,
    allow_additional_views: bool,
) -> bool:
    media_id = str(row["reference_media_id"])
    if media_id in state.selected_ids:
        return False
    if str(row["duplicate_group_id"]) in state.used_duplicate_groups:
        return False
    observer = str(row["observer_id"])
    if not allow_observer_reuse and (
        state.observer_counts[observer]
        >= policy.maximum_images_per_observer_before_reuse
    ):
        return False
    observation = str(row["reference_observation_id"])
    observation_count = state.observation_counts[observation]
    if not allow_additional_views:
        return observation_count < policy.maximum_images_per_observation
    if observation_count < policy.maximum_images_per_observation:
        return False
    if observation_count >= policy.maximum_distinct_views_per_observation:
        return False
    view = str(row["documented_view"])
    return (
        row["distinct_view_evidence"] != "none"
        and view != "unknown"
        and view not in state.observation_views[observation]
    )


def _selection_key(
    row: Mapping[str, object],
    policy: ProvisionalSelectionPolicy,
    state: _SelectionState,
) -> tuple[object, ...]:
    media_id = str(row["reference_media_id"])
    observer = str(row["observer_id"])
    tiebreak = canonical_semantic_fingerprint(
        {"selection_seed": policy.selection_seed, "reference_media_id": media_id}
    )
    return (
        state.observer_counts[observer],
        -float(row["quality_score"]),
        tiebreak,
        media_id,
    )


def _select(
    row: dict[str, object],
    *,
    state: _SelectionState,
    selection_round: str,
) -> None:
    media_id = str(row["reference_media_id"])
    observation = str(row["reference_observation_id"])
    observer = str(row["observer_id"])
    taxon = str(row["accepted_taxon_key"])
    observation_ordinal = state.observation_counts[observation] + 1
    observer_ordinal = state.observer_counts[observer] + 1
    selected = {
        **row,
        "selection_round": selection_round,
        "selection_rank": len(state.selected_rows) + 1,
        "observation_selection_ordinal": observation_ordinal,
        "observer_selection_ordinal": observer_ordinal,
        "observer_reuse_justified": observer_ordinal > 1,
        "distinct_additional_view_justified": observation_ordinal > 1,
    }
    state.selected_ids.add(media_id)
    state.selected_rows.append(selected)
    state.used_duplicate_groups.add(str(row["duplicate_group_id"]))
    state.observation_views[observation].add(str(row["documented_view"]))
    state.observation_counts[observation] += 1
    state.observer_counts[observer] += 1
    state.species_counts[taxon] += 1


def _decision_row(
    row: dict[str, object],
    *,
    selected: dict[str, object] | None,
    prefilter_reason: str | None,
    state: _SelectionState,
    policy: ProvisionalSelectionPolicy,
) -> dict[str, object]:
    if selected is not None:
        decision = "selected"
        reason = f"selected_{selected['selection_round']}"
        selection_round = selected["selection_round"]
        selection_rank = selected["selection_rank"]
        observation_ordinal = selected["observation_selection_ordinal"]
        observer_ordinal = selected["observer_selection_ordinal"]
        observer_reuse = selected["observer_reuse_justified"]
        distinct_view = selected["distinct_additional_view_justified"]
    else:
        decision = "skipped"
        reason = prefilter_reason or _skip_reason(row, state=state, policy=policy)
        selection_round = "not_selected"
        selection_rank = None
        observation_ordinal = None
        observer_ordinal = None
        observer_reuse = False
        distinct_view = False
    taxon = str(row["accepted_taxon_key"])
    output: dict[str, object] = {
        "schema_version": PROVISIONAL_SELECTION_SCHEMA_VERSION,
        **row,
        "decision": decision,
        "decision_reason": reason,
        "selection_round": selection_round,
        "selection_rank": selection_rank,
        "observation_selection_ordinal": observation_ordinal,
        "observer_selection_ordinal": observer_ordinal,
        "observer_reuse_justified": observer_reuse,
        "distinct_additional_view_justified": distinct_view,
        "species_quota": policy.quota_per_species,
        "species_selected_count": state.species_counts[taxon],
        "selection_policy_fingerprint": policy.fingerprint,
    }
    output["decision_fingerprint"] = canonical_semantic_fingerprint(output)
    return output


def _skip_reason(
    row: Mapping[str, object],
    *,
    state: _SelectionState,
    policy: ProvisionalSelectionPolicy,
) -> str:
    taxon = str(row["accepted_taxon_key"])
    if str(row["duplicate_group_id"]) in state.used_duplicate_groups:
        return "duplicate_group_already_selected"
    observation = str(row["reference_observation_id"])
    if state.observation_counts[observation]:
        if row["distinct_view_evidence"] == "none":
            return "observation_already_selected_without_distinct_view"
        if str(row["documented_view"]) in state.observation_views[observation]:
            return "documented_view_already_selected"
        return "observation_distinct_view_limit_reached"
    if state.species_counts[taxon] >= policy.quota_per_species:
        return "species_quota_reached"
    return "insufficient_independent_support"


def _required_text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


__all__ = [
    "DISTINCT_VIEW_EVIDENCE",
    "PROVISIONAL_SELECTION_CANDIDATE_SCHEMA",
    "PROVISIONAL_SELECTION_DECISION_SCHEMA",
    "PROVISIONAL_SELECTION_DECISIONS_FILE",
    "PROVISIONAL_SELECTIONS_FILE",
    "ProvisionalSelectionPolicy",
    "ProvisionalSelectionResult",
    "select_independent_provisional_support",
]
