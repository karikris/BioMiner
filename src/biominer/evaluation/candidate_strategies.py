"""Evidence-bound metrics for target-preserving candidate schedules."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isclose, isfinite
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.family_geo_candidates import (
    validate_family_geo_candidate_sets,
)
from biominer.candidates.strategy_ablation import (
    CANDIDATE_STRATEGIES,
    validate_candidate_strategy_plans,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.storage.parquet import write_parquet


CANDIDATE_STRATEGY_METRIC_SCHEMA_VERSION = "candidate-strategy-metric-v1.0.0"
CANDIDATE_STRATEGY_METRICS_FILE = "candidate_strategy_metrics.parquet"
FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION = (
    "family-pruning-counterfactual-v1.0.0"
)
FAMILY_PRUNING_COUNTERFACTUAL_FILE = "family_pruning_counterfactual.parquet"

_LABEL_FIELDS = frozenset(
    {
        "source_candidate_set_id",
        "reviewed_accepted_taxon_key",
        "reviewed_family_key",
        "label_status",
        "label_source",
    }
)
_MEASUREMENT_FIELDS = frozenset(
    {
        "strategy_plan_id",
        "k",
        "evaluated_candidate_count",
        "dot_product_count",
        "reference_member_count",
        "elapsed_time_ms",
        "peak_memory_bytes",
        "cache_reused_reference_members",
        "cache_new_reference_members",
        "measurement_source",
    }
)
_FINGERPRINT_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_METRIC_ID_PATTERN = re.compile(r"candidate-strategy-metric:[0-9a-f]{64}\Z")
_COUNTERFACTUAL_ID_PATTERN = re.compile(
    r"family-pruning-counterfactual:[0-9a-f]{64}\Z"
)
_SORT = (
    "evaluation_run_id",
    "strategy_name",
    "source_candidate_set_id",
    "strategy_plan_id",
    "k",
)


def candidate_strategy_metric_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "strategy_metric_id": pl.String,
        "strategy_metric_fingerprint": pl.String,
        "evaluation_run_id": pl.String,
        "strategy_plan_id": pl.String,
        "strategy_plan_fingerprint": pl.String,
        "source_candidate_set_id": pl.String,
        "run_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "strategy_name": pl.String,
        "strategy_version": pl.String,
        "k": pl.UInt32,
        "label_status": pl.String,
        "label_source": pl.String,
        "target_accepted_taxon_key": pl.String,
        "reviewed_accepted_taxon_key": pl.String,
        "reviewed_family_key": pl.String,
        "target_rank": pl.UInt32,
        "species_rank": pl.UInt32,
        "first_family_rank": pl.UInt32,
        "target_candidate_recall_at_k": pl.Float64,
        "species_candidate_recall_at_k": pl.Float64,
        "family_candidate_recall_at_k": pl.Float64,
        "candidate_set_size": pl.UInt32,
        "evaluated_candidate_count": pl.UInt32,
        "dot_product_count": pl.UInt64,
        "reference_member_count": pl.UInt64,
        "elapsed_time_ms": pl.Float64,
        "peak_memory_bytes": pl.UInt64,
        "cache_reused_reference_members": pl.UInt64,
        "cache_new_reference_members": pl.UInt64,
        "cache_reuse_fraction": pl.Float64,
        "geography_status": pl.String,
        "no_geo": pl.Boolean,
        "family_counterfactual_status": pl.String,
        "wrong_family": pl.Boolean,
        "measurement_source": pl.String,
    }


def family_pruning_counterfactual_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "counterfactual_id": pl.String,
        "counterfactual_fingerprint": pl.String,
        "evaluation_run_id": pl.String,
        "source_candidate_set_id": pl.String,
        "source_candidate_set_fingerprint": pl.String,
        "run_id": pl.String,
        "flickr_photo_id": pl.String,
        "organism_unit_id": pl.String,
        "scoring_stage": pl.String,
        "label_status": pl.String,
        "label_source": pl.String,
        "reviewed_accepted_taxon_key": pl.String,
        "reviewed_family_key": pl.String,
        "reviewed_species_candidate_row_fingerprint": pl.String,
        "complete_union_size": pl.UInt32,
        "hard_family_pool_size": pl.UInt32,
        "hard_family_pool_keys": pl.List(pl.String),
        "hard_family_keys": pl.List(pl.String),
        "reviewed_species_in_complete_union": pl.Boolean,
        "reviewed_species_in_hard_family_pool": pl.Boolean,
        "correct_species_family_evidence_status": pl.String,
        "correct_species_family_priority_match": pl.Boolean,
        "loss_eligible": pl.Boolean,
        "correct_species_lost": pl.Boolean,
        "geography_status": pl.String,
        "no_geo": pl.Boolean,
        "counterfactual_policy": pl.String,
        "counterfactual_policy_fingerprint": pl.String,
    }


def build_candidate_strategy_metrics(
    candidate_sets: pl.DataFrame,
    strategy_plans: Sequence[pl.DataFrame],
    *,
    evaluation_run_id: str,
    labels: Sequence[Mapping[str, object]],
    measurements: Sequence[Mapping[str, object]],
    ks: Sequence[int],
) -> pl.DataFrame:
    """Build one measured evaluation row per strategy plan and cutoff."""

    validate_family_geo_candidate_sets(candidate_sets)
    run_id = _required_text(evaluation_run_id, field="evaluation_run_id")
    cutoffs = _normalize_ks(ks)
    plans = _normalize_plans(strategy_plans, candidate_sets)
    label_by_set = _normalize_labels(labels)
    measurement_by_key = _normalize_measurements(measurements)
    source_by_set = {
        str(candidate_set_id): group.sort(
            "candidate_priority", "candidate_accepted_taxon_key"
        )
        for (candidate_set_id,), group in candidate_sets.group_by(
            "candidate_set_id", maintain_order=True
        )
    }
    expected_sets = set(source_by_set)
    if set(label_by_set) != expected_sets:
        raise ValueError("labels must cover every source candidate set exactly once")

    rows: list[dict[str, object]] = []
    expected_measurements: set[tuple[str, int]] = set()
    for plan in plans:
        for (plan_id,), group in plan.group_by(
            "strategy_plan_id", maintain_order=True
        ):
            ordered = group.sort("strategy_priority")
            first = ordered.row(0, named=True)
            source_set_id = str(first["source_candidate_set_id"])
            source = source_by_set[source_set_id]
            label = label_by_set[source_set_id]
            plan_id_text = str(plan_id)
            for k in cutoffs:
                measurement_key = (plan_id_text, k)
                expected_measurements.add(measurement_key)
                try:
                    measurement = measurement_by_key[measurement_key]
                except KeyError as exc:
                    raise ValueError(
                        f"missing measurement for strategy plan {plan_id_text!r} at k={k}"
                    ) from exc
                rows.append(
                    _metric_row(
                        evaluation_run_id=run_id,
                        plan=ordered,
                        source=source,
                        label=label,
                        measurement=measurement,
                        k=k,
                    )
                )
    extras = set(measurement_by_key) - expected_measurements
    if extras:
        raise ValueError("measurements contain plan/cutoff rows outside the evaluation")
    frame = pl.DataFrame(
        rows,
        schema=candidate_strategy_metric_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)
    validate_candidate_strategy_metrics(frame)
    return frame


def validate_candidate_strategy_metrics(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("candidate strategy metrics must be a Polars DataFrame")
    if frame.schema != candidate_strategy_metric_schema():
        raise ValueError("candidate strategy metric schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("candidate strategy metrics are not canonically sorted")
    if frame.height != frame["strategy_metric_id"].n_unique():
        raise ValueError("candidate strategy metric IDs are not unique")
    for row in frame.to_dicts():
        _validate_metric_row(row)


def write_candidate_strategy_metrics(
    frame: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    validate_candidate_strategy_metrics(frame)
    destination = Path(output_path)
    if destination.suffix.casefold() != ".parquet":
        destination /= CANDIDATE_STRATEGY_METRICS_FILE
    return write_parquet(frame, destination)


def build_family_pruning_counterfactual(
    candidate_sets: pl.DataFrame,
    *,
    evaluation_run_id: str,
    labels: Sequence[Mapping[str, object]],
) -> pl.DataFrame:
    """Measure reviewed-species loss under a hypothetical hard family gate."""

    validate_family_geo_candidate_sets(candidate_sets)
    run_id = _required_text(evaluation_run_id, field="evaluation_run_id")
    label_by_set = _normalize_labels(labels)
    groups = {
        str(candidate_set_id): group.sort(
            "candidate_priority", "candidate_accepted_taxon_key"
        )
        for (candidate_set_id,), group in candidate_sets.group_by(
            "candidate_set_id", maintain_order=True
        )
    }
    if set(label_by_set) != set(groups):
        raise ValueError("labels must cover every source candidate set exactly once")
    policy = "hard_family_priority_match_only"
    policy_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION,
            "policy": policy,
            "membership_rule": (
                "family_evidence_status=available and family_priority_match=true"
            ),
            "production_candidate_membership_changed": False,
        }
    )
    rows = [
        _family_pruning_row(
            evaluation_run_id=run_id,
            source=groups[set_id],
            label=label_by_set[set_id],
            policy=policy,
            policy_fingerprint=policy_fingerprint,
        )
        for set_id in sorted(groups)
    ]
    frame = pl.DataFrame(
        rows,
        schema=family_pruning_counterfactual_schema(),
        orient="row",
        strict=True,
    ).sort("evaluation_run_id", "source_candidate_set_id")
    validate_family_pruning_counterfactual(frame)
    return frame


def validate_family_pruning_counterfactual(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("family-pruning counterfactual must be a Polars DataFrame")
    if frame.schema != family_pruning_counterfactual_schema():
        raise ValueError("family-pruning counterfactual schema mismatch")
    if not frame.equals(frame.sort("evaluation_run_id", "source_candidate_set_id")):
        raise ValueError("family-pruning counterfactual is not canonically sorted")
    if frame.height != frame["counterfactual_id"].n_unique():
        raise ValueError("family-pruning counterfactual IDs are not unique")
    for row in frame.to_dicts():
        _validate_family_pruning_row(row)


def summarize_family_pruning_counterfactual(
    frame: pl.DataFrame,
) -> dict[str, object]:
    """Return denominator-explicit aggregate and no-geo loss statistics."""

    validate_family_pruning_counterfactual(frame)
    eligible = frame.filter(pl.col("loss_eligible"))
    no_geo = eligible.filter(pl.col("no_geo"))
    summary: dict[str, object] = {
        "schema_version": "family-pruning-counterfactual-summary-v1.0.0",
        "evaluated_label_count": frame.height,
        "eligible_correct_species_count": eligible.height,
        "correct_species_lost_count": int(eligible["correct_species_lost"].sum()),
        "correct_species_lost_rate": _loss_rate(eligible),
        "wrong_family_evidence_count": frame.filter(
            pl.col("correct_species_family_priority_match") == False  # noqa: E712
        ).height,
        "reviewed_species_missing_from_complete_union_count": frame.filter(
            ~pl.col("reviewed_species_in_complete_union")
        ).height,
        "mean_complete_union_size": _column_mean(frame, "complete_union_size"),
        "mean_hard_family_pool_size": _column_mean(
            frame, "hard_family_pool_size"
        ),
        "no_geo": {
            "eligible_correct_species_count": no_geo.height,
            "correct_species_lost_count": int(no_geo["correct_species_lost"].sum()),
            "correct_species_lost_rate": _loss_rate(no_geo),
        },
        "production_candidate_membership_changed": False,
    }
    summary["summary_fingerprint"] = canonical_semantic_fingerprint(summary)
    return summary


def write_family_pruning_counterfactual(
    frame: pl.DataFrame,
    output_path: str | Path,
) -> Path:
    validate_family_pruning_counterfactual(frame)
    destination = Path(output_path)
    if destination.suffix.casefold() != ".parquet":
        destination /= FAMILY_PRUNING_COUNTERFACTUAL_FILE
    return write_parquet(frame, destination)


def _normalize_plans(
    strategy_plans: Sequence[pl.DataFrame],
    candidate_sets: pl.DataFrame,
) -> list[pl.DataFrame]:
    if isinstance(strategy_plans, str | bytes) or not isinstance(
        strategy_plans, Sequence
    ):
        raise TypeError("strategy_plans must be a sequence of Polars DataFrames")
    if not strategy_plans:
        raise ValueError("at least one candidate strategy plan is required")
    plans: list[pl.DataFrame] = []
    strategies: set[str] = set()
    for plan in strategy_plans:
        validate_candidate_strategy_plans(plan, candidate_sets)
        strategy = str(plan["strategy_name"][0])
        if strategy in strategies:
            raise ValueError("strategy_plans contain a duplicate strategy")
        strategies.add(strategy)
        plans.append(plan)
    return sorted(plans, key=lambda frame: str(frame["strategy_name"][0]))


def _normalize_labels(
    labels: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, str]]:
    if isinstance(labels, str | bytes) or not isinstance(labels, Sequence):
        raise TypeError("labels must be a sequence of mappings")
    output: dict[str, dict[str, str]] = {}
    for raw in labels:
        if not isinstance(raw, Mapping) or set(raw) != _LABEL_FIELDS:
            raise ValueError("candidate strategy label fields do not match the contract")
        normalized = {
            field: _required_text(raw[field], field=field) for field in _LABEL_FIELDS
        }
        set_id = normalized["source_candidate_set_id"]
        if set_id in output:
            raise ValueError("labels contain a duplicate source candidate set")
        output[set_id] = normalized
    return output


def _normalize_measurements(
    measurements: Sequence[Mapping[str, object]],
) -> dict[tuple[str, int], dict[str, object]]:
    if isinstance(measurements, str | bytes) or not isinstance(
        measurements, Sequence
    ):
        raise TypeError("measurements must be a sequence of mappings")
    output: dict[tuple[str, int], dict[str, object]] = {}
    for raw in measurements:
        if not isinstance(raw, Mapping) or set(raw) != _MEASUREMENT_FIELDS:
            raise ValueError("candidate strategy measurement fields do not match the contract")
        plan_id = _required_text(raw["strategy_plan_id"], field="strategy_plan_id")
        k = _positive_int(raw["k"], field="k")
        normalized: dict[str, object] = {
            "strategy_plan_id": plan_id,
            "k": k,
            "evaluated_candidate_count": _nonnegative_int(
                raw["evaluated_candidate_count"], field="evaluated_candidate_count"
            ),
            "dot_product_count": _nonnegative_int(
                raw["dot_product_count"], field="dot_product_count"
            ),
            "reference_member_count": _nonnegative_int(
                raw["reference_member_count"], field="reference_member_count"
            ),
            "elapsed_time_ms": _nonnegative_float(
                raw["elapsed_time_ms"], field="elapsed_time_ms"
            ),
            "peak_memory_bytes": _nonnegative_int(
                raw["peak_memory_bytes"], field="peak_memory_bytes"
            ),
            "cache_reused_reference_members": _nonnegative_int(
                raw["cache_reused_reference_members"],
                field="cache_reused_reference_members",
            ),
            "cache_new_reference_members": _nonnegative_int(
                raw["cache_new_reference_members"],
                field="cache_new_reference_members",
            ),
            "measurement_source": _required_text(
                raw["measurement_source"], field="measurement_source"
            ),
        }
        key = (plan_id, k)
        if key in output:
            raise ValueError("measurements contain a duplicate plan/cutoff row")
        output[key] = normalized
    return output


def _metric_row(
    *,
    evaluation_run_id: str,
    plan: pl.DataFrame,
    source: pl.DataFrame,
    label: Mapping[str, str],
    measurement: Mapping[str, object],
    k: int,
) -> dict[str, object]:
    first = plan.row(0, named=True)
    ordered_keys = [str(value) for value in plan["candidate_accepted_taxon_key"]]
    target_key = str(first["target_accepted_taxon_key"])
    species_key = label["reviewed_accepted_taxon_key"]
    reviewed_family_key = label["reviewed_family_key"]
    source_rows = {
        str(row["candidate_accepted_taxon_key"]): row for row in source.to_dicts()
    }
    family_keys = [source_rows[key]["family_key"] for key in ordered_keys]
    target_rank = _rank(ordered_keys, target_key)
    species_rank = _rank(ordered_keys, species_key)
    family_rank = _rank(family_keys, reviewed_family_key)
    candidate_set_size = len(ordered_keys)
    evaluated_count = min(k, candidate_set_size)
    if int(measurement["evaluated_candidate_count"]) != evaluated_count:
        raise ValueError(
            "measured candidate count must equal the plan prefix at the requested k"
        )
    reference_count = int(measurement["reference_member_count"])
    cache_reused = int(measurement["cache_reused_reference_members"])
    cache_new = int(measurement["cache_new_reference_members"])
    if cache_reused + cache_new != reference_count:
        raise ValueError("cache reuse counts must partition reference members")
    geography_status = _geography_status(plan)
    family_status = _family_counterfactual_status(
        source_rows.get(species_key)
    )
    base: dict[str, object] = {
        "schema_version": CANDIDATE_STRATEGY_METRIC_SCHEMA_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "strategy_plan_id": str(first["strategy_plan_id"]),
        "strategy_plan_fingerprint": str(first["strategy_plan_fingerprint"]),
        "source_candidate_set_id": str(first["source_candidate_set_id"]),
        "run_id": str(first["run_id"]),
        "flickr_photo_id": str(first["flickr_photo_id"]),
        "organism_unit_id": str(first["organism_unit_id"]),
        "scoring_stage": str(first["scoring_stage"]),
        "strategy_name": str(first["strategy_name"]),
        "strategy_version": str(first["strategy_version"]),
        "k": k,
        "label_status": label["label_status"],
        "label_source": label["label_source"],
        "target_accepted_taxon_key": target_key,
        "reviewed_accepted_taxon_key": species_key,
        "reviewed_family_key": reviewed_family_key,
        "target_rank": target_rank,
        "species_rank": species_rank,
        "first_family_rank": family_rank,
        "target_candidate_recall_at_k": _recalled(target_rank, k),
        "species_candidate_recall_at_k": _recalled(species_rank, k),
        "family_candidate_recall_at_k": _recalled(family_rank, k),
        "candidate_set_size": candidate_set_size,
        "evaluated_candidate_count": evaluated_count,
        "dot_product_count": int(measurement["dot_product_count"]),
        "reference_member_count": reference_count,
        "elapsed_time_ms": float(measurement["elapsed_time_ms"]),
        "peak_memory_bytes": int(measurement["peak_memory_bytes"]),
        "cache_reused_reference_members": cache_reused,
        "cache_new_reference_members": cache_new,
        "cache_reuse_fraction": (
            cache_reused / reference_count if reference_count else None
        ),
        "geography_status": geography_status,
        "no_geo": geography_status != "available",
        "family_counterfactual_status": family_status,
        "wrong_family": family_status == "wrong_family",
        "measurement_source": str(measurement["measurement_source"]),
    }
    fingerprint = canonical_semantic_fingerprint(base)
    return {
        **base,
        "strategy_metric_id": _prefixed_id("candidate-strategy-metric", fingerprint),
        "strategy_metric_fingerprint": fingerprint,
    }


def _family_pruning_row(
    *,
    evaluation_run_id: str,
    source: pl.DataFrame,
    label: Mapping[str, str],
    policy: str,
    policy_fingerprint: str,
) -> dict[str, object]:
    first = source.row(0, named=True)
    source_rows = {
        str(row["candidate_accepted_taxon_key"]): row for row in source.to_dicts()
    }
    reviewed_key = label["reviewed_accepted_taxon_key"]
    reviewed_species = source_rows.get(reviewed_key)
    hard_pool = source.filter(
        (pl.col("family_evidence_status") == "available")
        & pl.col("family_priority_match").fill_null(False)
    )
    hard_pool_keys = sorted(str(value) for value in hard_pool["candidate_accepted_taxon_key"])
    hard_family_keys = sorted(set(str(value) for value in hard_pool["family_key"]))
    in_complete_union = reviewed_species is not None
    in_hard_pool = reviewed_key in hard_pool_keys
    family_match = (
        reviewed_species["family_priority_match"]
        if reviewed_species is not None
        else None
    )
    geography_status = _geography_status(source)
    base: dict[str, object] = {
        "schema_version": FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION,
        "evaluation_run_id": evaluation_run_id,
        "source_candidate_set_id": str(first["candidate_set_id"]),
        "source_candidate_set_fingerprint": str(
            first["candidate_set_fingerprint"]
        ),
        "run_id": str(first["run_id"]),
        "flickr_photo_id": str(first["flickr_photo_id"]),
        "organism_unit_id": str(first["organism_unit_id"]),
        "scoring_stage": str(first["scoring_stage"]),
        "label_status": label["label_status"],
        "label_source": label["label_source"],
        "reviewed_accepted_taxon_key": reviewed_key,
        "reviewed_family_key": label["reviewed_family_key"],
        "reviewed_species_candidate_row_fingerprint": (
            reviewed_species["candidate_row_fingerprint"]
            if reviewed_species is not None
            else None
        ),
        "complete_union_size": source.height,
        "hard_family_pool_size": hard_pool.height,
        "hard_family_pool_keys": hard_pool_keys,
        "hard_family_keys": hard_family_keys,
        "reviewed_species_in_complete_union": in_complete_union,
        "reviewed_species_in_hard_family_pool": in_hard_pool,
        "correct_species_family_evidence_status": (
            str(reviewed_species["family_evidence_status"])
            if reviewed_species is not None
            else "reviewed_species_not_in_union"
        ),
        "correct_species_family_priority_match": family_match,
        "loss_eligible": in_complete_union,
        "correct_species_lost": in_complete_union and not in_hard_pool,
        "geography_status": geography_status,
        "no_geo": geography_status != "available",
        "counterfactual_policy": policy,
        "counterfactual_policy_fingerprint": policy_fingerprint,
    }
    fingerprint = canonical_semantic_fingerprint(base)
    return {
        **base,
        "counterfactual_id": _prefixed_id(
            "family-pruning-counterfactual", fingerprint
        ),
        "counterfactual_fingerprint": fingerprint,
    }


def _validate_metric_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != CANDIDATE_STRATEGY_METRIC_SCHEMA_VERSION:
        raise ValueError("unsupported candidate strategy metric schema version")
    if row["strategy_name"] not in CANDIDATE_STRATEGIES:
        raise ValueError("unsupported candidate strategy metric strategy")
    if not _METRIC_ID_PATTERN.fullmatch(str(row["strategy_metric_id"])):
        raise ValueError("candidate strategy metric ID is invalid")
    if not _FINGERPRINT_PATTERN.fullmatch(
        str(row["strategy_metric_fingerprint"])
    ):
        raise ValueError("candidate strategy metric fingerprint is invalid")
    k = int(row["k"])
    size = int(row["candidate_set_size"])
    if k <= 0 or size <= 0:
        raise ValueError("candidate strategy metric cutoff and set size must be positive")
    if int(row["evaluated_candidate_count"]) != min(k, size):
        raise ValueError("candidate strategy evaluated count is inconsistent")
    for rank_field, recall_field in (
        ("target_rank", "target_candidate_recall_at_k"),
        ("species_rank", "species_candidate_recall_at_k"),
        ("first_family_rank", "family_candidate_recall_at_k"),
    ):
        rank = row[rank_field]
        if rank is not None and not 1 <= int(rank) <= size:
            raise ValueError(f"invalid {rank_field}")
        if float(row[recall_field]) != _recalled(
            int(rank) if rank is not None else None, k
        ):
            raise ValueError(f"{recall_field} is inconsistent with its rank")
    reference_count = int(row["reference_member_count"])
    cache_reused = int(row["cache_reused_reference_members"])
    cache_new = int(row["cache_new_reference_members"])
    if cache_reused + cache_new != reference_count:
        raise ValueError("candidate strategy cache counts are inconsistent")
    expected_fraction = cache_reused / reference_count if reference_count else None
    actual_fraction = row["cache_reuse_fraction"]
    if expected_fraction is None:
        if actual_fraction is not None:
            raise ValueError("zero reference members require undefined cache reuse")
    elif actual_fraction is None or not isclose(
        float(actual_fraction), expected_fraction, abs_tol=1e-12
    ):
        raise ValueError("candidate strategy cache reuse fraction is inconsistent")
    if bool(row["no_geo"]) != (row["geography_status"] != "available"):
        raise ValueError("candidate strategy no-geo status is inconsistent")
    if bool(row["wrong_family"]) != (
        row["family_counterfactual_status"] == "wrong_family"
    ):
        raise ValueError("candidate strategy wrong-family status is inconsistent")
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"strategy_metric_id", "strategy_metric_fingerprint"}
    }
    expected_fingerprint = canonical_semantic_fingerprint(payload)
    if row["strategy_metric_fingerprint"] != expected_fingerprint:
        raise ValueError("candidate strategy metric fingerprint does not match row")
    if row["strategy_metric_id"] != _prefixed_id(
        "candidate-strategy-metric", expected_fingerprint
    ):
        raise ValueError("candidate strategy metric ID does not match fingerprint")


def _validate_family_pruning_row(row: Mapping[str, object]) -> None:
    if row["schema_version"] != FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION:
        raise ValueError("unsupported family-pruning counterfactual schema")
    if not _COUNTERFACTUAL_ID_PATTERN.fullmatch(str(row["counterfactual_id"])):
        raise ValueError("family-pruning counterfactual ID is invalid")
    if not _FINGERPRINT_PATTERN.fullmatch(
        str(row["counterfactual_fingerprint"])
    ):
        raise ValueError("family-pruning counterfactual fingerprint is invalid")
    hard_pool_keys = list(row["hard_family_pool_keys"])
    hard_family_keys = list(row["hard_family_keys"])
    if hard_pool_keys != sorted(set(hard_pool_keys)):
        raise ValueError("hard-family pool keys are not canonical")
    if hard_family_keys != sorted(set(hard_family_keys)):
        raise ValueError("hard-family keys are not canonical")
    if int(row["hard_family_pool_size"]) != len(hard_pool_keys):
        raise ValueError("hard-family pool size does not match its keys")
    in_complete_union = bool(row["reviewed_species_in_complete_union"])
    in_hard_pool = bool(row["reviewed_species_in_hard_family_pool"])
    if in_hard_pool and not in_complete_union:
        raise ValueError("hard-family membership requires complete-union membership")
    if bool(row["loss_eligible"]) != in_complete_union:
        raise ValueError("family-pruning loss eligibility is inconsistent")
    if bool(row["correct_species_lost"]) != (
        in_complete_union and not in_hard_pool
    ):
        raise ValueError("family-pruning correct-species loss is inconsistent")
    if bool(row["no_geo"]) != (row["geography_status"] != "available"):
        raise ValueError("family-pruning no-geo status is inconsistent")
    payload = {
        key: value
        for key, value in row.items()
        if key not in {"counterfactual_id", "counterfactual_fingerprint"}
    }
    fingerprint = canonical_semantic_fingerprint(payload)
    if row["counterfactual_fingerprint"] != fingerprint:
        raise ValueError("family-pruning counterfactual fingerprint does not match row")
    if row["counterfactual_id"] != _prefixed_id(
        "family-pruning-counterfactual", fingerprint
    ):
        raise ValueError("family-pruning counterfactual ID does not match fingerprint")


def _family_counterfactual_status(
    reviewed_species: Mapping[str, object] | None,
) -> str:
    if reviewed_species is None:
        return "reviewed_species_not_in_union"
    if reviewed_species["family_evidence_status"] != "available":
        return "family_evidence_unavailable"
    if reviewed_species["family_priority_match"] is True:
        return "correct_family"
    if reviewed_species["family_priority_match"] is False:
        return "wrong_family"
    return "family_priority_unavailable"


def _geography_status(plan: pl.DataFrame) -> str:
    statuses = set(plan["geographic_evidence_status"].to_list())
    if "available" in statuses:
        return "available"
    if statuses == {"not_applicable"}:
        return "not_applicable"
    return "unavailable"


def _rank(values: Sequence[object], wanted: object) -> int | None:
    try:
        return values.index(wanted) + 1
    except ValueError:
        return None


def _recalled(rank: int | None, k: int) -> float:
    return 1.0 if rank is not None and rank <= k else 0.0


def _loss_rate(frame: pl.DataFrame) -> float | None:
    if frame.is_empty():
        return None
    return float(frame["correct_species_lost"].sum()) / frame.height


def _column_mean(frame: pl.DataFrame, column: str) -> float | None:
    value = frame[column].mean() if frame.height else None
    return float(value) if value is not None else None


def _normalize_ks(ks: Sequence[int]) -> tuple[int, ...]:
    if isinstance(ks, str | bytes) or not isinstance(ks, Sequence):
        raise TypeError("ks must be a sequence of positive integers")
    values = tuple(_positive_int(value, field="k") for value in ks)
    if not values:
        raise ValueError("at least one candidate strategy cutoff is required")
    if values != tuple(sorted(set(values))):
        raise ValueError("candidate strategy cutoffs must be unique and sorted")
    return values


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be non-empty")
    return text


def _positive_int(value: object, *, field: str) -> int:
    number = _nonnegative_int(value, field=field)
    if number <= 0:
        raise ValueError(f"{field} must be positive")
    return number


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a non-negative integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a non-negative integer") from exc
    if number < 0 or number != value:
        raise ValueError(f"{field} must be a non-negative integer")
    return number


def _nonnegative_float(value: object, *, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be a finite non-negative number") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{field} must be a finite non-negative number")
    return number


def _prefixed_id(prefix: str, fingerprint: str) -> str:
    return f"{prefix}:{fingerprint.removeprefix('sha256:')}"


__all__ = [
    "CANDIDATE_STRATEGY_METRICS_FILE",
    "CANDIDATE_STRATEGY_METRIC_SCHEMA_VERSION",
    "FAMILY_PRUNING_COUNTERFACTUAL_FILE",
    "FAMILY_PRUNING_COUNTERFACTUAL_SCHEMA_VERSION",
    "build_candidate_strategy_metrics",
    "build_family_pruning_counterfactual",
    "candidate_strategy_metric_schema",
    "family_pruning_counterfactual_schema",
    "summarize_family_pruning_counterfactual",
    "validate_candidate_strategy_metrics",
    "validate_family_pruning_counterfactual",
    "write_candidate_strategy_metrics",
    "write_family_pruning_counterfactual",
]
