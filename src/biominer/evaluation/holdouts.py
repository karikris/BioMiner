"""Frozen balanced and natural-stream target-evaluation holdouts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from fractions import Fraction
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
from typing import Any
from uuid import uuid4

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.leakage import (
    EVALUATION_LEAKAGE_REGISTER_FILE,
    EvaluationLeakageAudit,
    validate_reference_and_holdout_leakage,
)
from biominer.evaluation.labels import (
    normalize_reviewed_label_frame,
    validate_reviewed_label_frame,
)
from biominer.evaluation.sampling import (
    EVALUATION_SAMPLING_FRAME_SCHEMA,
    validate_evaluation_sampling_frame,
)
from biominer.reports.flickr_fetch import current_git_sha
from biominer.storage.parquet import write_parquet


NATURAL_STREAM_SELECTION_SCHEMA_VERSION = "natural-stream-selection-v1.0.0"
FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION = "frozen-evaluation-holdout-v1.0.0"
FROZEN_EVALUATION_HOLDOUT_REPORT_SCHEMA_VERSION = (
    "frozen-evaluation-holdout-report-v1.1.0"
)

NATURAL_STREAM_SELECTION_FILE = "natural_stream_selection.parquet"
BALANCED_CHALLENGE_HOLDOUT_FILE = "balanced_challenge_holdout.parquet"
NATURAL_STREAM_HOLDOUT_FILE = "natural_stream_holdout.parquet"
FROZEN_EVALUATION_HOLDOUT_REPORT_FILE = "frozen_evaluation_holdouts_report.json"
FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE = "frozen_evaluation_holdouts_report.md"

BALANCED_CHALLENGE_CATEGORIES = (
    "verified_target",
    "other_papilio",
    "other_papilionidae",
    "moths_and_other_insects",
    "artifacts",
    "pinned_specimens",
    "caterpillars",
)
BALANCED_CHALLENGE_CATEGORY_SET = frozenset(BALANCED_CHALLENGE_CATEGORIES)

FORBIDDEN_EVALUATION_USAGE_ROLES = frozenset(
    {
        "support",
        "support_bank",
        "support_train",
        "classifier_training",
        "training",
        "model_selection",
        "calibration",
        "threshold_selection",
    }
)
ALLOWED_EVALUATION_USAGE_ROLES = FORBIDDEN_EVALUATION_USAGE_ROLES | {
    "evaluation",
    "final_test",
    "balanced_challenge",
    "natural_stream",
}

DEFAULT_NATURAL_STRATIFICATION_FIELDS = (
    "geo_stratum",
    "primary_query_tier",
    "primary_query_term",
    "initial_visual_domain",
    "initial_reference_score_tail",
    "false_positive_genus_stratum",
    "text_image_reference_disagreement",
)
_ALLOWED_NATURAL_STRATIFICATION_FIELDS = frozenset(
    {
        "source",
        "geo_stratum",
        "primary_query_tier",
        "primary_query_term",
        "primary_query_field",
        "year_stratum",
        "yoloe_route",
        "subject_area_band",
        "initial_visual_domain",
        "initial_reference_score_band",
        "initial_reference_score_tail",
        "initial_competitor_margin_band",
        "initial_competitor_margin_tail",
        "false_positive_genus_stratum",
        "visual_input_disagreement_band",
        "text_image_reference_disagreement",
    }
)

USAGE_ASSIGNMENT_SCHEMA: dict[str, pl.DataType] = {
    "sampling_unit_id": pl.String,
    "usage_role": pl.String,
}

NATURAL_STREAM_SELECTION_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "selection_version": pl.String,
    "selection_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "sampling_frame_fingerprint": pl.String,
    "random_seed": pl.UInt64,
    "requested_sample_size": pl.UInt32,
    "minimum_per_stratum": pl.UInt32,
    "stratification_fields": pl.List(pl.String),
    "sampling_unit_id": pl.String,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "source_record_hash": pl.String,
    "sampling_stratum_id": pl.String,
    "sampling_stratum_json": pl.String,
    "population_size": pl.UInt32,
    "eligible_population_size": pl.UInt32,
    "population_stratum_size": pl.UInt32,
    "sample_stratum_size": pl.UInt32,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "selection_rank_within_stratum": pl.UInt32,
    "selection_rank": pl.UInt32,
}

FROZEN_EVALUATION_HOLDOUT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "holdout_version": pl.String,
    "holdout_fingerprint": pl.String,
    "configuration_fingerprint": pl.String,
    "sampling_frame_fingerprint": pl.String,
    "reviewed_labels_fingerprint": pl.String,
    "selection_fingerprint": pl.String,
    "holdout_kind": pl.String,
    "target_scope_accepted_taxon_key": pl.String,
    "evaluation_item_id": pl.String,
    "sampling_unit_id": pl.String,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "source_record_hash": pl.String,
    "detection_id": pl.String,
    "crop_hash": pl.String,
    "evaluation_class": pl.String,
    "target_present": pl.Boolean,
    "accepted_taxon_key": pl.String,
    "scientific_name": pl.String,
    "is_butterfly": pl.Boolean,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "label_certainty": pl.String,
    "life_stage": pl.String,
    "visual_domain": pl.String,
    "route": pl.String,
    "geo_cluster_id": pl.String,
    "source_query_tier": pl.String,
    "source_query_term": pl.String,
    "duplicate_group_id": pl.String,
    "observer_owner_group_id": pl.String,
    "dataset_split": pl.String,
    "reviewer_id": pl.String,
    "second_review_status": pl.String,
    "sampling_stratum_id": pl.String,
    "population_stratum_size": pl.UInt32,
    "sample_stratum_size": pl.UInt32,
    "inclusion_probability": pl.Float64,
    "sampling_weight": pl.Float64,
    "selection_rank": pl.UInt32,
}

_HOLDOUT_KIND_BALANCED = "balanced_challenge"
_HOLDOUT_KIND_NATURAL = "natural_stream"
_FINAL_TEST_SPLIT = "final_test"
_LABEL_KEY = ["source", "flickr_photo_id"]
_ARTIFACT_VISUAL_DOMAINS = frozenset({"artwork", "logo", "tattoo", "unsuitable"})
_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class FrozenHoldoutConfig:
    """Versioned selection policy for both evaluation holdouts."""

    holdout_version: str
    target_accepted_taxon_key: str
    challenge_per_category: int
    natural_sample_size: int
    random_seed: int = 42
    minimum_per_natural_stratum: int = 1
    natural_stratification_fields: tuple[str, ...] = (
        DEFAULT_NATURAL_STRATIFICATION_FIELDS
    )
    allowed_label_certainties: tuple[str, ...] = ("high",)
    allowed_second_review_statuses: tuple[str, ...] = (
        "completed",
        "not_required",
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "holdout_version",
            _required_text(self.holdout_version, field="holdout_version"),
        )
        object.__setattr__(
            self,
            "target_accepted_taxon_key",
            _required_text(
                self.target_accepted_taxon_key,
                field="target_accepted_taxon_key",
            ),
        )
        for field in ("challenge_per_category", "natural_sample_size"):
            value = _positive_integer(getattr(self, field), field=field)
            object.__setattr__(self, field, value)
        object.__setattr__(
            self,
            "minimum_per_natural_stratum",
            _positive_integer(
                self.minimum_per_natural_stratum,
                field="minimum_per_natural_stratum",
            ),
        )
        object.__setattr__(
            self,
            "random_seed",
            _nonnegative_integer(
                self.random_seed,
                field="random_seed",
                maximum=2**64 - 1,
            ),
        )
        fields = tuple(dict.fromkeys(self.natural_stratification_fields))
        if not fields:
            raise ValueError("natural_stratification_fields must not be empty")
        invalid_fields = sorted(set(fields) - _ALLOWED_NATURAL_STRATIFICATION_FIELDS)
        if invalid_fields:
            raise ValueError(
                "unsupported or outcome-bearing natural stratification fields: "
                f"{invalid_fields}"
            )
        object.__setattr__(self, "natural_stratification_fields", fields)
        certainties = _required_text_tuple(
            self.allowed_label_certainties,
            field="allowed_label_certainties",
        )
        statuses = _required_text_tuple(
            self.allowed_second_review_statuses,
            field="allowed_second_review_statuses",
        )
        object.__setattr__(self, "allowed_label_certainties", certainties)
        object.__setattr__(self, "allowed_second_review_statuses", statuses)

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION,
                "holdout_version": self.holdout_version,
                "target_accepted_taxon_key": self.target_accepted_taxon_key,
                "challenge_per_category": self.challenge_per_category,
                "natural_sample_size": self.natural_sample_size,
                "random_seed": self.random_seed,
                "minimum_per_natural_stratum": (self.minimum_per_natural_stratum),
                "natural_stratification_fields": list(
                    self.natural_stratification_fields
                ),
                "allowed_label_certainties": list(self.allowed_label_certainties),
                "allowed_second_review_statuses": list(
                    self.allowed_second_review_statuses
                ),
            }
        )


@dataclass(frozen=True, slots=True)
class FrozenEvaluationHoldoutPublication:
    output_dir: Path
    balanced_challenge_path: Path
    natural_stream_path: Path
    leakage_register_path: Path
    report_json_path: Path
    report_markdown_path: Path
    report: Mapping[str, Any]


def empty_natural_stream_selection() -> pl.DataFrame:
    return pl.DataFrame(schema=NATURAL_STREAM_SELECTION_SCHEMA)


def empty_frozen_evaluation_holdout() -> pl.DataFrame:
    return pl.DataFrame(schema=FROZEN_EVALUATION_HOLDOUT_SCHEMA)


def normalize_usage_assignments(
    assignments: pl.DataFrame | None,
) -> pl.DataFrame:
    if assignments is None:
        return pl.DataFrame(schema=USAGE_ASSIGNMENT_SCHEMA)
    if not isinstance(assignments, pl.DataFrame):
        raise TypeError("usage assignments must be a Polars DataFrame")
    missing = sorted(set(USAGE_ASSIGNMENT_SCHEMA) - set(assignments.columns))
    if missing:
        raise ValueError(f"usage assignments are missing columns: {missing}")
    frame = assignments.select(list(USAGE_ASSIGNMENT_SCHEMA)).cast(
        USAGE_ASSIGNMENT_SCHEMA
    )
    _require_nonblank(frame, list(USAGE_ASSIGNMENT_SCHEMA))
    frame = frame.unique().sort("sampling_unit_id", "usage_role")
    invalid = sorted(
        set(frame["usage_role"].to_list()) - ALLOWED_EVALUATION_USAGE_ROLES
    )
    if invalid:
        raise ValueError(f"unsupported evaluation usage roles: {invalid}")
    return frame


def select_natural_stream_candidates(
    sampling_frame: pl.DataFrame,
    config: FrozenHoldoutConfig,
    *,
    usage_assignments: pl.DataFrame | None = None,
    additionally_excluded_sampling_unit_ids: Collection[str] = (),
) -> pl.DataFrame:
    """Select the natural stream without accepting any label input."""

    _require_config(config)
    validate_evaluation_sampling_frame(sampling_frame)
    assignments = normalize_usage_assignments(usage_assignments)
    forbidden_ids = set(
        assignments.filter(
            pl.col("usage_role").is_in(FORBIDDEN_EVALUATION_USAGE_ROLES)
        )["sampling_unit_id"].to_list()
    )
    forbidden_ids.update(
        _required_text(value, field="excluded sampling_unit_id")
        for value in additionally_excluded_sampling_unit_ids
    )
    population_size = sampling_frame.height
    eligible = sampling_frame.filter(
        ~pl.col("sampling_unit_id").is_in(sorted(forbidden_ids))
    )
    if config.natural_sample_size > eligible.height:
        raise ValueError(
            "natural_sample_size exceeds the eligible candidate population: "
            f"requested={config.natural_sample_size}, eligible={eligible.height}"
        )
    sampling_frame_fingerprint = _sampling_frame_fingerprint(sampling_frame)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    stratum_json_by_id: dict[str, str] = {}
    for row in eligible.iter_rows(named=True):
        values = {field: row[field] for field in config.natural_stratification_fields}
        stratum_json = json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
        )
        stratum_id = "natural-stratum:" + canonical_semantic_fingerprint(
            {
                "fields": list(config.natural_stratification_fields),
                "values": values,
            }
        ).removeprefix("sha256:")
        grouped[stratum_id].append(row)
        stratum_json_by_id[stratum_id] = stratum_json
    counts = {stratum_id: len(rows) for stratum_id, rows in grouped.items()}
    quotas = _allocate_stratified_sample(
        counts,
        sample_size=config.natural_sample_size,
        minimum_per_stratum=config.minimum_per_natural_stratum,
    )
    selected: list[dict[str, object]] = []
    for stratum_id in sorted(grouped):
        ranked = sorted(
            grouped[stratum_id],
            key=lambda row: _selection_key(
                config.random_seed,
                str(row["sampling_unit_id"]),
                purpose="natural_stream",
            ),
        )
        sample_count = quotas[stratum_id]
        population_count = len(ranked)
        probability = sample_count / population_count
        weight = population_count / sample_count
        for within_rank, row in enumerate(ranked[:sample_count], start=1):
            selected.append(
                {
                    "schema_version": NATURAL_STREAM_SELECTION_SCHEMA_VERSION,
                    "selection_version": config.holdout_version,
                    "selection_fingerprint": "sha256:" + "0" * 64,
                    "configuration_fingerprint": config.fingerprint,
                    "sampling_frame_fingerprint": sampling_frame_fingerprint,
                    "random_seed": config.random_seed,
                    "requested_sample_size": config.natural_sample_size,
                    "minimum_per_stratum": config.minimum_per_natural_stratum,
                    "stratification_fields": list(config.natural_stratification_fields),
                    "sampling_unit_id": row["sampling_unit_id"],
                    "source": row["source"],
                    "flickr_photo_id": row["flickr_photo_id"],
                    "source_record_hash": row["source_record_hash"],
                    "sampling_stratum_id": stratum_id,
                    "sampling_stratum_json": stratum_json_by_id[stratum_id],
                    "population_size": population_size,
                    "eligible_population_size": eligible.height,
                    "population_stratum_size": population_count,
                    "sample_stratum_size": sample_count,
                    "inclusion_probability": probability,
                    "sampling_weight": weight,
                    "selection_rank_within_stratum": within_rank,
                    "selection_rank": 0,
                }
            )
    selected.sort(
        key=lambda row: _selection_key(
            config.random_seed,
            str(row["sampling_unit_id"]),
            purpose="natural_stream_global",
        )
    )
    for rank, row in enumerate(selected, start=1):
        row["selection_rank"] = rank
    selection_fingerprint = _rows_fingerprint(
        NATURAL_STREAM_SELECTION_SCHEMA_VERSION,
        selected,
        excluded_fields={"selection_fingerprint"},
    )
    for row in selected:
        row["selection_fingerprint"] = selection_fingerprint
    frame = pl.DataFrame(
        selected,
        schema=NATURAL_STREAM_SELECTION_SCHEMA,
    ).sort("selection_rank")
    validate_natural_stream_selection(frame)
    return frame


def validate_natural_stream_selection(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("natural-stream selection must be a Polars DataFrame")
    if dict(frame.schema) != NATURAL_STREAM_SELECTION_SCHEMA:
        raise ValueError("natural-stream selection physical schema mismatch")
    if frame.is_empty():
        raise ValueError("natural-stream selection must not be empty")
    if _single_text(frame, "schema_version") != NATURAL_STREAM_SELECTION_SCHEMA_VERSION:
        raise ValueError("natural-stream selection schema version is incompatible")
    if not frame.equals(frame.sort("selection_rank")):
        raise ValueError("natural-stream selection is not sorted by selection_rank")
    _require_unique(frame, ["sampling_unit_id"], label="natural-stream selection")
    ranks = frame["selection_rank"].to_list()
    if ranks != list(range(1, frame.height + 1)):
        raise ValueError("natural-stream selection ranks must be contiguous")
    if frame["requested_sample_size"].n_unique() != 1:
        raise ValueError("natural-stream requested sample size is inconsistent")
    if int(frame["requested_sample_size"][0]) != frame.height:
        raise ValueError("natural-stream selection row count is inconsistent")
    population_size = _single_integer(frame, "population_size")
    eligible_population_size = _single_integer(frame, "eligible_population_size")
    minimum_per_stratum = _single_integer(frame, "minimum_per_stratum")
    if not 0 < eligible_population_size <= population_size:
        raise ValueError("natural-stream population sizes are inconsistent")
    if minimum_per_stratum <= 0:
        raise ValueError("natural-stream minimum_per_stratum must be positive")
    if frame["stratification_fields"].n_unique() != 1:
        raise ValueError("natural-stream stratification fields are inconsistent")
    stratification_fields = frame["stratification_fields"].to_list()[0]
    if not isinstance(stratification_fields, list):
        raise TypeError("natural-stream stratification fields must be a list")
    if not stratification_fields or any(
        not isinstance(field, str) or not field.strip()
        for field in stratification_fields
    ):
        raise ValueError("natural-stream stratification fields are invalid")
    if len(set(stratification_fields)) != len(stratification_fields):
        raise ValueError("natural-stream stratification fields must be unique")
    if frame.filter(
        (pl.col("inclusion_probability") <= 0.0)
        | (pl.col("inclusion_probability") > 1.0)
        | ~pl.col("inclusion_probability").is_finite()
        | (pl.col("sampling_weight") <= 0.0)
        | ~pl.col("sampling_weight").is_finite()
    ).height:
        raise ValueError("natural-stream sampling probabilities are invalid")
    if frame.filter(
        (pl.col("sampling_weight") - (1.0 / pl.col("inclusion_probability"))).abs()
        > 1e-12
    ).height:
        raise ValueError(
            "natural-stream sampling weights are not inverse probabilities"
        )
    for _, group in frame.group_by("sampling_stratum_id"):
        population_count = int(group["population_stratum_size"][0])
        sample_count = int(group["sample_stratum_size"][0])
        if group.height != sample_count or sample_count > population_count:
            raise ValueError("natural-stream stratum counts are inconsistent")
        if sample_count < min(minimum_per_stratum, population_count):
            raise ValueError("natural-stream stratum minimum is not satisfied")
        if group["population_stratum_size"].n_unique() != 1:
            raise ValueError("natural-stream population stratum size is inconsistent")
        if group["sample_stratum_size"].n_unique() != 1:
            raise ValueError("natural-stream sample stratum size is inconsistent")
        within_ranks = group["selection_rank_within_stratum"].sort().to_list()
        if within_ranks != list(range(1, sample_count + 1)):
            raise ValueError("natural-stream within-stratum ranks are inconsistent")
        stratum_json = _single_text(group, "sampling_stratum_json")
        try:
            values = json.loads(stratum_json)
        except json.JSONDecodeError as exc:
            raise ValueError("natural-stream stratum JSON is invalid") from exc
        if not isinstance(values, dict) or list(sorted(values)) != list(
            sorted(stratification_fields)
        ):
            raise ValueError("natural-stream stratum fields are inconsistent")
        expected_stratum_id = "natural-stratum:" + canonical_semantic_fingerprint(
            {"fields": stratification_fields, "values": values}
        ).removeprefix("sha256:")
        if _single_text(group, "sampling_stratum_id") != expected_stratum_id:
            raise ValueError("natural-stream sampling_stratum_id is invalid")
        expected_probability = sample_count / population_count
        if group.filter(
            (pl.col("inclusion_probability") - expected_probability).abs() > 1e-12
        ).height:
            raise ValueError("natural-stream stratum probability is inconsistent")
    stratum_population = (
        frame.group_by("sampling_stratum_id").first()["population_stratum_size"].sum()
    )
    if int(stratum_population) != eligible_population_size:
        raise ValueError("natural-stream strata do not cover the eligible population")
    selection_fingerprint = _single_text(frame, "selection_fingerprint")
    expected = _rows_fingerprint(
        NATURAL_STREAM_SELECTION_SCHEMA_VERSION,
        frame.iter_rows(named=True),
        excluded_fields={"selection_fingerprint"},
    )
    if selection_fingerprint != expected:
        raise ValueError("natural-stream selection_fingerprint is invalid")


def write_natural_stream_selection(
    selection: pl.DataFrame,
    output: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Publish and round-trip verify the outcome-blind selection manifest."""

    validate_natural_stream_selection(selection)
    destination = Path(output)
    if destination.suffix.casefold() != ".parquet":
        destination /= NATURAL_STREAM_SELECTION_FILE
    written = write_parquet(selection, destination, overwrite=overwrite)
    loaded = pl.read_parquet(written)
    validate_natural_stream_selection(loaded)
    if not selection.equals(loaded):
        raise ValueError("natural-stream selection Parquet round-trip mismatch")
    _log_event(
        "natural_stream_selection_written",
        path=str(written),
        row_count=selection.height,
        byte_count=written.stat().st_size,
        sha256=_file_sha256(written),
        selection_fingerprint=_single_text(selection, "selection_fingerprint"),
    )
    return written


def load_natural_stream_selection(path: str | Path) -> pl.DataFrame:
    selection = pl.read_parquet(path)
    validate_natural_stream_selection(selection)
    return selection


def build_balanced_challenge_holdout(
    sampling_frame: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    config: FrozenHoldoutConfig,
    *,
    usage_assignments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build an intentionally label-balanced, review-complete challenge set."""

    _require_config(config)
    joined, labels_fingerprint, sampling_fingerprint = _eligible_reviewed_candidates(
        sampling_frame,
        reviewed_labels,
        config,
        usage_assignments=usage_assignments,
    )
    joined = joined.with_columns(
        _evaluation_class_expression(config.target_accepted_taxon_key).alias(
            "evaluation_class"
        )
    )
    challenge_pool = joined.filter(
        pl.col("evaluation_class").is_in(BALANCED_CHALLENGE_CATEGORIES)
    )
    counts = Counter(challenge_pool["evaluation_class"].to_list())
    shortfalls = {
        category: {
            "required": config.challenge_per_category,
            "available": counts.get(category, 0),
        }
        for category in BALANCED_CHALLENGE_CATEGORIES
        if counts.get(category, 0) < config.challenge_per_category
    }
    if shortfalls:
        raise ValueError(
            "balanced challenge label shortfalls: "
            + json.dumps(shortfalls, sort_keys=True, separators=(",", ":"))
        )
    selected_rows: list[dict[str, object]] = []
    for category in BALANCED_CHALLENGE_CATEGORIES:
        category_rows = challenge_pool.filter(
            pl.col("evaluation_class") == category
        ).to_dicts()
        chosen = _round_robin_challenge_selection(
            category_rows,
            quota=config.challenge_per_category,
            random_seed=config.random_seed,
            category=category,
        )
        population_count = len(category_rows)
        probability = config.challenge_per_category / population_count
        for rank, row in enumerate(chosen, start=1):
            selected_rows.append(
                _holdout_row(
                    row,
                    config=config,
                    holdout_kind=_HOLDOUT_KIND_BALANCED,
                    sampling_frame_fingerprint=sampling_fingerprint,
                    reviewed_labels_fingerprint=labels_fingerprint,
                    selection_fingerprint="sha256:" + "0" * 64,
                    sampling_stratum_id=f"challenge:{category}",
                    population_stratum_size=population_count,
                    sample_stratum_size=config.challenge_per_category,
                    inclusion_probability=probability,
                    sampling_weight=None,
                    selection_rank=rank,
                )
            )
    selection_fingerprint = _rows_fingerprint(
        "balanced-challenge-selection-v1.0.0",
        selected_rows,
        excluded_fields={"holdout_fingerprint", "selection_fingerprint"},
    )
    for row in selected_rows:
        row["selection_fingerprint"] = selection_fingerprint
    result = _finalize_holdout_rows(selected_rows)
    validate_frozen_evaluation_holdout(result)
    return result


def freeze_natural_stream_holdout(
    sampling_frame: pl.DataFrame,
    natural_selection: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    config: FrozenHoldoutConfig,
    *,
    usage_assignments: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Attach completed labels only after the natural sample is frozen."""

    _require_config(config)
    validate_natural_stream_selection(natural_selection)
    if _single_text(natural_selection, "selection_version") != config.holdout_version:
        raise ValueError("natural selection version does not match holdout config")
    if (
        _single_text(natural_selection, "configuration_fingerprint")
        != config.fingerprint
    ):
        raise ValueError(
            "natural selection configuration does not match holdout config"
        )
    joined, labels_fingerprint, sampling_fingerprint = _eligible_reviewed_candidates(
        sampling_frame,
        reviewed_labels,
        config,
        usage_assignments=usage_assignments,
    )
    expected_sampling_fingerprint = _single_text(
        natural_selection,
        "sampling_frame_fingerprint",
    )
    if sampling_fingerprint != expected_sampling_fingerprint:
        raise ValueError("natural selection was built from a different sampling frame")
    selected = natural_selection.join(
        joined,
        on=["sampling_unit_id", "source", "flickr_photo_id", "source_record_hash"],
        how="left",
        validate="1:1",
    )
    missing = selected.filter(pl.col("target_present").is_null())
    if missing.height:
        missing_ids = missing["sampling_unit_id"].to_list()
        raise ValueError(
            "natural-stream selection lacks completed final_test labels: "
            f"{missing_ids[:10]}"
        )
    selected = selected.with_columns(
        _evaluation_class_expression(config.target_accepted_taxon_key).alias(
            "evaluation_class"
        )
    )
    rows = [
        _holdout_row(
            row,
            config=config,
            holdout_kind=_HOLDOUT_KIND_NATURAL,
            sampling_frame_fingerprint=sampling_fingerprint,
            reviewed_labels_fingerprint=labels_fingerprint,
            selection_fingerprint=str(row["selection_fingerprint"]),
            sampling_stratum_id=str(row["sampling_stratum_id"]),
            population_stratum_size=int(row["population_stratum_size"]),
            sample_stratum_size=int(row["sample_stratum_size"]),
            inclusion_probability=float(row["inclusion_probability"]),
            sampling_weight=float(row["sampling_weight"]),
            selection_rank=int(row["selection_rank"]),
        )
        for row in selected.iter_rows(named=True)
    ]
    result = _finalize_holdout_rows(rows)
    validate_frozen_evaluation_holdout(result)
    return result


def validate_frozen_evaluation_holdout(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("frozen evaluation holdout must be a Polars DataFrame")
    if dict(frame.schema) != FROZEN_EVALUATION_HOLDOUT_SCHEMA:
        raise ValueError("frozen evaluation holdout physical schema mismatch")
    if frame.is_empty():
        raise ValueError("frozen evaluation holdout must not be empty")
    expected_sort = frame.sort("holdout_kind", "source", "flickr_photo_id")
    if not frame.equals(expected_sort):
        raise ValueError("frozen evaluation holdout is not deterministically sorted")
    _require_unique(frame, ["evaluation_item_id"], label="evaluation item")
    _require_unique(frame, ["sampling_unit_id"], label="evaluation sampling unit")
    if set(frame["dataset_split"].to_list()) != {_FINAL_TEST_SPLIT}:
        raise ValueError("frozen evaluation items must use final_test only")
    if (
        _single_text(frame, "schema_version")
        != FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION
    ):
        raise ValueError("frozen evaluation holdout schema version is incompatible")
    target_key = _single_text(frame, "target_scope_accepted_taxon_key")
    if frame.filter(
        pl.col("target_present") & (pl.col("accepted_taxon_key") != target_key)
    ).height:
        raise ValueError("frozen target-present rows do not match target scope")
    kind = _single_text(frame, "holdout_kind")
    if kind not in {_HOLDOUT_KIND_BALANCED, _HOLDOUT_KIND_NATURAL}:
        raise ValueError(f"unsupported frozen holdout kind: {kind}")
    if frame.filter(pl.col("target_present").is_null()).height:
        raise ValueError("frozen evaluation target_present values must be reviewed")
    if kind == _HOLDOUT_KIND_BALANCED:
        counts = Counter(frame["evaluation_class"].to_list())
        if set(counts) != BALANCED_CHALLENGE_CATEGORY_SET:
            raise ValueError("balanced challenge categories are incomplete")
        if len(set(counts.values())) != 1:
            raise ValueError("balanced challenge categories have unequal counts")
        if frame.filter(pl.col("sampling_weight").is_not_null()).height:
            raise ValueError(
                "balanced challenge rows must not carry prevalence weights"
            )
    else:
        if frame.filter(
            pl.col("sampling_weight").is_null()
            | (pl.col("sampling_weight") <= 0.0)
            | ~pl.col("sampling_weight").is_finite()
        ).height:
            raise ValueError("natural-stream holdout sampling weights are invalid")
        if frame.filter(
            (pl.col("sampling_weight") - (1.0 / pl.col("inclusion_probability"))).abs()
            > 1e-12
        ).height:
            raise ValueError("natural-stream holdout weights are inconsistent")
    holdout_fingerprint = _single_text(frame, "holdout_fingerprint")
    expected = _rows_fingerprint(
        FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION,
        frame.iter_rows(named=True),
        excluded_fields={"holdout_fingerprint"},
    )
    if holdout_fingerprint != expected:
        raise ValueError("frozen evaluation holdout_fingerprint is invalid")


def validate_evaluation_holdouts_disjoint(
    balanced_challenge: pl.DataFrame,
    natural_stream: pl.DataFrame,
) -> None:
    validate_frozen_evaluation_holdout(balanced_challenge)
    validate_frozen_evaluation_holdout(natural_stream)
    if _single_text(balanced_challenge, "holdout_kind") != _HOLDOUT_KIND_BALANCED:
        raise ValueError("balanced_challenge input has the wrong holdout kind")
    if _single_text(natural_stream, "holdout_kind") != _HOLDOUT_KIND_NATURAL:
        raise ValueError("natural_stream input has the wrong holdout kind")
    for field in (
        "holdout_version",
        "configuration_fingerprint",
        "sampling_frame_fingerprint",
        "target_scope_accepted_taxon_key",
    ):
        if _single_text(balanced_challenge, field) != _single_text(
            natural_stream,
            field,
        ):
            raise ValueError(f"evaluation holdouts have different {field} values")
    overlap = set(balanced_challenge["sampling_unit_id"].to_list()) & set(
        natural_stream["sampling_unit_id"].to_list()
    )
    if overlap:
        raise ValueError(
            f"balanced and natural-stream holdouts overlap: {sorted(overlap)[:10]}"
        )


def publish_frozen_evaluation_holdouts(
    balanced_challenge: pl.DataFrame,
    natural_stream: pl.DataFrame,
    output_dir: str | Path,
    *,
    leakage_register: pl.DataFrame,
    run_id: str | None = None,
) -> FrozenEvaluationHoldoutPublication:
    """Atomically publish both immutable holdouts plus compact reports."""

    started_at = datetime.now(UTC)
    leakage_audit = validate_reference_and_holdout_leakage(
        leakage_register,
        balanced_challenge,
        natural_stream,
    )
    effective_run_id = _required_text(
        run_id
        or "evaluation-holdouts-"
        + started_at.strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12],
        field="run_id",
    )
    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".{destination.name}.{uuid4().hex}.tmp"
    _log_event(
        "frozen_evaluation_holdouts_publish_started",
        command="evaluation.publish_frozen_holdouts",
        run_id=effective_run_id,
        output_dir=str(destination),
        started_at=started_at.isoformat(),
    )
    try:
        staging.mkdir(parents=False, exist_ok=False)
        challenge_staged = write_parquet(
            balanced_challenge,
            staging / BALANCED_CHALLENGE_HOLDOUT_FILE,
            overwrite=False,
        )
        natural_staged = write_parquet(
            natural_stream,
            staging / NATURAL_STREAM_HOLDOUT_FILE,
            overwrite=False,
        )
        leakage_staged = write_parquet(
            leakage_register,
            staging / EVALUATION_LEAKAGE_REGISTER_FILE,
            overwrite=False,
        )
        ended_at = datetime.now(UTC)
        report = _holdout_publication_report(
            balanced_challenge,
            natural_stream,
            challenge_path=challenge_staged,
            natural_path=natural_staged,
            leakage_path=leakage_staged,
            leakage_audit=leakage_audit,
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            final_output_dir=destination,
        )
        (staging / FROZEN_EVALUATION_HOLDOUT_REPORT_FILE).write_text(
            json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        (staging / FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE).write_text(
            _holdout_report_markdown(report),
            encoding="utf-8",
        )
        staging.replace(destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    challenge_path = destination / BALANCED_CHALLENGE_HOLDOUT_FILE
    natural_path = destination / NATURAL_STREAM_HOLDOUT_FILE
    leakage_path = destination / EVALUATION_LEAKAGE_REGISTER_FILE
    loaded_challenge = pl.read_parquet(challenge_path)
    loaded_natural = pl.read_parquet(natural_path)
    loaded_leakage = pl.read_parquet(leakage_path)
    validate_reference_and_holdout_leakage(
        loaded_leakage,
        loaded_challenge,
        loaded_natural,
    )
    if not balanced_challenge.equals(loaded_challenge):
        raise ValueError("balanced challenge Parquet round-trip mismatch")
    if not natural_stream.equals(loaded_natural):
        raise ValueError("natural-stream Parquet round-trip mismatch")
    if not leakage_register.equals(loaded_leakage):
        raise ValueError("evaluation leakage register Parquet round-trip mismatch")
    _log_event(
        "frozen_evaluation_holdouts_publish_completed",
        command="evaluation.publish_frozen_holdouts",
        run_id=effective_run_id,
        output_dir=str(destination),
        challenge_rows=balanced_challenge.height,
        natural_rows=natural_stream.height,
        leakage_register_fingerprint=leakage_audit.register_fingerprint,
        ended_at=ended_at.isoformat(),
    )
    return FrozenEvaluationHoldoutPublication(
        output_dir=destination,
        balanced_challenge_path=challenge_path,
        natural_stream_path=natural_path,
        leakage_register_path=leakage_path,
        report_json_path=destination / FROZEN_EVALUATION_HOLDOUT_REPORT_FILE,
        report_markdown_path=(
            destination / FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE
        ),
        report=report,
    )


def _eligible_reviewed_candidates(
    sampling_frame: pl.DataFrame,
    reviewed_labels: pl.DataFrame,
    config: FrozenHoldoutConfig,
    *,
    usage_assignments: pl.DataFrame | None,
) -> tuple[pl.DataFrame, str, str]:
    validate_evaluation_sampling_frame(sampling_frame)
    normalized = normalize_reviewed_label_frame(
        reviewed_labels,
        target_accepted_taxon_key=config.target_accepted_taxon_key,
    )
    findings = validate_reviewed_label_frame(normalized)
    fatals = [finding for finding in findings if finding.get("severity") == "fatal"]
    if fatals:
        raise ValueError(
            "reviewed labels failed validation: "
            + json.dumps(fatals, sort_keys=True, separators=(",", ":"))
        )
    assignments = normalize_usage_assignments(usage_assignments)
    forbidden_ids = set(
        assignments.filter(
            pl.col("usage_role").is_in(FORBIDDEN_EVALUATION_USAGE_ROLES)
        )["sampling_unit_id"].to_list()
    )
    sampling_identity = sampling_frame.select(
        "sampling_unit_id",
        "source",
        "flickr_photo_id",
        "source_record_hash",
        "geo_cluster_id",
        "geo_stratum",
        "primary_query_tier",
        "primary_query_term",
        "initial_visual_domain",
        "source_owner_group_id",
        "sampling_hash",
    )
    labels = normalized.join(
        sampling_identity,
        on=_LABEL_KEY,
        how="left",
        suffix="_sampling",
        validate="m:1",
    )
    final_labels = labels.filter(pl.col("dataset_split") == _FINAL_TEST_SPLIT)
    if final_labels.filter(pl.col("sampling_unit_id").is_null()).height:
        raise ValueError(
            "final_test reviewed labels are absent from the sampling frame"
        )
    if final_labels.filter(
        pl.col("sampling_unit_id").is_in(sorted(forbidden_ids))
    ).height:
        raise ValueError(
            "final_test reviewed labels overlap forbidden evaluation usage roles"
        )
    split_conflicts = (
        labels.filter(pl.col("sampling_unit_id").is_not_null())
        .group_by("sampling_unit_id")
        .agg(pl.col("dataset_split").drop_nulls().n_unique().alias("split_count"))
        .filter(pl.col("split_count") > 1)
    )
    if split_conflicts.height:
        raise ValueError("reviewed sampling units cross dataset splits")
    eligible = final_labels.filter(
        pl.col("target_present").is_not_null()
        & pl.col("label_certainty").is_in(config.allowed_label_certainties)
        & pl.col("second_review_status").is_in(config.allowed_second_review_statuses)
    )
    _require_unique(eligible, _LABEL_KEY, label="eligible reviewed candidate")
    target_mismatch = eligible.filter(
        pl.col("target_present")
        & (pl.col("accepted_taxon_key") != config.target_accepted_taxon_key)
    )
    if target_mismatch.height:
        raise ValueError("target-present labels do not match the configured target key")
    geo_mismatch = eligible.filter(
        pl.col("geo_cluster_id").is_not_null()
        & (pl.col("geo_cluster_id") != pl.col("geo_cluster_id_sampling"))
    )
    if geo_mismatch.height:
        raise ValueError("reviewed geo clusters do not match the sampling frame")
    query_tier_mismatch = eligible.filter(
        pl.col("source_query_tier").is_not_null()
        & (pl.col("source_query_tier") != pl.col("primary_query_tier"))
    )
    if query_tier_mismatch.height:
        raise ValueError("reviewed query tiers do not match the sampling frame")
    query_term_mismatch = eligible.filter(
        pl.col("source_query_term").is_not_null()
        & (pl.col("source_query_term") != pl.col("primary_query_term"))
    )
    if query_term_mismatch.height:
        raise ValueError("reviewed query terms do not match the sampling frame")
    eligible = eligible.with_columns(
        pl.col("geo_cluster_id_sampling").alias("resolved_geo_cluster_id"),
        pl.col("primary_query_tier").alias("resolved_source_query_tier"),
        pl.col("primary_query_term").alias("resolved_source_query_term"),
        pl.coalesce(
            pl.col("observer_owner_group_id"),
            pl.col("source_owner_group_id"),
        ).alias("resolved_observer_owner_group_id"),
    )
    return (
        eligible,
        _reviewed_labels_fingerprint(normalized),
        _sampling_frame_fingerprint(sampling_frame),
    )


def _evaluation_class_expression(target_key: str) -> pl.Expr:
    route = pl.col("route")
    life_stage = pl.col("life_stage")
    visual_domain = pl.col("visual_domain")
    target_present = pl.col("target_present")
    return (
        pl.when((route == "larval") | (life_stage == "larva"))
        .then(pl.lit("caterpillars"))
        .when((route == "pinned_specimen") | (visual_domain == "pinned_specimen"))
        .then(pl.lit("pinned_specimens"))
        .when(visual_domain.is_in(_ARTIFACT_VISUAL_DOMAINS))
        .then(pl.lit("artifacts"))
        .when(target_present & (pl.col("accepted_taxon_key") == target_key))
        .then(pl.lit("verified_target"))
        .when(
            ~target_present
            & pl.col("is_butterfly")
            & (pl.col("genus").str.to_lowercase() == "papilio")
        )
        .then(pl.lit("other_papilio"))
        .when(
            ~target_present
            & pl.col("is_butterfly")
            & (pl.col("family").str.to_lowercase() == "papilionidae")
        )
        .then(pl.lit("other_papilionidae"))
        .when(~pl.col("is_butterfly"))
        .then(pl.lit("moths_and_other_insects"))
        .when(pl.col("is_butterfly"))
        .then(pl.lit("other_butterfly"))
        .otherwise(pl.lit("reviewed_unclassified"))
    )


def _round_robin_challenge_selection(
    rows: Sequence[dict[str, object]],
    *,
    quota: int,
    random_seed: int,
    category: str,
) -> list[dict[str, object]]:
    strata: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        stratum = (
            str(row.get("geo_stratum") or "no_geo"),
            str(row.get("initial_visual_domain") or "ambiguous"),
            str(row.get("primary_query_term") or "unknown_term"),
        )
        strata[stratum].append(row)
    for stratum_rows in strata.values():
        stratum_rows.sort(
            key=lambda row: _selection_key(
                random_seed,
                str(row["sampling_unit_id"]),
                purpose=f"challenge:{category}",
            )
        )
    stratum_order = sorted(
        strata,
        key=lambda stratum: canonical_semantic_fingerprint(
            {
                "random_seed": random_seed,
                "category": category,
                "stratum": list(stratum),
            }
        ),
    )
    selected: list[dict[str, object]] = []
    cursor = 0
    while len(selected) < quota:
        progressed = False
        for _ in range(len(stratum_order)):
            stratum = stratum_order[cursor % len(stratum_order)]
            cursor += 1
            if strata[stratum]:
                selected.append(strata[stratum].pop(0))
                progressed = True
                break
        if not progressed:
            raise RuntimeError("could not satisfy balanced challenge quota")
    return selected


def _holdout_row(
    row: Mapping[str, object],
    *,
    config: FrozenHoldoutConfig,
    holdout_kind: str,
    sampling_frame_fingerprint: str,
    reviewed_labels_fingerprint: str,
    selection_fingerprint: str,
    sampling_stratum_id: str,
    population_stratum_size: int,
    sample_stratum_size: int,
    inclusion_probability: float,
    sampling_weight: float | None,
    selection_rank: int,
) -> dict[str, object]:
    evaluation_item_id = "evaluation-item:" + canonical_semantic_fingerprint(
        {
            "holdout_version": config.holdout_version,
            "source": row["source"],
            "flickr_photo_id": row["flickr_photo_id"],
            "detection_id": row.get("detection_id"),
        }
    ).removeprefix("sha256:")
    return {
        "schema_version": FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION,
        "holdout_version": config.holdout_version,
        "holdout_fingerprint": "sha256:" + "0" * 64,
        "configuration_fingerprint": config.fingerprint,
        "sampling_frame_fingerprint": sampling_frame_fingerprint,
        "reviewed_labels_fingerprint": reviewed_labels_fingerprint,
        "selection_fingerprint": selection_fingerprint,
        "holdout_kind": holdout_kind,
        "target_scope_accepted_taxon_key": config.target_accepted_taxon_key,
        "evaluation_item_id": evaluation_item_id,
        "sampling_unit_id": row["sampling_unit_id"],
        "source": row["source"],
        "flickr_photo_id": row["flickr_photo_id"],
        "source_record_hash": row["source_record_hash"],
        "detection_id": row.get("detection_id"),
        "crop_hash": row.get("crop_hash"),
        "evaluation_class": row["evaluation_class"],
        "target_present": row["target_present"],
        "accepted_taxon_key": row.get("accepted_taxon_key"),
        "scientific_name": row.get("scientific_name"),
        "is_butterfly": row["is_butterfly"],
        "family_key": row.get("family_key"),
        "family": row.get("family"),
        "genus_key": row.get("genus_key"),
        "genus": row.get("genus"),
        "label_certainty": row["label_certainty"],
        "life_stage": row["life_stage"],
        "visual_domain": row["visual_domain"],
        "route": row.get("route"),
        "geo_cluster_id": row["resolved_geo_cluster_id"],
        "source_query_tier": row["resolved_source_query_tier"],
        "source_query_term": row["resolved_source_query_term"],
        "duplicate_group_id": row.get("duplicate_group_id"),
        "observer_owner_group_id": row["resolved_observer_owner_group_id"],
        "dataset_split": row["dataset_split"],
        "reviewer_id": row["reviewer_id"],
        "second_review_status": row["second_review_status"],
        "sampling_stratum_id": sampling_stratum_id,
        "population_stratum_size": population_stratum_size,
        "sample_stratum_size": sample_stratum_size,
        "inclusion_probability": inclusion_probability,
        "sampling_weight": sampling_weight,
        "selection_rank": selection_rank,
    }


def _finalize_holdout_rows(rows: list[dict[str, object]]) -> pl.DataFrame:
    preliminary = pl.DataFrame(rows, schema=FROZEN_EVALUATION_HOLDOUT_SCHEMA).sort(
        "holdout_kind", "source", "flickr_photo_id"
    )
    fingerprint = _rows_fingerprint(
        FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION,
        preliminary.iter_rows(named=True),
        excluded_fields={"holdout_fingerprint"},
    )
    return preliminary.with_columns(pl.lit(fingerprint).alias("holdout_fingerprint"))


def _allocate_stratified_sample(
    counts: Mapping[str, int],
    *,
    sample_size: int,
    minimum_per_stratum: int,
) -> dict[str, int]:
    if not counts:
        raise ValueError("natural-stream eligible strata must not be empty")
    population = sum(counts.values())
    if sample_size <= 0 or sample_size > population:
        raise ValueError("sample_size must be positive and no larger than population")
    allocation = {key: min(count, minimum_per_stratum) for key, count in counts.items()}
    base = sum(allocation.values())
    if base > sample_size:
        raise ValueError(
            "minimum_per_natural_stratum requires more rows than natural_sample_size: "
            f"minimum_total={base}, sample_size={sample_size}"
        )
    remaining = sample_size - base
    while remaining:
        capacities = {
            key: counts[key] - allocation[key]
            for key in counts
            if counts[key] > allocation[key]
        }
        if not capacities:
            raise RuntimeError("natural-stream allocation exhausted its population")
        capacity_total = sum(capacities.values())
        exact = {
            key: Fraction(capacity * remaining, capacity_total)
            for key, capacity in capacities.items()
        }
        increments = {
            key: min(capacities[key], int(value)) for key, value in exact.items()
        }
        assigned = sum(increments.values())
        for key, increment in increments.items():
            allocation[key] += increment
        remaining -= assigned
        if not remaining:
            break
        order = sorted(
            capacities,
            key=lambda key: (
                exact[key] - int(exact[key]),
                canonical_semantic_fingerprint({"stratum": key}),
            ),
            reverse=True,
        )
        progressed = False
        for key in order:
            if remaining == 0:
                break
            if allocation[key] < counts[key]:
                allocation[key] += 1
                remaining -= 1
                progressed = True
        if not progressed and remaining:
            raise RuntimeError("natural-stream largest-remainder allocation stalled")
    return allocation


def _sampling_frame_fingerprint(frame: pl.DataFrame) -> str:
    return canonical_semantic_fingerprint(
        {
            "schema": {
                key: str(value)
                for key, value in EVALUATION_SAMPLING_FRAME_SCHEMA.items()
            },
            "rows": frame.select(
                "sampling_unit_id",
                "sampling_hash",
                "source_record_hash",
            )
            .sort("sampling_unit_id")
            .to_dicts(),
        }
    )


def _reviewed_labels_fingerprint(frame: pl.DataFrame) -> str:
    fields = [
        "schema_version",
        "source",
        "flickr_photo_id",
        "detection_id",
        "crop_hash",
        "target_present",
        "accepted_taxon_key",
        "scientific_name",
        "is_butterfly",
        "family_key",
        "family",
        "genus_key",
        "genus",
        "label_certainty",
        "life_stage",
        "visual_domain",
        "route",
        "geo_cluster_id",
        "source_query_tier",
        "source_query_term",
        "duplicate_group_id",
        "observer_owner_group_id",
        "dataset_split",
        "reviewer_id",
        "second_review_status",
    ]
    return canonical_semantic_fingerprint(
        {
            "schema": "reviewed-labels-v2",
            "rows": frame.select(fields)
            .sort("source", "flickr_photo_id", "detection_id")
            .to_dicts(),
        }
    )


def _rows_fingerprint(
    version: str,
    rows: Sequence[Mapping[str, object]] | Any,
    *,
    excluded_fields: set[str],
) -> str:
    normalized = [
        {key: value for key, value in dict(row).items() if key not in excluded_fields}
        for row in rows
    ]
    normalized.sort(
        key=lambda row: (
            str(row.get("sampling_unit_id") or ""),
            str(row.get("evaluation_item_id") or ""),
        )
    )
    return canonical_semantic_fingerprint({"version": version, "rows": normalized})


def _selection_key(seed: int, item_id: str, *, purpose: str) -> str:
    return canonical_semantic_fingerprint(
        {"random_seed": seed, "purpose": purpose, "item_id": item_id}
    )


def _holdout_publication_report(
    challenge: pl.DataFrame,
    natural: pl.DataFrame,
    *,
    challenge_path: Path,
    natural_path: Path,
    leakage_path: Path,
    leakage_audit: EvaluationLeakageAudit,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
    final_output_dir: Path,
) -> dict[str, Any]:
    return {
        "schema_version": FROZEN_EVALUATION_HOLDOUT_REPORT_SCHEMA_VERSION,
        "command": "evaluation.publish_frozen_holdouts",
        "run_id": run_id,
        "pid": os.getpid(),
        "git_sha": current_git_sha(),
        "status": "complete",
        "started_at": started_at.isoformat(),
        "ended_at": ended_at.isoformat(),
        "elapsed_seconds": max(0.0, (ended_at - started_at).total_seconds()),
        "network_requests": 0,
        "balanced_challenge_rows": challenge.height,
        "natural_stream_rows": natural.height,
        "balanced_counts_by_class": _counts(challenge["evaluation_class"]),
        "natural_counts_by_class": _counts(natural["evaluation_class"]),
        "natural_weight_sum": float(natural["sampling_weight"].sum()),
        "leakage_validation": {
            "status": "passed",
            "register_fingerprint": leakage_audit.register_fingerprint,
            "register_item_count": leakage_audit.register_item_count,
            "reference_item_count": leakage_audit.reference_item_count,
            "balanced_challenge_item_count": (
                leakage_audit.balanced_challenge_item_count
            ),
            "natural_stream_item_count": leakage_audit.natural_stream_item_count,
            "coverage_by_dimension": dict(leakage_audit.coverage_by_dimension),
        },
        "artifacts": {
            "balanced_challenge": {
                "path": str(final_output_dir / challenge_path.name),
                "byte_count": challenge_path.stat().st_size,
                "row_count": challenge.height,
                "sha256": _file_sha256(challenge_path),
            },
            "natural_stream": {
                "path": str(final_output_dir / natural_path.name),
                "byte_count": natural_path.stat().st_size,
                "row_count": natural.height,
                "sha256": _file_sha256(natural_path),
            },
            "leakage_register": {
                "path": str(final_output_dir / leakage_path.name),
                "byte_count": leakage_path.stat().st_size,
                "row_count": leakage_audit.register_item_count,
                "sha256": _file_sha256(leakage_path),
            },
        },
    }


def _holdout_report_markdown(report: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen evaluation holdouts",
            "",
            f"- Status: `{report['status']}`",
            f"- Run ID: `{report['run_id']}`",
            f"- Balanced challenge rows: `{report['balanced_challenge_rows']}`",
            f"- Natural-stream rows: `{report['natural_stream_rows']}`",
            f"- Natural-stream weight sum: `{report['natural_weight_sum']}`",
            f"- Leakage validation: `{report['leakage_validation']['status']}`",
            "- Leakage register: "
            f"`{report['leakage_validation']['register_fingerprint']}`",
            "",
        ]
    )


def _counts(series: pl.Series) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in series.to_list()).items()))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _require_config(config: object) -> None:
    if not isinstance(config, FrozenHoldoutConfig):
        raise TypeError("config must be a FrozenHoldoutConfig")


def _require_nonblank(frame: pl.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        if frame.filter(
            pl.col(column).is_null()
            | (pl.col(column).cast(pl.String).str.strip_chars() == "")
        ).height:
            raise ValueError(f"column {column} cannot contain blank values")


def _require_unique(
    frame: pl.DataFrame,
    columns: Sequence[str],
    *,
    label: str,
) -> None:
    if frame.select(columns).n_unique() != frame.height:
        raise ValueError(f"{label} rows must be unique by {list(columns)}")


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have one consistent value")
    return _required_text(values[0], field=field)


def _single_integer(frame: pl.DataFrame, field: str) -> int:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have one consistent value")
    value = values[0]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    return value


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} cannot be blank")
    return text


def _required_text_tuple(values: Sequence[object], *, field: str) -> tuple[str, ...]:
    normalized = tuple(
        dict.fromkeys(_required_text(value, field=field) for value in values)
    )
    if not normalized:
        raise ValueError(f"{field} must not be empty")
    return normalized


def _positive_integer(value: object, *, field: str) -> int:
    result = _nonnegative_integer(value, field=field)
    if result == 0:
        raise ValueError(f"{field} must be positive")
    return result


def _nonnegative_integer(
    value: object,
    *,
    field: str,
    maximum: int = 2**32 - 1,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{field} must be an integer")
    if not 0 <= value <= maximum:
        raise ValueError(f"{field} must be between zero and {maximum}")
    return value


def _log_event(event: str, **fields: object) -> None:
    _LOGGER.info(
        json.dumps(
            {"event": event, **fields},
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    )


__all__ = [
    "ALLOWED_EVALUATION_USAGE_ROLES",
    "BALANCED_CHALLENGE_CATEGORIES",
    "BALANCED_CHALLENGE_HOLDOUT_FILE",
    "FORBIDDEN_EVALUATION_USAGE_ROLES",
    "FROZEN_EVALUATION_HOLDOUT_REPORT_FILE",
    "FROZEN_EVALUATION_HOLDOUT_REPORT_MARKDOWN_FILE",
    "FROZEN_EVALUATION_HOLDOUT_SCHEMA",
    "FROZEN_EVALUATION_HOLDOUT_SCHEMA_VERSION",
    "NATURAL_STREAM_HOLDOUT_FILE",
    "NATURAL_STREAM_SELECTION_FILE",
    "NATURAL_STREAM_SELECTION_SCHEMA",
    "NATURAL_STREAM_SELECTION_SCHEMA_VERSION",
    "USAGE_ASSIGNMENT_SCHEMA",
    "FrozenEvaluationHoldoutPublication",
    "FrozenHoldoutConfig",
    "build_balanced_challenge_holdout",
    "empty_frozen_evaluation_holdout",
    "empty_natural_stream_selection",
    "freeze_natural_stream_holdout",
    "load_natural_stream_selection",
    "normalize_usage_assignments",
    "publish_frozen_evaluation_holdouts",
    "select_natural_stream_candidates",
    "validate_evaluation_holdouts_disjoint",
    "validate_frozen_evaluation_holdout",
    "validate_natural_stream_selection",
    "write_natural_stream_selection",
]
