"""Validation for Papilio adaptive initial-scoring evidence."""

from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.adaptive_pilot import load_adaptive_pilot_plan


INITIAL_PILOT_REPORT_SCHEMA_VERSION = "adaptive-pilot-initial-scoring-v1.0.0"
REQUIRED_STAGES = (
    "provisional_gbif_support",
    "reference_embeddings",
    "provisional_prototypes",
    "provisional_flickr_scoring",
)


def load_initial_pilot_report(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("initial pilot report must contain an object")
    validate_initial_pilot_report(payload)
    return payload


def validate_initial_pilot_report(report: Mapping[str, object]) -> None:
    if report.get("schema_version") != INITIAL_PILOT_REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported initial pilot report schema")
    plan_path = report.get("plan_path")
    if not isinstance(plan_path, str):
        raise ValueError("initial pilot report plan path is missing")
    plan = load_adaptive_pilot_plan(plan_path)
    if report.get("plan_fingerprint") != plan["plan_fingerprint"]:
        raise ValueError("initial pilot report plan fingerprint mismatch")
    if report.get("execution_mode") != "fixture_backed_integration":
        raise ValueError("initial pilot report must disclose fixture execution")
    current = _mapping(report, "current_execution")
    if current.get("live_status") != "not_executed_missing_local_artifacts":
        raise ValueError("live pilot execution status is not explicit")
    if current.get("fixture_status") != "passed":
        raise ValueError("fixture-backed pilot execution did not pass")
    stages = current.get("stages")
    if not isinstance(stages, list) or tuple(
        row.get("stage_id") for row in stages if isinstance(row, Mapping)
    ) != REQUIRED_STAGES:
        raise ValueError("initial pilot stage evidence is incomplete or reordered")
    if any(
        not isinstance(row, Mapping) or row.get("status") != "fixture_passed"
        for row in stages
    ):
        raise ValueError("every initial pilot fixture stage must pass")
    metrics = _mapping(current, "metrics")
    if metrics.get("reference_reviews_before_first_scoring") != 0:
        raise ValueError("adaptive initial scoring must require zero reference reviews")
    first_score_ms = metrics.get("time_to_first_provisional_scoring_ms")
    if not isinstance(first_score_ms, int | float) or first_score_ms <= 0:
        raise ValueError("initial pilot first-score timing must be measured")
    if metrics.get("evidence_basis") != "measured_fixture_baseline":
        raise ValueError("initial pilot metric evidence basis mismatch")
    unexecuted = current.get("unexecuted_live_steps")
    if not isinstance(unexecuted, list) or len(unexecuted) != len(REQUIRED_STAGES):
        raise ValueError("initial pilot live-step disclosure is incomplete")
    semantics = _mapping(report, "semantics")
    if semantics != {
        "provider_support_is_human_verified": False,
        "raw_scores_are_probabilities": False,
        "fixture_outcomes_are_live_outcomes": False,
        "scientific_release_authorized": False,
    }:
        raise ValueError("initial pilot scientific semantics were weakened")
    historical = _mapping(report, "historical_context")
    if historical.get("counted_as_current_execution") is not False:
        raise ValueError("historical pilot evidence cannot count as current execution")
    expected = canonical_semantic_fingerprint(
        {key: value for key, value in report.items() if key != "report_fingerprint"}
    )
    if report.get("report_fingerprint") != expected:
        raise ValueError("initial pilot report fingerprint mismatch")


def _mapping(parent: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = parent.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"initial pilot {field} must be an object")
    return value
