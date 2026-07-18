from __future__ import annotations

import json
from pathlib import Path

from biominer.bioclip.provisional_prototypes import (
    PROVISIONAL_PROTOTYPES_SCHEMA_VERSION,
    provisional_prototypes_schema,
)
from biominer.bioclip.provisional_ranking import (
    PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION,
    PROVISIONAL_SCORE_SEMANTICS,
    provisional_reference_ranking_schema,
)
from biominer.bioclip.reference_embeddings import (
    REFERENCE_EMBEDDINGS_SCHEMA_VERSION,
    reference_embeddings_schema,
)
from biominer.candidates.regional_union import (
    REGIONAL_CANDIDATE_POLICY_VERSION,
    REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION,
    regional_candidate_species_schema,
)
from biominer.references.admission import default_reference_admission_policy
from biominer.run.adaptive_config import AdaptiveReferenceSettings
from biominer.run.stages import (
    ADAPTIVE_REFERENCE_PRODUCTION_STAGES,
    DEFAULT_PRODUCTION_STAGES,
    MANUAL_REVIEW_STAGES,
    REFERENCE_FIRST_PRODUCTION_STAGES,
)


ROOT = Path(__file__).resolve().parents[1]
JSON_REPORT = ROOT / "reports/geo_dynamic_pooling/current_baseline.json"
MARKDOWN_REPORT = ROOT / "reports/geo_dynamic_pooling/current_baseline.md"
GITHITS_LEDGER = ROOT / "provenance/githits.jsonl"


def _report() -> dict[str, object]:
    return json.loads(JSON_REPORT.read_text(encoding="utf-8"))


def test_dynamic_pooling_baseline_matches_executable_contracts() -> None:
    report = _report()
    defaults = AdaptiveReferenceSettings()
    admission = default_reference_admission_policy()
    artifacts = report["artifact_contracts"]

    assert report["schema_version"] == "geo-dynamic-pooling-baseline-v1.0.0"
    assert report["evidence_scope"] == (
        "committed_code_and_fixture_backed_reports_no_live_execution_claim"
    )
    assert report["adaptive_reference_defaults"] == {
        "reference_admission_mode": defaults.reference_admission_mode,
        "reference_source": defaults.reference_source,
        "initial_scoring_mode": defaults.initial_scoring_mode,
        "flickr_release_requires_human_review": True,
        "statistical_reference_audit": True,
        "supported_compatibility_modes": [
            "human_verified_strict",
            "human_verified_flagged_only",
        ],
        "admission_policy_schema_version": admission.schema_version,
        "admission_policy_version": admission.policy_version,
        "admission_policy_fingerprint": admission.fingerprint,
        "admission_audit_policy_version": admission.audit_policy_version,
        "allowed_unreviewed_routes": list(admission.allowed_unreviewed_routes),
        "minimum_decoded_dimensions": [
            admission.minimum_decoded_width,
            admission.minimum_decoded_height,
        ],
        "minimum_subject_area_ratio": admission.minimum_subject_area_ratio,
        "maximum_images_per_observation": admission.maximum_images_per_observation,
        "maximum_images_per_observer_before_reuse": (
            admission.maximum_images_per_observer_before_reuse
        ),
    }
    assert artifacts["reference_embeddings"]["schema_version"] == (
        REFERENCE_EMBEDDINGS_SCHEMA_VERSION
    )
    assert artifacts["reference_embeddings"]["column_count"] == len(
        reference_embeddings_schema(1)
    )
    assert artifacts["provisional_prototypes"]["schema_version"] == (
        PROVISIONAL_PROTOTYPES_SCHEMA_VERSION
    )
    assert artifacts["provisional_prototypes"]["column_count"] == len(
        provisional_prototypes_schema(1)
    )
    assert artifacts["provisional_ranking"]["schema_version"] == (
        PROVISIONAL_REFERENCE_RANKING_SCHEMA_VERSION
    )
    assert artifacts["provisional_ranking"]["score_semantics"] == (
        PROVISIONAL_SCORE_SEMANTICS
    )
    assert artifacts["provisional_ranking"]["column_count"] == len(
        provisional_reference_ranking_schema()
    )
    assert artifacts["candidate_sets"]["schema_version"] == (
        REGIONAL_CANDIDATE_SPECIES_SCHEMA_VERSION
    )
    assert artifacts["candidate_sets"]["policy_version"] == (
        REGIONAL_CANDIDATE_POLICY_VERSION
    )
    assert artifacts["candidate_sets"]["column_count"] == len(
        regional_candidate_species_schema()
    )


def test_dynamic_pooling_baseline_stage_graph_and_evidence_state_are_truthful() -> None:
    report = _report()
    graph = report["stage_graph"]
    pilot = report["pilot_and_human_review_state"]
    tests = report["baseline_tests"]

    assert graph["default_production"] == [
        stage.value for stage in DEFAULT_PRODUCTION_STAGES
    ]
    assert graph["reference_first_production"] == [
        stage.value for stage in REFERENCE_FIRST_PRODUCTION_STAGES
    ]
    assert graph["adaptive_reference_production"] == [
        stage.value for stage in ADAPTIVE_REFERENCE_PRODUCTION_STAGES
    ]
    assert graph["manual_review_stages"] == sorted(
        stage.value for stage in MANUAL_REVIEW_STAGES
    )
    assert pilot["live_status"] == "not_executed_missing_local_artifacts"
    assert pilot["flickr_labels_reviewed"] == 0
    assert pilot["required_review_queue_records"] == 50
    assert pilot["scientific_release_authorized"] is False
    assert tests == {
        "command": "uv run pytest -q",
        "status": "passed",
        "passed": 2531,
        "failed": 0,
        "duration_seconds": 111.98,
        "run_scope": "full_repository",
    }
    assert report["downstream_repository_state"]["compatibility_review_required"]
    assert all(
        not report["downstream_repository_state"][repository][
            "consumed_in_this_subtask"
        ]
        for repository in ("taxalens", "butterflylens")
    )


def test_dynamic_pooling_baseline_markdown_and_githits_record_are_present() -> None:
    report_text = MARKDOWN_REPORT.read_text(encoding="utf-8")
    assert "2,531 tests" in report_text
    assert "zero Flickr labels have been human reviewed" in report_text
    assert "not a live biological-performance claim" in report_text
    assert "first-class dynamic global/local reference pool" in report_text

    records = [
        json.loads(line)
        for line in GITHITS_LEDGER.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    subtask = [row for row in records if row.get("task_id") == "geo-pool-0.1.1"]
    assert len(subtask) == 1
    assert subtask[0]["githits_status"] == "unavailable"
    assert subtask[0]["solution_id"] == "7adaebff-2209-44df-b947-8ea4ad3a7ada"
