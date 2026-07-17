from __future__ import annotations

from dataclasses import dataclass
import gc
import json
from pathlib import Path
from statistics import median
import time
import tracemalloc

from biominer.evaluation.flickr_release import (
    UnreviewedFlickrCandidateEvidence,
    score_unreviewed_flickr_candidate,
)
from biominer.references.admission import default_reference_admission_policy
from biominer.references.readiness import reference_readiness_allows_vision
from biominer.references.support_admission import evaluate_support_admission
from biominer.run.flickr_selective_rescore import (
    calculate_flickr_rescore_plan,
)
from biominer.run.incremental_feature_reuse import (
    FLICKR_EMBEDDING_SCOPE,
    REFERENCE_EMBEDDING_SCOPE,
    calculate_incremental_feature_reuse,
    feature_cache_entry_frame,
    feature_reuse_request_frame,
)
from test_adaptive_readiness_matrix import _provisional_payload
from test_flickr_selective_rescore import _revision_and_evidence
from test_incremental_feature_reuse import _cache, _request
from test_support_admission import _evidence


BASELINE_PATH = (
    Path(__file__).parent
    / "performance_baselines"
    / "adaptive_workflow_v1.json"
)
SAMPLE_COUNT = 5


@dataclass(frozen=True)
class AdaptiveWorkflowPerformance:
    time_to_first_provisional_scoring_ms: float
    reference_review_count_before_first_scoring: int
    embedding_reuse_count: int
    embedding_reuse_ratio: float
    selective_rerun_ratio: float
    peak_traced_memory_mib: float


def _measure_once() -> AdaptiveWorkflowPerformance:
    gc.collect()
    tracemalloc.start()
    try:
        started_ns = time.perf_counter_ns()
        admission_evidence = _evidence()
        admission = evaluate_support_admission(
            admission_evidence,
            default_reference_admission_policy(),
        )
        readiness = _provisional_payload()
        reference_reviews_before_scoring = int(
            admission_evidence.review_status == "completed"
        )
        candidate = score_unreviewed_flickr_candidate(
            UnreviewedFlickrCandidateEvidence(
                source_record_id="flickr:performance-guard",
                route="adult_field",
                embedding_artifact_sha256="sha256:" + "c" * 64,
                candidate_ranking=(
                    "Papilio demoleus",
                    "Papilio polytes",
                ),
                provisional_margin=0.17,
                review_priority=0.82,
            )
        )
        time_to_first_ms = (
            time.perf_counter_ns() - started_ns
        ) / 1_000_000

        assert admission.eligible and admission.provisional
        assert reference_readiness_allows_vision(readiness)
        assert readiness["permits_provisional_scoring"] is True
        assert candidate.scoring_state == "provisional_candidate_scored"
        assert candidate.eligible_for_final_occurrence_dataset is False

        requests = feature_reuse_request_frame(
            [
                _request(
                    FLICKR_EMBEDDING_SCOPE,
                    "flickr:retained",
                    "photo-retained",
                ),
                _request(
                    REFERENCE_EMBEDDING_SCOPE,
                    "reference:retained",
                    "reference-retained",
                ),
                _request(
                    REFERENCE_EMBEDDING_SCOPE,
                    "reference:new",
                    "reference-new",
                    newly_admitted=True,
                ),
            ]
        )
        cache = feature_cache_entry_frame(
            [
                _cache(
                    FLICKR_EMBEDDING_SCOPE,
                    "flickr-retained",
                    "photo-retained",
                ),
                _cache(
                    REFERENCE_EMBEDDING_SCOPE,
                    "reference-retained",
                    "reference-retained",
                ),
            ]
        )
        reuse_plan = calculate_incremental_feature_reuse(requests, cache)
        embedding_reuse_count = reuse_plan.filter(
            reuse_plan["cache_hit"]
        ).height

        revision, evidence = _revision_and_evidence()
        rescore_plan = calculate_flickr_rescore_plan(
            revision,
            evidence,
            margin_impact_band=0.1,
        )
        rescore_count = rescore_plan.filter(
            rescore_plan["rescore_required"]
        ).height
        _, peak_bytes = tracemalloc.get_traced_memory()

        return AdaptiveWorkflowPerformance(
            time_to_first_provisional_scoring_ms=time_to_first_ms,
            reference_review_count_before_first_scoring=(
                reference_reviews_before_scoring
            ),
            embedding_reuse_count=embedding_reuse_count,
            embedding_reuse_ratio=embedding_reuse_count / reuse_plan.height,
            selective_rerun_ratio=rescore_count / rescore_plan.height,
            peak_traced_memory_mib=peak_bytes / (1024 * 1024),
        )
    finally:
        tracemalloc.stop()


def _median_measurement() -> AdaptiveWorkflowPerformance:
    samples = [_measure_once() for _ in range(SAMPLE_COUNT)]
    return AdaptiveWorkflowPerformance(
        time_to_first_provisional_scoring_ms=median(
            sample.time_to_first_provisional_scoring_ms
            for sample in samples
        ),
        reference_review_count_before_first_scoring=int(
            median(
                sample.reference_review_count_before_first_scoring
                for sample in samples
            )
        ),
        embedding_reuse_count=int(
            median(sample.embedding_reuse_count for sample in samples)
        ),
        embedding_reuse_ratio=median(
            sample.embedding_reuse_ratio for sample in samples
        ),
        selective_rerun_ratio=median(
            sample.selective_rerun_ratio for sample in samples
        ),
        peak_traced_memory_mib=median(
            sample.peak_traced_memory_mib for sample in samples
        ),
    )


def test_adaptive_workflow_efficiency_stays_within_measured_baseline() -> None:
    baseline_document = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline_document["schema_version"] == (
        "adaptive-workflow-performance-baseline-v1.0.0"
    )
    baseline = baseline_document["baseline"]
    tolerance = baseline_document["tolerance"]
    measured = _median_measurement()

    # Counts and ratios describe selective-work correctness. They remain exact;
    # only host-sensitive time and traced memory receive a variance allowance.
    assert measured.reference_review_count_before_first_scoring == baseline[
        "reference_review_count_before_first_scoring"
    ]
    assert measured.embedding_reuse_count == baseline["embedding_reuse_count"]
    assert measured.embedding_reuse_ratio == baseline["embedding_reuse_ratio"]
    assert measured.selective_rerun_ratio == baseline[
        "selective_rerun_ratio"
    ]

    time_ceiling = (
        baseline["time_to_first_provisional_scoring_ms"]
        * (1 + tolerance["time_relative_increase"])
        + tolerance["time_absolute_ms"]
    )
    memory_ceiling = (
        baseline["peak_traced_memory_mib"]
        * (1 + tolerance["peak_memory_relative_increase"])
        + tolerance["peak_memory_absolute_mib"]
    )
    assert measured.time_to_first_provisional_scoring_ms <= time_ceiling
    assert measured.peak_traced_memory_mib <= memory_ceiling
