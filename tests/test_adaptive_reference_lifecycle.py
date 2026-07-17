from __future__ import annotations

import polars as pl
import pytest

from biominer.evaluation.flickr_export import (
    FlickrExportValidationError,
    validate_verified_flickr_export,
)
from biominer.evaluation.flickr_release import (
    UnreviewedFlickrCandidateEvidence,
    score_unreviewed_flickr_candidate,
)
from biominer.references.adaptive_bank_revision import (
    revise_adaptive_support_bank,
)
from biominer.references.admission import (
    default_reference_admission_policy,
    strict_reference_admission_policy,
)
from biominer.references.readiness import reference_readiness_allows_vision
from biominer.references.support_admission import evaluate_support_admission
from biominer.references.targeted_review import (
    build_targeted_reference_review_queue,
)
from biominer.reports.evidence_maturity import evidence_maturity_legend
from biominer.run.flickr_selective_rescore import (
    calculate_flickr_rescore_plan,
    flickr_rescore_evidence_frame,
    target_score_ids_to_rescore,
)
from biominer.run.incremental_feature_reuse import (
    FLICKR_EMBEDDING_SCOPE,
    REFERENCE_EMBEDDING_SCOPE,
    calculate_incremental_feature_reuse,
    feature_cache_entry_frame,
    feature_reuse_request_frame,
)
from test_adaptive_bank_revision import (
    _dependencies,
    _review_with_verify_and_exclude,
    _support_manifest,
)
from test_adaptive_readiness_matrix import _provisional_payload
from test_flickr_export import _valid_row
from test_flickr_selective_rescore import _evidence as _rescore_evidence
from test_incremental_feature_reuse import _cache, _request
from test_support_admission import _evidence as _admission_evidence
from test_targeted_reference_review import _inputs


def test_adaptive_reference_lifecycle_targets_only_affected_evidence() -> None:
    # 1. Provider-asserted GBIF support can reach provisional scoring before
    # reference review, but cannot authorize calibrated scoring or release.
    admission = evaluate_support_admission(
        _admission_evidence(), default_reference_admission_policy()
    )
    readiness = _provisional_payload()
    assert admission.eligible is True
    assert admission.provisional is True
    assert admission.evidence_path == "gbif_provider_asserted"
    assert reference_readiness_allows_vision(readiness) is True
    assert readiness["permits_provisional_scoring"] is True
    assert readiness["permits_scientific_release"] is False

    # 3-4. Statistical audit flags the underperformer and sends only that
    # species into targeted reference review.
    inputs = _inputs()
    escalations = inputs[1]
    flagged = escalations.filter(pl.col("flagged_for_reference_review"))
    assert flagged["target_species"].to_list() == ["Papilio demoleus"]
    assert escalations.filter(~pl.col("flagged_for_reference_review"))[
        "target_species"
    ].to_list() == ["Papilio machaon"]
    targeted = build_targeted_reference_review_queue(*inputs)
    assert targeted["scientific_name"].unique().to_list() == [
        "Papilio demoleus"
    ]

    # 5. The human review excludes one bad reference while retaining the
    # verified reference and the unrelated provisional reference.
    revision_inputs, review = _review_with_verify_and_exclude()
    revision = revise_adaptive_support_bank(
        _support_manifest(revision_inputs[0]),
        review,
        _dependencies(revision_inputs[0]),
    )
    excluded = revision.change_manifest.filter(
        pl.col("change_type") == "excluded_after_review"
    )
    assert excluded.height == 1
    excluded_id = str(excluded["reference_media_id"].item())
    excluded_support = revision.revised_support_manifest.filter(
        pl.col("reference_media_id") == excluded_id
    ).row(0, named=True)
    assert excluded_support["support_eligible"] is False

    # 6. Content-addressed identities preserve reusable Flickr and retained
    # reference embeddings across the review cycle.
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
    assert reuse_plan["cache_hit"].to_list() == [True, True]
    assert set(reuse_plan["action"].to_list()) == {
        "reuse_flickr_embedding",
        "reuse_reference_embedding",
    }

    # 7. The revised bank invalidates only evidence that depends on the
    # changed species; the unrelated Flickr score remains reusable.
    changed_species = next(
        str(row["accepted_taxon_key"])
        for row in revision.change_manifest.iter_rows(named=True)
        if not str(row["change_type"]).startswith("unchanged")
    )
    evidence = flickr_rescore_evidence_frame(
        [
            _rescore_evidence(
                "score:affected",
                bank_fingerprint=revision.old_reference_bank_fingerprint,
                target=changed_species,
            ),
            _rescore_evidence(
                "score:unrelated",
                bank_fingerprint=revision.old_reference_bank_fingerprint,
            ),
        ]
    )
    rescore_plan = calculate_flickr_rescore_plan(
        revision,
        evidence,
        margin_impact_band=0.1,
    )
    assert target_score_ids_to_rescore(rescore_plan) == ("score:affected",)
    unrelated = rescore_plan.filter(
        pl.col("target_score_id") == "score:unrelated"
    ).row(0, named=True)
    assert unrelated["rescore_action"] == "reuse_prior_score"

    # 8. Strict mode retains its historical human-verification requirement.
    strict = evaluate_support_admission(
        _admission_evidence(), strict_reference_admission_policy()
    )
    assert strict.eligible is False
    assert strict.evidence_path == "none"
    assert "policy_does_not_enable_provisional_support" in strict.reasons

    # 9. The reporting contract explicitly prevents raw provisional margins
    # from being represented as probabilities or release evidence.
    raw_score = evidence_maturity_legend().filter(
        pl.col("maturity_label") == "provisional_raw_score"
    ).row(0, named=True)
    assert raw_score["probability_semantics"] is False
    assert raw_score["release_authorizing"] is False
    assert "probability" in raw_score["prohibited_claims"]


def test_unreviewed_flickr_scoring_never_enters_final_dataset() -> None:
    # 2. An unreviewed Flickr candidate may be scored for triage, but its
    # release decision remains fail-closed until human review.
    candidate = score_unreviewed_flickr_candidate(
        UnreviewedFlickrCandidateEvidence(
            source_record_id="flickr:unreviewed",
            route="adult_field",
            embedding_artifact_sha256="sha256:" + "c" * 64,
            candidate_ranking=("Papilio demoleus", "Papilio polytes"),
            provisional_margin=0.17,
            review_priority=0.82,
        )
    )
    assert candidate.scoring_state == "provisional_candidate_scored"
    assert candidate.human_review_state == "unreviewed"
    assert candidate.eligible_for_final_occurrence_dataset is False

    # 10. The final export gate rejects the entire dataset when even one
    # Flickr record is unreviewed; no partial release is possible.
    unreviewed = {
        **_valid_row(),
        "source_record_id": candidate.source_record_id,
        "human_review_decision": candidate.human_review_state,
        "eligible_for_final_occurrence_dataset": (
            candidate.eligible_for_final_occurrence_dataset
        ),
        "release_state": candidate.release_state.value,
    }
    with pytest.raises(FlickrExportValidationError) as error:
        validate_verified_flickr_export(
            pl.DataFrame([_valid_row(), unreviewed])
        )

    assert "flickr:unreviewed" in error.value.blocked_records
