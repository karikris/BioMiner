from __future__ import annotations

from flickr_bio_occurrence.dwc.mapper import REQUIRED_DWC_FIELDS, map_candidate_to_dwc
from flickr_bio_occurrence.taxonomy.range_context import annotate_range_context


def test_dwc_required_fields_present() -> None:
    row = map_candidate_to_dwc(
        {
            "flickr_photo_id": "123",
            "resolved_scientific_name": "Papilio demoleus",
            "eventDate": "2024-01-15",
            "decimalLatitude": -27.4698,
            "decimalLongitude": 153.0251,
            "verbatimIdentification": "Papilio demoleus",
            "identificationVerificationStatus": "needs_review",
            "associatedReferences": "https://www.flickr.com/photos/example/123",
            "license": "cc-by",
            "rightsHolder": "example",
            "human_evidence": True,
        }
    )

    assert set(REQUIRED_DWC_FIELDS).issubset(row.keys())
    assert row["basisOfRecord"] == "HumanObservation"
    assert row["occurrenceID"]


def test_outside_known_range_never_auto_rejects() -> None:
    result = annotate_range_context(known_range_match=False, known_range_distance_km=2500)

    assert result.range_context_status == "range_extension_candidate"
    assert result.range_extension_candidate is True
    assert result.review_status != "rejected"


def test_range_extension_candidate_is_annotation_not_rejection() -> None:
    result = annotate_range_context(known_range_match=False, known_range_distance_km=2500)

    assert result.range_extension_candidate is True
    assert result.review_status == "needs_review"
