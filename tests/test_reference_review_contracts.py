from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import UTC, datetime, timedelta
import hashlib
from pathlib import Path

import polars as pl
import pytest

from biominer.references.schemas import (
    REFERENCE_LIFE_STAGES,
    REFERENCE_REVIEW_CONFIDENCE_VALUES,
    REFERENCE_REVIEW_DECISION_STATUSES,
    REFERENCE_REVIEW_DECISIONS_FILE,
    REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
    REFERENCE_REVIEW_QUEUE_FILE,
    REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
    REFERENCE_REVIEW_QUEUE_STATUSES,
    REFERENCE_VIEWS,
    REFERENCE_VISUAL_DOMAINS,
    make_reference_media_id,
    make_reference_review_decision_id,
    make_reference_review_request_id,
    reference_review_decision_schema,
    reference_review_decisions_frame,
    reference_review_queue_frame,
    reference_review_queue_schema,
    validate_reference_review_decisions,
    validate_reference_review_queue,
    write_reference_review_decisions,
    write_reference_review_queue,
)


NOW = datetime(2026, 7, 14, 1, 2, 3, tzinfo=UTC)
MEDIA_OBJECT_FINGERPRINT = "sha256:" + "a" * 64
INPUT_FINGERPRINT = "sha256:" + "b" * 64
DECISION_SOURCE_HASH = "sha256:" + "c" * 64
REFERENCE_BANK_VERSION = "papilio-demoleus-reference-bank-v1"
REFERENCE_OBSERVATION_ID = "reference-observation:" + "e" * 64
REFERENCE_MEDIA_ID = make_reference_media_id(
    "gbif",
    "provider-photo-1",
    REFERENCE_OBSERVATION_ID,
)
CANONICAL_REFERENCE_MEDIA_ID = "reference-media:" + "f" * 64


def _review_request(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION,
        "review_request_id": "",
        "reference_media_id": REFERENCE_MEDIA_ID,
        "reference_observation_id": REFERENCE_OBSERVATION_ID,
        "canonical_reference_media_id": CANONICAL_REFERENCE_MEDIA_ID,
        "accepted_taxon_key": "gbif:1938069",
        "scientific_name": "Papilio demoleus",
        "durable_preview_uri": (
            "s3://biominer-references/sha256/aa/" + "a" * 64 + ".jpg"
        ),
        "media_object_fingerprint": MEDIA_OBJECT_FINGERPRINT,
        "duplicate_group_id": "reference-duplicate-group:" + "1" * 32,
        "source": "gbif",
        "provider_media_id": "provider-photo-1",
        "provider_verification_status": "accepted",
        "creator": "Example observer",
        "rights_holder": "Example observer",
        "licence": "CC-BY-4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "licence_policy_status": "allowed",
        "attribution": "Example observer / CC BY 4.0",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "review_reason": "first scientific review",
        "review_priority": 10,
        "required_review_count": 1,
        "review_status": "pending",
        "created_at": NOW,
        "reference_bank_version": REFERENCE_BANK_VERSION,
        "input_fingerprint": INPUT_FINGERPRINT,
    }
    row.update(overrides)
    if "reference_media_id" not in overrides:
        row["reference_media_id"] = make_reference_media_id(
            row["source"],
            row["provider_media_id"],
            row["reference_observation_id"],
        )
    row["review_request_id"] = make_reference_review_request_id(
        reference_media_id=row["reference_media_id"],
        media_object_fingerprint=row["media_object_fingerprint"],
        reference_bank_version=row["reference_bank_version"],
        input_fingerprint=row["input_fingerprint"],
    )
    return row


def _review_decision(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "schema_version": REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION,
        "review_decision_id": "",
        "review_request_id": _review_request()["review_request_id"],
        "reference_media_id": REFERENCE_MEDIA_ID,
        "review_round": 1,
        "verified_by": "reviewer-1",
        "reviewed_at": NOW,
        "target_identity_verified": True,
        "verification_status": "verified",
        "life_stage": "adult",
        "visual_domain": "live_field",
        "view": "dorsal",
        "review_confidence": "high",
        "review_notes": "Diagnostic tail and wing pattern are visible.",
        "exclusion_reason": None,
        "second_review_required": False,
        "conflicts_with_decision_id": None,
        "decision_source_hash": DECISION_SOURCE_HASH,
    }
    row.update(overrides)
    row["review_decision_id"] = make_reference_review_decision_id(
        review_request_id=row["review_request_id"],
        reference_media_id=row["reference_media_id"],
        review_round=row["review_round"],
        verified_by=row["verified_by"],
        reviewed_at=row["reviewed_at"],
        target_identity_verified=row["target_identity_verified"],
        verification_status=row["verification_status"],
        life_stage=row["life_stage"],
        visual_domain=row["visual_domain"],
        view=row["view"],
        review_confidence=row["review_confidence"],
        review_notes=row["review_notes"],
        exclusion_reason=row["exclusion_reason"],
        second_review_required=row["second_review_required"],
        conflicts_with_decision_id=row["conflicts_with_decision_id"],
    )
    if "decision_source_hash" not in overrides:
        row["decision_source_hash"] = (
            "sha256:"
            + hashlib.sha256(str(row["review_decision_id"]).encode("ascii")).hexdigest()
        )
    return row


def test_reference_review_vocabularies_are_closed_and_versioned() -> None:
    assert REFERENCE_LIFE_STAGES == frozenset(
        {"adult", "larva", "pupa", "egg", "unknown"}
    )
    assert REFERENCE_VISUAL_DOMAINS == frozenset(
        {
            "live_field",
            "pinned_specimen",
            "artwork",
            "logo",
            "tattoo",
            "partial_wing",
            "dead_or_damaged_specimen",
            "ambiguous",
            "unsuitable",
        }
    )
    assert REFERENCE_VIEWS == frozenset(
        {"dorsal", "ventral", "lateral", "frontal", "oblique", "unknown"}
    )
    assert REFERENCE_REVIEW_CONFIDENCE_VALUES == frozenset(
        {"high", "medium", "low", "unknown"}
    )
    assert REFERENCE_REVIEW_QUEUE_STATUSES == frozenset(
        {
            "pending",
            "in_review",
            "completed",
            "conflict",
            "second_review_required",
            "cancelled",
        }
    )
    assert REFERENCE_REVIEW_DECISION_STATUSES == frozenset(
        {"verified", "excluded", "uncertain"}
    )
    assert REFERENCE_REVIEW_QUEUE_SCHEMA_VERSION == "reference-review-queue-v1.0.0"
    assert REFERENCE_REVIEW_DECISIONS_SCHEMA_VERSION == (
        "reference-review-decisions-v1.0.0"
    )
    assert REFERENCE_REVIEW_QUEUE_FILE == "reference_review_queue.parquet"
    assert REFERENCE_REVIEW_DECISIONS_FILE == "reference_review_decisions.parquet"


def test_reference_review_queue_schema_is_locked() -> None:
    assert reference_review_queue_schema() == {
        "schema_version": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "canonical_reference_media_id": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "durable_preview_uri": pl.String,
        "media_object_fingerprint": pl.String,
        "duplicate_group_id": pl.String,
        "source": pl.String,
        "provider_media_id": pl.String,
        "provider_verification_status": pl.String,
        "creator": pl.String,
        "rights_holder": pl.String,
        "licence": pl.String,
        "licence_uri": pl.String,
        "licence_policy_status": pl.String,
        "attribution": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_reason": pl.String,
        "review_priority": pl.UInt32,
        "required_review_count": pl.UInt8,
        "review_status": pl.String,
        "created_at": pl.Datetime("us", "UTC"),
        "reference_bank_version": pl.String,
        "input_fingerprint": pl.String,
    }


def test_reference_review_decision_schema_is_locked() -> None:
    assert reference_review_decision_schema() == {
        "schema_version": pl.String,
        "review_decision_id": pl.String,
        "review_request_id": pl.String,
        "reference_media_id": pl.String,
        "review_round": pl.UInt16,
        "verified_by": pl.String,
        "reviewed_at": pl.Datetime("us", "UTC"),
        "target_identity_verified": pl.Boolean,
        "verification_status": pl.String,
        "life_stage": pl.String,
        "visual_domain": pl.String,
        "view": pl.String,
        "review_confidence": pl.String,
        "review_notes": pl.String,
        "exclusion_reason": pl.String,
        "second_review_required": pl.Boolean,
        "conflicts_with_decision_id": pl.String,
        "decision_source_hash": pl.String,
    }


def test_reference_review_frames_preserve_exact_schema_and_deterministic_order() -> (
    None
):
    later = _review_request(
        provider_media_id="provider-photo-2",
        review_priority=20,
    )
    earlier = _review_request(
        provider_media_id="provider-photo-3",
        review_priority=5,
    )
    queue = reference_review_queue_frame([later, earlier])

    second = _review_decision(
        review_request_id=later["review_request_id"],
        reference_media_id=later["reference_media_id"],
        review_round=2,
        reviewed_at=NOW + timedelta(minutes=2),
    )
    first = _review_decision(
        review_request_id=earlier["review_request_id"],
        reference_media_id=earlier["reference_media_id"],
        reviewed_at=NOW + timedelta(minutes=1),
    )
    decisions = reference_review_decisions_frame([second, first])

    assert queue.schema == reference_review_queue_schema()
    assert decisions.schema == reference_review_decision_schema()
    assert queue["review_priority"].to_list() == [5, 20]
    expected_decisions = sorted(
        [first, second],
        key=lambda row: (
            row["reference_media_id"],
            row["review_round"],
            row["reviewed_at"],
            row["review_decision_id"],
        ),
    )
    assert decisions["review_decision_id"].to_list() == [
        row["review_decision_id"] for row in expected_decisions
    ]
    assert reference_review_queue_frame([]).schema == reference_review_queue_schema()
    assert reference_review_decisions_frame([]).schema == (
        reference_review_decision_schema()
    )


def test_reference_review_request_id_is_stable_and_input_bound() -> None:
    first = _review_request()
    repeated = _review_request(created_at=NOW + timedelta(days=1), review_priority=99)
    changed_object = _review_request(media_object_fingerprint="sha256:" + "9" * 64)
    changed_input = _review_request(input_fingerprint="sha256:" + "8" * 64)

    assert first["review_request_id"] == repeated["review_request_id"]
    assert first["review_request_id"] != changed_object["review_request_id"]
    assert first["review_request_id"] != changed_input["review_request_id"]


def test_reference_review_decision_id_is_stable_and_semantic() -> None:
    first = _review_decision()
    repeated = _review_decision(decision_source_hash="sha256:" + "7" * 64)
    changed_reviewer = _review_decision(verified_by="reviewer-2")
    changed_confidence = _review_decision(review_confidence="medium")

    assert first["review_decision_id"] == repeated["review_decision_id"]
    assert first["review_decision_id"] != changed_reviewer["review_decision_id"]
    assert first["review_decision_id"] != changed_confidence["review_decision_id"]
    assert reference_review_decisions_frame([repeated]).height == 1


def test_reference_review_queue_allows_unresolved_taxonomy() -> None:
    frame = reference_review_queue_frame(
        [_review_request(accepted_taxon_key=None, scientific_name=None)]
    )

    assert frame["accepted_taxon_key"].item() is None
    assert frame["scientific_name"].item() is None


def test_reference_review_queue_preserves_missing_provisional_routing() -> None:
    frame = reference_review_queue_frame(
        [_review_request(life_stage=None, visual_domain=None, view=None)]
    )

    assert frame.select("life_stage", "visual_domain", "view").row(0) == (
        None,
        None,
        None,
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("life_stage", "imago"),
        ("life_stage", "Adult"),
        ("visual_domain", "field"),
        ("visual_domain", "Live_Field"),
        ("view", "top"),
        ("review_status", "verified"),
    ],
)
def test_reference_review_queue_rejects_values_outside_closed_vocabularies(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError):
        reference_review_queue_frame([_review_request(**{field: value})])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_reason", "  "),
        ("required_review_count", 0),
        ("licence", None),
        ("verified_by", None),
    ],
)
def test_reference_review_contracts_require_review_evidence(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        if field == "verified_by":
            reference_review_decisions_frame([_review_decision(**{field: value})])
        else:
            reference_review_queue_frame([_review_request(**{field: value})])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("media_object_fingerprint", "sha256:ABC"),
        ("input_fingerprint", "b" * 64),
    ],
)
def test_reference_review_queue_requires_full_lowercase_fingerprints(
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        reference_review_queue_frame([_review_request(**{field: value})])


def test_reference_review_queue_rejects_stale_or_duplicate_identity() -> None:
    row = _review_request()
    mismatched = deepcopy(row)
    mismatched["review_request_id"] = "reference-review-request:stale"

    with pytest.raises(ValueError):
        reference_review_queue_frame([mismatched])
    with pytest.raises(ValueError):
        reference_review_queue_frame([row, deepcopy(row)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source", "inaturalist"),
        ("provider_media_id", "different-provider-photo"),
        ("reference_observation_id", "reference-observation:" + "9" * 64),
    ],
)
def test_reference_review_queue_binds_media_id_to_provider_provenance(
    field: str,
    value: str,
) -> None:
    row = _review_request()
    row[field] = value

    with pytest.raises(ValueError):
        reference_review_queue_frame([row])


def test_reference_review_queue_requires_one_canonical_id_per_duplicate_group() -> None:
    first = _review_request(provider_media_id="provider-photo-2")
    second = _review_request(
        provider_media_id="provider-photo-3",
        canonical_reference_media_id="reference-media:" + "9" * 64,
    )

    with pytest.raises(ValueError):
        reference_review_queue_frame([first, second])

    other_group = _review_request(
        provider_media_id="provider-photo-3",
        duplicate_group_id="reference-duplicate-group:" + "2" * 32,
    )
    with pytest.raises(ValueError):
        reference_review_queue_frame([first, other_group])


@pytest.mark.parametrize(
    ("factory", "field", "value"),
    [
        (_review_request, "review_status", " pending "),
        (_review_request, "life_stage", " adult "),
        (_review_request, "source", " gbif "),
        (_review_request, "input_fingerprint", INPUT_FINGERPRINT + " "),
        (_review_decision, "verification_status", " verified "),
        (_review_decision, "visual_domain", " live_field "),
        (_review_decision, "verified_by", " reviewer-1 "),
        (_review_decision, "decision_source_hash", DECISION_SOURCE_HASH + " "),
    ],
)
def test_reference_review_contracts_reject_noncanonical_whitespace(
    factory: Callable[..., dict[str, object]],
    field: str,
    value: str,
) -> None:
    with pytest.raises(ValueError):
        row = factory(**{field: value})
        if factory is _review_request:
            reference_review_queue_frame([row])
        else:
            reference_review_decisions_frame([row])


def test_verified_decision_requires_an_affirmative_identity_verdict() -> None:
    valid = reference_review_decisions_frame([_review_decision()])
    assert valid["target_identity_verified"].item() is True

    invalid_rows = [
        _review_decision(target_identity_verified=False),
        _review_decision(target_identity_verified=None),
        _review_decision(exclusion_reason="damaged specimen"),
        _review_decision(second_review_required=True),
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError):
            reference_review_decisions_frame([row])


@pytest.mark.parametrize("target_identity_verified", [True, False, None])
def test_excluded_decision_retains_independent_identity_evidence(
    target_identity_verified: bool | None,
) -> None:
    row = _review_decision(
        verification_status="excluded",
        target_identity_verified=target_identity_verified,
        visual_domain="artwork",
        exclusion_reason="not suitable as live-field support",
        review_confidence="medium",
    )

    frame = reference_review_decisions_frame([row])

    assert frame["verification_status"].item() == "excluded"
    assert frame["target_identity_verified"].item() is target_identity_verified


def test_excluded_decision_requires_a_reason() -> None:
    for reason in (None, "  "):
        with pytest.raises(ValueError):
            reference_review_decisions_frame(
                [
                    _review_decision(
                        verification_status="excluded",
                        target_identity_verified=False,
                        exclusion_reason=reason,
                    )
                ]
            )


def test_uncertain_decision_is_unresolved_and_requests_second_review() -> None:
    valid = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_confidence="unknown",
        review_notes="The identifying marks are occluded.",
        second_review_required=True,
    )
    frame = reference_review_decisions_frame([valid])
    assert frame["target_identity_verified"].item() is None
    assert frame["review_confidence"].item() == "unknown"

    invalid_rows = [
        _review_decision(
            verification_status="uncertain",
            target_identity_verified=True,
            second_review_required=True,
        ),
        _review_decision(
            verification_status="uncertain",
            target_identity_verified=None,
            review_notes=None,
            second_review_required=True,
        ),
        _review_decision(
            verification_status="uncertain",
            target_identity_verified=None,
            second_review_required=False,
        ),
    ]
    for row in invalid_rows:
        with pytest.raises(ValueError):
            reference_review_decisions_frame([row])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verification_status", "accepted"),
        ("verification_status", "pending"),
        ("life_stage", "Adult"),
        ("visual_domain", "field"),
        ("view", "top"),
        ("review_confidence", "0.9"),
        ("review_confidence", None),
    ],
)
def test_reference_review_decision_rejects_values_outside_closed_vocabularies(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError)):
        reference_review_decisions_frame([_review_decision(**{field: value})])


def test_conflict_pointer_is_only_valid_for_an_uncertain_decision() -> None:
    prior = _review_decision(verified_by="reviewer-2")
    conflicting_id = prior["review_decision_id"]
    valid = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_notes="Reviewer evidence conflicts with the prior decision.",
        review_confidence="low",
        review_round=2,
        reviewed_at=NOW + timedelta(minutes=1),
        second_review_required=True,
        conflicts_with_decision_id=conflicting_id,
    )
    assert reference_review_decisions_frame([valid, prior]).height == 2

    with pytest.raises(ValueError):
        reference_review_decisions_frame(
            [_review_decision(conflicts_with_decision_id=conflicting_id)]
        )

    dangling = deepcopy(valid)
    dangling["conflicts_with_decision_id"] = "reference-review-decision:" + "9" * 64
    dangling["review_decision_id"] = make_reference_review_decision_id(
        **{
            field: dangling[field]
            for field in (
                "review_request_id",
                "reference_media_id",
                "review_round",
                "verified_by",
                "reviewed_at",
                "target_identity_verified",
                "verification_status",
                "life_stage",
                "visual_domain",
                "view",
                "review_confidence",
                "review_notes",
                "exclusion_reason",
                "second_review_required",
                "conflicts_with_decision_id",
            )
        }
    )
    with pytest.raises(ValueError):
        reference_review_decisions_frame([dangling, prior])

    self_conflict = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_notes="Conflicting evidence.",
        second_review_required=True,
    )
    self_conflict["conflicts_with_decision_id"] = self_conflict["review_decision_id"]
    with pytest.raises(ValueError):
        reference_review_decisions_frame([self_conflict])


def test_conflict_pointer_requires_prior_same_media_decision_from_another_actor() -> (
    None
):
    same_actor_prior = _review_decision()
    same_actor_conflict = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_round=2,
        reviewed_at=NOW + timedelta(minutes=1),
        review_notes="Conflicting evidence.",
        second_review_required=True,
        conflicts_with_decision_id=same_actor_prior["review_decision_id"],
    )
    with pytest.raises(ValueError):
        reference_review_decisions_frame([same_actor_prior, same_actor_conflict])

    other_request = _review_request(provider_media_id="provider-photo-2")
    other_prior = _review_decision(
        review_request_id=other_request["review_request_id"],
        reference_media_id=other_request["reference_media_id"],
        verified_by="reviewer-2",
    )
    cross_media_conflict = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_round=2,
        reviewed_at=NOW + timedelta(minutes=1),
        review_notes="Conflicting evidence.",
        second_review_required=True,
        conflicts_with_decision_id=other_prior["review_decision_id"],
    )
    with pytest.raises(ValueError):
        reference_review_decisions_frame([other_prior, cross_media_conflict])

    future_prior = _review_decision(
        verified_by="reviewer-2",
        reviewed_at=NOW + timedelta(minutes=2),
    )
    forward_conflict = _review_decision(
        verification_status="uncertain",
        target_identity_verified=None,
        review_round=2,
        reviewed_at=NOW + timedelta(minutes=1),
        review_notes="Conflicting evidence.",
        second_review_required=True,
        conflicts_with_decision_id=future_prior["review_decision_id"],
    )
    with pytest.raises(ValueError):
        reference_review_decisions_frame([future_prior, forward_conflict])


def test_reference_review_decision_requires_positive_round_and_source_hash() -> None:
    with pytest.raises(ValueError):
        reference_review_decisions_frame([_review_decision(review_round=0)])
    with pytest.raises(ValueError):
        reference_review_decisions_frame(
            [_review_decision(decision_source_hash="SHA256:" + "C" * 64)]
        )


def test_source_decision_record_cannot_produce_multiple_semantic_decisions() -> None:
    first = _review_decision(decision_source_hash=DECISION_SOURCE_HASH)
    second = _review_decision(
        verified_by="reviewer-2",
        decision_source_hash=DECISION_SOURCE_HASH,
    )

    with pytest.raises(ValueError):
        reference_review_decisions_frame([first, second])


def test_reference_review_decision_rejects_stale_or_duplicate_identity() -> None:
    row = _review_decision()
    mismatched = deepcopy(row)
    mismatched["review_decision_id"] = "reference-review-decision:stale"

    with pytest.raises(ValueError):
        reference_review_decisions_frame([mismatched])
    with pytest.raises(ValueError):
        reference_review_decisions_frame([row, deepcopy(row)])


def test_reference_review_validators_reject_schema_and_sort_drift() -> None:
    queue = reference_review_queue_frame(
        [
            _review_request(
                provider_media_id="provider-photo-2",
                review_priority=1,
            ),
            _review_request(
                provider_media_id="provider-photo-3",
                review_priority=2,
            ),
        ]
    )
    decisions = reference_review_decisions_frame([_review_decision()])

    with pytest.raises(ValueError):
        validate_reference_review_queue(queue.reverse())
    with pytest.raises(ValueError):
        validate_reference_review_queue(queue.select(reversed(queue.columns)))
    with pytest.raises(ValueError):
        validate_reference_review_decisions(
            decisions.with_columns(pl.col("reviewed_at").dt.replace_time_zone(None))
        )


def test_reference_review_writers_validate_and_round_trip(tmp_path: Path) -> None:
    queue = reference_review_queue_frame([_review_request()])
    decisions = reference_review_decisions_frame([_review_decision()])

    queue_path = write_reference_review_queue(queue, tmp_path)
    decisions_path = write_reference_review_decisions(decisions, tmp_path)

    assert queue_path == tmp_path / REFERENCE_REVIEW_QUEUE_FILE
    assert decisions_path == tmp_path / REFERENCE_REVIEW_DECISIONS_FILE
    assert pl.read_parquet(queue_path).equals(queue)
    assert pl.read_parquet(decisions_path).equals(decisions)
    assert not list(tmp_path.glob("*.tmp"))

    with pytest.raises(FileExistsError):
        write_reference_review_decisions(
            reference_review_decisions_frame(
                [_review_decision(review_confidence="medium")]
            ),
            tmp_path,
        )
    assert pl.read_parquet(decisions_path).equals(decisions)


def test_reference_review_decision_writer_is_atomic_create_only(
    tmp_path: Path,
) -> None:
    output = tmp_path / "concurrent"
    frames = [
        reference_review_decisions_frame(
            [_review_decision(verified_by=f"reviewer-{index}")]
        )
        for index in range(8)
    ]

    def publish(frame: pl.DataFrame) -> Path | None:
        try:
            return write_reference_review_decisions(frame, output)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=len(frames)) as executor:
        results = list(executor.map(publish, frames))

    assert sum(result is not None for result in results) == 1
    published = pl.read_parquet(output / REFERENCE_REVIEW_DECISIONS_FILE)
    assert published["review_decision_id"].item() in {
        frame["review_decision_id"].item() for frame in frames
    }
    assert not list(output.glob(".*.tmp"))
