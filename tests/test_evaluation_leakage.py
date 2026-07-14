from __future__ import annotations

from dataclasses import replace
import hashlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from biominer.evaluation.leakage import (
    EVALUATION_LEAKAGE_REGISTER_SCHEMA,
    EvaluationLeakageError,
    EvaluationLeakageIdentity,
    build_evaluation_leakage_register,
    empty_evaluation_leakage_register,
    find_evaluation_leakage,
    validate_evaluation_leakage_register,
    validate_reference_and_holdout_leakage,
)
from test_evaluation_holdouts import _frozen_holdout_pair, _leakage_register


@pytest.mark.parametrize(
    ("field", "expected_dimension"),
    [
        ("visual_content_sha256", "exact_hash"),
        ("perceptual_duplicate_group_id", "perceptual_duplicate_group"),
        ("source_observation_id", "source_observation"),
        ("photographer_id", "photographer"),
        ("flickr_owner_id", "flickr_owner"),
        ("provider_mirror_group_id", "provider_mirror"),
        ("geographic_burst_group_id", "geographic_burst"),
    ],
)
def test_rejects_reference_holdout_leakage_for_every_required_identity(
    field: str,
    expected_dimension: str,
) -> None:
    support = _identity("support", partition="support_train")
    holdout = _identity("holdout", partition="balanced_challenge")
    holdout = replace(holdout, **{field: getattr(support, field)})

    with pytest.raises(EvaluationLeakageError) as error:
        build_evaluation_leakage_register(
            (support, holdout),
            register_version="leak-test-v1",
        )

    assert expected_dimension in set(error.value.findings["identity_dimension"])
    assert set(error.value.findings["partitions"].explode()) == {
        "support_train",
        "balanced_challenge",
    }


def test_register_is_deterministic_and_allows_shared_identity_within_partition() -> (
    None
):
    first_item = _identity("support-a", partition="support_train")
    second_item = replace(
        _identity("support-b", partition="support_train"),
        visual_content_sha256=first_item.visual_content_sha256,
        perceptual_duplicate_group_id=first_item.perceptual_duplicate_group_id,
        source_observation_id=first_item.source_observation_id,
        photographer_id=first_item.photographer_id,
        flickr_owner_id=first_item.flickr_owner_id,
        provider_mirror_group_id=first_item.provider_mirror_group_id,
        geographic_burst_group_id=first_item.geographic_burst_group_id,
    )

    first = build_evaluation_leakage_register(
        (first_item, second_item),
        register_version="same-partition-v1",
    )
    second = build_evaluation_leakage_register(
        (second_item, first_item),
        register_version="same-partition-v1",
    )

    assert_frame_equal(first, second)
    assert first.schema == EVALUATION_LEAKAGE_REGISTER_SCHEMA
    assert find_evaluation_leakage(first).is_empty()
    validate_evaluation_leakage_register(first)


def test_missing_optional_identities_do_not_create_false_groups() -> None:
    support = _identity(
        "support",
        partition="support_train",
        source="gbif",
        optional_identities=False,
    )
    holdout = _identity(
        "holdout",
        partition="balanced_challenge",
        source="gbif",
        optional_identities=False,
    )

    register = build_evaluation_leakage_register(
        (support, holdout),
        register_version="missing-optional-v1",
    )

    assert find_evaluation_leakage(register).is_empty()


def test_person_identity_leakage_is_detected_across_owner_field_names() -> None:
    support = _identity("support", partition="support_train")
    holdout = replace(
        _identity("holdout", partition="balanced_challenge"),
        flickr_owner_id=support.photographer_id,
    )

    with pytest.raises(EvaluationLeakageError) as error:
        build_evaluation_leakage_register(
            (support, holdout),
            register_version="cross-role-person-v1",
        )

    assert "person_or_owner" in set(error.value.findings["identity_dimension"])


def test_leakage_findings_are_deterministic_and_include_all_crossing_items() -> None:
    support = _identity("support", partition="support_train")
    challenge = replace(
        _identity("challenge", partition="balanced_challenge"),
        provider_mirror_group_id=support.provider_mirror_group_id,
    )
    natural = replace(
        _identity("natural", partition="natural_stream"),
        provider_mirror_group_id=support.provider_mirror_group_id,
    )

    with pytest.raises(EvaluationLeakageError) as first_error:
        build_evaluation_leakage_register(
            (support, challenge, natural),
            register_version="deterministic-leak-v1",
        )
    with pytest.raises(EvaluationLeakageError) as second_error:
        build_evaluation_leakage_register(
            (natural, challenge, support),
            register_version="deterministic-leak-v1",
        )

    assert_frame_equal(first_error.value.findings, second_error.value.findings)
    finding = first_error.value.findings.filter(
        pl.col("identity_dimension") == "provider_mirror"
    ).row(0, named=True)
    assert finding["partitions"] == [
        "balanced_challenge",
        "natural_stream",
        "support_train",
    ]
    assert finding["item_ids"] == ["challenge", "natural", "support"]


def test_register_rejects_noncanonical_or_incomplete_identity_data() -> None:
    assert empty_evaluation_leakage_register().schema == (
        EVALUATION_LEAKAGE_REGISTER_SCHEMA
    )
    with pytest.raises(TypeError, match="EvaluationLeakageIdentity"):
        build_evaluation_leakage_register(  # type: ignore[arg-type]
            (object(),),
            register_version="invalid-type-v1",
        )
    with pytest.raises(ValueError, match="full lowercase sha256"):
        replace(
            _identity("bad-hash", partition="support_train"),
            visual_content_sha256="not-a-hash",
        )
    with pytest.raises(ValueError, match="flickr_owner_id"):
        replace(
            _identity("missing-owner", partition="support_train"),
            flickr_owner_id=None,
        )
    with pytest.raises(ValueError, match="must be non-empty canonical text"):
        replace(
            _identity("blank", partition="support_train"),
            photographer_id=" ",
        )
    valid = build_evaluation_leakage_register(
        (_identity("nullable", partition="support_train"),),
        register_version="nullable-v1",
    )
    missing_observation = valid.with_columns(
        pl.lit(None, dtype=pl.String).alias("source_observation_id")
    )
    with pytest.raises(ValueError, match="source_observation_id"):
        validate_evaluation_leakage_register(missing_observation)


def test_reference_holdout_gate_requires_exact_coverage_and_artifact_identity() -> None:
    challenge, natural = _frozen_holdout_pair()
    register = _leakage_register(challenge, natural)

    audit = validate_reference_and_holdout_leakage(
        register,
        challenge,
        natural,
    )

    assert audit.register_item_count == 12
    assert audit.reference_item_count == 1
    assert audit.balanced_challenge_item_count == 7
    assert audit.natural_stream_item_count == 4
    assert dict(audit.coverage_by_dimension)["visual_content_sha256"] == 12

    missing_id = str(challenge["evaluation_item_id"][0])
    incomplete = build_evaluation_leakage_register(
        tuple(
            _identity_from_row(row)
            for row in register.filter(pl.col("item_id") != missing_id).iter_rows(
                named=True
            )
        ),
        register_version="papilio-demoleus-leakage-v1",
    )
    with pytest.raises(ValueError, match="identity coverage mismatch"):
        validate_reference_and_holdout_leakage(
            incomplete,
            challenge,
            natural,
        )


def test_reference_holdout_gate_rejects_stale_holdout_fingerprint() -> None:
    challenge, natural = _frozen_holdout_pair()
    register = _leakage_register(challenge, natural)
    challenge_id = str(challenge["evaluation_item_id"][0])
    stale = build_evaluation_leakage_register(
        tuple(
            replace(
                identity,
                source_artifact_fingerprint=_sha("stale-holdout"),
            )
            if identity.item_id == challenge_id
            else identity
            for identity in (
                _identity_from_row(row) for row in register.iter_rows(named=True)
            )
        ),
        register_version="papilio-demoleus-leakage-v1",
    )

    with pytest.raises(ValueError, match="wrong holdout fingerprint"):
        validate_reference_and_holdout_leakage(stale, challenge, natural)


def _identity(
    suffix: str,
    *,
    partition: str,
    source: str = "flickr",
    optional_identities: bool = True,
) -> EvaluationLeakageIdentity:
    return EvaluationLeakageIdentity(
        item_id=suffix,
        partition=partition,
        source_artifact_kind="fixture",
        source_artifact_fingerprint=_sha(f"artifact:{suffix}"),
        source=source,
        source_observation_id=f"observation:{suffix}",
        visual_content_sha256=_sha(f"content:{suffix}"),
        duplicate_group_id=(f"duplicate:{suffix}" if optional_identities else None),
        perceptual_duplicate_group_id=f"perceptual:{suffix}",
        observer_owner_group_id=(
            f"owner-group:{suffix}" if optional_identities else None
        ),
        photographer_id=(f"photographer:{suffix}" if optional_identities else None),
        flickr_owner_id=(
            f"flickr-owner:{suffix}" if source.startswith("flickr") else None
        ),
        provider_mirror_group_id=(f"mirror:{suffix}" if optional_identities else None),
        geographic_burst_group_id=(f"burst:{suffix}" if optional_identities else None),
    )


def _identity_from_row(row: dict[str, object]) -> EvaluationLeakageIdentity:
    return EvaluationLeakageIdentity(
        item_id=str(row["item_id"]),
        partition=str(row["partition"]),
        source_artifact_kind=str(row["source_artifact_kind"]),
        source_artifact_fingerprint=str(row["source_artifact_fingerprint"]),
        source=str(row["source"]),
        source_observation_id=str(row["source_observation_id"]),
        visual_content_sha256=str(row["visual_content_sha256"]),
        duplicate_group_id=_optional_text(row["duplicate_group_id"]),
        perceptual_duplicate_group_id=str(row["perceptual_duplicate_group_id"]),
        observer_owner_group_id=_optional_text(row["observer_owner_group_id"]),
        photographer_id=_optional_text(row["photographer_id"]),
        flickr_owner_id=_optional_text(row["flickr_owner_id"]),
        provider_mirror_group_id=_optional_text(row["provider_mirror_group_id"]),
        geographic_burst_group_id=_optional_text(row["geographic_burst_group_id"]),
    )


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()
