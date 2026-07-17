"""Canonical scientific evidence-maturity labels for adaptive reports."""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


EVIDENCE_MATURITY_SCHEMA_VERSION = "evidence-maturity-v1.0.0"
EVIDENCE_MATURITY_LABELS = (
    "provider_asserted_provisional_support",
    "human_verified_support",
    "human_reviewed_flickr_labels",
    "provisional_raw_score",
    "calibrated_probability",
    "final_release_status",
)

EVIDENCE_MATURITY_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "maturity_label": pl.String,
    "evidence_domain": pl.String,
    "human_reviewed": pl.Boolean,
    "probability_semantics": pl.Boolean,
    "release_authorizing": pl.Boolean,
    "allowed_claim": pl.String,
    "prohibited_claims": pl.List(pl.String),
    "maturity_fingerprint": pl.String,
}

_DEFINITIONS: dict[str, dict[str, object]] = {
    "provider_asserted_provisional_support": {
        "evidence_domain": "reference_identity_support",
        "human_reviewed": False,
        "probability_semantics": False,
        "release_authorizing": False,
        "allowed_claim": "GBIF asserted the reconciled taxon and automated gates permit provisional model support.",
        "prohibited_claims": [
            "human_verified_reference_identity",
            "flickr_ground_truth_label",
            "final_scientific_release",
        ],
    },
    "human_verified_support": {
        "evidence_domain": "reference_identity_support",
        "human_reviewed": True,
        "probability_semantics": False,
        "release_authorizing": False,
        "allowed_claim": "A human review resolved this reference as eligible model support.",
        "prohibited_claims": [
            "flickr_ground_truth_label",
            "calibrated_probability",
            "automatic_final_scientific_release",
        ],
    },
    "human_reviewed_flickr_labels": {
        "evidence_domain": "flickr_evaluation_label",
        "human_reviewed": True,
        "probability_semantics": False,
        "release_authorizing": False,
        "allowed_claim": "A source-bound human decision supplies an evaluation or release label, subject to conflict resolution.",
        "prohibited_claims": [
            "reference_identity_support",
            "model_probability",
            "release_without_resolved_review",
        ],
    },
    "provisional_raw_score": {
        "evidence_domain": "model_output",
        "human_reviewed": False,
        "probability_semantics": False,
        "release_authorizing": False,
        "allowed_claim": "An uncalibrated model score ranks provisional screening evidence.",
        "prohibited_claims": [
            "probability",
            "human_verified_identity",
            "final_scientific_release",
        ],
    },
    "calibrated_probability": {
        "evidence_domain": "model_output",
        "human_reviewed": False,
        "probability_semantics": True,
        "release_authorizing": False,
        "allowed_claim": "A fitted and provenance-bound calibrator maps model evidence to a probability estimate.",
        "prohibited_claims": [
            "human_verification",
            "taxonomic_ground_truth",
            "automatic_final_scientific_release",
        ],
    },
    "final_release_status": {
        "evidence_domain": "scientific_release_decision",
        "human_reviewed": True,
        "probability_semantics": False,
        "release_authorizing": True,
        "allowed_claim": "An explicit final status may authorize release only after all human-review and conflict rules pass.",
        "prohibited_claims": [
            "release_from_provider_assertion_alone",
            "release_from_raw_score_alone",
            "release_from_probability_alone",
        ],
    },
}


def evidence_maturity_legend() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for label in EVIDENCE_MATURITY_LABELS:
        row: dict[str, object] = {
            "schema_version": EVIDENCE_MATURITY_SCHEMA_VERSION,
            "maturity_label": label,
            **_DEFINITIONS[label],
            "maturity_fingerprint": "",
        }
        row["maturity_fingerprint"] = _fingerprint_without(
            row, "maturity_fingerprint"
        )
        rows.append(row)
    frame = pl.DataFrame(
        rows,
        schema=EVIDENCE_MATURITY_SCHEMA,
        orient="row",
        strict=True,
    )
    validate_evidence_maturity_legend(frame)
    return frame


def validate_evidence_maturity_legend(frame: pl.DataFrame) -> None:
    if frame.schema != EVIDENCE_MATURITY_SCHEMA:
        raise ValueError("evidence maturity schema mismatch")
    if frame["maturity_label"].to_list() != list(EVIDENCE_MATURITY_LABELS):
        raise ValueError("evidence maturity labels are incomplete or reordered")
    for row in frame.iter_rows(named=True):
        if row["schema_version"] != EVIDENCE_MATURITY_SCHEMA_VERSION:
            raise ValueError("unsupported evidence maturity version")
        expected = _DEFINITIONS[str(row["maturity_label"])]
        if any(row[field] != value for field, value in expected.items()):
            raise ValueError("evidence maturity semantics were weakened")
        if len(row["prohibited_claims"]) != len(set(row["prohibited_claims"])):
            raise ValueError("evidence maturity prohibited claims repeat")
        if row["maturity_fingerprint"] != _fingerprint_without(
            row, "maturity_fingerprint"
        ):
            raise ValueError("evidence maturity fingerprint mismatch")
    raw = frame.filter(pl.col("maturity_label") == "provisional_raw_score").row(
        0, named=True
    )
    calibrated = frame.filter(
        pl.col("maturity_label") == "calibrated_probability"
    ).row(0, named=True)
    release = frame.filter(pl.col("maturity_label") == "final_release_status").row(
        0, named=True
    )
    if raw["probability_semantics"] or raw["release_authorizing"]:
        raise ValueError("raw score maturity cannot imply probability or release")
    if not calibrated["probability_semantics"] or calibrated["release_authorizing"]:
        raise ValueError("calibrated probability maturity semantics are invalid")
    if not release["human_reviewed"] or not release["release_authorizing"]:
        raise ValueError("final release maturity requires human review")


def evidence_maturity_payload() -> dict[str, object]:
    legend = evidence_maturity_legend()
    return {
        "schema_version": EVIDENCE_MATURITY_SCHEMA_VERSION,
        "labels": legend.to_dicts(),
        "legend_fingerprint": canonical_semantic_fingerprint(
            {
                "schema": [
                    (name, str(dtype)) for name, dtype in legend.schema.items()
                ],
                "rows": legend.to_dicts(),
            }
        ),
    }


def validate_evidence_maturity_payload(value: object) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("report evidence maturity must be an object")
    if value != evidence_maturity_payload():
        raise ValueError("report evidence maturity legend mismatch")


def _fingerprint_without(row: Mapping[str, object], field: str) -> str:
    payload = dict(row)
    payload.pop(field)
    return canonical_semantic_fingerprint(payload)


__all__ = [
    "EVIDENCE_MATURITY_LABELS",
    "EVIDENCE_MATURITY_SCHEMA",
    "EVIDENCE_MATURITY_SCHEMA_VERSION",
    "evidence_maturity_legend",
    "evidence_maturity_payload",
    "validate_evidence_maturity_legend",
    "validate_evidence_maturity_payload",
]
