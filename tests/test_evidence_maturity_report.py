from __future__ import annotations

import polars as pl
import pytest

from biominer.reports.evidence_maturity import (
    EVIDENCE_MATURITY_LABELS,
    EVIDENCE_MATURITY_SCHEMA,
    evidence_maturity_legend,
    evidence_maturity_payload,
    validate_evidence_maturity_legend,
    validate_evidence_maturity_payload,
)


def test_legend_distinguishes_all_required_evidence_maturities() -> None:
    legend = evidence_maturity_legend()

    assert legend.schema == EVIDENCE_MATURITY_SCHEMA
    assert legend["maturity_label"].to_list() == list(EVIDENCE_MATURITY_LABELS)
    assert legend.filter(
        pl.col("maturity_label") == "provider_asserted_provisional_support"
    )["human_reviewed"].item() is False
    assert legend.filter(pl.col("maturity_label") == "human_verified_support")[
        "human_reviewed"
    ].item() is True
    assert legend.filter(
        pl.col("maturity_label") == "human_reviewed_flickr_labels"
    )["evidence_domain"].item() == "flickr_evaluation_label"


def test_score_probability_and_release_semantics_cannot_collapse() -> None:
    by_label = {
        row["maturity_label"]: row
        for row in evidence_maturity_legend().iter_rows(named=True)
    }

    assert by_label["provisional_raw_score"]["probability_semantics"] is False
    assert by_label["calibrated_probability"]["probability_semantics"] is True
    assert by_label["calibrated_probability"]["release_authorizing"] is False
    assert by_label["final_release_status"]["release_authorizing"] is True
    assert by_label["final_release_status"]["human_reviewed"] is True


def test_maturity_legend_and_embedded_payload_are_tamper_evident() -> None:
    legend = evidence_maturity_legend().with_columns(
        pl.when(pl.col("maturity_label") == "provisional_raw_score")
        .then(pl.lit(True))
        .otherwise(pl.col("probability_semantics"))
        .alias("probability_semantics")
    )
    with pytest.raises(ValueError, match="semantics were weakened"):
        validate_evidence_maturity_legend(legend)

    payload = evidence_maturity_payload()
    payload["labels"][0]["human_reviewed"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="legend mismatch"):
        validate_evidence_maturity_payload(payload)
