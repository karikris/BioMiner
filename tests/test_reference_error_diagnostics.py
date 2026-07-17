from __future__ import annotations

import polars as pl
import pytest

from biominer.evaluation.reference_error_diagnostics import (
    relate_errors_to_reference_quality,
)


def test_underperforming_species_errors_join_to_descriptive_reference_evidence() -> None:
    performance = pl.DataFrame(
        {
            "target_species": ["Papilio demoleus", "Danaus plexippus"],
            "competitor_species": ["Papilio polytes", "Danaus gilippus"],
            "region": ["geo:qld", "geo:qld"],
            "route": ["adult_field", "adult_field"],
            "metric_status": ["complete", "complete"],
            "false_positive_rate": [0.3, 0.0],
            "false_negative_rate": [0.2, 0.0],
        }
    )
    diagnostics = pl.DataFrame(
        {
            "species": ["Papilio demoleus", "Papilio demoleus"],
            "route": ["adult_field", "adult_field"],
            "embedding_outlier_score": [0.5, 0.1],
            "review_threshold": [0.35, 0.35],
            "prototype_influence": [0.2, 0.01],
            "route_domain_mismatch": [True, False],
            "taxon_misidentification_conclusion": ["not_assessed", "not_assessed"],
        }
    )
    prototypes = pl.DataFrame(
        {
            "species": ["Papilio demoleus", "Papilio demoleus"],
            "route": ["adult_field", "adult_field"],
            "dispersion": [0.2, 0.4],
        }
    )
    support = pl.DataFrame(
        {
            "scientific_name": [
                "Papilio demoleus",
                "Papilio demoleus",
                "Papilio demoleus",
                "Papilio polytes",
            ],
            "route": ["adult_field"] * 4,
            "provider_dataset_key": ["dataset:a", "dataset:a", "dataset:b", "dataset:c"],
            "geo_cluster_id": ["geo:qld", "geo:qld", None, "geo:qld"],
            "observer_id": ["observer:1", "observer:2", "observer:2", "observer:3"],
            "support_eligible": [True, True, True, True],
        }
    )

    result = relate_errors_to_reference_quality(
        performance,
        diagnostics,
        prototypes,
        support,
        underperforming_species=("Papilio demoleus",),
    )
    row = result.row(0, named=True)

    assert result.height == 1
    assert row["model_error_rate"] == pytest.approx(0.25)
    assert row["prototype_dispersion_mean"] == pytest.approx(0.3)
    assert row["prototype_dispersion_max"] == pytest.approx(0.4)
    assert row["reference_outlier_count"] == 1
    assert row["high_influence_reference_count"] == 1
    assert row["route_mismatch_reference_count"] == 1
    assert row["provider_dataset_count"] == 2
    assert row["largest_provider_dataset_fraction"] == pytest.approx(2 / 3)
    assert row["geographic_support_gap"] is False
    assert row["target_reference_count"] == 3
    assert row["competitor_reference_count"] == 1
    assert row["observer_diversity_ratio"] == pytest.approx(2 / 3)
    assert row["prioritization_evidence"] is True
    assert row["reference_identity_conclusion"] == "not_assessed"
    assert row["diagnostic_fingerprint"].startswith("sha256:")


def test_error_linkage_rejects_diagnostics_that_claim_misidentification() -> None:
    performance = pl.DataFrame(
        {
            "target_species": ["Papilio demoleus"],
            "competitor_species": ["Papilio polytes"],
            "region": ["global"],
            "route": ["adult_field"],
            "metric_status": ["complete"],
            "false_positive_rate": [0.3],
            "false_negative_rate": [0.2],
        }
    )
    diagnostics = pl.DataFrame(
        {
            "species": ["Papilio demoleus"],
            "route": ["adult_field"],
            "embedding_outlier_score": [0.5],
            "review_threshold": [0.35],
            "prototype_influence": [0.2],
            "route_domain_mismatch": [False],
            "taxon_misidentification_conclusion": ["misidentified"],
        }
    )
    prototypes = pl.DataFrame({"species": [], "route": [], "dispersion": []})
    support = pl.DataFrame(
        schema={
            "scientific_name": pl.String,
            "route": pl.String,
            "provider_dataset_key": pl.String,
            "geo_cluster_id": pl.String,
            "observer_id": pl.String,
            "support_eligible": pl.Boolean,
        }
    )

    with pytest.raises(ValueError, match="must not assert misidentification"):
        relate_errors_to_reference_quality(
            performance,
            diagnostics,
            prototypes,
            support,
            underperforming_species=("Papilio demoleus",),
        )
