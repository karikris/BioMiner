from __future__ import annotations

import pyarrow as pa

from biominer.gbif_quality.policy import build_field_policy, field_policy_table


def test_policy_covers_schema_and_preserves_originals() -> None:
    schema = pa.schema(
        [
            ("gbifID", pa.string()),
            ("species", pa.string()),
            ("infraspecificEpithet", pa.string()),
            ("coordinateUncertaintyInMeters", pa.string()),
            ("media_format", pa.string()),
        ]
    )
    policies = build_field_policy(schema)
    assert [policy.field_name for policy in policies] == schema.names
    assert all(policy.preserve_original for policy in policies)
    by_name = {policy.field_name: policy for policy in policies}
    assert by_name["species"].applicability_rule == "species_rank_or_below"
    assert by_name["infraspecificEpithet"].applicability_rule == "below_species_rank"
    assert by_name["coordinateUncertaintyInMeters"].applicability_rule == "coordinates_present"
    assert by_name["media_format"].scope == "media_assertion"
    assert field_policy_table(policies).num_rows == len(schema)


def test_policy_keeps_identification_evidence_independent() -> None:
    schema = pa.schema(
        [
            ("identifiedBy", pa.string()),
            ("identificationVerificationStatus", pa.string()),
        ]
    )
    by_name = {policy.field_name: policy for policy in build_field_policy(schema)}
    assert "Must not be inferred" in by_name["identifiedBy"].policy_note
    assert "Must not be inferred" in by_name["identificationVerificationStatus"].policy_note
    assert not by_name["identifiedBy"].allowed_derivation_sources


def test_gbif_taxon_keys_remain_strings() -> None:
    policy = build_field_policy(pa.schema([("acceptedTaxonKey", pa.string())]))[0]

    assert policy.valid_type == "gbif_taxon_key_string"
