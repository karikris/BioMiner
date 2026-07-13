from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import polars as pl
import pytest

from biominer.candidates.regional_occurrence import regional_taxon_occurrence_schema
from biominer.candidates.regional_union import (
    RegionalCandidateConfig,
    build_regional_candidate_species,
)
from biominer.candidates.relationships import (
    COMPETITOR_RELATIONSHIPS_FILE,
    COMPETITOR_RELATIONSHIPS_SCHEMA_VERSION,
    COMPETITOR_RELATIONSHIP_SOURCE_SCHEMA_VERSION,
    compile_competitor_relationships,
    competitor_relationships_schema,
    load_competitor_relationship_source,
    write_competitor_relationships,
)


TARGET = "gbif:1938069"
CONGENER = "gbif:100"
GRAPHIUM_SPECIES = "gbif:101"
MIMIC = "gbif:102"
SHA_A = "sha256:" + "a" * 64
SHA_B = "sha256:" + "b" * 64
PILOT_GENERA = {
    "Graphium": "gbif:1937188",
    "Losaria": "gbif:1939221",
    "Ornithoptera": "gbif:1937440",
    "Pachliopta": "gbif:1939152",
    "Protographium": "gbif:1939129",
}


def _taxon(
    key: str,
    name: str,
    rank: str,
    family: str,
    genus: str,
) -> dict[str, object]:
    return {
        "accepted_taxon_key": key,
        "scientific_name": name,
        "rank": rank,
        "taxonomic_status": "ACCEPTED",
        "family": family,
        "genus": genus,
        "in_scope": True,
    }


def _taxa() -> pl.DataFrame:
    rows = [
        _taxon(TARGET, "Papilio demoleus", "SPECIES", "Papilionidae", "Papilio"),
        _taxon(CONGENER, "Papilio polytes", "SPECIES", "Papilionidae", "Papilio"),
        _taxon(
            GRAPHIUM_SPECIES,
            "Graphium agamemnon",
            "SPECIES",
            "Papilionidae",
            "Graphium",
        ),
        _taxon(MIMIC, "Danaus chrysippus", "SPECIES", "Nymphalidae", "Danaus"),
        _taxon("gbif:papilio", "Papilio", "GENUS", "Papilionidae", "Papilio"),
        _taxon("gbif:danaus", "Danaus", "GENUS", "Nymphalidae", "Danaus"),
    ]
    rows.extend(
        _taxon(key, genus, "GENUS", "Papilionidae", genus)
        for genus, key in PILOT_GENERA.items()
    )
    return pl.DataFrame(rows)


def _row(
    relationship_type: str,
    scope_type: str,
    scope_id: str,
    *,
    record_id: str,
    evidence_version: str = "relationship-evidence-v1",
) -> dict[str, object]:
    return {
        "subject_accepted_taxon_key": TARGET,
        "object_scope_type": scope_type,
        "object_scope_id": scope_id,
        "relationship_type": relationship_type,
        "source": "reviewed-pilot",
        "source_record_id": record_id,
        "evidence_version": evidence_version,
        "evidence_note": "Reviewed local classifier comparison evidence.",
        "review_status": "reviewed",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-07-13T10:00:00+10:00",
        "enabled": True,
    }


def _source(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": COMPETITOR_RELATIONSHIP_SOURCE_SCHEMA_VERSION,
        "relationships": list(rows),
    }


def _all_relationships() -> list[dict[str, object]]:
    visual = _row(
        "visual_neighbour",
        "species",
        GRAPHIUM_SPECIES,
        record_id="visual",
    )
    visual["prototype_fingerprint"] = SHA_A
    visual["model_fingerprint"] = SHA_B
    return [
        _row("known_mimic", "species", MIMIC, record_id="mimic"),
        _row("close_congener", "species", CONGENER, record_id="congener"),
        _row(
            "historical_false_positive_species",
            "species",
            GRAPHIUM_SPECIES,
            record_id="false-species",
        ),
        _row(
            "historical_false_positive_genus",
            "genus",
            "Graphium",
            record_id="false-genus",
        ),
        _row(
            "taxonomic_neighbour",
            "genus",
            "Graphium",
            record_id="taxonomic",
        ),
        visual,
    ]


def test_compiles_all_relationship_types_with_stable_physical_schema() -> None:
    rows = _all_relationships()
    first = compile_competitor_relationships(_source(*rows), _taxa())
    second = compile_competitor_relationships(_source(*reversed(rows)), _taxa())

    assert first.schema == competitor_relationships_schema()
    assert first.equals(second)
    assert first["schema_version"].unique().to_list() == [
        COMPETITOR_RELATIONSHIPS_SCHEMA_VERSION
    ]
    assert set(first["relationship_type"]) == {
        "known_mimic",
        "close_congener",
        "historical_false_positive_species",
        "historical_false_positive_genus",
        "taxonomic_neighbour",
        "visual_neighbour",
    }
    assert all(
        value.startswith("sha256:") and len(value) == 71
        for value in first["relationship_fingerprint"]
    )
    assert first["reviewed_at"].unique().to_list() == ["2026-07-13T00:00:00Z"]


def test_rejects_unknown_or_unaccepted_taxonomy_identities() -> None:
    unknown_subject = _row(
        "known_mimic", "species", MIMIC, record_id="unknown-subject"
    )
    unknown_subject["subject_accepted_taxon_key"] = "gbif:missing"
    with pytest.raises(ValueError, match="subject is not an accepted"):
        compile_competitor_relationships(_source(unknown_subject), _taxa())

    unknown_genus = _row(
        "historical_false_positive_genus",
        "genus",
        "Missinggenus",
        record_id="unknown-genus",
    )
    with pytest.raises(ValueError, match="not one accepted in-scope genus"):
        compile_competitor_relationships(_source(unknown_genus), _taxa())

    synonym = _taxa().with_columns(
        pl.when(pl.col("accepted_taxon_key") == MIMIC)
        .then(pl.lit("SYNONYM"))
        .otherwise(pl.col("taxonomic_status"))
        .alias("taxonomic_status")
    )
    with pytest.raises(ValueError, match="object is not an accepted"):
        compile_competitor_relationships(
            _source(_row("known_mimic", "species", MIMIC, record_id="synonym")),
            synonym,
        )


@pytest.mark.parametrize(
    ("relationship_type", "scope_type", "scope_id"),
    [
        ("historical_false_positive_genus", "species", GRAPHIUM_SPECIES),
        ("historical_false_positive_species", "genus", "Graphium"),
        ("visual_neighbour", "genus", "Graphium"),
    ],
)
def test_enforces_relationship_object_scope(
    relationship_type: str,
    scope_type: str,
    scope_id: str,
) -> None:
    row = _row(relationship_type, scope_type, scope_id, record_id="bad-scope")
    with pytest.raises(ValueError, match="requires object scope"):
        compile_competitor_relationships(_source(row), _taxa())


def test_enforces_congener_and_visual_evidence_constraints() -> None:
    not_congener = _row(
        "close_congener",
        "species",
        GRAPHIUM_SPECIES,
        record_id="not-congener",
    )
    with pytest.raises(ValueError, match="share the subject genus"):
        compile_competitor_relationships(_source(not_congener), _taxa())

    visual = _row(
        "visual_neighbour", "species", GRAPHIUM_SPECIES, record_id="visual"
    )
    with pytest.raises(ValueError, match="requires prototype and model"):
        compile_competitor_relationships(_source(visual), _taxa())
    visual["prototype_fingerprint"] = "sha256:not-a-digest"
    visual["model_fingerprint"] = SHA_B
    with pytest.raises(ValueError, match="full lowercase sha256"):
        compile_competitor_relationships(_source(visual), _taxa())


def test_enabled_relationship_requires_complete_review_provenance() -> None:
    pending = _row("known_mimic", "species", MIMIC, record_id="pending")
    pending.update(
        {
            "review_status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
        }
    )
    with pytest.raises(ValueError, match="must be reviewed"):
        compile_competitor_relationships(_source(pending), _taxa())

    incomplete = _row("known_mimic", "species", MIMIC, record_id="incomplete")
    incomplete["reviewed_by"] = None
    with pytest.raises(ValueError, match="requires review provenance"):
        compile_competitor_relationships(_source(incomplete), _taxa())

    naive_time = _row("known_mimic", "species", MIMIC, record_id="naive-time")
    naive_time["reviewed_at"] = "2026-07-13T00:00:00"
    with pytest.raises(ValueError, match="include a timezone"):
        compile_competitor_relationships(_source(naive_time), _taxa())


def test_rejects_multiple_enabled_versions_of_one_relationship() -> None:
    first = _row("known_mimic", "species", MIMIC, record_id="mimic-v1")
    second = _row(
        "known_mimic",
        "species",
        MIMIC,
        record_id="mimic-v2",
        evidence_version="relationship-evidence-v2",
    )
    with pytest.raises(ValueError, match="multiple enabled evidence versions"):
        compile_competitor_relationships(_source(first, second), _taxa())

    first["enabled"] = False
    result = compile_competitor_relationships(_source(first, second), _taxa())
    assert result.height == 2
    assert result.filter(pl.col("enabled")).height == 1


def test_writer_refuses_tampered_fingerprint_and_preserves_schema(tmp_path: Path) -> None:
    frame = compile_competitor_relationships(
        _source(_row("known_mimic", "species", MIMIC, record_id="mimic")),
        _taxa(),
    )
    output = write_competitor_relationships(frame, tmp_path)
    assert output == tmp_path / COMPETITOR_RELATIONSHIPS_FILE
    restored = pl.read_parquet(output)
    assert restored.schema == competitor_relationships_schema()
    assert restored.equals(frame)

    tampered = frame.with_columns(pl.lit(SHA_A).alias("relationship_fingerprint"))
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        write_competitor_relationships(tampered, tmp_path / "tampered.parquet")


def test_papilio_pilot_seed_is_reviewed_and_registry_reconciled() -> None:
    source = (
        Path(__file__).parents[1]
        / "config"
        / "candidates"
        / "papilio_demoleus_competitor_relationships.json"
    )
    payload = load_competitor_relationship_source(source)
    result = compile_competitor_relationships(payload, _taxa())

    assert result.height == 5
    assert set(result["object_scope_id"]) == set(PILOT_GENERA)
    assert result["subject_accepted_taxon_key"].unique().to_list() == [TARGET]
    assert result["relationship_type"].unique().to_list() == [
        "historical_false_positive_genus"
    ]
    assert result["enabled"].all()
    assert result["review_status"].unique().to_list() == ["reviewed"]


def test_compiled_genus_relationships_expand_in_regional_candidate_union() -> None:
    graphium_relationship = _row(
        "historical_false_positive_genus",
        "genus",
        "Graphium",
        record_id="false-genus",
    )
    relationships = compile_competitor_relationships(
        _source(graphium_relationship), _taxa()
    )
    clusters = pl.DataFrame(
        [
            {
                "geo_cluster_id": "cluster-a",
                "target_accepted_taxon_key": TARGET,
                "candidate_distribution_only": True,
                "cluster_configuration_hash": SHA_A,
            }
        ]
    )
    result = build_regional_candidate_species(
        target_accepted_taxon_key=TARGET,
        geo_clusters=clusters,
        regional_occurrence=pl.DataFrame(schema=regional_taxon_occurrence_schema()),
        taxa=_taxa(),
        registry_version="registry-test-v1",
        competitor_relationships=relationships,
        config=RegionalCandidateConfig(minimum_local_same_family_candidates=0),
    )

    graphium = result.filter(
        pl.col("candidate_accepted_taxon_key") == GRAPHIUM_SPECIES
    ).row(0, named=True)
    assert graphium["historical_false_positive"] is True
    assert graphium["candidate_reason"] == ["historical_false_positive"]
    assert (
        "relationships:relationship-evidence-v1" in graphium["source_versions"]
    )


def test_source_loader_rejects_unknown_root_and_row_fields() -> None:
    source = _source(_row("known_mimic", "species", MIMIC, record_id="mimic"))
    source["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        load_competitor_relationship_source(source)

    row = _row("known_mimic", "species", MIMIC, record_id="mimic")
    row["typo"] = "value"
    with pytest.raises(ValueError, match="unknown fields"):
        compile_competitor_relationships(_source(row), _taxa())

    missing = deepcopy(row)
    missing.pop("typo")
    missing.pop("evidence_note")
    with pytest.raises(ValueError, match="missing fields"):
        compile_competitor_relationships(_source(missing), _taxa())
