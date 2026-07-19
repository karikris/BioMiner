from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from biominer.flickr_fetch.metadata_poller import MetadataPollState, poll_once
from biominer.flickr_fetch.query_planner import FlickrQuery, plan_pages_from_count, plan_queries_from_count
from biominer.run.taxon_scope import resolve_species_context_from_registry
from biominer.species.context import CommonName, SpeciesContext


def _write_registry_fixture(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family_key": "gbif:10",
                "family": "Papilionidae",
                "genus_key": "gbif:90",
                "genus": "Papilio",
                "species_key": "gbif:100",
                "species": "Papilio demoleus",
            }
        ]
    ).write_parquet(base / "taxa.parquet")
    pl.DataFrame(
        [
            {
                "name_id": "name:accepted",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Papilio demoleus",
                "display_name": "Papilio demoleus",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "accepted_scientific",
                "source": "gbif",
                "source_record_id": "gbif:100",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
            },
            {
                "name_id": "name:synonym",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "Papilio erithonius",
                "display_name": "Papilio erithonius",
                "language": "la",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "scientific_synonym",
                "source": "gbif",
                "source_record_id": "gbif:syn:1",
                "trust_tier": "T1",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
            },
            {
                "name_id": "name:common",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "lime butterfly",
                "display_name": "lime butterfly",
                "language": "en",
                "script": "Latn",
                "region": "AU",
                "bbox": "112.92,-43.74,153.64,-10.05",
                "name_class": "vernacular",
                "source": "gbif",
                "source_record_id": "gbif:vernacular:1",
                "trust_tier": "T2",
                "precision_tier": "high",
                "confidence": "high",
                "enabled": True,
                "disabled_reason": "",
                "query_eligible": True,
                "query_disabled_reason": "",
                "species_specificity_score": 0.95,
            },
            {
                "name_id": "name:weak",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "verbatim_name": "lime",
                "display_name": "lime",
                "language": "en",
                "script": "Latn",
                "region": "",
                "bbox": "",
                "name_class": "vernacular_alias",
                "source": "gbif",
                "source_record_id": "gbif:vernacular:weak",
                "trust_tier": "T2",
                "precision_tier": "low",
                "confidence": "low",
                "enabled": True,
                "disabled_reason": "",
                "query_eligible": False,
                "query_disabled_reason": "generic_single_token",
                "species_specificity_score": 0.25,
            },
        ]
    ).write_parquet(base / "names.parquet")
    pl.DataFrame(
        [
            {
                "evidence_id": "evidence:1",
                "name_id": "name:accepted",
                "registry_version": "registry-v1",
                "accepted_taxon_key": "gbif:100",
                "source": "gbif",
                "source_record_id": "gbif:100",
                "source_response_hash": "sha256:fixture",
                "retrieved_at": "2026-01-01T00:00:00+00:00",
                "licence": "",
                "trust_tier": "T1",
                "review_state": "accepted",
            }
        ]
    ).write_parquet(base / "name_evidence.parquet")
    pl.DataFrame(
        [{"source": "gbif", "source_version": "fixture", "retrieved_at": "2026-01-01T00:00:00+00:00"}]
    ).write_parquet(base / "source_snapshots.parquet")
    pl.DataFrame([]).write_parquet(base / "flickr_query_definitions.parquet")
    (base / "manifest.json").write_text(json.dumps({"registry_version": "registry-v1"}), encoding="utf-8")


def test_species_context_resolves_papilio_fixture_from_registry_names(tmp_path) -> None:
    registry = tmp_path / "registry"
    _write_registry_fixture(registry)

    context = resolve_species_context_from_registry(scientific_name="Papilio demoleus", registry_dir=registry)

    assert context.accepted_taxon_key == "gbif:100"
    assert context.synonyms == ("Papilio erithonius",)
    assert [name.name for name in context.common_names] == ["lime butterfly"]
    assert [term.term for term in context.search_terms] == ["Papilio demoleus", "Papilio erithonius", "lime butterfly"]
    assert "lime" not in context.target_terms()


def test_query_planner_preserves_registry_provenance_through_pages_and_accessible_window() -> None:
    probe = FlickrQuery(
        term="Danaus plexippus",
        language="la",
        search_field="text",
        lane="count_probe",
        has_geo=0,
        query_definition_id="q-danaus-text",
        registry_version="registry-v1",
        accepted_taxon_key="gbif:5133240",
        accepted_scientific_name="Danaus plexippus",
        family_key="gbif:7017",
        genus_key="gbif:5133234",
        species_key="gbif:5133240",
    )

    page = plan_pages_from_count(probe, total=1)[0]
    capped = plan_queries_from_count(probe, total=4001)[0]

    for query in (page, capped):
        assert query.registry_version == "registry-v1"
        assert query.query_definition_id == "q-danaus-text"
        assert query.accepted_taxon_key == "gbif:5133240"
        assert query.accepted_scientific_name == "Danaus plexippus"
        assert query.family_key == "gbif:7017"
        assert query.genus_key == "gbif:5133234"
        assert query.species_key == "gbif:5133240"


def test_no_compact_shard_writes_canonical_folded_evidence_rows(tmp_path) -> None:
    state = MetadataPollState(tmp_path / "poller.sqlite")
    state.enqueue_work_item(FlickrQuery(term="Danaus plexippus", language="la", search_field="text", lane="normal_page", page=1, per_page=250))
    state.enqueue_work_item(FlickrQuery(term="monarch butterfly", language="en", search_field="tags", lane="normal_page", page=1, per_page=250))

    poll_once(
        state_db=state.path,
        raw_root=tmp_path / "raw",
        evidence_output=tmp_path / "evidence" / "poll.parquet",
        max_api_calls=2,
        fetch_metadata=lambda item: {
            "photos": {
                "total": "1",
                "pages": "1",
                "page": "1",
                "perpage": "250",
                "photo": [{"id": "m1", "title": "monarch", "url_l": "https://live.staticflickr.com/m1.jpg"}],
            }
        },
        run_id="run-1",
        worker_id="worker-1",
        storage_prefix=tmp_path / "staging",
        compact_after_run=False,
    )

    shards = sorted((tmp_path / "staging" / "evidence" / "stage=poll_once" / "run_id=run-1" / "worker=worker-1").glob("*.parquet"))
    frame = pl.read_parquet(shards)
    row = frame.to_dicts()[0]

    assert frame.height == 1
    assert row["text_search_terms"] == ["Danaus plexippus"]
    assert row["tag_search_terms"] == ["monarch butterfly"]
    assert row["all_query_labels"] == ["text:Danaus plexippus", "tags:monarch butterfly"]
    assert row["query_hit_count"] == 2


def test_species_context_target_terms_skip_query_ineligible_common_names() -> None:
    context = SpeciesContext(
        scientific_name="Papilio demoleus",
        accepted_taxon_key="gbif:100",
        canonical_name="Papilio demoleus",
        family="Papilionidae",
        genus="Papilio",
        family_key="gbif:10",
        genus_key="gbif:90",
        species_key="gbif:100",
        registry_version="registry-v1",
        common_names=(
            CommonName(name="lime butterfly", language="en", review_state="accepted"),
            CommonName(name="lime", language="en", review_state="query_ineligible"),
        ),
    )

    assert context.target_terms() == ("Papilio demoleus", "lime butterfly")


def test_production_modules_do_not_hardcode_papilio_species() -> None:
    root = Path("src/biominer")
    forbidden = (
        "Papilio demoleus",
        "PAPILIO_DEMOLEUS_ANCHOR_TERMS",
        "PAPILIO_DEMOLEUS_REGION_BBOXES",
        "load_papilio_demoleus_terms_from_json",
        "build_papilio_demoleus_count_probes_from_json",
        "papilio_demoleus_known_region_for_coordinate",
        "outside_known_papilio_demoleus_regions",
        "TARGET_SPECIES",
        "TARGET_TERMS",
    )
    offenders = []
    for path in root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        offenders.extend(f"{path}: {token}" for token in forbidden if token in text)

    assert offenders == []
