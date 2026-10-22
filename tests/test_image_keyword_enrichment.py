"""Synthetic coordinator tests: no provider clients, requests or real databases."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.registry.enrichment import SpeciesContext
from biominer.registry.image_keyword_enrichment import SourceBundle, WorkState, enrich_inventory


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _collection(tmp_path: Path, *, conflicts: bool = False) -> Path:
    collection = tmp_path / "collection"
    collection.mkdir()
    rows = [
        {"species_ordinal": 1, "accepted_taxon_key": "gbif:3", "species_key": "3",
         "scientific_name": "Papilio demoleus", "family_key": "9417", "family": "Papilionidae",
         "genus": "Papilio", "image_rows": 3, "name_variant_count": 1, "query_eligible": not conflicts,
         "binding_status": "conflicting_source_names_or_families" if conflicts else "source_bound"},
        {"species_ordinal": 2, "accepted_taxon_key": "gbif:20", "species_key": "20",
         "scientific_name": "Danaus plexippus", "family_key": "7017", "family": "Nymphalidae",
         "genus": "Danaus", "image_rows": 2, "name_variant_count": 1, "query_eligible": True,
         "binding_status": "source_bound"},
    ]
    pq.write_table(pa.Table.from_pylist(rows), collection / "species.parquet")
    pq.write_table(pa.Table.from_pylist([
        {"accepted_taxon_key": row["accepted_taxon_key"], "scientific_name": row["scientific_name"]}
        for row in rows
    ]), collection / "source_name_variants.parquet")
    artifacts = [{"path": name, "sha256": _sha(collection / name)}
                 for name in ("species.parquet", "source_name_variants.parquet")]
    (collection / "manifest.json").write_text(json.dumps({
        "schema_version": "biominer-image-species-collection/v1", "species_count": len(rows),
        "semantic_fingerprint": "sha256:" + "a" * 64, "artifacts": artifacts,
    }))
    return collection


@pytest.fixture(autouse=True)
def _stable_code_inputs(monkeypatch):
    # Exercise resume configuration without coupling fixture work to concurrent
    # source edits by other implementation owners.
    monkeypatch.setattr("biominer.registry.image_keyword_enrichment.code_inputs", lambda: {"fixture": "a" * 64})


def _assertion(context, *, record="record-1", source="GBIF", name=None, taxon=None):
    return {"accepted_taxon_key": taxon or context.accepted_taxon_key,
            "display_name": name or context.accepted_scientific_name, "language": "la",
            "name_class": "accepted_scientific", "source": source, "source_record_id": record,
            "source_taxon_id": context.accepted_taxon_key.removeprefix("gbif:"),
            "lineage_check": "exact_species_identity", "trust_tier": "T1",
            "confidence": "high", "precision_tier": "high", "enabled": True,
            "review_state": "accepted", "disabled_reason": ""}


def _latest(output: Path):
    latest = json.loads((output / "latest.json").read_text())
    manifest_path = output / latest["manifest"]
    assert _sha(manifest_path) == latest["sha256"]
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    for item in manifest["artifacts"]:
        path = manifest_path.parent / item["path"]
        assert _sha(path) == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        table = pq.read_table(path)
        assert table.num_rows == item["rows"]
        artifacts[path.stem] = table.to_pylist()
    return manifest, artifacts


def test_actual_species_and_source_denominators_numeric_order_and_export_provenance(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    calls, events = [], []

    def get_names(source):
        def get(context):
            calls.append((context.accepted_taxon_key, source))
            return {"name_assertions": [_assertion(context, source=source)],
                    "source_snapshots": [{"source": source, "source_response_hash": "sha256:" + "b" * 64,
                                          "source_path": "https://example.org/taxon", "retrieved_at": "2026-01-01T00:00:00+00:00"}]}
        return get

    result = enrich_inventory(collection, output, sources=("gbif_vernacular", "col"),
                              handlers={"gbif_vernacular": get_names("GBIF"), "col": get_names("CoL")},
                              event_sink=events.append, locales=("en", "fr"))
    assert result["state"] == "complete"
    assert calls == [("gbif:3", "GBIF"), ("gbif:3", "CoL"), ("gbif:20", "GBIF"), ("gbif:20", "CoL")]
    started = [e for e in events if e.get("status") == "started"]
    assert [(e["species_ordinal"], e["species_total"], e["source_ordinal"], e["source_total"]) for e in started] == [
        (1, 2, 1, 2), (1, 2, 2, 2), (2, 2, 1, 2), (2, 2, 2, 2)]
    manifest, artifacts = _latest(output)
    assert manifest["species_total"] == manifest["source_total"] == 2
    assert manifest["source_status_counts"] == {"complete": 4}
    assert manifest["collection_manifest"]["sha256"] == _sha(collection / "manifest.json")
    assert manifest["runtime_registered"] is manifest["production_migration_performed"] is False
    names = artifacts["names"]
    assert len(names) == 4
    assert all(row["source_response_hash"].startswith("sha256:") and row["retrieved_at"] for row in names)
    assert all(row["query_eligible"] for row in names)
    assert {row["source_key"] for row in names} == {"gbif_vernacular", "col"}
    progress = (output / "progress.log").read_text()
    assert "1/2 species | 1/2 sources" in progress and "2/2 species | 2/2 sources" in progress


def test_same_name_from_distinct_provider_records_and_sources_retains_all_provenance(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    enrich_inventory(collection, output, sources=("gbif_vernacular", "col"), handlers={
        "gbif_vernacular": lambda context: {"name_assertions": [_assertion(context, record="first"), _assertion(context, record="second")]},
        "col": lambda context: {"name_assertions": [_assertion(context, record="first", source="CoL")]},
    }, locales=("en",))
    _, artifacts = _latest(output)
    assert len(artifacts["names"]) == 6
    first_species = [row for row in artifacts["names"] if row["accepted_taxon_key"] == "gbif:3"]
    assert len({row["evidence_id"] for row in first_species}) == 3
    assert len({row["display_name"] for row in first_species}) == 1
    assert {(row["source_key"], row["source_record_id"]) for row in first_species} == {
        ("gbif_vernacular", "first"), ("gbif_vernacular", "second"), ("col", "first")}


@pytest.mark.parametrize("has_crosswalk", [False, True])
def test_wikidata_common_names_need_recorded_same_taxon_crosswalk_proof(tmp_path, has_crosswalk):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    def names(context):
        assertion = _assertion(context, source="Wikidata", name="Lime swallowtail")
        assertion.update(source_taxon_id="Q123", language="en", name_class="vernacular",
            trust_tier="T3", lineage_check="legacy_taxon+accepted_taxon_key+family+genus")
        links = [{"accepted_taxon_key": context.accepted_taxon_key, "source": "Wikidata",
            "source_taxon_id": "Q123", "match_confidence": "high",
            "match_method": "P225+P846+GBIF_COL_XR"}] if has_crosswalk else []
        return {"name_assertions": [assertion], "external_links": links}
    enrich_inventory(collection, output, sources=("wikidata",), handlers={"wikidata": names}, locales=("en",))
    _, artifacts = _latest(output)
    assert all(row["query_eligible"] == has_crosswalk for row in artifacts["names"])
    for row in artifacts["names"]:
        assert bool(json.loads(row["source_assertion_json"]).get("source_binding_confident")) == has_crosswalk


def test_error_unavailable_not_applicable_and_zero_results_remain_distinct(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"

    def failed(context):
        raise RuntimeError("fixture provider failed")

    result = enrich_inventory(collection, output, sources=("gbif_vernacular", "col", "wikidata", "inaturalist"), handlers={
        "gbif_vernacular": failed,
        "col": lambda context: {"work_status": "unavailable", "reason": "snapshot_missing"},
        "wikidata": lambda context: {"work_status": "not_applicable", "reason": "outside_source_scope"},
        "inaturalist": lambda context: {"name_assertions": []},
    }, locales=("en",))
    assert result["state"] == "completed_with_gaps"
    manifest, artifacts = _latest(output)
    assert manifest["source_status_counts"] == {"error": 2, "unavailable": 2, "not_applicable": 2, "complete": 2}
    assert artifacts["names"] == []
    jobs = {row["source"]: row for row in artifacts["jobs"] if row["taxon"] == "gbif:3"}
    assert jobs["gbif_vernacular"]["error_class"] == "RuntimeError"
    assert jobs["col"]["status"] == "unavailable"
    assert jobs["wikidata"]["status"] == "not_applicable"
    assert jobs["inaturalist"]["status"] == "complete" and jobs["inaturalist"]["name_count"] == 0


def test_interrupt_preserves_per_source_checkpoint_and_resume_skips_completed_calls(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    calls = []
    interrupt = [True]

    def first(context):
        calls.append((context.accepted_taxon_key, "first"))
        return {"name_assertions": [_assertion(context)]}

    def second(context):
        calls.append((context.accepted_taxon_key, "second"))
        if interrupt[0]:
            interrupt[0] = False
            raise KeyboardInterrupt("synthetic interruption")
        return {"name_assertions": []}

    handlers = {"gbif_vernacular": first, "col": second}
    with pytest.raises(KeyboardInterrupt):
        enrich_inventory(collection, output, sources=("gbif_vernacular", "col"), handlers=handlers,
                         checkpoint_every=250, locales=("en",))
    manifest, artifacts = _latest(output)
    assert manifest["state"] == "interrupted"
    assert [(job["taxon"], job["source"], job["status"]) for job in artifacts["jobs"]] == [("gbif:3", "gbif_vernacular", "complete")]
    assert len(artifacts["names"]) == 1
    result = enrich_inventory(collection, output, sources=("gbif_vernacular", "col"), handlers=handlers, locales=("en",))
    assert result["state"] == "complete"
    assert calls.count(("gbif:3", "first")) == 1
    assert calls.count(("gbif:3", "second")) == 2
    assert calls.count(("gbif:20", "first")) == calls.count(("gbif:20", "second")) == 1
    assert _latest(output)[0]["source_status_counts"] == {"complete": 4}


def test_resume_retries_error_but_not_complete_unavailable_or_nonapplicable(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    calls = dict.fromkeys(("gbif_vernacular", "col", "wikidata", "inaturalist"), 0)
    failing = [True]

    def handler(source):
        def run(context):
            calls[source] += 1
            if source == "gbif_vernacular" and failing[0]:
                raise ValueError("retry fixture")
            if source == "col": return {"work_status": "unavailable"}
            if source == "wikidata": return {"work_status": "not_applicable"}
            return {"name_assertions": []}
        return run

    handlers = {key: handler(key) for key in calls}
    arguments = dict(sources=tuple(calls), handlers=handlers, locales=("en",))
    enrich_inventory(collection, output, **arguments)
    failing[0] = False
    enrich_inventory(collection, output, **arguments)
    assert calls == {"gbif_vernacular": 4, "col": 2, "wikidata": 2, "inaturalist": 2}
    manifest, artifacts = _latest(output)
    assert manifest["source_status_counts"] == {"complete": 4, "unavailable": 2, "not_applicable": 2}
    assert all(row["attempts"] == 2 for row in artifacts["jobs"] if row["source"] == "gbif_vernacular")


@pytest.mark.parametrize("change", ["source_order", "locales", "code_inputs"])
def test_resume_rejects_changed_source_locale_or_code_configuration(tmp_path, monkeypatch, change):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    handlers = {"gbif_vernacular": lambda context: {}, "col": lambda context: {}}
    enrich_inventory(collection, output, sources=("gbif_vernacular", "col"), handlers=handlers, locales=("en",))
    arguments = dict(sources=("gbif_vernacular", "col"), handlers=handlers, locales=("en",))
    if change == "source_order": arguments["sources"] = ("col", "gbif_vernacular")
    elif change == "locales": arguments["locales"] = ("en", "fr")
    else: monkeypatch.setattr("biominer.registry.image_keyword_enrichment.code_inputs", lambda: {"fixture": "b" * 64})
    with pytest.raises(ValueError, match="configuration changed"):
        enrich_inventory(collection, output, **arguments)


def test_rejects_assertion_for_different_species_and_rolls_back_whole_source_result(tmp_path):
    collection = _collection(tmp_path)
    output = tmp_path / "enrichment"
    with pytest.raises(ValueError, match="different species"):
        enrich_inventory(collection, output, sources=("gbif_vernacular",), handlers={
            "gbif_vernacular": lambda context: {"name_assertions": [
                _assertion(context), _assertion(context, record="foreign", taxon="gbif:999")]},
        }, locales=("en",))
    manifest, artifacts = _latest(output)
    assert manifest["state"] == "interrupted"
    assert artifacts["names"] == artifacts["jobs"] == []


def test_input_conflicts_skip_provider_calls_but_keep_actual_species_denominator(tmp_path):
    collection = _collection(tmp_path, conflicts=True)
    calls = []
    result = enrich_inventory(collection, tmp_path / "out", sources=("gbif_vernacular",), handlers={
        "gbif_vernacular": lambda context: calls.append(context.accepted_taxon_key) or {},
    }, locales=("en",))
    assert result["species_total"] == 2
    assert result["state"] == "completed_with_gaps"
    assert calls == ["gbif:20"]
    assert _latest(tmp_path / "out")[0]["source_status_counts"] == {"input_conflict": 1, "complete": 1}


def test_tampered_collection_fails_before_handlers_or_work_state(tmp_path):
    collection = _collection(tmp_path)
    (collection / "species.parquet").write_bytes(b"tampered")
    calls = []
    with pytest.raises(ValueError, match="collection artifact changed"):
        enrich_inventory(collection, tmp_path / "out", sources=("gbif_vernacular",), handlers={
            "gbif_vernacular": lambda context: calls.append(context) or {},
        }, locales=("en",))
    assert not calls
    assert not (tmp_path / "out/work.sqlite3").exists()


def _context(row):
    return SpeciesContext(row["accepted_taxon_key"], row["scientific_name"],
                          row["family_key"], row["family"], "", row["genus"], ())


def test_source_bundle_bulk_reuse_keeps_original_physical_request_receipts_for_each_species(tmp_path):
    collection = _collection(tmp_path)
    rows = pq.read_table(collection / "species.parquet").to_pylist()
    output = tmp_path / "out"
    output.mkdir()
    state = WorkState(output / "work.sqlite3", {"fixture": "bulk-receipt-reuse"})
    receipt = {"host": "example.org", "path": "/bulk-taxonomy", "response_sha256": "c" * 64,
               "request_sha256": "d" * 64, "retrieved_at": "2026-01-01T00:00:00+00:00",
               "attempt_id": 7, "cache_hit": False}

    class FakeHTTP:
        def __init__(self): self.job_receipts = []
        def reset_job_receipts(self): self.job_receipts.clear()

    http = FakeHTTP()
    calls = []

    class FakeBulkClient:
        def enrich_registry(self, *, taxa_rows, name_rows):
            calls.append((taxa_rows, name_rows))
            http.job_receipts.append(receipt)
            return {"name_assertions": [_assertion(_context(row), source="Open Tree") for row in rows],
                    "source_snapshots": [{"source": "Open Tree", "source_path": "https://example.org/bulk-taxonomy",
                                          "source_response_hash": "sha256:" + "c" * 64}]}

    # Build only the minimal bundle collaborators for this public run branch;
    # the normal constructor would instantiate unrelated live provider adapters.
    bundle = SourceBundle.__new__(SourceBundle)
    bundle.http, bundle.state, bundle.rows = http, state, rows
    bundle.locales, bundle.qids = ("en",), {}
    bundle.clients = {"open_tree": FakeBulkClient()}
    bundle.bulk, bundle.bulk_receipts = {}, {}
    bundle.taxa = [{"accepted_taxon_key": row["accepted_taxon_key"], "scientific_name": row["scientific_name"],
                    "family": row["family"], "genus": row["genus"], "rank": "SPECIES"} for row in rows]
    events = []
    bundle.emit = lambda **event: events.append(event)
    try:
        for row in rows:
            http.reset_job_receipts()
            result = bundle.run("open_tree", _context(row), row, 1)
            assert len(result["name_assertions"]) == 1
            assert result["name_assertions"][0]["accepted_taxon_key"] == row["accepted_taxon_key"]
            assert http.job_receipts == [receipt]
            state.commit(row, "open_tree", 1, result, receipts=http.job_receipts)
        assert len(calls) == 1
        assert len(calls[0][0]) == 2
        assert [event["status"] for event in events] == ["started", "complete"]
        state.export(output, {"state": "complete"})
    finally:
        state.con.close()
    _, artifacts = _latest(output)
    assert len(artifacts["names"]) == len(artifacts["jobs"]) == 2
    assert all(json.loads(job["receipts"]) == [receipt] for job in artifacts["jobs"])
    assert {name["source_response_hash"] for name in artifacts["names"]} == {
        canonical_semantic_fingerprint([receipt["response_sha256"]])}
    assert all(name["source_url"] == "https://example.org/bulk-taxonomy" for name in artifacts["names"])


@pytest.mark.parametrize("empty_result", [True, False])
def test_translation_unit_receipts_survive_reopen_and_export_even_without_names(tmp_path, empty_result):
    collection = _collection(tmp_path)
    row = pq.read_table(collection / "species.parquet").to_pylist()[0]
    output = tmp_path / "out"
    output.mkdir()
    config = {"fixture": "translation-unit-receipts"}
    unit = canonical_semantic_fingerprint([row["accepted_taxon_key"], "Lime Swallowtail", "fr"])
    receipt = {"host": "mymemory.translated.net", "path": "/get", "response_sha256": "e" * 64,
               "request_sha256": "f" * 64, "retrieved_at": "2026-01-01T00:00:00+00:00",
               "attempt_id": 11, "cache_hit": False}
    translated = _assertion(_context(row), source="MyMemory", name="Voilier échiquier")
    translated.update(language="fr", name_class="generated_translation", trust_tier="T5",
                      enabled=False, review_state="candidate", disabled_reason="dictionary_translation_requires_review")
    result = {"name_assertions": [] if empty_result else [translated]}
    state = WorkState(output / "work.sqlite3", config)
    try:
        assert not state.translation_done(unit)
        state.translation_commit(unit, row, 17, result, [receipt])
        assert state.translation_done(unit)
    finally:
        state.con.close()
    # A new coordinator must be able to skip the already committed physical
    # dictionary request, including a successful request that returned no names.
    state = WorkState(output / "work.sqlite3", config)
    try:
        assert state.translation_done(unit)
        state.export(output, {"state": "complete"})
    finally:
        state.con.close()
    manifest, artifacts = _latest(output)
    assert "translation_units" in artifacts
    assert len(manifest["artifacts"]) == 5
    assert len(artifacts["translation_units"]) == 1
    saved = artifacts["translation_units"][0]
    assert saved["id"] == unit
    assert json.loads(saved["receipts"]) == [receipt]
    assert saved["name_count"] == (0 if empty_result else 1)
    assert saved["finished_at"]
    if empty_result:
        assert artifacts["names"] == []
    else:
        name = artifacts["names"][0]
        assert not name["enabled"] and not name["query_eligible"]
        assert name["source_response_hash"] == canonical_semantic_fingerprint([receipt["response_sha256"]])
        assert name["source_url"] == "https://mymemory.translated.net/get"
