from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from biominer.registry.image_species import RAW_ARCHIVE_SHA256, collect_image_species


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source(tmp_path: Path, rows: list[tuple[object, ...]]) -> tuple[Path, Path]:
    database = tmp_path / "images.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute("""CREATE TABLE gbif_media (
            speciesKey VARCHAR, species VARCHAR, familyKey VARCHAR,
            family VARCHAR, genus VARCHAR, media_identifier VARCHAR,
            media_type VARCHAR, media_format VARCHAR, taxonRank VARCHAR)""")
        if rows:
            connection.executemany("INSERT INTO gbif_media VALUES (?,?,?,?,?,?,?,?,?)", rows)
    raw = tmp_path / "raw.json"
    members = []
    artifacts = {}
    for role in ("occurrence", "multimedia", "verbatim"):
        path = tmp_path / f"{role}.parquet"
        path.write_bytes(role.encode())
        members.append({"member": role + ".txt", "path": path.name,
                        "physical_bytes": path.stat().st_size, "column_count": 2,
                        "row_count": 10})
        if role != "verbatim":
            artifacts[role] = {"path": str(path), "physical_bytes": path.stat().st_size,
                               "column_count": 2, "physical_sha256": _sha(path)}
    raw.write_text(json.dumps({"schema_version": "biominer-dwca-members-parquet-v1",
                               "source": {"archive_sha256": RAW_ARCHIVE_SHA256},
                               "outputs": members}))
    joined = tmp_path / "joined.json"
    joined.write_text(json.dumps({"schema_version": "biominer-gbif-occurrence-multimedia-join/v1",
        "input": {"artifacts": artifacts, "source_manifest": {"path": raw.name, "physical_sha256": _sha(raw)}},
        "counts": {"occurrence_rows": 10, "multimedia_rows": 10},
        "output": {"physical_sha256": "a" * 64}}))
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"schema_version": "biominer-gbif-media-url-database/v1",
        "database": {"path": database.name, "physical_sha256": _sha(database),
                     "physical_bytes": database.stat().st_size, "row_count": len(rows),
                     "table": "gbif_media", "image_url_column": "media_identifier"},
        "input": {"physical_sha256": "a" * 64,
                  "source_manifest": {"path": joined.name, "physical_sha256": _sha(joined)}}}))
    return database, manifest


def _row(key="20", name="Danaus plexippus", **values):
    record = dict(speciesKey=key, species=name, familyKey="7017", family="Nymphalidae",
                  genus="Danaus", media_identifier="https://example.org/image.jpg",
                  media_type="StillImage", media_format="image/jpeg", taxonRank="SPECIES")
    record.update(values)
    return tuple(record.values())


def test_extracts_actual_image_species_count_and_numeric_order_without_rank_loss(tmp_path):
    database, manifest = _source(tmp_path, [
        _row("20"), _row("3", "Papilio demoleus", familyKey="9417", family="Papilionidae", genus="Papilio", taxonRank="SUBSPECIES"),
        _row("20", taxonRank="SUBSPECIES"),
        _row("40", "Not image", media_type="Sound", media_format="audio/wav"),
        _row("50", "Missing URL", media_identifier=None),
        _row(None, "Missing key"), _row("-2", "Invalid key"),
        _row("51", "Outside scope", familyKey="999"),
        _row("52", "Blank name", species=" "),
    ])
    events = []
    result = collect_image_species(database, manifest, tmp_path / "collection", logger=events.append)
    rows = pq.read_table(tmp_path / "collection/species.parquet").to_pylist()
    assert result["species_count"] == 2
    assert result["eligible_image_rows"] == 3
    assert [(r["species_ordinal"], r["species_key"], r["image_rows"]) for r in rows] == [(1, "3", 1), (2, "20", 2)]
    assert all(r["query_eligible"] for r in rows)
    assert result["raw_archive_sha256"] == RAW_ARCHIVE_SHA256
    assert len(result["identity"]["manifest_chain"]) == 3
    raw_bindings = result["identity"]["manifest_chain"][1]["raw_member_bindings"]
    assert len(raw_bindings) == 2
    assert all(not binding["fresh_member_sha256_verified"] for binding in raw_bindings)
    assert any(event["event"] == "input_hash_progress" for event in events)
    diagnostics = json.loads((tmp_path / "collection/exclusions.json").read_text())
    assert diagnostics["exclusion_counts"] == {
        "not_explicit_image": 1, "missing_or_invalid_http_url": 1,
        "missing_or_invalid_species_key": 2, "outside_butterfly_scope": 1, "missing_species_name": 1,
    }


def test_retains_all_source_variants_and_fails_closed_for_name_family_conflicts(tmp_path):
    database, manifest = _source(tmp_path, [
        _row(), _row(name="Danaus alternate"),
        _row(familyKey="9417", family="Papilionidae"),
        _row("30", "Danaus other", family="Wrong family name"),
    ])
    result = collect_image_species(database, manifest, tmp_path / "out")
    species = pq.read_table(tmp_path / "out/species.parquet").to_pylist()
    variants = pq.read_table(tmp_path / "out/source_name_variants.parquet").to_pylist()
    assert result["species_count"] == 2
    assert result["query_eligible_species_count"] == 0
    assert len(variants) == 4
    assert species[0]["name_variant_count"] == 2
    assert species[0]["image_rows"] == 3
    assert all(not row["query_eligible"] for row in variants)
    assert all(row["binding_status"] == "conflicting_source_names_or_families" for row in species)


def test_accepts_explicit_image_mime_but_rejects_non_http_and_unknown_media(tmp_path):
    database, manifest = _source(tmp_path, [
        _row(media_type=None), _row("21", "Danaus next", media_identifier="ftp://example.org/image.jpg"),
        _row("22", "Danaus third", media_type=None, media_format=None),
    ])
    result = collect_image_species(database, manifest, tmp_path / "out")
    assert result["species_count"] == 1
    assert result["eligible_image_rows"] == 1


def test_resume_reuses_verified_complete_outputs(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    output = tmp_path / "out"
    first = collect_image_species(database, manifest, output)
    before = (output / "manifest.json").read_bytes()
    events = []
    second = collect_image_species(database, manifest, output, logger=events.append)
    assert first == second
    assert (output / "manifest.json").read_bytes() == before
    assert events[-1]["event"] == "species_collection_reused"


def test_resume_rejects_changed_database_or_output(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    output = tmp_path / "out"
    collect_image_species(database, manifest, output)
    (output / "species.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact checksum"):
        collect_image_species(database, manifest, output)
    with database.open("r+b") as stream:
        stream.seek(100)
        stream.write(b"x")
    with pytest.raises(ValueError, match="database checksum"):
        collect_image_species(database, manifest, tmp_path / "other")


def test_rejects_broken_transitive_manifest_checksum_before_database_scan(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    raw = tmp_path / "raw.json"
    raw.write_text(raw.read_text() + " ")
    events = []
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        collect_image_species(database, manifest, tmp_path / "out", logger=events.append)
    assert not events


def test_rejects_wrong_raw_archive_even_if_manifest_links_are_rebound(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    raw = tmp_path / "raw.json"
    data = json.loads(raw.read_text())
    data["source"]["archive_sha256"] = "b" * 64
    raw.write_text(json.dumps(data))
    joined = tmp_path / "joined.json"
    data = json.loads(joined.read_text())
    data["input"]["source_manifest"]["physical_sha256"] = _sha(raw)
    joined.write_text(json.dumps(data))
    data = json.loads(manifest.read_text())
    data["input"]["source_manifest"]["physical_sha256"] = _sha(joined)
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="retained raw GBIF archive"):
        collect_image_species(database, manifest, tmp_path / "out")


def test_empty_cohort_has_typed_outputs_and_zero_denominator(tmp_path):
    database, manifest = _source(tmp_path, [_row(media_type="Sound", media_format="audio/wav")])
    result = collect_image_species(database, manifest, tmp_path / "out")
    assert result["species_count"] == 0
    assert pq.read_table(tmp_path / "out/species.parquet").num_rows == 0
    assert str(pq.read_schema(tmp_path / "out/species.parquet").field("species_ordinal").type) == "int64"


def test_refuses_incomplete_existing_output(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(FileNotFoundError):
        collect_image_species(database, manifest, output)


def test_rejects_parent_artifact_link_mismatch(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    data = json.loads(manifest.read_text())
    data["input"]["physical_sha256"] = "c" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="source manifest output"):
        collect_image_species(database, manifest, tmp_path / "out")


def _rebind_join(tmp_path, mutate):
    joined = tmp_path / "joined.json"
    data = json.loads(joined.read_text())
    mutate(data)
    joined.write_text(json.dumps(data))
    manifest = tmp_path / "input.json"
    data = json.loads(manifest.read_text())
    data["input"]["source_manifest"]["physical_sha256"] = _sha(joined)
    manifest.write_text(json.dumps(data))


@pytest.mark.parametrize("missing", ["parent_sha", "child_output", "child_sha"])
def test_rejects_missing_required_single_artifact_hash_bindings(tmp_path, missing):
    database, manifest = _source(tmp_path, [_row()])
    if missing == "parent_sha":
        data = json.loads(manifest.read_text())
        del data["input"]["physical_sha256"]
        manifest.write_text(json.dumps(data))
    else:
        def remove(data):
            if missing == "child_output":
                del data["output"]
            else:
                del data["output"]["physical_sha256"]
        _rebind_join(tmp_path, remove)
    with pytest.raises(ValueError, match="required SHA-256"):
        collect_image_species(database, manifest, tmp_path / "out")


@pytest.mark.parametrize("change", ["missing", "bytes", "columns", "rows", "path", "sha"])
def test_rejects_incomplete_or_mismatched_raw_member_bindings(tmp_path, change):
    database, manifest = _source(tmp_path, [_row()])
    def mutate(data):
        artifact = data["input"]["artifacts"]["multimedia"]
        if change == "missing":
            del data["input"]["artifacts"]["multimedia"]
        elif change == "bytes":
            artifact["physical_bytes"] += 1
        elif change == "columns":
            artifact["column_count"] += 1
        elif change == "rows":
            data["counts"]["multimedia_rows"] += 1
        elif change == "path":
            artifact["path"] = str(tmp_path / "occurrence.parquet")
        else:
            del artifact["physical_sha256"]
    _rebind_join(tmp_path, mutate)
    with pytest.raises(ValueError, match="raw"):
        collect_image_species(database, manifest, tmp_path / "out")


@pytest.mark.parametrize("change", ["missing", "duplicate"])
def test_rejects_missing_or_duplicate_raw_member_output(tmp_path, change):
    database, manifest = _source(tmp_path, [_row()])
    raw = tmp_path / "raw.json"
    data = json.loads(raw.read_text())
    if change == "missing":
        data["outputs"].pop()
    else:
        data["outputs"].append(data["outputs"][0])
    raw.write_text(json.dumps(data))
    _rebind_join(tmp_path, lambda data: data["input"]["source_manifest"].update(physical_sha256=_sha(raw)))
    with pytest.raises(ValueError, match="raw receipt"):
        collect_image_species(database, manifest, tmp_path / "out")


def test_rejects_fractional_exponent_and_signed_keys_and_embedded_url_space(tmp_path):
    database, manifest = _source(tmp_path, [
        _row(), _row("20.4"), _row("2e1"), _row("+20"), _row("9223372036854775808"),
        _row("21", "Danaus other", familyKey="7017.4"),
        _row("22", "Danaus last", media_identifier="https://example.org/invalid image.jpg"),
    ])
    result = collect_image_species(database, manifest, tmp_path / "out")
    assert result["species_count"] == 1
    assert result["eligible_image_rows"] == 1
    assert pq.read_table(tmp_path / "out/species.parquet").to_pylist()[0]["species_key"] == "20"


def test_resume_rechecks_lineage_that_changes_during_input_hash(tmp_path):
    database, manifest = _source(tmp_path, [_row()])
    output = tmp_path / "out"
    collect_image_species(database, manifest, output)
    before = (output / "manifest.json").read_bytes()

    def change_during_hash(event):
        if event["event"] == "input_hash_progress" and event["path"] == str(database):
            raw = tmp_path / "raw.json"
            raw.write_text(raw.read_text() + " ")

    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        collect_image_species(database, manifest, output, logger=change_during_hash)
    assert (output / "manifest.json").read_bytes() == before
