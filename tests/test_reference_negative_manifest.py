from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys

import polars as pl
import pytest

from biominer.references.negative_manifest import (
    CURATED_VISUAL_DOMAIN_NEGATIVE_SOURCE_SCHEMA_VERSION,
    REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE,
    REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE,
    REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION,
    REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION,
    CuratedVisualDomainNegativeSourceAdapter,
    compile_curated_visual_domain_negative_manifest,
    curated_visual_domain_negative_manifest_schema,
    load_curated_visual_domain_negative_source,
    publish_curated_visual_domain_negative_manifest,
    validate_curated_visual_domain_negative_manifest,
    write_curated_visual_domain_negative_manifest,
)


_CATEGORY_DOMAINS = {
    "artwork": "artwork",
    "logo": "logo",
    "tattoo": "tattoo",
    "non_butterfly_insect_illustration": "artwork",
    "partial_wing": "partial_wing",
    "misleading_pattern": "unsuitable",
}
_SOURCE_KINDS = (
    "institutional_repository",
    "licensed_media_repository",
    "creator_supplied",
    "commissioned",
    "internal_original",
)
_ROOT_FIELDS = {
    "schema_version",
    "manifest_version",
    "source_snapshot_version",
    "target_accepted_taxon_key",
    "negatives",
}
_ROW_FIELDS = {
    "source_kind",
    "source",
    "source_record_id",
    "provider_media_id",
    "source_record_uri",
    "media_uri",
    "source_sha256",
    "negative_category",
    "target_presence",
    "creator",
    "creator_uri",
    "rights_holder",
    "licence",
    "licence_uri",
    "attribution",
    "rights_evidence_uri",
    "review_status",
    "reviewed_by",
    "reviewed_at",
    "review_confidence",
    "review_notes",
    "exclusion_reason",
    "enabled",
}


def _sha(value: int) -> str:
    return f"sha256:{value:064x}"


def _row(
    category: str = "artwork",
    *,
    index: int = 1,
    target_presence: str = "absent",
    review_status: str = "verified",
    licence: str = "CC BY 4.0",
    licence_uri: str = "https://creativecommons.org/licenses/by/4.0/",
    enabled: bool = True,
) -> dict[str, object]:
    reviewed = review_status in {"verified", "excluded"}
    return {
        "source_kind": _SOURCE_KINDS[(index - 1) % len(_SOURCE_KINDS)],
        "source": "reviewed-local-collection",
        "source_record_id": f"record-{index}",
        "provider_media_id": f"media-{index}",
        "source_record_uri": f"https://archive.example/records/{index}",
        "media_uri": f"https://media.example/images/{index}.jpg",
        "source_sha256": _sha(index),
        "negative_category": category,
        "target_presence": target_presence,
        "creator": f"Creator {index}",
        "creator_uri": f"https://archive.example/creators/{index}",
        "rights_holder": f"Rights Holder {index}",
        "licence": licence,
        "licence_uri": licence_uri,
        "attribution": f"Creator {index} / {licence}",
        "rights_evidence_uri": f"https://archive.example/records/{index}/rights",
        "review_status": review_status,
        "reviewed_by": "reviewer@example.test" if reviewed else None,
        "reviewed_at": "2026-07-14T10:00:00+10:00" if reviewed else None,
        "review_confidence": "high" if reviewed else "unknown",
        "review_notes": "Confirmed against the source record." if reviewed else None,
        "exclusion_reason": "Not suitable for model input."
        if review_status == "excluded"
        else None,
        "enabled": enabled,
    }


def _source(*rows: dict[str, object]) -> dict[str, object]:
    return {
        "schema_version": CURATED_VISUAL_DOMAIN_NEGATIVE_SOURCE_SCHEMA_VERSION,
        "manifest_version": "papilio-demoleus-negatives-v1",
        "source_snapshot_version": "manual-review-2026-07-14",
        "target_accepted_taxon_key": "1938069",
        "negatives": list(rows),
    }


def _all_categories() -> list[dict[str, object]]:
    return [
        _row(category, index=index)
        for index, category in enumerate(_CATEGORY_DOMAINS, start=1)
    ]


def _expected_schema() -> dict[str, pl.DataType]:
    string_columns = [
        "schema_version",
        "negative_reference_id",
        "manifest_version",
        "source_snapshot_version",
        "target_accepted_taxon_key",
        "source_kind",
        "source",
        "source_record_id",
        "provider_media_id",
        "source_record_uri",
        "media_uri",
        "source_sha256",
        "negative_category",
        "visual_domain",
        "target_presence",
        "creator",
        "creator_uri",
        "rights_holder",
        "licence",
        "licence_uri",
        "attribution",
        "rights_evidence_uri",
        "canonical_licence",
        "licence_policy_status",
        "licence_policy_reason",
        "licence_policy_version",
        "licence_policy_fingerprint",
        "review_status",
        "reviewed_by",
    ]
    schema = {name: pl.String for name in string_columns}
    schema["reviewed_at"] = pl.Datetime(time_unit="us", time_zone="UTC")
    schema.update(
        {
            "review_confidence": pl.String,
            "review_notes": pl.String,
            "exclusion_reason": pl.String,
            "enabled": pl.Boolean,
            "row_fingerprint": pl.String,
        }
    )
    return schema


def test_compiles_all_categories_with_exact_schema_and_domain_mapping() -> None:
    result = compile_curated_visual_domain_negative_manifest(
        _source(*_all_categories())
    )

    assert curated_visual_domain_negative_manifest_schema() == _expected_schema()
    assert result.schema == _expected_schema()
    assert result.height == len(_CATEGORY_DOMAINS)
    assert result["schema_version"].unique().to_list() == [
        REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_SCHEMA_VERSION
    ]
    assert (
        dict(result.select("negative_category", "visual_domain").iter_rows())
        == _CATEGORY_DOMAINS
    )
    assert result["licence_policy_status"].unique().to_list() == ["allowed"]
    assert result["canonical_licence"].unique().to_list() == ["cc-by"]
    assert result["enabled"].all()
    assert result["reviewed_at"].unique().to_list()[0].isoformat() == (
        "2026-07-14T00:00:00+00:00"
    )
    assert result["negative_reference_id"].n_unique() == result.height
    assert all(
        value.startswith("sha256:") and len(value) == 71
        for value in result["row_fingerprint"]
    )


def test_compilation_is_stable_under_source_row_order() -> None:
    rows = _all_categories()

    first = compile_curated_visual_domain_negative_manifest(_source(*rows))
    second = compile_curated_visual_domain_negative_manifest(_source(*reversed(rows)))

    assert first.equals(second)


@pytest.mark.parametrize("field", sorted(_ROOT_FIELDS))
def test_loader_rejects_missing_root_fields(field: str) -> None:
    source = _source(_row())
    del source[field]

    with pytest.raises((TypeError, ValueError), match="missing"):
        load_curated_visual_domain_negative_source(source)


def test_loader_rejects_unknown_root_fields() -> None:
    source = _source(_row())
    source["search_query"] = "butterfly logo"

    with pytest.raises(ValueError, match="unknown"):
        load_curated_visual_domain_negative_source(source)


def test_loader_takes_a_deep_snapshot_of_mapping_input() -> None:
    source = _source(_row())

    loaded = load_curated_visual_domain_negative_source(source)
    source["negatives"][0]["creator"] = "Mutated creator"
    loaded["negatives"][0]["rights_holder"] = "Mutated rights holder"

    assert loaded["negatives"][0]["creator"] == "Creator 1"
    assert source["negatives"][0]["rights_holder"] == "Rights Holder 1"


@pytest.mark.parametrize("field", sorted(_ROW_FIELDS))
def test_compiler_rejects_missing_row_fields(field: str) -> None:
    row = _row()
    del row[field]

    with pytest.raises((TypeError, ValueError), match="missing"):
        compile_curated_visual_domain_negative_manifest(_source(row))


def test_compiler_rejects_unknown_row_fields() -> None:
    row = _row()
    row["web_search_rank"] = 1

    with pytest.raises(ValueError, match="unknown"):
        compile_curated_visual_domain_negative_manifest(_source(row))


@pytest.mark.parametrize(
    "field",
    [
        "creator",
        "rights_holder",
        "licence",
        "licence_uri",
        "attribution",
        "rights_evidence_uri",
    ],
)
def test_explicit_rights_and_licence_fields_must_be_nonblank(field: str) -> None:
    row = _row()
    row[field] = "  "

    with pytest.raises(ValueError, match=field):
        compile_curated_visual_domain_negative_manifest(_source(row))


def test_compiler_never_opens_a_network_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"network access attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(socket, "create_connection", fail_network)

    adapter = CuratedVisualDomainNegativeSourceAdapter()
    result = adapter.compile(_source(*_all_categories()))

    assert result.height == len(_CATEGORY_DOMAINS)


@pytest.mark.parametrize("target_presence", ["present", "absent", "unknown"])
def test_target_presence_is_retained(target_presence: str) -> None:
    row = _row(target_presence=target_presence)

    result = compile_curated_visual_domain_negative_manifest(_source(row))

    assert result["target_presence"].item() == target_presence
    assert result["enabled"].item() is True


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("negative_category", "generic_negative", "negative_category"),
        ("source_kind", "web_search", "source_kind"),
        ("source_record_uri", "/relative/record", "source_record_uri"),
        ("media_uri", "ftp://media.example/image.jpg", "media_uri"),
        (
            "rights_evidence_uri",
            "https://user:secret@archive.example/rights",
            "rights_evidence_uri",
        ),
        ("source_sha256", "SHA256:" + "a" * 64, "source_sha256"),
        ("source_sha256", "sha256:short", "source_sha256"),
    ],
)
def test_rejects_invalid_categories_sources_uris_and_checksums(
    field: str,
    value: object,
    message: str,
) -> None:
    row = _row()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        compile_curated_visual_domain_negative_manifest(_source(row))


def test_central_licence_policy_retains_status_but_gates_enabled_rows() -> None:
    allowed = _row(index=1)
    research = _row(
        index=2,
        licence="CC BY-NC 4.0",
        licence_uri="https://creativecommons.org/licenses/by-nc/4.0/",
        enabled=False,
    )
    quarantined = _row(
        index=3,
        licence="Provider Custom Licence",
        licence_uri="https://rights.example/licences/custom",
        enabled=False,
    )

    result = compile_curated_visual_domain_negative_manifest(
        _source(allowed, research, quarantined)
    )
    by_record = {row["source_record_id"]: row for row in result.iter_rows(named=True)}

    assert by_record["record-1"]["licence_policy_status"] == "allowed"
    assert by_record["record-1"]["enabled"] is True
    assert by_record["record-2"]["licence_policy_status"] == "research_only"
    assert by_record["record-2"]["enabled"] is False
    assert by_record["record-3"]["licence_policy_status"] == "quarantined"
    assert by_record["record-3"]["licence_policy_reason"] == (
        "unrecognised_media_licence"
    )
    assert by_record["record-3"]["enabled"] is False

    for row in (research, quarantined):
        row["enabled"] = True
        with pytest.raises(ValueError, match="enabled"):
            compile_curated_visual_domain_negative_manifest(_source(row))


def test_review_provenance_status_and_enabled_rules_are_fail_closed() -> None:
    pending = _row(review_status="pending", enabled=False)
    compile_curated_visual_domain_negative_manifest(_source(pending))

    pending_with_actor = deepcopy(pending)
    pending_with_actor["reviewed_by"] = "reviewer@example.test"
    with pytest.raises(ValueError, match="pending"):
        compile_curated_visual_domain_negative_manifest(_source(pending_with_actor))

    verified_without_actor = _row()
    verified_without_actor["reviewed_by"] = None
    with pytest.raises(ValueError, match="verified"):
        compile_curated_visual_domain_negative_manifest(_source(verified_without_actor))

    verified_with_exclusion = _row()
    verified_with_exclusion["exclusion_reason"] = "contradictory evidence"
    with pytest.raises(ValueError, match="exclusion"):
        compile_curated_visual_domain_negative_manifest(
            _source(verified_with_exclusion)
        )

    excluded = _row(review_status="excluded", enabled=False)
    compiled = compile_curated_visual_domain_negative_manifest(_source(excluded))
    assert compiled["exclusion_reason"].item() == "Not suitable for model input."

    excluded_without_reason = deepcopy(excluded)
    excluded_without_reason["exclusion_reason"] = None
    with pytest.raises(ValueError, match="excluded"):
        compile_curated_visual_domain_negative_manifest(
            _source(excluded_without_reason)
        )

    enabled_pending = deepcopy(pending)
    enabled_pending["enabled"] = True
    with pytest.raises(ValueError, match="enabled|disabled"):
        compile_curated_visual_domain_negative_manifest(_source(enabled_pending))


@pytest.mark.parametrize("review_status", ["verified", "excluded"])
def test_reviewed_rows_require_known_confidence(review_status: str) -> None:
    row = _row(review_status=review_status, enabled=False)
    row["review_confidence"] = "unknown"

    with pytest.raises(ValueError, match="confidence.*unknown"):
        compile_curated_visual_domain_negative_manifest(_source(row))


def test_pending_rows_require_unknown_confidence() -> None:
    row = _row(review_status="pending", enabled=False)
    row["review_confidence"] = "medium"

    with pytest.raises(ValueError, match="pending.*confidence.*unknown"):
        compile_curated_visual_domain_negative_manifest(_source(row))


def test_low_confidence_verified_rows_must_remain_disabled() -> None:
    row = _row(enabled=False)
    row["review_confidence"] = "low"
    compiled = compile_curated_visual_domain_negative_manifest(_source(row))
    assert compiled["review_confidence"].item() == "low"

    row["enabled"] = True
    with pytest.raises(ValueError, match="enabled.*confidence.*high or medium"):
        compile_curated_visual_domain_negative_manifest(_source(row))


@pytest.mark.parametrize(
    "duplicate_field",
    ["source_identity", "media_uri", "source_sha256"],
)
def test_rejects_duplicate_source_media_and_content_identities(
    duplicate_field: str,
) -> None:
    first = _row(index=1)
    second = _row("logo", index=2)
    if duplicate_field == "source_identity":
        second["source"] = first["source"]
        second["source_record_id"] = first["source_record_id"]
        second["provider_media_id"] = first["provider_media_id"]
    else:
        second[duplicate_field] = first[duplicate_field]

    with pytest.raises(ValueError, match="duplicate|unique"):
        compile_curated_visual_domain_negative_manifest(_source(first, second))


def test_validator_and_writer_reject_fingerprint_tampering(tmp_path: Path) -> None:
    frame = compile_curated_visual_domain_negative_manifest(_source(_row()))
    tampered = frame.with_columns(
        pl.lit("Different Rights Holder").alias("rights_holder")
    )

    with pytest.raises(ValueError, match="fingerprint"):
        validate_curated_visual_domain_negative_manifest(tampered)
    with pytest.raises(ValueError, match="fingerprint"):
        write_curated_visual_domain_negative_manifest(
            tampered,
            tmp_path / "tampered.parquet",
        )


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ValueError, match="empty|at least one"):
        compile_curated_visual_domain_negative_manifest(_source())


def test_writer_round_trips_validated_manifest(tmp_path: Path) -> None:
    frame = compile_curated_visual_domain_negative_manifest(_source(*_all_categories()))

    path = write_curated_visual_domain_negative_manifest(frame, tmp_path)

    assert path == tmp_path / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE
    assert pl.read_parquet(path).equals(frame)
    assert not list(tmp_path.glob(".*.tmp"))


def test_publication_is_atomic_create_only_and_reports_no_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_network(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"network access attempted: {args!r} {kwargs!r}")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    source = _source(*_all_categories())
    output = tmp_path / "published"

    paths = publish_curated_visual_domain_negative_manifest(
        source,
        output,
        run_id="negative-run",
    )

    assert set(paths) == {"manifest", "report"}
    assert paths["manifest"] == (
        output / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE
    )
    assert paths["report"] == (
        output / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE
    )
    expected = compile_curated_visual_domain_negative_manifest(source)
    assert pl.read_parquet(paths["manifest"]).equals(expected)

    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    assert report["schema_version"] == (
        REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION
    )
    assert report["status"] == "complete"
    assert report["command"] == "references.compile_visual_domain_negatives"
    assert report["run_id"] == "negative-run"
    assert report["started_at"] <= report["ended_at"]
    assert report["elapsed_seconds"] >= 0.0
    assert report["source_fingerprint"].startswith("sha256:")
    assert report["network_requests"] == 0
    assert report["row_count"] == len(_CATEGORY_DOMAINS)
    assert report["enabled_count"] == len(_CATEGORY_DOMAINS)
    assert report["counts_by_category"] == {
        category: 1 for category in sorted(_CATEGORY_DOMAINS)
    }
    artifact = report["artifact"]
    payload = paths["manifest"].read_bytes()
    assert artifact["uri"] == str(paths["manifest"])
    assert artifact["byte_count"] == len(payload)
    assert artifact["row_count"] == len(_CATEGORY_DOMAINS)
    assert artifact["sha256"] == ("sha256:" + hashlib.sha256(payload).hexdigest())

    with pytest.raises(FileExistsError):
        publish_curated_visual_domain_negative_manifest(
            source,
            output,
            run_id="duplicate-run",
        )
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.glob(".published.*.staging"))


def test_invalid_source_publication_writes_failed_audit(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    output = tmp_path / "invalid-published"

    with caplog.at_level("INFO"), pytest.raises(ValueError, match="at least one"):
        publish_curated_visual_domain_negative_manifest(
            _source(),
            output,
            run_id="invalid-negative-run",
        )

    audits = list(tmp_path.glob(".invalid-published.*.failed.json"))
    assert len(audits) == 1
    report = json.loads(audits[0].read_text(encoding="utf-8"))
    assert report == {
        "schema_version": REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_SCHEMA_VERSION,
        "command": "references.compile_visual_domain_negatives",
        "run_id": "invalid-negative-run",
        "pid": report["pid"],
        "git_sha": report["git_sha"],
        "status": "failed",
        "started_at": report["started_at"],
        "ended_at": report["ended_at"],
        "elapsed_seconds": report["elapsed_seconds"],
        "network_requests": 0,
        "output_dir": str(output),
        "error_type": "ValueError",
        "error": "curated negative source must contain at least one negative",
        "artifact": "not_committed",
    }
    assert report["started_at"] <= report["ended_at"]
    assert report["elapsed_seconds"] >= 0.0
    assert not output.exists()
    assert not list(tmp_path.rglob("*.tmp"))
    assert "reference_visual_domain_negative_publication_started" in caplog.text
    assert "reference_visual_domain_negative_publication_failed" in caplog.text


def test_report_write_failure_never_commits_partial_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "failed-after-manifest"
    original_write_text = Path.write_text

    def fail_committed_report(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        if path.name == REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE:
            raise OSError("injected report write failure")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_committed_report)

    with pytest.raises(OSError, match="injected report write failure"):
        publish_curated_visual_domain_negative_manifest(
            _source(_row()),
            output,
            run_id="failed-after-manifest-run",
        )

    assert not output.exists()
    assert len(list(tmp_path.glob(".failed-after-manifest.*.failed.json"))) == 1
    assert not list(tmp_path.rglob("*.tmp"))


def test_concurrent_cli_publications_commit_exactly_one_directory(
    tmp_path: Path,
) -> None:
    source = tmp_path / "negative-source.json"
    source.write_text(json.dumps(_source(_row())), encoding="utf-8")
    output = tmp_path / "concurrent-publication"
    command = [
        sys.executable,
        "-m",
        "biominer.cli",
        "references",
        "compile-visual-domain-negatives",
        "--source-manifest",
        str(source),
        "--output-dir",
        str(output),
    ]

    processes = [
        subprocess.Popen(  # noqa: S603 - fixed local interpreter and module.
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    results = [process.communicate(timeout=20) for process in processes]

    assert sorted(process.returncode for process in processes) == [0, 2]
    assert (
        sum(json.loads(stdout).get("status") == "complete" for stdout, _ in results)
        == 1
    )
    assert (output / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_FILE).is_file()
    assert (output / REFERENCE_VISUAL_DOMAIN_NEGATIVE_MANIFEST_REPORT_FILE).is_file()
    assert not list(tmp_path.rglob("*.tmp"))
