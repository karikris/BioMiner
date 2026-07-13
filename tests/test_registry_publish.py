from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from biominer.cli import build_parser, run
from biominer.registry.geographic_spread import (
    GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE,
    GEOGRAPHIC_SPREAD_MANIFEST_FILE,
    TAXON_GEOGRAPHIC_SPREAD_FILE,
    geographic_occurrence_evidence_schema,
    geographic_spread_schema,
)
from biominer.registry.geographic_summary import (
    GEOGRAPHIC_QA_FINDINGS_FILE,
    GEOGRAPHIC_SUMMARY_MANIFEST_FILE,
    TAXON_GEOGRAPHIC_SUMMARY_FILE,
    geographic_qa_schema,
    geographic_spread_fingerprint,
    geographic_summary_schema,
)
from biominer.registry.publish import PUBLISHED_REGISTRY_ARTIFACTS, publish_registry


TARGET_KEY = "gbif:1938069"
REGISTRY_VERSION = "butterflies-v1"
RETRIEVED_AT = "2026-07-13T00:00:00Z"


def test_publish_registry_merges_geographic_audit_artifacts_and_inventory(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "build"
    _write_publishable_registry(registry)

    result = publish_registry(registry, output_dir=tmp_path / "current")

    published = tmp_path / "current"
    assert result["artifacts"] == list(PUBLISHED_REGISTRY_ARTIFACTS)
    assert set(path.name for path in published.iterdir()) == set(PUBLISHED_REGISTRY_ARTIFACTS)
    assert pl.read_parquet(published / TAXON_GEOGRAPHIC_SPREAD_FILE).is_empty()
    summary = pl.read_parquet(published / TAXON_GEOGRAPHIC_SUMMARY_FILE)
    assert summary.row(0, named=True)["data_deficient"] is True
    qa = pl.read_parquet(published / "qa_findings.parquet")
    assert qa.to_dicts() == [
        {
            "severity": "warning",
            "code": "geographic_no_georeferenced_evidence",
            "subject": TARGET_KEY,
        }
    ]
    snapshots = pl.read_parquet(published / "source_snapshots.parquet")
    geographic = snapshots.filter(pl.col("source_version") == "gbif-download:empty-fixture")
    assert geographic.height == 1
    assert geographic.row(0, named=True)["source"] == "GBIF"
    manifest = json.loads((published / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["geographic_spread_rows"] == 0
    assert manifest["geographic_summary_rows"] == 1
    assert manifest["geographic_data_deficient_species"] == 1
    assert manifest["geographic_qa_warning_count"] == 1
    assert manifest["qa_warning_count"] == 1
    assert manifest["qa_status"] == "passed"
    assert set(manifest["artifact_inventory"]) == set(PUBLISHED_REGISTRY_ARTIFACTS) - {
        "manifest.json"
    }
    for filename, entry in manifest["artifact_inventory"].items():
        path = published / filename
        assert entry["byte_count"] == path.stat().st_size
        assert entry["sha256"] == _sha256(path)


def test_publish_registry_blocks_fatal_geographic_qa(tmp_path: Path) -> None:
    registry = tmp_path / "build"
    _write_publishable_registry(registry)
    pl.DataFrame(
        [
            {
                "severity": "fatal",
                "code": "geographic_impossible_cell_identifier",
                "subject": TARGET_KEY,
            }
        ],
        schema=geographic_qa_schema(),
    ).write_parquet(registry / GEOGRAPHIC_QA_FINDINGS_FILE)
    _refresh_geographic_manifests(registry)

    with pytest.raises(ValueError, match="fatal geographic QA"):
        publish_registry(registry, output_dir=tmp_path / "current")

    assert not (tmp_path / "current").exists()


def test_publish_registry_requires_geographic_summary_artifact(tmp_path: Path) -> None:
    registry = tmp_path / "build"
    _write_publishable_registry(registry)
    (registry / TAXON_GEOGRAPHIC_SUMMARY_FILE).unlink()

    with pytest.raises(FileNotFoundError, match=TAXON_GEOGRAPHIC_SUMMARY_FILE):
        publish_registry(registry, output_dir=tmp_path / "current")


def test_publish_registry_rejects_missing_species_summary_as_malformed_not_absent(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "build"
    _write_publishable_registry(registry)
    pl.DataFrame(schema=geographic_summary_schema()).write_parquet(
        registry / TAXON_GEOGRAPHIC_SUMMARY_FILE
    )

    with pytest.raises(ValueError, match="summary must cover every accepted species"):
        publish_registry(registry, output_dir=tmp_path / "current")


def test_publish_registry_rejects_stale_summary_fingerprint(tmp_path: Path) -> None:
    registry = tmp_path / "build"
    _write_publishable_registry(registry)
    summary = pl.read_parquet(registry / TAXON_GEOGRAPHIC_SUMMARY_FILE).with_columns(
        pl.lit("sha256:" + ("9" * 64)).alias("spread_fingerprint")
    )
    summary.write_parquet(registry / TAXON_GEOGRAPHIC_SUMMARY_FILE)
    _refresh_geographic_manifests(registry)

    with pytest.raises(ValueError, match="spread fingerprint mismatch"):
        publish_registry(registry, output_dir=tmp_path / "current")


def test_registry_publish_cli_smoke(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    registry = tmp_path / "build"
    output = tmp_path / "current"
    _write_publishable_registry(registry)

    exit_code = run(
        build_parser().parse_args(
            [
                "registry",
                "publish",
                "--registry-dir",
                str(registry),
                "--output-dir",
                str(output),
            ]
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "published"
    assert payload["manifest_written_last"] is True
    assert (output / "manifest.json").exists()


def _write_publishable_registry(registry: Path) -> None:
    registry.mkdir(parents=True)
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": TARGET_KEY,
                "rank": "SPECIES",
                "taxonomic_status": "ACCEPTED",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    path_row: dict[str, object] = {"accepted_taxon_key": TARGET_KEY, "enabled": True}
    for rank in ("kingdom", "phylum", "class", "order", "family", "genus", "species"):
        path_row[f"{rank}_node_id"] = f"gbif:{rank}"
    pl.DataFrame([path_row]).write_parquet(registry / "species_paths.parquet")
    pl.DataFrame(
        [
            {
                "normalized_match_key": "papilio demoleus",
                "is_canonical_keyword": True,
            }
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame(
        [
            {
                "logical_query_id": "query-1",
                "normalized_match_key": "papilio demoleus",
                "search_field": "tags",
            }
        ]
    ).write_parquet(registry / "flickr_query_definitions.parquet")
    pl.DataFrame(
        [
            {
                "source": "Catalogue of Life",
                "source_version": "COL26.6 XR",
                "retrieved_at": RETRIEVED_AT,
                "source_path": "fixture",
                "source_response_hash": "sha256:" + ("1" * 64),
                "licence": "CC BY 4.0",
                "source_url": "https://www.catalogueoflife.org/",
                "citation": "Fixture source",
            }
        ]
    ).write_parquet(registry / "source_snapshots.parquet")
    pl.DataFrame(schema=geographic_qa_schema()).write_parquet(registry / "qa_findings.parquet")
    (registry / "manifest.json").write_text(
        json.dumps(
            {
                "registry_version": REGISTRY_VERSION,
                "qa_status": "passed",
                "qa_fatal_count": 0,
                "qa_warning_count": 0,
            }
        ),
        encoding="utf-8",
    )
    spread = pl.DataFrame(schema=geographic_spread_schema())
    spread.write_parquet(registry / TAXON_GEOGRAPHIC_SPREAD_FILE)
    pl.DataFrame(schema=geographic_occurrence_evidence_schema()).write_parquet(
        registry / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE
    )
    created = datetime(2026, 7, 13, tzinfo=UTC)
    pl.DataFrame(
        [
            {
                "schema_version": "taxon-geographic-summary-v1.0.0",
                "registry_version": REGISTRY_VERSION,
                "accepted_taxon_key": TARGET_KEY,
                "scientific_name": "Papilio demoleus",
                "geographic_evidence_version": "sha256:" + ("3" * 64),
                "cell_counts_by_resolution": [],
                "countries": [],
                "admin_regions": [],
                "occupied_envelope": None,
                "disconnected_range_component_count": 0,
                "occurrence_density_summary": {
                    "min": None,
                    "p50": None,
                    "p95": None,
                    "max": None,
                },
                "data_deficient": True,
                "data_deficient_reasons": [
                    "no_georeferenced_evidence",
                    "no_range_inference_eligible_occurrences",
                ],
                "suspicious_outlier_cell_count": 0,
                "range_source_coverage": [],
                "known_introduced_regions": [],
                "current_evidence_count": 0,
                "historical_evidence_count": 0,
                "spread_fingerprint": geographic_spread_fingerprint(spread),
                "created_at": created,
            }
        ],
        schema=geographic_summary_schema(),
    ).write_parquet(registry / TAXON_GEOGRAPHIC_SUMMARY_FILE)
    pl.DataFrame(
        [
            {
                "severity": "warning",
                "code": "geographic_no_georeferenced_evidence",
                "subject": TARGET_KEY,
            }
        ],
        schema=geographic_qa_schema(),
    ).write_parquet(registry / GEOGRAPHIC_QA_FINDINGS_FILE)
    (registry / GEOGRAPHIC_SPREAD_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "status": "complete",
                "source": "GBIF",
                "source_query_hash": "sha256:" + ("2" * 64),
                "source_snapshot_version": "gbif-download:empty-fixture",
                "retrieved_at": RETRIEVED_AT,
            }
        ),
        encoding="utf-8",
    )
    (registry / GEOGRAPHIC_SUMMARY_MANIFEST_FILE).write_text(
        json.dumps({"status": "complete", "qa_status": "passed"}),
        encoding="utf-8",
    )
    _refresh_geographic_manifests(registry)


def _refresh_geographic_manifests(registry: Path) -> None:
    spread_manifest = json.loads(
        (registry / GEOGRAPHIC_SPREAD_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    spread_manifest["files"] = {
        "geographic_occurrence_evidence": _manifest_entry(
            registry / GEOGRAPHIC_OCCURRENCE_EVIDENCE_FILE
        ),
        "taxon_geographic_spread": _manifest_entry(
            registry / TAXON_GEOGRAPHIC_SPREAD_FILE
        ),
    }
    (registry / GEOGRAPHIC_SPREAD_MANIFEST_FILE).write_text(
        json.dumps(spread_manifest),
        encoding="utf-8",
    )
    qa = pl.read_parquet(registry / GEOGRAPHIC_QA_FINDINGS_FILE)
    summary_manifest = {
        "status": "complete",
        "qa_status": "failed" if qa.filter(pl.col("severity") == "fatal").height else "passed",
        "files": {
            "taxon_geographic_summary": _manifest_entry(
                registry / TAXON_GEOGRAPHIC_SUMMARY_FILE
            ),
            "geographic_qa_findings": _manifest_entry(
                registry / GEOGRAPHIC_QA_FINDINGS_FILE
            ),
        },
    }
    (registry / GEOGRAPHIC_SUMMARY_MANIFEST_FILE).write_text(
        json.dumps(summary_manifest),
        encoding="utf-8",
    )


def _manifest_entry(path: Path) -> dict[str, object]:
    return {
        "file": path.name,
        "row_count": pl.read_parquet(path).height,
        "byte_count": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
