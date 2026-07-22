from __future__ import annotations

import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from biominer.gbif_quality.inventory import (
    INVENTORY_SCHEMA_VERSION,
    SourceInventoryConfig,
    build_source_inventory,
)


def test_source_inventory_recalculates_lineage_and_physical_metadata(
    tmp_path: Path,
) -> None:
    config = _fixture(tmp_path)

    inventory = build_source_inventory(config)

    assert inventory.schema_version == INVENTORY_SCHEMA_VERSION
    assert inventory.source_download_key == "0000001-260101010101001"
    assert inventory.source_package_id == "fixture-package"
    assert inventory.source_publication_date == "2026-01-02"
    assert all(inventory.validation.values())
    assert inventory.artifact_table().num_rows == 6
    by_role = {row["artifact_role"]: row for row in inventory.artifacts}
    assert by_role["occurrence_core"]["row_count"] == 2
    assert by_role["multimedia_extension"]["row_count"] == 3
    assert by_role["rights_filtered_v3"]["column_count"] == 2
    assert by_role["dwca_archive"]["row_count_status"] == "NOT_APPLICABLE"
    assert all(row["row_groups_complete"] for row in inventory.artifacts[1:])


def test_source_inventory_fails_closed_on_archive_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    with config.archive.open("ab") as handle:
        handle.write(b"drift")

    with pytest.raises(ValueError, match="archive checksum mismatch"):
        build_source_inventory(config)


def test_source_inventory_fails_closed_on_parquet_count_drift(tmp_path: Path) -> None:
    config = _fixture(tmp_path)
    dwca = json.loads(config.dwca_manifest.read_text(encoding="utf-8"))
    dwca["outputs"][0]["row_count"] = 999
    config.dwca_manifest.write_text(json.dumps(dwca), encoding="utf-8")

    with pytest.raises(ValueError, match="source inventory validation failed"):
        build_source_inventory(config)


def _fixture(root: Path) -> SourceInventoryConfig:
    archive = root / "download.zip"
    metadata = b'''<?xml version="1.0"?>
<eml:eml xmlns:eml="https://eml.ecoinformatics.org/eml-2.2.0"
 packageId="fixture-package">
 <dataset><title>GBIF Occurrence Download 0000001-260101010101001</title>
 <pubDate>2026-01-02</pubDate></dataset>
</eml:eml>'''
    with ZipFile(archive, "w", ZIP_DEFLATED) as handle:
        handle.writestr("metadata.xml", metadata)
        handle.writestr("meta.xml", "<archive />")
        handle.writestr("occurrence.txt", "gbifID\n1\n2\n")
        handle.writestr("multimedia.txt", "gbifID\n1\n1\n2\n")
        handle.writestr("verbatim.txt", "gbifID\n1\n2\n")
    archive_sha = _sha256(archive)

    occurrence = root / "occurrence.parquet"
    multimedia = root / "multimedia.parquet"
    verbatim = root / "verbatim.parquet"
    joined = root / "joined.parquet"
    v3 = root / "v3.parquet"
    pq.write_table(pa.table({"gbifID": ["1", "2"]}), occurrence, row_group_size=1)
    pq.write_table(pa.table({"gbifID": ["1", "1", "2"]}), multimedia, row_group_size=2)
    pq.write_table(pa.table({"gbifID": ["1", "2"]}), verbatim, row_group_size=1)
    pq.write_table(
        pa.table({"gbifID": ["1", "1", "2"], "identifier": ["a", "b", "c"]}),
        joined,
        row_group_size=2,
    )
    pq.write_table(
        pa.table({"gbifID": ["1", "2"], "media_identifier": ["a", "c"]}),
        v3,
        row_group_size=1,
    )

    dwca_manifest = root / "dwca.json"
    dwca_manifest.write_text(
        json.dumps(
            {
                "source": {"archive_sha256": archive_sha},
                "outputs": [
                    {"member": "occurrence.txt", "row_count": 2, "column_count": 1},
                    {"member": "multimedia.txt", "row_count": 3, "column_count": 1},
                    {"member": "verbatim.txt", "row_count": 2, "column_count": 1},
                ],
            }
        ),
        encoding="utf-8",
    )
    joined_manifest = root / "joined.json"
    joined_manifest.write_text(
        json.dumps(
            {
                "output": {
                    "row_count": 3,
                    "column_count": 2,
                    "physical_sha256": _sha256(joined),
                }
            }
        ),
        encoding="utf-8",
    )
    v3_manifest = root / "v3.json"
    v3_manifest.write_text(
        json.dumps(
            {
                "input": {
                    "row_count": 2,
                    "column_count": 2,
                    "physical_sha256": _sha256(v3),
                }
            }
        ),
        encoding="utf-8",
    )
    prior = root / "prior.json"
    prior.write_text(
        json.dumps(
            {
                "inputs": {
                    "parquets": [
                        {"path": str(path), "sha256": _sha256(path)}
                        for path in (occurrence, multimedia, verbatim)
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    return SourceInventoryConfig(
        repository_root=root,
        archive=archive,
        dwca_manifest=dwca_manifest,
        occurrence_parquet=occurrence,
        multimedia_parquet=multimedia,
        verbatim_parquet=verbatim,
        joined_parquet=joined,
        joined_manifest=joined_manifest,
        v3_parquet=v3,
        v3_manifest=v3_manifest,
        prior_intake_manifest=prior,
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
