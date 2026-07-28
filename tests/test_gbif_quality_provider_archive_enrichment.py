from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from biominer.gbif_quality.provider_archive_enrichment import (
    publish_provider_archive_enrichment,
)
from biominer.gbif_quality.provider_review import publish_provider_review_sample


MULTIMEDIA_META = """\
<archive xmlns="http://rs.tdwg.org/dwc/text/">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" fieldsEnclosedBy=""
        ignoreHeaderLines="1" rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <id index="0"/>
    <field index="0" term="http://rs.tdwg.org/dwc/terms/occurrenceID"/>
  </core>
  <extension encoding="UTF-8" fieldsTerminatedBy="\\t" fieldsEnclosedBy=""
             ignoreHeaderLines="1"
             rowType="http://rs.gbif.org/terms/1.0/Multimedia">
    <files><location>multimedia.txt</location></files>
    <coreid index="0"/>
    <field index="1" term="http://purl.org/dc/terms/identifier"/>
    <field index="2" term="http://purl.org/dc/terms/creator"/>
    <field index="3" term="http://purl.org/dc/terms/license"/>
    <field index="4" term="http://purl.org/dc/terms/rightsHolder"/>
    <field index="5" term="http://purl.org/dc/terms/format"/>
    <field index="6" term="http://purl.org/dc/terms/type"/>
  </extension>
</archive>
"""

CORE_ONLY_META = """\
<archive xmlns="http://rs.tdwg.org/dwc/text/">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" fieldsEnclosedBy=""
        ignoreHeaderLines="1" rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <id index="0"/>
    <field index="0" term="http://rs.tdwg.org/dwc/terms/occurrenceID"/>
    <field index="1" term="http://purl.org/dc/terms/license"/>
  </core>
</archive>
"""


def test_provider_archive_enrichment_is_item_scoped_and_retains_outcomes(
    tmp_path: Path,
) -> None:
    dataset_naturemapr = "7ebef267-9d72-4c21-a276-cc84281a8590"
    dataset_vermont = "cf3bdc30-370c-48d3-8fff-b587a39d72d6"
    dataset_danish = "963a6b96-4d22-4428-86e4-afee52cf4a8e"
    v3 = tmp_path / "v3.parquet"
    quality = tmp_path / "quality.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                _source_row(
                    "1",
                    dataset_naturemapr,
                    "occ-1",
                    "https://images.test/1.jpg",
                    creator=None,
                    media_license=None,
                    rights_holder=None,
                ),
                _source_row(
                    "2",
                    dataset_naturemapr,
                    "occ-2",
                    "https://images.test/2.jpg",
                    creator="Old Creator",
                    media_license="CC BY 4.0",
                    rights_holder="Existing Holder",
                ),
                _source_row(
                    "3",
                    dataset_naturemapr,
                    "occ-3",
                    "https://images.test/3.jpg",
                    creator=None,
                    media_license=None,
                    rights_holder=None,
                ),
                _source_row(
                    "4",
                    dataset_vermont,
                    "occ-4",
                    "https://images.test/4.jpg",
                    creator=None,
                    media_license=None,
                    rights_holder=None,
                ),
                _source_row(
                    "5",
                    dataset_danish,
                    "occ-5",
                    "https://images.test/5.jpg",
                    creator=None,
                    media_license="CC BY 4.0",
                    rights_holder=None,
                ),
            ]
        ),
        v3,
    )
    pq.write_table(
        pa.table(
            {
                "source_row_id": [f"source-{index}" for index in range(1, 6)],
                "media_assertion_id": [
                    f"assertion-{index}" for index in range(1, 6)
                ],
            }
        ),
        quality,
    )

    archive_root = tmp_path / "archives"
    archive_root.mkdir()
    naturemapr = archive_root / "naturemapr.zip"
    with zipfile.ZipFile(naturemapr, "w") as bundle:
        bundle.writestr("meta.xml", MULTIMEDIA_META)
        bundle.writestr(
            "occurrence.txt",
            "occurrenceID\nocc-1\nocc-2\nocc-3\n",
        )
        bundle.writestr(
            "multimedia.txt",
            "coreid\tidentifier\tcreator\tlicense\trightsHolder\tformat\ttype\n"
            "occ-1\thttps://images.test/1.jpg\tNew Creator\tCC BY 4.0\t"
            "New Holder\timage/jpeg\tStillImage\n"
            "occ-2\thttps://images.test/2.jpg\tNew Creator\tCC BY 4.0\t"
            "Existing Holder\timage/jpeg\tStillImage\n"
            "occ-3\thttps://images.test/replaced-3.jpg\t\tCopyright\t\t"
            "image/jpeg\tStillImage\n",
        )
    vermont = archive_root / "vermont.zip"
    with zipfile.ZipFile(vermont, "w") as bundle:
        bundle.writestr("meta.xml", CORE_ONLY_META)
        bundle.writestr(
            "occurrence.txt",
            "occurrenceID\tlicense\nocc-4\tCC BY-NC 4.0\n",
        )

    manifest = archive_root / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "archives": [
                    _archive_entry(
                        provider="NatureMapr",
                        dataset_key=dataset_naturemapr,
                        path=naturemapr,
                    ),
                    _archive_entry(
                        provider="Vermont Center for Ecostudies",
                        dataset_key=dataset_vermont,
                        path=vermont,
                    ),
                    {
                        "provider": "Danish SGAV",
                        "dataset_key": dataset_danish,
                        "source_url": "https://provider.test/danish.zip",
                        "path": None,
                        "physical_bytes": None,
                        "sha256": None,
                        "intake_status": "UNRESOLVED_ACCESS_DENIED",
                        "reason": "access denied",
                    },
                ]
            }
        )
        + "\n",
        encoding="utf-8",
    )

    output = tmp_path / "output"
    result = publish_provider_archive_enrichment(
        v3_parquet=v3,
        media_quality_parquet=quality,
        archive_manifest=manifest,
        output_directory=output,
        source_snapshot_id="sha256:source",
        expected_media_rows=5,
        code_commit="commit",
        batch_rows=2,
    )

    assert result["counts"]["target_media_rows"] == 5
    assert result["counts"]["media_outcomes"] == 5
    assert result["counts"]["exact_identifier_matches"] == 2
    assert result["counts"]["occurrence_context_matches"] == 3
    assert result["counts"]["explicit_denied_context_items"] == 1
    assert result["counts"]["new_assertions"] == 3
    assertions = pq.read_table(
        output / "provider_derived_assertions.parquet"
    ).to_pylist()
    assert {
        (row["source_row_id"], row["target_field"], row["derived_value"])
        for row in assertions
    } == {
        ("source-1", "media_license", "CC BY 4.0"),
        ("source-1", "media_creator", "New Creator"),
        ("source-1", "media_rightsHolder", "New Holder"),
    }
    assert all(row["original_value"] is None for row in assertions)
    conflicts = pq.read_table(output / "provider_conflicts.parquet").to_pylist()
    assert [
        (
            row["source_row_id"],
            row["target_field"],
            row["original_value"],
            row["provider_value"],
        )
        for row in conflicts
    ] == [("source-2", "media_creator", "Old Creator", "New Creator")]
    outcomes = pq.read_table(output / "provider_media_outcomes.parquet").to_pylist()
    reasons = {row["source_row_id"]: row["provider_enrichment_reason"] for row in outcomes}
    assert reasons["source-3"] == "no_exact_occurrence_and_identifier_match"
    assert reasons["source-4"] == "archive_has_no_multimedia_table"
    assert reasons["source-5"].startswith("archive_unavailable:")
    contexts = pq.read_table(
        output / "provider_occurrence_context.parquet"
    ).to_pylist()
    assert {row["occurrenceID"] for row in contexts} == {
        "occ-1",
        "occ-2",
        "occ-3",
    }
    assert all(row["item_binding_status"] == "NOT_ITEM_BOUND" for row in contexts)
    assert all(not row["automatic_repair_permitted"] for row in contexts)
    review_output = tmp_path / "provider-review"
    review = publish_provider_review_sample(
        provider_enrichment_directory=output,
        output_directory=review_output,
        code_commit="commit",
        sample_per_stratum=1,
    )
    assert review["counts"]["explicit_denied_context_review_rows"] == 1
    review_rows = pq.read_table(
        review_output / "provider_review_sample.parquet"
    ).to_pylist()
    denied = [
        row
        for row in review_rows
        if row["review_reason"]
        == "current_archive_contains_explicitly_denied_item"
    ]
    assert len(denied) == 1
    assert denied[0]["expected_review_decision"] == (
        "DO_NOT_AUTOMATICALLY_REPAIR"
    )
    assert (output / "manifest.json").is_file()


def _source_row(
    gbif_id: str,
    dataset_key: str,
    occurrence_id: str,
    identifier: str,
    *,
    creator: str | None,
    media_license: str | None,
    rights_holder: str | None,
) -> dict[str, object]:
    return {
        "gbifID": gbif_id,
        "datasetKey": dataset_key,
        "occurrenceID": occurrence_id,
        "media_identifier": identifier,
        "media_license": media_license,
        "media_creator": creator,
        "media_rightsHolder": rights_holder,
        "media_format": "image/jpeg",
        "media_type": "StillImage",
        "media_publisher": None,
        "publisher": "Fixture Provider",
    }


def _archive_entry(
    *, provider: str, dataset_key: str, path: Path
) -> dict[str, object]:
    return {
        "provider": provider,
        "dataset_key": dataset_key,
        "source_url": f"https://provider.test/{path.name}",
        "path": path.name,
        "physical_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "intake_status": "PASS",
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
