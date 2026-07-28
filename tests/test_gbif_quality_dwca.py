from __future__ import annotations

from pathlib import Path
import zipfile

import pytest

from biominer.gbif_quality.dwca import inspect_dwca, iter_dwca_rows


META = """\
<archive xmlns="http://rs.tdwg.org/dwc/text/">
  <core encoding="UTF-8" fieldsTerminatedBy="\\t" fieldsEnclosedBy=""
        ignoreHeaderLines="1" rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.txt</location></files>
    <id index="0"/>
    <field index="0" term="http://rs.tdwg.org/dwc/terms/occurrenceID"/>
    <field index="2" term="http://purl.org/dc/terms/license"/>
  </core>
  <extension encoding="UTF-8" fieldsTerminatedBy="\\t" fieldsEnclosedBy=""
             ignoreHeaderLines="1"
             rowType="http://rs.gbif.org/terms/1.0/Multimedia">
    <files><location>multimedia.txt</location></files>
    <coreid index="0"/>
    <field index="1" term="http://purl.org/dc/terms/identifier"/>
    <field index="2" term="http://purl.org/dc/terms/creator"/>
  </extension>
</archive>
"""


def test_dwca_metadata_drives_bounded_row_stream(tmp_path: Path) -> None:
    archive = tmp_path / "fixture.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("meta.xml", META)
        bundle.writestr(
            "occurrence.txt",
            "occurrenceID\tunused\tlicense\n"
            "o-1\tx\tCC BY 4.0\n"
            "o-2\tx\n",
        )
        bundle.writestr(
            "multimedia.txt",
            "coreid\tidentifier\tcreator\n"
            "o-1\thttps://example.org/1.jpg\tA Person\n",
        )

    core, multimedia = inspect_dwca(archive)
    assert core.member == "occurrence.txt"
    assert core.field_separator == "\t"
    assert multimedia.core_id_index == 0

    core_rows = list(iter_dwca_rows(archive, core))
    assert core_rows[0].record_id == "o-1"
    assert core_rows[0].values["license"] == "CC BY 4.0"
    assert core_rows[0].source_row_number == 2
    assert core_rows[0].width_status == "PASS"
    assert core_rows[1].width_status == "FAIL"
    assert core_rows[1].values["license"] is None

    media_rows = list(iter_dwca_rows(archive, multimedia))
    assert media_rows[0].core_id == "o-1"
    assert media_rows[0].values == {
        "identifier": "https://example.org/1.jpg",
        "creator": "A Person",
    }


def test_dwca_rejects_member_path_escape(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "meta.xml",
            META.replace("occurrence.txt", "../occurrence.txt"),
        )
        bundle.writestr("../occurrence.txt", "id\n")
        bundle.writestr("multimedia.txt", "id\n")

    with pytest.raises(ValueError, match="unsafe Darwin Core member path"):
        inspect_dwca(archive)
