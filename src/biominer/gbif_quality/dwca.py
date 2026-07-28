from __future__ import annotations

import csv
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from typing import Iterator
from xml.etree import ElementTree
import zipfile


@dataclass(frozen=True, slots=True)
class DwcaField:
    index: int | None
    term: str
    default: str | None

    @property
    def name(self) -> str:
        return self.term.rsplit("/", 1)[-1].rsplit("#", 1)[-1]


@dataclass(frozen=True, slots=True)
class DwcaTable:
    role: str
    row_type: str
    member: str
    fields: tuple[DwcaField, ...]
    id_index: int | None
    core_id_index: int | None
    encoding: str
    field_separator: str
    enclosed_by: str | None
    ignored_header_rows: int

    @property
    def maximum_index(self) -> int:
        indexes = [
            field.index for field in self.fields if field.index is not None
        ]
        indexes.extend(
            value
            for value in (self.id_index, self.core_id_index)
            if value is not None
        )
        return max(indexes, default=-1)


@dataclass(frozen=True, slots=True)
class DwcaRow:
    table_role: str
    row_type: str
    member: str
    source_row_number: int
    record_id: str | None
    core_id: str | None
    values: dict[str, str | None]
    observed_field_count: int
    expected_minimum_field_count: int
    width_status: str


def inspect_dwca(archive: str | Path) -> tuple[DwcaTable, ...]:
    source = Path(archive)
    if not source.is_file():
        raise FileNotFoundError(source)
    with zipfile.ZipFile(source) as bundle:
        if "meta.xml" not in bundle.namelist():
            raise ValueError("Darwin Core Archive has no meta.xml")
        with bundle.open("meta.xml") as handle:
            root = ElementTree.parse(handle).getroot()
        tables = []
        for element in root:
            role = _local_name(element.tag)
            if role not in {"core", "extension"}:
                continue
            tables.append(_parse_table(element, role=role, members=bundle.namelist()))
    if not tables or tables[0].role != "core":
        raise ValueError("Darwin Core Archive has no core table")
    return tuple(tables)


def iter_dwca_rows(
    archive: str | Path,
    table: DwcaTable,
) -> Iterator[DwcaRow]:
    source = Path(archive)
    with zipfile.ZipFile(source) as bundle:
        if table.member not in bundle.namelist():
            raise ValueError(f"Darwin Core member is missing: {table.member}")
        with bundle.open(table.member) as raw, TextIOWrapper(
            raw,
            encoding=table.encoding,
            errors="strict",
            newline="",
        ) as text:
            reader = csv.reader(
                text,
                delimiter=table.field_separator,
                quotechar=table.enclosed_by,
                quoting=(
                    csv.QUOTE_MINIMAL
                    if table.enclosed_by is not None
                    else csv.QUOTE_NONE
                ),
            )
            for _ in range(table.ignored_header_rows):
                next(reader, None)
            for row_number, raw_values in enumerate(
                reader,
                start=table.ignored_header_rows + 1,
            ):
                if not raw_values or not any(value.strip() for value in raw_values):
                    continue
                expected = table.maximum_index + 1
                width_status = "PASS" if len(raw_values) >= expected else "FAIL"
                values = {
                    field.name: (
                        _value(raw_values, field.index)
                        if field.index is not None
                        else field.default
                    )
                    for field in table.fields
                }
                yield DwcaRow(
                    table_role=table.role,
                    row_type=table.row_type,
                    member=table.member,
                    source_row_number=row_number,
                    record_id=_value(raw_values, table.id_index),
                    core_id=_value(raw_values, table.core_id_index),
                    values=values,
                    observed_field_count=len(raw_values),
                    expected_minimum_field_count=expected,
                    width_status=width_status,
                )


def _parse_table(
    element: ElementTree.Element,
    *,
    role: str,
    members: list[str],
) -> DwcaTable:
    member_values = [
        child.text.strip()
        for child in element.iter()
        if _local_name(child.tag) == "location"
        and child.text
        and child.text.strip()
    ]
    if len(member_values) != 1:
        raise ValueError(f"Darwin Core {role} must name exactly one member")
    member = member_values[0]
    _validate_member(member)
    if member not in members:
        raise ValueError(f"Darwin Core member is missing: {member}")
    fields = tuple(
        DwcaField(
            index=(
                _positive_index(child.attrib.get("index"))
                if child.attrib.get("index") is not None
                else None
            ),
            term=_required_text(child.attrib.get("term"), field="term"),
            default=_optional_text(child.attrib.get("default")),
        )
        for child in element
        if _local_name(child.tag) == "field"
    )
    if any(field.index is None and field.default is None for field in fields):
        raise ValueError("Darwin Core unindexed field requires a default")
    indexes = [field.index for field in fields if field.index is not None]
    if len(indexes) != len(set(indexes)):
        raise ValueError(f"Darwin Core {role} repeats a field index")
    id_index = _single_index(element, "id")
    core_id_index = _single_index(element, "coreid")
    if role == "core" and id_index is None:
        raise ValueError("Darwin Core core table has no id index")
    if role == "extension" and core_id_index is None:
        raise ValueError("Darwin Core extension has no coreid index")
    return DwcaTable(
        role=role,
        row_type=_required_text(element.attrib.get("rowType"), field="rowType"),
        member=member,
        fields=fields,
        id_index=id_index,
        core_id_index=core_id_index,
        encoding=(element.attrib.get("encoding") or "UTF-8").strip(),
        field_separator=_separator(
            element.attrib.get("fieldsTerminatedBy", ","),
            field="fieldsTerminatedBy",
        ),
        enclosed_by=_enclosure(element.attrib.get("fieldsEnclosedBy", "")),
        ignored_header_rows=_nonnegative_integer(
            element.attrib.get("ignoreHeaderLines", "0"),
            field="ignoreHeaderLines",
        ),
    )


def _single_index(element: ElementTree.Element, tag: str) -> int | None:
    matches = [
        child
        for child in element
        if _local_name(child.tag) == tag
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError(f"Darwin Core table repeats {tag}")
    return _positive_index(matches[0].attrib.get("index"))


def _separator(value: str, *, field: str) -> str:
    decoded = {"\\t": "\t", "\\n": "\n", "\\r": "\r"}.get(value, value)
    if len(decoded) != 1:
        raise ValueError(f"{field} must decode to one character")
    return decoded


def _enclosure(value: str) -> str | None:
    if value == "":
        return None
    return _separator(value, field="fieldsEnclosedBy")


def _validate_member(member: str) -> None:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe Darwin Core member path: {member}")


def _positive_index(value: str | None) -> int:
    index = _nonnegative_integer(value, field="index")
    return index


def _nonnegative_integer(value: str | None, *, field: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Darwin Core {field} is not an integer") from exc
    if parsed < 0:
        raise ValueError(f"Darwin Core {field} must be nonnegative")
    return parsed


def _required_text(value: str | None, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"Darwin Core {field} is required")
    return result


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _value(values: list[str], index: int | None) -> str | None:
    if index is None or index >= len(values):
        return None
    result = values[index].strip()
    return result or None


def _local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1]


__all__ = [
    "DwcaField",
    "DwcaRow",
    "DwcaTable",
    "inspect_dwca",
    "iter_dwca_rows",
]
