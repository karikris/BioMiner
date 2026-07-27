from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime
import re


PARSER_VERSION = "biominer-gbif-temporal-parser/v1"

_DATE_PATTERN = re.compile(r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}\Z")
_TIMESTAMP_PATTERN = re.compile(
    r"\A[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}"
    r"(?::[0-9]{2}(?:\.[0-9]{1,6})?)?"
    r"(?:Z|[+-][0-9]{2}:[0-9]{2})?\Z"
)
_PARTIAL_PATTERN = re.compile(r"\A(?:[0-9]{4}|[0-9]{4}-[0-9]{2})\Z")


@dataclass(frozen=True, slots=True)
class TemporalDerivation:
    derived_year: int | None = None
    derived_month: int | None = None
    derived_day: int | None = None
    method: str | None = None
    status: str = "unsupported_format"
    derived_components: str | None = None
    interval_start: str | None = None
    interval_end: str | None = None
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class _Endpoint:
    value: date
    source: str
    timestamp: bool
    instant: datetime | None


def derive_temporal_components(
    *,
    event_date: object | None,
    year: object | None,
    month: object | None,
    day: object | None,
) -> TemporalDerivation:
    """Derive only blank temporal components from a strict ISO eventDate.

    Intervals use their start boundary. Existing components are never replaced,
    and any conflict causes the record to fail closed.
    """

    original = (_trimmed(year), _trimmed(month), _trimmed(day))
    missing = tuple(value is None for value in original)
    if not any(missing):
        return TemporalDerivation(status="not_needed")

    raw = _trimmed(event_date)
    if raw is None:
        return TemporalDerivation(status="missing_event_date")
    if _PARTIAL_PATTERN.fullmatch(raw):
        try:
            date.fromisoformat(raw + ("-01-01" if len(raw) == 4 else "-01"))
        except ValueError:
            return TemporalDerivation(status="invalid_calendar_date")
        return TemporalDerivation(status="insufficient_precision")

    values = raw.split("/")
    if len(values) > 2:
        return TemporalDerivation(status="unsupported_format")
    start, start_error = _parse_endpoint(values[0])
    if start is None:
        return TemporalDerivation(status=start_error)
    end: _Endpoint | None = None
    if len(values) == 2:
        end, end_error = _parse_endpoint(values[1])
        if end is None:
            return TemporalDerivation(status=end_error)
        comparison = _compare_endpoints(start, end)
        if comparison is None:
            return TemporalDerivation(
                status="unsupported_format",
                interval_start=start.source,
                interval_end=end.source,
            )
        if comparison > 0:
            return TemporalDerivation(
                status="reversed_interval",
                interval_start=start.source,
                interval_end=end.source,
            )

    parsed = (start.value.year, start.value.month, start.value.day)
    if _has_conflict(original, parsed):
        return TemporalDerivation(
            status="existing_component_conflict",
            interval_start=start.source,
            interval_end=end.source if end else None,
        )

    derived = tuple(value if is_missing else None for value, is_missing in zip(parsed, missing))
    component_names = tuple(
        name
        for name, value
        in zip(("year", "month", "day"), derived)
        if value is not None
    )
    if not component_names:
        return TemporalDerivation(status="not_needed")
    if end is None:
        method = "event_date_single_timestamp" if start.timestamp else "event_date_single_date"
    else:
        method = (
            "event_date_interval_start_timestamp"
            if start.timestamp
            else "event_date_interval_start_date"
        )
    result = TemporalDerivation(
        derived_year=derived[0],
        derived_month=derived[1],
        derived_day=derived[2],
        method=method,
        status="derived",
        derived_components="|".join(component_names),
        interval_start=start.source,
        interval_end=end.source if end else None,
    )
    if result.derived_year is not None and result.derived_year < 1960:
        return replace(
            result,
            status="excluded_pre_1960",
            exclusion_reason="derived_year_pre_1960",
        )
    return result


def _parse_endpoint(raw: str) -> tuple[_Endpoint | None, str]:
    value = raw.strip()
    timestamp = bool(_TIMESTAMP_PATTERN.fullmatch(value))
    if not timestamp and not _DATE_PATTERN.fullmatch(value):
        return None, "unsupported_format"
    try:
        instant: datetime | None = None
        if timestamp:
            instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
            parsed = instant.date()
        else:
            parsed = date.fromisoformat(value)
    except ValueError:
        return None, "invalid_calendar_date"
    return (
        _Endpoint(
            value=parsed,
            source=value,
            timestamp=timestamp,
            instant=instant,
        ),
        "derived",
    )


def _compare_endpoints(start: _Endpoint, end: _Endpoint) -> int | None:
    if start.instant is None or end.instant is None:
        return (start.value > end.value) - (start.value < end.value)
    start_aware = start.instant.utcoffset() is not None
    end_aware = end.instant.utcoffset() is not None
    if start_aware != end_aware:
        return None
    return (start.instant > end.instant) - (start.instant < end.instant)


def _has_conflict(
    original: tuple[str | None, str | None, str | None],
    parsed: tuple[int, int, int],
) -> bool:
    for raw, expected in zip(original, parsed):
        if raw is None:
            continue
        try:
            if int(raw) != expected:
                return True
        except ValueError:
            return True
    return False


def _trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


__all__ = ["PARSER_VERSION", "TemporalDerivation", "derive_temporal_components"]
