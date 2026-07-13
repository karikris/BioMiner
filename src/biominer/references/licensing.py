from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from urllib.parse import urlparse


REFERENCE_LICENCE_POLICY_VERSION = "reference-licences-v1"

_KNOWN_CREATIVE_COMMONS_LICENCES = frozenset(
    {
        "cc0",
        "cc-by",
        "cc-by-sa",
        "cc-by-nc",
        "cc-by-nc-sa",
        "cc-by-nd",
        "cc-by-nc-nd",
    }
)
_KNOWN_CREATIVE_COMMONS_VERSIONS = frozenset({"1.0", "2.0", "2.5", "3.0", "4.0"})
_KNOWN_CC0_VERSIONS = frozenset({"1.0"})
_CODE_PATTERN = re.compile(
    r"(?:cc[- _]*)?(?P<code>0|zero|by(?:[- _]+(?:nc|nd|sa)){0,2})"
    r"(?:[- _]+v?(?P<version>\d+(?:\.\d+)?))?\Z",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ReferenceLicenceDecision:
    status: str
    canonical_licence: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class _ResolvedLicence:
    canonical: str
    version: str | None


@dataclass(frozen=True, slots=True)
class ReferenceLicencePolicy:
    version: str = REFERENCE_LICENCE_POLICY_VERSION
    broadly_reusable: tuple[str, ...] = ("cc0", "cc-by", "cc-by-sa")
    research_only: tuple[str, ...] = ("cc-by-nc", "cc-by-nc-sa")
    attribution_required: tuple[str, ...] = (
        "cc-by",
        "cc-by-sa",
        "cc-by-nc",
        "cc-by-nc-sa",
    )
    licence_aliases: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        version = str(self.version or "").strip()
        if not version:
            raise ValueError("licence policy version must be nonblank")
        object.__setattr__(self, "version", version)
        for field in ("broadly_reusable", "research_only", "attribution_required"):
            values = _canonical_policy_values(getattr(self, field), field=field)
            object.__setattr__(self, field, values)
        aliases = _normalise_aliases(self.licence_aliases)
        object.__setattr__(self, "licence_aliases", aliases)
        overlap = _overlapping_policy_entries(
            self.broadly_reusable,
            self.research_only,
        )
        if overlap:
            raise ValueError(
                "broadly reusable and research-only licence allowlists overlap: "
                f"{sorted(overlap)}"
            )
        allowed = self.broadly_reusable + self.research_only
        unknown_attribution = [
            value
            for value in self.attribution_required
            if not _policy_entry_is_covered(value, allowed)
        ]
        if unknown_attribution:
            raise ValueError(
                "attribution-required licences must also be allowed: "
                f"{sorted(unknown_attribution)}"
            )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "version": self.version,
                "broadly_reusable": self.broadly_reusable,
                "research_only": self.research_only,
                "attribution_required": self.attribution_required,
                "licence_aliases": self.licence_aliases,
            }
        )

    def evaluate(
        self,
        *,
        media_licence: object,
        licence_uri: object,
        attribution: object,
    ) -> ReferenceLicenceDecision:
        supplied = [
            str(value).strip()
            for value in (media_licence, licence_uri)
            if str(value or "").strip()
        ]
        if not supplied:
            return ReferenceLicenceDecision(
                status="quarantined",
                canonical_licence=None,
                reason="missing_media_licence",
            )
        allowed = set(self.broadly_reusable) | set(self.research_only)
        aliases = dict(self.licence_aliases)
        resolved_licences = [
            _resolve_policy_licence(value, allowed=allowed, aliases=aliases)
            for value in supplied
        ]
        if any(value is None for value in resolved_licences):
            return ReferenceLicenceDecision(
                status="quarantined",
                canonical_licence=None,
                reason="unrecognised_media_licence",
            )
        canonical_values = frozenset(
            value.canonical for value in resolved_licences if value is not None
        )
        explicit_versions = frozenset(
            value.version
            for value in resolved_licences
            if value is not None and value.version is not None
        )
        if len(canonical_values) != 1 or len(explicit_versions) > 1:
            return ReferenceLicenceDecision(
                status="quarantined",
                canonical_licence=None,
                reason="conflicting_media_licence",
            )
        value = next(iter(canonical_values))
        explicit_version = next(iter(explicit_versions), None)
        if _policy_allows(
            self.broadly_reusable,
            canonical=value,
            version=explicit_version,
        ):
            status = "allowed"
        elif _policy_allows(
            self.research_only,
            canonical=value,
            version=explicit_version,
        ):
            status = "research_only"
        else:
            return ReferenceLicenceDecision(
                status="denied",
                canonical_licence=value,
                reason=f"media_licence_not_allowed:{value}",
            )
        if (
            _policy_allows(
                self.attribution_required,
                canonical=value,
                version=explicit_version,
            )
            and not str(attribution or "").strip()
        ):
            return ReferenceLicenceDecision(
                status="quarantined",
                canonical_licence=value,
                reason="missing_required_attribution",
            )
        return ReferenceLicenceDecision(
            status=status,
            canonical_licence=value,
            reason=None,
        )


def canonicalise_creative_commons_licence(value: object) -> str | None:
    resolved = _parse_creative_commons_licence(value)
    return resolved.canonical if resolved is not None else None


def _parse_creative_commons_licence(value: object) -> _ResolvedLicence | None:
    text = str(value or "").strip().casefold()
    if not text:
        return None
    try:
        parsed = urlparse(text)
    except ValueError:
        return None
    if parsed.scheme or parsed.netloc:
        if parsed.scheme not in {"http", "https"} or parsed.query or parsed.fragment:
            return None
        host = (parsed.hostname or "").removeprefix("www.")
        if host != "creativecommons.org" or parsed.username or parsed.password:
            return None
        try:
            port = parsed.port
        except ValueError:
            return None
        default_port = 80 if parsed.scheme == "http" else 443
        if port is not None and port != default_port:
            return None
        parts = [part for part in parsed.path.casefold().split("/") if part]
        if (
            len(parts) in {3, 4}
            and parts[:2] == ["publicdomain", "zero"]
            and parts[2] in _KNOWN_CC0_VERSIONS
            and (len(parts) == 3 or _valid_licence_document(parts[3]))
        ):
            return _ResolvedLicence(canonical="cc0", version=parts[2])
        if (
            len(parts) in {3, 4}
            and parts[0] == "licenses"
            and parts[2] in _KNOWN_CREATIVE_COMMONS_VERSIONS
            and (len(parts) == 3 or _valid_licence_document(parts[3]))
        ):
            candidate = f"cc-{parts[1]}"
            if candidate in _KNOWN_CREATIVE_COMMONS_LICENCES:
                return _ResolvedLicence(canonical=candidate, version=parts[2])
            return None
        return None
    normalized = text.replace("creative commons", "cc")
    normalized = re.sub(r"\b(?:licen[cs]e|legalcode)\b", "", normalized)
    normalized = re.sub(r"[()]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" -_")
    match = _CODE_PATTERN.fullmatch(normalized)
    if match is None:
        return None
    code = match.group("code").casefold().replace("_", "-").replace(" ", "-")
    candidate = "cc0" if code in {"0", "zero"} else f"cc-{code}"
    version = match.group("version")
    known_versions = (
        _KNOWN_CC0_VERSIONS if candidate == "cc0" else _KNOWN_CREATIVE_COMMONS_VERSIONS
    )
    if (
        candidate not in _KNOWN_CREATIVE_COMMONS_LICENCES
        or version is not None
        and version not in known_versions
    ):
        return None
    return _ResolvedLicence(canonical=candidate, version=version)


def _valid_licence_document(value: str) -> bool:
    return (
        value == "legalcode"
        or re.fullmatch(r"deed(?:\.[a-z]{2,3})?", value) is not None
    )


def _canonical_policy_values(values: tuple[str, ...], *, field: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not all(
        isinstance(value, str) for value in values
    ):
        raise TypeError(f"{field} must be a tuple of strings")
    canonical: list[str] = []
    for raw in values:
        resolved = _parse_creative_commons_licence(raw)
        if resolved is None and _looks_like_creative_commons_licence(raw):
            raise ValueError(f"invalid Creative Commons licence: {raw!r}")
        value = (
            _resolved_policy_identifier(resolved)
            if resolved is not None
            else _policy_identifier(raw)
        )
        canonical.append(value)
    if len(canonical) != len(set(canonical)):
        raise ValueError(f"{field} contains duplicate licences")
    return tuple(sorted(canonical))


def _normalise_aliases(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if not isinstance(values, tuple):
        raise TypeError("licence_aliases must be a tuple")
    aliases: dict[str, str] = {}
    for value in values:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(
                "licence_aliases entries must be (source, canonical) tuples"
            )
        if not all(isinstance(item, str) for item in value):
            raise TypeError("licence alias source and canonical value must be strings")
        source = str(value[0] or "").strip().casefold()
        if not source:
            raise ValueError("licence alias source must be nonblank")
        if _looks_like_creative_commons_licence(source):
            raise ValueError(
                "licence aliases cannot override Creative Commons licences"
            )
        resolved = _parse_creative_commons_licence(value[1])
        if resolved is None and _looks_like_creative_commons_licence(value[1]):
            raise ValueError(f"invalid Creative Commons licence: {value[1]!r}")
        canonical = (
            _resolved_policy_identifier(resolved)
            if resolved is not None
            else _policy_identifier(value[1])
        )
        existing = aliases.get(source)
        if existing is not None and existing != canonical:
            raise ValueError(f"licence alias is ambiguous: {value[0]!r}")
        aliases[source] = canonical
    return tuple(sorted(aliases.items()))


def _resolve_policy_licence(
    value: str,
    *,
    allowed: set[str],
    aliases: dict[str, str],
) -> _ResolvedLicence | None:
    exact = value.strip().casefold()
    if exact in aliases:
        alias = aliases[exact]
        return _parse_creative_commons_licence(alias) or _ResolvedLicence(
            canonical=alias,
            version=None,
        )
    creative_commons = _parse_creative_commons_licence(value)
    if creative_commons is not None:
        return creative_commons
    if _looks_like_creative_commons_licence(value):
        return None
    try:
        identifier = _policy_identifier(value)
    except ValueError:
        return None
    if identifier in allowed:
        return _ResolvedLicence(canonical=identifier, version=None)
    return None


def _looks_like_creative_commons_licence(value: object) -> bool:
    text = str(value or "").strip().casefold()
    if not text:
        return False
    try:
        parsed = urlparse(text)
    except ValueError:
        return text.startswith(("creative commons", "cc"))
    if (parsed.hostname or "").removeprefix("www.") == "creativecommons.org":
        return True
    return re.match(r"(?:creative\s+commons\b|cc(?:0|[- _]))", text) is not None


def _policy_identifier(value: object) -> str:
    identifier = re.sub(
        r"[- _]+",
        "-",
        str(value or "").strip().casefold(),
    ).strip("-")
    if not identifier or re.fullmatch(r"[a-z0-9][a-z0-9.+:-]*", identifier) is None:
        raise ValueError(f"invalid licence policy identifier: {value!r}")
    return identifier


def _resolved_policy_identifier(value: _ResolvedLicence) -> str:
    if value.version is None:
        return value.canonical
    return f"{value.canonical}-{value.version}"


def _policy_entry(value: str) -> _ResolvedLicence:
    return _parse_creative_commons_licence(value) or _ResolvedLicence(
        canonical=value,
        version=None,
    )


def _policy_allows(
    entries: tuple[str, ...],
    *,
    canonical: str,
    version: str | None,
) -> bool:
    for raw in entries:
        entry = _policy_entry(raw)
        if entry.canonical != canonical:
            continue
        if entry.version is None or entry.version == version:
            return True
    return False


def _overlapping_policy_entries(
    first: tuple[str, ...],
    second: tuple[str, ...],
) -> set[str]:
    overlaps: set[str] = set()
    for first_raw in first:
        first_entry = _policy_entry(first_raw)
        for second_raw in second:
            second_entry = _policy_entry(second_raw)
            if first_entry.canonical != second_entry.canonical:
                continue
            if (
                first_entry.version is None
                or second_entry.version is None
                or first_entry.version == second_entry.version
            ):
                overlaps.add(first_entry.canonical)
    return overlaps


def _policy_entry_is_covered(value: str, allowed: tuple[str, ...]) -> bool:
    required = _policy_entry(value)
    for raw in allowed:
        candidate = _policy_entry(raw)
        if candidate.canonical != required.canonical:
            continue
        if candidate.version is None or candidate.version == required.version:
            return True
    return False


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "REFERENCE_LICENCE_POLICY_VERSION",
    "ReferenceLicenceDecision",
    "ReferenceLicencePolicy",
    "canonicalise_creative_commons_licence",
]
