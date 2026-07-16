from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommonName:
    name: str
    language: str = "und"
    script: str | None = None
    region: str | None = None
    bbox: str | None = None
    source: str | None = None
    source_record_id: str | None = None
    trust_tier: str | None = None
    confidence: str | None = None
    review_state: str | None = None


@dataclass(frozen=True)
class SpeciesSearchTerm:
    term: str
    language: str = "und"
    term_class: str = "unknown"
    source: str | None = None
    source_record_id: str | None = None
    trust_tier: str | None = None
    precision_tier: str | None = None
    confidence: str | None = None
    region: str | None = None
    bbox: str | None = None
    enabled: bool = True
    review_state: str | None = None


@dataclass(frozen=True)
class RegionHint:
    region: str
    bbox: str | None = None
    source: str | None = None
    confidence: str | None = None


@dataclass(frozen=True)
class SpeciesContext:
    scientific_name: str
    accepted_taxon_key: str
    canonical_name: str
    family: str
    genus: str
    family_key: str
    genus_key: str
    species_key: str
    registry_version: str
    synonyms: tuple[str, ...] = ()
    common_names: tuple[CommonName, ...] = ()
    search_terms: tuple[SpeciesSearchTerm, ...] = ()
    regions: tuple[RegionHint, ...] = ()
    source_versions: dict[str, str] = field(default_factory=dict)

    def target_terms(self) -> tuple[str, ...]:
        terms = [self.scientific_name, self.canonical_name, *self.synonyms]
        terms.extend(name.name for name in self.common_names if _review_state_allows_term(name.review_state))
        terms.extend(term.term for term in self.search_terms if term.enabled and _review_state_allows_term(term.review_state))
        return _unique_texts(terms)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SpeciesContext:
        return cls(
            scientific_name=str(payload["scientific_name"]),
            accepted_taxon_key=str(payload["accepted_taxon_key"]),
            canonical_name=str(payload.get("canonical_name") or payload["scientific_name"]),
            family=str(payload.get("family") or ""),
            genus=str(payload.get("genus") or ""),
            family_key=str(payload.get("family_key") or ""),
            genus_key=str(payload.get("genus_key") or ""),
            species_key=str(payload.get("species_key") or payload.get("accepted_taxon_key") or ""),
            registry_version=str(payload.get("registry_version") or ""),
            synonyms=tuple(str(value) for value in payload.get("synonyms", ())),
            common_names=tuple(CommonName(**item) for item in payload.get("common_names", ())),
            search_terms=tuple(SpeciesSearchTerm(**item) for item in payload.get("search_terms", ())),
            regions=tuple(RegionHint(**item) for item in payload.get("regions", ())),
            source_versions={str(key): str(value) for key, value in dict(payload.get("source_versions", {})).items()},
        )

    def write_json(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    @classmethod
    def read_json(cls, path: str | Path) -> SpeciesContext:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _unique_texts(values: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        cleaned = " ".join(str(value or "").split())
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return tuple(output)


def _review_state_allows_term(value: object) -> bool:
    normalized = "_".join(str(value or "").casefold().split())
    return normalized in {"", "accepted", "reviewed", "query_approved", "curator_reviewed", "manual_reviewed"}
