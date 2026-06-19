from __future__ import annotations

from collections import Counter
from time import monotonic
from typing import Any

from biominer.registry.gbif import GBIFClient, resolve_family
from biominer.registry.scope import ButterflyScope


def build_gbif_source_snapshot(
    client: GBIFClient,
    scope: ButterflyScope,
    *,
    retrieved_at: str,
    page_limit: int = 1000,
) -> dict[str, Any]:
    started = monotonic()
    root_match = client.match_name(scope.root_scientific_name, rank=scope.root_rank)
    root_key = root_match.get("acceptedUsageKey") or root_match.get("usageKey")
    if root_key is None:
        raise ValueError(f"GBIF did not resolve scope root {scope.root_scientific_name!r}")
    root_usage = client.usage(root_key)
    taxa = [_taxon_row(root_usage, parent_key="", family={})]
    names = [_scientific_name_row(root_usage, name_class="accepted_scientific")]
    assertions: list[dict[str, Any]] = []

    for family_name in scope.included_families:
        resolution = resolve_family(client, family_name, root_name=scope.root_scientific_name)
        family_key = _bare_key(resolution.accepted_taxon_key)
        family_usage = client.usage(family_key)
        taxa.append(_taxon_row(family_usage, parent_key=f"gbif:{root_key}", family=family_usage))
        names.append(_scientific_name_row(family_usage, name_class="accepted_scientific"))
        assertions.append(
            {
                "configured_name": family_name,
                "accepted_taxon_key": resolution.accepted_taxon_key,
                "matched_usage_key": resolution.matched_usage_key,
                "match_type": resolution.match_type,
                "confidence": resolution.confidence,
                "lineage_names": list(resolution.lineage_names),
            }
        )

        for genus in client.children(family_key, rank="GENUS", limit=page_limit):
            genus_key = genus.get("key")
            taxa.append(_taxon_row(genus, parent_key=f"gbif:{family_key}", family=family_usage, genus=genus))
            names.append(_scientific_name_row(genus, name_class="accepted_scientific"))
            for species in client.children(genus_key, rank="SPECIES", limit=page_limit):
                species_key = species.get("key")
                taxa.append(
                    _taxon_row(
                        species,
                        parent_key=f"gbif:{genus_key}",
                        family=family_usage,
                        genus=genus,
                        species=species,
                    )
                )
                names.append(_scientific_name_row(species, name_class="accepted_scientific"))
                for synonym in client.synonyms(species_key, limit=page_limit):
                    names.append(_scientific_name_row(synonym, name_class="scientific_synonym", accepted_key=species_key))
                for vernacular in client.vernacular_names(species_key, limit=page_limit):
                    names.append(_vernacular_name_row(vernacular, accepted_key=species_key))

    return {
        "source": "GBIF",
        "source_version": "gbif-species-api",
        "retrieved_at": retrieved_at,
        "taxa": taxa,
        "names": names,
        "source_assertions": assertions,
        "metrics": _metrics(taxa, names, assertions, gbif_calls=client.call_count, elapsed_seconds=monotonic() - started),
    }


def _metrics(
    taxa: list[dict[str, str]],
    names: list[dict[str, object]],
    assertions: list[dict[str, Any]],
    *,
    gbif_calls: int,
    elapsed_seconds: float,
) -> dict[str, Any]:
    taxa_by_rank = Counter(row["rank"] for row in taxa)
    name_classes = Counter(str(row["name_class"]) for row in names)
    return {
        "gbif_calls": gbif_calls,
        "taxa_rows": len(taxa),
        "taxa_by_rank": dict(sorted(taxa_by_rank.items())),
        "name_rows": len(names),
        "synonym_rows": name_classes.get("scientific_synonym", 0),
        "vernacular_rows": name_classes.get("vernacular", 0),
        "source_assertion_rows": len(assertions),
        "elapsed_seconds": round(elapsed_seconds, 6),
    }


def _taxon_row(
    usage: dict[str, Any],
    *,
    parent_key: str,
    family: dict[str, Any],
    genus: dict[str, Any] | None = None,
    species: dict[str, Any] | None = None,
) -> dict[str, str]:
    key = usage.get("key")
    rank = str(usage.get("rank") or "")
    return {
        "accepted_taxon_key": f"gbif:{key}",
        "scientific_name": _scientific_name(usage),
        "rank": rank,
        "parent_key": parent_key,
        "family_key": f"gbif:{family.get('key')}" if family.get("key") else "",
        "family": _scientific_name(family),
        "genus_key": f"gbif:{(genus or {}).get('key')}" if (genus or {}).get("key") else "",
        "genus": _scientific_name(genus or {}),
        "species_key": f"gbif:{(species or {}).get('key')}" if (species or {}).get("key") else "",
        "species": _scientific_name(species or {}) if rank == "SPECIES" else "",
    }


def _scientific_name_row(usage: dict[str, Any], *, name_class: str, accepted_key: object | None = None) -> dict[str, object]:
    key = accepted_key or usage.get("key")
    name = _scientific_name(usage)
    return {
        "accepted_taxon_key": f"gbif:{key}",
        "verbatim_name": name,
        "display_name": name,
        "language": "la",
        "script": "Latn",
        "region": "",
        "bbox": "",
        "name_class": name_class,
        "source": "GBIF",
        "source_record_id": f"gbif:{usage.get('key')}",
        "trust_tier": "T1",
        "precision_tier": "high",
        "confidence": "high",
        "enabled": True,
    }


def _vernacular_name_row(usage: dict[str, Any], *, accepted_key: object) -> dict[str, object]:
    name = str(usage.get("vernacularName") or "")
    return {
        "accepted_taxon_key": f"gbif:{accepted_key}",
        "verbatim_name": name,
        "display_name": name,
        "language": str(usage.get("language") or ""),
        "script": "",
        "region": str(usage.get("country") or ""),
        "bbox": "",
        "name_class": "vernacular",
        "source": "GBIF",
        "source_record_id": f"gbif:vernacular:{accepted_key}:{name}",
        "trust_tier": "T2",
        "precision_tier": "medium",
        "confidence": "medium",
        "enabled": True,
    }


def _scientific_name(usage: dict[str, Any]) -> str:
    return str(usage.get("canonicalName") or usage.get("scientificName") or "")


def _bare_key(value: str) -> str:
    return value.removeprefix("gbif:")
