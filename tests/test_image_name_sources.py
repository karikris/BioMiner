"""Provider boundary controls: fixture HTTP only, never live services."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest

from biominer.registry.enrichment import SpeciesContext
from biominer.registry.enrichment_sources import WikidataClient, COLXRLegacyResolution
from biominer.registry.query_eligibility import assess_name_query_eligibility
from biominer.registry.image_name_sources import (
    ALAImageNamesClient,
    CommonsImageNamesClient,
    GBIFImageNamesClient,
    ProviderPaginationLimit,
    ProviderTaxonMismatch,
)


@pytest.fixture
def species():
    return SpeciesContext("gbif:123", "Papilio demoleus", "gbif:10", "Papilionidae", "gbif:11", "Papilio", ())


def _gbif_taxon():
    return {"key": 123, "canonicalName": "Papilio demoleus", "rank": "SPECIES", "family": "Papilionidae"}


@pytest.mark.parametrize("matched", [False, True])
def test_numeric_col_key_never_bypasses_wikidata_legacy_crosswalk(species, matched):
    calls = []
    class Resolver:
        def resolve(self, context, legacy):
            calls.append((context.accepted_taxon_key, legacy))
            return COLXRLegacyResolution(matched=matched, request_count=2)
    def get(path, params):
        return {"results": {"bindings": [{"taxon": {"value": "https://www.wikidata.org/entity/Q123"},
            "scientificName": {"value": "Papilio demoleus"}, "gbifId": {"value": "123"},
            "commonName": {"value": "Lime butterfly", "xml:lang": "en"}}]}}
    result = WikidataClient(http_get=get, col_xr_resolver=Resolver(),
        legacy_ids_require_crosswalk=True).enrich_species(species)
    assert calls == [("gbif:123", "123")]
    assert len(result["external_links"]) == len(result["name_assertions"]) == int(matched)
    if matched:
        assert result["external_links"][0]["match_method"] == "P225+P846+GBIF_COL_XR"


def test_gbif_binds_identity_and_paginates_names_without_collapsing_sources(species):
    calls = []

    def get(path, params):
        calls.append((path, params))
        if path == "/species/123":
            return _gbif_taxon()
        if path.endswith("synonyms"):
            return {"results": [{"acceptedKey": 123, "canonicalName": "Papilio erithonius", "key": 456}, {"acceptedKey": 999, "canonicalName": "Wrong species"}], "endOfRecords": True}
        offset = params["offset"]
        return {"results": [{"taxonKey": 123, "vernacularName": "Lime Swallowtail", "language": "en-AU", "country": "AU", "sourceTaxonKey": 88 + offset}], "endOfRecords": offset == 1}

    result = GBIFImageNamesClient(http_get=get, page_size=1).enrich_species(species)
    assert result["request_count"] == 4
    names = result["name_assertions"]
    assert [x["display_name"] for x in names] == ["Lime Swallowtail", "Lime Swallowtail", "Papilio erithonius"]
    assert names[0]["language"] == "en-AU" and names[0]["region"] == "AU"
    assert names[0]["source_record_id"] != names[1]["source_record_id"]
    assert names[2]["name_class"] == "scientific_synonym"
    assert all(x["enabled"] for x in names)
    assert all(x["lineage_check"] == "accepted_taxon_key" for x in names)
    assert all(x["source_binding_evidence"] == "exact_species_identity" for x in names)
    assert [p[1]["offset"] for p in calls if p[0].endswith("vernacularNames")] == [0, 1]
    assert all(x["source_response_hash"].startswith("sha256:") and x["retrieved_at"] for x in result["source_snapshots"])


@pytest.mark.parametrize("field,value", [("key", 999), ("canonicalName", "Papilio machaon"), ("rank", "GENUS"), ("family", "Nymphalidae")])
def test_gbif_rejects_taxon_mismatch_before_fetching_names(species, field, value):
    calls = []

    def get(path, params):
        calls.append(path)
        taxon = _gbif_taxon()
        taxon[field] = value
        return taxon

    with pytest.raises(ProviderTaxonMismatch):
        GBIFImageNamesClient(http_get=get).enrich_species(species)
    assert calls == ["/species/123"]


def test_gbif_missing_language_retained_disabled_and_foreign_taxon_rejected(species):
    def get(path, params):
        if path == "/species/123":
            return _gbif_taxon()
        rows = [] if path.endswith("synonyms") else [{"taxonKey": 123, "vernacularName": "Chequered Swallowtail"}, {"taxonKey": 999, "vernacularName": "Unrelated name", "language": "en"}]
        return {"results": rows, "endOfRecords": True}

    result = GBIFImageNamesClient(http_get=get).enrich_species(species)
    assert len(result["name_assertions"]) == 1
    assert result["name_assertions"][0]["enabled"] is False
    assert result["name_assertions"][0]["disabled_reason"] == "missing_language"


def test_gbif_pagination_bound_is_explicit_failure(species):
    def get(path, params):
        if path == "/species/123":
            return _gbif_taxon()
        return {"results": [{"vernacularName": "Lime Swallowtail", "language": "en"}], "endOfRecords": False}

    with pytest.raises(ProviderPaginationLimit):
        GBIFImageNamesClient(http_get=get, max_pages=1).enrich_species(species)


def _gbif_exact_match():
    return {"usageKey": 123, "canonicalName": "Papilio demoleus", "family": "Papilionidae",
            "rank": "SPECIES", "matchType": "EXACT", "confidence": 100, "status": "ACCEPTED"}


def test_gbif_resolves_opaque_source_key_without_rewriting_inventory_identity(species):
    opaque = replace(species, accepted_taxon_key="gbif:SLPT", family_key="gbif:DGC")
    calls = []

    def get(path, params):
        calls.append((path, params))
        if path == "/species/match":
            return _gbif_exact_match()
        if path == "/species/123":
            return _gbif_taxon()
        return {"results": [] if path.endswith("synonyms") else [{"taxonKey": 123, "vernacularName": "Lime Swallowtail", "language": "en"}], "endOfRecords": True}

    result = GBIFImageNamesClient(http_get=get).enrich_species(opaque)
    assert calls[0] == ("/species/match", {"name": "Papilio demoleus", "rank": "SPECIES", "family": "Papilionidae", "strict": "true"})
    assert [path for path, _ in calls] == ["/species/match", "/species/123", "/species/123/vernacularNames", "/species/123/synonyms"]
    assert result["name_assertions"][0]["accepted_taxon_key"] == "gbif:SLPT"
    link = result["external_links"][0]
    assert link["accepted_taxon_key"] == "gbif:SLPT" and link["source_taxon_id"] == "123"
    assert link["match_method"] == "source_opaque_key+exact_scientific_name+family+numeric_usage"
    assert result["request_count"] == 4


@pytest.mark.parametrize("source_key", ["12345", "00123"])
def test_gbif_name_bridge_does_not_treat_numeric_col_key_as_legacy_usage_key(species, source_key):
    source = replace(species, accepted_taxon_key="gbif:" + source_key)
    calls = []

    def get(path, params):
        calls.append(path)
        if path == "/species/match":
            return _gbif_exact_match()
        if path == "/species/123":
            return _gbif_taxon()
        return {"results": [], "endOfRecords": True}

    result = GBIFImageNamesClient(http_get=get, resolve_legacy_by_name=True).enrich_species(source)
    assert calls == ["/species/match", "/species/123", "/species/123/vernacularNames", "/species/123/synonyms"]
    assert "/species/" + source_key not in calls
    link = result["external_links"][0]
    assert link["accepted_taxon_key"] == "gbif:" + source_key and link["source_taxon_id"] == "123"
    assert link["match_method"] == "source_taxon_key+exact_scientific_name+family+numeric_usage"


@pytest.mark.parametrize("source_key,resolve", [("00123", False), ("0", False), ("000", True)])
def test_gbif_legacy_key_syntax_is_not_relaxed_by_source_identity_support(species, source_key, resolve):
    def get(path, params):
        pytest.fail("invalid key must fail before a provider request")

    with pytest.raises(ProviderTaxonMismatch):
        GBIFImageNamesClient(http_get=get, resolve_legacy_by_name=resolve).enrich_species(
            replace(species, accepted_taxon_key="gbif:" + source_key))


@pytest.mark.parametrize("field,value", [
    ("matchType", "FUZZY"), ("matchType", "HIGHERRANK"), ("confidence", 99),
    ("canonicalName", "Papilio machaon"), ("family", "Nymphalidae"),
    ("rank", "GENUS"), ("status", "SYNONYM"), ("acceptedUsageKey", 999),
    ("usageKey", "SLPT"), ("family", ""),
])
def test_gbif_opaque_resolution_rejects_unproven_or_synonym_matches(species, field, value):
    calls = []

    def get(path, params):
        calls.append(path)
        match = _gbif_exact_match()
        match[field] = value
        return match

    with pytest.raises(ProviderTaxonMismatch):
        GBIFImageNamesClient(http_get=get).enrich_species(replace(species, accepted_taxon_key="gbif:SLPT"))
    assert calls == ["/species/match"]


def _ala_profile():
    return {
        "taxonConcept": {"guid": "urn:test:123", "nameString": "Papilio demoleus", "rankString": "species"},
        "classification": {"family": "Papilionidae"},
        "commonNames": [{"nameString": "Lime Swallowtail", "language": "en-AU", "infoSourceURL": "https://example.org/name/1"}, {"nameString": "Unknown-language name"}],
        "synonyms": [{"nameString": "Papilio erithonius", "nameGuid": "urn:name:2"}],
    }


def _ala_search():
    return {"searchResults": {"totalRecords": 1, "results": [{"guid": "urn:test:123", "scientificName": "Papilio demoleus", "rank": "species", "family": "Papilionidae"}]}}


def test_ala_exact_match_and_profile_preserve_language_region_and_synonyms(species):
    calls = []

    def get(path, params):
        calls.append(path)
        return _ala_search() if path == "/search" else _ala_profile()

    result = ALAImageNamesClient(http_get=get).enrich_species(species)
    assert calls == ["/search", "/species/urn%3Atest%3A123"]
    names = result["name_assertions"]
    assert [x["display_name"] for x in names] == ["Lime Swallowtail", "Unknown-language name", "Papilio erithonius"]
    assert names[0]["language"] == "en-AU" and names[0]["region"] == "AU"
    assert names[1]["enabled"] is False
    assert names[2]["name_class"] == "scientific_synonym"
    assert names[0]["lineage_check"] == "accepted_taxon_key"
    assert names[0]["source_binding_evidence"] == "exact_species_identity"
    assert result["request_count"] == 2


@pytest.mark.parametrize("change", ["guid", "name", "rank", "family"])
def test_ala_rejects_profile_identity_drift(species, change):
    profile = _ala_profile()
    if change == "guid":
        profile["taxonConcept"]["guid"] = "urn:other"
    elif change == "name":
        profile["taxonConcept"]["nameString"] = "Papilio machaon"
    elif change == "rank":
        profile["taxonConcept"]["rankString"] = "genus"
    else:
        profile["classification"]["family"] = "Nymphalidae"
    with pytest.raises(ProviderTaxonMismatch):
        ALAImageNamesClient(http_get=lambda path, params: _ala_search() if path == "/search" else profile).enrich_species(species)


def test_ala_rejects_multiple_exact_concepts(species):
    payload = _ala_search()
    second = deepcopy(payload["searchResults"]["results"][0])
    second["guid"] = "urn:second"
    payload["searchResults"]["results"].append(second)
    payload["searchResults"]["totalRecords"] = 2
    with pytest.raises(ProviderTaxonMismatch):
        ALAImageNamesClient(http_get=lambda path, params: payload).enrich_species(species)


@pytest.mark.parametrize("classification", [None, {}, {"genus": "Papilio"}])
def test_ala_requires_profile_family_even_if_search_omits_family(species, classification):
    search = _ala_search()
    del search["searchResults"]["results"][0]["family"]
    profile = _ala_profile()
    if classification is None:
        del profile["classification"]
    else:
        profile["classification"] = classification
    with pytest.raises(ProviderTaxonMismatch):
        ALAImageNamesClient(http_get=lambda path, params: search if path == "/search" else profile).enrich_species(species)


def test_ala_missing_search_family_can_be_proven_by_exact_profile_family(species):
    search = _ala_search()
    del search["searchResults"]["results"][0]["family"]
    result = ALAImageNamesClient(http_get=lambda path, params: search if path == "/search" else _ala_profile()).enrich_species(species)
    assert result["name_assertions"][0]["enabled"] is True


def test_ala_no_exact_species_is_no_match_not_translated_fallback(species):
    payload = _ala_search()
    payload["searchResults"]["results"][0]["scientificName"] = "Papilio machaon"
    result = ALAImageNamesClient(http_get=lambda path, params: payload).enrich_species(species)
    assert result["status"] == "no_match"
    assert result["name_assertions"] == result["external_links"] == []
    assert result["request_count"] == 1


def _commons(text, *, title="Category:Papilio demoleus", qid="Q123"):
    return {"query": {"pages": {"7": {"pageid": 7, "ns": 14, "title": title, "pageprops": {"wikibase_item": qid}, "revisions": [{"revid": 71, "slots": {"main": {"*": text}}}]}}}}


def test_commons_requires_expected_qitem_and_reads_only_explicit_vn_fields(species):
    payload = _commons("caption=Some invented blue butterfly\n{{VN|en=Lime Swallowtail|fr=Voilier échiquier|sciname=Papilio demoleus}}\n[[Category:Butterflies]]")
    result = CommonsImageNamesClient(http_get=lambda path, params: payload, expected_wikidata_items={species.accepted_taxon_key: {"Q123"}}).enrich_species(species)
    names = result["name_assertions"]
    assert [x["display_name"] for x in names] == ["Lime Swallowtail", "Voilier échiquier"]
    assert all(x["enabled"] and x["source_taxon_id"] == "Q123" for x in names)
    assert all(x["lineage_check"] == "accepted_taxon_key" for x in names)
    assert all(x["source_binding_evidence"] == "wikibase_item_exact" for x in names)
    assert assess_name_query_eligibility(names[0]).query_eligible is True
    assert all(x["source_record_id"].startswith("commons:7:71:") for x in names)
    assert result["external_links"][0]["source_taxon_id"] == "Q123"


@pytest.mark.parametrize("expected", [{}, {"gbif:123": {"Q999"}}])
def test_commons_unbound_names_are_disabled_candidates(species, expected):
    payload = _commons("{{VN|en=Lime Swallowtail}}")
    result = CommonsImageNamesClient(http_get=lambda path, params: payload, expected_wikidata_items=expected).enrich_species(species)
    assert result["external_links"] == []
    row = result["name_assertions"][0]
    assert row["enabled"] is False and row["review_state"] == "candidate"
    assert row["disabled_reason"] == "commons_taxon_binding_unconfirmed"
    assert row["source_taxon_id"] == "" and row["observed_source_taxon_id"] == "Q123"
    assert row["lineage_check"] == ""
    assert assess_name_query_eligibility(row).query_eligible is False


@pytest.mark.parametrize("text,title", [
    ("{{VN|en=Lime Swallowtail}}", "Category:Butterfly art"),
    ("{{VN|en=Lime Swallowtail|sciname=Papilio machaon}}", "Category:Papilio demoleus"),
    ("{{VN|en=Lime Swallowtail|useWikidata=Q999}}", "Category:Papilio demoleus"),
    ("A butterfly image caption [[Category:Lime Swallowtail]]", "Category:Papilio demoleus"),
    ("{{VN|en={{unresolved template}}}}", "Category:Papilio demoleus"),
])
def test_commons_does_not_promote_foreign_templates_captions_or_unparsed_markup(species, text, title):
    result = CommonsImageNamesClient(http_get=lambda path, params: _commons(text, title=title), expected_wikidata_items={"gbif:123": {"Q123"}}).enrich_species(species)
    assert result["name_assertions"] == []


@pytest.mark.parametrize("client", [GBIFImageNamesClient, ALAImageNamesClient, CommonsImageNamesClient])
def test_provider_api_errors_propagate_explicitly(species, client):
    with pytest.raises(ValueError, match="API error"):
        client(http_get=lambda path, params: {"error": {"code": "limited"}}).enrich_species(species)
