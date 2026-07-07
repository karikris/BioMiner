# Registry Multilingual Sources

BioMiner treats multilingual names as registry evidence, not as taxonomic authority. GBIF accepted taxon keys remain the production identity. Supplemental sources can add source-backed names, regional context, and review candidates, but they do not replace the accepted spine.

## Data Flow

1. Build the accepted GBIF family, genus, and species spine.
2. Optionally discover occurrence-backed country coverage from GBIF occurrence facets.
3. Convert occurrence countries and region seed files into language targets.
4. Enrich names from curated APIs, curated static CSV snapshots, and translation providers.
5. Compile Flickr query definitions only from enabled, query-eligible names.

Occurrence-driven language discovery keeps the question narrow: where does this accepted taxon have evidence, and which languages are relevant in those regions? It does not assign lower-rank taxonomy to broad Flickr hits and does not validate identifications.

## Source Roles

| Source class | Role | Query policy |
|---|---|---|
| GBIF accepted taxonomy | Accepted taxon key spine, hierarchy, accepted/synonym relations | T1 scientific species names and accepted synonyms can be query-eligible |
| Catalogue of Life | Supplemental accepted/synonym and vernacular evidence, discrepancy QA | T1/T2 after binding to an accepted GBIF taxon |
| GBIF/CoL/ITIS/TMD vernaculars | Source-backed common names | T2 names may be query-eligible when species-specific and collision-free |
| Wikidata/Wikimedia | Labels, aliases, interlanguage titles, QIDs, external IDs | T3 names require confident same-taxon binding |
| iNaturalist/community sources | Regional preferred or place-linked names | T4 names require review or corroboration |
| Curated static CSV sources | Reviewed snapshots from biodiversity sources without a stable API | T2 by default when metadata, licence, scope, and fixture tests are present |
| Translation providers | Dictionary or generated candidates | T5; retained for audit/review and disabled for Flickr queries by default |

Curated source assertions and translation candidates are separate artifacts. Curated assertions are written to `source_name_assertions.parquet`; translation providers write `translation_candidates.parquet` and optional T5 evidence. The compiler keeps both auditable while preventing unreviewed generated names from becoming broad Flickr search traffic.

## Trust Tiers

| Tier | Meaning | Default query behavior |
|---|---|---|
| T1 | Accepted taxonomy and taxonomically verified scientific names | Query-eligible for species-level scientific names and accepted synonyms |
| T2 | Curated biodiversity vernacular names | May be query-eligible when source-backed, regional, and collision-free |
| T3 | Aggregator names with confident external taxon links | May be query-eligible after binding checks |
| T4 | Community or weakly curated names | Require review or corroboration |
| T5 | Dictionary or generated translations | Disabled for Flickr queries unless reviewed, corroborated, or backed by same-taxon language-source evidence |

Query eligibility is stricter than name enablement. A name can remain enabled as evidence and still have `query_eligible = false` with a concrete `query_disabled_reason`.

## Region And Caution Policy

Regional names keep their region in `names.parquet` and `flickr_query_definitions.parquet`. A vernacular name should only become query-eligible in the region where the source backs it. Names from taxonomically cautionary regions are query-ineligible unless the row carries explicit accepted-taxon resolution such as `lineage_check = accepted_taxon_key` or another confident same-taxon binding.

Collision checks block broad or risky terms:

- same normalized common name used by multiple accepted species;
- source rows mapped only to a genus, species complex, or ambiguous synonym;
- common words and broad group terms that are poor Flickr queries;
- disabled reasons containing ambiguity, collision, or taxonomic caution.

## Why ALA And Swedish Providers Are Not Included

ALA is not included in this implementation because the current Papilio demoleus work treats Australia/New Guinea as taxonomically cautionary. Adding ALA safely would require a dedicated adapter, reviewed taxon-resolution rules for the demoleus/sthenelus complex, licence/citation metadata, fixture tests, and QA that prevents cautionary Australian names from becoming global query terms.

Swedish/SLU/Artdatabanken providers are not included because they are outside the configured Papilio demoleus regional language target set and outside this implementation's source policy. Adding them later should follow the same adapter/static-source process below and must preserve regional scope, source licence, source IDs, and taxonomic binding evidence.

## Maintainer Guide For Future Sources

Use a Python API adapter when:

- the source has a stable API or structured downloadable dataset;
- rows need pagination, retry, rate-limit handling, or source-specific matching;
- taxon IDs, external links, or source snapshots must be preserved per request;
- incremental checkpointing is needed.

Use the static CSV loader when:

- the source is a curated snapshot without a stable API;
- the dataset is small enough to review and version in `data/source_snapshots/`;
- all rows can fit the static schema with source ID, accepted name, vernacular name, language, region, rank, licence, source URL, and citation.

Every new source needs:

- source key, display name, source type, source version, snapshot version;
- language, script, country/region scope, trust tier, precision tier;
- source IDs and source record IDs where available;
- licence and citation fields, even when the value is `not_configured`;
- fixture tests for successful mapping, ambiguous mapping, broad-rank rejection, disabled/caution rows, metadata preservation, and deterministic output;
- no live network in tests.

Do not add a source only to increase query volume. Add it when it improves provenance, regional coverage, or reviewability without weakening taxonomic identity.
