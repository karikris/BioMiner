# Registry Trust Tiers

BioMiner's registry separates taxonomic identity, name evidence, and query eligibility. The GBIF accepted taxon key is the production identity for accepted taxa. Other sources can add evidence and aliases, but they do not replace the accepted spine.

## Tiers

```text
T1  accepted taxonomic spine and reviewed first-party registry assertions
T2  source-backed vernaculars or synonyms from trusted biodiversity sources
T3  Wikidata labels and aliases with confident external taxon links
T4  unreviewed community/source enrichment that is useful for discovery but needs caution
T5  generated or machine-translated candidates
```

T1 and reviewed T2 assertions are the preferred basis for evidence matching. T3 assertions can support discovery only when the external taxon linkage is confident or the assertion has been explicitly accepted by review. T4 can support discovery when its source and review state make that safe for the current registry policy. T5 generated and dictionary translations are retained as low-trust registry name evidence for audit and matching, but they do not replace the GBIF accepted taxonomic spine and do not become Flickr queries unless separately query-eligible.

## Enablement

Name records should carry at least:

```text
display_name
language
script
source
trust_tier
precision_tier
confidence
review_state
enabled
licence
source_taxon_key or accepted_taxon_key
registry_version
```

Generated translations are retained as:

```text
trust_tier = T5
enabled = true
review_state = accepted
```

They are accepted registry name evidence for discovery and matching, with their source and confidence retained for audit. A generated name must not replace the accepted GBIF taxon identity.

## Query Generation

Flickr query definitions are compiled from names where `enabled = true` and `query_eligible = true`. Tags and text searches remain separate atomic query definitions, and each definition preserves:

```text
query_definition_id
search_field
term
language
accepted_taxon_key
family_key
genus_key
species_key
registry_version
source evidence fields
review_state
query_eligible
query_disabled_reason
species_specificity_score
```

T5 rows keep `trust_tier = T5` and `name_class = generated_translation` in `names.parquet`. They appear in `flickr_query_definitions.parquet` only after manual review, corroboration, or exact same-taxon language-source support makes them query-eligible.
Registry manifests record `enabled_t5_name_rows`, `t5_query_definition_rows`, and query-eligible/query-ineligible name counts so retained low-trust evidence remains auditable without becoming broad Flickr search traffic.

When Flickr rediscoveries hit the same photo, those query definitions are folded into the canonical source record provenance arrays. The registry remains the source of term provenance; Flickr title, tags, description, and comments are discovery or review evidence, not taxonomic authority.

## Review Policy

Comments and metadata may strengthen ambiguous records only when they match accepted scientific names, synonyms, or reviewed common-name evidence. They cannot promote records that visual triage has marked as hard negatives. Operational failures stay retryable and must not be converted into biological negatives.
