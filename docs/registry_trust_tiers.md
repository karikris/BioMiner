# Registry Trust Tiers

BioMiner's registry separates taxonomic identity, name evidence, and query eligibility. The GBIF accepted taxon key is the production identity for accepted taxa. Other sources can add evidence and aliases, but they do not replace the accepted spine.

## Tiers

```text
T1  accepted taxonomic spine and reviewed first-party registry assertions
T2  source-backed vernaculars or synonyms from trusted biodiversity sources
T3  source-backed article labels, aliases, or names requiring weaker interpretation
T4  unreviewed community/source enrichment that is useful for discovery but needs caution
T5  generated or machine-translated candidates
```

T1 and reviewed T2 assertions are the preferred basis for production search terms and evidence matching. T3 and T4 can support discovery when their source and review state make that safe for the current registry policy. T5 records are disabled by default.

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

Generated translations must remain:

```text
trust_tier = T5
enabled = false
review_state = needs_review
```

They can become query terms only after explicit review or independent corroboration by source-backed evidence. A generated name should never be treated as accepted vernacular evidence merely because it was useful for search expansion.

## Query Generation

Flickr query definitions are compiled only from enabled registry names. Tags and text searches remain separate atomic query definitions, and each definition preserves:

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
```

When Flickr rediscoveries hit the same photo, those query definitions are folded into the canonical source record provenance arrays. The registry remains the source of term provenance; Flickr title, tags, description, and comments are discovery or review evidence, not taxonomic authority.

## Review Policy

Comments and metadata may strengthen ambiguous records only when they match accepted scientific names, synonyms, or reviewed common-name evidence. They cannot promote records that visual triage has marked as hard negatives. Operational failures stay retryable and must not be converted into biological negatives.
