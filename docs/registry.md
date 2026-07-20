# Taxonomy and keyword enrichment

No registry is currently published. Registry production begins only after the
GBIF three-member DWCA has been fingerprinted and validated; see
[GBIF ground-zero pipeline](PIPELINE_GROUND_ZERO.md).

The future registry must retain, rather than collapse, the evidence required to
derive Flickr discovery terms:

- GBIF taxon and occurrence identities;
- accepted scientific names, synonyms, spelling variants, and rank paths;
- source-bound common names from iNaturalist, Wikimedia, Catalogue of Life,
  and other documented providers;
- language, region, retrieval date, provider/source identifier, trust decision,
  and homonym/query-risk state for every name assertion; and
- the logical query definition and every source-name association.

Flickr discovery starts with eligible species-level terms and expands through
the approved enrichment set. It preserves every logical query association even
when a physical API request or returned photo is deduplicated. A name is
retrieval evidence, not a taxonomic label for an image.

Published registry artifacts, if and when produced, must be immutable Parquet
tables with explicit schemas, counts, checksums, provenance, and a manifest
written last. They must not reuse historical staging outputs.
