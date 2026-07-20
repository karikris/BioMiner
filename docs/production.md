# Production workflow

## Status

Production is reset to the [GBIF ground-zero intake](PIPELINE_GROUND_ZERO.md).
The three-member GBIF DWCA and its checksum-bound Zstandard Parquet
representation are the only current production input. No previous run, report,
cache, registry, Flickr acquisition, model output, or release artifact is part
of this cycle.

## Required production order

```text
immutable GBIF DWCA
  → source/member fingerprinting and schema validation
  → normalized taxonomy and source-bound name enrichment
  → species-first Flickr query planning and exhaustive bounded paging
  → media validation, rights handling, and provenance-preserving deduplication
  → YOLOE route/quality optimization for butterfly, moth, and other-insect domains
  → full-frame BioCLIP evidence across taxonomic ranks
  → review, quality, release, and downstream-handoff gates
```

Each arrow produces an immutable, versioned artifact and a manifest written
last. A later stage cannot infer completion from a local cache, a historical
report, or an unversioned file. Missing evidence remains unavailable; a failed
or incomplete stage cannot be represented as zero or success.

The DWCA-to-Parquet converter is a bounded physical intake operation. It keeps
all source fields as strings and does not normalize, enrich, filter, or score
records. See [the pipeline scope](PIPELINE_GROUND_ZERO.md) for scientific
boundaries and [storage handoffs](storage_handoffs.md) for completed-artifact
transfer rules.
