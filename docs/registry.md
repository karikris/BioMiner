# Unified butterfly registry

BioMiner uses one CoL XR identity registry for taxonomy, BioCLIP routing, and Flickr discovery evidence. The pinned primary source is ChecklistBank dataset `315557`, release `COL26.6 XR`, DOI `10.48580/dgy8b`.

## BioCLIP paths

Every accepted species has one row in `species_paths.parquet` using the ranks BioCLIP supports:

```text
KINGDOM → PHYLUM → CLASS → ORDER → FAMILY → GENUS → SPECIES
```

When an intermediate source rank is absent, the nearest observed parent is carried forward as a routing proxy. Proxy rows retain `target_rank`, their true semantic rank, `candidate_kind=carry_forward_proxy`, and `proxy_source_node_id`. Prompt text uses the semantic rank and never asserts that the parent belongs to the missing rank. A proxy score is routing evidence, not taxonomy.

The classifier reads `species_paths.parquet` directly from `--registry-dir`.

## Canonical Flickr keywords

`names.parquet` retains every keyword/source/taxon association. Unicode NFKC normalization, case folding, whitespace collapse, and typographic punctuation normalization produce the request identity. Exactly one row per normalized term is canonical; lower-priority identical terms are marked `suppressed_duplicate` while retaining their associations.

Trust order is T1 through T5. A term's highest available trust becomes `effective_trust_tier`. `flickr_query_definitions.parquet` contains exactly two logical definitions per actionable normalized term: one `tags` request and one `text` request. Registry version, language, taxon, and source are association data and do not enter logical or physical request identity.

The persistent SQLite discovery ledger stores canonical keywords, keyword associations, logical queries, physical requests, query results, and relational photo-keyword evidence. Completed physical partitions are never recreated. Newly associated keyword IDs are backfilled from existing results without calling Flickr or re-queuing a known image.

## Published artifacts

`registry publish` validates complete species paths, unique canonical terms,
logical-query uniqueness, exact geographic schemas, geographic summary coverage
for every accepted species, and fatal QA. Geographic build QA is merged into the
registry QA table, and geographic occurrence provenance is merged into the
registry source-snapshot table. Publication then emits:

```text
taxa.parquet
species_paths.parquet
names.parquet
flickr_query_definitions.parquet
source_snapshots.parquet
qa_findings.parquet
taxon_geographic_spread.parquet
taxon_geographic_summary.parquet
manifest.json
```

The geographic build staging directory must also contain
`geographic_occurrence_evidence.parquet`, `geographic_qa_findings.parquet`,
`geographic_spread_manifest.json`, and `geographic_summary_manifest.json`.
These support validation and provenance merging but are not copied into the
runtime registry.

`manifest.json` is written last and inventories every published Parquet file by
row count, byte count, and SHA-256 checksum. A summary row marked
`data_deficient` is valid. A species with no spread rows is also valid: missing
geographic evidence is unknown evidence, not a hard-negative range assertion.

## Commands

```bash
uv run biominer registry build --output-dir data/registry/build --registry-version COL26.6-XR
uv run biominer registry audit --registry-dir data/registry/build --report-dir reports
uv run biominer registry publish --registry-dir data/registry/build --output-dir data/registry/current --replace-existing
```

The retired classification-v3 staged-rank embedding-cache command is preserved
only in history. Current reference and Flickr embeddings are built by their
own full-frame, content-addressed contracts.
