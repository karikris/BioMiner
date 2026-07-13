# BioMiner

BioMiner builds a CoL XR-grounded butterfly identity registry, discovers Flickr metadata without duplicating requests, removes non-butterfly material, and screens eligible images with BioCLIP 2.5. Model output remains screening evidence, never taxonomic validation.

## Registry and discovery

The primary identity source is CoL XR dataset `315557`, release `COL26.6 XR`, DOI `10.48580/dgy8b`. Each accepted species has one BioCLIP-supported path:

```text
KINGDOM → PHYLUM → CLASS → ORDER → FAMILY → GENUS → SPECIES
```

Missing ranks use semantic-parent routing proxies. `names.parquet` retains every taxon/source association while selecting one canonical normalized term. Flickr executes one logical query per normalized term and field (`tags` or `text`) and stores all many-to-many evidence in SQLite.

## BioCLIP butterfly funnel

Visual classification follows this bounded funnel:

```text
family top 1
→ genera within that family top 20
→ genus top 3
→ species beneath those genera top 20
→ distinct-prompt species top 5
→ final species top 1
```

If genus top-1 confidence is strictly above 0.90, the guardrail routes species through that one genus. Otherwise the broader genus top-20 then top-3 branch remains active. Every shortlist, score, margin, count, and routing decision is recorded.

Durable tables are Parquet, analytical QA uses DuckDB, and local operational state uses SQLite. Downloaded images are temporary and deleted after durable classification unless a debug option explicitly retains them.

## Quick start

```bash
uv sync
uv run biominer registry build \
  --output-dir data/registry/build \
  --registry-version COL26.6-XR

uv run biominer registry audit \
  --registry-dir data/registry/build \
  --report-dir reports

# After the geographic spread and summary builders populate this build directory:
uv run biominer registry publish \
  --registry-dir data/registry/build \
  --output-dir data/registry/current \
  --replace-existing

uv run biominer dev vision build-text-embedding-cache \
  --registry-dir data/registry/current \
  --output data/cache/taxonomy/current/classification_text_embeddings.parquet

uv run biominer bioclip screen --help
```

`registry publish` requires completed geographic spread and summary builds. It
merges geographic provenance and QA into the registry audit tables, then emits
`taxa.parquet`, `species_paths.parquet`, `names.parquet`,
`flickr_query_definitions.parquet`, `source_snapshots.parquet`,
`qa_findings.parquet`, `taxon_geographic_spread.parquet`,
`taxon_geographic_summary.parquet`, and `manifest.json`. The manifest is written
last and contains row counts, byte counts, and checksums for every published
Parquet artifact. A data-deficient species or an empty spread is publishable;
absence of geographic evidence means unknown, never taxonomic or biological
absence.

## Verification

```bash
uv run pytest -q
uv run biominer --help
uv run biominer bioclip screen --help
```

Tests use fake clients and classifiers and do not require Flickr calls or model downloads.

See [registry documentation](docs/registry.md), [production workflow](docs/production.md), and the [unified-registry cutover](docs/migrations/unified-registry.md).
