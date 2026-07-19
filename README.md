# BioMiner

BioMiner builds a CoL XR-grounded butterfly identity registry, discovers
Flickr metadata without duplicating requests, constructs geography-conditioned
reference evidence, routes unsuitable visual material, and screens eligible
full-frame images with BioCLIP 2.5. Model output remains screening evidence,
never taxonomic validation.

## Adaptive GBIF reference fast-start

The production reference default is `adaptive_gbif_fast_start`. Qualifying GBIF
provider assertions may enter a provisional support bank after deterministic
taxon, rights, decode, duplicate, independence and YOLOE route checks. Reference
human review does not block the first provisional BioCLIP comparison. A
species-level probability audit using source-bound human-reviewed Flickr labels
later targets reference review and selective reruns only where evidence
warrants it.

Provider-asserted support is not human verification, raw similarity and margin
values are not probabilities, and every final Flickr occurrence still requires
source-bound human review. `human_verified_strict` remains available for
publication-critical work. See the [fast-start workflow](docs/adaptive_gbif_fast_start.md)
and [migration guide](docs/migrations/adaptive-gbif-reference-default.md).

## Registry and discovery

The primary identity source is CoL XR dataset `315557`, release `COL26.6 XR`, DOI `10.48580/dgy8b`. Each accepted species has one BioCLIP-supported path:

```text
KINGDOM → PHYLUM → CLASS → ORDER → FAMILY → GENUS → SPECIES
```

Missing ranks use semantic-parent routing proxies. `names.parquet` retains every taxon/source association while selecting one canonical normalized term. Flickr executes one logical query per normalized term and field (`tags` or `text`) and stores all many-to-many evidence in SQLite.

## Geography-conditioned dynamic reference pooling

The adaptive production workflow does not use family or geography as a hard
identity gate. It builds a complete candidate union, preserving the configured
target and safety candidates, then combines:

```text
family retrieval evidence + regional candidate evidence
→ complete target-preserving candidate union
→ diverse global reference safety pool
→ local reference pool, or an explicit local-unavailable reason
→ complete raw global/local score components
→ provisional fusion, review sampling and statistical audit
```

Family evidence accelerates retrieval and batching; it cannot catastrophically
prune the target. Geography prioritizes candidate and reference evidence; it
does not certify identity, and missing geography is not biological absence.
The target-aware route uses the canonical full frame and persists one immutable
embedding per media identity. Changing pool membership reuses that embedding
rather than re-encoding the image.

Raw similarities, margins, component scores, and fusion values are not
probabilities. Human verification, calibration, statistical support,
release-ready maturity, and downstream publication remain separate gates. The
bounded fixture pilot evaluated all 24 declared candidate/pool/fusion variants
but selected no production default because reviewed precision, subgroup
support, comparable runtime, and MPS evidence are unavailable. See the
[pilot report](reports/geo_dynamic_pooling/pilot/geography_conditioned_pooling_report.md).

The historical family/genus cascade, 0.90 genus shortcut, crop materialization,
and bucketed visual modes remain explicit `legacy` or compatibility paths for
existing artifacts. They are not the adaptive production default and must not
silently control adaptive output.

Durable tables are Parquet, analytical QA uses DuckDB, and local operational
state uses SQLite. Production defaults to S3-compatible storage and a
PostgreSQL workstore. Downloaded images are temporary and deleted only after
durable commit unless a debug option explicitly retains them.

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

uv run biominer run --help
uv run biominer references --help
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
uv run biominer run --help
uv run biominer references --help
```

Tests use fake clients and classifiers and do not require Flickr calls or model downloads.

See [registry documentation](docs/registry.md), [production workflow](docs/production.md),
[operation-efficient storage handoffs](docs/storage_handoffs.md), and the
[dynamic-pooling architecture](docs/architecture/geography_conditioned_dynamic_pooling.md).
