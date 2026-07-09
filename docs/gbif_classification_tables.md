# GBIF Classification Tables

Registry builds add compact artifacts for family-first BioCLIP classification. The accepted GBIF-backed registry remains the taxonomic authority. These tables are derived candidate-selection and prompt-label artifacts; they do not validate occurrences and they do not replace `taxa.parquet`, `names.parquet`, or registry QA.

## Artifacts

Registry builds emit these files by default:

```text
butterfly_classification_taxa.parquet
butterfly_family_labels.parquet
butterfly_species_labels.parquet
butterfly_classification_manifest.json
butterfly_classification_qa_findings.parquet
```

`butterfly_classification_taxa.parquet` is metadata-only. It keeps accepted species rows with GBIF species, family, and genus keys where available, plus enable/disable status for visual classification. It intentionally does not include unbounded synonym or vernacular arrays.

`butterfly_family_labels.parquet` contains a small set of BioCLIP-friendly family prompts, such as `a photo of a butterfly in the family Papilionidae`.

`butterfly_species_labels.parquet` contains a small set of species prompts, such as `a photo of Papilio demoleus` and field/close-up variants. These rows are designed for optional text-embedding caches and family-constrained top-20 species scoring.

The manifest records the classification table version, prompt variant version, source registry QA status, row counts, label counts, disabled species count, and expected artifacts. QA findings record fatal reference/schema issues and non-fatal warnings such as missing genus values.

## Build

A normal registry build writes the artifacts unless explicitly skipped:

```bash
uv run biominer registry build \
  --output-dir data/registry/version=2026-07-09 \
  --registry-version 2026-07-09
```

To rebuild only the classification artifacts from an existing registry:

```bash
uv run biominer registry build-classification-table \
  --registry-dir data/registry/current \
  --output-dir data/registry/current
```

Use `--skip-classification-table` only for compatibility builds that should omit visual candidate artifacts.

## Runtime Use

The hierarchical workflow shape is:

```text
YOLOE butterfly_like crop
-> BioCLIP top 3 butterfly families
-> select top family
-> BioCLIP top 20 species from that family only
-> rerank those 20 into top 5
```

`hierarchical_butterfly_classification` validates the classification taxa, family labels, and species labels, then uses them for open BioCLIP classification. Prompt-template scores are mean-aggregated by taxon. The species candidate pool is always restricted to the selected top family, and the reranker scores all first-pass top-20 species rather than only the first five. The mode never falls back to `target_scope_object_screening`.

Example command:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilionoidea_hierarchical \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table s3://biominer/biominer/registry/current \
  --device mps \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --delete-images-after-commit
```

The taxonomy store API can load a local registry directory:

```python
from biominer.bioclip.taxonomy_store import ButterflyTaxonomyStore

store = ButterflyTaxonomyStore.read("data/registry/current")
families = store.family_candidates()
papilionidae_species = store.species_for_family("gbif:9417")
labels = store.species_prompt_labels_for_family("gbif:9417")
```

## Scale

For roughly 18,000 butterfly species, the metadata-only taxa table should normally remain well under 0.1 GB. Prompt-label tables are also small because the current build uses a small set of family and species prompt templates. Optional text embedding caches may be larger, roughly 0.1-0.3 GB depending on prompt count, embedding dimension, and float16 versus float32 storage.
