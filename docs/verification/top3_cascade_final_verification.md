# Six-rank cascade final repository verification

Date: 2026-07-11. Confidence: high.

This verification ran from the Phase 8 Task 8.5 working tree based on
`8121c32`. All data was synthetic and model-free; no Flickr request, model
weight, CUDA device, or MPS device was used.

## Repository checks

| Check | Result |
|-|-|
| `git diff --check` | passed |
| `uv run ruff check .` | passed |
| `.venv/bin/pytest -q` | 986 passed in 30.72 seconds |
| Removed-hierarchy scan of authoritative docs | no old five-rank production hierarchy found |
| Cumulative-pruning scan outside the historical benchmark | no production match found |
| Removed-module/config scan | no five-rank or classification-v2 production match found |
| Removed-field/config scan | no `genus_top8`, `family_top_k`, or `DEFAULT_FAMILY_TOP_K` production match found |

The historical cumulative selector remains isolated in
`src/biominer/benchmarks/path_cascade.py`; it exists only to prove that the old
and required algorithms diverge.

## Classification-v3 artifact build

A temporary accepted GBIF fixture containing Papilionidae, Papilio, and
Papilio demoleus was built with:

```bash
uv run biominer registry build-classification \
  --registry-dir <tmp>/base \
  --output-dir <tmp>/classification-v3 \
  --source-json config/taxonomy/papilionoidea_classification_v3.json
```

The promoted manifest and `PathTaxonomyStore.read(...)` validation reported:

- `classification_version`: `butterfly-classification-v3.0.0`;
- `prompt_version`: `butterfly-six-rank-prompts-v4`;
- rank order: `FAMILY, SUBFAMILY, TRIBE, SUBTRIBE, GENUS, SPECIES`;
- accepted/mapped species: 1/1;
- reviewed SUBTRIBE skips: 1;
- fatal findings: 0;
- warnings: 1 (`optional_subtribe_skipped`);
- QA status: `passed`;
- every declared artifact checksum and hierarchy/classification fingerprint
  passed store validation.

The object-score schema contains all 123 shared and cascade columns. Every
field inherited from `PATH_CASCADE_OUTPUT_SCHEMA` has the exact canonical
physical dtype. The output identity is
`butterfly-cascade-output-v1.0.0`; old `genus_top8`, accepted-family-key,
generic species-score, and duplicate JSON audit aliases are absent.

## Hierarchical dry run

The local dry run used separate base, overlay, cache, and run roots and
resolved the Papilionidae family scope to one accepted species. It completed
with the fixed production configuration:

- `beam_strategy = global_rank_top_k`;
- `rank_beam_width = 3`;
- `species_first_pass_top_k = 20`;
- `species_rerank_top_k = 5`;
- `species_report_top_k = 3`.

Dry run verified parsing, scope resolution, path propagation, and the emitted
manifest. By contract it recorded, but did not open, the intentionally absent
temporary embedding-cache path.

## Model-free benchmarks

`benchmark-rolling-matrix --records 1000` completed with status `ok` across
72 bounded-worker/configuration variants. Its observed total time was 0.057319
seconds; this synthetic control-flow timing is not a model-throughput claim.

`benchmark-cascade` completed with status `ok` and verified:

- seven FAMILY candidates and a global rank width of three;
- reviewed optional-SUBTRIBE skip handling;
- three retained genera defining 75 species candidates;
- species first pass `75 → 20`;
- distinct-prompt rerank `20 → 5`;
- the versioned current-rank pruning trace;
- divergence from the retired cumulative-path selector.

The cascade benchmark's observed time was 0.313154 seconds. It is a synthetic
algorithm/contract benchmark, not biological evidence or model performance.
