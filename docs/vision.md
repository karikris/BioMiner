# Vision and classification

YOLOE identifies butterfly and life-stage objects. Production preserves metadata for ineligible detections but does not create their crops. Eligible crops use profile-controlled padding, resize, batching, and memory limits; fine wing patterns use high-quality image resizing.

BioCLIP 2.5 runs as a persistent worker. Build text embeddings from reviewed
rank prompts with `biominer dev vision build-text-embedding-cache`, then pass
the Parquet artifact to production with `--taxonomy-text-embedding-cache`.
Production validates the cache against classification version, prompt version,
hierarchy fingerprint, complete staged-label set, model ID, model checkpoint,
embedding dimensions, and cache fingerprint. Image embeddings are batched and
compared with normalized cached text embeddings. A missing, stale, incomplete,
or model-mismatched cache fails the hierarchical production stage.

## Six-rank global cascade

The ordered production hierarchy is:

```text
FAMILY → SUBFAMILY → TRIBE → SUBTRIBE → GENUS → SPECIES
```

At each intermediate rank, the candidates are the deduplicated union of actual
nodes in the surviving reviewed leaf paths. Each node is ordered only by its
current-rank raw similarity: the mean normalized image/text embedding dot
product across that node's reviewed prompts. The fixed
`global_rank_top_k` beam retains at most 3 nodes across the whole rank. All
three may belong to one parent. It is not top 3 per parent, and no cumulative,
multiplied, or averaged cross-rank path score affects pruning.

SUBTRIBE remains a real position in the rank order. A path without a supported
subtribe survives only through a sourced, reviewed SUBTRIBE skip. Skip paths
are carried forward without a fabricated node, score, or beam position; actual
SUBTRIBE nodes in other active paths still compete for the same global top 3.

The global genus top 3 defines the species candidate universe. Species are
ordered solely by the species first-pass prompt score and truncated to top 20.
Exactly those candidates are scored again with the distinct species-rerank
prompt stage. The reranked top 5 and reported top 3 are persisted; the reranked
top 1 determines the final winning taxonomic path. Target injection is not
used for open classification.

Each crop records:

- `<rank>_top3`, `<rank>_top3_node_ids`, and `<rank>_top3_scores` for FAMILY through GENUS;
- rank-local `<rank>_top1`, `<rank>_top1_node_id`, `<rank>_top1_score`, and `<rank>_margin` fields;
- final winning-path `selected_<rank>`, `selected_<rank>_node_id`, and `selected_<rank>_score` fields;
- `species_top20` with first-pass scores, reranked `species_top5`, and reported `species_top3` with accepted GBIF keys;
- candidate, retained, and active-path counts by rank, plus the versioned pruning trace and reviewed skip reasons;
- `classification_version`, `prompt_version`, `classification_fingerprint`, `hierarchy_fingerprint`, and `embedding_cache_fingerprint`.

Rank-local top 1 and final selected path answer different questions. For
example, `family_top1` is the highest family raw similarity at the FAMILY step;
`selected_family` is the family containing the final reranked species winner
and may differ. A reviewed SUBTRIBE skip on the winning path leaves its selected
SUBTRIBE fields null and is recorded in `skipped_ranks`;
`fully_skipped_ranks` identifies a rank skipped by every active path at that
step.

Classifier output remains screening evidence. Open classifications enter `in_review`; GBIF and reviewed registry evidence define identity.

## Runtime profiles and benchmarks

Production visual settings flow from the selected runtime profile into detection, crop materialization, and BioCLIP. The Mac profile is `config/vision_profiles/mac_m5pro_64gb.json`.

Developer-only checks and benchmarks live under `biominer dev vision`:

```bash
uv run biominer dev vision bioclip-runtime-check --device mps
uv run biominer dev vision yoloe26-runtime-check --device mps
uv run biominer dev vision benchmark-plumbing --records 1000 --output-dir reports/vision_benchmarks/plumbing
uv run biominer dev vision benchmark-rolling-matrix --records 1000 --output-dir reports/vision_benchmarks/rolling
uv run biominer dev vision benchmark-cascade --output-dir reports/vision_benchmarks/cascade
```

Benchmarks measure plumbing, throughput, and deterministic cascade invariants,
not biological accuracy. Accuracy evaluation uses reviewed labels through
`biominer evaluation classify`.
