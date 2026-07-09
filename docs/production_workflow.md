# Production Workflow

BioMiner has one production entry point for taxon-scoped work:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio_demoleus \
  --vision-profile mac_m5pro_64gb \
  --classification-mode target_scope_object_screening \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --device mps
```

`--rank` accepts `auto`, `family`, `genus`, or `species`. The run resolver reads the registry, resolves the accepted taxon, expands the species scope for family and genus runs, and writes that scope into the run manifest. Broad seed search is not a production mode; query work is compiled from the versioned registry.

## Default Backends

Production defaults are S3-compatible artifact storage and a Postgres workstore:

```text
--storage-backend s3
--workstore-backend postgres
```

Local filesystem and SQLite are explicit development overrides only:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix staging/runs/papilio_demoleus \
  --storage-backend local \
  --workstore-backend sqlite \
  --dry-run
```

## Stages

The orchestrator records every stage in the manifest and can run a subset with `--stages` when debugging. The production stage order is:

```text
resolve_taxon_scope
build_registry
compile_queries
enqueue_flickr_work
poll_flickr
detect_objects
score_bioclip
join_evidence
summarize
```

Stage responsibilities:

- `resolve_taxon_scope`: resolves the input family, genus, or species against `taxa.parquet`.
- `build_registry`: records registry availability and version metadata for the selected taxon scope.
- `compile_queries`: compiles deterministic Flickr text and tag query definitions from enabled, query-eligible registry names; generic/collided/generated terms remain retained evidence unless review or corroboration makes them eligible.
- `enqueue_flickr_work`: writes resumable Flickr work items into the workstore.
- `poll_flickr`: fetches Flickr metadata, stores raw JSON audit payloads when configured, and writes canonical source records.
- `detect_objects`: runs YOLOE/YOLO26-style object proposals over temporary image loads.
- `score_bioclip`: scores BioCLIP 2.5 detector crops for `butterfly_like` detections. Whole-image and segmentation modes are explicit ablation/debug modes, not the production default.
- `join_evidence`: joins canonical records, detections, and BioCLIP scores into object and photo evidence outputs.
- `summarize`: writes run metrics, review queues, and report artifacts.

## Data Rules

Canonical source records are keyed by `source` and `flickr_photo_id`. Repeated discoveries of the same Flickr photo are folded into provenance arrays such as `text_search_terms`, `tag_search_terms`, and `all_query_labels`; duplicate discoveries do not create duplicate evidence rows.

Metadata filtering records metadata flags for review and routing. It does not perform final biological classification. Metadata flags are kept as evidence fields and cannot override hard-negative visual triage.

BioCLIP is screening evidence only. The GBIF accepted taxonomic spine remains the production taxonomic identity, and geography is a candidate prior rather than validation.

`target_scope_object_screening` is the default classification mode. It scores detector crops against the run taxon scope and registry candidate-set context for target support screening. Output fields named `family_top3`, `species_top20`, and `species_top5` are screening fields, not taxonomic validation.

Registry builds now emit GBIF-derived classification artifacts by default:

```text
butterfly_classification_taxa.parquet
butterfly_family_labels.parquet
butterfly_species_labels.parquet
butterfly_classification_manifest.json
butterfly_classification_qa_findings.parquet
```

These files are derived from the accepted registry and are candidate-selection inputs only. `hierarchical_butterfly_classification` now uses them for open classification: family top 3 across configured butterfly families, selected top family, species top 20 restricted to that family, and species top 5 reranked from all first-pass top-20 species. It never injects the run target species into hierarchical reranking and it keeps open-classification evidence separate from target-scope support.

The Mac M5 Pro / 64 GB production profile is `mac_m5pro_64gb`. It uses `device=mps`, YOLOE checkpoint `yoloe-26s-seg.pt`, YOLO image size `768`, detector batch size `16`, BioCLIP crop batch size `24`, crop target `336`, crop padding `0.08`, zstd Parquet part files, and delete-after-commit cached image cleanup. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running MPS sidecars.

For family or superfamily hierarchical runs, `--taxonomy-text-embedding-cache` is an optional performance input. The cache is accepted only when its classification table version, prompt variant version, BioCLIP model/checkpoint metadata, label hash, embedding dimension, and dtype match the requested taxonomy labels. Direct prompt scoring remains available when the cache is absent.

Adaptive batching is opt-in with `--adaptive-batching`. It is intended for memory pressure on YOLOE or BioCLIP batches and must not be used to hide non-memory failures.

Local hierarchical command shape:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir data/registry/current \
  --output-prefix runs/local_debug/papilionoidea_hierarchical \
  --storage-backend local \
  --workstore-backend sqlite \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table data/registry/current \
  --device mps \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --delete-images-after-commit
```

Cloud/S3 hierarchical command shape:

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

## Outputs

Run manifests include:

```text
run_id
storage_backend
workstore_backend
taxon_scope
model_configs
query_counts
detection_counts
bioclip_counts
evidence_counts
stage statuses
output artifact URIs
```

Production artifacts are written as immutable Parquet and JSON objects under the configured S3 prefix. Workers must write unique shard paths rather than append to shared cloud files. Visual stages write zstd parts such as:

```text
evidence/stage=detect_objects/run_id=<run_id>/worker=<worker_id>/part=<part_id>.parquet
evidence/stage=score_bioclip/run_id=<run_id>/worker=<worker_id>/part=<part_id>.parquet
```

The workstore registers only successfully written parts. Cached Flickr images are deleted after committed detection, score, and evidence outputs; failed score or part writes leave the cached image retryable.

Vision stages also write compact reports:

```text
vision_stage_metrics.json
vision_stage_summary.md
```

These reports expose detector skip counts, eligible BioCLIP detections, selected family counts, species top-1 counts, batching settings, adaptive retries, throughput estimates, and cache-use flags. `bioclip_counts.objects_scored` should remain tied to eligible `butterfly_like` detections rather than all canonical source records.

## Evaluation And QA Reports

Production summaries write review queues as run artifacts. Local runs write `reports/review_queue.parquet`; cloud runs write immutable review-queue shards and record them in the manifest. These rows are review priorities, not taxonomic truth or occurrence publication records.

Evaluate a completed hierarchical run against human-reviewed labels:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical
```

The report writer emits `evaluation_metrics.json`, family/species confusion matrices, `calibration_bins.parquet`, `review_error_examples.parquet`, and `evaluation_summary.md`. Add `--write-charts` for local PNG charts. BioCLIP scores are candidate-set-relative, so calibration is a review-prioritisation signal rather than absolute biological confidence.

Build a local standalone review queue when inspecting artifacts outside a full production `summarize` run:

```bash
uv run biominer evaluation review-queue \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --photo-summary runs/local_debug/papilionoidea_hierarchical/photo_evidence_summary.parquet \
  --output reports/review_queue.parquet
```

Run the Xie-style metrics profile against the same BioMiner outputs:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical \
  --evaluation-profile xie_style_metrics_only
```

Xie-style is a metrics profile only. It does not change the production architecture, does not score all images with BioCLIP, and does not replace GBIF-derived candidate scope. Human-reviewed labels are required for real accuracy claims; synthetic fixtures only validate metric and QA logic.
