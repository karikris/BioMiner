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

`target_scope_object_screening` is the current default classification mode. It scores detector crops against the run taxon scope and registry candidate-set context for target support screening. Output fields named `family_top3`, `species_top20`, and `species_top5` are therefore screening fields, not proof of a family-first classifier. They are not yet constrained by `family_top1`; the current rerank metadata records the target-screening strategy and top-k settings used for the row.

`hierarchical_butterfly_classification` is reserved for the later GBIF classification-table workflow. Dry-run planning can record this mode and the future `--taxonomy-candidate-table`, but non-dry `score_bioclip` fails clearly rather than running target-scope screening silently.

The Mac M5 Pro / 64 GB production profile is `mac_m5pro_64gb`. It uses `device=mps`, YOLOE checkpoint `yoloe-26s-seg.pt`, YOLO image size `768`, detector batch size `16`, BioCLIP crop batch size `24`, crop target `336`, crop padding `0.08`, zstd Parquet part files, and delete-after-commit cached image cleanup. Set `PYTORCH_ENABLE_MPS_FALLBACK=1` when running MPS sidecars.

Reserved future hierarchical command shape:

```bash
uv run biominer run \
  --taxon "Papilionoidea" \
  --rank family \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilionoidea_hierarchical \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --classification-mode hierarchical_butterfly_classification \
  --taxonomy-candidate-table s3://biominer/biominer/registry/current/butterfly_classification_taxa.parquet \
  --family-top-k 3 \
  --species-first-pass-top-k 20 \
  --species-rerank-top-k 5 \
  --device mps \
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
