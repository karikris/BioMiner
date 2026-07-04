# Production Workflow

BioMiner has one production entry point for taxon-scoped work:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio_demoleus
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
- `compile_queries`: compiles deterministic Flickr text and tag query definitions from enabled registry names.
- `enqueue_flickr_work`: writes resumable Flickr work items into the workstore.
- `poll_flickr`: fetches Flickr metadata, stores raw JSON audit payloads when configured, and writes canonical source records.
- `detect_objects`: runs YOLOE/YOLO26-style object proposals over temporary image loads.
- `score_bioclip`: scores whole-image, detector-crop, and detector-crop-segmentation visual evidence with BioCLIP 2.5 when the requested mode is available.
- `join_evidence`: joins canonical records, detections, and BioCLIP scores into object and photo evidence outputs.
- `summarize`: writes run metrics, review queues, and report artifacts.

## Data Rules

Canonical source records are keyed by `source` and `flickr_photo_id`. Repeated discoveries of the same Flickr photo are folded into provenance arrays such as `text_search_terms`, `tag_search_terms`, and `all_query_labels`; duplicate discoveries do not create duplicate evidence rows.

Metadata filtering records metadata flags for review and routing. It does not perform final biological classification. Metadata flags are kept as evidence fields and cannot override hard-negative visual triage.

BioCLIP is screening evidence only. The GBIF accepted taxonomic spine remains the production taxonomic identity, and geography is a candidate prior rather than validation.

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

Production artifacts are written as immutable Parquet and JSON objects under the configured S3 prefix. Workers must write unique shard paths rather than append to shared cloud files.
