# Production Workflow Examples

These examples use the production defaults: S3-compatible artifact storage and a Postgres workstore. Run them from `./BioMiner`.

Set the required environment before live runs:

```bash
export BIOMINER_STORAGE_BACKEND=s3
export BIOMINER_WORKSTORE_BACKEND=postgres
export BIOMINER_S3_ENDPOINT_URL="https://s3.<region>.backblazeb2.com"
export BIOMINER_S3_ACCESS_KEY_ID="<access-key-id>"
export BIOMINER_S3_SECRET_ACCESS_KEY="<secret-access-key>"
export BIOMINER_S3_REGION="<region>"
export BIOMINER_S3_BUCKET="biominer"
export BIOMINER_S3_PREFIX="biominer"
export BIOMINER_WORKSTORE_DSN="postgresql://user:password@host:5432/postgres"
export BIOMINER_WORKER_ID="worker-001"
export FLICKR_API_KEY="<flickr-api-key>"
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

`FLICKR_SECRET_KEY` can be present for future signed Flickr operations, but current metadata polling uses `FLICKR_API_KEY`.

## Mac M5 Pro Runtime Examples

Install the optional Python 3.12 sidecar runtimes outside the main Python 3.14 environment:

```bash
bash scripts/setup_yoloe26_user_py312.sh
bash scripts/setup_bioclip25_user_py312.sh
```

Check the YOLOE-26 and BioCLIP sidecars on Apple MPS:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision yoloe26-runtime-check \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device mps

PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --device mps
```

Run the integrated local detector-first screen:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer vision screen \
  --input runs/local_debug/papilio_demoleus/canonical_source_records.parquet \
  --output-dir runs/local_debug/papilio_demoleus/vision_screen \
  --species-context runs/local_debug/papilio_demoleus/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --delete-images-after-commit
```

Run the production detector-first workflow:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio_demoleus \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --delete-images-after-commit
```

## Species Run

Use `--rank species` when the requested taxon is one accepted species. The run expands to one species context and compiles Flickr query definitions from enabled registry names for that species.

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio_demoleus \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --delete-images-after-commit
```

## Genus Run

Use `--rank genus` when the requested taxon should expand to every species under the accepted genus in the registry.

```bash
uv run biominer run \
  --taxon "Papilio" \
  --rank genus \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --delete-images-after-commit
```

## Family Run

Use `--rank family` when the requested taxon should expand to every species under the accepted family in the registry.

```bash
uv run biominer run \
  --taxon "Papilionidae" \
  --rank family \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilionidae \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-backend yoloe26 \
  --vision-profile mac_m5pro_64gb \
  --device mps \
  --bioclip-model hf-hub:imageomics/bioclip-2.5-vith14 \
  --delete-images-after-commit
```

## Bounded Smoke Run

Use limits during first live checks. Limits are operational bounds, not changes to query semantics.

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir s3://biominer/biominer/registry/current \
  --output-prefix s3://biominer/biominer/runs/papilio_demoleus_smoke \
  --storage-backend s3 \
  --workstore-backend postgres \
  --vision-profile mac_m5pro_64gb \
  --limit-records 10
```

Local filesystem and SQLite are only explicit development overrides. Use `--storage-backend local --workstore-backend sqlite --dry-run` for parser or manifest checks that must not touch production services.
