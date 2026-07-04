# BioMiner

BioMiner is a taxonomically grounded Flickr butterfly-discovery and image-triage pipeline.

It builds a reviewed multilingual butterfly name registry, compiles deterministic Flickr search definitions, fetches Flickr metadata, records metadata review flags, finds candidate objects with YOLOE/YOLO26-style coarse detectors, scores temporary whole images and crops with BioCLIP 2.5, assigns evidence buckets, and uses targeted Flickr comment review to strengthen ambiguous records.

BioMiner separates three forms of evidence:

- **taxonomic identity** from the versioned Step 0 registry;
- **discovery evidence** from Flickr queries and metadata;
- **screening evidence** from BioCLIP and comment review.

BioCLIP output is screening evidence only. BioMiner does not claim taxonomic validation, does not publish verified Darwin Core occurrences, and does not keep a permanent Flickr image archive.

## Pipeline

```text
Step 0: taxonomic registry
  GBIF accepted spine
  + synonyms and vernacular names
  + optional CoL/iNaturalist/ITIS/EOL evidence
  + reviewed translation candidates
  -> versioned registry Parquet
  -> atomic Flickr tags/text query definitions

Step 1: Flickr metadata discovery
  registry query definitions
  -> fixed upload-date slices
  -> Flickr photos.search metadata
  -> one canonical evidence row per photo with folded query-term provenance

Step 2: metadata flagging
  metadata/evidence Parquet
  -> preserve review flags for obvious non-biodiversity and hard-negative hints

Step 3: detector-first BioCLIP screening
  temporary image download
  -> YOLOE/YOLO26 coarse object proposals
  -> BioCLIP 2.5 whole-image, detector-crop, and segmentation-crop scoring
  -> Gold/Silver/Bronze/Bin/InReview
  -> delete temporary image

Step 4: targeted comment review
  Bronze/ambiguous records
  -> matching species/synonym evidence
  -> promotion where all bucket rules are satisfied
```

## Object-Level BioCLIP Pipeline

BioMiner's production visual path is detector-first and object-evidence based:

```text
canonical source records
  -> object detections with source + flickr_photo_id join keys
  -> ephemeral detector crops
  -> BioCLIP object scores against SpeciesContext/candidate-set labels
  -> whole-image, detector-crop, and detector-crop-segmentation rows when masks are available
  -> object_evidence_joined.parquet and photo_evidence_summary.parquet
  -> reports/review_queue.parquet for Bronze and InReview photo summaries
```

The core Python 3.14 environment keeps heavy vision dependencies optional. `vision detect --backend fake` is available for offline tests and deterministic local plumbing. YOLOE-26 and SAM/SAM2-style adapters are lazy-loaded from optional vision environments and fail with clear runtime errors when their dependencies are absent. The YOLOE-26 prototype treats YOLOE only as an object finder; BioCLIP 2.5 Huge remains the biological classifier and species scorer.

Example command shape:

```bash
uv run biominer vision detect \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --output staging/species_runs/example/object_detections.parquet \
  --backend fake

uv run biominer vision detect \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --output staging/species_runs/example/object_detections_yoloe26.parquet \
  --backend yoloe26 \
  --runtime-python "../YOLO26/venv/bin/python" \
  --checkpoint yoloe-26s-seg.pt \
  --device auto \
  --conf 0.20 \
  --iou 0.50 \
  --max-det 8

uv run biominer vision score \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --detections staging/species_runs/example/object_detections.parquet \
  --species-context staging/species_runs/example/species_context.json \
  --output staging/species_runs/example/object_bioclip_scores.parquet \
  --ablation-mode detector_crop

uv run biominer vision ablate \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --detections staging/species_runs/example/object_detections.parquet \
  --species-context staging/species_runs/example/species_context.json \
  --output-dir staging/species_runs/example/ablations \
  --modes whole_image,detector_crop,detector_crop_segmentation

uv run biominer evidence join \
  --input staging/species_runs/example/canonical_source_records.parquet \
  --detections staging/species_runs/example/object_detections.parquet \
  --scores staging/species_runs/example/object_bioclip_scores.parquet \
  --joined-output staging/species_runs/example/object_evidence_joined.parquet \
  --photo-summary-output staging/species_runs/example/photo_evidence_summary.parquet \
  --species-context staging/species_runs/example/species_context.json
```

Object-level tables are not standalone silos. Every detection and score row keeps `source`, `flickr_photo_id`, and object-level `detection_id`/`crop_hash` where applicable. Geography is recorded as a soft prior and can route strong visual conflicts to review; it is not an absolute discard rule.

For the combined YOLOE-26 plus BioCLIP detector-crop prototype, see `docs/yoloe26_prototype.md`.

The active package is `biominer` under `src/`. Run commands through:

```bash
uv run biominer
```

## Production Docs

- `docs/production_workflow.md` describes the single `biominer run` workflow, rank handling, stages, and production data rules.
- `docs/registry_trust_tiers.md` defines registry trust tiers and generated-name enablement policy.
- `docs/vision_workflow.md` documents the public vision commands, dev-only runtime utilities, and YOLOE/YOLO26 object-finder boundary.
- `docs/storage_postgres_s3.md` summarizes S3 artifact storage, Postgres workstore state, and explicit local overrides.
- `docs/deprecated_removed_commands.md` records removed public commands and their replacements.
- `examples/production_workflow.md` gives species, genus, and family `biominer run` examples using S3/Postgres defaults.

## Core invariants

1. **Step 0 is mandatory before Flickr query generation.**
2. **GBIF accepted taxon keys define the production taxonomic spine.**
3. **Deduplicate evidence rows by Flickr photo ID and fold discovery terms into canonical provenance arrays.**
4. **Tags and text searches are separate atomic query definitions.**
5. **Flickr metadata is retained even when an image is never downloaded.**
6. **Downloaded Flickr images are temporary and deleted after classification.**
7. **BioCLIP is screening evidence, not taxonomic authority.**
8. **Generated translations are disabled until reviewed or independently corroborated.**
9. **Fatal registry QA blocks promotion to `data/registry/current`.**
10. **Long-running API work must be resumable, bounded, and observable.**

## Repository status

The active implementation covers:

- GBIF-backed Papilionoidea registry construction;
- seven pinned butterfly-family identities;
- accepted family, genus, and species traversal;
- species synonyms and vernacular names;
- bounded concurrent species enrichment;
- retry, progress, and checkpoint controls;
- deterministic atomic Flickr query compilation;
- fixed upload-date Flickr search slicing;
- bounded metadata polling with a shared SQLite ledger;
- in-memory metadata text hints for review evidence, without a path-backed anti-keyword config;
- temporary image caching and deletion;
- BioCLIP 2.5 register-based classification helpers;
- evidence buckets and category/life-stage fields;
- targeted Flickr comment review;
- compact JSON, Markdown, Parquet, and DuckDB-based QA.

Supplementary Catalogue of Life, iNaturalist, ITIS, EOL, and generated-translation adapters remain lower-trust enrichment layers and must not replace GBIF accepted identities.

## Repository layout

```text
src/biominer/
  cli.py
  registry/
    build.py
    compiler.py
    gbif.py
    gbif_source.py
    scope.py
    audit.py
    ...
  run/
  evidence/
  detection/
  flickr_fetch/
  filter/
  bioclip/
  flickr_comments/
  common/
  storage/
  reports/

config/
  butterfly_scope.json
  vision_profiles/
  ...

tests/
data/
staging/
reports/
logs/
run/
```

Generated data, raw payloads, registry outputs, model files, downloaded images, local databases, credentials, and caches must not be committed.

## Production Storage Status

BioMiner separates durable artifact storage from operational work state:

- S3-compatible storage for Parquet shards, manifests, reports, and raw-response audit payloads;
- Postgres workstore rows for queues, API-call ledgers, completed keys, run state, shard inventory, and resume state;
- explicit local filesystem and SQLite backends for tests and isolated development only.

Production defaults are `BIOMINER_STORAGE_BACKEND=s3` and `BIOMINER_WORKSTORE_BACKEND=postgres`. Local filesystem + SQLite must be selected explicitly with local/dev config or `--storage-backend local --workstore-backend sqlite`. Workers write unique immutable Parquet shards such as `evidence/stage=poll_once/run_id=<run_id>/worker=<worker_id>/batch=<batch_id>.parquet`. See `docs/cloud_storage.md`.

## Requirements

- Python 3.14 or newer;
- `uv`;
- network access for live GBIF or Flickr work;
- Flickr API key only for commands that call Flickr;
- BioCLIP/OpenCLIP/PyTorch runtime only for Step 3.

The standard GIL-enabled CPython 3.14 build is sufficient. Step 0 concurrency is network-I/O concurrency and does not require free-threaded Python.
PyTorch is intentionally kept out of the main Python 3.14 BioMiner environment. Step 3 uses a separate Python 3.12 BioCLIP worker environment.

## Installation

Create and synchronise the project environment:

```bash
cd ./BioMiner
unset VIRTUAL_ENV
uv sync --extra test
```

Create the optional YOLOE-26 and BioCLIP worker environments only on machines that will run detector-first or BioCLIP work:

```bash
bash scripts/setup_yoloe26_user_py312.sh
bash scripts/setup_bioclip25_user_py312.sh
```

These create external Python 3.12 runtimes under `./YOLO26` and `./BioCLIP25` next to `./BioMiner`. Commands are run from `./BioMiner`, so runtime flags use `../YOLO26` and `../BioCLIP25`. Set `BIOMINER_BASE_PATH=/path/to/base` on macOS, WSL, or Ubuntu when the sibling folders live outside the inferred base path. The worker environments, model files, and caches are local-only and must not be committed.

Verify the CLI:

```bash
uv run biominer --help
uv run biominer registry build --help
```

Run the local test suite:

```bash
uv run pytest -q
```

Set Flickr credentials only for commands that call Flickr:

```bash
cp .env.example .env
# edit .env and set FLICKR_API_KEY
```

BioMiner CLI startup also loads secrets automatically when a secrets file is present. The lookup order is:

```text
BIOMINER_SECRETS_ENV
/Applications/secrets/secrets.env
../secrets/secrets.env next to ./BioMiner
/mnt/c/Applications/secrets/secrets.env for WSL
```

The Flickr variables expected by current commands are `FLICKR_API_KEY` and, when future signed Flickr operations need it, `FLICKR_SECRET_KEY`. Existing shell environment variables are preserved unless a caller explicitly asks the loader to override them. Never commit `.env`, `secrets.env`, or API keys.

# Step 0 — Taxonomic registry

Step 0 creates the authoritative versioned taxonomic name graph used by every downstream query and evidence row.

## Butterfly scope

Configured root:

```yaml
root:
  scientific_name: Papilionoidea
  rank: SUPERFAMILY
  gbif_taxon_key: 1875
```

Configured families:

| Family | GBIF key |
|---|---:|
| Hesperiidae | 6953 |
| Papilionidae | 9417 |
| Pieridae | 5481 |
| Lycaenidae | 5473 |
| Riodinidae | 1933999 |
| Nymphalidae | 7017 |
| Hedylidae | 6951 |

For each configured family BioMiner:

1. runs GBIF name matching and retains the matcher result;
2. uses the pinned accepted family key for production identity;
3. requires rank `FAMILY`;
4. requires accepted status;
5. requires the pinned key to resolve to the configured family name;
6. checks its lineage;
7. accepts `Lepidoptera` as the recorded fallback when GBIF omits the intermediate `Papilionoidea` node;
8. fails on a conflicting valid family-level match.

A GBIF `HIGHERRANK` result may be retained as source evidence but must never replace the pinned family identity. This is required for Hedylidae, whose matcher may return a higher-rank fallback while the reviewed accepted family key remains valid.

## Step 0A — GBIF spine

GBIF supplies:

- accepted taxon keys;
- root, family, genus, and species hierarchy;
- accepted/synonym relationships;
- first-layer vernacular names.

Traversal:

```text
Papilionoidea
-> seven configured families
-> accepted genera
-> accepted species
-> species synonyms
-> species vernacular names
```

The accepted spine is enumerated deterministically. Independent species enrichment is performed concurrently.

## Concurrent production build

The production registry command exposes:

```bash
uv run biominer registry build \
  --output-dir data/registry/<version> \
  --registry-version <version> \
  --scope-json config/butterfly_scope.json \
  --report-dir reports \
  --workers 8 \
  --progress-every 100 \
  --checkpoint-every 500 \
  --max-retries 5
```

Behavior:

- bounded `ThreadPoolExecutor` species enrichment;
- default eight workers;
- bounded task submission;
- pooled HTTP connections;
- retry of HTTP 429, 502, 503, 504, timeouts, and transient transport failures;
- `Retry-After` support;
- exponential backoff with jitter;
- permanent 4xx errors are not retried;
- worker results are merged only in the main thread;
- deterministic sorting before output;
- family-level resumable checkpoints;
- structured progress logs.

The implementation parallelises species synonym and vernacular-name retrieval. Taxonomic spine construction remains ordered and deterministic.

## Start a production registry build

Use sequential shell assignments so `V`, `OUT`, and `LOG` cannot reuse stale values:

```bash
cd ~/BioMiner && unset VIRTUAL_ENV && \
ROOT="$PWD" && \
V="$(date -u +%Y.%m.%d.%H%M%S)" && \
OUT="data/registry/$V" && \
LOG="logs/step0_$V.log" && \
export ROOT V OUT LOG && \
mkdir -p "$OUT" reports logs run && \
{
  nohup bash -c '
set -euo pipefail
cd "$ROOT"

BIOMINER_LOG_LEVEL=INFO PYTHONUNBUFFERED=1 \
uv run biominer registry build \
  --output-dir "$OUT" \
  --registry-version "$V" \
  --scope-json config/butterfly_scope.json \
  --report-dir reports \
  --workers 8 \
  --progress-every 100 \
  --checkpoint-every 500 \
  --max-retries 5

uv run python -c "
import json, pathlib, sys
m = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(json.dumps(m, indent=2))
sys.exit(0 if m.get(\"qa_fatal_count\", 1) == 0 else 1)
" "$OUT/manifest.json"

uv run biominer registry audit \
  --registry-dir "$OUT" \
  > "reports/registry_audit_$V.json"

ln -sfn "$V" data/registry/current
' >"$LOG" 2>&1 </dev/null &

  PID=$!
  printf 'V=%q\nPID=%q\nOUT=%q\nLOG=%q\n' \
    "$V" "$PID" "$OUT" "$LOG" \
    > run/latest_step0.env

  printf 'Registry started: version=%s pid=%s output=%s log=%s\n' \
    "$V" "$PID" "$OUT" "$LOG"
}
```

## Follow the build

```bash
cd ~/BioMiner
source run/latest_step0.env
tail --pid="$PID" -n 60 -f "$LOG"
```

Pressing `Ctrl+C` stops only the log follower.

Expected progress events include:

```text
registry.build.start
registry.gbif.root
registry.gbif.family.start
registry.gbif.family.genera
registry.gbif.family.progress
registry.gbif.family.checkpoint
registry.gbif.family.complete
registry.gbif.complete
registry.build.compile.start
registry.build.compile.complete
registry.build.complete
```

## Resume an interrupted build

Reuse the same version and output directory:

```bash
cd ~/BioMiner
source run/latest_step0.env

{
  nohup bash -c '
set -euo pipefail
cd "$ROOT"

BIOMINER_LOG_LEVEL=INFO PYTHONUNBUFFERED=1 \
uv run biominer registry build \
  --output-dir "$OUT" \
  --registry-version "$V" \
  --scope-json config/butterfly_scope.json \
  --report-dir reports \
  --workers 8 \
  --progress-every 100 \
  --checkpoint-every 500 \
  --max-retries 5
' >>"$LOG" 2>&1 </dev/null &

  PID=$!
  printf 'V=%q\nPID=%q\nOUT=%q\nLOG=%q\n' \
    "$V" "$PID" "$OUT" "$LOG" \
    > run/latest_step0.env

  echo "Resumed version=$V pid=$PID"
}
```

A checkpoint is reusable only when its registry version, scope hash, source, and schema version match.

## Registry outputs

A successful registry version contains:

```text
data/registry/<version>/
  taxa.parquet
  taxon_relations.parquet
  names.parquet
  name_evidence.parquet
  source_snapshots.parquet
  flickr_query_definitions.parquet
  qa_findings.parquet
  manifest.json
  gbif_source_snapshot.json
  checkpoints/
```

The successful version is promoted through:

```text
data/registry/current
```

Promotion occurs only after compilation, manifest validation, registry audit, and fatal QA checks pass.

## Registry audit

```bash
uv run biominer registry audit \
  --registry-dir data/registry/current
```

The audit queries Parquet through DuckDB and reports:

- taxa by rank and family;
- accepted, synonym, and vernacular name counts;
- language/source/trust distributions;
- Flickr query counts by field and priority;
- fatal and warning QA findings.

## Step 0B — Supplemental names

GBIF remains the accepted taxonomic spine.

Supplementary sources:

| Source | Use |
|---|---|
| Catalogue of Life | accepted/synonym evidence, vernacular evidence, discrepancy QA |
| iNaturalist | regional preferred and place-linked names |
| ITIS | common names and source taxon links |
| EOL | optional additional vernacular evidence |
| Translation providers | disabled candidate translations only |

iNaturalist and ITIS names must first be linked to an accepted GBIF taxon and must not replace GBIF accepted identity.

Regional names retain geographic scope. An Australian preferred name must not be promoted to a global name without preserving that scope.

Default trust policy:

| Source | Tier |
|---|---|
| GBIF accepted taxonomy | T1 |
| Catalogue of Life taxonomy | T1 |
| GBIF or CoL vernacular | T2 |
| Wikidata labels and aliases with confident taxon links | T3 |
| iNaturalist or other community names | T4 unless reviewed |
| Dictionary or generated translation candidates | T5 |

T5 translation candidates remain disabled until reviewed or independently corroborated.

## Step 0C — Flickr query compilation

Each enabled registry name is compiled into separate atomic Flickr definitions.

Priority order:

| Priority | Definition |
|---:|---|
| 10 | species scientific name — tags |
| 20 | species common name — tags |
| 30 | genus scientific name — tags |
| 40 | family scientific/common name — tags |
| 50 | species scientific name — text |
| 60 | species common name — text |
| 70 | genus/family — text |
| 80 | reviewed anchored broad terms — tags |
| 90 | reviewed anchored broad terms — text |
| 100+ | disabled experimental translation candidates |

Each definition retains:

- registry version;
- deterministic query-definition ID;
- accepted GBIF taxon key;
- accepted scientific name;
- rank and lineage keys;
- source term and normalized query term;
- language, script, region, and bbox;
- term class;
- trust and precision tier;
- search field;
- priority;
- enabled/review state.

Tags are scheduled before text.

## Step 0D — Registry QA

The production build checks:

- configured taxon identity, rank, status, and key;
- accepted/synonym relationships;
- lineage and root-validation mode;
- source discrepancies;
- language and script codes;
- name collisions across taxa;
- duplicate names and query definitions;
- suspicious translations;
- untranslated or generic terms;
- taxa without accepted names;
- taxa without enabled query definitions;
- checkpoint compatibility and completeness.

Fatal findings prevent promotion. Warnings remain visible in `qa_findings.parquet` and reports.

# Step 1 — Flickr metadata discovery

Step 1 consumes enabled definitions from:

```text
data/registry/current/flickr_query_definitions.parquet
```

It fetches Flickr metadata only.
Production polling does not seed generic multilingual searches automatically; work items must be created from registry-derived query definitions. Broad probes such as `butterfly` are explicit dev/review workflows and must remain anchored to accepted taxa before they can enter production discovery.

Outputs include:

- raw Flickr response payloads;
- canonical photo metadata;
- folded query-term provenance;
- API ledger and work state;
- evidence Parquet;
- compact reports.

## Deduplication rule

**Deduplicate evidence rows by Flickr photo ID and fold discovery terms into canonical provenance arrays.**

BioMiner keeps one canonical source/evidence row per Flickr photo ID. When another search term rediscovers the same photo, BioMiner folds that provenance into list fields on the canonical row:

- `text_search_terms`
- `tag_search_terms`
- `all_query_labels`
- `query_hit_count`
- `duplicate_query_hit_count`

This reduces Parquet row inflation while preserving the search-term provenance needed for auditing. First-query metadata remains available as `first_query_field`, `first_query_term`, and `first_query_language`.

Optional QA reports may still summarize duplicate rediscoveries. If generated, they should include:

```text
flickr_photo_id
query_definition_id
query_term
search_field
accepted_taxon_key
family_key
genus_key
species_key
registry_version
deduplication_reason
```

For each photo BioMiner supports:

- query-derived candidate taxonomy;
- metadata-derived keyword matches from title, description, tags, machine tags, and comments.

A broad term such as `butterfly` must not imply a family, genus, or species by itself.

## Flickr result and rate limits

Keep these separate:

```text
SOFT_API_CALLS_PER_HOUR = 3500
HARD_API_CALLS_PER_HOUR = 3600
STABLE_RESULT_THRESHOLD = 4000
FLICKR_SEARCH_RESULT_WINDOW = 4000
```

- 3,500 calls/hour is BioMiner's operating budget.
- 3,600 calls/hour is the hard local ceiling.
- 4,000 results is Flickr's accessible search window per query.
- 4,000 is therefore the stable BioMiner leaf threshold.

## Flickr search policy

Default request fields:

```text
media=photos
safe_search=1
content_types=0
extras=description,license,date_upload,date_taken,owner_name,last_update,geo,tags,machine_tags,o_dims,views,media,url_l,url_m
```

Image URL preference:

```text
url_l -> url_m
```

Do not use `url_o` by default.

Page sizes:

```text
count probes: per_page=1
normal pages: per_page=500
bbox/geotagged pages: per_page=250
```

## Explicit broad-probe coverage

Broad probes are not implicit production seeds. When retained for dev QA or a reviewed broad-query experiment, they use deterministic upload-date slices:

```text
start: 2004-02-10
slice length: 5 days
end: current date
initial page: 1
per_page: 500
accessible pages: at most 8
```

Workflow:

1. enqueue page 1 for each five-day slice;
2. fetch and store the real page-1 metadata;
3. read `photos.total`, `photos.pages`, `photos.page`, and `photos.perpage`;
4. enqueue pages 2 through `min(photos.pages, 8)`;
5. mark a slice saturated when page 8 returns 500 rows;
6. stop cleanly when the API budget is exhausted;
7. resume pending work in deterministic SQLite order.

Broad searches do not recursively count-probe the full time range.

## Metadata polling

Production runs poll Flickr metadata through `uv run biominer run`. The direct poller is retained only as a dev/debug entry point:

```bash
uv run biominer dev flickr poll-once \
  --max-api-calls 3500 \
  --workers 8 \
  --state-db data/state/flickr_poller.sqlite \
  --raw-root data/raw \
  --evidence-output staging/evidence/poll_once_evidence.parquet
```

Each worker must reserve an API-call token before making a request.

# Step 2 — Metadata flagging

Inputs:

```text
staging/evidence/poll_once_evidence.parquet
```

Metadata text hints are retained as review and evidence-bucket flags rather than exposed as a public hard-drop command or external text-list path. Flag obvious non-biodiversity material:

```text
artwork
tattoo
AI/generated
logo/brand
object/product
textile/pattern
museum/pinned specimen
other insect
not Lepidoptera
```

Keep butterfly life stages:

```text
adult
egg
caterpillar
larva
pupa
chrysalis
```

Step 2 does not make final species decisions.

# Step 3 — BioCLIP 2.5 screening

BioCLIP uses temporary image downloads and register-based processing.

BioCLIP 2.5 Huge runs through a separate Python 3.12 sidecar environment:

```bash
bash scripts/setup_bioclip25_user_py312.sh
```

Verify the sidecar runtime without loading the model:

```bash
uv run biominer dev vision bioclip-runtime-check \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --device auto
```

Prefetch the BioCLIP 2.5 Huge safetensors snapshot:

```bash
uv run biominer dev vision bioclip-prefetch-model \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface"
```

Run object-first scoring after `vision detect` has produced object detections:

```bash
uv run biominer vision score \
  --input staging/evidence/canonical_source_records.parquet \
  --detections staging/evidence/object_detections.parquet \
  --species-context staging/evidence/species_context.json \
  --species-candidates data/registry/current/species_candidates.parquet \
  --output staging/evidence/object_bioclip_scores.parquet \
  --runtime-python "../BioCLIP25/venv/bin/python" \
  --hf-cache-dir "../BioCLIP25/cache/huggingface" \
  --device auto \
  --ablation-mode detector_crop
```

Rules:

- use one persistent model worker per run;
- download an image temporarily;
- score whole-image/object-crop visual modes through BioCLIP object scoring;
- write object score rows;
- delete the staged image;
- idempotently skip successful records by source/photo/image/model/checkpoint;
- use fake classifiers in tests.

Successful-record identity:

```text
source
flickr_photo_id
image_url
model_id
model_version
model_checkpoint
```

## Evidence buckets

### Gold

- butterfly at any life stage;
- BioCLIP species score above 0.70;
- matching scientific or accepted common-name evidence;
- image URL;
- event date;
- latitude and longitude;
- no hard-negative category.

### Silver

- species score from 0.35 through 0.70 with matching species evidence; or
- Gold-strength evidence missing event date or geolocation;
- no hard-negative category.

### Bronze

- remaining butterfly/life-stage records;
- insufficient species agreement;
- records needing comment or human review.

### Bin

- no butterfly detected;
- hard-negative visual/material category;
- irrelevant non-biodiversity record.

Operational download or runtime failures remain retryable and must not be silently converted into biological negatives.

## Category model

Primary category column:

```text
image_category
```

Allowed initial values:

```text
adult_butterfly
life_stage_non_adult
museum_specimen
artwork
tattoo
ai_generated
logo_or_brand
object_or_product
textile_or_pattern
other_insect
not_lepidoptera
unknown
```

Life stages:

```text
adult_butterfly
egg
caterpillar
larva
pupa
chrysalis
unknown
```

# Step 4 — Targeted Flickr comment review

Comments are not fetched globally.

Default queue:

```text
Bronze records
```

Additional ambiguous records may be queued for:

- BioCLIP/Flickr text mismatch;
- suspected species conflict;
- missing date;
- missing geo;
- unknown category or life stage;
- low confidence;
- incomplete evidence.

Commands:

```bash
uv run biominer dev comments queue \
  --input staging/evidence/classified.parquet \
  --state-db data/state/comment_review.sqlite

uv run biominer dev comments review-once \
  --state-db data/state/comment_review.sqlite \
  --max-api-calls 300

uv run biominer dev comments apply-decisions \
  --input staging/evidence/classified.parquet \
  --output staging/evidence/classified_with_comments.parquet \
  --state-db data/state/comment_review.sqlite
```

Promotion rules:

```text
Bronze -> Gold
  comment species/synonym matches BioCLIP candidate
  Gold score/date/geo/category rules also pass
  no hard negative

Bronze -> Silver
  comment species/synonym matches
  species evidence is present
  Gold metadata requirements are incomplete

Remain Bronze
  no match
  generic comments only
  species conflict
  non-adult life stage without sufficient evidence
  incomplete evidence
```

Comments must not override hard negatives or fabricate structured coordinates from free text.

# Common commands

Build or audit the registry:

```bash
uv run biominer registry build --help
uv run biominer registry audit --help
```

Run the rank-aware production workflow:

```bash
uv run biominer run \
  --taxon "Papilio demoleus" \
  --rank species \
  --registry-dir data/registry/current \
  --output-prefix s3://biominer/runs \
  --storage-backend s3 \
  --workstore-backend postgres
```

Validate production configuration:

```bash
uv run biominer storage doctor
uv run biominer workstore doctor
```

Local production runs write `run_manifest.json`, `reports/run_metrics.json`, and `reports/review_queue.parquet` under the run directory. The review queue contains Bronze and InReview photo summaries for targeted human/comment review.

Run object detection and BioCLIP scoring as debug subcommands:

```bash
uv run biominer vision detect --help
uv run biominer vision score --help
uv run biominer evidence join --help
```

Use dev-only utilities for direct Flickr/comment debugging:

```bash
uv run biominer dev --help
```

# Data stack

BioMiner uses:

- **Polars** for dataframe operations;
- **Parquet** for durable tabular storage;
- **DuckDB** for local analytical queries and QA;
- **SQLite** for operational work queues and rate-limit ledgers;
- **JSON** only for compact configuration, manifests, checkpoints, and reports.

Do not introduce pandas or CSV workflows unless a strict external boundary requires them. Do not add Seaborn; use Matplotlib directly.

# Metrics and reports

Every run writes compact JSON and Markdown under `reports/`.

Include when applicable:

```text
command
git_sha
run_id
pid
status
started_at
ended_at
effective_workers
retry_count
rate_limit_events
checkpoint_count
resume_state
api_calls_used
api_calls_remaining
calls_per_hour
records_per_call
avg/p50/p95_seconds_per_call
rows_in
rows_out
dedupe_count
error_count
bucket_counts
category_counts
life_stage_counts
total_seconds
rows_or_images_per_second
artifact_bytes
checkpoint_bytes
cache_bytes_before
cache_bytes_after
rss_peak_memory
gpu_memory_peak
```

Unsupported metrics must be `null` or `"not_instrumented"`, never guessed.

Expected report families include:

```text
reports/registry_build_<version>.json
reports/registry_build_<version>.md
reports/registry_audit_<version>.json
reports/api_budget_profile.json
reports/fetch_profile.json
reports/metadata_flag_profile.json
reports/bioclip_profile.json
reports/occurrence_bin_profile.json
reports/comment_review_profile.json
reports/cache_profile.json
reports/idempotency_profile.json
```

# Long-running processes

For registry builds, Flickr fetches, and BioCLIP runs:

- start one detached local process;
- redirect output to `logs/`;
- write PID, version, output, and log paths to `run/`;
- write compact manifests to `reports/`;
- use structured progress logging;
- checkpoint resumable work;
- do not continuously poll from an agent session;
- inspect progress only when requested.

# Tests

Run all tests:

```bash
uv run pytest -q
```

Focused test areas:

```text
CLI surface
GBIF root and family-key validation
GBIF synonym accepted-usage handling
GBIF lineage fallback
bounded species-enrichment concurrency
retry/backoff behavior
checkpoint compatibility and resume
deterministic registry output
registry query compilation
registry fatal QA gates
Flickr endpoint constraints
API-budget enforcement
fixed upload-date slicing
deterministic resume order
metadata flag rules
category/life-stage rules
BioCLIP worker behavior with fakes
temporary image deletion
idempotency
comment-review queueing
comment-derived promotions
```

Tests must remain local and small. They must not require:

```text
network
Flickr credentials
CUDA
real BioCLIP weights
real downloaded images
large Parquet artifacts
large DuckDB artifacts
model caches
```

# Out of scope

The active BioMiner scope excludes:

- Darwin Core occurrence publication;
- taxonomic-validation claims;
- permanent Flickr image archival;
- global comment fetching;
- multi-key Flickr quota multiplication;
- blind deep paging beyond Flickr's accessible result window;
- network/CUDA/model-weight requirements in unit tests.
