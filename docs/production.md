# Production workflow

`biominer run` is the integrated production pipeline entry point. Concrete
reference workflow operations are exposed under `biominer references`; the run
orchestrator and CLI call the same owning application functions.

## Stages

```text
resolve scope
  → build/validate registry
  → plan geographic spread and compile discovery queries
  → enqueue and poll Flickr metadata
  → cluster Flickr geographic evidence
  → route Flickr full-frame visual inputs and build/reuse one embedding per photo
  → acquire, route and provisionally admit GBIF reference support
  → build/reuse reference embeddings and a geography index
  → build the complete family/geography candidate union
  → plan bounded global safety and local reference pools
  → score complete raw global/local components
  → provisional fusion and Flickr screening
  → build a representative probability sample
  → pause for source-bound Flickr human verification
  → risk-controlled and statistical reference audit
  → targeted reference review
  → rebuild affected reference evidence and selectively rescore affected records
  → final release gate
```

The default reference mode is `adaptive_gbif_fast_start`: automated reference
admission blocks first scoring, but reference human review does not. Flickr
human verification always blocks final occurrence release. Strict projects set
`--reference-admission-mode human_verified_strict`. The complete admission,
readiness, evidence-maturity and selective-rerun contract is documented in
[Adaptive GBIF fast-start](adaptive_gbif_fast_start.md).

`biominer run` owns one adaptive stage graph, planning, scope resolution,
manual-review pauses, support-dependency preflight, and manifest publication.
Concrete live stage operations are supplied by explicit stage owners; an
unconfigured live stage fails closed. Dry-run records the plan without opening
the workstore or running acquisition and model code. Reference-pool and matrix
identities are independently versioned, so pool changes reuse compatible
Flickr and reference embeddings.

Cloud runs require S3-compatible storage and a PostgreSQL-compatible workstore. Local development can use filesystem storage and SQLite. Work claims, retry state, committed shard inventories, and source evidence make runs resumable and idempotent.

## Adaptive dynamic-pooling contract

The registry stores BioCLIP-supported identity paths, but the adaptive visual
route does not treat a family winner as an identity proof. Family retrieval and
regional evidence form a complete, deterministic candidate union. The target
and safety-union candidates must remain present; family and geography hard
pruning are forbidden.

Each candidate receives a diverse global reference pool and, where evidence is
available, a geographically relevant local pool. A missing or inadequate local
pool remains an explicit unavailable state and falls back to global evidence;
it is never converted into zero support or biological absence. Pool selection
is bounded, class-balanced, observation/observer-aware, deterministic, and
fingerprinted.

The canonical target-aware model input is the full frame. YOLOE routes and
measures suitability; it does not classify species. One raw BioCLIP embedding
is persisted per compatible media/model/transform identity. Candidate, family,
global, and local matrices are cached separately, and scoring is ordered for
locality. Every raw component, disagreement, rank movement, coverage state,
fusion method, tie, and alternative remains available downstream.

No raw similarity, distance, detector score, margin, SVM value, component
score, or provisional fusion score is a probability. Candidate evidence,
calibrated probability, human verification, representative statistical
support, occurrence-release maturity, and downstream handoff maturity are
separate contracts. Unreviewed Flickr records cannot enter the verified
occurrence export.

The alternate `legacy` and `reference-first` workflow selectors, one-off Build
Week mode, family/genus cascade, strictly-above-0.90 genus shortcut,
per-detection crop materialization, bucketed visual modes, and rolling cloud
worker were removed on 2026-07-19. There is no supported runtime fallback.
Existing historical artifacts remain in Git and cannot substitute for
adaptive full-frame dynamic-pool output. See the
[workflow migration](migrations/alternate-workflow-removal.md) and
[vision-runtime migration](migrations/cascade-crop-runtime-removal.md).

## Example

```bash
uv run biominer --config config/biominer.cloud.example.toml run \
  --taxon Papilionidae \
  --rank family \
  --registry-dir s3://biominer/registry/butterflies-v2 \
  --output-prefix s3://biominer/runs/current \
  --reference-admission-mode adaptive_gbif_fast_start \
  --reference-source gbif \
  --initial-scoring-mode provisional_reference_ranking \
  --flickr-release-requires-human-review \
  --statistical-reference-audit
```

Use `storage doctor`, `workstore doctor`, and `run --dry-run` before a live run.
Dry-run resolves scope and records the plan and configured artifact paths; it
does not execute live acquisition, visual models, review, statistical audit, or
release. Stage-specific artifact, schema, fingerprint, and readiness checks run
and fail closed when their live adapter initializes.

The former seven-command `biominer dynamic-pooling` plan wrapper was removed on
2026-07-19. It had no execution adapters, and several declared input/output
bindings did not match the validated Parquet contracts. Typed dynamic-pooling
settings remain authoritative. Stage planning belongs to `biominer run`, while
artifact operations belong to the concrete `biominer references` commands; see
the [migration note](migrations/dynamic-pool-plan-cli-removal.md).

The Phase 15 fixture pilot reports an `insufficient_evidence` production
decision. Its review projection is not a selected default. Current runtime
settings remain unselected and unchanged; real source-bound review, precision
bounds, subgroup support, comparable computation, and MPS measurements are
still required. See the
[pilot report](../reports/geo_dynamic_pooling/pilot/geography_conditioned_pooling_report.md).

## Durability and observability

Every stage reports command, run ID, PID, git SHA, inputs, outputs, timestamps, elapsed time, rows, bytes, retries, errors, and artifact paths. Unsupported metrics are `null` or `not_instrumented`. Long jobs write structured progress and checkpoints; repeated polling by operators is not part of the execution model.

Images, raw API dumps, models, caches, generated registry builds, large Parquet files, and secrets are runtime state and must not be committed.

Filesystem/SQLite local runs and S3/PostgreSQL cloud runs use the same semantic
artifact contracts. Mode and policy fingerprints, object checksums, immutable
readiness pins and checkpoint identities determine reuse; moving an artifact
between backends does not weaken validation or human-review gates.
