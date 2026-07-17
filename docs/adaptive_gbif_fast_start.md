# Adaptive GBIF fast-start workflow

BioMiner uses `adaptive_gbif_fast_start` by default to reduce time to the first
useful BioCLIP comparison without treating provider metadata as ground truth.
Use `human_verified_strict` when every reference identity must be independently
reviewed before scoring, or `human_verified_flagged_only` while remediating an
existing provisional bank.

## Lifecycle and evidence maturity

```text
GBIF metadata
  → automated taxon, rights, media, duplicate and independence gates
  → YOLOE route/domain QA (quality and life-stage routing, never species ID)
  → GBIF provider-asserted provisional support
  → content-addressed BioCLIP embeddings
  → route-separated prototypes / nearest-reference evidence
  → provisional raw Flickr scores and review priorities
  → source-bound human Flickr labels
  → weighted species/region audit
  → targeted human reference review for flagged species only
  → bank revision, affected-only rebuild and selective Flickr rescore
  → human-gated final occurrence export
```

| Evidence | Human reviewed | Probability | May authorize final release |
|---|---:|---:|---:|
| GBIF provider-asserted provisional support | No | No | No |
| Human-verified reference support | Yes | No | No |
| Provisional raw score or margin | No | No | No |
| Independently calibrated probability | Labels only | Yes | No |
| Decisive, source-bound Flickr release decision | Yes | No | Only with every other release gate |

## Reference acquisition and admission

Acquire GBIF still-image candidates with accepted taxon and dataset provenance.
Admission requires accepted reconciliation, non-fossil occurrence, accepted
licence and attribution, successful download/decode and SHA-256, canonical
duplicate resolution, observation/photographer independence, a compatible
YOLOE route and sufficient subject area. Adult, larval and specimen routes form
separate banks. YOLOE supplies route and domain evidence; it does not assert the
species.

An admitted unreviewed row carries the mode, policy version/fingerprint,
`identity_evidence_basis = gbif_provider_asserted`, provider taxon/dataset
fields, `provisional_support = true`, and `statistical_audit_required = true`.
A later human rejection always overrides the provider assertion.

## Review and readiness

Reference review and Flickr review are separate append-only workflows. Adaptive
reference review is generated only for statistically flagged species and may
verify, exclude, correct metadata, remain uncertain or request a second review.
Flickr candidates may be scored before review, but Skip, Can't view, uncertain,
conflicting or stale-hash decisions never enter final export.

`ready_provisional` permits reference embedding, prototype construction and
provisional scoring. It blocks calibrated claims and scientific release and
requires a statistical audit. Strict `ready` requires human-verified support.
The immutable readiness directory and trusted JSON SHA-256 are required before
vision; a mode or policy fingerprint change invalidates the permit.

## BioCLIP, orchestration and selective reruns

BioCLIP workers load the model once and reuse content-addressed Flickr and
reference embeddings when image, model and preprocessing identities match.
Provisional ranking may combine prototypes, nearest references, competitor
margins, geography and compatible domains; its raw outputs remain
non-probabilistic. A probability requires a separately fitted calibrator using
eligible human-reviewed labels.

The adaptive stage graph does not auto-complete human steps. Automated admission
blocks first scoring; Flickr verification blocks release; the audit blocks
species-quality approval; targeted reference review activates only for flags;
revision/rebuild/rescore stages activate only for affected dependencies.
Unchanged detector output, embeddings, prototypes and Flickr scores remain
reusable through validated impact plans and checkpoints.

## CLI and local/cloud execution

Inspect the current surfaces before a run:

```bash
uv run biominer references --help
uv run biominer references validate-readiness --help
uv run biominer run --help
```

The key production options are:

```text
--reference-admission-mode adaptive_gbif_fast_start
--reference-source gbif
--initial-scoring-mode provisional_reference_ranking
--flickr-release-requires-human-review
--statistical-reference-audit
```

Use `run --dry-run` to inspect the resolved stage plan. Local execution uses
filesystem storage and SQLite; production cloud execution uses S3-compatible
storage and a PostgreSQL-compatible workstore. Both preserve the same Parquet,
JSON manifest, immutable readiness, checksum, checkpoint and human-gate
semantics. Never copy a readiness pin across admission modes.

## Evaluation and release

Species-level metrics use only human-reviewed Flickr labels. Representative and
targeted samples remain separate; weighted estimates are required when sampling
probabilities differ. Insufficient labels produce `insufficient_sample` and a
review queue rather than invented metrics. Statistical flags persist their
thresholds and reasons and prioritize review; they never prove a reference is
misidentified.

After a reviewed bank revision, reuse unchanged vectors, rebuild only affected
models/prototypes, and rescore only Flickr records whose target, competitor,
candidate set, reference dependency or safety-band margin is affected. Final
export remains fail-closed for every unreviewed Flickr record.
