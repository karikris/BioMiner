# Strict reference verification baseline

- Task: `gbif-fast-0.1`
- Snapshot: `dcd494321abc0666ea692b5759f84bc4c7e08ba9`
- Recorded: 2026-07-17T10:04:10Z
- Confidence: high for committed code and tests; unknown where live artifacts are unavailable.

## Result

The normal reference-bank path is fail-closed and strict. A reference row can
be support-eligible only after review is complete, the resolved verification
status is `verified`, and `target_identity_verified` is true. The readiness
compiler then enforces configured target and competitor support minima. Only
`ready` and `ready_with_documented_shortfalls` permit vision, and target
shortfalls cannot be waived.

BioMiner also contains a separate, explicit Build Week prototype path that
accepts `provider_supported` reference rows for prototype classification. That
path does not claim human verification, calibrated probabilities, production
default status, or scientific-release authority. It is evidence that much of
the required fast-start machinery exists, but it is not the current general
reference-admission contract.

There is no `DEFAULT_REFERENCE_ADMISSION_MODE` constant and no versioned
three-mode admission policy at this snapshot. The strict requirement is
encoded across review, support-manifest, readiness, embedding, dependency, and
orchestration contracts rather than represented by one explicit mode.

## Reproducible environment

| Item | Baseline |
|---|---|
| Branch | `main` |
| Starting main SHA | `dcd494321abc0666ea692b5759f84bc4c7e08ba9` |
| Remote main SHA | `dcd494321abc0666ea692b5759f84bc4c7e08ba9` |
| Ahead / behind | `0 / 0` |
| Python | `3.14.4` |
| uv | `0.11.19` |
| `uv.lock` SHA-256 | `49e0c67867b37b25cc5de0522889c76c0943b3efd9c2de757e87484be9432a7d` |
| `pyproject.toml` SHA-256 | `6b0e611cd1b605b8f4b05a77e67f941594bbafcf6beccad4ce2bec86f6b49d1e` |
| Lock check | `uv lock --check` passed; 44 packages resolved |
| Full baseline suite | `uv run --locked pytest -q` |
| Full baseline result | **2,264 passed in 82.69 seconds** |

Pre-existing untracked files were excluded from the baseline and remain
untouched:

- `config/papilio_demoleus_flickr_estimator.sh2`
- `config/papilio_demoleus_multilingual_keywords.json`
- `docs/superpowers/`
- `duplicate_query_terms_skipped`
- `logs/`
- `query_terms_added`

## Current schema ledger

| Contract | Current schema |
|---|---|
| Reference observations | `reference-observations-v1.2.0` |
| Reference media candidates | `reference-media-candidates-v1.0.0` |
| Acquisition plan | `reference-acquisition-plan-v1.1.0` |
| Acquisition selections | `reference-acquisition-selections-v1.0.0` |
| Reference media objects | `reference-media-objects-v1.1.0` |
| Duplicate relationships | `reference-media-duplicate-relationships-v1.0.0` |
| Review queue | `reference-review-queue-v1.0.0` |
| Review decisions | `reference-review-decisions-v1.0.0` |
| Review decision import | `reference-review-decision-import-v1.0.0` |
| Review outcomes | `reference-review-outcomes-v1.0.0` |
| Review conflicts | `reference-review-conflicts-v1.0.0` |
| Resolved review media | `reference-review-resolved-media-v1.0.0` |
| Review queue provenance | `reference-review-queue-provenance-v1.0.0` |
| Review history | `reference-review-history-v1.0.0` |
| Review history head | `reference-review-history-head-v1.0.0` |
| Split assignments | `reference-bank-split-assignments-v1.0.0` |
| Support manifest | `reference-support-manifest-v2.0.0` |
| Bank summary | `reference-bank-summary-v1.0.0` |
| Bank readiness | `reference-bank-readiness-v2.0.0` |
| Readiness policy | `reference-bank-readiness-policy-v1.0.0` |
| Model input identity | `reference-model-input-identity-v2.0.0` |
| Reference embeddings | `reference-embeddings-v2.0.0` |
| Reference prototypes | `reference-prototypes-v2.0.0` |
| Prototype support | `prototype-support-manifest-v1.0.0` |
| Prototype readiness | `reference-bank-prototype-readiness-v1.0.0` |
| Reference CLI settings | `target-aware-reference-cli-settings-v1` |

## Current defaults

| Area | Current behavior |
|---|---|
| Production workflow | `legacy`; reference-first is opt-in |
| Classification mode | `target_scope_object_screening` |
| Normal support admission | Implicitly strict through support/readiness validators |
| Accepted licence policy in `ReferenceBankReadinessPolicy` | `allowed` only unless explicitly changed |
| Readiness permitting statuses | `ready` and `ready_with_documented_shortfalls` |
| Reference-first manual stage | `reference_review` before embeddings and prototypes |
| GBIF media verification value | `unreviewed` |
| GBIF reconciliation | `accepted_key_exact`, `conflict`, or `unresolved` |
| Default planned stratum | adult / unreviewed, 20 requested per species |
| Planned licence states | allowed, research-only, and unreviewed candidates; final support is stricter |
| Prototype target adult minimum | 5 support-train rows |
| Prototype regional competitor minimum | 1 species |
| BioCLIP production model | `imageomics/bioclip-2.5-vith14` in the orchestrator |
| YOLOE checkpoint | `yoloe-26s-seg.pt` |
| Explicit adaptive admission modes | Not implemented |

The acquisition planner may select unreviewed candidates and the downloader may
materialize them. The strict boundary occurs when review outcomes are compiled
into the normal support manifest, not at metadata discovery.

## Strict path trace

1. **Reference acquisition**
   (`src/biominer/references/gbif.py:GBIFReferenceAdapter`) requests GBIF still
   images, normalizes occurrence and media provenance, reconciles accepted
   taxon keys, and marks GBIF media `verification_status=unreviewed`. Fossils,
   absent occurrences, unsuitable bases of record, uncertain taxa, and missing
   media licences become exclusion or quarantine evidence.
2. **Reference schemas**
   (`src/biominer/references/schemas.py`) validate exact Polars schemas,
   deterministic order, taxon reconciliation, download/decode state, licence
   state, duplicate evidence, review queue state, and review decisions.
3. **Media acquisition and decoding**
   (`src/biominer/references/downloader.py`) produces content-addressed media
   objects with content type, decode state, SHA-256, dimensions, licensing, and
   attribution evidence.
4. **Duplicate resolution**
   (`src/biominer/references/deduplication.py`) resolves exact, provider mirror,
   resized-copy, burst, and perceptual relationships while retaining canonical
   media and unresolved/conflict states.
5. **Review queue**
   (`src/biominer/references/review.py:build_reference_review_queue`) binds each
   request to the acquisition selection, observation, media candidate, media
   object, duplicate evidence, image hash, licence, and attribution.
6. **Review decisions**
   (`src/biominer/references/review.py:import_reference_review_decisions`) keeps
   append-only decision history, reviewer identity, review round, conflicts,
   uncertainty, second-review requirements, and source bindings.
7. **Review resolution**
   (`resolve_reference_review_statuses` and `select_verified_reference_media`)
   yields completed, verified, excluded, conflict, and pending outcomes.
   `_support_blockers` rejects incomplete, non-verified, noncanonical,
   unresolved-taxonomy, disallowed-licence, missing-attribution, and unresolved
   duplicate rows.
8. **Readiness and support manifest**
   (`src/biominer/references/readiness.py:build_reference_bank_readiness`)
   validates the entire immutable input chain, resolves reviews, builds support
   rows, fingerprints the bank, evaluates minima and leakage, and issues a
   readiness status.
9. **Strict row validator**
   (`validate_reference_support_manifest`) requires every eligible row to have
   `verification_status=verified`, `review_status=completed`, and
   `target_identity_verified=true`, plus complete source, licence, attribution,
   canonical-media, route, split, and fingerprint evidence.
10. **Embedding permit**
    (`src/biominer/bioclip/reference_embeddings.py:build_reference_embeddings`)
    filters to `support_eligible` rows and requires a pinned readiness permit,
    model identity, checkpoint hash, preprocessing identity, full-frame input
    contract, source image hashes, and support fingerprints.
11. **Prototypes**
    (`src/biominer/bioclip/reference_prototypes.py`) build deterministic,
    route-separated prototypes from frozen `support_train` embeddings.
12. **Support dependency preflight**
    (`src/biominer/run/support_dependencies.py`) requires current candidates,
    strict readiness, embeddings, classifier, and calibrator; it checks
    verified target-adult minima before Flickr vision.
13. **Orchestration**
    (`src/biominer/run/stages.py`) places `reference_review` before reference
    embeddings/prototypes in `REFERENCE_FIRST_PRODUCTION_STAGES` and marks it as
    the manual stage. The normal path loads strict support dependencies.
14. **CLI**
    (`src/biominer/cli.py` and `src/biominer/reference_workflow_cli.py`) exposes
    acquisition, download, duplicate resolution, review export/import,
    readiness validation, embeddings, prototypes, classifier, calibrator, and
    target-aware scoring as explicit artifact-bound commands.

## Locations enforcing completed review

| Location | Enforcement |
|---|---|
| `references/review.py:_resolve_request` | Pending, uncertain, insufficient, or conflicting decisions cannot become `completed` |
| `references/review.py:_support_blockers` | Adds `review_not_completed` unless a decisive signature exists |
| `references/review.py:select_verified_reference_media` | Returns only support-eligible resolved verified media |
| `references/readiness.py:validate_reference_support_manifest` | Eligible rows require `review_status=completed` |
| `references/readiness.py:_build_support_rows` | Support rows inherit resolved review outcomes |
| `references/readiness.py:_build_reference_bank_summary` | Counts reviewed, verified, pending, excluded, and shortfall states |
| `references/readiness.py:_readiness_checks` | Pending included media keep the strict verification check pending |
| `references/readiness.py:_readiness_status` | Pending verification produces `awaiting_manual_review` |
| `references/readiness.py:_outcome_is_human_verified` | Requires completed + verified + identity true |
| `run/stages.py:REFERENCE_FIRST_PRODUCTION_STAGES` | Manual review precedes embeddings/prototypes |

## Locations enforcing verified identity

| Location | Enforcement |
|---|---|
| `references/schemas.py:validate_reference_review_decisions` | A `verified` decision requires `target_identity_verified=true` |
| `references/review.py:_support_blockers` | A completed outcome must be verified and identity-true |
| `references/readiness.py:validate_reference_support_manifest` | Rejects eligible rows not verified or lacking verified identity |
| `references/readiness.py:_outcome_is_human_verified` | Defines strict verified outcome predicate |
| `references/readiness.py:_readiness_checks` | `verified_support_only` fails or remains pending for unverified included media |
| `references/readiness.py:_validate_readiness_payload` | Revalidates strict check/status consistency when loading artifacts |
| `run/support_dependencies.py:_validate_target_support` | Reports and blocks deficient verified target support |

## Locations enforcing support minima

| Location | Enforcement |
|---|---|
| `references/readiness.py:ReferenceBankRequirement` | Every requirement has a positive minimum |
| `references/readiness.py:ReferenceBankReadinessPolicy` | Requires a positive target `adult_field` minimum |
| `references/readiness.py:ReferenceBankReadinessPolicy` | Forbids documented target shortfalls |
| `references/readiness.py:_readiness_checks` | Computes target and competitor shortfalls from eligible `support_train` rows |
| `references/readiness.py:_readiness_status` | Missing target support blocks readiness; approved competitor shortfalls are explicit |
| `run/support_dependencies.py:_validate_target_support` | Rechecks observed target minima before Flickr vision |
| `references/prototype_freeze.py:_readiness` | Prototype-only path separately defaults to target minimum 5 and competitor minimum 1 |

## Existing provider-supported prototype exception

The exception is intentionally separate:

- `references/prototype_freeze.py` permits `human_verified`,
  `provider_high_trust`, and `provider_supported` rows for a
  `prototype_only` bank after metadata, licence, decode, duplicate, route, and
  QA checks.
- Prototype readiness can set `classification_authorised=true` while keeping
  `human_verification_complete=false` and
  `human_verification_required_for_scientific_release=true`.
- `bioclip/prototype_support.py` validates metadata-qualified support and
  rejects any false human-verification claim.
- `run/orchestrator.py` uses this permit only for the explicit
  `build_week_target_aware_prototype` classification mode. Other modes load the
  strict readiness/dependency chain.
- The production default remains `target_scope_object_screening` and the CLI
  workflow default remains `legacy`.

This prototype path is the safest starting point for the adaptive design, but
its prototype-only schemas must not be silently reused as production
admission, calibration, final-test, or release authority.

## Papilio demoleus evidence available at baseline

The committed Phase 14 report supplies the following measured prototype
metrics:

| Metric | Value |
|---|---:|
| Selected reference media | 93 |
| Frozen prototype support | 81 |
| Excluded references | 12 |
| Retryable reference-source failures | 10 |
| GBIF provider-supported references | 81 |
| Human-verified references | 0 |
| Independent observations | 81 |
| Allowed-licence support | 2 |
| Research-only support | 79 |
| Adult-field support | 80 |
| Larval support | 1 |
| Frozen support embeddings | 81 |
| Planned Flickr records | 13,501 |
| Classified Flickr records | 13,496 |
| Retryable Flickr failures | 5 |
| Candidate score rows | 634,312 |
| Measured throughput | 2.274524 records/second |
| Measured peak RSS | 1,765,261,312 bytes |
| Persistent BioCLIP model loads | 1 |
| Resumed/reused support embeddings | 81 |

The report explicitly records:

- classification accuracy unavailable because no independent human taxonomic
  labels exist;
- calibration unavailable because independently reviewed labels are
  insufficient;
- raw scores are not probabilities;
- the bank is prototype-only;
- scientific release and production-default changes were not authorized; and
- reference review and Flickr final-inclusion review still remain human work.

The following strict-baseline metrics are **not measured** by committed
artifacts and must not be fabricated:

- elapsed time blocked specifically on reference review;
- strict workflow time to first reference embedding;
- strict workflow time to first prototype;
- strict workflow time to first Flickr score;
- manual reference-review minutes; and
- a strict-versus-adaptive paired runtime comparison.

## GitHits design evidence

Focused query:

> Python open-source reference image bank admission quality gates for weakly
> supervised biological species labels where provider-asserted records remain
> distinct from human-verified ground truth; include provenance manifests,
> fail-closed screening, and downstream human review.

Solution: `39e961a1-6eae-47d7-8ffa-be2d21bc262a`.

Reviewed references:

- `QinghongLin/data2story-skill@63a55c1`, MIT, validation/fail-closed patterns;
- `sanjaysgk/ipg#48`, MIT, provenance and review-boundary patterns.

Adopted concepts:

- keep provider assertion and human verification in distinct fields;
- use immutable source/image hashes and explicit admission reasons;
- fail closed on missing required metadata and contradictory evidence; and
- retain review as an explicit later state transition.

Rejected concepts:

- copying the generated implementation;
- JSON as the durable tabular workflow format;
- placing every provisionally admitted reference into an immediate mandatory
  review queue; and
- treating a batch-level reject as proof of biological invalidity.

BioMiner uses Polars/Parquet, its existing content-addressed artifact chain,
route-aware support, append-only review decisions, and statistical escalation.
No external code or prose was copied.

## Baseline conclusion

The strict baseline is reproducible and green. The architectural change must
not delete the strict path. It must introduce an explicit, fingerprinted
admission mode that allows only qualifying GBIF provider-asserted provisional
support to reach embeddings/provisional scoring, while preserving strict
review, calibration, Flickr final-inclusion, and scientific-release gates.
