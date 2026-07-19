# Post-uplift architecture and simplification audit

Date: 2026-07-19  
Status: implementation baseline; no live scientific result is claimed

## Decision summary

The geography-conditioned dynamic-pooling uplift added the right scientific
boundaries, but it did not finish the production cutover. Between baseline
`c7eaa9bf3696a25a0c8229837819dccec4fb9d66` and the first cleanup commit
`eef126a1fa0ecc61d512e8b1cdde50244b9165ef`, the repository changed 278 files,
adding 79,336 lines and deleting 359. It added 62 Python source modules while
deleting none. The current tree contains 201,769 physical Python source lines
and 111,221 physical Python test lines.

The implementation therefore accumulated the new adaptive contracts beside
the old cascade, bucket and comment-promotion application instead of replacing
it. This is not just excess code:

- the main adaptive workflow declares 31 stages, but the built-in orchestrator
  implements only five of them (`resolve_taxon_scope`, `build_registry`,
  `compile_queries`, `enqueue_flickr_work`, and `poll_flickr`);
- the remaining non-manual adaptive stages fail with
  `stage_handler_not_configured` unless a caller injects a handler;
- the seven dynamic-pooling CLI operations rejected non-dry-run execution,
  described their adapter as `not_connected`, and declared several bindings
  that differed from the owning Parquet contracts;
- the old default workflow still executes detector-crop, cascade scoring,
  bucket summarisation, and Flickr-comment promotion;
- the initial BioMiner ButterflyLens consumer pin predated 62 committed
  ButterflyLens changes and did not cover the newer quality,
  occurrence-release, sensitive-location, Darwin Core, or ALA contribution
  contracts; the current pin audit now closes that mismatch.

The production direction is one adaptive full-frame workflow with explicit
manual-review stops and immutable product handoffs. Compatibility code must not
remain callable merely because it has tests.

## Evidence and method

This audit uses the current source and test import graph, CLI dispatch,
orchestrator stage dispatch, Git history from the dynamic-pooling baseline,
committed downstream consumer objects, exact-symbol searches, Ruff, Vulture,
and focused tests. Static findings are treated as leads and confirmed against
runtime entry points and contracts before removal.

GitHits is unavailable in the current runtime and the user instructed that it
must not be called for the remainder of this goal. No GitHits result is claimed
or fabricated. Repository history, tests, schemas, manifests, and pinned
consumer contracts are the provenance sources for this audit.

Existing untracked BioMiner runtime/configuration files and dirty TaxaLens and
ButterflyLens documentation files are user-owned. They are not deletion
candidates and are excluded from all cleanup commits.

## Prior-phase quality review

| Phase group | What is sound | What remains incomplete or misleading | Required action |
|---|---|---|---|
| 0–3: contracts and geographic units | Explicit geography, canonical Flickr units, immutable identities, and missing-geography semantics are appropriate. | These are primarily contracts and pure transformations, not an executed production route. | Retain; connect them through the single runtime. |
| 4–5: candidates and pools | Target-preserving candidates, bounded expansion, source axes, and explicit fallback reasons are scientifically sound. | Fixture strategy selection is not evidence that one strategy is empirically superior. | Retain all auditable strategies; do not label one scientifically selected without live evidence. |
| 6–8: vision, scoring, caching | Full-frame routing, separate raw components, reusable embeddings/matrices, and explicit efficiency accounting are appropriate. | The old detector-crop/cascade runtime remains the implemented production path. | Remove old runtime after adaptive execution coverage exists. |
| 9–12: review, evaluation, and selective rebuild | Representative versus targeted review, source independence, grouped quality, and dependency fingerprints are strong boundaries. | Many modules are fixture-tested but have no production adapter or orchestrator execution. | Connect immutable inputs/outputs and preserve manual-review stops. |
| 13: orchestration | The 31-stage topological graph correctly names the intended boundaries. | Most graph nodes are declarations only. A skipped/failed node cannot be reported as an executed phase. | Implement concrete dispatch and fail closed on absent artifacts. |
| 14: downstream handoffs | Product-specific, immutable, fail-closed handoff manifests are a sound boundary. | TaxaLens was already current; ButterflyLens had expanded its contract surface and retired fingerprint v1.0. | The current committed consumers are now pinned; preserve v1.1-only fingerprints and downstream-owned release controls. |
| 15: pilot | Deterministic fixture coverage and ablation mechanics are useful software evidence. | Fixture outputs are not live Flickr/GBIF/YOLOE/BioCLIP evidence and cannot select a release strategy. | Keep as regression fixtures; label empirical outcomes unavailable. |
| 16: release report | The report correctly preserves the fixture-only limitation. | Calling the workflow complete obscures disconnected adapters and unhandled stages. | Replace completion language with software-contract/fixture completion and runtime readiness gates. |

## Layer disposition

### Remove

1. **Comment-to-occurrence promotion.** `flickr_comments/comment_review.py`
   infers species/date/location from unverified comment text and can mutate
   records to Gold or Silver. `comments_enrichment.py` can promote comment terms
   into new query work. This conflicts with the current rule that Flickr text
   is evidence for review, never an occurrence decision or an autonomous query
   authority. Remove its CLI, state wrapper, stages, paths, metrics, schemas,
   tests, and output columns.

2. **Bucket authority and legacy evidence summary.** Gold/Silver/Bronze bucket
   assignment is coupled to the old detector/cascade workflow. The adaptive
   path owns raw score components, evidence maturity, review state, and the
   downstream release gate. Remove bucket-only code when the old orchestrator
   stages are removed; do not translate old buckets into new evidence labels.

3. **Hierarchical classification-v3 cascade runtime.** The classification
   overlay, taxonomy embedding cache, `20 -> 5 -> 3` cascade, cascade output,
   benchmark, crop-based production dispatch, prototype switches, and their
   dedicated tests are diagnostic/legacy code. They must not remain a second
   production architecture after full-frame dynamic pooling. Preserve old
   artifacts and Git revisions; do not migrate them in place.

   The callable model-free cascade benchmark and its synthetic fixture have now
   been removed. The production cascade cutover remains a separate gated
   subtask because it is intertwined with the old cloud runner and evidence
   join.

4. **Legacy/default/reference-first workflow layers.** After adaptive handlers
   are connected, remove alternate workflow selectors, aliases, configuration
   fields, and stage plans. One canonical production graph is easier to audit
   and cannot silently fall back to obsolete scientific semantics.

5. **Inert switches and forwarding helpers.** The first cleanup removed the
   hidden registry `--skip-classification` switch, two no-op wrappers, and stale
   manifest/report fields (143 lines removed). Continue removing only helpers
   proven to forward without adding validation, identity, resource ownership,
   or a consumer boundary.

### Retain

- Product-specific handoff builders plus the shared
  `integration/product_handoff.py` invariant layer. The shared module is used
  by both products and exporters for canonical artifact ordering, Git/SHA
  validation, content-derived identity, path safety, and atomic writes.
- Storage handoff bundling and receipt verification. This is an immutable
  transport boundary, not a second product abstraction.
- Dynamic-pool component scores, candidate strategies, audit frames, review
  contracts, evidence-maturity labels, source-independence logic, and selective
  rebuild identities.
- Current-schema checkpointing, leases, idempotency, and content-addressed
  caches. Checkpoints are not legacy when they are required to resume current
  durable work safely; compatibility readers for older schema shapes are
  legacy.
- Explicit configuration values that change scientific identity or resource
  limits. Remove only aliases, inferred fallbacks, duplicate defaults, and
  values belonging exclusively to deleted architectures.

## Cross-repository alignment baseline

TaxaLens committed HEAD is
`e845dd98493979f37b04dbb6538e0d7b8758ca11`, matching BioMiner's current
TaxaLens pin. TaxaLens owns product replay, human review, geographic impact,
and occurrence release. BioMiner must publish evidence and maturity; it must
never authorize TaxaLens occurrence release.

ButterflyLens committed HEAD and BioMiner's audited pin are
`3d6486da87f32136c35e29aeed6cb6291da66a17`. Exact committed-object tests cover
the current consumer schemas and confirm that BioMiner does not export database
primary keys, reviewer identities, service credentials, sensitive-location or
occurrence-release decisions, Darwin Core publication authority, or ALA
submission authority. The consumer's retired fingerprint v1.0 reader is
compatible with BioMiner because the producer already emits v1.1 only.

Sibling worktrees remain read-only during this goal. Alignment is performed by
reading committed objects and changing BioMiner's producer/contracts/tests,
not by importing product implementation code.

## Implementation phases and gates

### A. Remove misleading legacy authority

1. Remove comment enrichment/review and all automatic promotion.
2. Remove bucket/cascade/classification-v3 production surfaces and artifacts.
3. Remove alternate workflow plans and legacy-only configuration.

Gate: no removed symbol is importable or parseable; current adaptive schema,
science, CLI, and orchestrator tests pass; migration documentation states how
to retain historical artifacts without reinterpretation.

### B. Connect the adaptive runtime

1. Remove the disconnected plan-only dynamic command layer; use the concrete
   `references` application commands for artifact work.
2. Route the canonical orchestrator stages through those same owning
   application functions instead of duplicating transformations.
3. Keep manual-review stages awaiting explicit signed review input.
4. Fail closed on missing, stale, incompatible, or partially written inputs.

Gate: a bounded fixture-backed run crosses every automated stage, pauses at
manual gates, resumes from immutable review artifacts, and produces verified
handoffs. Dry-run remains side-effect free. No live scientific result is
claimed.

### C. Align product consumers

1. Re-audit TaxaLens current committed consumer and preserve its exact pin.
2. Preserve the exact ButterflyLens pin and committed-object parity tests for
   every BioMiner-owned export role and downstream-owned release gate.
3. Report unavailable product-owned artifacts explicitly rather than creating
   placeholders or inferring approvals.

Gate: pinned-object contract tests pass for both products and all authority
boundaries fail closed under mutation.

### D. Consolidate and release

1. Re-run import/dead-code/duplication audits after removals.
2. Consolidate only exact repeated validators where ownership and error
   semantics remain clear.
3. Run focused, adaptive, downstream, lint, and complete regression gates.
4. Update current-state and migration documentation with measured before/after
   source, test, module, and command-surface reductions.

Gate: the repository exposes one production architecture; documentation and
tests do not describe deleted compatibility paths; every incomplete live-data
or human-review dependency is explicit.

## Non-goals

- No historical artifacts, checkpoints, reports, caches, or user runtime state
  are deleted or rewritten.
- No fixture result is promoted to a biological, performance, or release claim.
- No manual review is synthesized.
- No product repository is made dependent on BioMiner implementation imports.
- No helper is removed solely because a static tool reports it unused.
