# Post-uplift architecture and simplification audit

Date: 2026-07-19  
Status: architecture simplification complete; no live scientific result is claimed

## Decision summary

The geography-conditioned dynamic-pooling uplift added the right scientific
boundaries beside an older cascade, bucket and comment-promotion application.
The cleanup has now completed the architecture cutover: BioMiner exposes one
adaptive full-frame production graph, explicit manual-review stops, and
immutable TaxaLens and ButterflyLens handoffs. Historical implementations
remain recoverable from Git and frozen artifacts, not callable as compatibility
paths.

Against the pre-cleanup commit
`ae6a18509b7be48da5c6ca69ab0caacf4632cc70`, the final code implementation
commit `499990779131518f8bf6a712a7010e83c2f1a321` changes 224 files, with 2,193
insertions and 59,356 deletions (net -57,163 lines). Production Python changes
by +423/-34,315 (net -33,892) and tests by +510/-24,522 (net -24,012).
Production modules fall from 309 to 256 and test modules from 288 to 230;
physical Python lines fall from 201,841 to 167,949 in `src/biominer` and from
111,286 to 87,274 in `tests`.

The cutover removes the old detector/crop/cascade and bucket authority,
comment-based occurrence promotion, disconnected planning and Build Week
command stacks, hierarchical evaluation compatibility, alternate workflow
plans, obsolete readiness migration, implicit prompt provenance, test-only
facades, and orphan cloud/worker/model-state layers. The 31-value `RunStage`
enum now equals the 31-node adaptive graph exactly; concrete reference commands
self-identify instead of borrowing phantom production-stage labels.

Live stage handlers beyond the five built-in orchestration operations remain
explicitly injected. Missing handlers fail closed with
`stage_handler_not_configured`; manual stages cannot be auto-completed. This is
an honest external integration boundary, not a fallback to the deleted runtime
and not evidence that a live scientific run has occurred.

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
| 0–3: contracts and geographic units | Explicit geography, canonical Flickr units, immutable identities, and missing-geography semantics are appropriate. | Live acquisition remains an operator-owned action. | Retained as the single contract surface; missing live inputs fail closed. |
| 4–5: candidates and pools | Target-preserving candidates, bounded expansion, source axes, and explicit fallback reasons are scientifically sound. | Fixture strategy selection is not evidence that one strategy is empirically superior. | All auditable strategies retained; none is labelled empirically selected. |
| 6–8: vision, scoring, caching | Full-frame routing, separate raw components, reusable embeddings/matrices, and explicit efficiency accounting are appropriate. | Live model execution remains pending. | Old detector/crop/cascade production and benchmark paths removed; current durable caches retained. |
| 9–12: review, evaluation, and selective rebuild | Representative versus targeted review, source independence, grouped quality, and dependency fingerprints are strong boundaries. | Human review and sufficient live statistical evidence remain unavailable. | Immutable review gates retained; old heuristic review and hierarchical compatibility removed. |
| 13: orchestration | The 31-stage topological graph correctly names the intended boundaries. | Most live handlers are application-injected rather than built in. | Stage enum and graph now match exactly; absent handlers and artifacts fail closed. |
| 14: downstream handoffs | Product-specific, immutable, fail-closed handoff manifests are a sound boundary. | Product-owned review/release evidence cannot be manufactured by BioMiner. | Exact current TaxaLens and ButterflyLens consumer pins and authority-boundary tests retained. |
| 15: pilot | Deterministic fixture coverage and ablation mechanics are useful software evidence. | Fixture outputs are not live Flickr/GBIF/YOLOE/BioCLIP evidence and cannot select a release strategy. | Current regression fixtures retained; one-off Build Week replay/report commands removed. |
| 16: release report | The report preserves the fixture-only limitation and fail-closed release. | Live execution, review and scientific selection remain pending. | Software/fixture completion is explicit and separated from scientific completion. |

## Layer disposition

### Removed

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
   been removed. The classification-v3 text-cache builder and the remaining
   plumbing, rolling, and M5Pro crop/cascade benchmark harnesses were removed
   first. The production cascade, crop generator, object-evidence buckets,
   cloud wrappers, and rolling worker are now removed as one coherent cutover;
   the stable detector schema retains nullable historical crop fields.

4. **Alternate/default/reference-first workflow layers.** Completed on
   2026-07-19: the alternate workflow selector, legacy/default and
   reference-first stage plans, and one-off Build Week runtime configuration
   and permit wrappers were removed. `run` now resolves one adaptive graph.
   `RunStage` now contains exactly the adaptive graph nodes. Concrete reference
   commands identify themselves by executable command rather than a second
   display-only stage vocabulary.

5. **Inert switches and forwarding helpers.** The first cleanup removed the
   hidden registry `--skip-classification` switch, two no-op wrappers, and stale
   manifest/report fields (143 lines removed). Continue removing only helpers
   proven to forward without adding validation, identity, resource ownership,
   or a consumer boundary. Later passes removed the test-only reference-work
   facade, registry query forwarder, bucket triage policy, crop image loader,
   and unused artifact-path declarations.

6. **Historical command and infrastructure layers.** The one-off Build Week
   benchmark/report command stack, standalone cloud poller, storage compactor,
   duplicate shard-path helper, and model-worker state wrapper had no current
   application caller. Their current invariants remain owned by the reference
   commands, metadata poller, evidence-shard path, WorkStore and embedding
   cache respectively.

7. **Compatibility defaults and upgrades.** The v2 support-manifest upgrader
   and implicit `PromptVariant` provenance defaults were used only by their own
   compatibility tests. Current manifests and prompt variants now require
   explicit versioned identity.

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
- The `ExecutorFactory` protocol in the current detection pipeline. It is an
  actual resource-injection seam used to bound executor ownership and test
  concurrency; it is not a constructor wrapper or alternate architecture.
- The on-wire label `reference-first-run-artifacts-v1.0.0`. It versions an
  immutable artifact layout and is not a selectable reference-first workflow.

## Cross-repository alignment baseline

TaxaLens committed HEAD is
`e845dd98493979f37b04dbb6538e0d7b8758ca11`, matching BioMiner's current
TaxaLens pin. TaxaLens owns product replay, human review, geographic impact,
and occurrence release. BioMiner must publish evidence and maturity; it must
never authorize TaxaLens occurrence release.

ButterflyLens committed HEAD and BioMiner's audited pin are
`1ca6d9e15b03147df26a15deb309d32aed7ea9f7`. Exact committed-object tests cover
the current consumer schemas and confirm that BioMiner does not export database
primary keys, reviewer identities, service credentials, sensitive-location or
occurrence-release decisions, Darwin Core publication authority, or ALA
submission authority. The consumer's retired fingerprint v1.0 reader is
compatible with BioMiner because the producer already emits v1.1 only.
The 16-commit movement from `3d6486da87f32136c35e29aeed6cb6291da66a17`
removed a separate analyst runtime; the audited contract, Flickr-policy,
migration, and database-fixture Git trees remained identical.

Sibling worktrees remain read-only during this goal. Alignment is performed by
reading committed objects and changing BioMiner's producer/contracts/tests,
not by importing product implementation code.

## Implementation phases and gates

### A. Remove misleading legacy authority

Status: complete.

1. Remove comment enrichment/review and all automatic promotion.
2. Remove frozen cascade-specific evaluation compatibility after confirming no
   current report or consumer depends on it.
3. Separate concrete reference-operation labels from adaptive production-stage
   vocabulary.

Gate: no removed symbol is importable or parseable; current adaptive schema,
science, CLI, and orchestrator tests pass; migration documentation states how
to retain historical artifacts without reinterpretation.

### B. Connect the adaptive runtime

Status: software and fixture contracts complete; live application handlers and
human inputs remain external and deliberately fail closed.

1. Remove the disconnected plan-only dynamic command layer; use the concrete
   `references` application commands for artifact work.
2. Bind each live canonical stage to one explicit owning adapter; unconfigured
   live stages fail closed instead of falling back to old transformations.
3. Keep manual-review stages awaiting explicit signed review input.
4. Fail closed on missing, stale, incompatible, or partially written inputs.

Gate: a bounded fixture-backed run crosses every automated stage, pauses at
manual gates, resumes from immutable review artifacts, and produces verified
handoffs. Dry-run remains side-effect free. No live scientific result is
claimed.

### C. Align product consumers

Status: complete at TaxaLens pin
`e845dd98493979f37b04dbb6538e0d7b8758ca11` and ButterflyLens pin
`1ca6d9e15b03147df26a15deb309d32aed7ea9f7`.

1. Re-audit TaxaLens current committed consumer and preserve its exact pin.
2. Preserve the exact ButterflyLens pin and committed-object parity tests for
   every BioMiner-owned export role and downstream-owned release gate.
3. Report unavailable product-owned artifacts explicitly rather than creating
   placeholders or inferring approvals.

Gate: pinned-object contract tests pass for both products and all authority
boundaries fail closed under mutation.

### D. Consolidate and release

Status: architecture cleanup complete; scientific release remains a non-goal
until live evidence and human review exist.

1. Re-run import/dead-code/duplication audits after removals.
2. Consolidate only exact repeated validators where ownership and error
   semantics remain clear.
3. Run focused, adaptive, downstream, lint, and complete regression gates.
4. Update current-state and migration documentation with measured before/after
   source, test, module, and command-surface reductions.

Gate: the repository exposes one production architecture; documentation and
tests do not describe deleted compatibility paths; every incomplete live-data
or human-review dependency is explicit.

## Final verification

- The implementation-only regression gate passed 2,622 tests with the one
  agent-pack manifest consistency test deferred until this report and current
  state were hashed (138.22 seconds). After manifest regeneration, the complete
  suite passed all 2,623 tests in 137.45 seconds.
- Ruff and Vulture at 90% confidence report no findings on the cleaned source
  tree; a production-source search has zero whole-word `legacy` matches.
- Exact committed-object tests cover both sibling consumer pins. No sibling
  implementation code was copied and no sibling worktree was modified.
- Migration notes document each deleted runtime boundary. Historical reports,
  caches, checkpoints and user-owned runtime/configuration paths were not
  rewritten or deleted.
- `uv build` produced both the source distribution and wheel successfully; the
  root and retained `dev vision` CLI help surfaces both parse successfully.
- GitHits was not called, in accordance with the user's directive; each task
  provenance record states `skipped_user_directive` and has no solution ID.

## Non-goals

- No historical artifacts, checkpoints, reports, caches, or user runtime state
  are deleted or rewritten.
- No fixture result is promoted to a biological, performance, or release claim.
- No manual review is synthesized.
- No product repository is made dependent on BioMiner implementation imports.
- No helper is removed solely because a static tool reports it unused.
