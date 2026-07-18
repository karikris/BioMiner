# Current BioMiner state and active-goal safety

## Observed repository baseline

Updated from the local repository on 2026-07-18. Confirm Git and reports again
at the start of every task.

- Repository: `karikris/BioMiner`
- Default branch: `main`
- Phase 0 dynamic-pooling design baseline:
  `299914548b407b439cd36d1aa99397b41aa827f1`
- Latest verified dynamic-pooling implementation commit:
  `30319fa7b7e9f24256556b293cf2e2db6e6ce2e7`
  (`docs(pilot): report geography-conditioned pooling`)
- Latest verified dynamic-pooling task report commit:
  `65b98ce4bed1d8b799ef0396fea5515921621e68`
  (`docs(provenance): record task 15.1 push`)
- Active goal family: geography-conditioned dynamic global/local reference
  pooling
- Dynamic-pooling phases: Phase 0 baseline, audit and design complete; Phase 1
  cross-repository contract alignment complete; Phase 2 geographic reference
  indexing complete; Phase 3 canonical Flickr photo, organism, association,
  candidate, scoring-geography and work-partition contracts complete through
  Task 3.1.3. Phase 4 implements all three target-preserving candidate
  schedules, per-k strategy metrics, hard-family-pruning counterfactual and
  fail-closed selection gate through Task 4.2.3. Phase 5 implements the
  immutable reference-pool policy, deterministic observation planner,
  diversity/class balancing, coverage shortfalls, raw uncertainty evidence,
  cached identity expansion and bounded stop decisions through Task 5.2.3.
  Phase 6 implements the canonical YOLOE route contract, explicit full-frame
  BioCLIP input policy, durable one-time Flickr embeddings, bounded family,
  candidate and pool matrix indexes and deterministic cache-local work ordering
  through Task 6.2.3. Phase 7 implements complete raw family, global and local
  component evidence plus explicit disagreement, rank movement and coverage
  through Task 7.1.4. Task 7.2 preserves every component through four
  versioned provisional fusion methods, complete per-method candidate rankings,
  ties and alternatives through Task 7.2.3. Phase 8 separates encoding from
  vector scoring, adds memory-aware image batching and bounds shared
  pool-matrix working sets through Task 8.1.3. Task 8.2 reports observed
  embedding and matrix reuse plus plan-derived selective score reuse without
  guessed savings through Task 8.2.3. Phase 9 defines the complete
  dynamic-pool audit frame, stratified probability sample, separate targeted
  failure queue and complete fail-closed occurrence-release review queue
  through Task 9.1.4. Task 9.2 defines the preregistered review-evidence policy,
  dynamic exact-binomial reference count with explicit weighted/grouped design
  inflation, and immutable adaptive milestone evaluation through Task 9.2.3.
  Phase 10 creates transitive reviewed-Flickr source-independence components
  and freezes balanced calibration/validation/final-test assignments through
  Task 10.1.2 plus its explicit post-push balance correction. Task 10.2 builds
  a 75-dimension raw-evidence feature table, fits grouped-OOF sigmoid-calibrated
  route models on calibration rows, reports independent validation reliability
  and selects screening-only thresholds from conservative precision lower
  bounds through Task 10.2.3. Task 10.3 projects a complete, mutually exclusive
  human-reviewed release, screening-only and unresolved partition through Task
  10.3.3. Phase 11 Task 11.1 reports the same grouped/weighted quality contract
  overall and by family, genus, species and five explicit geographic levels
  through Task 11.1.4. Task 11.2 maps audited failures to typed human actions,
  source-bound GBIF review candidates and design-gated Flickr follow-up through
  Task 11.2.3. Phase 12 Task 12.1 propagates reference changes through exact
  dynamic-pool, matrix and individual scoring-record dependencies through Task
  12.1.3. Task 12.2 binds content-addressed reference/Flickr reuse into a
  preflighted selective pool, matrix and record execution DAG through Task
  12.2.3. Phase 13 Task 13.1 integrates the explicit dynamic-pooling stages,
  dependency order and non-automatic human gates through Task 13.1.3. Task 13.2
  adds typed fingerprinted settings, seven plan-first CLI commands, persisted
  plan validation and a fail-closed live-adapter boundary through Task 13.2.3.
  Phase 14 publishes immutable TaxaLens and ButterflyLens handoffs verified
  against exact committed consumer contracts. Phase 15 Task 15.1 freezes and
  executes a seven-case fixture-backed dynamic-pooling pilot across three
  candidate schedules, global-only and dynamic global/local pools, and four
  raw fusion methods, then creates separate representative and targeted review
  work registers. Task 15.2 publishes a complete 24-variant, nine-criterion
  selection table and returns `insufficient_evidence`: zero variants are
  eligible and runtime settings remain unchanged. No candidate strategy,
  pooling variant, or fusion method is selected or production-defaulted.
- Phase 1 contract alignment remains a completed historical boundary; it is
  not the current next step.
- Prior adaptive GBIF fast-start phases: 0–13 complete, with the final
  self-identifying report resolved by the commit containing
  `reports/gbif_fast_start/final_report.json`.
- Current release-boundary evidence:
  - dynamic-pooling Task 1.2 full regression: 2,660 passed in 100.67 seconds;
  - Task 1.2 focused cross-repository contracts: 33 passed in 0.17 seconds;
  - Task 2.1 final full regression: 2,730 passed in 101.37 seconds;
  - Task 2.1 reference/geography gate: 133 passed in 1.77 seconds;
  - Task 2.1 artifact round-trip: four Parquet artifacts and the JSON manifest
    passed semantic fingerprint and complete supplied physical-checksum checks;
  - Task 3.1 final full regression: 2,745 passed in 104.48 seconds;
  - Task 3.1 canonical Flickr grain gate: 118 passed in 1.01 seconds;
  - Task 3.1 artifact round-trip: five organism assignments across four
    partitions, with one shared model-input reuse and explicit no-geo evidence;
  - Task 4.1 final full regression: 2,756 passed in 102.92 seconds;
  - Task 4.1 strategy and target-preservation gate: 82 passed in 1.25 seconds;
  - Task 4.1 fixture round-trip: all three schedules retained identical
    five-taxon membership; no strategy is selected or production-defaulted;
  - Task 4.2 final full regression: 2,766 passed in 104.06 seconds;
  - Task 4.2 strategy evaluation gate: 93 passed in 7.06 seconds;
  - Task 4.2 fixture ablation: 18 metric rows across three strategies and three
    cutoffs; the hard-family counterfactual lost one of two eligible correct
    species, and selection failed closed on non-fixture evidence;
  - Task 5.1 final full regression: 2,796 passed in 103.97 seconds;
  - Task 5.1 pool policy/planner gate: 82 passed in 1.40 seconds;
  - Task 5.1 fixture round-trip: one plan, three independent members, two pool
    summaries and one complete coverage row with deterministic fingerprints;
  - Task 5.2 final full regression: 2,816 passed in 106.31 seconds;
  - Task 5.2 expansion/cache gate: 117 passed in 2.54 seconds;
  - Task 5.2 fixture round-trip: a two-member initial plan retained both cached
    identities, added three cached identities, invoked no encoder and stopped
    at the mandatory rescore boundary;
  - Task 6.1 final full regression: 2,838 passed in 104.28 seconds;
  - Task 6.1 route/input/cache gate: 173 passed in 1.19 seconds;
  - Task 6.1 reuse fixture: two Flickr photos and three route units shared one
    persisted embedding; the rerun invoked no encoder and added no model load;
  - Task 6.2 final full regression: 2,855 passed in 107.26 seconds;
  - Task 6.2 cache/order gate: 89 passed in 3.97 seconds;
  - Task 6.2 matrix fixture: family, candidate and pool caches each reused one
    matrix on the second request; candidate/pool work signatures formed two
    deterministic locality runs with independent canonical result ordering;
  - Task 7.1 final full regression: 2,876 passed in 104.89 seconds;
  - Task 7.1 focused numeric/matrix gate: 27 passed in 0.20 seconds;
  - Task 7.1 adjacent dynamic-pooling gate: 269 passed in 19.74 seconds;
  - Task 7.1 fixture: two candidates retained separate prototype, nearest and
    top-k components; one exact local-unavailable state stayed null, while a
    second fixture exposed a complete global/local rank reversal;
  - Task 7.2 final full regression: 2,890 passed in 104.98 seconds;
  - Task 7.2 fusion ablation and exact-semantics gate: 80 passed in 0.58
    seconds;
  - Task 7.2 fixture: all four methods retained two complete candidates and
    explicit alternatives; inverted global/local components produced method
    disagreement and exact score ties, with method selection left unset;
  - Task 8.1 final full regression: 2,908 passed in 110.43 seconds;
  - Task 8.1 MPS, memory, matrix and worker gate: 92 passed in 0.98 seconds;
  - Task 8.1 fixtures: five fake-MPS images formed batches `[2, 2, 1]`; a
    bounded memory retry repeated only its failed slice; three vector work items
    formed batches `[2, 1]` over three unique matrices and reported zero encoder
    or image work;
  - Task 8.2 final full regression: 2,921 passed in 106.61 seconds;
  - Task 8.2 metrics/no-guess gate: 91 passed in 3.03 seconds;
  - Task 8.2 fixture: seven embedding requests produced five reuse events and
    two materializations; seven worker-cache matrix requests plus batch sharing
    produced seven distinct reuse events; one of two score records was marked
    `reuse_prior_score`, with runtime savings left `not_instrumented`;
  - Task 9.1 final full regression: 2,944 passed in 107.24 seconds;
  - Task 9.1 sampling/release gate: 55 passed in 1.56 seconds;
  - Task 9.1 fixture: six candidate rows formed five connected
    duplicate/observation probability units and three selected reviews with
    exact inclusion probabilities; targeted failure discovery retained two
    heuristic rows with null statistical weights; occurrence release retained
    both final candidates, including a shared duplicate, in a fail-closed queue;
  - Task 9.2 final full regression: 2,970 passed in 106.83 seconds;
  - Task 9.2 statistical planning/milestone gate: 73 passed in 3.10 seconds;
  - Task 9.2 references: a one-look, one-sided 95% exact all-success design
    crosses a 95% lower bound at 59 independent decisive reviews (58 remains
    below); the default four-look Bonferroni policy requires 86 under the same
    all-success assumptions; a fixture weight/group/external design effect of
    2.2032 inflates 59 effective reviews to 130 nominal reviews;
  - Task 10.1 final full regression: 2,985 passed in 104.20 seconds;
  - Task 10.1 leakage gate: 39 passed in 0.91 seconds;
  - Task 10.1 fixture: a seven-row identity chain became one atomic component,
    while unrelated no-geo rows remained independent; 18 independent reviewed
    components froze to calibration/validation/final-test counts 7/5/6 under
    40/30/30 weights, with both supported and error outcomes in every split;
  - Task 10.2 final full regression: 3,005 passed in 109.62 seconds;
  - Task 10.2 calibration/reliability/risk-control gate: 76 passed in 5.68
    seconds;
  - Task 10.2 fixture: 36 rows and 75 features froze to 14/11/11; grouped OOF
    fitting used four folds, validation Brier/log-loss/ECE were
    0.0055986/0.0751940/0.0722312, and a permissive fixture threshold selected
    six components at a 0.6069622 conservative precision lower bound. These are
    software-fixture metrics; the default 0.95 objective and 30-item/component
    floors remain unmet and occurrence release remains unauthorized;
  - Task 10.3 final full regression: 3,030 passed in 110.65 seconds;
  - Task 10.3 language/release/calibration/outcome gate: 68 passed in 4.11
    seconds;
  - Task 10.3 outcome contract: every source item enters exactly one of a
    human-reviewed release, exact-label screening-only or explicit unresolved
    lane; screening and unresolved rows have no occurrence-release authority
    and are rejected by the verified Flickr export validator;
  - Task 11.1 final full regression: 3,048 passed in 109.73 seconds;
  - Task 11.1 grouped/weighted hierarchy gate: 84 passed in 6.23 seconds;
  - Task 11.1 quality contract: 13 descriptive metrics use the minimum of
    row-weight and independence-component effective sample sizes; overall,
    family, genus, species and five geographic levels retain targeted-exclusion
    counts and explicit null-estimate insufficient-sample states;
  - Task 11.2 final full regression: 3,071 passed in 117.36 seconds;
  - Task 11.2 escalation/queue gate: 79 passed in 2.34 seconds;
  - Task 11.2 remediation contract: quality triggers remain non-mutating human
    actions; GBIF targeting preserves `not_assessed` identity and existing
    disposition, while representative Flickr expansion has no weights or
    estimation eligibility before a probability design;
  - Task 12.1 final full regression: 3,082 passed in 121.48 seconds;
  - Task 12.1 impact gate: 25 passed in 2.28 seconds;
  - Task 12.1 impact contract: exact member/new-eligibility pool impact flows to
    declared matrix and record dependencies; unaffected identities are marked
    reusable and affected scoring rows retain reusable Flickr embeddings;
  - Task 12.2 final full regression: 3,091 passed in 121.11 seconds;
  - Task 12.2 selective reuse/rerun gate: 85 passed in 2.34 seconds;
  - Task 12.2 fixture contract: 14 planned operations partition into 7 executed,
    6 reused and 1 excluded; executed work covers 2 reference vectors, 1 Flickr
    vector, 1 pool, 1 matrix and 2 exact score records, while runtime savings
    remain `not_instrumented`;
  - Task 13.1 final full regression: 3,096 passed in 114.13 seconds;
  - Task 13.1 stage graph gate: 118 passed in 2.89 seconds;
  - Task 13.1 graph contract: 31 adaptive stages include all 9 requested
    dynamic boundaries; all active dependencies are topological, legacy/default
    compatibility sequences are unchanged, and Flickr human verification
    pauses before risk-controlled audit;
  - Task 13.2 final full regression: 3,123 passed in 115.05 seconds;
  - Task 13.2 CLI gate: 137 passed in 11.29 seconds;
  - Task 13.2 CLI contract: seven commands declare 22 exact named inputs and 15
    intended outputs; dry-run plans are deterministic and fingerprinted,
    selection readiness remains explicit, and live adapters fail closed;
  - Task 14.2 final full regression: 3,169 passed in 118.21 seconds;
  - Task 14.2 ButterflyLens contract gate: 46 passed in 3.70 seconds; the exact
    pinned consumer parity runner passed 24 schemas, 20 valid cases, 20 invalid
    cases, 20 version checks and 15 vocabulary checks;
  - Task 15.1 final full regression: 3,205 passed in 134.18 seconds;
  - Task 15.1 bounded-pilot gate: 36 passed in 12.63 seconds;
  - Task 15.1 fixture pilot: seven cases retain complete five-taxon candidate
    unions under all three schedules; 168 raw score projections reuse 14
    encoder-free vector work items; six located cases change target raw score
    under local pooling without changing top candidate, and the no-geography
    case has exact global-fallback parity;
  - Task 15.1 review plan: seven fixture items enter both a complete
    within-fixture probability design and a separate targeted queue, zero enter
    release review, no reviews are assigned or completed, and the shortfall to
    the preregistered 86 effective real reviews remains 86;
  - Task 15.2 final full regression: 3,225 passed in 164.55 seconds;
  - Task 15.2 complete pilot evaluation gate: 56 passed in 70.61 seconds;
  - Task 15.2 selection table: all 24 candidate/pool/fusion variants retain
    explicit reviewed-precision, subgroup, workload, computation, reuse, MPS,
    target-pruning and statistical-claim states; no variant is eligible;
  - Task 15.2 decision: three software/fixture gates pass, six criteria remain
    blocking, and current/resulting runtime settings retain exact fingerprint
    `sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`;
  - dynamic-pooling Task 0.2 ADR/audit gate: 17 passed;
  - strict gate: 86 passed;
  - adaptive gate: 65 passed;
  - CLI gate: 143 passed;
  - pilot/fixture gate: 66 passed;
  - Ruff lint and locked dependency audit passed;
  - type and format checks retain explicit unconfigured failing baselines;
  - live source smoke and live Papilio scientific evaluation were not run.

This is an observed baseline, not a permanent assertion. Re-read local Git,
phase reports, and current tests before relying on these numbers.

GitHub cannot show uncommitted work from a running Codex session. The local
worktree is authoritative for the active task; the final report distinguishes
the ending implementation SHA from its self-referential containing commit.

## Active-goal protocol

For the remainder of the active geography-conditioned dynamic-pooling goal,
the user has explicitly disabled further GitHits calls. Do not call GitHits.
Required task and subtask provenance records must use
`githits_status: "skipped_user_directive"` and `solution_id: null`, must state
that no call was made, and must not invent repositories, results, or external
contributions.

Before touching a file:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
git diff --stat
git diff --cached --stat
```

Then inspect:

```text
reports/geo_dynamic_pooling/
reports/gbif_fast_start/
provenance/githits.jsonl
provenance/task_pushes.jsonl
docs/architecture/geography_conditioned_dynamic_pooling.md
docs/architecture/statistical_support_and_human_verification.md
docs/architecture/adaptive_gbif_reference_admission.md
the current goal/prompt or phase report
```

If a long process may be running, inspect the repository's workstore, PID,
lease, checkpoint, manifest, and structured logs. Do not infer process state
from silence.

### Never do during another active session

- `git reset`, `git restore`, `git checkout --`, `git clean`;
- branch switching without explicit goal authority;
- pull/rebase/merge on a dirty worktree;
- repository-wide format or import rewrites;
- staging all files blindly;
- amending or squashing another session's commits;
- deleting checkpoints, caches, workstore rows, reports, or output roots;
- starting duplicate acquisition, model, evaluation, or publication jobs.

If the requested task overlaps active files, stop and report the exact overlap
instead of attempting to merge incomplete work.

## Current production direction

`ProductionRunRequest` currently defaults to:

```text
storage backend                  s3
workstore backend                postgres
reference admission mode        adaptive_gbif_fast_start
reference source                gbif
initial scoring mode            provisional_reference_ranking
Flickr release human review     required
statistical reference audit     required
stages                          adaptive reference production stages
```

Three reference modes exist:

```text
adaptive_gbif_fast_start
human_verified_strict
human_verified_flagged_only
```

The adaptive stage plan is:

```text
resolve taxon scope
→ build registry
→ geographic spread
→ compile queries
→ enqueue and poll Flickr
→ Flickr geographic clustering
→ Flickr detection and embedding
→ regional candidate generation
→ reference metadata and media
→ reference deduplication
→ reference quality routing
→ reference admission
→ reference embeddings
→ reference geography index
→ reference prototypes
→ Flickr geo/taxon partitioning
→ family retrieval routing
→ dynamic pool planning and scoring
→ provisional Flickr scoring
→ representative review-sample planning
→ Flickr human verification
→ risk-controlled audit
→ statistical reference audit
→ targeted reference review
→ affected reference rebuild
→ affected record rescore
→ final quality gate
```

Manual review stages may not be auto-completed.

## Accepted adaptive semantics

The accepted ADR is:

```text
docs/architecture/adaptive_gbif_reference_admission.md
```

It establishes:

- provisional, provider-asserted GBIF support as the default fast-start path;
- strict human-verified support as a compatibility/high-stakes path;
- targeted review and selective rerun for flagged species;
- mandatory human review before final Flickr inclusion;
- non-probabilistic provisional ranking;
- statistical auditing with human-reviewed Flickr labels;
- explicit evidence maturity and fail-closed release.

Key implementation locations:

```text
src/biominer/references/admission.py
src/biominer/references/admission_eligibility.py
src/biominer/references/admission_compiler.py
src/biominer/references/readiness.py
src/biominer/references/escalation.py
src/biominer/references/targeted_review.py
src/biominer/references/bank_revision.py

src/biominer/bioclip/reference_embeddings.py
src/biominer/bioclip/provisional_prototypes.py
src/biominer/bioclip/provisional_ranking.py
src/biominer/bioclip/reference_quality.py

src/biominer/run/stages.py
src/biominer/run/adaptive_config.py
src/biominer/run/support_dependencies.py
src/biominer/run/orchestrator.py

src/biominer/evaluation/
src/biominer/reports/
```

Verify exact filenames locally; the active goal may add or move modules.

## Accepted dynamic-pooling semantics

Phase 0 accepted two additional ADRs:

```text
docs/architecture/geography_conditioned_dynamic_pooling.md
docs/architecture/statistical_support_and_human_verification.md
```

They establish:

- immutable cached embedding identity separate from versioned global/local
  comparison-plan membership;
- an explicit, diverse global safety pool and a local pool or exact
  local-unavailable reason;
- family evidence as a batching/retrieval accelerator, never a hard gate;
- geography as candidate/reference evidence, never identity proof or absence;
- deterministic, bounded uncertainty expansion that reuses embeddings;
- raw component scores and margins as non-probabilistic evidence;
- separate human-review, calibration, statistical-support, release-readiness
  and downstream-publication authorities; and
- immutable TaxaLens/ButterflyLens handoffs pinned to committed contracts.

Phase 0 remains architecture evidence only. Implemented behavior through Phase
13 is recorded in task reports and tests; handoffs, pilot evidence and release
verification remain work in Phases 14–16.

## Legacy documentation warning

At the observed baseline:

- `README.md` still describes the old family-first funnel and 0.90 genus
  shortcut.
- `docs/production.md` still describes crop materialization and hierarchical
  production.
- `src/biominer/run/orchestrator.py` retains legacy cascade and visual-mode
  compatibility fields alongside adaptive defaults.
- `src/biominer/run/stages.py` retains legacy, strict reference-first, and
  adaptive stage plans.

Therefore:

- do not treat legacy README prose as active-goal authority;
- do not delete compatibility paths without an explicit migration task;
- do not let compatibility paths silently control adaptive output;
- update stale docs only when the active goal reaches its documentation phase
  or explicitly requests the change.

## Remaining implementation, live and human work

The implementation, migration/documentation, fixture-backed pilot, and release
verification phases are complete. This does not complete the scientific run.
The authoritative remaining-work ledger is in
`reports/gbif_fast_start/final_report.json` and currently includes:

- live current-policy GBIF acquisition and durable-media admission;
- live BioCLIP embedding, prototype, and Flickr scoring with instrumentation;
- 50 source-bound representative Flickr reviews;
- a sufficient-sample statistical audit;
- targeted reference review and selective rerun only if legitimately flagged.

Provider-asserted GBIF support remains provisional, current quality metrics are
unavailable, and no live production improvement or scientific release is
claimed.

For the active dynamic-pooling goal, the next implementation boundary is Phase
16 Task 16.1: update the README, production guide, architecture/schema docs,
human decisions and agent topic files to reflect the implemented dynamic-pool
workflow and its fail-closed pilot result. The parallel family/geography union
is only the current review projection. No strategy is selected or
production-defaulted because only fixture evidence has run. No strategy is
empirically superior, and no live dynamic pool, live score, calibrated
probability, completed human review, statistical-support result or new
release-ready occurrence is claimed by the completed phases.

## Repository map

```text
src/biominer/registry/          taxonomy, names, query compilation, spread
src/biominer/flickr_fetch/      planner, pollers, rate accounting, geography
src/biominer/geography/         cells, distance, validation
src/biominer/candidates/        regional and visual candidate unions
src/biominer/references/        acquisition, QA, admission, review, readiness
src/biominer/detection/         detector interfaces and routing policy
src/biominer/vision/            full-frame inputs, gates, rolling/cloud work
src/biominer/bioclip/           embeddings, prototypes, ranking, scoring
src/biominer/ml/                classifiers, calibration, non-match policy
src/biominer/evaluation/        labels, splits, metrics, sampling, reports
src/biominer/run/               stages, orchestration, paths, manifests
src/biominer/workstore/         SQLite/PostgreSQL work state and leases
src/biominer/storage/           local/cloud objects, Parquet, handoffs
src/biominer/reports/           structured stage and scientific reporting
tests/                          deterministic default suite
reports/gbif_fast_start/        active-goal evidence and phase reports
reports/geo_dynamic_pooling/    dynamic-pooling baselines and phase reports
```
