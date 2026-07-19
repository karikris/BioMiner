# BioMiner geography-conditioned dynamic pooling — final report

Status: **software and fixture goal complete; live scientific work pending**.

BioMiner now implements the complete target-aware, geography-conditioned
global/local reference-pooling architecture from reference indexing through
candidate scheduling, pool construction, cached scoring, review/statistics,
selective remediation, CLI planning and exact downstream handoffs. The final
technical gate passes 3,233 tests plus named strict, adaptive, dynamic-pool,
CLI, schema/parity and handoff suites. All ten scientific-semantics checks pass.

The production result is deliberately fail-closed: **insufficient evidence**.
Zero of 24 pilot variants are eligible, runtime settings are unchanged, and no
candidate strategy, pool variant or fusion method is selected. There are zero
completed real reviews. No occurrence release is authorized.

## Goal identity and delivery

| Field | Value |
|---|---|
| Starting BioMiner SHA | `c7eaa9bf3696a25a0c8229837819dccec4fb9d66` |
| Phase 0 design SHA | `299914548b407b439cd36d1aa99397b41aa827f1` |
| Pre-report workflow SHA | `98c64ec27e0aaa6aa3da333b3e4d37df3fc1c30b` |
| Final report commit | `self`; resolved by Task 16.2 completion record |
| Branch | `main` |
| TaxaLens pin | `e845dd98493979f37b04dbb6538e0d7b8758ca11` |
| ButterflyLens pin | `1cea643623f2f20a2bea72afc754c7b194db3278` |
| Primary model | `bounded-model` |
| Reasoning effort | `xhigh` |
| Codex session | `019f660b-6398-7a22-a1e8-ad5bc6abc23c` |

The goal contains 33 numbered tasks, 105 numbered subtask provenance records
and one explicit corrective record. Every subtask's exact commit and subject is
listed in its immutable `task_*_completion.json` report. Thirty-two prior task
pushes are verified; the final Task 16.2 push and exact remote SHA are resolved
by its separate completion record after this self-identifying report commit.
Work remained directly on `main`; no feature branch, PR, force-push or history
rewrite was used.

## What was built

The implemented dependency chain is:

```text
registry + admitted references + immutable embeddings
  → normalized reference geography / index / anchors / neighbours
  → canonical Flickr photo / organism / scoring-unit / geography partitions
  → complete family + geography + visual + safety candidate union
  → geography-first / family-first-safe / parallel-union schedules
  → immutable global / local / safety pool plans, members and shortfalls
  → cached full-frame query vectors and cached family/candidate/pool matrices
  → raw family / global / local / nearest / top-k / disagreement components
  → four complete provisional fusion rankings
  → representative review design + separate targeted failure discovery
  → source-independent splits / calibration / grouped quality / outcome lanes
  → exact reference impact / selective pool, matrix and score-record rerun
  → immutable TaxaLens and ButterflyLens evidence handoffs
```

The minimum requested artifact surface is implemented as versioned contracts.
Where the final implementation uses a safer or more precise canonical name,
the machine report records the exact mapping. Important replacements include:

- `pool_shortfalls.parquet` → `dynamic_pool_coverage_shortfalls.parquet`;
- review sampling frame → `dynamic_pool_audit_frame.parquet` plus separate
  probability register/sample and targeted queues;
- dataset split manifest → `dynamic_pool_evaluation_splits.parquet`;
- training features → `dynamic_pool_features.parquet`;
- final review-required output → the explicit unresolved-candidate lane; and
- revision/rebuild/rescore manifests → typed pool/matrix/record impact,
  selective-rerun plan/receipt and `flickr_rescore_plan.parquet` contracts.

All contracts, grains, versions, validators and authority boundaries are
indexed in `docs/schemas/geography_conditioned_dynamic_pooling_contracts.md`.
Their presence does not mean a live artifact was materialized. Current evidence
is software-contract and deterministic-fixture evidence unless explicitly
labelled otherwise.

## Fixture metrics versus live metrics

### Reference index and Flickr work

| Metric | Live | Deterministic fixture |
|---|---:|---:|
| Reference geography/index rows | unavailable | 2 / 2 |
| Global anchor rows | unavailable | 2 |
| Geographic-neighbour rows | unavailable | 18 |
| Species with local/global-only support | unavailable | not aggregated as a species claim |
| No-geo fallbacks | unavailable | 1 pilot case |
| Flickr photos processed | unavailable | 7 scoring cases |
| YOLOE-eligible photos | unavailable | not run; cached vectors used |
| Unique Flickr/query embeddings | unavailable | 7 |
| Query embedding consumptions / reuse events | unavailable | 14 / 7 |
| Encoder invocations / image materializations | unavailable | 0 / 0 |
| Family matrix requests / hits | unavailable | 14 / 13 |
| Unique pool matrices / references / hits | unavailable | 35 / 100 / 65 |

No live current-policy reference corpus or Flickr/YOLOE/BioCLIP run was
executed. The pilot's zero encoder work proves cached-vector reuse in that
fixture; it does not claim live encoding throughput.

### Pools, candidates and performance

| Metric | Result |
|---|---|
| Average live global/local/total pool size | unavailable; no live pool artifacts |
| Fixture rows per unique pool matrix | 2.0 (70 rows / 35 matrices) |
| Pool expansions / shortfalls / nearest distance | unavailable for live work; expansion not executed in the bounded pilot |
| Candidate strategy selected | none |
| Reviewed family/species candidate recall | unavailable; zero real reviews |
| Complete-union preservation | 1.0 across every fixture strategy |
| Fixture target recall at 5 | 1.0 across every strategy; structural, not reviewed accuracy |
| Hard-family counterfactual loss | 1 of 2 eligible correct species (0.5) in the deterministic counterexample |
| Average fixture candidate set | 5 taxa |
| Dynamic scoring throughput | unavailable; comparable runtime not instrumented |
| Peak MPS memory | unavailable; cached-vector fixture did not run MPS |
| MPS policy limit | 536,870,912 bytes |

The 24-variant pilot covers three candidate schedules, global-only versus
dynamic global/local pools and four raw fusion methods. It emits 168 result
projections from 14 score work items. Six located cases change target raw score
under local evidence without changing the top fixture candidate. This is
structural raw-score behavior, not biological performance.

## Selection, review and statistical evidence

The frozen nine-criterion policy has three passing software/fixture gates:
observed embedding/matrix reuse, zero target-pruning regressions and no
unsupported statistical claims. Six criteria block selection:

1. target/candidate recall is fixture structural evidence, not reviewed recall;
2. reviewed precision and its confidence lower bound are unavailable;
3. family and geographic subgroup estimates are unavailable;
4. zero effective real reviews leaves the full 86-review shortfall;
5. comparable computation is not instrumented; and
6. MPS peak memory is not measured.

| Review/statistical field | Value |
|---|---:|
| Representative human reviews completed | 0 |
| Decisive human reviews completed | 0 |
| Effective real review count / shortfall | 0 / 86 |
| Target precision objective / confidence level | 0.95 / 0.95 |
| Estimated precision / lower bound | unavailable / unavailable |
| Calibration | unavailable; no eligible reviewed labels |
| Sufficient species / families / geographies | 0 / 0 / 0 |
| Fixture representative / targeted items planned | 7 / 7, kept separate |
| Fixture analysis strata | 7; not live statistical support |

Targeted work has null inclusion probabilities and weights and cannot support
unweighted population claims. Statistical support cannot create an item-level
human decision. Human review alone cannot create an occurrence release.

Current and resulting settings share fingerprint
`sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`.
The decision fingerprint is
`sha256:43d034983485e789b8fa7c0428131f13c826d695781a28130b011e13b3bf3fb2`,
the selection table is
`sha256:7368a623b9fbd9a665e2ae135c7da24c2adf9ed3a87b8e36241a3a2f14a676ec`,
and the integrated pilot report is
`sha256:ade039c9914c6fc720773eee7fbfb2141ff087f3abf869d9ab56b5f54dfa5d09`.

## Outcome and remediation status

| Field | Count/status |
|---|---:|
| Human-reviewed release candidates | 0 |
| Statistically supported screening candidates | 0 |
| Live review-required candidates | 0 |
| Unreviewed occurrence exports | **0** |
| References statistically flagged/reviewed/excluded | 0 / 0 / 0 |
| Affected species rebuilt | 0 |
| Flickr records selectively rescored | 0 |
| Fixture query-embedding reuse events | 7 |

Zero here means the live action did not occur; it is not evidence of zero
biological errors. Remediation cannot run legitimately until eligible review
and statistical evidence creates a versioned flag.

## Downstream handoffs

TaxaLens consumes an immutable artifact-first handoff pinned to
`e845dd98493979f37b04dbb6538e0d7b8758ca11`. Exact contract tests preserve
review probabilities, unavailable quality, geographic impact and the boundary
that review is not release.

ButterflyLens consumes an immutable artifact-first handoff pinned to
`1cea643623f2f20a2bea72afc754c7b194db3278`. Exact committed JSON Schema,
Python, TypeScript, migration, pgTAP and vocabulary fixtures pass. BioMiner
cannot assign reviewers, write ButterflyLens database IDs, bypass RLS or
authorize release.

Compatibility is verified through committed objects; no sibling implementation
source was bulk-copied. A live consumer database or publication was not run.

## Final technical gate

| Gate | Result |
|---|---:|
| Full pytest | 3,233 passed in 169.62 s |
| Ruff, all tracked Python | passed |
| Strict mode | 77 passed in 17.04 s |
| Adaptive mode | 68 passed in 7.56 s |
| Dynamic pooling | 428 passed in 60.94 s |
| Scientific semantics | 21 passed in 0.34 s |
| CLI | 146 passed in 8.73 s |
| Configured schema/parity | 44 passed in 0.84 s |
| Handoffs | 85 passed in 5.87 s |
| Locked dependency audit | 0 known vulnerabilities; local unpublished package skipped |
| Secret scan | 453 classified heuristics; 0 private-key/common live-token prefix matches |
| Large-file inspection | 0 tracked files over 1 MiB; no source media/model weights |
| `git diff --check` | passed |

Focused suites overlap the full regression by design. The technical receipt
retains exact commands, counts, environment, lock hash, classified secret
findings and limitations. Secret scanning remains heuristic.

## Scientific-semantics acceptance

The release gate verifies that:

- wrong-family evidence cannot remove the target or complete candidate union;
- geography is evidence, never identity proof or biological absence;
- automatic GBIF references remain provider-asserted provisional support;
- pool scoring consumes cached embeddings without encoder or image work;
- raw evidence is not probability;
- statistical support is not human verification;
- no unreviewed outcome lane can enter an occurrence export;
- insufficient strata retain null estimates and exact reasons;
- targeted work cannot support unweighted population quality; and
- TaxaLens/ButterflyLens maturity and authority boundaries survive handoff.

All 70 requested acceptance criteria are mapped in the JSON report. Architecture,
dynamic-pool, scientific and cross-repository criteria pass as software or
exact fixture contracts. Efficiency and statistical-quality criteria pass at
the contract/fixture layer while live throughput, MPS and population estimates
remain unavailable. The final task-push criterion is resolved only after this
report commit is pushed and its Task 16.2 closure record is committed.

## GitHits contribution and current directive

The goal contains 139 GitHits provenance records: 33 task, 105 numbered
subtask and one corrective record. Status totals are:

| Status | Records |
|---|---:|
| Pattern-only, no code copied | 3 |
| Available | 1 |
| Timeout | 23 |
| Unavailable | 1 |
| No qualified result | 1 |
| Skipped by user directive | 110 |

GitHits made no direct code contribution. Its material architecture influence
was limited to early precedent: baseline/cache/artifact audit framing,
artifact-first handoffs and one typed source/image/candidate-locator example.
BioMiner's actual architecture comes from its local ADRs, existing adaptive
contracts, exact consumer fixtures, tests and human governance. No external
implementation was copied.

After the user disabled GitHits, no further call was made. Every remaining task
and subtask records `githits_status: skipped_user_directive` and
`solution_id: null`; no repository, result or contribution was invented.

## Claims, limitations and next action

Allowed claims:

- the full software architecture, schemas, validators, CLI plans, selective
  rerun graph and exact handoff fixtures are implemented and tested;
- all 24 fixture variants were structurally evaluated and cached-vector reuse
  was observed;
- the correct policy result is insufficient evidence, zero eligible variants,
  unchanged settings and no default; and
- the complete technical and scientific-semantics software gates pass.

Blocked claims:

- empirical superiority or production selection of any strategy/pool/fusion;
- biological accuracy, reviewed precision, calibrated quality or statistical
  support from fixture ordering;
- measured production throughput, avoided runtime or MPS peak memory;
- any live Flickr item is verified, release ready or published; and
- geography absence is biological absence or provider labels are ground truth.

Live work still required includes current-policy reference/Flickr/YOLOE/BioCLIP
execution, comparable timing and MPS instrumentation, at least 86 effective
source-bound representative reviews, subgroup floors of 30 independent records,
reviewed calibration/held-out evaluation, any legitimately triggered
remediation, live consumer ingestion and separate occurrence-release approval.

Exact next action: run one bounded, instrumented current-policy live execution
that materializes the reference index and dynamic-pool score artifacts, records
comparable throughput and MPS peak memory, then execute the frozen
representative human-review plan before reconsidering any production default.
