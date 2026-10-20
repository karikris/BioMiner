# Phase 3: Polars 2 compatibility

Phase starts on `main` at `e723354ec27016c17b0c5ad36b9caef01d88548d`.
The verified phase-2 push handshake is
`/home/toffe/.local/state/biominer/upgrades/20261021/phase-2-pushed.json`,
SHA-256 `be7e78758322b74adec6e5a7ea820350d0a3946d3111aa5e7b6e78023afa8885`.
The continuation gate returned `ready` for phase 3.

Unrelated scientific changes remain outside this task's commits and tests.
The frozen patched phase-2 fallback and active development environment are
preserved. No production scientific run or hosted-service qualification is
claimed here.

## Task 3.1: explicit list expansion

Both production `explode` calls now specify
`empty_as_null=True, keep_nulls=True`. Sampling still rejects empty, null,
mixed-null and unresolved query definitions. Original query IDs and duplicate
discovery hits remain intact; structured provenance ordering remains explicit.
Flickr reports retain duplicate label counts, lexical ties and observed query
hit distributions, while absent instrumentation remains distinct from zero.
Two test-only expansion calls now declare the same semantics.

GitHits inspected tagged Polars 1.44.2 implementation and warning tests and
the [Polars 2 migration guide](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/docs/source/releases/upgrade/2.md).
The two explicit arguments preserve the approved row contract across the
default change. No upstream implementation was copied.

The new tests produced nine expected warning failures before the repair.
Afterwards, the four focused test modules passed with deprecations treated as
errors: **56 passed in 2.05 seconds** on Polars 1.44.2, and **56 passed in
2.56 seconds** on Polars 2.0.0 with custom Arrow
`25.0.1+biominer.f918dfd84c8c`. Both use Python 3.14.7.
Independent automated specification and quality review approved the six
changed files; no human review is claimed.

Receipts are retained beneath
`/home/toffe/.local/state/biominer/upgrades/20261021/phase3/task3.1/`:

| Artifact | SHA-256 |
|---|---|
| Expected failing test receipt | `6fd0677fc0844cf98cf0c4e573cd138ee9977fb20e484ae620e8d161c3ea09f0` |
| Polars 1 test receipt | `dcd336e344c084b45616890f960dc09a70796f8f5ca6595f9bbf3cad99b16d57` |
| Polars 2 test receipt | `a775d1d55c5e99626a67259e7e494a3dd8db90cf393683a091b5f9ddf7523d75` |
| Six-file source hash inventory | `0a2eeb17efc6204d67fc2c060fea4b33469564c031ba0f6f357357360b912fed` |

Task 3.1 commit: `18bc040d4dcabfbfc26b3c575cb6380bed86460c`.

## Task 3.2: membership, artifact contracts and resource controls

Task starts at the task-3.1 commit above. Shard merging now explicitly treats
successful photo IDs as a collection with `Series.implode()`. Public tests
preserve sorted, disjoint success/failure IDs and terminal operational failures.
They exposed the previous deprecation warning before the repair.

Storage regressions preserve nullable lists, duplicate IDs, source order,
UTC timestamps, missing-column null insertion and explicit empty schemas.
Strict cast failures retain the previous artifact byte-for-byte; interrupted
batch ingestion publishes no partial artifact. The 56 storage/shard checks
passed on both Polars versions (1.17/1.27 seconds). Another 81 existing workload,
enrichment and translation checks passed on both (7.84/7.89 seconds).

The source audit reviewed lazy collection ordering and the full upstream
migration-guide headings/removals. Relevant production paths already sort
species/context/provenance outputs explicitly or use ordered streaming batches.
Timestamp parsing uses `str.to_datetime`, not removed temporal casts.
An AST audit of 329 committed production Python files found no direct Polars
CSV, SQLContext, Enum, cut/qcut or entropy calls; a second bounded scan found
none of the inspected removed method/parameter names. Dynamic lookup and
assignment aliases are outside that absence claim.

An all-terminal-failure shard run already produces a zero-column result frame
and rejects sorting under both Polars majors. The regression preserves its
fail-closed behavior and verifies absence of final merged artifacts. This
pre-existing limitation is not successful all-failure handling or a schema fix.

The new standard-library supervisor establishes budgets before a fresh Python
worker imports Polars. Defaults are four threads, 1073 decimal MB of engine
memory and 4294 decimal MB of spill, rounded down from the approved GiB limits.
Explicit positive budgets and thread counts are retained; an explicit memory
fraction alone does not receive an extra default absolute cap. Each run owns a
private spill directory. Linux child supervision includes detached descendants,
and cleanup precedes a finished receipt. Timeouts, signals, worker failures,
changed scratch ownership and receipt/cleanup failures remain visible.

The supervisor/native-launcher tests passed **111 tests in 7.05 seconds**.
Independent specification and quality review found an initial-receipt failure
could leave scratch behind; a regression reproduced it and the correction
was independently rechecked. Both the supervisor and compatibility subset
are now approved by automated review, without a human-review claim.
The coordinator's combined run of all eight focused modules then passed
**248 tests in 12.85 seconds**. Its receipt SHA-256 is
`31615417514d26748b74973d5c1a6d23b25095a15b9649b95568a5785c2c8965`.

Exploratory real Polars 2 probes used a 200,000-row streaming sort. With an
explicit 4 MB memory/64 MB disk budget, IPC spill files were observed and
the query succeeded. A 16 MB disk budget created files then rejected the query
with the expected budget error; both runs removed their private scratch.
These probes establish local behavior, not RSS limits or final admission.

The root compatibility report is
`phase3/task3.2-root/report.json` under the external evidence root, SHA-256
`6ea6ab293717fa318d0167c7df39c76d60c9f87dcdcb36251fc3a9fd5756bcc9`.
Source-audit SHA-256 values are
`a354a2eb30e61873736d6b34877e98f4f5f1963e5a4944ff9b7f3834e41d4272`
and `1ee124a01ae551a4e0ed8e5493f8b40f1295e5de43026173a4b6c8de16ca11e3`.
Supervisor probes and receipts are retained under `phase3/task3.2/`.

Task 3.2 commit: `6255b0a26a88139a240e3790129f6813ead7ac34`.

## Task 3.3: locked target and paired qualification

Task starts at the task-3.2 commit above. The dependency requirement is now
`polars>=2.0.0,<3`, and the lock changes only `polars` and `polars-runtime-32`
from 1.44.2 to 2.0.0. The installed `biominer` console entry point now invokes
the resource supervisor before importing the CLI. Offline `uv --no-config lock
--check --offline` passes without changing an existing environment.

Two opt-in qualification cases require explicit, hashed runtime contexts;
missing context fails rather than skips. The semantics case checks identical
application sources, all installed distribution versions, image/classifier
behavior and logical Parquet artifacts across the two Polars versions. Only
declared version labels and physical Parquet encoding hashes are excluded from
logical comparison; original physical hashes remain recorded. The resource
case binds the loaded supervisor and synthetic worker to committed sources,
observes actual spills, requires the specific disk-budget error, and verifies
scratch removal plus development-runtime integrity before and after execution.

Both cases collect successfully. The preflight receipt is
`phase3/task3.3-preflight/report.json` under the external evidence root, SHA-256
`5fbdafc3e04d9efd5424275b89921b1e15289f67150749fdad25db399a77da32`.
Collection is not qualification. Actual same-source runtime builds, paired
execution, clean-snapshot full regression and the phase push handshake remain
pending at this implementation commit.

Changing dependency configuration does not activate a target runtime. The
previous development receipt no longer verifies against the updated main
configuration; its environment remains untouched. The frozen phase-2 fallback
uses its own preserved configuration and remains independent of this change.
