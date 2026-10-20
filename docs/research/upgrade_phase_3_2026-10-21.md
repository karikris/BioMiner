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

Task 3.3 commit: `1d2b176a348dcfb528af100e2bd45c662e07390d`.

## First committed-source runtime pair

The external builder completed 29 commands successfully in 180.435 seconds.
Both fresh isolated generations and separate development environments passed
their runtime verification. The two application wheels contain 331 identical
application members. Wheel comparison permits only the Polars requirement in
package metadata and the resulting `RECORD` entries to differ.

Evidence lives under `phase3/pair-1/` in the external evidence root.
The `built-pair.json` SHA-256 is
`5a19fd3bffdfad8bad7f4eb99ff1a692cd64c165c28cfad7642461a24e444acf`.
The runtimes remain unactivated. This verifies isolated test runtimes; the
profile and development metadata do not claim production promotion.

| Pair member | Profile SHA-256 | Development receipt SHA-256 |
|---|---|---|
| Polars 1.44.2 / custom Arrow 25 | `aec002589809ac6778407e96c1922900e8fccfc5d87b9f7a405621aec6961df2` | `cd4e39b8abdd49fb9f2ce3ac8bb20114974d5aa8465cefeb067c3fab2675b04e` |
| Polars 2.0.0 / custom Arrow 25 | `f131fcff442daa75f91b35380198fd7977d692f4e4b707edc40cf07675e8b06d` | `710a047e1b98ff89d9a7a0db97ff8704985d51a8fb4419f1ba409d521a65d33f` |

This first pair is preserved as historical build evidence. The source preflight
below superseded it for phase qualification; no phase checks ran against it.

The pair's scoped security audit checked all 47 native wheel members in each
of four generation/development environments, plus the 101 declared native files
per environment. Actual imports retain Python 3.14.7, OpenSSL 3.5.9, AnyIO 4.15.1,
libxml2 2.15.4, libxslt 1.1.45 and libiconv 1.19; pip is absent. The patched zlib
claim rests on matching compiled bytes and source/object provenance, not its
version string. Only the two Polars distributions differ between pair members.

`phase3/security-pair-1/security-final-v2.json`, SHA-256
`8597555ee21dcb9fe5d81c151e692c1f86b1c9f59cefb5169315963bed250e19`,
preserves all nine finding IDs/scopes and the separate lxml attestation gap.
Its predecessor-aware security validation passed. Native-SBOM limits,
unassessed package metadata and the zlib vendor-catalog discrepancy remain
explicit; this does not repeat the earlier process inventory or qualify live
services.

## Task 3.3 correction: distinguish source from output

Before phase execution, the full source-context preflight rejected ordinary
API-key variable expressions in four production files. The output credential
regex was being applied to code bytes that were only read for hash validation.
The failure receipt is `phase3/source-validation-preflight-failure.json`,
SHA-256 `d616292e7c79cc9d2d8f4eb31c3ac0b0390adcd31c373c28928f845860a49b3b`.

The qualifier now reads source bytes for exact committed/execution/helper hash
comparison without interpreting expressions as output. File existence, safe
paths, commit identity and all hash checks remain enforced. Credential filtering
for actual logs, manifests and reports is unchanged.

A committed-source regression failed before the correction and passed after it;
the existing secret-output rejection still passes. The qualification and phase
gate suites passed **96 tests in 9.12 seconds** in the retained reviewed run.
Its receipt SHA-256 is
`32f44e03eccd951035d2c546f9cc07676242913adfd21cc58d5765623851bae4`.
A corrective task-3.3 commit
preserves the original history rather than amending it. The phase must build
and qualify a fresh pair bound to that corrected commit; the first pair and
unexecuted gate preparation remain preserved.

The first corrective task-3.3 implementation commit is
`ff59a7c758d62ae7d0485b77edcc8ff441bceb35`.
Its fresh `phase3/pair-2/` build completed all 29 commands successfully in
194.258 seconds; `built-pair.json` SHA-256 is
`8aa72771492f4e3990349f876ff9a835cd5a34b68c95c7849b02f92e7fe9e2c1`.
Both complete 714-file source contexts then passed validation without narrowing.
The preflight receipt SHA-256 is
`c04ce803bb2c5f76675d7018bd7ded6fb721e7a3e94f0c2e7d2e9a285ea53eb4`.

All four installed `biominer --help` console checks passed across generation
and development environments for both Polars versions. Each used the permanent
interpreter, completed its resource lifecycle and removed its private scratch.
`phase3/pair-2-console-checks/consoles.json` has SHA-256
`0cb363b5e50a1439535467c57a86c3f4d40c7f7c5a09db7fa2d1a5d16d386a72`.
This checks the installed bootstrap; it does not activate a runtime or exercise
live services.

## First phase execution and harness correction

The paired semantics case passed **1 test in 27.48 seconds** on the corrected
commit. Image, classifier and logical Parquet observations match exactly.
Single-file Parquet physical hashes differ between Polars versions, while
batched Parquet hashes match. Both physical observations remain in the original
receipts; no existing artifact or fingerprint was overwritten.
The comparison receipt SHA-256 is
`d3cd7dbc7074ddbaa08f9f6df5a606d676cb2ce9a62a1c4f0d99b9fb17c28fb1`.

The real resource case passed **1 test in 23.34 seconds**. With 4 MB engine
memory and 64 MB disk, the streaming sort succeeded and 82 spill files were
observed. With 16 MB disk, it created two observed spill files and raised the
required disk-budget error (expected worker exit 7). Both private scratch
directories were removed and process cleanup passed. These are engine-budget
observations, not process-RSS or aggregate limits.
The qualification receipt SHA-256 is
`b65347e7333ac0cce6e9670b515580a1ad8105abbd9bfa1f47c73d9861845a99`.

The full suite then reported **11 failed, 3,083 passed in 138.29 seconds**.
All eleven failures rejected the operator's temporary directory, which was
outside `/tmp` and the repository's `.native-runtime` root. The production
output policy remains unchanged. The failed log SHA-256 is
`5979cd13138027602d522824587d5c388c04d052cb1f0fc1fe49f48ca5b33e24`.
The supervisor returned 125 after cleaning descendants; no phase candidate
or push handshake was published.

A revised external operator creates a unique private directory under `/tmp`.
Its actual emitted launcher passed **77 native-runtime tests in 5.36 seconds**,
but still returned 125. This isolates the temporary-path repair from the
remaining descendant-accounting failure. The preserved handoff is
`phase3/operator-gate-prep-v3/handoff.json`, SHA-256
`044ab4affd3721b3092052a2b3757cc1125fa83c1790281f28ef4e57fb0733ec`.
The supervisor failure requires separate diagnosis and a clean rerun.

The bounded diagnostic then reproduced the actual native timeout test:
pytest passed, but supervision returned 125. All three descendant observations
identified the same directly adopted child in state `Z` (already exited).
The observer recorded process state without changing the original ownership
result. Its receipt, `phase3/supervisor-correction-1/diagnostic-receipt.json`,
has SHA-256
`d826b509a0de479a50af80e8c38211f623a8630c673de83433e80558dfea2fd3`.
This establishes the cause of that reproduced failure; it does not imply that
arbitrary descendant-cleanup failures may be ignored.

## Task 3.3 correction: collect exited adopted children

After collecting the direct subprocess, the supervisor now uses nonblocking
`waitpid` to collect already-exited adopted children before its existing
descendant check. It preserves the direct subprocess's exit status. Live
descendants still cause exit 125 and cleanup; timeouts still cause exit 124
and cleanup.

Real subprocess regressions reproduced two false failures before the repair.
Afterward, all five behavior cases passed: parent exits 0 and 7 with an
already-exited child, a live detached child, and both timeout variants.
The supervised qualification, gate and native-runtime suites passed
**176 tests in 13.62 seconds**, with outer exit 0. The focused receipt is
`phase3/supervisor-correction-1/green-focused.json`, SHA-256
`79312067a87900be77e5784aa00fd604540a4b67d9d5788836411b99d54c4eb3`.

GitHits source review of Tini and dumb-init corroborated nonblocking child
collection while preserving the primary exit status. Both inspected sources
are MIT licensed; no upstream implementation was copied. Returning success
without collecting exited children, or suppressing live-leak failures, was
rejected.

This is another additive task-3.3 corrective commit. The original attempts
remain evidence; a fresh pair and complete phase gate must bind the corrected
source before phase 3 can be pushed.

The supervisor correction was committed as
`81ac71e79e029ba79d8c441d52d38ea871d76a03`. Its fresh third pair completed all
29 build commands in 179.315 seconds, and both 714-file context validations
passed. `phase3/pair-3/built-pair.json` has SHA-256
`4cebd06db5ce3cbff7d06e12f8ec89be2e894a8ba2585f4b34ca35759906c4fe`.
All four installed console checks also passed; their receipt SHA-256 is
`6ccdf4a584f82b50a9a7d46c5aef0962fb1ac6180c4587d0210f87bc86e6fae0`.
No new phase gate ran against this pair.

## Newly established Arrow 25 security blocker

While reviewing Arrow 26's changes, primary upstream
[PR 50843](https://github.com/apache/arrow/pull/50843) established a heap
overwrite in three enabled compute kernels: chunked boolean forward fill,
backward fill, and masked replacement. The exact Arrow 25 source used by the
existing custom wheel contains all three faulty allocations. The upstream
fix uses the existing fixed-width allocation helper, which handles bit-packed
boolean data correctly.

An independent review confirmed that the exact upstream patch applies to both
the Arrow 25 implementation and its regression tests without unrelated Arrow 26
changes. The installed Arrow 25 library remains affected even though the
bounded BioMiner call-site audit found no direct use of these functions.
No CVE/GHSA identifier is claimed from the primary issue/PR evidence.

This invalidates the plan's assumption that the existing Arrow 25 fallback
needs no further security repair. The plan now requires an exact backport,
fresh native build and qualification, a safe replacement fallback, and a new
paired Polars comparison before phase 3 can pass. The third pair and earlier
receipts remain historical evidence. No gate was relaxed, no new phase push
occurred, and the previously retained fallback is no longer described as
currently safe for this defect.

## Task 3.3 correction: require the Arrow 25 backport

The builder now applies the exact Apache Arrow upstream patch
`a769c291e01093b73d03a075179cf7a09bf92ad8` to the pinned Arrow 25 archive.
Both changed files are verified before and after patching; the original license,
authorship and notice remain. No unrelated Arrow 26 source is substituted.
The captured recipe, patch application and resulting sources are retained in
immutable build inputs. Their digest defines the wheel's local version.

Admission requires that complete reviewed chain, including the exact recipe
SHA-256 `dfb95a9f94962bfbc1b2b48d16cba5670f52dc6aa5f195d0f08c0dc76c828541`.
Relocated evidence resolves through retained checked bytes without rewriting
historical manifests. Missing, tampered or relabelled receipts fail closed.
Resume also rejects changed source inventories, output symlinks and completed
artifacts, preserving previous attempts. Outer compilation is bounded to four
jobs and nested CMake/Make work is serial.

Qualification now requires a snapshot containing the current admission helper
before any runtime subprocess. Fifteen native observations are defined across
the three affected kernels: large boolean chunks, sliced offsets and nullable
masks, empty chunks, all-null chunks and numeric controls. They independently
check values, buffer capacity, chunk behavior and unchanged inputs. These
actual kernel checks have **not yet run**. The Polars comparison reads the
exact Arrow version and wheel digest from verified profiles and requires both
sides to use identical Arrow bytes.

The final combined builder, admission, snapshot, assembly, qualification and
phase-gate unit suites passed **287 tests in 21.45 seconds**, outer exit 0.
Source hashes were captured before execution and remained unchanged afterward.
The receipt is `phase3/arrow25-backport-combined/checks.json`, SHA-256
`6c37a7507edaa0e9dcdbf26d54b1ac62cba2a3de9e37e5407d0caa45a4797b11`.
The exact full-archive source preparation and strict resume also passed,
without compiling or executing Arrow. Independent specification and quality
reviews approved the final source changes.

A fresh isolated build environment retains all 2,321 file hashes and twelve
exact package pins. GitHits metadata, batched upgrade reviews and changelogs
support the setuptools 84.2.0 and scikit-build-core 1.1.1 selections. The
builder receipt is `phase3/arrow25-backport-builder/builder.json`, SHA-256
`9c6d8d2485ba5c37a88228771bcc7a614bedefdf1fcf1fb5fe5a5c834f18dc9f`.
Catalog/deprecation gaps remain explicit. This source commit does not close
the Arrow security finding: compilation, actual native behavior, replacement
fallback qualification, registered-entrypoint migration and the fresh paired
phase gate are still required.

The unchanged upstream patch contains four blank context lines prefixed with
the unified-diff space marker. `.gitattributes` disables only end-of-line
blank checks for that exact patch path so `git diff --check` accepts its
required bytes; whitespace checks remain enabled for implementation files.
The patch digest remains unchanged.
