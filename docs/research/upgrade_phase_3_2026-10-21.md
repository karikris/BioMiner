# Phase 3: Polars 2 compatibility

Phase starts on `main` at `e723354ec27016c17b0c5ad36b9caef01d88548d`.
The verified phase-2 push handshake is
`/home/toffe/.local/state/biominer/upgrades/20261021/phase-2-pushed.json`,
SHA-256 `be7e78758322b74adec6e5a7ea820350d0a3946d3111aa5e7b6e78023afa8885`.
The continuation gate returned `ready` for phase 3.

Unrelated scientific changes remain outside this task's commits and tests.
This report records successive task evidence. Phase-2 artifacts remain historical
evidence; later sections record the replacement of affected entrypoints and
qualification of the current two-patch Arrow 25 fallback. The old environment
is not a currently safe fallback. No production scientific run or hosted-service
qualification is claimed here.

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

## Patched Arrow 25 build and compiler correction

The corrective source commit is
`0dee88002c517354b2770dcee76310e7b4510f1e`. Its first actual C++ build
completed, but Python wheel configuration failed because scikit-build-core
selected `clang` from the approved Python interpreter's build settings.
Clang was unavailable in the isolated build path. The first attempt's final
receipt is `phase3/arrow25-backport-builder/build-finished.json`, SHA-256
`d4e72bd1a5e06bcb55831321d88a1e9acabea973d33616a144fe05c24765b0d4`.
The observed compiler maximum was four; this attempt produced no wheel.

GitHits source inspection confirmed that scikit-build-core 1.1.1 uses
Python's recommended compiler only when `CC`/`CXX` are absent. The reviewed
retry selects `/usr/bin/gcc` and `/usr/bin/g++`, whose resolved binary hashes
match the original C++ configuration's `cc`/`c++`. It uses the unchanged
committed recipe and immutable build-input manifest, preserving the version
`25.0.1+biominer.e9697ff28bd8`.

All 5,267 archive files matched their original or reviewed transformed bytes.
The failed backend had added one generated version file. The retry preserves
that exact file, the failed receipt and all overwritten logs before removing
only that generated file and rerunning the strict source checks. The
independent pre-resume review is
`phase3/arrow25-backport-resume-prep/independent-pre-resume-review.json`,
SHA-256 `a6f0aa10d87b3f5cebfe83a2c6dc2aad01203be6d4f6094febe0bf8cf11d7c82`.
Historical first-attempt paths that the recipe updates are mapped to their
preserved copies in the retry's `preserved-failure.json`.

The retry passed source and manifest validation before compilation. No
source guard, dependency pin or security gate was relaxed.

The retry completed in 561.345 seconds with outer exit 0, no monitor or cleanup
error, and an observed maximum of four compiler frontends. The resulting wheel
is `25.0.1+biominer.e9697ff28bd8`, SHA-256
`2657f0336db3a09d780af50542db4ac539b23fdff3a8a7adbd706882593508ca`.
The compiled receipt SHA-256 is
`a418f8246e2f8c6fd10e6e25272d632a82e21978cd815013f1ecfc73e3e157da`.

The independent final artifact audit binds patched source to the actual unity
object, compute linkmap, library, installed prefix and wheel. All 37 wheel ELF
members match their receipt; seven C++ wheel libraries match the final prefix.
All 37 resolved host files match the already-patched deployment inventories.
The 154 static archive origins include one explicit, byte-identical c-ares
relocation from its build directory into the owned install prefix. Recompilation
changed 49 archive hashes; external archives and static OpenSSL are unchanged.
The final audit is `phase3/arrow25-generated-audit-prep/final-artifact-3/audit.json`,
SHA-256 `e0d30a6649184a77b9c9938a46acd5e2546f0c9bee18b6a46bebb0e8b685b617`.
This establishes artifact identity; actual kernel qualification remains a
separate requirement.

The first patched runtime-pair build stopped at command 16, before publishing
a complete pair. Its assembler rejected conflicting host `libssl.so.3` hashes:
the new Arrow build correctly records the patched deployment library, while
the unchanged PostgreSQL-driver build receipt records the library present at
its original build. Neither source receipt was rewritten. The failed command
receipt is `phase3/arrow25-pair-1/commands/016-p1a25-assemble.json`, SHA-256
`ee948ec74c6c68847d6b9f7da87ed1b18cb957a7efc5565cb7beb8673c71ad20`.
The stopped pair remains unqualified evidence.


## Explicit deployment host reconciliation

Starting from `0dee88002c517354b2770dcee76310e7b4510f1e` on `main`, task
3.3 now accepts an explicitly checksummed deployment profile for host library
inventory only. It verifies current package policy, file ownership and bytes,
retains stronger mandatory version floors, and records every historical to
current mapping. It preserves both historical build receipts and the exact
selected profile. Wheel and interpreter conflicts still fail; no historical
Arrow wheel is admitted through this option. The candidate remains unready
until installation and qualification succeed.

GitHits inspection of auditwheel at
`63a51b2f62936e38b6504faa60db5ca042559f97` informed the separation between
external host libraries and wheel contents. No upstream code was copied.
The unavailable Ubuntu CVE tracker lookup supplied no evidence.

The focused assembler suite passed **40 tests in 1.64 seconds**. An initial
neighbor run had five failures and 170 passes because the refresh fixture
loads a separate native policy module without the new synthetic Arrow pins.
The fixture now supplies those same synthetic pins to its private module;
production checks remain unchanged. The corrected assembly, host-refresh,
admission and snapshot suites passed **175 tests in 12.35 seconds**, supervised
outer exit 0. Exact command, source hashes and log are retained in
`phase3/assembler-host-neighbors-2/checks.json`, SHA-256
`9cae71ecb6353eac8a71dd0e7e4438bf942f49aa517723d9a7eeec4d9fb3b230`.
The original failed result remains in `assembler-host-neighbors-1`.

An isolated preflight used the actual failed pair's five artifact receipts
and exact new assembler. Assembly, artifact validation and host refresh each
exited 0 under supervision. All 44 selected host mappings matched current
files. Candidate and refreshed profile bytes were identical, SHA-256
`d6b1a4cc83b511b0965c9eb3dac4f58c0c8d78b32cf77e737fffbbd110cad8ee`.
The result is `phase3/assembler-host-preflight-1/result.json`, SHA-256
`ea809b52107a622a3bdc2eaf3a4482b8f9ffccff12dea2b4826e3f3633f94d05`.
Independent integration review:
`phase3/assembler-host-preflight-1/independent-integration-review.json`,
SHA-256 `3afbbbf66666c8f7b3c8805d226b3a85e593f13552e7f709e61e055535c12ea4`.
This preflight installed no environment and ran no Arrow kernels. Runtime
qualification, registered runtime replacement and the phase push remain
pending. Unrelated scientific work is excluded from this task's commit.


The inventory correction was committed as
`5d905ade29b916ca91c9859de42d22fde080e0fa`. The next fresh attempt,
`phase3/arrow25-pair-2`, passed assembly and host refresh, then failed at
command 18 while freezing the profile. No environment was installed or
activated. Two historical observations in the reconciliation mapping used
`path` and `sha256`, which the snapshot format reserves for live references.
The strict recursive verifier therefore rejected the intentionally historical
libssl/libcrypto hashes. Current selected host records were correct. The
failed tree and exact command receipt remain preserved; the correction must
use explicit historical observation fields and retain strict validation of
live references. The original five build receipts remain unchanged.


The historical rows now use `recorded_path` and `recorded_sha256`, preserving
both observations and ownership without declaring stale bytes as current
inputs. The selected deployment rows retain normal live references. The
unchanged freezer is exercised by two new regressions: retained history
survives freezing, while changed current library bytes still fail verification.
The focused assembler and snapshot suites passed **71 tests in 6.40 seconds**.
Their three expected pre-fix failures and successful run are retained in
`phase3/assembler-history-reference-1/{red,green}.json`.

The extended actual-input preflight passed all five supervised commands:
assembly, host refresh, freeze, frozen verification and frozen artifact
validation. Its result is `phase3/assembler-history-preflight-1/result.json`,
SHA-256 `a89aee4d645737a6cca9ef4ae558f43009a7a04c5288eedf0277910171c55338`.
It uses exact current source bytes and the previous attempt's actual artifact
receipts, in a fresh directory. No kernel tests, environment installation,
activation or finding closure is claimed by this preflight.

## Actual pair qualification exposed a second upstream defect

The fresh pair at `dec69d2e53a5d775fa3b36bd9c7564f12689b7f5` completed all
30 build, installation and verification commands. Its receipt is
`phase3/arrow25-pair-3/built-pair.json`, SHA-256
`eab7696fed966eec62b1a8cbf8b94ac2322e77429895aa5bae36550cb7817b82`.
Both application wheels contain identical application payloads; their Polars
requirements differ as planned. These checks establish identity only.

Native qualification then stopped at `sliced_boolean.forward` in the first
runtime's feature check. The preceding artifact, loaded-version, TLS and
import-order checks passed. The second runtime's native checks did not run.
The actual outer execution exited 1, retained at
`phase3/patched25-pair-3-native-outer/execution.json`, SHA-256
`f95a19a714d1f4589935f7176aac1ae00a95917f26f5ca275a689c08e7d8a981`.
The independent prerequisite review retains this blocking result.

A supervised reproduction recorded 21 operations: 15 incorrect sliced Boolean
results and six passing controls. For example, offset-one input
`[null, true, null]` forward-filled to `[null, false, false]`, rather than
`[null, true, true]`. A separate nearest-non-null oracle matched the committed
expected values. Offset-zero Boolean and sliced integer controls passed.
The analysis is `phase3/arrow25-sliced-investigation/analysis.json`, SHA-256
`8fc9c3638eb6cb3e83a8acd4f90de3f9f3615d1a688c1f0a0c549e275433a51b`.

GitHits search, grep, exact source reads and release-history inspection located
the missing `ArraySpan.offset` in `CopyDataUtils<BooleanType, ArraySpan>`.
[Upstream PR 51240](https://github.com/apache/arrow/pull/51240) fixes it in
commit `0efbdec7a10da1a4043fc1210c61e73c88bc9b4f` and adds sliced-array
regressions. The exact mail patch SHA-256 is
`6c64e115aa53621ae7e1992417b26ab7c6c51fb265d3e64216580f7dcae87d50`.
The Arrow 26 changelog lists GH-51223 under improvements; GitHits classified
the entry `no_impact`. That classification is contradicted for this enabled
Arrow 25 runtime by the actual regression and source comparison. No CVE/GHSA
or additional memory-safety claim is inferred from the correctness defect.

The separate identity auditor initially rejected one conservative inventory
entry: a 4,008-byte Parquet fixture whose filename contains `.so` as part of
`.some-named-index`. The corrected external auditor verifies its exact wheel
membership, owner, hash and non-ELF bytes separately. All four runtimes have
37 matching Arrow ELF files plus this explicitly classified data asset.
Fourteen focused auditor checks, fresh execution and independent review pass;
the original failure remains preserved. Its bounded handoff is
`phase3/security-patched25-auditor-v2-prep/handoff.json`, SHA-256
`19c58815189684c72104086e4b7f91ea55cce12515bafeca98f519c6aca37059`.
This identity result does not waive the native behavior failure.

The allocation-only pair remains unqualified. Shared development replacement,
registered runtime cutover, security closure and the phase-3 push remain
pending. The next corrective build must include both exact upstream patches
and pass the unchanged behavioral gate before those steps can proceed.

The corrective recipe now applies both exact patches in order. It checks every
stage's input and output bytes, verifies all original archive files, and binds
the three final source files separately from the shared test file's intermediate
hash. Resume rejects partial or altered patch chains. Admission requires this
exact recipe and both patches; allocation-only receipts cannot qualify under
the new policy. Original patch bytes and the native expected-value cases are
unchanged.

The new rejection tests failed before implementation. An initial recipe run
then exposed a synthetic-fixture hash collision between unrelated sample
contents; the assertion now checks the overlapping file's own final hash.
The retained final
recipe run passed 47 tests, and actual full-archive preparation plus strict
resume passed under supervision without compiling. That actual source check is
`phase3/arrow25-offset-recipe-1/actual-source.json`, SHA-256
`198cbeadc2c0055bb95500a86be415dc012a50600758be0c4adedc73a484dc79`.

The combined builder, admission, snapshot, host-refresh, assembly, qualification
and phase-validator suites passed **346 tests in 22.11 seconds**, with outer
exit 0 and all tested source bytes unchanged. The command and log are bound by
`phase3/arrow25-sliced-combined/checks.json`, SHA-256
`930b97c43f871e6ab89fc0382a6900100509ac6118d9fb46d58a04c464787faa`.
The recipe SHA-256 is
`65077fc43ba4542eb051b0a07a53ba147614551d192e774e13773aef0567da83`;
the admission policy SHA-256 is
`78db872a76a1557f2386f803be53c1fe5f5fb9b21719f56913c8cc32a70402fd`.
Independent source review covers both changes and the isolated build operator.
The review is `phase3/arrow25-sliced-source-review/final-source-operator-review.json`,
SHA-256 `5214f19686716ce63fc36f7f5b9360d98935d5cd5b3663d393cf0aaf8eb92c3a`.
These results authorize the fresh build; they do not assert native kernel
qualification, migration or phase completion.

## Two-patch native qualification and build-cleanup race

At `08af8fbf2514d26f381ba1f7b3949b2bfdcf8565`, the fresh Arrow build completed
in 1,112.137 seconds, with at most four compiler frontends observed and no
budget, monitor or cleanup error. The wheel is
`25.0.1+biominer.d601668ee1b9`, SHA-256
`2f7c85d9049f130870bb71a4c719d144f67eae79b8b32c76521f33b25df24112`.
Independent artifact review binds both ordered patches, three final source
files, both shared-header consumers through their objects and linked libraries,
37 wheel ELF members and seven exact prefix copies. Its receipt is
`phase3/arrow25-sliced-artifact-audit-actual-1/independent-actual-artifact-review.json`,
SHA-256 `5cd84a848ecb838c03812318f724adb5ded0cf2234660fd84ffdfbed68d29d99`.

Pair 4 completed all 30 build commands and its outer process exited 0. The
pair receipt is `phase3/arrow25-pair-4/built-pair.json`, SHA-256
`932599e6c19cd29ea1ab65fda001fdd6001827451e4b30e5971242c3562bc4e5`.
Both runtimes passed all five native checks, including all 30 Boolean/control
observations across the pair and the previously failing sliced-array values.
The actual native report is
`phase3/patched25-pair-4-native/paired-native-qualification.json`, SHA-256
`718d0dbc89df539f3cde0f8d8a4bf005636c7bfa682c695801b5009e97afa394`;
its outer supervisor and runner both exited 0 without interruption.
The four-environment identity audit and all four installed console smoke
checks also passed. These are local qualification results, not migration or
hosted-service evidence.

The paired Polars semantics and bounded spill checks passed in 29.38 and
24.56 seconds. Full regression then reported **1 failed, 3,209 passed in
96.90 seconds**. `test_timed_out_build_terminates_compiler_descendants`
observed a compiler child in state `R` immediately after the native build
timeout handler returned. Its unchanged assertion requires absent or exited
children. The full-suite log is
`phase3/patched25-pair-4-prechecks/full_suite.log`, SHA-256
`013d30bf9ce1ab148da87e40b7ff055b38516f4d0c94d6f0c4bdfd990e477090`.
The outer precheck receipt records supervisor/runner exit 1 and no completed
precheck report; it remains a failed prerequisite.

Source inspection established that `_bounded_process` sends `SIGKILL` to the
recipe's process group, then `communicate()` waits only for the direct child.
With file-backed logs, the leader can exit before a signalled compiler child
finishes. [Python's subprocess contract](https://docs.python.org/3.14/library/subprocess.html#subprocess.Popen.communicate)
and GitHits inspection of
[psutil's bounded process waiting](https://github.com/giampaolo/psutil/blob/d094fd6c/psutil/__init__.py#L1949)
support treating signal delivery and completed termination separately.
No psutil code or dependency is added. The correction must observe the owned
process group until its members are absent or terminal, fail closed at its
cleanup deadline, and retain the real descendant regression. A test-only
delay or accepting runnable children would not repair the lifecycle contract.

The independent blocked review is
`phase3/patched25-pair-4-independent-review/prerequisite-blocked-review.json`,
SHA-256 `a92ecafc6e218648a09cd0aa1087b4076dbf3ac84c69f4697991684e5130714f`.
Pair 4 has not replaced the shared runtime. A fresh source-bound pair and
complete successful prerequisites are required after this correction; the
verified Arrow wheel can be reused because its recipe and binaries are
unchanged. Cutover, security closure and the phase-3 push remain pending.

The correction now shares one five-second monotonic cleanup deadline between
leader communication and observation of the original process group/session.
It preserves the original timeout or interruption after successful cleanup,
and reports a distinct cleanup failure for live or unverifiable members.
No additional process group is signalled; escaped sessions remain under the
separate qualification supervisor's existing ownership contract.

Nine deterministic public-build regressions failed against the preserved
original source. With the correction, `tests/test_native_runtime.py` passed
**124 tests in 6.44 seconds**, including the unchanged real compiler-descendant
test. The qualification, native-check, profile-assembly and host-refresh
neighboring suites passed **115 tests in 6.88 seconds**. Both supervised runs
exited 0 and retained unchanged tested source hashes under
`phase3/native-timeout-cleanup-1/`. The native-runtime source SHA-256 is
`33fc0e8ed8bf91770447a4731bcbd75e3b81e1b8c903f1ed7c187062969b5238`;
its test SHA-256 is
`67c170983cdc7f25c35351ca09bd98f7655b97e086240660f5e0bc9a9b20dca1`.
These focused results do not replace the pending fresh full-suite prerequisite.

## Qualified replacement pair and registered runtime migration

The timeout correction is committed as
`e6f32872c2fc113272d484ec1d0fbf64752c8007`, the final task-3.3 implementation
boundary. Fresh pair 5 completed all 30 build commands in 176.685 seconds.
Its archived application sources are identical across Polars 1.44.2 and
2.0.0; both use the unchanged, independently reviewed two-patch Arrow wheel
`25.0.1+biominer.d601668ee1b9`. The build receipts bind the new source boundary
without rebuilding or relabelling that unchanged native artifact.

Both runtimes passed all five native cases, including 30 Boolean/control
observations across the pair, TLS, import orders, installed artifacts and
checkpoint/resume. The prerequisite semantics and resource checks passed in
29.24 and 23.59 seconds. The clean archived checkout then passed **3,219 tests
in 98.25 seconds**, including the unchanged real compiler-descendant regression.
Native and precheck outer supervisors and runners exited 0 with no interruption.
All four installed console checks and the four-environment identity audit
passed. These results supersede pair 4's failed prerequisite; its failure
receipts remain preserved.

The new shared development environment passed fresh verification and independent
review. Its editable main checkout retains two unrelated scientific source/test
deviations; neither is represented as qualified by the clean archived suite.
The fresh registered-consumer inventory found no consumers or unknown processes
in its declared readable same-user scope. The reviewed cutover repeated scans
while holding the activation lock, atomically replaced `.venv`, verified the
replacement, and retired the two exact old environment directories with mode
`000`. Both execution probes returned `EACCES`; the final scoped scans remained
clear. No jobs were killed or restarted. The outer cutover exited 0 in
152.984 seconds without interruption.

The shared target is `.native-runtime/development/arrow25-pair-5-shared/env`.
The safe fallback is the immutable Polars-1 generation under
`phase3/arrow25-pair-5/p1a25/assembly/.native-runtime/generations/qualified/env`
in the external evidence root. Reopening either retired old environment would
invalidate the security resolution; historical isolated unsafe pairs remain
unapproved retained evidence.

All paths below are relative to the external evidence root
`/home/toffe/.local/state/biominer/upgrades/20261021/`:

| Artifact | SHA-256 |
|---|---|
| `phase3/arrow25-pair-5/built-pair.json` | `db7b836d005779420868924d591190f46c3574e61d24546a1136743812f87015` |
| `phase3/patched25-pair-5-native/paired-native-qualification.json` | `f35b64a1c0032e2f41ab67f8ce62bc327aa832c44efb702231ee7c1ec5b7123d` |
| `phase3/patched25-pair-5-prechecks/prechecks.json` | `872b40748a6a889281fac8bb72bac5e8505bb9b369eb26534139f73830e0c3cf` |
| `phase3/security-patched25-pair-5/pair-identity.json` | `22a55cd85d3d9d56760305747fcb972ef3637f26d9016208506856bf31c9da3b` |
| `phase3/patched25-pair-5-shared/shared-development.json` | `19781eb89b37913f94df844b5e70d93d6dd487a94a64bd00a0fde6cd4dd3c753` |
| `phase3/patched25-pair-5-cutover/migration.json` | `785a91770c84e315c640e176fff7bb9a5cb097efc04caf3c7b0ef78863127982` |
| `phase3/patched25-pair-5-cutover-outer/execution.json` | `f815cddce49f4ccb82778361ff06541a1afdeff339f6a8a312025cb307837b3b` |
| `phase3/patched25-pair-5-independent-review/security-evidence-review.json` | `d3f2f372cf4e12a51acebcbfd0a53115c3583deba8f9768d39f68900ef45bc47` |

This is registered-local runtime evidence, not host-wide vulnerability clearance
or coverage of other-user mappings, remote launchers or future manual launches.
The final security report and official phase checks are recorded below;
the external push handshake remains required. Live PostgreSQL/S3, Arrow 26,
performance and production migration remain later phases; no scientific release
is implied.

The security finalizer then exited 0 with supervisor/runner exit 0 and no
interruption. Its `phase3/security-patched25-pair-5-final/security-final.json`
has SHA-256
`7017b267c28212d70cc310b4f966cf96413b3aa7011573a8b06588ac525d2a31`.
It retains all ten finding IDs and their original scopes, plus the separate
lxml attestation gap, with evidence-bound resolved dispositions. The sliced
Boolean correctness defect is retained as a prerequisite; no additional CVE or
GHSA assignment is invented. Package/advisory coverage limits and native-SBOM
limits remain explicit. Independent final security review is
`phase3/patched25-pair-5-independent-review/final-security-review.json`, SHA-256
`ef255e6967a77a7d82303fcd3cfa7043d9cc8bc91d8b4d9758858bcdd15afef6`.

## Official gate interrupted by exhausted temporary storage

The first official run at the corrected `e6f32872` boundary passed semantics
in 34.70 seconds and resources in 23.67 seconds. Its full suite stopped with
**267 failed, 2,189 passed, 28 errors in 82.63 seconds**, followed by pytest's
internal `OSError: [Errno 28] No space left on device` while flushing output.
The 36 GiB `/tmp` tmpfs was full. Individual failure details could not be
rendered, so those counts do not establish 267 independent source regressions.
The outer run exited 1 after 187.688 seconds; security did not run and no
passing `phase.json` was produced. The failed outer receipt is
`phase3/patched25-timeout-gate-run-1-outer/execution.json`, SHA-256
`0b62c06004557731739707f8ad65ecbd670f14f88d4119be6518e2351d336ac0`.

The retained diagnosis is
`phase3/official-gate-tmpfs-diagnosis-1/diagnosis.json`, SHA-256
`fd9064373f5446ff89af5631ffb434201b9844e15b9808b0e78c10925edbe2ef`.
Independent blocked review is
`phase3/patched25-pair-5-independent-review/official-gate-enospc-blocked-review.json`,
SHA-256 `9e1f1845378593504c59f7d0342a1bde3c69a6342e3160764f56fbf2a8ded80a`.

Moving pytest scratch outside `/tmp` would conflict with the native output
policy; placing generated test files inside the archived checkout would conflict
with its cleanliness check. Both checks remain enforced. Recovery therefore
archives inactive, successful-test scratch owned by this task before removing
any of it. Ownership and reference inspection identified six eligible roots
without durable hash references into their contents; a failed-test root was
explicitly excluded. The current failed run, referenced evidence and other
sessions' temporary directories remain preserved. A fresh complete official
run is required after verified capacity recovery; the failed run is not reused
as passing evidence.

The first scratch archival attempt stopped at its post-archive process check,
without deleting any source files. That operator did not persist the rejected
scan, so the original failure cannot be classified as an active consumer or an
unreadable process. A later clear scan does not resolve that historical gap.
The successor persists every sanitized scan before asserting, raises on traversal
errors, and verifies owner read/search permission on every archived directory.
It completed all six archives with supervisor/runner exit 0 and no interruption;
its aggregate is `phase3/tmp-reclamation-2/archives.json`, SHA-256
`d78a62675f9661ba90c6a4523a4cb2156ffdb3a10420e99ac4601321df0ed8c6`.
The first failed attempt and its partial archives remain preserved separately.

Independent archive and cleanup reviews passed before the exact six-root
removal. Cleanup then exited 0 in 89.45 seconds with no interruption, retained
the verified archives, and recovered 6,118,596,608 bytes available in `/tmp`.
The independent actual-cleanup review confirms all six removals and preserves
the excluded failed root and current failed-gate scratch. Its receipt is
`phase3/tmp-reclamation-2/independent-cleanup-actual-review.json`, SHA-256
`589a7443f8544f5a66bc41077f1c50c7551e1c10989476feb6b5b1059533a29f`.
The actual removal receipt is `phase3/tmp-reclamation-2/cleanup/finished.json`,
SHA-256 `70b454bf667f754b97abe140e276c27da5f01845e18a23125d8c6c9347415a95`. The retry requires a fresh
4 GiB capacity observation before preparation and again immediately before
execution; capacity is not reserved against concurrent writers. The tested
source, native output policy, four check commands and security finding IDs
remain unchanged.

## Successful official phase-3 retry

The fresh official run completed all four mandatory checks against the clean
pair-5 P2 source snapshot at `e6f32872c2fc113272d484ec1d0fbf64752c8007`.
Its emitted `python-phase3` launcher freshly verified the permanent runtime
before each check and retained the unchanged isolated environment and supervisor.
No partial result from the failed run was replayed.

| Check / exact arguments after `python-phase3` | Actual result |
|---|---|
| `-m pytest -q scripts/upgrade_checks/test_polars_semantics_cases.py::test_paired_polars_versions_and_public_artifacts` | 1 passed in 48.11 seconds; no logical differences |
| `-m pytest -q scripts/upgrade_checks/test_polars_resource_cases.py::test_polars_engine_spill_and_disk_exhaustion` | 1 passed in 23.16 seconds |
| `-m pytest -q` | 3,219 passed in 95.24 seconds |
| Pinned `read_security.py` with exact report, context and predecessor hashes | exit 0; ten finding IDs and one gap preserved with reviewed resolutions |

The actual wrapper and security argv are bound by
`phase3/patched25-capacity-gate-prepared-1/plan-reviewed.json`, SHA-256
`e703827a03a804eb0ef2f740c015de7dfc02f30fb08d30da2ac4ca3be2568d5c`.
The whole run finished in 247.912 seconds with supervisor and runner exit 0
and no interruption. Its outer receipt is
`phase3/patched25-capacity-gate-run-1-outer/execution.json`, SHA-256
`990a1deebd689404087f4ff5aab466dcd5ee71253b63a99b33025ebd0ca99a78`.
The immediately preceding capacity check observed 6,118,596,608 bytes free.

The result candidate binds all bounded semantics/resource outputs, direct nested
references, four command receipts/logs, capacity evidence and actual outer
execution: `phase3/patched25-capacity-gate-run-1/phase.with-outputs.json`,
SHA-256 `20e39bbb7ea2563423c5688c6ac12f93601b31e6b1b9874e24f12946701dfd34`
with 628 unique artifact references. The original failed run, unclassified first
archive failure, their partial artifacts and the successful recovery remain
separate immutable evidence.

| Task | Implementation commit |
|---|---|
| 3.1 | `18bc040d4dcabfbfc26b3c575cb6380bed86460c` |
| 3.2 | `6255b0a26a88139a240e3790129f6813ead7ac34` |
| 3.3, including corrective ancestry | `e6f32872c2fc113272d484ec1d0fbf64752c8007` |

These checks establish the scoped Polars-2/patched-Arrow-25 compatibility and
runtime security evidence. Arrow 26, the four-runtime matrix, live PostgreSQL/S3
qualification, three-run performance comparison and final migration remain
phases 4–6. No production scientific release or universal security clearance
is claimed.

Independent post-run spec and quality reviews approved the actual results with
no findings. Their receipts have SHA-256
`ba2b34f35440b2a47d3e9e25aec8d98726cff206d32d622c70b9b1777280c043`
and `64f5eb80e79d0ca91078a1246d5b329316153d5e55ee0a418194b621f5886716`,
respectively, beneath the run's `independent-post-run-reviews/` directory.
The unchanged phase validator accepted the final reviewed candidate at
`phase3/patched25-capacity-gate-run-1/phase.final-reviewed.json`, SHA-256
`56e4a74237426298bc4398a51020db1eeb2a5a2d9dfc3eb50c2654175e18a0f5`.

This report and its curated provenance form a separate documentation-only phase
boundary. The create-only external `phase-3-pushed.json` handshake must bind the
actual resulting main commit and observed remote SHA after push; its digest
cannot be embedded here without circularity. Phase 4 implementation remains
conditional on the existing continuation validator returning `ready` for that
actual handshake. No manual success receipt substitutes for these checks.
