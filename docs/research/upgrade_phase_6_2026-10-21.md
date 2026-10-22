# Phase 6: synthetic canary and recovery qualification

The phase-5 summary was normally pushed to main as
`12ec32a0c843713a628d59aa8c4d9a870140b3e8`; local and remote SHAs matched.
Its actual finalized pushed receipt has SHA-256
`18b4b0f3979be2d2c412d108a2e3510319581cd7394edbd00ffb9fdd57d436aa`.
Fresh continuation into phase 6 passed, with result/execution hashes
`12e2c69958444f347dfa5278e2474d376ee4b344ac130b3595f503f496114355`
and `bb94911111c9ec7baee0178fb20fbac340845aa44dc592692b751ee956fe4db4`.
No Git commit intervened before the first execution. This report follows the
2026-10-21 upgrade cohort; actual phase-6 execution began on 2026-10-22.
External paths below are relative to
`/home/toffe/.local/state/biominer/upgrades/20261021/phase4/`.

The scope is six sequential public-driver commands across the complete Arrow25
recovery and Arrow26 target application candidates: two tiny synthetic probes,
two collections and 35 recovery cases per runtime. It uses artificial features,
locally generated images, fake network/storage clients and local/injected state.
It does not establish production fallback or cross-runtime checkpoint portability.
Both runtime registrations, all admission flags and production migration stay
closed. Scientific production inputs and unrelated active work are untouched.

## Task 6.1: execution and strict artifact-schema validation

Starting main SHA: `12ec32a0c843713a628d59aa8c4d9a870140b3e8`.
Initial coordinator/context/launch hashes were `2af907bc…`, `3bddc635…` and
`95c11725…`. Strict launch review `17ae96a5…` bound the actual phase-5 push and
continuation. Environment disposition `622e9f61…` explicitly distinguishes the
illustrative row environment from the actual pinned helper-generated environment:
per-command HOME/TMPDIR/cache under its records, four Polars threads, BLAS/MKL/OMP
one, no ambient Python paths or credentials, offline package controls, and explicit
private `/tmp` pytest basetemp. Actual planned records retain the environment used.

The first public recovery probe passed (exit0, 2.821 seconds); its process group
was clear and owned scratch removed. The coordinator then stopped during
classifier artifact metadata capture, before command two. Final whole-input
integrity passed. Actual arrays matched all three raw probe hashes, while the
versioned manifests correctly contained `sha256:<digest>`. Two coordinator
comparisons incorrectly equated that versioned field with a raw digest. The
application serializer and probe are correct; their source confirms the distinct
representations. The stopped attempt, original coordinator and all outputs are
retained unchanged. Diagnosis receipt `d8afac7d…` binds the exact three artifacts.

The initial outer execution is `4c9125d4…`, public command `006f1872…`,
finished report `d123e8b4…`, and actual probe `b6b5ff37…`. This is a failed
coordinator attempt with a successful public probe, not a completed six-command
qualification. This checkpoint required a strict validator correction and
fresh owned retry roots; no canary/recovery pass was inferred from the failed attempt.

Test-first correction: 17 actual-shaped metadata cases ran against the preserved
`2af907bc…` coordinator: four expected assertion failures and 13 passes, with
no import/setup errors (`candidate-canary-hash-schema-red-1/execution.json`,
`d2d00c9b…`). Both valid tagged-manifest cases failed, and both bare-manifest
rejection cases incorrectly passed the old validator. The existing 42-case
suite keeps its assertions; only its synthetic manifest fixture hashes acquire
the actual schema prefix. The source correction is limited to the two hash
comparisons. Three fixed metadata fingerprints are rebound because the retry
uses new owned paths. Root verified that reversing those path/hash substitutions
restores all three original metadata files byte for byte. GitHits CPython
`823f0323`, `Modules/_hashopenssl.c:626`, confirms that `hexdigest` returns hex;
the application serializer supplies the manifest's explicit algorithm tag.

The corrected coordinator is
`candidate-canary-recovery-execution-prep-3/qualify_canary_recovery.py`,
SHA-256 `cea54ce66b8b1c1f0362471e7d4a763e2ec1c24634d30f23d945f8f051c1ab92`.
Root's combined pure controls passed **59 tests in 0.34 seconds**, outer/runner
exit0, execution `43143ce9…`. The successor still stops for any physical Parquet
hash difference. A successful unit run does not substitute for the six public
commands or approve any observed encoding difference. Those were still pending
at that unit-test checkpoint.

The fresh retry completed **all six public commands**, outer/runner exit0,
without interruption (`candidate-canary-recovery-outer-execution-2/execution.json`,
`94dc422f…`). Result `candidate-canary-recovery-pair-2/execution-result.json`
has SHA-256 `65b0f9c9c299b0868f4fca5d292288a0ea23fee55ac882f007dd78555067ce54`,
status `commands_completed`, and final `all_inputs_unchanged=true`. Each public
report passed with process-group clearance and scratch cleanup. Both collections
contained the same 35 expanded cases; both recovery runs passed all 35 cases.
The two probes reused the same five frozen synthetic images. No Git commit
intervened between phase-five continuation and this successful execution.

Exact retry bindings: context `ef987d5e…`, launch `0825afc7…`, strict independent
launch review `c02a505e…`, code/context review `e24c243a…`, and actual metadata
check `ead9d926…`. Pair/commands/contract hashes are `43826824…`, `02c29f25…`
and `86affd64…`. The original failed attempt and its original outputs remain
unchanged.

The original extractor then stopped before publishing a pair observation,
canary or recovery result (`811a5ddb…`). Diagnostic trace `b17f5bf9…` identified
a second reader issue: collection reports `tests/...` node IDs while pytest's
PASSED summary reports the same source tests relative to the private working
directory. Both JUnit documents contain all 35 matching class/name pairs with
no failure, error or skip. This task records successful commands; the separate
path-bound extraction and physical-difference review were required before
qualification.

Independent completed-command review `d327d94c9766385ada8b134d21c9ebbd06daa22b7b8b590806cace09a620b4ae` approves task 6.1 only. Its separate extraction follow-up is tracked in task 6.2 below.

## Task 6.2: bound extraction and physical encoding disposition

Starting main SHA: `02beab73cad7926a76a12fb84c4a94278ced0103`.
The six workload commands are not repeated. A separate metadata companion was
tested to bind each PASSED summary path, resolved from its actual planned
working directory, to the exact frozen test file and collected case. It retains
the original producer, launch, source, raw logs, JUnit and execution receipts.
The derived context names its own harness explicitly. GitHits pytest 9.1.1,
`cf470ec0`, `src/_pytest/terminal.py:1507`, confirms that the terminal formatter
uses `config.cwd_relative_nodeid(rep.nodeid)`.

Root's read-only original comparison (`f82b8812…`) reports exactly
`batches.parquet:physical_encoding_requires_review`. Byte inspection `ecf7a14b…`
binds both 2,116-byte batches files: the footer is 1,360 bytes, starts at byte748,
and contains one writer marker starting at byte2061. The two changed bytes,
at offsets2088 and2092, spell Arrow `25.0.1` versus `26.0.0`; replacing exactly
that versioned marker makes every byte identical. The single,
empty and malformed files are already identical. This inspection alone did not
approve the difference; the required independent logical, provenance and
physical review follows below.
GitHits Arrow `44616716`, `cpp/src/parquet/parquet_version.h.in:29`, defines
the version-specific `CREATED_BY_VERSION` marker.

Independent disposition
`phase6-physical-disposition-independent-1/disposition.json`, SHA-256
`c1c043ab7897426a2e64ca1cbd52958e4fdb49b497f44771e5045219320c7c45`,
approves only that exact batches pair. Its byte/source/logical proof is
`9347cfb25b50d979cfe44719f630164e6a4d4acaf2bbb48dc85d3af37378fe2b`.
A bounded compact-protocol traversal identifies the marker as FileMetaData
field6 `created_by`; no native footer reader or workload rerun was needed.
The proof also checks matching image/Parquet logical outputs, all three raw
NPZ pairs and manifests differing only in the declared source identity fields.

The new companion's 30 controls were first run before its implementation
existed: **30 setup errors in 0.08 seconds**, receipt `10696219…`. This is an
absence check, not a reproduced body-level regression. The original 59
coordinator controls and original executor remain unchanged.

The initial companion passed **89 combined controls in 0.52 seconds** and its
read-only actual-input check passed (`bf21aaa8…`, `12b315f1…`). Independent
review then found one closure omission: the new context file itself was not in
the original guard's pinned-file set, although its nested references and the
outer recorder already bound it. Review `7ffa9f2c…` requested the correction;
the prior quality approval was explicitly superseded by `00ebf875…`. No derived
results were created. Two new behavioral tests reproduced missing closure and
undetected context mutation, **2 failed in 0.02 seconds** (`1ff6c8a3…`). The
successor adds only an explicit context pin after loading the original run.

The final companion in `candidate-canary-recovery-extraction-prep-2/` has
SHA-256 `540d4e5c86ea4741f56aebac2bd9f3198c409122d2056a4787e5ca222fdeefbb`;
its context is `246b0a3fd3b1670ec718566e51f32150cd06e72178c40f119a0131ec231e75ff`.
Removing the one pin-registration line restores the initial companion byte for
byte. Root's first combined test invocation accidentally applied the companion
operator override to 17 coordinator cases; those failed with AttributeError,
while 74 passed (`c605e9f6…`). The corrected invocation lets each unchanged
file use its own operator: **91 passed in 0.51 seconds**, outer/runner exit0,
receipt `9aec556e…`. No test assertion or production source was altered to
correct that invocation. That checkpoint preceded the final actual-input
metadata check reported below.

The successor actual-input check passed, outer/runner exit0, receipt
`5a78d650…`. Final independent SPEC/QUALITY approvals are `f88ee4dc…` and
`3c73d825…`. The reviewed metadata-only derivation then passed without any
public workload rerun: outer execution `4ba47cae…`; derived result
`a387a4031a43fa5f020e0d8b8072ad09cb4df24dce8f1749e649d91041d0c784`.
Both before/after guards explicitly contain the context file and report all
inputs unchanged (`273890a1…`, `668ae7e7…`).

The immutable derived canary and recovery results have SHA-256
`ea496004601267e32d864d4d4104781ae7198bd6af6dc47cd7b294fd66a98328`
and `4a47861404eeb7163e1c45fa7603dd90a69e846cd86cff52fc3fd7e3b0fc9b76`.
They report `canary_passed` and `recovery_passed`, no unexplained logical or
provenance differences, and removed workload scratch. Interruption, stale-claim,
resume and idempotency categories each retain both runtime executions and their
exact collected cases. The original 35 cases per runtime remain the execution
scope; category coverage uses those same executions, not additional runs.

Observation `c3f36459f5f268cb41f8bb7d875e8a3e58fa8728868d9dc2926746a807350dd9`
retains the raw physical difference, both original hashes and the separate
reviewed resolution. Derived PASSED-summary views retain each raw log backlink
and exact canonical path/case bindings. The original failed extractor, raw
summaries, producer, context and six executions remain unchanged.

Independent actual-result review `834b2247a20f99e98a09f11ddb8a7d946f7026a1ad0baea13231270b1d8e9068` approves task 6.2 with no blockers. Separate physical-resolution preservation review `faa48a90…` confirms the exact limited disposition survives the derived records. Final phase-manifest joins, reviews and publication remain task 6.3.

## Task 6.3: final compatibility bindings and phase handoff

Starting main SHA: `0abbaa55ca4472f6663dbc47892be2063aa0d4b0`.
The final source map preserves all original 358 entries and adds ten existing
probe/test/helper files. Their bytes match the frozen application source,
its source commit and committed main; no application code changes accompany
these additions. Addendum `bdc01708…` preserves the earlier six-file addendum
`8c783548…` and explicitly adds the four fixture helpers used by the probe.
Both check records bind all 13 relevant source files plus `pyproject.toml`.
The original three phase-five subjects remain identical; the additional
recovery application is explicitly the complete Arrow25 candidate with its
own wheel, source/build and installation identities.

Root ran the unchanged `44ff68a2…` canary/recovery semantic reader against
both actual results. Each record carries the original six public executions,
the successful original outer execution and separate derivation execution;
each also binds four actual before/after integrity records and 14 frozen source
files. All 368 source hashes match committed task bytes. Actual semantic/source
verification passed, outer/runner exit0 (`65c036d6d4dd1401845fa840c383317eb2ae2dbd7e308c7c13dceda8b21809a5`).
The draft subject for this check is `54b07673…`; it deliberately remains a draft
until this task commit and the final digest-bound independent phase reviews.

The amended qualification scope now has passing evidence for:

| Check | Verified outcome |
|---|---|
| Arrow25/26 native capabilities and target API | Both candidate families passed their separate phase-four proofs |
| Polars1/2 × Arrow25/26 | Four cells, 59 cases each, no logical differences |
| Complete source and installed application | 4,254 tests per suite and three CLI checks passed on the frozen target inputs |
| Disposable PostgreSQL and S3 | Both passed, with verified namespace cleanup |
| Paired resources | Twelve sequential workers, three repetitions per cell, no 15% review flags |
| Synthetic canary and recovery | Both probes and 35 recovery cases per runtime passed; 91 final coordinator/extractor controls passed |

The phase-five S3 configuration correction used the supplied bucket name rather
than its bucket ID. Only that local setting changed; credential values were not
reported or committed. The exact live result and configuration evidence remain
in the phase-five report and provenance.

This work qualifies the selected candidates for the agreed compatibility scope.
Cybersecurity clearance, runtime admission/registration/activation, production
migration and a currently safe production fallback remain excluded. All six
corresponding flags stay false. No scientific production workload was run.
Failed attempts, withdrawn reviews and original observations remain available.
The final independent phase reviews, create-only publication, distinct summary
commit, normal main push and pushed-receipt finalization follow this task.
