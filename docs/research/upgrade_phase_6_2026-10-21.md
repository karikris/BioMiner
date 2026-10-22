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
qualification. A strict validator correction and fresh owned retry roots are
being prepared; no canary/recovery pass is inferred at this checkpoint.

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
commands or approve any observed encoding difference. Those remain pending.

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
path-bound extraction and physical-difference review remain required.

Independent completed-command review `d327d94c9766385ada8b134d21c9ebbd06daa22b7b8b590806cace09a620b4ae` approves task 6.1 only. Its separate extraction follow-up remains open.
