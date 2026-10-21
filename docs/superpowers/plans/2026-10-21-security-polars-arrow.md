# Security remediation and Polars 2 / Arrow 26 implementation plan

Approved in conversation on 21 October 2026. Starting `main`:
`5bac7f168c1b257184615baf1524f12d5c580bc6`.

## Execution contract

Use subagent-driven implementation and independent spec/quality review.
One tested commit directly to `main` per numbered task; push each phase and
verify the remote SHA. Automated evidence handshakes release the next phase.
No routine approval pauses. Missing evidence, failed mandatory tests, stale
hashes, unresolved applicable vulnerabilities, remote divergence or unsafe
workload overlap block advancement. Do not fabricate a passing gate.

Preserve unrelated dirty scientific work, old profiles, runtime generations
and production artifacts. No branch, history rewrite, duplicate scientific
job, automatic uncoordinated reboot, or restart of another owner's workload.
The current task's explicit commit/push cadence supersedes the generic
dirty-tree checkpoint wording; unrelated work must not be committed.

Use GitHits metadata, batched upgrade reviews, changelogs, pinned advisories
and upstream search/grep/read. Use search_status only when offered. Preserve
primary-source references and coverage gaps. Raw evidence and secrets stay
outside Git; append curated provenance to `provenance/githits.jsonl`.

## Task boundaries

| Phase | Task | Deliverable |
|---|---|---|
| 1 | 1.1 | Runtime/process/launcher/credential-availability inventory and immutable baseline evidence |
| 1 | 1.2 | Refreshed installed/baseline/target security and compatibility evidence; baseline fixtures |
| 1 | 1.3 | Tested automated phase receipt validator, runner and post-push handshake |
| 2 | 2.1 | Self-contained patched baseline/fallback with frozen launcher, checks, configuration and lock |
| 2 | 2.2 | Drain eligible workloads at durable checkpoints; migrate registered entrypoints and replace development environment only after consumers stop |
| 2 | 2.3 | Verify actual loaded versions, fresh admission, TLS/subprocess/resume and independent fallback operation |
| 3 | 3.1 | Explicit explode semantics preserving candidate provenance rejection and report label counts |
| 3 | 3.2 | Resolve relevant schema/coercion/order/ingestion changes; explicit bounded per-run spill configuration |
| 3 | 3.3 | Repair the confirmed Arrow 25 boolean-allocation and sliced-boolean copy defects; qualify a safe patched fallback, then run paired Polars 1/2 tests with that Arrow 25 build, lock Polars 2 and run full regression |
| 4 | 4.1 | Pinned Arrow 26 recipe, reviewed crypto/build patches and static dependencies; four compiler jobs maximum |
| 4 | 4.2 | Audit stricter/removed Arrow APIs; reader-depth and malformed-input rejection tests |
| 4 | 4.3 | Custom Arrow 26 wheel and native receipts; lock/admission changes and new verified generation |
| 5 | 5.1 | Four-combination Polars 1/2 × Arrow 25/26 compatibility matrix and provenance-safe artifact comparisons |
| 5 | 5.2 | Bounded live PostgreSQL/S3 qualification in task-owned disposable namespaces, including cleanup |
| 5 | 5.3 | Full combined suite, packaging/CLI/security/native checks and three-run resource benchmark |
| 6 | 6.1 | Bounded canary, then sequential checkpoint-safe workload migration |
| 6 | 6.2 | Interrupted-run recovery, currently safe fallback and retirement of affected entrypoints |
| 6 | 6.3 | Final migration/security/evidence report and verified final push |

## Phase gates

1. Complete inventory, reproducible baseline, security/compatibility evidence
   and tested gate behavior.
2. Eligible workloads migrated; remaining exposure scoped/contained; fallback
   self-contained and freshly verified. Unowned/uncheckpointable active
   consumers block migration rather than being killed.
3. No unexplained logical output changes or relevant migration warnings;
   resource/spill cleanup checks and full suite pass.
4. Exact build provenance, five native installation checks, feature/import-order
   checks and fresh verification pass.
5. All four compatibility combinations, live tests and cleanup, full regression,
   current security review and bounded performance checks pass.
6. All registered workloads use the target, recovery works, affected entrypoints
   are retired and final independent review passes.

Each handshake binds phase/task IDs, task commits, tested source/configuration
hashes, artifact hashes, predecessor receipt hash, exact commands/results/log
hashes, independent review, findings/blockers, and live/cleanup/migration
evidence where required. Test evidence binds implementation commits; a separate
create-only post-push receipt binds the final phase commit and observed remote
SHA, avoiding circular hashes.

## Compatibility and security contracts

### Execution correction: Arrow 25 security prerequisite

During phase 3, primary upstream PR 50843 and the exact installed Arrow 25
source confirmed a heap overwrite in `FillNullForwardChunked`,
`FillNullBackwardChunked`, and `ReplaceMaskChunked`. These belong to the enabled
compute library. Absence of direct BioMiner calls does not establish that the
installed library is unaffected. No CVE/GHSA assignment is asserted from the
issue/PR evidence alone.

The original plan assumed that the phase-2 Arrow 25 fallback needed no further
native correction before the Polars comparison. That assumption conflicts with
the newly established defect and the strict phase-3 security gate. Resolve it
by adding corrective work within task 3.3: backport the exact reviewed upstream
allocation fix, build a new isolated Arrow 25 wheel with bound source and build
evidence, qualify a replacement fallback, and rebuild the Polars pair against
that same patched Arrow version. Record the newly applicable finding and its
actual resolution. Preserve old receipts and failed/blocked attempts, without
continuing to describe the old fallback as currently safe.

Do not relax the gate, suppress the finding, or begin phase-4 implementation
before the phase-3 push handshake. Additional corrective task-3.3 commits remain
in the ancestry; its final commit binds the actual qualified source. Preserve
active workloads and replace registered entrypoints only after the new fallback
passes verification and checkpoint/ownership checks.

The allocation-only wheel subsequently failed the actual native regression:
an offset-one Boolean array `[null, true, null]` forward-filled to
`[null, false, false]`. A separate bounded reproduction confirmed the committed
oracle and failures in all three affected operations; offset-zero Boolean and
sliced integer controls passed. Upstream
[PR 51240](https://github.com/apache/arrow/pull/51240), commit
`0efbdec7a10da1a4043fc1210c61e73c88bc9b4f`, fixes the missing input-array
offset in Boolean bitmap copying. Arrow 26 includes this fix. Its changelog
entry alone and automated `no_impact` classification did not establish
compatibility; exact source inspection and actual observations take precedence.

Task 3.3 therefore requires this second exact backport after the unchanged
allocation patch. Bind both ordered patch stages and the three final source
files, including the test file modified by both stages. Build a fresh isolated
wheel and pair, retaining the failing regression and all failed receipts.
This is an established correctness defect; no additional CVE, GHSA or memory
exploit is asserted. Identity checks passing for the allocation-only wheel do
not satisfy native qualification or close the allocation security finding.

### Execution correction: transitive network and compression providers

During phase 4, the complete configured-provider inventory and the retained
Arrow 25 ELF receipt exposed a separate system zlib provider through system
curl and libxml2. Ubuntu marks the installed zlib package as affected by
[CVE-2026-85091](https://ubuntu.com/security/CVE-2026-85091), and the installed
curl package as affected by
[CVE-2026-19931](https://ubuntu.com/security/CVE-2026-19931).
The patched private zlib used by lxml does not repair this separate provider.
Retain the material applicability conflict: the advisory describes 1.3.1.2–1.3.2,
Ubuntu also marks older packages vulnerable and questions the affected range,
and the earlier exact 1.3.1 source inspection lacks the later stale-pointer
branch. This review does not newly prove exploitability of installed 1.3.1 or
erase that source evidence. Replacing Arrow's provider removes the conflicting
host dependency; the independent curl finding already blocks its admission.

Reopen `NATIVE-ZLIB-CVE-2026-85091` in its existing
`shared_and_approved_runtime` scope and add the separately confirmed
`NATIVE-CURL-CVE-2026-19931` finding in that scope. Preserve earlier receipts as
historical observations. Their previous resolved statuses do not establish
current safety of the registered Arrow 25 development or fallback runtimes.

Add corrective task-4.1 work before phase-4 admission: build a source-bound,
private static network/XML prefix with curl 8.23.0, libxml2 2.15.4, libiconv 1.19,
zlib 1.3.2 with `df84af25`, and the reviewed private OpenSSL 3.6.5. Preserve HTTP,
HTTPS, HTTP/2, verified TLS, gzip and the enabled Arrow feature contract. Bind
the exact nghttp2 archive, headers, distro security/backport evidence and actual
PIC consumer proof; retain its unavailable source-archive evidence explicitly.
Select one private zlib provider, exact imported targets and headers, then
reject residual affected host providers in the final resolved ELF/link-map
closure. Neither prefix compilation nor a configure inventory admits a runtime.

Build a separate corrective Arrow 25 recipe and policy for the safe fallback
and four-runtime matrix. The old recipe, ordered compute patches and receipts
remain byte-exact. Do not hide changed execution behind an old recipe hash.
Qualify the corrected Arrow 25 and Arrow 26 installations before closing either
finding or advancing the phase.

With no safe fallback established, stop affected registered entrypoints at a
durable boundary after fresh ownership, consumer and scheduler checks. Use a
separate pytest-only executor for synthetic policy/build-tool tests. Preserve
unowned workloads, scientific artifacts and historical evidence; do not kill
jobs or modify OS packages. Continue isolated remediation, keeping the phase
and installed security gates blocked until replacement qualification succeeds.

### Required target behavior

- Targets: Polars 2.0.0 (`>=2.0.0,<3`) and custom Arrow
  `26.0.0+biominer.<input-digest>` (`>=26.0.0,<27`), after refreshed review.
- Both explode call sites use explicit `empty_as_null=True, keep_nulls=True`.
  Empty/null candidate query definitions remain rejected, never dropped.
- Preserve schema, ordering, null/list/enum/timestamp, join, provenance,
  checkpoint, image and classifier contracts. No scientific schema/model/pin
  changes. Logical Parquet equivalence may coexist with versioned physical
  encoding hashes; never overwrite prior artifacts or suppress invalidation.
- Preserve declared job budgets. Otherwise four threads, 1 GiB Polars engine
  memory budget and 4 GiB spill budget; these are not process RSS guarantees.
  Use isolated scratch and cleanup only task-owned files.
- Preserve all enabled Arrow capabilities/codecs, S3, Flight and encryption.
  Gandiva remains disabled. Review simdjson/static crypto source requirements,
  full ELF closure and import-order behavior. Stock Arrow wheels do not qualify.
- Keep Parquet depth protection enabled. Malformed/over-depth data must fail
  without partially publishing output.
- Approved live targets: unique disposable schema in configured Supabase
  project and unique disposable prefix in configured S3 bucket. Synthetic
  records/objects only. No production tables, objects or access policies changed.
  API keys are not PostgreSQL passwords; suitable connection access is required.
- PostgreSQL tests cover authenticated TLS, transactions/rollback, CRUD,
  leases and idempotency. S3 covers Parquet round-trip/ranges/listing/retries,
  manifest-last publication, scoped deletion and verified cleanup.
- Compare four runtime combinations using identical bounded fixtures.
  Run three comparable benchmark repetitions; unexplained median wall-time
  or peak-RSS regression over 15%, budget violations or resume/provenance loss
  block the gate. Benchmarks stay outside ordinary unit tests.
- Runtime migration, not lockfile modification, closes installed exposure.
  An unsafe fallback is forbidden; pause at a durable checkpoint when no
  currently qualified, checkpoint-compatible fallback exists.

## Completion evidence

Record exact task and push SHAs, artifact/profile hashes, tests, source/advisory
references and gaps, live cleanup, process migration and remaining human/live
requirements. Do not claim universal compatibility, native security clearance,
production promotion or scientific release from package tests alone.
