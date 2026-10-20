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
| 3 | 3.3 | Repair the newly confirmed Arrow 25 boolean-allocation defect; qualify a safe patched fallback, then run paired Polars 1/2 tests with that Arrow 25 build, lock Polars 2 and run full regression |
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
