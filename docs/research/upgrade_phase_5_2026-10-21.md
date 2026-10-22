# Phase 5: combined candidate compatibility

Phase 4 was pushed to `main` as
`48e3c469a803a8c8e4b60d4d68f7794ce8cb92d8`. Its create-only compatibility
push receipt has SHA-256
`3a4bcb355d1527c62de2a1118c666417699969012b9c45ef83d13621baaebf77`.
Fresh continuation into phase 5 passed before execution; result and execution
hashes are `3b49a147e5b63b678ae53f0040fe62a7cd568dc4a6ee1b858ed38686b22b17e6`
and `00b05b6c82737d82ff11127e844c6925e1d3c18b709c031c0e93d07ab92b049a`.
External paths below are relative to
`/home/toffe/.local/state/biominer/upgrades/20261021/phase4/`.

## Task 5.1: four-cell matrix

All four fresh offline environments passed their complete 59-case ordinary
API selections, with no skipped or failed cases. Both native import orders,
all ordinary fixtures and final whole-environment/source inventory checks
passed in every cell. Logical differences are empty.

| Polars | Arrow candidate | Passed cases |
|---|---|---|
| 1.44.2 | 25.0.1+biominer.53ac869eb6c0 | 59 |
| 2.0.0 | 25.0.1+biominer.53ac869eb6c0 | 59 |
| 1.44.2 | 26.0.0+biominer.58dcc66d447e | 59 |
| 2.0.0 | 26.0.0+biominer.58dcc66d447e | 59 |

Each environment contains exactly nine pinned distributions, rather than the
full application or its 45 native dependencies. Six recorded commands per
cell cover offline installation/metadata, import orders, tests and ordinary
observations. The complete 46-package application/source/CLI qualification
remains the separate phase-4 result for unchanged frozen inputs.

Matrix result `candidate-compatibility-matrix-2/result.json` has SHA-256
`cc8db7ea72cf0288ce78cc914566c03a6fd5f7142e99b2882bb8c6c8649a7f36`.
The outer execution exited 0 without interruption, SHA-256
`26ed0a47071c126b83719187e59add615ab8e6459032c539b6c31a387662f8e3`.
The owned coordinator completed in 208.565 seconds. Its inner execution lacks
an argv field, so the generic phase evidence uses the actual outer command
receipt and retains the inner observation separately.

The unchanged matrix helper/context hashes are `1296748d…` and `57b33aed…`.
Independent final matrix/live launch review is
`4e9d8cfa33467bc494ba21b1b2fe24761eda0dc41aa6042415939d32c62ea899`;
all 532 pinned files matched. Exact launch specs are
`matrix-concrete-inputs-1/run-6.spec.json` and `run-outer-6.spec.json`.
No candidate was registered or promoted. Live backend qualification, paired
resource comparison, phase-5 publication/push and phase-6 recovery remain
required. The first live attempt passed PostgreSQL and cleaned both namespaces,
but S3 failed; resource preflight rejected an alias. These later-task failures
remain unresolved at this checkpoint and do not invalidate the passed matrix.


## Task 5.3a: resource harness preflight

Starting main SHA: `9dac03387158e08abd433ca6526df05fd28bf630`.
The first metadata-only preflight rejected a retained matrix builder's Python
symlink. No benchmark worker or result was produced. The matrix already allows
this declared link to the exact SHA-bound canonical base interpreter. The
resource helper now applies that same narrow policy to original matrix inputs,
records the literal link chain, and checks it again with target and byte hashes.
Other aliases, changed link text and resource-owned aliases remain rejected.
A prior correction compares retained wheel copies by exact name, version and
whole-archive SHA rather than requiring equal filesystem paths.

The new alias cases first produced 8 failures and 4 passes against the old
helper; all 65 controls then passed. The actual corrected metadata preflight
passed without creating qualification outputs. Independent SPEC/QUALITY review
approved the concrete input set. The 12 measured workers remain unrun at this
commit; no performance claim follows from these controls.

The corrected helper/context are `ac1cdaac…` / `81059294…`; actual GREEN and
metadata execution hashes are `e1d9b6ef…` and `d7ff62e3…`. Review is
`candidate-resource-benchmark-alias-independent-review-2.json`, SHA-256
`9f0f04a0ff2f87c75f2ca2b3127a8f0175f943dd3f3faa0efb43748695144a08`.
All original failed preflight evidence and prior helper bytes remain retained.


## Task 5.4: prepare the phase-6 synthetic recovery handoff

Starting main SHA: `ee33b5262c65bffe4dcf7f95eef21f80ec6d927e`.
The reviewed finite coordinator is prepared for six sequential commands across
the two complete application candidates: paired tiny synthetic probes, paired
collection, and 35 recovery cases per runtime. It verifies whole frozen
installations, source, private Git metadata, 44 common dependency archives and
331 identical application payloads. Each candidate retains its correct Arrow
floor and exact application/source identity. Both remain unregistered.

The coordinator's 42 controls passed, including actual authorization tests for
full-suite success and independent-review identities. Root's actual metadata
check passed without workers, credentials, authorization or qualification
outputs. Closing independent review approved this preparation. Its helper,
context and metadata receipt hashes are `2af907bc…`, `3bddc635…` and
`a3e59448…`; the review is `622a427f…`.

Execution still requires the actual completed phase-5 push and a fresh phase-6
continuation, bound independent launch review, six successful commands and
unchanged final inputs. Unexplained logical, provenance or physical differences
stop extraction. Any observed writer-version difference needs a concrete,
independently reviewed disposition preserving both original artifact hashes.
No canary, recovery, fallback or production result is claimed by preparation.


## Task 5.2a: live diagnostic retry

Starting main SHA: `12a7e53c0944cf2bb0e25e9b28a958e4fb7c4ada`.
The first combined live attempt passed PostgreSQL, failed S3 with an unclassified
OSError, and cleaned both namespaces. Its preinitialized Parquet counts described
planned work, not completed publications. That failed report remains unchanged.
The diagnostic successor records only finite operation/error tokens and advances
publication counts after successful public writes. Client options, TLS checks,
cleanup, journals, call limits and time bounds are unchanged. All 73 original
controls remain byte-identical; 22 new controls first failed and then all 95
passed. Independent review and actual metadata validation passed.

Root executed the reviewed retry in a fresh namespace, with a 900-second outer
and 600-second worker limit. PostgreSQL passed and its schema was cleaned. S3
returned `NO_SUCH_BUCKET` during `PutObject` at `marker_write`, before any Parquet
publication (observed count zero). S3 cleanup found no remaining owned objects
and no errors. Both frozen whole-installation/source checks passed. The outer
exited 1 in 31.618 seconds without interruption, accurately preserving failure.
This establishes the endpoint response, not whether the underlying cause is a
bucket setting or request routing. No client option was changed on speculation.
The user was asked to verify bucket and endpoint in the approved local secrets
file; no credentials were printed or committed. Offline work continues.

Report `live-backend-qualification-2/report.json` has SHA-256
`20c8e060bddd19bb96864de8caa3298afab9804612cc1eaa8ead04165a37683b`;
outer receipt is `6f3e00ee…`, final integrity `23ec5a00…` and independent
review `144e9f30…`. Phase-5 live qualification remains incomplete. No claim of
successful object publication, security clearance or runtime admission follows.


## Task 5.3b: completed paired resource comparison

Starting main SHA: `ad33fc475b07ca071660505ca4dd304127834263`.
All 12 sequential fresh workers passed: three interleaved repetitions in each
nine-distribution matrix cell, identical 200,000-row CSV input, four Polars
threads and one BLAS thread. Every output matched the same 171,428-row logical
oracle and schema, every owned scratch directory was removed, and before/after
whole frozen-input checks passed. No compared work-time or RSS pair/median
exceeded the reviewed 15% threshold; flags are empty. There was no overlapping live/native
qualification during measurement.

| Cell | Median work (ms) | Median peak process RSS (KiB) |
|---|---:|---:|
| Polars 1 / Arrow 25 | 96.251 | 287380 |
| Polars 2 / Arrow 25 | 107.057 | 306888 |
| Polars 1 / Arrow 26 | 98.414 | 289864 |
| Polars 2 / Arrow 26 | 109.766 | 306860 |

On Polars 2, Arrow 26/25 median work ratio is 1.0253 and RSS ratio 0.9999.
Across Arrow versions, Polars 2/1 work ratios are 1.1123 and 1.1153; RSS ratios
are 1.0679 and 1.0586. These observations describe this tiny synthetic fixture.
Work timing excludes validation, import and logical-check stages. Peak process
RSS includes imports, work and checks. Temporary/spill observations are end-of-
work snapshots, not peak-spill or engine memory-limit evidence. They are not
production performance or runtime-admission claims.

Result `candidate-resource-benchmark-1/result.json` has SHA-256
`27d4c00260e412664b06436e0d6e65ee1362d7e847732d6cb54120c6baccaefb`.
Outer execution `b1c946a0…` exited 0/0 in 424.461 seconds, without interruption.
The exact reviewed inner/outer specs are `run-2.spec.json` / `run-outer-2.spec.json`
in `candidate-resource-benchmark-concrete-1/`; final launch review is `6dea4b59…`.
This completes the measured resource task. Successful S3 qualification and the
phase-5 reviewed publication/push remain required before phase-6 execution.

Independent result SPEC/QUALITY review approved (`690219a2…`). A separate
independent read-only post-run integrity observation (`0ca8dafb…`) verified
10,232 relevant files / 1.750 GB against the original four cell and frozen-source
inventories. This is review evidence; the benchmark did not emit a separate
post-run inventory JSON or an integrity boolean. Its reviewed code checked the
same frozen inputs before and after every worker. Both forms remain distinct.


## Task 5.5: preserve the pending-live handoff

Starting main SHA: `d3a7d732dbf0d20916226429f5f0d19fe2d70cbf`.
A finite metadata-only assembler joined the completed matrix, actual resource
result and exact unchanged phase-4 full-suite row. Root inspected the helper
before execution. The actual bounded assembly exited 0/0; it performed no
qualification, semantic phase validation, service call or publication. The
immutable output is `compatibility-phase5-concrete-1/manifest.assembly.pending-live-1.json`,
SHA-256 `2729a5ebb5061e0d1c62ec13b0cbf62e1404cf686b9a2afed051de691c5111a2`.
Its execution receipt is `c36a5ff8…`. Three subjects and 358 source identities
match the published phase-4 manifest. The live result remains null, its sole
qualification blocker remains explicit, and every readiness/admission/production
flag remains false. The draft ledger contains five actual task commits through
`d3a7d73`; this later handoff commit and future live work must be added to a new
immutable final assembly before phase completion. The unchanged compatibility
builder remains the sole semantic validator and publisher.

Fresh retry preparation changes only the nonce and owned output roots. The
worker/client settings remain unchanged. Actual metadata validation passed
without credentials or live calls (`826be9e4…`), and independent preparation
review approved (`9529a02a…`). Context/guard hashes are `5cc2a9f3…` / `1232d26a…`.
This is preparation only: user confirmation of the approved local bucket and
endpoint, final concrete launch/pin review and serialized scheduling are still
required. No third live attempt was made.

GitHits source follow-up (`a00e4b96…`) confirms that Arrow can classify
`NO_SUCH_BUCKET` as NotFound during existence checks. HEAD and PUT share the
same endpoint resolver, bucket/key parameters and addressing choice in the
reviewed paths. Neither an existenceFalse result nor the current evidence
establishes a Put-only routing defect. No checksum or addressing setting was
changed speculatively.

Current handoff: matrix 4 × 59 passed; unchanged full application 4,254 source
and 4,254 installed cases plus three CLI stages passed; resources 12 passed;
combined live qualification failed on S3 while PostgreSQL and both cleanups
passed. Phase 5 is not published or pushed; remote main remains the completed
phase-4 SHA `48e3c469a803a8c8e4b60d4d68f7794ce8cb92d8`. Phase 6 remains prepared
and unrun behind its required phase-5 handshake. Existing scientific work and
contained runtimes remain untouched. The cybersecurity component stays excluded.

Independent checkpoint review (`39499cde…`) approved this coherent incomplete
handoff only. It verified the exact draft, actual assembly, protected identities,
completed evidence joins, preparation-only retry scope and explicit null live
result. It is not phase approval or publication authorization.


## Task 5.2b: successful PostgreSQL and S3 qualification

Starting main SHA: `8d310128cea30a886ca91b4e02fc8fe1bf8f5cce`.
On 2026-10-22 the user supplied additional S3 details in the approved local
secrets file. A value-free check established that the configured bucket equalled
the supplied Backblaze bucket ID; the matching application-key block provided a
separate valid bucket name. Root changed only `BIOMINER_S3_BUCKET` to that name,
using an atomic local write. Endpoint, key and every other parsed setting stayed
unchanged. No credential or bucket value is included in this report or commit.
Backblaze's [S3 calling documentation](https://www.backblaze.com/docs/en/cloud-storage-call-the-s3-compatible-api)
uses the bucket name in the URL; GitHits confirmed the SDK PutObject bucket-name
contract at `aws/aws-sdk-cpp@d207a37e` in `PutObjectRequest.h`.

The exact independently reviewed fresh retry passed. PostgreSQL checks passed
with verified client TLS, six fixture work items, 40 explicit execute calls and
48 explicit SQL statements; rollback left no durable attempted item. S3 passed
both public Parquet publications (two objects, eight written rows), exact reads,
batched reads/writes, checksum/materialization, create-only collision rejection,
manifest round-trip and lazy scan. Observed filesystem method calls were 46/128
for fixture work and 36/96 for cleanup; these are not SDK/Rust transport counts.
The private PostgreSQL schema and S3 namespace were cleaned; S3 remaining objects
and cleanup errors are empty. Whole frozen-input checks passed.

Actual report `live-backend-qualification-3/report.json` has SHA-256
`6d7dd743eb0fe131013811124830d77d261b7c49591e0675e1c3cf384b8aa0e4`.
Outer execution `09410de6…` exited 0/0 in 90.939 seconds without interruption;
final whole-input integrity is `17a054ac…`. Final launch review is `2ae92647…`;
reviewed launch spec is `2d407b11…`. The worker remains `8c1bed13…`, context
`5cc2a9f3…`, and guard `1232d26a…`. Both earlier failures remain unchanged.
This closes successful live qualification; final phase-5 evidence joining,
independent phase reviews and publication/push remain required.

Independent actual-result SPEC/QUALITY review approved (`5d470247…`), including
all 17 journal references and exact task-owner/context joins. This result proves
ordinary disposable PostgreSQL/S3 compatibility and cleanup for these frozen
inputs; it is not phase-level approval, production activation or admission.


## Task 5.3c: exact resource-observation metadata view

Starting main SHA: `21b84d4d773137d79b87a7500bcaf10c4d76173f`.
The actual resource result merged two supervisor fields (`execution` and
`launch_wall_seconds`) into each copied observation. The unchanged phase reader
requires the copied value to equal the original observation JSON exactly. Root
reproduced that representation rejection (`693c4cc0…`); it was not a benchmark
or logical-data failure. Neither the original result nor the pushed phase reader
was edited.

A finite stdlib-only adapter accepts only the exact completed raw result
`27d4c002…`. It verifies all 12 original observations, their exact two extra
fields, successful execution/cleanup/subreaper records and matching launch
elapsed values. The derived view retains the two supervisor fields separately,
uses each original observation verbatim, and explicitly links the raw result,
adapter and original measurement context/harness. Every top-level metric,
comparison, original reference, measurement and source byte remains unchanged.
Unknown differences, changed observations, failed/interrupted executions,
elapsed mismatch, aliasing and output overwrite are rejected.

The 22 controls first produced setup errors while the adapter was intentionally
absent; then all 22 passed in 1.14 seconds with the controlled executor and exact
code/test pins. Independent code review approved (`aa422eaa…`). Actual derivation
exited 0/0 (`ed00d665…`) and created the read-only view
`candidate-resource-observation-view-1/result.json`, SHA-256
`81cd448af06a7a650ca5f335c79bb61f3e6f0ca99ae28c4aa7d825930778fb9f`.
Root's actual unchanged-reader check accepted both this resource view and the
successful live result (`46543cb4…`), after separately asserting exact original
observations and all unchanged metrics. No benchmark or live job was rerun.
The original `27d4c002…` result and `44ff68a2…` phase reader remain immutable.
Final phase-level reviews, publication/push and phase-6 qualification remain.

Independent actual-view closing review approved (`9dc4c4f3…`), explicitly
separating unchanged raw measurements from the derived representation. GitHits
also confirmed Python 3.14.7's exclusive-create output mode maps to
`O_EXCL | O_CREAT` in `python/cpython@823f0323`, `Modules/_io/fileio.c`.


## Phase-5 candidate publication and summary boundary

All four compatibility checks passed and the unchanged finite validator accepted
the complete evidence. The eight chronological task commits are:

| Task | Commit |
|---|---|
| 5.1 matrix | `9dac033` |
| 5.3a resource preflight | `ee33b52` |
| 5.4 phase-6 preparation | `12a7e53` |
| 5.2a live diagnostics / preserved failure | `ad33fc4` |
| 5.3b resource measurements | `d3a7d73` |
| 5.5 pending-live checkpoint | `8d31012` |
| 5.2b successful live qualification | `21b84d4` |
| 5.3c exact resource metadata view | `84c7aef` |

Tested commit is `84c7aefc50c51d8bfb304a6be987e84e9c61010e`.
The final manifest has SHA-256
`f2aa60d0802e032eab46c1c13f830ba9286d369d8d4d03b9a415f698dfe28fbb`;
its reviewed core digest is
`50ccfaf77bf5fea9062cd7b0f09b061c298acf2c64d6debc1f4f563c68a07eca`.
The original three subject identities and 358 committed source hashes are
unchanged from the actually pushed phase-4 boundary. Two stale preparation
state strings were corrected in an immutable successor before approval; the
old ready1 digest was never approved.

Independent final SPEC and QUALITY reports are `091fd687…` and `11502398…`;
approved wrappers are `3a5568f2…` and `b2afa2a7…`. Formal validation passed
(exit 0/0, `bbc0edd2…`); publication revalidated the evidence and passed
(exit 0/0, `d89f6076…`). The create-only compatibility candidate receipt is
`/home/toffe/.local/state/biominer/upgrades/20261021/phase-5-compatibility-candidate.json`,
SHA-256 `9694d828578cbcaa0c50b2f7c1e0895135f47e1e18b0c6d789c1990054263318`.

This distinct documentation/provenance summary is the required next commit.
After its normal push and remote-SHA verification, root must finalize the pushed
receipt and run the fresh continuation into phase 6. No Git commit may intervene
between that summary boundary and the first phase-6 execution. Push and next-
phase authorization are recorded by their actual receipts, not inferred here.
Phase-6 synthetic canary/recovery remain unrun at this publication checkpoint.
Every security/admission/registration/production-migration flag remains closed;
this is compatibility qualification only. Unrelated scientific work is preserved.
