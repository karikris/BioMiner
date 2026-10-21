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
