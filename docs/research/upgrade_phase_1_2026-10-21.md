# Security and compatibility upgrade — phase 1

Phase 1 completed its inventory, baseline, evidence and handoff implementation
on `main`. This is an **inventory gate**, not runtime security clearance or
production promotion. The shared environment and existing generation remain
unmigrated; their applicable findings must be resolved by later gates.

## Commits and evidence

Starting commit: `5bac7f168c1b257184615baf1524f12d5c580bc6`.

| Task | Commit | Result |
|---|---|---|
| 1.1 | `c4176c4e05f68817a6ba6832697b2eb0a133ebf9` | Runtime/process/entrypoint inventory and ownership boundaries |
| 1.2 | `15ebde9b192d8b5e8b641d1580b2a2186e0e326d` | Current advisory/source evidence and reproducible synthetic fixtures |
| 1.3 | `3c3c5c7cf7d03e12309292fe7c392d03a71fd5a2` | Tested phase validator, runner and post-push handshakes |

The tested source is an isolated, clean snapshot of task 1.3. Its owned test
environment uses Python 3.14.7, Polars 1.44.2 and custom Arrow
`25.0.1+biominer.f918dfd84c8c`. An editable installation binds BioMiner to that
snapshot, including subprocess imports. A hashed launcher supplies an empty
provider environment and thread limits of four for Polars and one for each
BLAS/OpenMP pool. The shared `.venv` was not modified.

The validated candidate receipt is
`/home/toffe/.local/state/biominer/upgrades/20261021/phase1/phase-reviewed.json`,
SHA-256 `9165ae56453a2c34af175d41ddde25fb8f98136dadeadce8506a3ab7bebe99e9`.
It binds exact task commits, source/configuration and artifact hashes,
independent spec/quality reviews, executed argv, logs and exit codes.
Its source digest is
`4f95ac3424cf8cb4aaef1592aa633254e48a2e486db15b467f101278a775ab56`.
Raw inventory, advisory and synthetic-fixture evidence remains under ignored
`.native-runtime/upgrades/20261021/`.

The distinct documentation commit containing this report is the phase's final
commit. After its normal push, the create-only `phase-1-pushed.json` in the
external upgrade evidence directory records the exact final and observed
remote SHAs. Phase 2 may begin only after `finalize` and `continue --phase 2`
verify that handshake.

## Verification

| Check | Actual result |
|---|---|
| Clean snapshot, full `python -m pytest -q` | 2,888 passed; 18 warnings; 75.00 seconds |
| Phase gate tests | 69 passed; 3.59 seconds |
| Native runtime focused harness recovery check | 54 passed; 5.03 seconds |
| `uv lock --check --python <approved Python 3.14.7>` | Passed; 48 packages resolved |
| Inventory/security report adapters | Passed; retained evidence hashes checked |
| Candidate receipt validation | Passed |
| Spec and quality review | Independent AI reviews passed, exact source/task bindings retained |

Two earlier attempts remain visible. Run 1 rejected captured output as
credential material before publishing a test log; its underlying test outcome
is unavailable. Run 2, with credentials excluded, recorded 11 failures and
2,877 passes: the harness's `PYTHONPATH` setting correctly triggered the native
loader safeguard. Installing the snapshot through the existing editable-package
mechanism and removing that override fixed the harness. No test, production
loader check or secret filter was weakened. The final run's 18 warnings include
the Polars explode migration warnings addressed by phase 3.

## Open findings and next boundary

The gate records nine findings and one explicit lxml backport-attestation gap.
See the [security/source review](security_compatibility_evidence_2026-10-21.md)
and [runtime inventory](runtime_exposure_2026-10-21.md). Shared AnyIO, pip and
OpenSSL exposure requires entrypoint migration, not merely a lockfile change.
The embedded libxml2 finding prevents treating the existing lxml wheel as a
qualified fallback.

Preparation for the fallback additionally identified
`NATIVE-ZLIB-CVE-2026-85091` and
`NATIVE-LIBICONV-GB18030-BUFFER-OVERRUN` in the older lxml dependency lineage.
Both are included in the final security command receipt and must carry forward
into phase 2's security closure evidence. The unchanged full-suite receipts are
retained; a separately executed security supplement and independent review
bind these additional findings.
The selected rebuild strategy uses patched libxml2/libxslt, an explicitly
reviewed zlib fix and libiconv 1.19. Exact applicability of the zlib advisory to
other native copies must be documented separately; no blanket native-library
clearance is claimed. The
[zlib fix](https://github.com/madler/zlib/commit/df84af25dc1942490e1d1c899a07619152a46148)
and [GNU libiconv release distribution](https://www.gnu.org/software/libiconv/)
are primary upstream references.

No scientific workload, source media, production output, schema, checkpoint,
model, consumer pin or unrelated dirty EDA work was changed. The inventory's
restricted-process and external-scheduler limits remain. Live PostgreSQL/S3,
benchmarks, target builds, migration, retirement and production promotion have
not occurred. Direct PostgreSQL qualification still needs the provider CA:
the supplied credentials file did not contain a certificate or certificate
path. TLS verification remains enabled.
