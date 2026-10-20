# Phase 2 — patched baseline and environment migration

Plan: [security remediation and Polars/Arrow migration](../superpowers/plans/2026-10-21-security-polars-arrow.md).
Task 2.1 started on `main` at
`355333a2fe64f8d169387091b5d1c933a5186894`, after the verified phase 1 push.
Unrelated scientific changes and all historical profiles/generations were preserved.

## Task 2.1: qualified fallback

The new baseline installs through its frozen launcher and passes all five
installation checks: actual effective versions, native inventory in both import
orders, compiled DuckDB extension policy, local certificate-verifying TLS
fixtures, and geometry/capability checks. Its subsequent fresh verification
passes the three mandatory runtime checks. This is local qualification, not
hosted PostgreSQL/S3 qualification or completion of workload migration.

The custom lxml 6.1.3 wheel contains libxml2 2.15.4, libxslt 1.1.45,
GNU libiconv 1.19 and zlib 1.3.2 with upstream patch
`df84af25dc1942490e1d1c899a07619152a46148`. Admission requires its exact wheel,
source and patch hashes, loaded lxml/native versions, compiled/runtime parser
features, retained license files and qualified relink materials.
The eight retained libxml features are catalog, HTML, iconv, regexp,
Schematron, XML Schema, XPath and zlib.

The recipe uses private PIC static libraries with at most four compiler jobs.
It binds original archives, the zero-fuzz patch application, patched
`gzwrite.c`, compiled archives, linker commands/maps and resulting extensions.
The `gzwrite.o` member in `libz.a` matches the freshly compiled object, and the
etree link map includes it. Actual ELF dependencies are libc, libm and the
loader. Existing Python, Arrow, DuckDB and Psycopg artifacts remain pinned.
The host OpenSSL admission floor advances to Ubuntu `3.5.5-1ubuntu3.7`
([USN-8861-1](https://ubuntu.com/security/notices/USN-8861-1)); no host package
was changed by this task.

GNU libiconv/libcharset notices and license texts accompany the wheel.
Corresponding source, modifications, objects, libraries, headers and replay
instructions are retained in the companion relink bundle. All seven native
extensions were reproduced byte-identically after relocating that bundle and
supplying a separate libiconv archive. The GPL iconv command is not included.
These materials must remain available alongside any distribution of the wheel.

The frozen snapshot contains byte-identical launcher/check scripts together,
project configuration, lockfile, original and relocated profiles, and an
inventory published last. Tampered, missing, extra, writable or symlinked
snapshot files fail verification. An explicit profile-bound generation root
allows persistent installation after the original repository control files
change. Abbreviated CLI options cannot override the frozen profile.

Supported operations are `artifacts`, `install`, `verify` and `exec`.
Frozen `recipes=[]` prevents rebuilding through either launcher; original
recipes remain provenance. The snapshot still requires its retained
hash-bound wheels, build materials, Python, tools and qualified host libraries.
It is independent of mutable repository control files, not of those retained
dependencies or the host.

| Artifact | SHA-256 |
|---|---|
| Built lxml wheel | `6e9f79caa8015fda671cff52f6b6c8cd3d4c3c4c351a616bf7be78956f1dfe21` |
| lxml build receipt | `3b679a784286fe584c2c56c3c4a54990d2b4b8ac0420f27f6be541aa578f370c` |
| Refreshed baseline profile | `f459aac745f0c3cde550ce05a2bc091042cdd6c917db6de59fe4d7c435b20ca7` |
| Frozen profile | `7fca0dd75f4541986cc0262bf4d8211c0b8c73543dbabc9b452a6691265f3ded` |
| Snapshot manifest | `15296af0799c9b3019545674fbd1b982886d6590392bd8b78b51cca432857b9b` |
| Installed generation receipt | `d7356ce4463d84bf2e75553b90d67f6cacc05082e92a6874af8246348467ebb0` |

Snapshot: `.native-runtime/snapshots/security-baseline-20261021-1`.
Generation: `.native-runtime/generations/security-baseline-20261021-1`.
New repository profiles retain the pre-refresh candidate and host-refresh
receipt; historical profiles are untouched. A checked invocation is:

```bash
.native-runtime/python-runtime/python/bin/python3.14 -I -B \
  .native-runtime/snapshots/security-baseline-20261021-1/scripts/freeze_native_profile.py \
  run .native-runtime/snapshots/security-baseline-20261021-1 \
  exec --generation .native-runtime/generations/security-baseline-20261021-1 \
  -- -m biominer.cli --help
```

## Verification and retained failures

The task's six focused test modules passed **182 tests in 11.13 seconds**:
`test_native_runtime`, `test_native_runtime_checks`,
`test_assemble_native_profile`, `test_refresh_native_host`,
`test_native_profile_snapshot` and `test_native_lxml`.
They ran with the owned Python 3.14.7 baseline test environment, without
`PYTHONPATH` or provider credentials. `uv pip check` against the new generation
reported all 42 installed packages compatible.

A synthetic presentation exercises Unicode text, tables, an empty cell and an
image. Reopening it succeeds under both the previous and new lxml builds.
All 39 canonical package parts match; physical ZIP hashes differ and are
retained separately. The new runtime reports libxml2 2.15.4. Its eleven native
capability checks include explicit gzip decompression, the secure decompression
default and external parameter-entity rejection. No scientific input was used.

The first lxml generation compiled but failed qualification because Python's
linker flags hid native version symbols. Its failed output remains preserved.
The second build retained linker hardening while narrowing archive symbol
hiding and passed qualification.

Initial profile assembly rejected a historical Psycopg input that referred to
the mutable lockfile and a raw DuckDB receipt that referred to a retired
temporary interpreter. The current raw DuckDB descriptor also identifies a
different policy build. Assembly now selects the exact retained artifacts and
receipts from the previous approved profile, preserving every hash check.
Previously retained backend bytes are reused without requiring retired
temporary paths. An incomplete candidate with missing DuckDB remains in the
external evidence directory and was never installed.

Command receipts, test output, presentation comparison and failed candidate
evidence are under
`/home/toffe/.local/state/biominer/upgrades/20261021/phase2/`.
Independent source, build-artifact and regression reviews were performed by
the separate native-review agent; this is not a human review claim.

## Remaining phase work

Tasks 2.2 and 2.3 remain pending at this task boundary. The shared development
environment has not been replaced, and the phase 2 push/handshake has not run.
The latest readable process scan found no shared-environment consumer, but
that scan must be repeated immediately before cutover. Unreadable processes,
other hosts and external schedulers remain outside the inventory.

The nine phase 1 security findings and its coverage gap are not globally
closed by building this fallback. Their exact scopes and remaining affected
entrypoints must be reconciled in the phase security receipt. The hosted
PostgreSQL test still needs the project CA certificate; TLS verification
remains enabled. No hosted PostgreSQL/S3 namespace has been created here.
