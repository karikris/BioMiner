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

## Task 2.2: development cutover and old entrypoints

Starting commit: `cf99ca1a75e313cf7e08f944292a9f27674821c4`.
The new development environment was built at its permanent path,
`.native-runtime/development/security-baseline-20261021-1/env`, before activation.
Locked ordinary dependencies are installed first, the four exact custom native
wheels next, and the editable BioMiner checkout last, with setuptools 84.2.0
as the build constraint. A private cache and copy mode avoid changing shared
cache permissions. All five native checks, package consistency, editable origin,
permanent parent/child interpreter identity and CLI help passed.

The dependency tree is read-only, with a separately pinned development receipt:
`246b4be1dc59e27c8bd59284797082bcd1cbadbe97f50830f57900b3aa7b1e3f`.
Its 8,264 inventory entries bind modes, links and file bytes. This is an editable
development contract; production application-wheel verification is unchanged.
Three real disposable `uv` attempts—sync, automatic run synchronization and
interpreter replacement—failed without changing dependency bytes, lock bytes
or the environment alias. A read-only `.lock` alone is insufficient: upstream
uv warns and continues when that lock cannot be acquired.

There are 46 installed distributions: BioMiner and 45 third-party packages.
The four test additions match the previous pinned advisory coverage. Metadata
discovery also sees two existing checkout `egg-info` records, including the
legacy `flickr-bio-occurrence` name; these are not extra installed dependencies
and were preserved. Independent inspection matched all 47 native files across
the four custom wheels and passed eleven fresh lxml feature checks.

Immediately before cutover, repeated scans found no eligible `.venv` consumers.
The operator-owned `.venv` directory's device/inode was checked before moving it
under `.native-runtime/quarantine/shared-venv-20261021-1/env`; the quarantine
parent then received mode `000`. Only then was `.venv` published as a link to
the new environment. The cutover does not automatically restore unsafe bytes
after a failure.

Two unreadable system bootstrap processes were excluded through an explicit,
reviewed scope option. Independent systemd observations bind exactly the user
manager and its PAM helper to root-controlled executable/unit hashes,
MainPID, UID, parent PID, start ticks, arguments and `init.scope` membership.
The pinned scope hash is
`2f12abe3aab7bbe088ad290af1cca2d39dede308868edba4ac45e74c5086ad1d`.
Activation regenerated that evidence before switching. This excludes no
descendants, services, queued jobs or future launches. Their private mappings
and environment remain unreadable; other-user processes, external schedulers
and remote hosts remain outside coverage. Visible systemd timers, unit files
and the current user's crontab contained no BioMiner launchers.

The previous approved generation's `env` directory was separately changed to
mode `000` after a fresh clear consumer scan and a check that the replacements
do not reference it. Its original directory identity, mode, complete inventory
and unchanged generation receipt are retained. Actual execution attempts through
both old entrypoints failed with `PermissionError`; subsequent scans were clear
within the stated scope. No process was killed or restarted and no scientific
workload needed draining. Historical bytes remain preserved, and the owner/root
can deliberately restore permissions; they are not an allowed fallback.

The old-entrypoint retirement receipt SHA-256 is
`4b217081bf31472d22a9d880986cd15e7cc82832f0eedf4d57a83692b7e54f63`.
The actual build used archived builder bytes with SHA-256
`5e263fc037ee6f4c2392ad6ca0beb5e9667c3c4324996582b2b5e109a9b3c394`;
activation used the finalized scope-capable script with SHA-256
`28476d02d9fe5a5881d5719b1296a2f3f39dcf3598f423302974f5fd064b963a`.
The new shared entrypoint actually reports Python 3.14.7, OpenSSL 3.5.9,
AnyIO 4.15.1 and the qualified native packages. Its console script points to the
permanent development interpreter. Long-running jobs must use permanent paths:
invoking Python through `.venv` preserves that alias in child launches.

The focused development suite passed **30 tests**; the development and
neighboring native suites passed **135 tests**. Independent source and
cutover-scope review passed. Build, inspection, activation, negative-launch and
post-cutover observations are retained in the phase evidence directory.

## PostgreSQL certificate preflight

The supplied Supabase CA file is present locally and excluded by the anchored
`/prod-ca-2021.crt` Git ignore rule. Its PEM SHA-256 is
`700723581420dd1ac98fd7e9ac529f0ef210eadcaf87fc868a3ad7d114c2f3b7`.
The existing credential configuration uses the project pooler. Authentication
and a read-only query now succeed with `sslmode=verify-full`; libpq confirms
TLS on that client connection and the server reports PostgreSQL 17.6.
No namespace or table was created.

Earlier preflight failures remain preserved: the first guard expected the
direct hostname, and the second incorrectly required TLS on the pooler's
server-side backend connection. The corrected check inspects the actual client
connection through `PQsslInUse`. The backend TLS field was false and is retained
separately; client verification does not attest that internal transport.
Successful preflight receipt SHA-256:
`f874056ac4fa426ea4066ae5782b4e97a8778c87cecf68b73e74e367e225d778`.

## Remaining phase work

The final security report retains all nine original finding IDs/scopes and the
replacement-attestation gap. They are resolved for the registered local
entrypoints, with the inventory and retained-artifact limitations above.
Report SHA-256:
`b2a21033ee83061fd3448d818f2c6455c1c6d01af07d6cf3b02495a6aa8824c2`.

Task 2.3, full phase regression and the phase 2 push/handshake remain pending
at this task boundary. Complete hosted
PostgreSQL/S3 behavior and cleanup tests remain for phase 5. No scientific
production promotion or general host security clearance is claimed.
