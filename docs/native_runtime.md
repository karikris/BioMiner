# Verified native runtime generations

BioMiner's ordinary Python dependency lock does not attest OpenSSL bundled into
native wheels or DuckDB extensions. The native runtime adds a separate,
host-specific artifact profile. Its supported target is Ubuntu 26.04, Linux
x86_64, CPython 3.14.7 with the audited Astral 20260929 build/OpenSSL 3.5.9.
It does not certify other hosts, containers, extension caches, or deployments.

The current phase-4 provider review reopened the registered Arrow 25 runtimes'
transitive curl finding and zlib applicability conflict. The shared `.venv`
points to `.native-runtime/development/arrow25-pair-5-shared/env`; that environment
and the registered fallback are disabled at their environment roots. No
currently safe qualified fallback is established. Preserve their files and
historical receipts while isolated replacement builds and qualification proceed.
See the [phase 4 report](research/upgrade_phase_4_2026-10-21.md) for containment,
source evidence and remaining gates. The private network prefix has compiled,
but that alone does not qualify an Arrow wheel or application runtime.

The [phase 2 report](research/upgrade_phase_2_2026-10-21.md) records an earlier
qualification boundary; its commands and resolved statuses do not override the
later findings. Do not use `.venv` for current development tests or replace it
through ordinary `uv sync` or automatic `uv run` synchronization. Pure policy
tests use the separately recorded pytest-only executor, which contains no
application/native dependency graph. Application tests require a new reviewed
environment through `scripts/development_runtime.py`, followed by qualification
and a stopped-consumer cutover.

Long-running jobs must use a permanent generation path. Python invoked through
the `.venv` alias retains that alias in `sys.executable`, including child launches.
Development console scripts already name their permanent environment.
The development receipt is not production-generation admission.

The old shared environment is preserved beneath an inaccessible quarantine.
The old approved generation's `env` directory is also inaccessible to ordinary
user launches; neither is an allowed rollback. The subsequently registered
patched fallback is also disabled by the current containment. Phase 2's full
regression and native qualification are historical observations; its phase
evidence and push handshake remain recorded in the phase report.
PostgreSQL authentication now passes with certificate verification through the
configured project pooler; the complete hosted PostgreSQL/S3 qualification is
still unrun.

Newly built runtimes use `biominer.polars_runtime` before importing the requested
Python workload. The [phase 3 report](research/upgrade_phase_3_2026-10-21.md)
records the completed Polars 2 semantic/resource checks; current combined
qualification awaits replacement native providers. This does not retrofit
spill behavior into historical Polars 1 generations. The supervisor
preserves explicit job budgets. Otherwise it selects four threads, a 1 GiB
engine memory budget and a 4 GiB spill budget, rounded down to Polars' decimal
MB units (1073 and 4294 MB). These are per-process engine accounting limits,
not process RSS or aggregate worker limits.

`POLARS_OOC_SPILL_DIR` names a requested parent directory. Each invocation uses
its own private child and removes it only after its worker and descendants stop.
An explicit memory fraction alone does not gain an additional default absolute
cap. Invalid settings fail before a worker starts. Started/finished receipts
retain applied budgets, exit and cleanup outcomes without recording raw command
arguments, credentials or worker output.

For explicit use, invoke the permanent environment interpreter:

```bash
/absolute/permanent/environment/bin/python -I -B -m biominer.polars_runtime \
  --scratch-parent /var/tmp --report-dir /absolute/new/report-directory \
  -- -m biominer.cli --help
```

The current installed console script has not yet moved to the new bootstrap;
that requires the next qualified environment. Direct Python scripts and library
imports require this wrapper. The supervisor requires Linux, a single Python
thread and no existing child processes. It cannot clean scratch after its own
SIGKILL or host failure; retained started receipts identify those interrupted
runs for ownership-checked recovery.

The 20 October dependency renewal retains the original profile and generation
as historical evidence. Their repository configuration and script references
are mutable: changing those files invalidates their verification. The host has
also advanced to Ubuntu OpenSSL package revision `3.5.5-1ubuntu3.7`; the earlier
generation currently fails its recorded apt-origin check. Preserve its bytes
and create a new profile and generation rather than editing its receipt.

## Historical 20 October qualification

The monthly dependency upgrade is qualified locally with
`native-runtime-2026-10-20.lock` and
`.native-runtime/generations/dependencies-20261020-1`.
The profile SHA-256 is
`099e9996661085c7766c89e55cbdaece7fd1c1edcc7f27ec238bf87e2ef35708`.
Its application wheel contains source commit
`827fa5ce4b98aa4e759350580661b6301d32c0ae` plus the reviewed dependency
configuration. The three custom native wheels and their historical build
receipts are unchanged. Host refresh records the currently installed Ubuntu
libraries separately from their build-time evidence.

At that qualification boundary all five installation checks, fresh verification,
installed package consistency and launcher help passed. Its historical command
was:

```bash
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/native_runtime.py exec --profile native-runtime-2026-10-20.lock --generation .native-runtime/generations/dependencies-20261020-1 -- -m biominer.cli --help
```

At that boundary the shared `.venv` and existing workers had not been migrated. Live PostgreSQL
and hosted S3 checks, coordinated reboot and worker retirement remain unverified.
See the [dependency review](research/backend_dependency_review_2026-10-20.md)
for version decisions, security findings, compatibility evidence and exact
qualification commands. The generic examples below describe new builds;
existing profiles and generations must not be reused as output paths.

## Build and qualify a candidate

Use a separate workspace under ignored `.native-runtime/`; preserve existing
environments and active scientific jobs. Read the source-pinned build recipes
before use. Compiler/build dependencies are developer tools, not production
Python dependencies. The PostgreSQL extra now requires a compiler, Python
headers, PostgreSQL headers and `pg_config` when installed from source.

Build Psycopg C 3.3.6 against Ubuntu libpq. Build Arrow 25.0.1 against a private,
static OpenSSL 3.6.5 prefix, including Flight, S3, Parquet encryption and the
previous wheel's other capabilities. Build DuckDB Python/core 1.5.6 with Spatial
statically linked, retaining GDAL/GEOS/network support and a pinned OpenSSL
3.6.5 vcpkg overlay. Build recipes record exact source checksums and outputs;
an unfinished compile is never a substitute for a wheel receipt.

The initial candidate uses application source frozen by `git archive` from
`074076e4ec0af5990117944b37e761613faf005a`, plus the task-owned dependency
configuration overlay. It excludes unrelated scientific changes in the dirty
checkout. Retain source, licence notices and required relinking materials if
distributing the native artifacts.

After build receipts exist, assemble the effective artifact profile:

```bash
python scripts/attest_native_python.py --output .native-runtime/python-runtime
python scripts/assemble_native_profile.py --python .native-runtime/python-runtime/python/bin/python3.14 --preserve-inputs --output native-runtime.lock
python scripts/native_runtime.py artifacts --profile native-runtime.lock
```

The assembler writes `ready: false`, records missing wheels explicitly, and
never invents artifact checksums. An incomplete profile fails admission. The
profile contains local absolute artifact paths; preserve those files or
regenerate a reviewed profile when moving them. Arrow and DuckDB custom wheels
carry explicit `+biominer.<input digest>` versions.

Assembly now requires a new output filename. Application builds require a new
output directory, separate from their build environment. Supply
`--biominer-receipt <new-build>/receipt.json` when dependencies change.
Admission checks that the application build's exact `pyproject.toml` matches
the profile configuration; an older wheel cannot qualify against new requirements.

Run build recipes through the checked profile with `native_runtime.py build
--profile native-runtime.lock --output .native-runtime/new-build`. Recipe
arguments identify already prepared, hashed compiler environments and source
inputs. Inspect the individual build logs and receipts on failure. Reassemble
the profile from the newly built artifacts before installation; the initial
assembler defaults identify the initial build directories.

The user approved eight compiler jobs total, allocated four each to Arrow and
DuckDB/Spatial. Nested builds must obey the same per-builder limit.

## Host security prerequisite

An administrator must refresh package indexes and apply Ubuntu's OpenSSL
security update. On Ubuntu 26.04, `openssl`, `libssl3t64` and `libssl-dev` must
be at least `3.5.5-1ubuntu3.6`. Use Debian package-version comparison rather
than the upstream `openssl version` string: Ubuntu applies security backports.
[USN-8847-1](https://ubuntu.com/security/notices/USN-8847-1) requires a reboot
after the system update.

Coordinate draining/rebooting with every active job owner at durable
checkpoints. Do not reboot or kill unrelated jobs automatically. Rebuild or
re-attest the system-linked adapter after the update, then regenerate the
profile to bind the updated library hashes and Ubuntu package-origin evidence.
After the coordinated host update/reboot, reattest host-linked library bytes
without rewriting the retained build-time receipts:

```bash
python scripts/refresh_native_host.py --profile native-runtime.lock --output native-runtime.post-update.lock
python scripts/native_runtime.py install --profile native-runtime.post-update.lock --generation .native-runtime/generations/reviewed-1
```

The output profile must share the input profile’s directory so relative
references retain their meaning. Host refresh fails until package security floors
and Ubuntu-origin checks pass.
It preserves wheel/source/static artifact hashes, writes a separate candidate
profile and requires all installation checks again. No existing generation or
profile is overwritten. Changing apt indexes or installed libraries can invalidate an existing profile;
review the change and create a new verified generation.

## Install and launch

```bash
python scripts/native_runtime.py install --profile native-runtime.lock --generation .native-runtime/generations/reviewed-1
python scripts/native_runtime.py verify --profile native-runtime.lock --generation .native-runtime/generations/reviewed-1
python scripts/native_runtime.py exec --profile native-runtime.lock --generation .native-runtime/generations/reviewed-1 -- -m biominer.cli --help
```

Installation selects ordinary packages, including the PostgreSQL, geometry and
ML extras, from the Python lock and installs
approved native/application wheels separately with hashes and no dependency
resolution. It runs mandatory checks before freezing the new environment.
Verification checks installed bytes, interpreter/native-library identity,
package ownership, host security floors, and compiled extension policy.
Launch rejects unapproved loader overrides and uses isolated installed code.

Do not run ordinary `uv sync` against a verified generation. Updates create a
new generation; existing generations remain read-only. The profile is trusted,
reviewed operator input, not a self-authenticating certificate or protection
against arbitrary code executed by the same user or root.

## Extensions and acceptance

Spatial is built in. Initial production policy allows no external extensions.
Audited external extensions require a new custom DuckDB generation containing
their exact allowlist and dependency policy. The loader retains official
signature/ABI checks and loads a sealed snapshot of approved bytes; unsigned
fallbacks and unknown dependencies are rejected. Audited dependencies must already be mapped from the exact approved paths
and hashes before external `LOAD`; a separate audited profile must validate
its controlled preload step. The signed INET positive test used explicit
preloads, not the initial production launcher policy. Relative dependency
resolution that cannot be verified under sealed-file loading is unsupported.

Mandatory checks cover effective wheels, mapped native libraries, compiled
extension policy, local TLS certificate acceptance/rejection, and spatial/Arrow
capabilities. Local TLS fixtures are not hosted database CRUD or S3 integration
tests. Live checks without credentials remain explicitly unrun.

Promotion requires passing mandatory checks, a verified host update, and
migration of inventoried local production entrypoints to this launcher. Until
then, the build is a candidate and the existing exposure remains unresolved.
Keep old generations for investigation; do not restore network workloads using
known affected artifacts. Pause affected jobs if no security-approved generation
can run. Scientific release and downstream handoff gates remain separate.

## Historical 1 October qualification

The retained production DuckDB receipt is
`.native-runtime/duckdb/root-production-wheel-receipt.json`, pointing at preserved
source materials. Separate test-policy relinks must not replace it. When
refreshing the initial profile after such a test build, pass
`--duckdb-receipt .native-runtime/duckdb/root-production-wheel-receipt.json` to the
assembler. Production policy remains the empty external allowlist.

The frozen application baseline passed 2,708 tests with the actual custom
wheels; local TLS passed six certificate checks. These candidate checks do not
attest retirement of existing processes or a coordinated reboot. Detailed artifact fingerprints and
remaining actions are in the [OpenSSL review](research/openssl_security_review_2026-10-01.md).

On 1 October, `.native-runtime/generations/security-20261001-1` passed verification,
bound to profile SHA-256
`e47aff33c1020f3812e6e4b7b58db9b57771f161cbc682a39976fe975682e61e`.
Its five installation gates and subsequent verification passed at that time.
The historical launch command below is currently blocked by changed host and
repository inputs; it is retained for traceability:

```bash
python scripts/native_runtime.py exec --profile native-runtime.lock --generation .native-runtime/generations/security-20261001-1 -- -m biominer.cli --help
```

The host security revision was observed changing externally to the required
Ubuntu update during qualification. Fresh-process library admission was verified then;
coordinated reboot and retirement of old workers remain unconfirmed.
