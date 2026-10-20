# Runtime exposure inventory — 21 October 2026

Task `1.1` of the [approved migration plan](../superpowers/plans/2026-10-21-security-polars-arrow.md).
Starting branch/commit: `main`,
`5bac7f168c1b257184615baf1524f12d5c580bc6`.
Observed at `2026-10-20T13:17:16Z` / `2026-10-21T00:17:16+11:00`.

The shared `.venv` remains an older installed environment. The separate
`dependencies-20261020-1` generation contains the reviewed dependencies, but
this inventory does not promote it. No readable process consumed the shared
environment at the observation instant. The older Python processes inspected
are Codex's Headroom MCP helpers, not identified BioMiner scientific workers.
Restricted processes and external schedulers remain outside that conclusion.

| Component | Shared `.venv` | Separate generation |
|---|---|---|
| Python | 3.14.5 | 3.14.7 |
| OpenSSL reported by a fresh Python `ssl` import | 3.5.6 | 3.5.9 |
| AnyIO | 4.14.0 | 4.15.1 |
| Polars | 1.41.2 | 1.44.2 |
| PyArrow | 24.0.0 | 25.0.1+biominer.f918dfd84c8c |
| DuckDB | 1.5.4 | 1.5.6+biominer.f8be3729f88a |
| Psycopg / Psycopg C | 3.3.4 / absent | 3.3.6 / 3.3.6 |
| NumPy | 2.4.6 | 2.5.4 |
| lxml | 6.1.1 | 6.1.3 |
| skops | 0.14.0 | 0.16.0 |

Both interpreter probes exited zero using `-I -B`; installed package metadata
and interpreter paths are retained in the receipt. These are fresh-process
facts, not assertions about modules already imported by another process.
The [20 October dependency review](backend_dependency_review_2026-10-20.md)
contains the earlier advisory analysis; task 1.2 must refresh applicable
security evidence. No new external advisory claim is made here.

The `/proc` scan covered all visible process directories, including processes
with another working directory: 54 considered and 27 inaccessible executable
or mapping records. The inaccessible same-user entries were `systemd` and
`(sd-pam)`; the other inaccessible entries belonged to system accounts.
Readable Python PIDs `89866`, `1009725`, and `1215354` all used the uv-managed
CPython 3.14.5 executable and were `headroom` MCP children of Codex PID `89587`,
owned by `toffe`. Their package roots were under
`~/.local/share/uv/tools/headroom-ai`. PID `1009725` mapped six Arrow `.so.2400`
libraries from that tool environment. No separate `_ssl`, `libssl` or
`libcrypto` mapping was observed in these helpers; their in-process crypto
version is therefore unestablished. Library pathname hashes are retained but
are not hashes of mapped memory. PID `1149245` was absent.

The helpers' filesystem descriptor basenames were only `urandom` and
`python3.14`; no scientific source, output or checkpoint descriptor was found.
No readable process referenced BioMiner `.venv` through its executable argument
or mapped files, or mapped a qualified generation. This point-in-time scan must
be repeated immediately before replacing `.venv`. It does not authorize
restarting the Codex/MCP processes, identify an owner-approved scientific
restart checkpoint, or attest other PID namespaces or hosts.

The installed `.venv/bin/biominer` launcher still has a shebang targeting
`.venv/bin/python3`; direct `.venv/bin/python` invocation is also an affected
entrypoint. System and user timer listings succeeded with no BioMiner match.
Readable definitions under `/etc/systemd/system`, `~/.config/systemd/user`,
`/etc/cron.d`, and `/etc/crontab` had no BioMiner, repository or native-runtime
reference. `crontab -l` reported no user crontab.
`/var/spool/cron/crontabs` was inaccessible. Root/other-user cron content,
remote orchestration, container launchers, shell history and future ad hoc
commands were not established. No scheduled launcher was changed.

Credential availability was checked without recording values. Nonempty
`BIOMINER_WORKSTORE_DSN` is a PostgreSQL URI with hostname, username, password,
database and `sslmode` parameter; common placeholder markers were absent.
Thus connection credentials are structurally present, but this task did not
authenticate them. `SUPABASE_KEY` and `SUPABASE_SECRET_KEY` are also present;
their presence is not PostgreSQL authentication evidence. S3 endpoint,
access-key, secret-key, bucket and prefix environment variables are present.
`BIOMINER_S3_REGION` is absent. No `.pgpass` or standard PostgreSQL service
configuration was found. PostgreSQL/S3 connectivity, permissions, qualified
client TLS, CRUD and cleanup remain for the authorized live-test task.

The coordinator's subsequent read-only preflight confirmed Supabase MCP SQL
access to PostgreSQL 17.6. Direct Psycopg connection with `sslmode=verify-full`
and the public certifi bundle failed certificate validation before
authentication could be established. This is recorded in the separate
`postgres-preflight.json` and its detailed follow-up
`postgres-tls-preflight.json` (SHA-256
`7747f09ce77c2f7ee9b8f5ead1dc0dd802fd91c2786ed2fe8338edde568dde3a`);
the original inventory receipt remains unchanged.
The provider-specific trusted CA must be obtained and verified before live
qualification. TLS verification was not disabled. Supabase documents this
requirement in its [SSL guide](https://supabase.com/docs/guides/platform/ssl-enforcement).

The raw intake manifest and archive are owned by `toffe`. The manifest is
`data/reference/gbif_global_papilionoidea_parquet/dwca_parquet_manifest.json`,
schema `biominer-dwca-members-parquet-v1`, SHA-256
`a55722bc0eddc5d052d3149463b7afa4ad7a791cc8cfdd1a44b830a96e8b7ee4`.
The archive's observed size is 24,398,287,955 bytes. Its manifest records
SHA-256 `3494944db9bca6917e0176e63852f8c233cdd7da95904c041c55ff8e8d31c6b9`;
the archive itself was not rehashed. The manifest agrees with the observed
size; `docs/PIPELINE_GROUND_ZERO.md` instead says 24,398,227,955 bytes.
This documentation discrepancy is recorded without changing the intake.
Filesystem ownership does not establish
the responsible live job owner. Historical staging reports and `run/` PID files
were not used as restart evidence. Dirty scientific scripts, EDA files, tests,
`AGENTS.md`, goal documents and output directories were preserved.

Observed Ubuntu packages were OpenSSL/libssl `3.5.5-1ubuntu3.7`,
libpq `18.6-0ubuntu0.26.04.1`, libcurl `8.18.0-1ubuntu2.8`, and
libxml2 `2.15.2+dfsg-0.1ubuntu0.3`. `/run/reboot-required` was absent.
Neither that absence nor installed package versions proves coordinated reboot,
retirement of old loaded code, or complete host security clearance.

Evidence is retained in ignored
`.native-runtime/upgrades/20261021/inventory/inventory.json`, SHA-256
`b10f840486320f119b31965e34004d5eecee7d4b4670d1e15bc2ff163fdcffec`,
with the allowlisted collection script alongside it. The receipt binds:

| Baseline artifact | SHA-256 |
|---|---|
| `pyproject.toml` | `7092cfedccd9b9b06957747742d33c0c90626b610e26a075080b03b6da27dcf1` |
| `uv.lock` | `ebb2716263ae2fd3f34277c78c13d41e8f7d99974cf62c123ab1d2d544a9d6da` |
| `native-runtime-2026-10-20.lock` | `099e9996661085c7766c89e55cbdaece7fd1c1edcc7f27ec238bf87e2ef35708` |
| Generation `generation.json` | `71cdf92d0445c5ad79be03ee77d37f85de192b0bab0e23bfae735620acbf08f5` |

The retained generation receipt says `status: installed`, `ready: false`;
fresh mandatory admission was not repeated for this metadata inventory.
Collection command:
`python3 -B .native-runtime/upgrades/20261021/inventory/collect_inventory.py`
(exit zero). The receipt is create-only. No credentials, raw command lines,
environment values, production records or source media were retained.
This local evidence task required no GitHits implementation precedent or
new scientific test. Commit/push evidence belongs to the coordinator's task
receipt; no commit or push was performed by the inventory worker.
