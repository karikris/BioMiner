# Backend dependency renewal — 20 October 2026

Task: `backend-dependencies-20261020`. Starting branch/SHA:
`main` / `074076e4ec0af5990117944b37e761613faf005a`.
The user authorized continuation of the existing dependency/native changes.
Unrelated `AGENTS.md`, scientific scripts, EDA changes, tests and goal documents
were preserved. No production dataset or model job was started.

## Execution and boundaries

1. Configure and authenticate the project-scoped Supabase MCP server.
2. Inventory both the installed environment and dependency lock. Review current
   package metadata, upgrade ranges, release history, advisories and source.
3. Verify a candidate independently, checkpoint the authorized native work,
   apply the reviewed dependency patch, and qualify a new immutable generation.
4. Review exact staged scope, commit on `main`, push, and verify remote identity.

Supabase was registered with the requested project/features URL. Both the
automatic add flow and `codex mcp login supabase` succeeded; a filtered
`codex mcp list --json` returned `enabled: true`, `auth_status: o_auth`.
The interactive `/mcp` UI command was not executed by the shell.
Supabase skills were already installed. No project database, functions or
permissions were changed. Setup followed the
[Supabase MCP guide](https://supabase.com/docs/guides/getting-started/mcp).

The native checkpoint is
`827fa5ce4b98aa4e759350580661b6301d32c0ae`, pushed and independently verified
as `origin/main` before applying the monthly patch. It completes the existing
native tooling and fixes application-configuration admission and output reuse.

## Installed versions are not the lock

The shared `.venv` still reports Python 3.14.5/OpenSSL 3.5.6 and dependencies
predating the October 1 lock. Examples: AnyIO 4.14.0, lxml 6.1.1, skops 0.14.0,
DuckDB 1.5.4, Arrow 24.0.0 and NumPy 2.4.6. It was preserved; updating a lock
does not patch existing environments or running workers.

Current GitHits advisory lookup identifies three findings affecting installed
AnyIO 4.14.0:

| Advisory | Finding | Fixed from |
|---|---|---|
| [GHSA-82r6-8w77-94w6](https://github.com/advisories/GHSA-82r6-8w77-94w6) | Critical TLS hostname encoding/certificate spoofing risk | 4.14.2 |
| [GHSA-3w57-8xmc-8v26](https://github.com/advisories/GHSA-3w57-8xmc-8v26) | Supplementary groups can be retained by subprocesses | 4.14.2 |
| [GHSA-5p39-cfhj-2xmp](https://github.com/advisories/GHSA-5p39-cfhj-2xmp) | Process-pool stderr can block workers | 4.14.2 |

The lock already selects AnyIO 4.15.1. A fresh batch review also covers the 29
installed-to-locked version differences. Skops 0.15 adds type-trust, random
generator and archive-protocol hardening even though version-specific advisory
lookup returns no active match for 0.14.0. BioMiner's classifier artifact format
remains numeric NPZ/JSON, without introducing untrusted skops loading.
[Upstream release](https://github.com/skops-dev/skops/releases/tag/v0.15).

The candidate's 47 third-party locked versions all returned no active direct
advisory matches on this retrieval. This is registry coverage, not proof that
embedded libraries, other platforms, undisclosed issues or running deployments
are safe.

## Selected changes

| Package | Previous lock → selected | Relevant changes |
|---|---|---|
| Pillow | 12.3.0 → 13.0.0 | TIFF bounds/lifetime fixes, decompression-bomb checks, stricter malformed image handling; removes deprecated APIs and Python 3.10 support. Synthetic JPEG/PNG/TIFF/WebP decoding and downloader decisions were compared. |
| Pydantic | 2.13.5 → 2.14.1 | Stricter Decimal, timedelta, positive `multiple_of`, Fraction and NameEmail validation. No direct production use found in the inspected source. |
| pydantic-core | 2.46.5 → 2.50.1 | Exact requirement of Pydantic 2.14.1, confirmed in upstream `pyproject.toml`; not selected independently. |
| NumPy | 2.5.3 → 2.5.4 | SVD/least-squares reject infinities; masked-array fill values are corrected after dtype changes. Synthetic classifier margins and stored arrays remain identical. |
| certifi | 2026.7.22 → 2026.10.19 | Removes emSign ECC Root CA - G3. Source comparison found unchanged `core.py`; offline TLS contexts loaded 121 versus 120 roots, with exactly that deletion. |
| fonttools | 4.66.1 → 4.67.0 | Font parsing/layout fixes, stricter malformed CFF/feature handling; VARC binary-format changes require rebuilding affected fonts. No VARC generation was introduced. |
| iniconfig | 2.3.0 → 2.3.1 | UTF-8 BOM support and linear handling of continued values. |
| narwhals | 2.26.0 → 2.27.1 | Backend/null/string compatibility corrections and Polars compatibility maintenance. |
| pluggy | 1.6.0 → 1.7.0 | Python 3.14 annotation registration fixes, hookspec defaults and stricter/deprecated hook declarations. Full pytest execution checks the consumer. |
| tzdata | 2026.4 → 2026.5 | IANA 2026e timezone update; Windows-only resolution is not runtime-tested on this Linux host. |
| wcwidth | 0.9.1 → 0.9.2 | Terminal-width maintenance; no added dependency/advisory findings in the reviewed range. |

Release evidence:
[Pillow](https://github.com/python-pillow/Pillow/releases/tag/13.0.0),
[Pydantic](https://github.com/pydantic/pydantic/releases/tag/v2.14.1),
[Pydantic core requirement](https://github.com/pydantic/pydantic/blob/b2212c78/pyproject.toml),
[NumPy](https://github.com/numpy/numpy/releases/tag/v2.5.4),
[certifi source comparison](https://github.com/certifi/python-certifi/compare/2026.07.22...2026.10.19),
[fonttools](https://github.com/fonttools/fonttools/blob/b0fa59394498d146b73e2482ae7c17444ddea50b/NEWS.rst),
[iniconfig](https://github.com/pytest-dev/iniconfig/releases/tag/v2.3.1),
[narwhals](https://github.com/narwhals-dev/narwhals/releases/tag/v2.27.1),
[pluggy](https://github.com/pytest-dev/pluggy/blob/78e3f45d6c18a6efe6c46a2a79b63073f45a44bd/CHANGELOG.rst),
[tzdata](https://github.com/python/tzdata/releases/tag/2026.5),
[wcwidth](https://github.com/jquast/wcwidth/releases/tag/0.9.2).

The setuptools build floor moves from 84 to 84.2. Upstream 84.1 fixes namespace
path validation in legacy egg installation and archive extraction traversal.
Source inspection confirms dotted-name validation and destination containment.
The named GHSA pages returned 404 through Valyu and GitHits returned no active
84.0 advisory match: these are documented upstream security fixes, with that
advisory coverage gap retained. BioMiner uses PEP 517, not `setup_requires`.
[Upstream changes](https://github.com/pypa/setuptools/blob/301e27a484f17e175acf6b692fccc634f9a20c49/NEWS.rst),
[validation implementation](https://github.com/pypa/setuptools/blob/301e27a4/setuptools/wheel.py).

The fresh, separate application builder uses setuptools 84.2.0, build 1.6.1,
wheel 0.48.0, packaging 26.3 and pyproject-hooks 1.3.3. Build 1.4→1.6.1 was
reviewed separately: it improves isolation, backend validation and constraint
handling while removing a deprecated exception argument unused here.
These are build tools, not additions to the runtime graph.
[Build changelog](https://github.com/pypa/build/blob/a1887073399bc1d92a0e94bf1650524f983b5574/CHANGELOG.rst).

## Deliberately retained versions

Polars remains 1.44.2 with `<2`. Source confirms that 2.0 changes
`explode(empty_as_null=False)`; existing query-provenance processing and warnings
need a dedicated semantics migration. Other 2.0 changes include SQL/Decimal,
ragged-row rejection and out-of-core defaults.
[Release](https://github.com/pola-rs/polars/releases/tag/py-2.0.0),
[implementation](https://github.com/pola-rs/polars/blob/22a147de/py-polars/src/polars/expr/expr.py).

Arrow remains the approved custom 25.0.1 artifact. Arrow 26 was reviewed,
including null-fill buffer hardening and Parquet schema-depth checks.
Source establishes the default depth limit of 100 and Python's optional
`schema_depth_limit` argument. Its static build requirements, removed APIs and
custom OpenSSL/S3/Flight qualification require a separate native rebuild;
installing the stock 26 wheel would discard this project's native attestation.
This deferral does not claim that Arrow 25 contains all Arrow 26 hardening.
[Changelog](https://github.com/apache/arrow/blob/ee382216db4f8d764ed17c16c5baef45afe3f757/CHANGELOG.md),
[reader properties](https://github.com/apache/arrow/blob/44616716/cpp/src/parquet/properties.h).

DuckDB 1.5.6, psycopg/c 3.3.6 and the attested Python 3.14.7 remain unchanged.
The actual custom Arrow/DuckDB/Psycopg wheel checksums were verified before
installation in both test environments. No model weights, scientific schemas,
score semantics, consumer pins or production Parquet were changed.

## Verification evidence

Paired checks ran before and after upgrading, using identical synthetic files.
Decoded RGB bytes, image perceptual hashes, downloader outcomes and storage
side effects, nullable/list/empty Parquet values and bytes, and all three
classifier margins, predictions, class order and numeric artifact bytes matched.
Malformed, missing and null cases were included. Classifier artifacts also
cross-load between environments. NumPy's version change correctly alters
training/classifier fingerprints and manifests; this expected provenance change
was isolated rather than suppressed.

The candidate at the starting source commit passed 192 focused tests and then
2,708 full-suite tests, with 18 existing Polars warnings. Source distribution,
wheel build, CLI help, `uv pip check` and `uv lock --check` passed. The first
full archive run had 21 Git-identity failures; detached metadata for the exact
source commit resolved all 21, followed by the complete passing run. No source
behavior or test contract was weakened.

Native regressions first produced nine expected failures. The repaired tooling
passed 111 tests on Python 3.14.7 and independent spec/quality review.
The old generation fails fresh host-origin verification after an external
Ubuntu OpenSSL update to `3.5.5-1ubuntu3.7`; its historical receipt is not current
admission evidence. No OS upgrade or reboot was performed here.

Exact tool responses, paired probe source, fixture outputs, command receipts and
initial failures are retained locally under
`/tmp/biominer-deps-20261020-z424m0b2`.
The review used `pkg_info`, four retained `pkg_upgrade_review` batches,
`pkg_changelog`, pinned `pkg_vulns`, `search`, `grep` (the current replacement
for `code_grep`), `read`, `code_diff` and an eligible `search_status` continuation.
The retained evidence includes 51 package metadata responses, 15 changelog
responses, and 47 candidate advisory lookups. Upgrade-review batch sizes were
6 direct candidates, 9 additional dependencies, 29 installed-to-locked
differences and 2 build tools.
Final review found that the build-tools responses were absent from the initial
raw export. That batch and Build changelog were retrieved again and saved in
`githits-build-tools-evidence.json`; the refreshed results support the same
version and compatibility decisions.
Missing certifi/core release bodies were investigated in source. GitHits
occasionally served a marked older compatible index snapshot; resolved commits
and that coverage limitation remain in the raw evidence. No private source or
credentials were sent to upstream research tools.

## Integrated and native qualification

The final source snapshot is checkpoint
`827fa5ce4b98aa4e759350580661b6301d32c0ae` plus the reviewed `pyproject.toml`
and `uv.lock` overlays. Integrity checks covered 729 tracked files; unrelated
dirty scientific edits were excluded. The complete integrated suite passed
**2,819 tests**, with 18 warnings, in 73.38 seconds. Its 46 installed package
versions, including custom native wheels, matched the tested candidate.

Exact integrated test command, from the isolated `final-source` directory:

```bash
POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /tmp/biominer-deps-20261020-z424m0b2/candidate-venv/bin/python \
  -m pytest -q --basetemp /tmp/biominer-deps-20261020-z424m0b2/pytest-integrated
```

The application wheel was rebuilt using the separate persistent builder:

```bash
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/build_native_project.py \
  --python .native-runtime/builds/dependencies-20261020-1-build-env/bin/python \
  --root /home/toffe/github/karikris/BioMiner \
  --output .native-runtime/builds/dependencies-20261020-1
```

Wheel SHA-256:
`11bd927b848a6899b66e0d51a3b392509575fe8b48bd838d4752e92eeaccd681`.
Its preserved source receipt is
`54068e375f218015ea795a8b90c2615134a75ea6656917e93021d3355e3b498d`.

The first profile assembly correctly refused an original Psycopg receipt whose
live `uv.lock` input had changed. A new candidate was instead derived from the
historical profile's already preserved receipts. It retains the exact native
artifacts, source receipts, interpreter and extension policy. Only application
artifact/receipt, project/lock fingerprints, current recipe fingerprints and
application builder path changed. No historical receipt was rewritten to
claim a new build. The derivation script and receipt are retained locally.

Candidate SHA-256:
`07d541a0326ae583e17bfd7083b8783a6b7062104f4b8f2156d578651905462a`.
Artifact verification passed before host refresh. Host refresh observed four
changed libraries: `libssl`/`libcrypto` from Ubuntu `3.5.5-1ubuntu3.7`,
`libxml2` from `2.15.2+dfsg-0.1ubuntu0.3`, and `libcurl` from `8.18.0-1ubuntu2.8`.
These updates were external to this task; their observed versions and ownership
are deployment evidence, not a claim of a complete OS advisory audit.

```bash
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/refresh_native_host.py \
  --profile .native-runtime/profiles/dependencies-20261020-1/candidate.lock \
  --output .native-runtime/profiles/dependencies-20261020-1/qualified.lock
```

The resulting profile was copied byte-for-byte, with exclusive creation, to
`native-runtime-2026-10-20.lock`. All file references are absolute or generation
placeholders. Its SHA-256 is
`099e9996661085c7766c89e55cbdaece7fd1c1edcc7f27ec238bf87e2ef35708`.

```bash
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/native_runtime.py install \
  --profile native-runtime-2026-10-20.lock \
  --generation .native-runtime/generations/dependencies-20261020-1
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/native_runtime.py verify \
  --profile native-runtime-2026-10-20.lock \
  --generation .native-runtime/generations/dependencies-20261020-1
.native-runtime/python-runtime/python/bin/python3.14 -I -B scripts/native_runtime.py exec \
  --profile native-runtime-2026-10-20.lock \
  --generation .native-runtime/generations/dependencies-20261020-1 \
  -- -m biominer.cli --help
uv lock --check --python .native-runtime/python-runtime/python/bin/python3.14
uv pip check --python .native-runtime/generations/dependencies-20261020-1/env/bin/python
```

Installation passed `effective_versions`, `native_inventory`, `extension_policy`,
`tls` and `geometry`; the frozen receipt inventories 7,337 files. Fresh
verification passed, launcher help succeeded, lock validation resolved 48
packages, and all 42 runtime packages passed dependency consistency checks.
The profile itself retains candidate status; successful local generation
verification does not promote production workloads.

Independent final review passed for the six staged paths, including the
supplemental build evidence. Staged JSON/TOML, exact artifact/configuration
hashes, distribution hosts/checksums, credential-pattern screening and
`git diff --check` passed. No dependency graph additions or removals occurred.

Live provider, PostgreSQL CRUD and hosted S3 integration remain unrun.
Existing workers and the shared `.venv` were not migrated. No biological
performance, occurrence release, complete native-security clearance or
production promotion is claimed.
