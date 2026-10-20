# Native OpenSSL review — 1 October 2026

**20 October continuation:** the user authorized completing the existing native
work. The original qualification results below remain historical. A fresh
verification of the original generation now fails with
`host apt origin evidence changed or absent: openssl`; the observed Ubuntu
OpenSSL revision is `3.5.5-1ubuntu3.7`. No OS update or reboot was performed in
this continuation. Application/configuration binding and create-only build and
profile publication were added after nine failing regression cases. The native
tooling suite passes **111 tests on Python 3.14.7**. Independent spec and quality
reviews found no blocking issues. A new monthly dependency profile must be
qualified before launch; old profile/generation bytes are retained.

The registry wheel upgrades below retain affected native copies. The follow-up remediation now builds separate candidate artifacts with patched private OpenSSL and source-linked Psycopg; **production migration and reboot/process retirement remain pending**. See the follow-up section for the replacement artifacts and qualification evidence. The privately tested CPython build contains the current patched OpenSSL 3.5.9. The shared virtual environment and host packages were preserved; repository version pins do not migrate existing processes. See the [backend upgrade review](backend_dependency_review_2026-10-01.md) for package changes and verification, and the [separate frontend findings](frontend_tooling_security_review_2026-10-01.md) for the report-only sibling scope.

Task: `backend-dependencies-20261001`. Baseline: `main` at `0ba1784f673888ceff8da81c4eb3b6652011dbc4`. Inspection covers Linux x86_64 glibc, standard CPython 3.14 and the exact wheel/extension artifacts below. Advisory status was retrieved on 1 October 2026. Counts below mean published version-range matches, not proven exploitable defects; API use, platform, build options and documented security backports matter.

## Follow-up implementation — preserve the complete native stack

Task: `native-openssl-remediation-20261001`; implementation baseline `main` at
`074076e4ec0af5990117944b37e761613faf005a`. The earlier inventory and range matrix
below describe registry artifacts, not the newly built custom artifacts.
Unrelated science work, the shared `.venv`, and existing signed extension cache
were preserved. TaxaLens and ButterflyLens remain report-only.

The PostgreSQL extra now selects `psycopg[c]` 3.3.6. Its actual CPython 3.14 wheel
uses Ubuntu libpq 18.6 rather than `psycopg-binary`'s bundled OpenSSL 3.5.8.
This introduces compiler/PostgreSQL-header requirements for ordinary source
installation and makes the Ubuntu security update a deployment prerequisite.
Arrow 25.0.1 was rebuilt with private static OpenSSL 3.6.5, preserving S3,
Flight, Parquet encryption, Dataset, Acero/Compute, Azure, GCS, HDFS, ORC,
Substrait, CSV and JSON capabilities. The AWS build uses OpenSSL for its
crypto provider; its alternative AWS-LC path was removed for this candidate.
DuckDB Python/core 1.5.6 and Spatial 04270fe were rebuilt with the GDAL/GEOS/
CURL network feature set and private OpenSSL 3.6.5. Spatial is statically
registered, retaining ordinary `LOAD spatial` behavior.

Release notes could not establish those build properties. GitHits source
inspection, exact source archives, linker maps/static archives, actual ELF
closure and actual wheel probes supplied that evidence. Source-level fixes
were needed for AWS crypto selection, third-party unity-build symbol conflicts,
DuckDB's upstream `DONT_LINK` Spatial registration and the callback registration
for its metadata-check setting. Failed intermediate candidates were not admitted.
The inventory distinguishes the actual provider from CPython’s exact
PSF-2.0 `get_fips_mode` documentation string mentioning OpenSSL 3.0.0;
other old-provider markers still fail. The private OpenSSL build passed five upstream test files containing 168 tests;
non-FIPS preparation was skipped. These are regression results, not exploitability
certification for every advisory in the matrix.

The production DuckDB loader admits no external extensions. A separate compiled
test policy successfully loaded the exact signed official INET artifact through
a sealed snapshot; it was not substituted for the production wheel. Official
signature and ABI checks remain active. Unknown artifacts, affected cached
Spatial and unsigned/metadata-check bypass settings are rejected. The launcher
blocks unapproved loader and GDAL plugin overrides. This policy assumes an
operator-controlled process baseline, not arbitrary hostile code already running
inside the process.

The persistent interpreter is freshly extracted from the official Astral
20260929 archive, whose SHA-256 is
`da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e`.
It reports Python 3.14.7/OpenSSL 3.5.9. Archive contents, bytecode and unexpected
files/symlinks are checked; the executable hash alone is insufficient.
The application wheel uses committed baseline source plus the task-owned
`pyproject.toml` overlay, excluding the other session's scientific edits.

Build inputs, toolchain bytes, backend files, source archives, static libraries
and linker maps are retained under ignored `.native-runtime/build-materials`.
The machine-specific `native-runtime.lock` binds source receipts, actual wheel
checksums, library ownership, mandatory checks and extension policy. It remains
`ready: false`; artifact existence alone cannot approve deployment. See
[the operator workflow](../native_runtime.md) for building a new generation,
post-update reattestation and verified launch.

During qualification, `openssl`, `libssl3t64` and `libssl-dev` changed from
**3.5.5-1ubuntu3.5** to the required **3.5.5-1ubuntu3.6**, with official Ubuntu
security/updates archive provenance. This agent did not perform that update.
The earlier native profile correctly rejected the changed library hashes; a
Host reattestation subsequently passed: only the two OpenSSL deployment
library hashes changed; build receipts remained unchanged. All 35 Ubuntu
package provenance/version gates passed.
[USN-8847-1](https://ubuntu.com/security/notices/USN-8847-1) specifies a reboot.
No reboot-required marker was observed, which does not attest that every old
process was retired or the notice’s reboot requirement was completed. Coordination
with active job owners, process retirement, entrypoint migration and production
promotion remain unperformed. In the original 1 October session, Git metadata
was read-only, so that session did not commit or push this follow-up.
The 20 October continuation has separate commit/push evidence.
Live PostgreSQL CRUD and hosted S3 tests remain unrun.

### Candidate artifacts and actual verification

| Package | Actual candidate version | Wheel SHA-256 |
|---|---|---|
| biominer | 0.1.0 | `0db1c50386dc4fc33153b08c66a9deb511cc95cd4caa99ef218d79a46e2acc84` |
| psycopg-c | 3.3.6 | `fd12633b428ca00ad35d8ae6a0db769ca15ee80ec2401eb3fbcd8b2afc32f6bc` |
| pyarrow | 25.0.1+biominer.f918dfd84c8c | `67b10809882d255635e19b048c69ba7b60d90d25cd406fe9aed712631a7142ee` |
| duckdb | 1.5.6+biominer.f8be3729f88a | `83a527979c560899777c274002674ebc989d126789716b0d8c105ff13b9ba8a0` |

The committed baseline source passed **2,708 tests**, with 18 existing Polars
deprecation warnings, in 61.98 seconds on the patched host using the actual replacement wheels.
The candidate environment passes `uv pip check` for 46 packages and the
installed CLI help smoke check. Effective package identity, geometry/Arrow
capabilities, extension-policy rejection and all six local TLS certificate
checks passed. Audited test-only policies directly demonstrated rejection of
wrong ABI metadata and corrupt signatures after passing the native SHA gate,
alongside signed INET acceptance and immutable loaded bytes after path replacement.
The final profile SHA-256 is
`e47aff33c1020f3812e6e4b7b58db9b57771f161cbc682a39976fe975682e61e`.
The new immutable local generation
`.native-runtime/generations/security-20261001-1` passed all five installation
gates and subsequent verification; its verified launcher ran CLI help.
It inventories 7,335 generation files and passes `uv pip check` for 42 packages.
The native tooling/security suite passed **100 tests** in 10.15 seconds.
Both opposite native import orders pass with patched OS library hashes.
`ready: true` applies to that verified local generation, not to retirement of
old workers or a complete production deployment. The artifact profile itself
continues to say `ready: false` until an installed generation is verified.
All binary/probe artifacts stay in ignored storage.

## Observed inventory and remaining work

| Runtime owner | Original observed version | Selected/tested artifact | Remaining exposure and remedy |
|---|---|---|---|
| uv-managed CPython | Python 3.14.5, OpenSSL 3.5.6; 38 catalog range matches | Private Python **3.14.7+20260929**, OpenSSL **3.5.9** | Tested build passes the reviewed affected ranges. Migrate the actual runtime and restart processes; verify its OpenSSL rather than relying on `.python-version` alone. Shared `.venv` remains on the original runtime. |
| `psycopg-binary` | 3.3.4, OpenSSL 3.5.4; 57 matches | **3.3.6**, OpenSSL **3.5.8** | **13 September advisories remain**. Obtain a wheel with OpenSSL >=3.5.9 or documented equivalent backports. A separately validated `psycopg[c]` deployment against patched libpq/system OpenSSL is another option; this task did not change the deployment backend. |
| PyArrow `libarrow`/`libarrow_flight` | 24.0.0, OpenSSL 3.6.0 markers; 61 matches | **25.0.1**, OpenSSL **3.6.2** markers | **41 matches remain**: 17 June, 11 August, 13 September. Obtain a validated vendor wheel/rebuild with OpenSSL >=3.6.5 or documented equivalent backports, covering both native copies. |
| DuckDB Spatial | Previous extension cache preserved and not inventoried | Signed **DuckDB 1.5.6 / Spatial 04270fe**, OpenSSL **3.6.0** marker | **61 matches**. Provision a signed upstream extension rebuild with >=3.6.5 or documented equivalent backports. The current signed extension permits local geography verification but does not resolve native advisories. |
| Ubuntu 26.04 host | `openssl`/`libssl3t64` **3.5.5-1ubuntu3.5** | No host update performed | August fixes were backported by the vendor. September fixes require **3.5.5-1ubuntu3.6** or later supported vendor package, plus process restarts. Assess this package by Ubuntu's revision, not just its upstream 3.5.5 marker. |

Current fixed upstream lines are [OpenSSL 3.5.9 and 3.6.5](https://openssl-library.org/news/secadv/20260929.txt). Ubuntu's installed August backports are documented in [USN-8678-3](https://ubuntu.com/security/notices/USN-8678-3); its September security update is [USN-8847-1](https://ubuntu.com/security/notices/USN-8847-1), corroborated by the current [CVE-2026-35189 vendor status](https://ubuntu.com/security/CVE-2026-35189).

Updating system OpenSSL does not replace statically embedded Arrow/Spatial code or Psycopg's bundled libraries. Updating CPython repairs its own TLS implementation only. A zero-finding PyPI audit cannot attest these native components.

## Security fixes and reachable routes

| Finding | Effect of this upgrade | Application context |
|---|---|---|
| **CVE-2025-15467, High:** malformed encrypted CMS data can overflow a parser stack | Original Psycopg 3.5.4 and Arrow 3.6.0 are in range; fixes are 3.5.5/3.6.1. Both wheel upgrades pass those ranges. Spatial's newly examined 3.6.0 remains in range. | Requires the affected CMS/PKCS7 parsing API. Ordinary HTTPS/PostgreSQL TLS does not establish that trigger. [January advisory](https://openssl-library.org/news/secadv/20260127.txt). |
| **CVE-2026-45447, High:** PKCS7 signature verification can use freed memory | Original Psycopg, CPython and Arrow versions are in range. Selected Psycopg and private CPython pass the 3.5.7 fix; selected Arrow 3.6.2 and Spatial 3.6.0 remain below 3.6.3. | No direct BioMiner PKCS7/S-MIME verification call was found in the reviewed routes. [June advisory](https://openssl-library.org/news/secadv/20260609.txt). |
| **CVE-2026-84782, High:** DTLS handshake retransmission can use a stale buffer offset | Psycopg, Arrow and Spatial candidates remain below 3.5.9/3.6.5. Private CPython passes the range. | This requires DTLS. The reviewed HTTPX, Arrow S3 and PostgreSQL connections use ordinary TLS; their existence does not prove DTLS exposure. [September advisory](https://openssl-library.org/news/secadv/20260929.txt). |
| **CVE-2026-35189, Low:** a certificate's relative CRL distribution-point name can cause excessive allocation | Remains in the three native candidates and the unupdated host package. Private CPython 3.5.9 fixes its copy. | Certificate extension processing is reachable during ordinary TLS; enabling CRL checking is not required. GitHits read the actual fixed code, which defers expensive name expansion until CRL matching. [September advisory](https://openssl-library.org/news/secadv/20260929.txt), [fixed source](https://github.com/openssl/openssl/blob/45e844fa2a14ec92d146bd8f5778ac130b6625fb/crypto/x509/v3_purp.c#L350). |

BioMiner's current source-media resolver uses HTTPX at `src/biominer/gbif_media_resolution/resolver.py:350`, so the Python interpreter's TLS build matters. Arrow's independent native S3 connection is created at `src/biominer/storage/s3.py:528`; upstream [S3 implementation](https://github.com/apache/arrow/blob/beccec0d0c451b7aa3e4530416ac431b3c035c69/cpp/src/arrow/filesystem/s3fs.cc#L1160) selects HTTPS and certificate verification. PostgreSQL uses `psycopg.connect(self.dsn)` at `src/biominer/workstore/postgres.py:758`, reaching packaged libpq/OpenSSL when TLS is negotiated. No live provider, S3 or PostgreSQL checks ran.

Spatial's exact [CMake source](https://github.com/duckdb/duckdb-spatial/blob/04270fe7bb4eb412765cce8afde0276a9d81d3ba/CMakeLists.txt#L29) defaults `SPATIAL_USE_NETWORK` to `ON` and links CURL/OpenSSL. Its [GDAL build recipe](https://github.com/duckdb/duckdb-spatial/blob/04270fe7bb4eb412765cce8afde0276a9d81d3ba/vcpkg_ports/gdal/portfile.cmake#L90) enables those features and requests static OpenSSL. GDAL HTTPS drivers/filesystems can therefore reach native certificate parsing; the exact path depends on the driver and DuckDB filesystem integration. Local GeoJSON/geometry tests do not exercise this route. A custom `SPATIAL_USE_NETWORK=OFF` build is an option only for a separately validated offline deployment, with extension trust and compatibility reviewed; no such build or trust-policy change was performed.

## Artifact and source identity

| Inspected artifact | SHA-256 |
|---|---|
| Psycopg 3.3.6 `cp314` manylinux2014 x86_64 wheel | `31cd942c23f613276b81a6e6598cefa12960058b0f46e1e874b540c793f6aca5` |
| Arrow 25.0.1 `cp314` manylinux_2_28 x86_64 wheel | `9171748cdf796972d85a4b60157c279913e242992e350c90c7450182a9838b2a` |
| Private CPython 3.14.7 executable from Astral 20260929 build | `1747426f232ecbe022fc83eef7e56ccedcb9841c3acea82c43f1f6a013c05027` |
| DuckDB 1.5.6 Linux amd64 Spatial extension, 80,982,694 bytes | `bcaa2dd6b967b20985bf34aef3f9f1c04621d81546580cf6ee897c388bdd67c0` |

Psycopg's version was obtained from its actual `OpenSSL_version(0)` result through `ctypes`; ELF inspection locates its bundled `libcrypto`/`libssl`. Arrow and Spatial have embedded release markers and no dynamic system `libssl`/`libcrypto` dependency. Marker evidence does not provide a complete vendor patch attestation; no documented equivalent backports were found for these candidates. The managed interpreter reports `ssl.OPENSSL_VERSION`, and its built-in `_ssl` has no dynamic system crypto dependency.

GitHits exact release source: Psycopg `3.3.6` at `a67654d1e7afbf9b3a619557838f62de1c790e7c`, `.github/workflows/packages-bin.yml` pins OpenSSL 3.5.8 and `tools/ci/build_libpq.sh` shows no observed patch step; `docs/news.rst` and `docs/basic/install.rst` fill missing package release bodies. Arrow `apache-arrow-25.0.1` at `beccec0d0c451b7aa3e4530416ac431b3c035c69` supplied third-party build and S3 implementation evidence, supplemented by the exact wheel inspection. Spatial source is `04270fe7bb4eb412765cce8afde0276a9d81d3ba`.

The [Astral build recipe](https://github.com/astral-sh/python-build-standalone/blob/4a7348fcaa53d686c894674a5071f52f8b54dac6/pythonbuild/downloads.json#L306) pins OpenSSL 3.5.9. The tested [20260929 interpreter artifact](https://releases.astral.sh/github/python-build-standalone/releases/download/20260929/cpython-3.14.7%2B20260929-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz) corroborates it. Local uv 0.11.19 knew only Python 3.14.5; privately downloaded uv 0.12.21 supplied the current catalog. An old cached or differently built Python 3.14.7 must still be inspected.

## Coverage and provenance

GitHits supplied `pkg_info`, batched `pkg_upgrade_review`, `pkg_changelog`, `search`, explicit `search_status` continuations, `code_grep` and bounded `read`. PyPI reviews returned no direct Psycopg/Arrow package advisories, which was treated as incomplete native coverage. Unpinned OpenSSL HEAD initially served an older snapshot; exact `openssl-3.5.9` source resolved the implementation question. GitHits did not expose the current OpenSSL/Ubuntu advisory website inventory, so Valyu/web retrieved official primary pages as a disclosed fallback. No secondary article supplies fix versions. Headroom compressed long output; relevant hashes include `0f4c5d3f71c27ee946df964b` and `8d72057aedbc0e2a6f3a5647`.

The baseline scan also inspected 265 other site-packages native-library files: no other OpenSSL marker or ELF `NEEDED` crypto dependency was identified. DuckDB core source uses Mbed TLS; this does not cover its separately distributed extensions. Only the newly installed Spatial 1.5.6 extension was inspected, not other extension caches, containers, platforms or marker-free copies. This is bounded inventory evidence, not a complete SBOM or security certification.

Temporary detailed inventories and matched advisory ranges remain under `/tmp/biominer-dependencies-20261001-k3kxy0ga/openssl`; no wheels, interpreter, raw payloads or native binaries are committed. No host package update, shared environment sync, live attacker-input check or production deployment was performed. The main task installed/loaded the signed Spatial extension from `https://extensions.duckdb.org` to satisfy an existing prerequisite; the subsequent independent native inspection was read-only. Signatures were verified by the normal loader, with no unsigned override.

Source was read for evidence only. OpenSSL/Arrow are Apache-2.0, Psycopg LGPL-3.0-only and uv MIT OR Apache-2.0; the python-build-standalone licence was not independently inspected. No third-party implementation was copied into BioMiner.

## Published version-range matrix

The matrix below records the complete union of matches for the inspected original copies and three native candidates against the [current OpenSSL advisory catalog](https://openssl-library.org/news/vulnerabilities/index.html). `yes` means the observed release identifier falls within a published range. `—` means it falls outside that reviewed range, not that the application has received a full security certification. Original Arrow and the new Spatial extension share the 3.6.0 identifier. The private 3.5.9 interpreter falls outside all rows. Ubuntu is assessed separately through its vendor revisions/backports and is deliberately excluded from an upstream-only matrix.

| CVE | Upstream severity | Original Psycopg 3.5.4 | Original Python 3.5.6 | Original Arrow / new Spatial 3.6.0 | New Psycopg 3.5.8 | New Arrow 3.6.2 | Fixed 3.6 release |
|---|---|---|---|---|---|---|---|
| CVE-2026-35189 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-35191 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-42772 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-54872 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-54873 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-54875 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-72897 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-75804 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-75805 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-75806 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-77696 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-84782 | High | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-84784 | Low | yes | yes | yes | yes | yes | 3.6.5 |
| CVE-2026-75803 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-14457 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-18798 | Moderate | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-54874 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-63072 | Moderate | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-63073 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-63074 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-63075 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-63076 | Moderate | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-14456 | Low | yes | yes | yes | — | yes | 3.6.4 |
| CVE-2026-54876 | Low | — | — | yes | — | yes | 3.6.4 |
| CVE-2026-34180 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-34181 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-34182 | Moderate | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-34183 | Moderate | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-35188 | Moderate | — | — | yes | — | yes | 3.6.3 |
| CVE-2026-42764 | Moderate | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-42765 | Low | — | — | yes | — | yes | 3.6.3 |
| CVE-2026-42766 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-42767 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-42768 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-42769 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-42770 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-45445 | Moderate | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-45446 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-45447 | High | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-7383 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-9076 | Low | yes | yes | yes | — | yes | 3.6.3 |
| CVE-2026-28386 | Low | — | — | yes | — | — | 3.6.2 |
| CVE-2026-28387 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-28388 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-28389 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-28390 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-31789 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-31790 | Moderate | yes | — | yes | — | — | 3.6.2 |
| CVE-2026-2673 | Low | yes | — | yes | — | — | 3.6.2 |
| CVE-2025-11187 | Moderate | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-15467 | High | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-15468 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-15469 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-66199 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-68160 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-69418 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-69419 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-69420 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2025-69421 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2026-22795 | Low | yes | — | yes | — | — | 3.6.1 |
| CVE-2026-22796 | Low | yes | — | yes | — | — | 3.6.1 |
