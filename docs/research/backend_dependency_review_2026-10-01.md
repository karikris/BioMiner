# Backend dependency review — 1 October 2026

BioMiner's backend lock now contains **31 version upgrades and one added transitive package**. Direct requirements have reviewed lower bounds, explicit AnyIO/lxml security floors, and a skops 0.16 compatibility range. Python is pinned to 3.14.7; the tested Astral 20260929 build contains OpenSSL 3.5.9. Native libraries still have separate OpenSSL findings described in [the native-runtime review](openssl_security_review_2026-10-01.md). [TaxaLens and ButterflyLens findings](frontend_tooling_security_review_2026-10-01.md) are report only.

Task: `backend-dependencies-20261001`. Starting branch/SHA: `main` / `0ba1784f673888ceff8da81c4eb3b6652011dbc4`. One coherent dependency/evidence commit is required on main, followed by a normal push; its exact local and remote SHA is reported in the completion response. No merge or history rewrite is authorized by this maintenance task.

## Review method and ownership

GitHits supplied latest `pkg_info`, three backend upgrade batches covering all 31 upgrades, a separate build-backend review, pinned advisory checks, release histories, and upstream source. `search_status` was used only for explicit search continuations. Source `code_grep`/`read` resolved questions that release bodies could not answer. The review distinguishes current-version advisories from package-wide advisory history and registry-resolved dependency ranges from the exact installed/locked graph.

Initial GBIF EDA/report work, `AGENTS.md`, scripts and their tests, and `veritaxa_codex_goal.md` belonged to another session and remain untouched. The instruction to first commit dirty main conflicts with the active-work prohibition on staging another session's changes; this task stages only its own paths. The historical dynamic-pooling GitHits opt-out does not apply to this explicitly authorized monthly GitHits review. No old scientific run was resumed, no model weights or media were fetched, and no handoff pins or production artifacts were changed.

Verification uses `/tmp/biominer-dependencies-20261001-k3kxy0ga/venv`. The shared `.venv`, global Python, and Ubuntu libraries were not updated. A matching signed DuckDB Spatial extension was installed into a previously absent `v1.5.6/linux_amd64` cache; the previous version cache was preserved. No raw tool output, wheels, caches, interpreter, database, or test artifacts are committed.

## Security fixes affecting the baseline

| Baseline component | Finding and reviewed fix | Application relevance |
|---|---|---|
| AnyIO 4.14.0 | Critical [GHSA-82r6-8w77-94w6](https://github.com/advisories/GHSA-82r6-8w77-94w6), CVE-2026-63374: IDNA 2003 hostname mapping can validate the wrong TLS certificate. Fixed in 4.14.2; selected 4.15.1. | The public TLS wrapper is affected. HTTPX 0.28.1 already encodes IDNA hostnames before passing `raw_host` to HTTPCore, so this is not evidence of an exploitable normal BioMiner HTTPX request. The explicit floor closes the underlying dependency defect. |
| AnyIO 4.14.0 | High [GHSA-3w57-8xmc-8v26](https://github.com/advisories/GHSA-3w57-8xmc-8v26), CVE-2026-63349: `extra_groups` was ignored. Medium [GHSA-5p39-cfhj-2xmp](https://github.com/advisories/GHSA-5p39-cfhj-2xmp), CVE-2026-64847: process-worker stderr deadlock. Both fixed in 4.14.2. | No direct BioMiner use of these AnyIO subprocess/process-pool APIs was found. Installed-version applicability is retained without claiming a demonstrated privilege or denial-of-service exploit. |
| lxml 6.1.1 | Default/internal-only XML parsing still loaded external parameter entities. [6.1.3 source changelog](https://github.com/lxml/lxml/blob/lxml-6.1.3/CHANGES.txt) documents the correction. Selected 6.1.3 and added a direct floor. | The baseline test actually reads an external fixture DTD; the upgraded parser rejects it. Current pinned GitHits/PyPI advisory checks did not enumerate this issue, so it is reported as an upstream security fix rather than an invented CVE. |
| skops 0.14.0 | [0.15 source history](https://github.com/skops-dev/skops/blob/bcc69da9/docs/changes.rst) hardens protocol validation, tree-state trust, random-generator auditing, and the CLI updater's implicit trust. Selected 0.16.0. | The optional dependency is installed and exposed to these defects, but BioMiner currently uses numeric NPZ/JSON classifier persistence rather than skops loading. No production skops artifact migration is claimed. |
| DuckDB 1.5.4 | [1.5.5 release](https://github.com/duckdb/duckdb/releases/tag/v1.5.5) backports out-of-bounds security hardening from [PR 23100](https://github.com/duckdb/duckdb/pull/23100). Selected 1.5.6. | New checks cover malformed Parquet RLE/VARIANT and ALP/ALPRD/Patas/native RLE decoding. This is relevant to untrusted data readers; advisory tooling reported zero direct findings and did not substitute for release/source inspection. |
| Supported setuptools floor 69 | [GHSA-cx63-2mw6-8hw5](https://github.com/advisories/GHSA-cx63-2mw6-8hw5) command injection; [GHSA-5rjg-fvgr-3xxf](https://github.com/advisories/GHSA-5rjg-fvgr-3xxf) traversal/write; [GHSA-h35f-9h28-mq5c](https://github.com/advisories/GHSA-h35f-9h28-mq5c) Unicode manifest exclusion bypass. Reviewed 69.0.0 to 84.0.0; fixed respectively from 70.0.0, 78.1.1, 83.0.0. | This describes versions permitted by the old build requirement, not an observed installed setuptools 69 runtime. The build lower bound is now 84, and source/wheel builds were exercised. |

The baseline exact requirements audit found three advisories in AnyIO; the upgraded Linux/all-extras requirements audit found **zero advisory records in 45 packages**. This result does not cover embedded OpenSSL, downloaded DuckDB extensions, OS libraries, other platforms, or undisclosed/unindexed issues.

## Selected upgrades and what they contain

| Package | Locked baseline → selected | Release/source assessment |
|---|---|---|
| annotated-types | 0.7.0 → 0.8.0 | `DocInfo` renamed to `Doc`; typing-extensions dependency removed. Pydantic/configuration tests cover the consumer path. |
| anyio | 4.14.0 → 4.15.1 | Three security fixes above; 4.15 adds lazy submodule loading, Python 3.14 threading fixes and concurrency utilities; 4.15.1 restores direct submodule access compatibility. |
| certifi | 2026.6.17 → 2026.7.22 | Updated CA bundle; no current direct advisory. Detailed root-by-root release notes were unavailable; live provider TLS was not exercised. |
| contourpy | 1.3.3 → 1.4.0 | Adds contour algorithm-name property and Python 3.15 wheels; NumPy minimum becomes 2.0, satisfied by this graph. |
| duckdb | 1.5.4 → 1.5.6 | Reader security hardening, malformed Parquet page rejection and memory-manager deadlock fix in 1.5.5; 1.5.6 fixes active-query races, optimizer NULL/pushdown behavior and ICU timezone state, and stabilizes C API versioning. |
| fonttools | 4.63.0 → 4.66.1 | Font parsing/subsetting and DesignSpace name fixes; Python minimum becomes 3.11. Report generation remains compatible in tests. |
| idna | 3.18 → 3.20 | Unicode 18 tables, stricter canonical Punycode and domain lengths; 3.19 restores STD3 rules and ignores the deprecated transitional argument. |
| joblib | 1.5.3 → 1.6.0 | Fixes cross-process cache identity/writes, argument handling and unordered-generator concurrency; unvendors cloudpickle. Python minimum becomes 3.10. |
| kiwisolver | 1.5.0 → 1.5.1 | Packaging/build maintenance; no direct dependency/advisory changes in the review. Rendering tests exercise the consumer. |
| lxml | 6.1.1 → 6.1.3 | External parameter-entity fix; sdist/build and error handling corrections in 6.1.2. Tested Linux libxml2 reports 2.14.6. |
| matplotlib | 3.11.0 → 3.11.2 | Layout/image/font/PDF fixes, GIF frame skew correction and wheel/system-library conflict prevention; fonttools minimum increases to 4.28.2. |
| narwhals | 2.24.0 → 2.26.0 | Dataframe compatibility maintenance, including deprecated category-access handling for newer Polars. |
| numpy | 2.4.6 → 2.5.3 | Removes distutils and expires older deprecations; fixes datetime Cython, masking and StringDType handling. Invalid UTF-8 byte-to-StringDType conversion now raises `TypeError`. |
| packaging | 26.2 → 26.3 | Requires Python 3.9+; no graph or advisory changes in review. |
| polars | 1.41.2 → 1.44.2 | Streaming/join/slice/Arrow interoperability and SQL NULL fixes; adds API deprecations and out-of-core work. Existing streaming/order parameters remain available in inspected source. |
| polars-runtime-32 | 1.41.2 → 1.44.2 | Exact matching native runtime required by Polars; same release line. |
| psycopg / psycopg-binary | 3.3.4 → 3.3.6 | Keep Python adapter and binary build aligned. Source/package build and actual wheels establish updated libpq/OpenSSL; tested libpq reports 18.6. OpenSSL remains separately affected. |
| pyarrow | 24.0.0 → 25.0.1 | 25 changes single-file partition inference and introduces Feather deprecation. 25.0.1 fixes aarch64 SVE double decoding and a first-load/non-main-thread mimalloc crash; it narrows Feather deprecation to V1. |
| pydantic | 2.13.4 → 2.13.5 | Validator reuse, GC traversal and smart-union field-count corrections. |
| pydantic-core | 2.46.4 → 2.46.5 | Exact version required by Pydantic 2.13.5; do not substitute latest 2.49.0 independently. GC/union fixes are covered by the Pydantic release. |
| pygments | 2.20.0 → 2.21.0 | Lexer improvements including YAML alias/non-breaking-space handling; release keyword heuristics are not treated as proof of a breaking API. |
| pyparsing | 3.3.2 → 3.3.3 | Python 3.15 support; no direct advisory/dependency changes. |
| scikit-learn | 1.9.0 → 1.9.1 | Metadata routing, metrics and estimator fixes. Source history was read after package changelog initially supplied no bodies; linear SVC fit/class ordering was also inspected. |
| scipy | 1.18.0 → 1.18.1 | Bugfix release and Python 3.15 wheels; source-build GCC minimum becomes 10.3.0. |
| skops | 0.14.0 → 0.16.0 | Security changes above; adds timezone-aware datetime support and faster imports. Protocol/type-trust changes require attention for external skops users. |
| threadpoolctl | 3.6.0 → 3.7.0 | No package release-note bodies or dependency/advisory changes returned; scientific classifier tests exercise its installed consumer path. A complete source-change catalogue is not claimed. |
| typing-extensions | 4.15.0 → 4.16.0 | Inheritance/deprecated-class fixes and current typing support; meets AnyIO's new requirement. |
| typing-inspection | 0.4.2 → 0.4.4 | Python minimum rises to 3.10; typing-extensions minimum becomes 4.15.0. One release body was missing. |
| tzdata | 2026.2 → 2026.4 | Timezone-data refresh for Windows Psycopg dependency; not installed or executed on this Linux host. |
| wcwidth | 0.8.2 → 0.9.1 | Native width implementation and fixes for long control sequences, wide graphemes and formatting; terminal/table consumer tests remain compatible. |

Added `cloudpickle 3.1.2` because joblib 1.6 unvendors it. Pinned GitHits advisory lookup returned no active finding. It is a process-serialization dependency, not a new untrusted artifact loader. No package was removed. HTTPX 0.28.1, HTTPCore 1.0.9, H3 4.5.0, Pillow 12.3.0, python-pptx 1.0.2 and pytest 9.1.1 were already current; direct floors were aligned where needed without inventing version upgrades.

## Compatibility and deployment attention

1. **Interpreter builds:** `.python-version` and `Requires-Python >=3.14.7` select the patch line; verify `ssl.OPENSSL_VERSION` on the actual build. The tested current Astral build is `3.14.7+20260929`, OpenSSL 3.5.9. An older cached build or another vendor's build need not contain the same OpenSSL. uv 0.11.19's catalog only knew 3.14.5; the private current uv 0.12.21 supplied the patched build.
2. **DuckDB extensions:** install/load the matching signed Spatial extension for DuckDB 1.5.6 before boundary enrichment. Source confirms version/platform are part of cache lookup. The first full-suite run failed at this existing prerequisite; installing the current extension resolved the focused test. No production auto-install or fallback was introduced. Its own embedded OpenSSL marker is included in the native review.
3. **Polars:** inspected `collect_batches`/`sink_parquet` retain explicit streaming and order controls, but the batch API is marked unstable upstream. New deprecations concern read/scan `rechunk`, category casts/access and temporal casts. Eighteen suite warnings identify `explode`'s future `empty_as_null=False` change in Polars 2.0; `<2` now prevents an unlocked install from silently changing that behavior. Plan a separate semantics-preserving 2.0 migration.
4. **Arrow:** check workflows that depended on implicit single-file Hive partition columns or Feather V1. S3 credential/network contracts were tested using fakes; cloud endpoints, aarch64 SVE and macOS were not exercised. No production Parquet was rewritten.
5. **skops and XML:** future archive protocols now fail closed, native tree states require explicit review, and the updater no longer trusts every declared type. lxml's internal-only parser stops parameter-entity processing; workflows needing external DTDs require an explicit trusted parsing policy. BioMiner's classifier artifacts remain numeric NPZ/JSON; frozen weights, model policies and consumer pins were not changed.
6. **Numerical evidence:** NumPy/scikit-learn upgrades can affect newly fitted numerical results. BioMiner records dependency versions in training identity, and round-trip/margin/calibration tests passed; this does not certify historical model equivalence or authorize resuming a pre-ground-zero run.

Deployment sequence after coordinating any running jobs: use a current uv catalog, select the verified patched Python build, run `uv sync --locked` with only the extras needed by that workflow, verify the selected interpreter's OpenSSL, and provision matching signed DuckDB extensions. For the Spatial setup tested here, the explicit repository command is `INSTALL spatial FROM 'https://extensions.duckdb.org'`. Existing `.venv` contents require migration; editing a lock or interpreter pin does not patch an already running process. Host Ubuntu, Psycopg, Arrow and Spatial native copies require their separate remedies; see the native review.

## Verification receipts

All scientific inputs were fixtures. No live provider/model/PostgreSQL/S3 checks ran. Tests used `POLARS_MAX_THREADS=4 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` and the private interpreter, not the shared environment.

| Check | Observed result |
|---|---|
| Original `.venv/bin/python -m pytest -q tests/test_dependency_security.py` | 3 expected failures: old TLS mapping, external DTD accepted, future skops protocol accepted. |
| Private `python -m pytest -q` on security, HTTP downloader, registry, storage, classifier persistence/training/calibration, DWCA conversion and EDA files | 197 passed in 14.80s, including all three security regressions. |
| Initial full offline suite | 2730 passed, 1 failed, 18 warnings in 60.94s; missing version-matched Spatial extension. Preserved receipt. |
| After Spatial setup: `python -m pytest -q tests/test_gbif_quality_geography.py tests/test_dependency_security.py` | 5 passed in 0.68s. |
| Final full offline suite: private `python -m pytest -q --junitxml=<temporary-root>/pytest-full-verified.xml` | **2731 passed, 0 failures/errors/skips, 18 warnings in 61.80s**. All three new security regressions ran. |
| `uv pip check --python <private-venv>/bin/python` | 46 packages checked; all compatible. |
| Exact upgraded all-extras requirements: `uvx pip-audit --no-deps --disable-pip --progress-spinner off --format json --requirement <upgraded-requirements.txt>` | 45 Linux packages, zero known advisory records; native coverage limitations retained. |
| `uv lock --check` with the patched interpreter | Successful, 48 lock packages including project and Windows-only entries. |
| `uv build --quiet --out-dir <temporary-dist>` | Source distribution and wheel built; package data includes registry language metadata. No build was published. |
| Private `biominer --help` console entry point | Successful. `python -m biominer` is not a supported entry point and was corrected to the declared script. |

Final full-suite log and JUnit receipts are `full-suite-verified.log` and `pytest-full-verified.xml` under `/tmp/biominer-dependencies-20261001-k3kxy0ga`. The suite observes the shared working tree, including the other session's existing EDA changes; those files are excluded from this task's commit. Only this narrative, companion reviews, plan, dependency metadata, behavior tests and a provenance record are committed. Final local/remote commit identifiers and publication status are supplied in the completion response to avoid a self-referential commit hash in this report.

Final dependency/input SHA-256: `pyproject.toml` = `92faa05860f199bd77b48354373330d74748e5d2c307662984ed0442e2a7546d`; `uv.lock` = `b1130337069092433e9496b481afd0f6a6103d13cbb02903ffd0e7d62bcf0bfe`; `.python-version` = `922fe0c3de073b01988e23348ea184456161678c5e329e6f34be89be24383f93`; security tests = `c791d87670e4a3df9ef09fcd87ce83567c391c37598a7a8456cf7e48b5b09fb4`. Built wheel SHA-256 = `4504e33d7b2c549e593ca43e28739f5e1bd7c150f944dec7cb60be056c4545fa`; sdist = `2901bc942febc623707528428cddbe3e0e1ac6a6e313d142eec4bc6ae9d2cba1`. These private build artifacts were not published.

## Source provenance and evidence limits

Representative GitHits questions: `trusted types load security persistence format numpy scikit learn compatibility`; `sink_parquet collect_batches streaming partition maintain_order`; `dataset write_dataset ParquetFile arrow schema compatibility`; `TLSStream wrap hostname IDNA`; `parameter entity resolve_entities internal XMLParser`; `RLE bit_width bounds malformed`; `extension_directory home_directory load extension`; `server_hostname`; `def read_schema`; `Version 1.9.1`; `Release 1.6.0`; `v84.0.0`; and native OpenSSL build/TLS queries in the companion review. Package reviews cover exact from/to versions in the table.

Bounded implementation reads include AnyIO `src/anyio/streams/tls.py:95–180`; HTTPX `_urlparse.py:388–392` and `_transports/default.py:374–394`; skops `io/_persist.py` and `_utils.py:173–213`; lxml `parser.pxi` and `CHANGES.txt`; Polars `lazyframe/frame.py:4333–4410`; Arrow `dataset.py`, `parquet/core.py` and `s3fs.cc`; scikit-learn `svm/_classes.py:290–359` and `doc/whats_new/v1.9.rst`; joblib `CHANGES.rst:4–47`; setuptools `NEWS.rst:1–65`; DuckDB `parquet_rle_bp_decoder.hpp:17–28` and `extension_install.cpp:95–163`. GitHits package and repository locators retain exact versions/refs. Some documentation results used a disclosed fallback snapshot; these were not represented as a complete source diff or latest security catalog.

GitHits package histories initially lacked release bodies for Psycopg, skops, lxml, scikit-learn and several small transitives. Source histories filled the material questions. GitHits could not serve DuckDB core release pages or the current OpenSSL/Ubuntu advisory site inventory; official primary release/security pages were retrieved with Valyu/web as a disclosed fallback. DuckDB's core release evidence and the actual RLE decoder were checked, rather than assuming the Python wrapper notes contained all security fixes. No secondary article defines a fix version.

Headroom compressed long review/source outputs; retrieval hashes include `f0ed012cabae467adeacfedc`, `99881528e8186720e0e993fc` and `0f4c5d3f71c27ee946df964b`. Exact omitted transitive details were retrieved before decisions. Raw research remains temporary.

Reviewed repositories include `duckdb/duckdb`, `duckdb/duckdb-python`, `agronholm/anyio`, `encode/httpx`, `pola-rs/polars`, `apache/arrow`, `pydantic/pydantic`, `numpy/numpy`, `scikit-learn/scikit-learn`, `skops-dev/skops`, `psycopg/psycopg`, `lxml/lxml`, `joblib/joblib`, `pypa/setuptools`, `openssl/openssl`, `astral-sh/uv` and `astral-sh/python-build-standalone`. Licence implications remain those of the dependencies: notably Arrow/OpenSSL Apache-2.0, Psycopg LGPL-3.0-only, certifi MPL-2.0, and the Python/scientific ecosystem's MIT/BSD/PSF licences. No third-party source was copied into runtime code.

Applied skills: ADHD preflight (direct maintenance review), using-superpowers, writing-plans, executing-plans, subagent-driven-development, test-driven-development, systematic-debugging, requesting/receiving-code-review, verification-before-completion and Headroom. The writing-skills skill was inspected but not applied because it concerns skill authoring, not this repository report. Independent final review passed with no substantive findings: the nine staged paths, preserved prior provenance, dependency/build hashes, test/audit receipts, all 61 native matrix rows and deduplicated frontend advisory tables agree with the evidence. Staged `git diff --check` passed, and structured lock inspection verified 575 hashed official-index artifacts and every dependency reference. Exact model/session identifiers and human review were not available and are not invented.

Scientific claims allowed: tested software compatibility and observed package/native inventories only. Biological performance, production occurrence release, live source availability, and a complete native security all-clear remain unclaimed.
