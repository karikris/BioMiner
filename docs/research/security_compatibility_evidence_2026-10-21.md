# Security and compatibility evidence — 21 October 2026

Task `security-polars-arrow-20261021-1.2`; research baseline `main` at
`5bac7f168c1b257184615baf1524f12d5c580bc6`.
This refresh supports the [approved implementation plan](../superpowers/plans/2026-10-21-security-polars-arrow.md).
It changes no environment, host package, workload, scientific data or model.
The coordinator owns commits, migration, compatibility fixtures and phase gates.

**The October 20 generation is not yet a sufficiently evidenced safe fallback.**
Its lxml wheel embeds libxml2 2.14.6, outside the host's patched library closure.
That finding must be resolved before fallback admission. Package advisory
results alone do not close native exposure.

## Inventories and advisory findings

| Scope | Observed identity | Result |
|---|---|---|
| Shared `.venv` | Python 3.14.5 / OpenSSL 3.5.6; AnyIO 4.14.0; pip 26.1.1; Polars 1.41.2; Arrow 24.0.0 | Older installation remains exposed; no migration claimed. |
| October 20 generation | Python 3.14.7 / OpenSSL 3.5.9; AnyIO 4.15.1; Polars 1.44.2; custom Arrow `25.0.1+biominer.f918dfd84c8c`; custom DuckDB `1.5.6+biominer.f8be3729f88a`; psycopg/c 3.3.6 | Registry checks have no active direct matches; embedded lxml finding below remains blocking. |
| Current application builder | build 1.6.1, packaging 26.3, pyproject-hooks 1.3.3, setuptools 84.2.0, wheel 0.48.0 | No active direct matches. Neither this builder nor the approved runtime contains pip. |
| Proposed data stack | Polars/runtime-32 2.0.0; custom Arrow 26.0.0 | Upstream versions have no active direct matches. Actual custom Arrow wheel has not been built or qualified by this task. |

Pinned GitHits checks identify these shared-environment advisories:

| Component | Advisory | Fixed version / action |
|---|---|---|
| AnyIO 4.14.0 | [GHSA-82r6-8w77-94w6 / CVE-2026-63374](https://github.com/advisories/GHSA-82r6-8w77-94w6), critical TLS hostname handling | 4.14.2; planned baseline already selects 4.15.1. |
| AnyIO 4.14.0 | [GHSA-3w57-8xmc-8v26 / CVE-2026-63349](https://github.com/advisories/GHSA-3w57-8xmc-8v26), supplementary subprocess groups | 4.14.2. |
| AnyIO 4.14.0 | [GHSA-5p39-cfhj-2xmp / CVE-2026-64847](https://github.com/advisories/GHSA-5p39-cfhj-2xmp), process-pool stderr blocking | 4.14.2. |
| pip 26.1.1 | [GHSA-wf93-45jw-7689 / CVE-2026-8643](https://github.com/advisories/GHSA-wf93-45jw-7689), entrypoint path traversal | 26.1.2; remove the obsolete entrypoint or use reviewed 26.2.1. |
| pip 26.1.1 | [GHSA-qwm4-qh6w-59xr / CVE-2026-13346](https://github.com/advisories/GHSA-qwm4-qh6w-59xr), double-decoded index URL | 26.2; 26.2.1 lookup has no active matches. |

These are version-range matches, not demonstrated exploitation of BioMiner.
AnyIO's hostname fix is source-bound to
[`68f58915f82d9be8109ebbbd8f5d70577d43f2ce`](https://github.com/agronholm/anyio/commit/68f58915f82d9be8109ebbbd8f5d70577d43f2ce).
pip's URL fix is
[`10dfb6b9005484578b386f64b9f36982e3dc6679`](https://github.com/pypa/pip/commit/10dfb6b9005484578b386f64b9f36982e3dc6679);
its [release history](https://github.com/pypa/pip/blob/634a6ec1/NEWS.rst)
also confirms entrypoint validation.
If pip is retained, account for 26.2's separate build constraints and stricter
downloaded metadata validation.

GitHits labels the pip entrypoint advisory high/8, while the current GitHub
Reviewed record labels it moderate. The URL advisory scores also differ
(GitHits 6; GitHub 5.6). Affected/fixed ranges agree. AnyIO publication dates
and numeric scores differ between the aggregation and primary records too.
Use the linked primary records for those details; this report does not resolve
the discrepancies by silently selecting the more severe value.

## Native libraries: host fixes do not patch embedded copies

`dpkg-query` and local `apt-cache policy` identify official Ubuntu origins:

| Host component | Installed revision | Current primary fix evidence |
|---|---|---|
| OpenSSL / libssl3t64 / libssl-dev | `3.5.5-1ubuntu3.7` | [USN-8861-1](https://ubuntu.com/security/notices/USN-8861-1), QUIC CPU/memory fixes CVE-2026-42772 and CVE-2026-54873. |
| libcurl4t64 | `8.18.0-1ubuntu2.8` | [USN-8927-1](https://ubuntu.com/security/notices/USN-8927-1), including proxy credential, HSTS, HTTP/2, MIME and protocol handling fixes. |
| libxml2-16 | `2.15.2+dfsg-0.1ubuntu0.3` | [USN-8910-1](https://ubuntu.com/security/notices/USN-8910-1), CVE-2026-76781, 86138, 86139, 86142, 86143 and 86144. |
| libpq5 | `18.6-0ubuntu0.26.04.1` | Installed version/origin recorded; no complete PostgreSQL advisory audit is claimed here. |

These revisions match the reviewed vendor notices. No apt refresh, package
update or reboot was performed. Already mapped libraries require separate
process evidence and retirement; developer MCP processes are a separate
inventory scope, not automatically BioMiner production.

The complete retrieved [OpenSSL 3.5 catalog](https://openssl-library.org/news/vulnerabilities-3.5/)
and [3.6 catalog](https://openssl-library.org/news/vulnerabilities-3.6/)
contain 61 distinct entries each. Numeric affected-range evaluation matches
38 entries for observed shared-interpreter 3.5.6 and zero for 3.5.9 or 3.6.5.
The custom Arrow/DuckDB receipts specify 3.6.5; that is receipt evidence,
not a new inspection of every loaded native copy. Ubuntu backports must be
evaluated by vendor revision, not by the upstream `3.5.5` substring.
The [September 29 advisory](https://openssl-library.org/news/secadv/20260929.txt)
and [3.6.5 source history](https://github.com/openssl/openssl/blob/c8bd5a57/CHANGES.md)
describe the DTLS stale-buffer and QUIC fixes. For example, the 3.6.5 QUIC
reassembly fix is
[`32d0ed8afe1b8c3e7ece725b44663da3d7087a09`](https://github.com/openssl/openssl/commit/32d0ed8afe1b8c3e7ece725b44663da3d7087a09).

**Blocking embedded-library gap:** fresh imports of lxml 6.1.1 and 6.1.3
both report compiled/runtime libxml2 2.14.6 and libxslt 1.1.43. Their etree ELF
dependencies contain no dynamic libxml2/libxslt, so updating the host libraries
does not replace those copies. The approved etree binary SHA-256 is
`0527412673571e96528365835cc0052b1b4887070e55425226251d9e479eef88`.
[lxml's release source](https://github.com/lxml/lxml/blob/3c1a4c7f/CHANGES.txt)
documents the 2.14.6 Linux/macOS lineage, the separate 6.1.3 parameter-entity
fix and explicit libxslt 1.1.43 security backports. Do not infer that the
libxslt fixes are missing from its version string.

For libxml2, the published pre-2.15.4 range includes 2.14.6.
The [upstream xmlDictAddQString fix](https://github.com/GNOME/libxml2/commit/a4cba4b5b5a8c42e155ed42d2d2a44955465a2e4)
adds allocation arithmetic guards; USN-8910-1 associates this defect with
CVE-2026-86138. No byte-bound libxml2 backport attestation was found for these
wheels. This is an unresolved applicable native finding, not proof of an
exploit through the application's bounded parser paths. Task 2.1 must obtain
such an attestation or rebuild/attest lxml against reviewed patched libraries,
then test XML/PPTX behavior and native admission. Do not declare the existing
generation safe merely because PyPI lxml has no active advisory.

## Compatibility decisions bound to local calls

Polars 2.0.0 resolves to
[`22a147de3d2bb2e44b97338a2510816c7105c9f2`](https://github.com/pola-rs/polars/tree/22a147de3d2bb2e44b97338a2510816c7105c9f2).
Arrow 26.0.0 resolves to
[`446167169a0dc547e00e1bc29f417a40c0ca267a`](https://github.com/apache/arrow/tree/446167169a0dc547e00e1bc29f417a40c0ca267a).
Exact GitHub ref responses independently corroborate GitHits revisions.

| Change | Local consequence and required verification |
|---|---|
| Polars `explode` defaults `empty_as_null=False`, `keep_nulls=True` | `evaluation/sampling.py:_candidate_query_provenance` must retain empty/null query rows so its unresolved-definition check rejects them. `reports/flickr_fetch.py:_query_provenance_summary` must preserve label counts. Both require explicit `True, True` and paired null/empty/duplicate fixtures. [Source](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/py-polars/src/polars/dataframe/frame.py). |
| Polars `is_in` coercion and order-sensitive operations | Many status/taxon filters use string membership; `flickr_fetch/yoloe_sharded.py` also passes a Series. Compare coercion/null cases and stable provenance ordering rather than assuming unchanged semantics. |
| Polars automatic spill defaults | Upstream uses 80% memory fraction and 64 GB disk defaults. Actual controls are `POLARS_OOC_MEMORY_BUDGET_MB`, `POLARS_OOC_DISK_BUDGET_MB` and `POLARS_OOC_SPILL_DIR`. Implement the plan's bounded per-run policy and failure/cleanup tests; these controls do not guarantee process RSS. [Source](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/crates/polars-config/src/lib.rs). |
| Arrow Parquet nesting limit | Numerous `pq.ParquetFile`/`pq.read_table` paths in `gbif_final`, `gbif_quality` and media resolution receive the default limit of 100. Keep it enabled; test malformed/over-depth input and no partial publication. [C++ property](https://github.com/apache/arrow/blob/446167169a0dc547e00e1bc29f417a40c0ca267a/cpp/src/parquet/properties.h), [Python propagation](https://github.com/apache/arrow/blob/446167169a0dc547e00e1bc29f417a40c0ca267a/python/pyarrow/parquet/core.py). |
| Arrow null-fill buffer fixes; stricter deserialize/readinto/resize handling | No direct tracked BioMiner calls to Arrow `fill_null_forward/backward`, `FunctionOptions.deserialize` or these Arrow buffer operations were found. Preserve regression probes because the enabled Compute capability remains available. These are upstream hardening changes, not newly assigned critical advisories. [Implementation](https://github.com/apache/arrow/blob/446167169a0dc547e00e1bc29f417a40c0ca267a/cpp/src/arrow/compute/kernels/vector_replace.cc). |

The release review additionally flags SQL numeric/window behavior, Parquet ENUM
mapping, ragged rows and String-to-Time casts. Focused tracked-source searches
found no direct use of Polars SQLContext, Enum, cut/qcut or entropy. This is
bounded call-site evidence, not a universal absence claim. Existing joins,
nullable/list schemas, ingestion and physical/logical artifact comparisons
remain mandatory in the four-environment matrix.

## Native and build requirements

Arrow 26 enables simdjson for Parquet/JSON, requires at least 4.3.0, and bundles
4.6.4 with SHA-256
`b091107844fe928158c5c2265c20360fff312889ddf7ebc4528a0f0f8f2ff9cd`.
Its [build logic](https://github.com/apache/arrow/blob/446167169a0dc547e00e1bc29f417a40c0ca267a/cpp/cmake_modules/ThirdpartyToolchain.cmake)
adds the vendored target to static libraries; the exact pin is in
[versions.txt](https://github.com/apache/arrow/blob/446167169a0dc547e00e1bc29f417a40c0ca267a/cpp/thirdparty/versions.txt).

Review a 4.6.11 pin for the new build: its
[official release](https://github.com/simdjson/simdjson/releases/tag/v4.6.11)
documents six memory/streaming corrections reproduced on 4.6.10.
The released 4.x backport is
[`e21cbc451510df5128b1d4f1b8d5f1e1f06f65a3`](https://github.com/simdjson/simdjson/commit/e21cbc451510df5128b1d4f1b8d5f1e1f06f65a3);
4.6.11 is `f5de14f09256982933af2849beb43778bd421ca7`.
The broader PR 2852 was closed without merging; its author's 17-finding
severity claims are not adopted. This is a source-hardening recommendation,
not a CVE finding or proof that all affected APIs are exercised by Arrow.
Pin/hash and test any replacement; do not silently move to 5.x.

The existing Arrow recipe forces private static OpenSSL and replaces the
AWS-LC crypto target with OpenSSL. Reapply only after exact source review;
retain S3, Flight, encryption and all enabled capabilities, with Gandiva
disabled. Stock Arrow wheels cannot inherit the custom receipt.

Preserved native-build metadata adds 22 distinct package/version checks beyond
the runtime/lock/builder inventory. No active matches were returned. These
include historical setuptools 83/84, NumPy 2.5.3, Cython 3.3.0 and
scikit-build-core 1.1.0. Historical build receipts are not live environments:
old Arrow builder interpreter links are no longer executable.
The new builder must use current reviewed inputs. Setuptools 84.1+ documents
archive containment and legacy egg namespace fixes despite an advisory
coverage gap; maintain the 84.2 floor. [Source](https://github.com/pypa/setuptools/blob/301e27a484f17e175acf6b692fccc634f9a20c49/NEWS.rst).
No complete advisory clearance of every bundled C/C++ dependency is claimed.

## Receipts, verification and limits

Ignored evidence root: `.native-runtime/upgrades/20261021/evidence/`.
`summary.json` records reproducible counts, findings and source/artifact hashes.
Retained GitHits coverage is **71 unique package metadata responses** and
**112 unique pinned advisory responses**. The latter cover all 47 locked
third-party versions, 46 shared-environment third-party versions, 41 approved
runtime versions (custom local suffixes mapped to their upstream bases),
current/historical builders and proposed versions; overlaps are deduplicated.
Five accepted upgrade-review batches contain **30, 2, 3, 2 and 1** comparisons.
One initial oversized 32-row request was rejected and split; it contributes
no evidence. All direct affected matches are the five AnyIO/pip records above.

| Retained receipt | SHA-256 |
|---|---|
| `githits-package-evidence.json` | `75fc39055a421dec764c8dc89ba4088b394613ab3cc485c721e8730519bb7af3` |
| `source-evidence.json` | `daecf89785e1c858f33a03fd21f3befff140f2830b808fe66503265884dc9829` |
| `supplemental-evidence.json` | `a34e53718b71a6fbbf7bb86766253b4ec770e29b613c714884f0032f7c4d78c2` |
| `lxml-native-probe.json` | `846887251194bb955199c8df1d6d393f82441417be10cb23b4f51f3ad45a2b8e` |

Valyu supplied official advisory/release pages. Direct primary fetches retained
complete OpenSSL catalogs and exact release refs; Ubuntu direct HTTP fetches
timed out, with Valyu notice text retained instead. Repository-local pip
advisory URLs returned 404; the global GitHub Reviewed records and upstream
pip source supplied the evidence. Some source results were marked older or
provisional; one explicitly offered libxml2 `search_status` continuation was
used. A GitLab shorthand resolved to the wrong host and failed; the official
GNOME GitHub mirror was then discovered through primary-source search.
Native advisories are not inferred from unrelated Go/Java/PHP package results.

This task ran inventory/import/ELF probes and validated receipt JSON and hashes.
Compatibility test results belong to the coordinator's separate baseline
receipt and later matrix. No live service, exploit, benchmark, native rebuild,
workload migration or production promotion is claimed here. No third-party
implementation was copied: source was inspected as evidence. Polars is MIT,
lxml BSD, and Arrow/OpenSSL Apache-2.0; this review introduces no dependency
or upstream-code licence obligations.
