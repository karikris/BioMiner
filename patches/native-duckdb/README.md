# BioMiner DuckDB native build

`python3 scripts/build_native_duckdb.py --python /absolute/venv/bin/python`
fetches checksum-pinned upstream sources, builds the OpenSSL 3.6.5 overlay and
Spatial dependency tree, and builds a custom DuckDB wheel. It defaults to four compiler
jobs within the agreed eight-job total budget and stores all build material under ignored `.native-runtime/duckdb`.
`--prepare-only` verifies and prepares source inputs without compiling a wheel.

The source patch preserves DuckDB's signature and ABI verification, and changes
external loading to hash, verify and load the same sealed Linux memory-file
snapshot. `allow_unsigned_extensions=true` is rejected. A metadata-bypass
setting is also rejected at connection configuration and runtime SET. The initial compiled allowlist is
empty; Spatial is linked into DuckDB using the upstream built-in extension
loader, so `LOAD spatial` does not need an unsigned extension file.

An audited external extension requires a new build using `--policy reviewed.json`.
The policy must contain exactly `extensions`, with entries containing `name`,
`sha256` and `dependencies`. Every dependency contains `soname`, an absolute
canonical `path`, and `sha256`. All transitive ELF dependencies must be declared
and already loaded at their approved paths. RPATH/RUNPATH, ELF audit/filter
mechanisms, unknown dependencies and unsupported platforms fail admission.
This intentionally limits the extension set; it does not promise compatibility
with arbitrary upstream extensions or protect against arbitrary same-user code.

The `_duckdb` shared object exports `biominer_extension_policy_id()` for runtime
attestation. This input/policy digest is separate from the final wheel hash,
avoiding a circular build identity.

A `prepared`, `failed`, or `built_unverified` receipt is not deployment approval.
A wheel requires independent runtime, Spatial semantics, native linkage, TLS,
and admission checks before promotion. The official signed-extension positive
case must be tested in an audited test profile rather than enabling unsigned
extensions in production.

The custom build retains upstream licenses. DuckDB is MIT; OpenSSL is Apache
2.0. Preserve the complete Spatial dependency notices and any source/relinking
requirements when distributing static GEOS/GDAL dependency artifacts.
