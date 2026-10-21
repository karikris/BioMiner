# Phase 4: Arrow 26 build and compatibility

Phase 4 starts on `main` at
`007396255b6afe6acef9cf80d8d7960a1290a1cc`, after the verified phase-3 push.
The external `phase-3-pushed.json` handshake has SHA-256
`2b8d0926bda797e2a30222631390962402095d8721324f798d58e3b7fbe58c3c`.
The existing continuation validator returned `ready` for phase 4; its command
receipt is `phase3/patched25-capacity-gate-prep/boundary-finalization-1/continue.json`,
SHA-256 `6e245740835a0ac928ca55ac76844757179c30654f7a69af2e7527d0f5139fe6`.
External paths are relative to
`/home/toffe/.local/state/biominer/upgrades/20261021/`.

The two-patch Arrow 25 recipe and artifacts remain unchanged as historical
evidence. Current provider research below reopened their transitive zlib
exposure and identified an applicable curl vulnerability; a corrected fallback
is required. Unrelated scientific work remains outside this task. Arrow 26
installation, runtime qualification, phase completion and production promotion
are not established by preparatory research or builder tooling checks.

## Task 4.1: exact source and isolated builder

Arrow 26 is pinned to commit `446167169a0dc547e00e1bc29f417a40c0ca267a`.
Its source includes both allocation and sliced-Boolean fixes; neither Arrow 25
patch is replayed. The separate recipe must preserve all fourteen Arrow
features, six codecs, encryption, S3, Flight and allocators, with Gandiva and
test/benchmark/integration targets disabled. Outer compilation remains bounded
to four jobs and nested CMake/Make work to one.

The reviewed Python builder inputs contain twelve exact wheel pins. Nine are
unchanged from the prior builder; the changes are NumPy 2.5.3 → 2.5.4,
setuptools 84.2.0 → 84.2.1 and vcs-versioning 2.5.0 → 2.6.0. These are isolated
build-tool choices, not changes to application NumPy. GitHits package information,
batched upgrade reviews and primary release/source evidence are retained in
`phase4/python-builder-research-e6f3287/`. Its recommendation SHA-256 is
`578ef75666e918cb3181ffd46672dedc531f4c56a4e8e4938bb1b2764641dd6a`;
the exact hash-required requirements file SHA-256 is
`6b5496c6dd9f446e80b583c28f94a26da4359e914bf7290f248637e1f97542c1`.
Top-level advisory results do not establish security of every vendored component.

Independent source review of the exact setuptools wheel found nested vendored
`dist-info` metadata and its import-time addition of `_vendor` to `sys.path`.
The builder verifier therefore selects the wheel's unique top-level metadata
and enumerates installed distributions through explicit in-environment
purelib/platlib paths. The full file inventory still includes vendored files.
Installation uses local verified wheels, no index or dependency resolution,
required hashes and copy linking, followed by metadata/RECORD, import and
dependency checks. Freezing must never follow interpreter symlinks or change
external hardlinks.

The initial new-recipe tests produced the expected missing-implementation result:
one assertion failure and one fixture setup error in 0.05 seconds. Both identify
the absent `scripts/build_native_arrow26.py`; they do not claim the later source
or build guards have been exercised. The retained supervised receipt is
`phase4/task4.1-red-1/execution.json`, SHA-256
`4ecd226bedd6ca759a6a802f882cce37970c97fb7ea034a1cfdbee74a8628a99`.

The second observed RED run exercises CLI input requirements, intermediate-output
tampering and required crypto/zlib linkage. Five cases failed in 0.11 seconds
because those implementations were absent. Its receipt is
`phase4/task4.1-red-2/execution.json`, SHA-256
`80be48ebc2864b7cc1d15eb817c6c68a5ec82b66a2b136a14be4afcd67b9c6b7`.

The source-reviewed offline builder provisioning completed with supervisor and
runner exit 0 and no interruption. All ten commands passed. Its inventory
contains 2,330 files, 318 directories and four symlinks, with no writable files
or directories. Before/after-freeze probes agree on the twelve distributions,
Python 3.14.7, OpenSSL 3.5.9 and successful imports. The receipt is
`phase4/builder-1/builder.json`, SHA-256
`303c3f5561d8e93e9e60d5bef476c3da8ebfedaa33c379433bda26e5352da561`;
the captured execution receipt is `phase4/builder-execution-1/execution.json`,
SHA-256 `2f6cd5a03ad4cd69613759379eb58c6e7b83f939dbbb03c4ade78bf8614fc56b`.
Independent actual SPEC/QUALITY review passes with no findings. The reviewer
rehashed the complete inventory and directly compared 2,245 wheel payload
members and 2,315 installed RECORD rows. Its receipt is
`phase4/builder-1/independent-actual-qualification-review.json`, SHA-256
`4b553fbd4e3c7477799adc29d85ec6fbf2eb5fde163e202f4b471d0cc588a307`.
This qualifies only the isolated Python build tooling.

The separate recipe is now implemented in `scripts/build_native_arrow26.py`,
with SHA-256 `373110382bb94eeb29877b4ace7743c5283ffb282aa5dc1c081d2cb6e4fe3da3`.
It checks the complete source archive, four integrated fix files, exact
configuration transformations and frozen builder before building. It forces
simdjson 4.6.11, uriparser 1.0.2, Boost 1.92.0 and zlib 1.3.2 with the exact
`df84af25` patch. Private OpenSSL 3.6.5 archives, source receipt and header are
pinned; its include tree is captured and rechecked. The old Arrow 25 recipe and
both patches remain byte-for-byte unchanged.

Configuration stops for review of actual dependency targets and source/provider
identities. Compilation requires explicit hashes for both that configuration
and the complete intermediate-output checkpoint. Review found and corrected two
resume problems: expected new build outputs were incorrectly compared with
configuration-time output inventory, and a rejected configuration hash could
replace the prior receipt. New tests first failed for those exact reasons and
now pass. Preexisting source/selection identities remain strict, and rejecting an
approval hash preserves the prior receipt and entire checkpoint tree. Failed
Python builds that modify generated source require a fresh generation; the
failed evidence is retained.

Root-executed task checks, through the committed supervisor:

| Check | Actual result | External receipt SHA-256 |
|---|---|---|
| `pytest -q tests/test_build_native_arrow26.py` | 55 passed in 0.52 s | `b66b830b747ab8df0b5806bbbfdbbd7cc9b5c98d1039499af2fce90ed3842e87` |
| `pytest -q tests/test_build_native_arrow.py` | 47 passed in 0.42 s | `998fb4001180bfde1aa4183c82a74d571a9a6366c2273ce9ef72a24eb2e93c66` |
| Recipe `--check` with the actual frozen builder, five source archives and private OpenSSL | Exit 0; no build generation created | `75a69c91b4998d2ef9ee1cd1ddf903661338700bf3758522e56a81a00d3fe189` |

These are respectively `phase4/task4.1-focused-3/execution.json`,
`phase4/task4.1-neighbor-1/execution.json` and
`phase4/task4.1-input-check-1/execution.json`. Each records supervisor/runner
exit 0 and no interruption; exact argv, environment, source hashes and logs are
retained. Final recipe/test hashes were unchanged after the checks, and
`git diff --check` passed. No additional repository lint/type command is
configured in `pyproject.toml`.

Actual Arrow configuration, compilation, dependency/advisory review, runtime
admission and phase gates remain pending. The new raw build receipt is explicitly
`compiled_unverified`; task 4.3 must implement its distinct admission policy.

Task 4.1 was committed on `main` as
`3fcd4e5fad78a950ef20acbba398417389f6110e`. Independent SPEC approval is
`phase4/task4.1-independent-spec-review-1/spec-review.json`, SHA-256
`1a984a68e7506ff3be0fdc4fde9d1bd2e085f3c6686b75f7b91c996678c1f13e`;
QUALITY approval is `phase4/task4.1-independent-quality-review-1/quality-review.json`,
SHA-256 `76b3c2f1315492f6a0cc0173998e94d54b48cbc3e9786b782744aa4cae7208d3`.
The additional create-only QUALITY v2 receipt binds the final staged verification:
SHA-256 `2adc9a6acc97d345bcdd45967738c52999cc7550a53bca7df6f47c0e0bee800d`.
The phase push remains pending the complete phase-4 gates.


### Configuration correction: use CMake's native target model

The first actual configure attempt failed during generation. The new inventory
hook evaluated `LINK_ONLY` outside link context and mapped distinct configured
targets to the same output filename. Independent inspection also found that
root-scope target lookups could misidentify child-local imported targets. The
failed generation is preserved at
`.native-runtime/arrow26-security-20261021-1/build`. Its outer receipt is
`phase4/configure-outer-1/execution.json`, SHA-256
`0c61ca8fcae1dc157e5156274a73e027f0efc4397ad8dd7e5adca84826480be4`;
inner receipt `phase4/configure-execution-1/execution.json` has SHA-256
`d2543ee8bd33ab090693347c8aa59dc00af5bd5bbfa995aa598dd28235bcb2fa`.
Both exit codes are 1, interruption is null, cleanup succeeded, and the sampled
compiler maximum was one during configuration probes. No Arrow target build ran.

The correction removes that custom generator hook and uses the pinned CMake
4.2.3 File API codemodel 2.9. It retains built and abstract/imported targets with
native scoped IDs, resolved artifact paths and evaluated link fragments. The
successful index, model and every consumed target reply are hash-bound. Missing,
escaping or mismatched references and older models fail closed. The empty query
is part of the immutable precompile inputs. System native inputs retain provider
and source-package identities; bare `-l` flags still require final link-map
resolution. Existing source, crypto, compiler, feature and resume guards remain.

A real bounded CMake fixture confirms three `Repeated` imported targets retain
three distinct root/child IDs and archive paths, and both `LINK_ONLY` consumers
resolve their own child archive. Its parser verification is
`phase4/cmake-file-api-scope-2/parser-verification.json`, SHA-256
`faf29ac8262b0b81d7f078d89a30ca78ff22749df91283e9b56bafea5838d3a2`.
The initial fixture ordering was invalid CMake; that failed fixture and both
minimal reproductions of the original hook defects are preserved. A proposed
raw-metadata/sidecar repair was rejected before implementation because it did
not resolve directory scope. Those RED results remain historical evidence.

The new File API tests first failed in all five cases against the old collector,
then the complete recipe suite passed 60 tests in 0.90 seconds. The supervised
GREEN receipt is `phase4/task4.1-focused-4/execution.json`, SHA-256
`729a000c9cea32fce9712c221e55da864806b6ea161dcaef0ce945ab541a267a`;
supervisor/runner exit 0, interruption null. Current corrected recipe SHA-256 is
`d375b96ba64dd5380a4b57469670321bbb6183d0f25ce58658a4782d941e2fb1`.
The exact upstream CMake source/docs and GitHits search/read/grep/status results
are retained in `phase4/cmake-inventory-research-1/githits.json`. The next actual
configure uses a fresh generation; the failed receipt and source stay intact.


### Configuration correction: retain the standalone Python provider

The second run completed CMake configuration/generation, then correctly stopped
because its newly complete inventory encountered `libpython3.14.so` outside the
OS package database. This library belongs to the already qualified standalone
Python runtime. The failed generation-2 receipt is preserved with SHA-256
`eee74d2e4a935dca5c8974c61792c52ec0ef0b571530a6474a6c2d28b89a9b10`.
Outer `phase4/configure-outer-2/execution.json` has SHA-256
`6ea5b2866c71eb0d3d3073a31018556181025cdc76b5a15be25af2ae50c72c2a`;
inner `phase4/configure-execution-2/execution.json` has SHA-256
`dc07a95f556c511a1c471c8cdb2b10d587b04dec3789107249fc031ca1e769c1`.
Both exits are 1 with no interruption; cleanup succeeded and maximum sampled
compiler concurrency was one. Arrow target compilation remains unrun.

The correction binds the exact existing Python source receipt
`43a7e96c319752ec519a3c7e213cbb063bb8d140bbb598e69ceb6b49b2c787f4`
and archive `da5357f47b1d9c8d4c439004d463d893ef688a44c27438a62b47073d26d1377e`.
It rechecks all 3,518 original payload files before and after building, retaining
293 native/header file descriptors in the build manifest and final raw receipt.
Selected files require exact attested membership and bytes. Existing
`python-runtime:Astral20260929` ownership applies to the ELF closure; all other
external inputs still require a known private OpenSSL archive or an OS provider.
This adds no arbitrary unowned-library fallback.

The provider regression first failed on the missing implementation, then all
61 recipe tests passed in 0.67 seconds. The GREEN receipt is
`phase4/task4.1-focused-5/execution.json`, SHA-256
`1b0522e672dc4f08a664a419922cb29ed934dc5d2ea7b61850af0d3064e7de3f`,
with supervisor/runner exit 0 and no interruption. A read-only actual-runtime
preflight revalidated the full payload and source/provider membership:
`phase4/private-python-provider-check-1.json`, SHA-256
`b648fd750d0061afba87a74c246a36637e6fb04fc054d9fa1bf4709184d82822`.
Current recipe SHA-256 is
`1629657fa2ec79fac6045a314d675371f95e756f6bb3da71729df1f440ab2182`.

The retained configured-provider audit contains 468 targets and 15 existing
external native inputs: `phase4/configured-native-provider-audit-2.json`, SHA-256
`9a5243017bc8f23490f1493a72b5d33a2788f7209a8129a4907a8bcd1c9cc2da`.
It establishes provider identity for a configured superset. Final linkage,
current source/advisory applicability and a fresh successful configuration remain
required before runtime admission. The next run uses a new generation 3.

### Security correction: actual transitive zlib and curl exposure

The configured-provider audit led to inspection of the actual Arrow 25 receipt
at `.native-runtime/arrow25-security-slices-20261021-1/build/build-receipt.json`.
It records system `libz.so.1.3.1`, `libcurl.so.4.8.0` and
`libxml2.so.16.1.2`, with the same bytes selected during Arrow 26 configuration.
Both system network/XML libraries directly require `libz.so.1`.

Ubuntu marks the installed zlib source package
`1:1.3.dfsg+really1.3.1-1ubuntu3.1` vulnerable to
[CVE-2026-85091](https://ubuntu.com/security/CVE-2026-85091). The installed curl
source package `8.18.0-1ubuntu2.8` remains vulnerable to Negotiate ambient-user
connection reuse,
[CVE-2026-19931](https://ubuntu.com/security/CVE-2026-19931). Its October
backports fix other issues and do not close this finding. The private patched
zlib in lxml does not fix the separately loaded system provider.

The zlib applicability conflict was already recorded in phase 2 and must not be
presented as new proof of exploitability. The advisory describes 1.3.1.2–1.3.2;
Ubuntu marks older packages vulnerable while questioning the affected range.
The earlier exact 1.3.1 source inspection found no `gz_vacate` or `state->again`
branch introduced by `81cc0beb`. That source evidence remains valid within its
stated scope, including the separately inspected DuckDB static 1.3.1 source.
It does not establish whole-zlib safety or erase Ubuntu's unresolved older-version
classification. Arrow's replacement uses the patched provider and removes that
host dependency. The independently confirmed curl finding already blocks the
current Arrow runtime.

`NATIVE-ZLIB-CVE-2026-85091` is reopened in its existing scope;
`NATIVE-CURL-CVE-2026-19931` is an additional finding in
`shared_and_approved_runtime`. Prior phase-3 receipts remain historical,
including their earlier security dispositions. Neither the current development
environment nor its registered Arrow 25 fallback is currently established safe.
Phase-4 admission and advancement remain blocked.

The corrective source choice is private curl 8.23.0, libxml2 2.15.4, libiconv
1.19, patched zlib 1.3.2 and private OpenSSL 3.6.5. The curl source archive was
downloaded and matched the immutable upstream release-asset digest; the other
source inputs were copied from exact retained archives and rehashed.
`phase4/private-network-inputs-1/inputs.json` has SHA-256
`82b6ef34cfa93ccffd9e9493335e280e537d01c180f323adb665cab0e188af14`.
The source-choice receipt has SHA-256
`7a18833bf4521e6161be46d6d84783f8578a5c17d9aa0cf4ce4f8d2c58a976c9`.
HTTP/2 must retain its exact reviewed nghttp2 provider and pass an actual PIC
consumer link. Exact distro source-archive/patch bytes for nghttp2 were not
obtained; official source-package/backport evidence and installed-byte checks
are scoped separately. Complete native/host security clearance is not claimed.

Eight GitHits package-information calls, batched reviews, eight changelog calls
and source search/grep/read/status results were retained alongside official
upstream and Ubuntu evidence. GitHits has no vcpkg vulnerability dataset; zero
rows did not establish safety. The 100-artifact research inventory is
`phase4/native-provider-research-78778aaa-1/provider-security-research.json`,
SHA-256 `341864bb5e7d65bc78e46783d499f06b495495bd665e0574f96da980a2d2d7df`.

A separate immutable pytest executor now contains only the five existing pure
Python test-tool distributions, copied from 506 exact recorded files. Its
initial probe loaded no Arrow, curl, libxml2 or system zlib library. It is not a
BioMiner fallback. Its receipt is
`.native-runtime/pytest-executor-20261021-1/executor.json`, SHA-256
`5481049905dd8e59871aa3c686fa0e9baa94ec362612be0d34281a66bf8d4b08`;
the supervised provisioning receipt is
`phase4/pytest-executor-provision-1/execution.json`, SHA-256
`d6a35079a612930d5bd00d9847e1eef64e1be1a624c6a66abdd9295cb1515443`,
with supervisor/runner exit 0 and no interruption.

Containment then completed with supervisor/runner exit 0 and no interruption.
A fresh inventory under the cutover lock covered the two registered runtime
roots, exact bootstrap identities, readable same-user consumers, nine scheduler
commands and visible launcher configuration. No eligible active workload or
matching launcher was found. Only the exact development and fallback environment
root directories were changed from mode 0555 to 0000, using verified directory
descriptors and durable intent records. No job was stopped or restarted; the
`.venv` alias still points at its now-disabled target. Their payload bytes and
historical receipts remain intact. There is no currently qualified safe fallback.

The actual containment receipt is
`phase4/current25-containment-1/containment.json`, SHA-256
`9080b152f07c57b5976bbf473274199312076f6d5a1b5a31217519c550233a76`;
execution receipt `phase4/current25-containment-execution-1/execution.json` has
SHA-256 `4a1859d2c222c0b942a783ba14e17b7aefe0668ebb0ea4a289d7bc086f95bd23`.
Independent source review approved the descriptor-bound mutation and intent
journal: SHA-256
`2e99ed859cc7e383798c4b97944d357523bd4e5fa6697f2a2ecd5653524de067`.
Other users, remote schedulers and future reconstructed/manual environments
remain outside this scope. Private-provider builds and replacement runtime
qualification remain required; containment does not pass the phase-4 security
gate.

## Task 4.2: storage API compatibility

Starting commit is `3fcd4e5fad78a950ef20acbba398417389f6110e`. The refreshed
committed-source audit confirms that all 330 application-source hashes match the
retained 1,420-call audit. Its direct-call and dynamic-dispatch limitations remain
explicit. No direct affected use of the removed/restricted APIs was found, so
no speculative production migration or reader-depth override is introduced.
The read-only handoff is
`phase4/research-api-preparation/task4.2-handoff-0073962/handoff.json`, SHA-256
`4ca8669f3cee0307ba8a2acdcace18443671dcc10c837fffa1deafb793bcb654`.

Nine parameter-expanded storage cases exercise actual public local and fake-S3
batch I/O: ordered nested/null/timestamp schemas, non-Parquet bytes and truncated
footers, preservation of an existing artifact after an interrupted batch source,
and rejection of a corrupted staged S3 footer before promotion. The combined
`pytest -q tests/test_storage_backends.py tests/test_polars_compatibility.py`
baseline run passed 59 tests in 3.82 seconds on protected Arrow 25 with Polars 2.
Its supervised receipt is `phase4/task4.2-baseline-1/execution.json`, SHA-256
`cac3916013c43e6eb9f1488bb10aa738c44cefe76f3349580359b0e0f6c8916b`,
with supervisor/runner exit 0 and no interruption.

Arrow 26-specific depth qualification and the same storage cases on the actual
Arrow 26 generation remain pending. The new depth parameter does not govern
the separate eager Polars reader; no such compatibility claim is made.

The separate opt-in `test_arrow26_schema_depth_limits` case requires an explicit
committed-source/runtime context and a custom Arrow 26 profile. It runs the
preserved launcher verify → bounded worker → verify sequence. The worker binds
installed application bytes and runtime identity, preserves depth-22/depth-102
fixtures, tests explicit 21 rejection and 22/default acceptance, then observes
the untouched default-100 rejection through public local/fake-S3 batch readers.
This is a test implementation; actual target behavior remains unrun until the
new runtime qualifies.

An invocation without context correctly failed at its explicit guard before
creating qualification output (one expected failure in 0.05 seconds). Receipt
`phase4/task4.2-missing-context-1/execution.json`, SHA-256
`32c71e939531a4b5804bcbfaee66d67a1c0430f6a89d7226a454b4a23add9f5a`,
records supervisor/runner exit 1 and no interruption. That negative guard check
is not a passing Arrow 26 compatibility result. Independent source SPEC and
QUALITY approvals are `phase4/task4.2-independent-review-1/spec-review.json`
(`46045ebf7a605aed97188236630d9699ee8f83ad5709190f9182b2dfcc6e4de8`)
and `quality-review.json`
(`88551d4a7a556006d76071c66535fc7622fd5b31a858951c63ed0da5cc29ab3e`).

## Task 4.3: candidate admission foundation

The new Arrow 26 adapter normalizes its distinct raw receipt through an
`arrow26_build_receipt` backlink. It preserves nested JSON bytes and retains the
complete referenced source/build material, including relative inventories.
Validation binds integrated source fixes, the reviewed recipe, builder, Python
and OpenSSL attestations, dependency selection, actual commands, feature/codec
flags, compiler budgets and static link evidence. Indexed retained hashes and
source records avoid quadratic scans of large source closures. The Arrow 25
policy body and its ten existing version/policy assignments remain unchanged.

Synthetic cases exercise complete receipt acceptance, both assembly preservation
modes, relocation, source/static tampering, malformed evidence, the unchanged
freezer and frozen-source rejection. Arrow 26 candidates omit the legacy Arrow
25 rebuild recipe and explicitly reject unsupported rebuilds before environment
probes. This prevents a candidate from invoking the wrong family recipe.

Root observed the relevant missing-policy, command/provider and rebuild REDs.
Independent review additionally found array, path-validation-order, nested File
API shape and text-decoding errors. These were reproduced and corrected with
narrow guards. RED 8 included two fixture serialization failures; corrected
RED 9 then reproduced both actual Unicode failures. No broad exception masking
or Arrow 25 policy relaxation was introduced.

The final exact four-file check on the separate pytest executor was
`pytest -q tests/test_native_runtime.py tests/test_assemble_native_profile.py`:
217 passed in 11.56 seconds. The supervised receipt is
`phase4/task4.3-neighbor-3/execution.json`, SHA-256
`1a587903e78752b92c64020c4cef0bd5af4a0deb1c072ebc0c96844d8e53a7d5`,
with supervisor/runner exit 0 and no interruption. Successor independent SPEC
review has SHA-256
`246bde406a296bee5db69833cb328987e1290a6220807e549186d1b8bc6135d3`;
QUALITY review has SHA-256
`0b95c4e258e0a814727a5d13354347da00fadc4f2cddcde49c05c3d6cb5e9b5e`.
Both bind these final tested files; the earlier SPEC receipt applies only to
its older source hashes.

This foundation validates the earlier recipe schema. Its actual Arrow 26 raw
receipt pin remains unset, so no real Arrow 26 runtime can be admitted. The
private-network correction requires a new explicit schema/pins and actual
compiled-source/provider/ELF review. A custom wheel, installed native checks,
gzip/API qualification, safe fallback, lock update and complete phase gates
remain pending. The synthetic test results do not supply those observations.

### Task 4.3: codec qualification

Native capabilities now require all six configured codecs. They exercise
explicit Snappy, gzip, Brotli, LZ4 and Zstd Parquet round trips with nullable
integer and string values, plus a bzip2 byte-stream round trip. Parquet has no
bzip2 codec. Pinned Arrow 26 streaming tests confirm the selected stream API;
Snappy streaming is unsupported and is deliberately exercised through Parquet.
The existing source/profile binding retains the returned codec evidence.

Root observed eight initial missing-function failures. The final neighboring
policy tests on the pure executor passed: 39 passed, two deselected in 0.05
seconds. Receipt `phase4/codec-check-neighbor-2/execution.json` has SHA-256
`d0cf8293833da3baeb062ac4418e02bfbf6e8f7d6232160ff804cb67888da9e6`,
supervisor/runner exit 0 and no interruption. Independent SPEC and QUALITY
review passed for both exact tested source hashes. These cases reject missing
providers, changed values in each Parquet codec and corrupted bzip2 output.

This is qualification implementation, not observed native codec success.
Actual codec, TLS and spatial checks await a newly qualified runtime; the two
live native tests were explicitly excluded from this pure-executor run.

## Task 4.1: private network build tool

The new isolated builder pins curl 8.23.0, libxml2 2.15.4, libiconv 1.19,
zlib 1.3.2 plus `df84af25`, and the same patched Ubuntu nghttp2 source version
`1.68.0-2ubuntu0.2`. It retains the existing private OpenSSL 3.6.5 inputs.
Six private prefix archives, full source/object/build inventories, terminal
command receipts, static consumer maps and whole-archive PIC probes are
required. Its pure verification interface returns the complete retained
closure for downstream Arrow builds; a candidate receipt never admits a host.

The installed nghttp2 archive failed actual whole-archive shared linking with
`R_X86_64_PC32` against `stderr`. Its exact package identity was insufficient
PIC evidence. Receipt `phase4/nghttp2-pic-preflight-1/execution.json` has SHA-256
`84c1c2985108039a5d037c73667359df0adcfcf70dc6fb8dd2cd3fd742e0730b`.
The replacement uses the same HTTP/2 version and changes no OS package.
Official Launchpad source files and DSC checksums are retained. The local
keyring could not verify the signing key; signature verification is explicitly
unavailable. All three Debian patches applied in order without fuzz or offset,
with 1,226 original/final file hashes and every intermediate changed-file hash
retained in `phase4/nghttp2-source-access-1/exact-source-preparation-1/`.

Root reproduced missing source/PIC-plan, ordered-patch and compile-evidence
guards before implementation. Independent review then found that curl backend
variants and relative archive-member paths escaped the provider rejection
matcher. Both counterexamples were reproduced and corrected. The final exact
helper/test pair passed 73 tests in 1.70 seconds; supervised receipt
`phase4/network-nghttp2-focused-2/execution.json` has SHA-256
`44955f18ce3c625eacbb6d11f35454eb60c9951e21bdcd9ca1aefcfddb6ccbd2`.
The actual nine-input check also passed without creating a build generation;
receipt `phase4/network-nghttp2-input-check-2/execution.json` has SHA-256
`affbe05d6511e0c9f6c576245e86b155ade819fb3a90485cbe2fa1a3b74218a3`.
Both have supervisor/runner exit 0 and no interruption. Independent SPEC and
QUALITY review passed for those exact source hashes.

These results qualify the build tool's guards and inputs. Actual private
compilation, PIC/capability probes, Arrow consumer linkage and replacement
runtime qualification remain pending. The current registered runtimes remain
disabled and the phase security gates remain blocked.

## Task 4.1: separate corrective Arrow 25 recipe

The new `build_native_arrow25_network.py` preserves the old recipe and both
ordered compute patches. It uses the exact Arrow 25 archive, full source
inventories, private providers and the reviewed frozen builder. Inspection of
the real archive corrected a backend assumption: this version already uses
`_build_backend` and scikit-build-core. S3 is within `libarrow`; the build has
seven ABI 2500 libraries and no separate `ArrowS3` package. Boost selection is
explicitly transformed from 1.88 to the reviewed 1.92 archive.

All stages verify private-network and immutable inputs before execution and
again in `finally`, including failures. Final consumer linker maps must show
object contributions from each of seven core private archives. Merely loading
an archive is insufficient. The unused charset archive retains its separate
prefix PIC proof requirement. Host-provider rejection remains active.

Root reproduced the missing recipe and eight independent review regressions.
After correction and binding the committed helper, 59 tests passed in 0.66
seconds. Receipt `phase4/arrow25-network-final-pin-2/execution.json` has SHA-256
`fd8dfd8196563c873d0e92c5e96ce7206f40abd5e4901a5604d0377b0df6db80`,
supervisor/runner exit 0 and no interruption. Independent SPEC/QUALITY review
and a source comparison limited to the helper pin and one removed trailing
newline bind this final candidate.
The reviewer disclosed authorship of shared Arrow 26 guards; its Arrow 25
source, backend, patch and orchestration integration was separately assessed.

The actual Arrow 25 build and fresh fallback qualification remain pending.
This source commit does not reactivate either disabled runtime or close the
installed-provider findings.

## Task 4.1: Arrow 26 private-provider correction

Source policy `arrow26-integrated-private-network-v2` selects the approved
private prefix in both C++ and Python CMake projects. Global imported targets
carry the complete static CURL/XML/Iconv/zlib/OpenSSL/HTTP2 interfaces. The
helper source is captured and hashed before execution; complete provider and
immutable build inputs are rechecked around each stage, including failures.
Configured and final native evidence rejects protected host-provider inputs.

The unused bundled zlib patch path is removed. Its original archive is
explicitly retained only as a source pin; the selected compiled provider must
be the single patched private zlib. All enabled Arrow features, codecs,
encryption and allocators remain required. Final shared-library consumer maps
must prove object contributions from all seven core private archives, excluding
configure probes and dependency build maps. Root reproduced eight LOAD-only
counterexamples before the member-consumption correction.

After binding the committed helper, 127 focused tests passed in 1.68 seconds.
Receipt `phase4/arrow26-network-final-pin-1/execution.json` has SHA-256
`e8fbf1aa02cb8c741a79444faf45bb4941e649bd605749addc7c99fededed206`,
supervisor/runner exit 0 and no interruption. Independent SPEC/QUALITY review
plus an exact helper-pin-only comparison bind the final recipe. Real archive
inspection also verified both source transformations, recorded under
`phase4/arrow-family-source-policy-preflight-1/`.

Actual configure selection, compilation, final ELF review, native API/codec
qualification and admission remain pending. This supersedes the earlier v1
candidate schema without admitting any of its artifacts.

## Actual private prefix build

Attempt 1 stopped when the compiler observer encountered `ProcessLookupError`
while reading a disappearing `/proc` entry. The observer interrupted its owned
build; this was not a native compile failure or a budget violation. It observed
at most four frontends and produced no completed build receipt. Its directory
and outer/inner receipts remain intact. A new observer tolerates only this
transient enumeration/read race, resamples incomplete snapshots and preserves
the independent timeout, four-job limit and cleanup. Independent SPEC/QUALITY
review approved that narrow correction; primary psutil source corroborates
both vanished-entry exception types.

Fresh attempt 2 completed in 42.083 seconds with supervisor/runner exit 0,
no interruption and at most four frontends across 415 samples. Exact nghttp2
source and three patches produced 26 pinned-GCC PIC compilations with matching
target objects, archive members and installed headers. Whole-archive shared
links passed for the seven core archives and separate charset. Actual probes
confirmed curl 8.23.0, OpenSSL 3.6.5, zlib 1.3.2, libxml2 2.15.4, libiconv 1.19
and nghttp2 1.68.0. They reported HTTP/HTTPS/HTTP2 support and exercised
XML/XPath, encoding and gzip/XML. This offline probe made no HTTP request or
TLS certificate handshake; installed TLS checks remain required.
Recorded consumer dependencies contain only libc/libm, without TEXTREL or
protected host providers. All 54 commands succeeded with terminal cleanup;
retained licence copies match their sources.

Raw receipt `.native-runtime/private-network-security-20261021-2/build-receipt.json`
has SHA-256 `9371f44d18d198179e2a4dc808dc534feef7faca9e3d27328f4bcbfcb84c6b99`.
Outer receipt `phase4/network-build-outer-2/execution.json` has SHA-256
`3482c25a555e54783096eed723ed65c77f5737ca83509e3381baaecbf4603775`;
inner receipt has SHA-256
`b9b9b0b19155641783282bc96be400e9f007824f74a242a3eff93ddeda7a81ca`.
The pinned helper's post-build pure verification rehashed 12,479 closure files
and 70 private files; its execution receipt has SHA-256
`95b58ebc186b97d6a811cc314eaea224e15047285c5a551129efbaa4b384f754`.
Independent actual review approved this exact prefix for Arrow configuration.

The receipt remains `compiled_unverified`, `ready:false`. No Arrow installation,
runtime admission, safe fallback or whole-host security clearance follows from
these prefix observations. Installed-provider findings remain open pending
replacement runtime qualification.

## Task 4.3: private-provider receipt families

The new corrective Arrow 25 and Arrow 26 v2 families share strict evidence
validation while retaining distinct raw formats, source/patch contracts and
normalizer backlinks. The obsolete, never-admitted Arrow 26 v1 policy is
removed. The original Arrow 25 security function, its nine `ARROW_` assignments
and the separate `VERSIONS` identity remain unchanged. New-family rebuilds
reject the legacy recipe before probes or output creation.

Both assembly modes, relocation and the unchanged freezer preserve exact
nested bytes and source/provider/compiler/File API/static-consumer evidence.
The normalizer never executes the network helper. Historical original source
inventories bind original archives, not the current patched files; external
observation receipts remain exact retained leaves. Independent review found
that owned compiler-command JSON arrays were rejected as objects. Root
reproduced both family failures, then the nested-only dict/list reader fixed
the mismatch while keeping control receipts object-only.

The final four-file test run passed 259 tests in 11.22 seconds. Receipt
`phase4/private-family-admission-final-prefix-pins-1/execution.json` has SHA-256
`e1d6283f757af25f5afb7db017c9bc09c054c57f908a15c6c4a5eef194dbc316`,
supervisor/runner exit 0 and no interruption. Independent SPEC/QUALITY review
passed; only the two approved prefix/helper pins changed afterward, with an
exact source comparison and test rerun. A separate read-only network-subroutine
probe accepted all 12,479 actual prefix descriptor files and 70 private files.
It did not synthesize an Arrow compiled receipt or run full Arrow admission.

Both actual Arrow raw receipt pins remain unset. The real third configure
attempt completed CMake, then rejected ORC's `ZLIB_LIBRARY=ZLIB::ZLIB` cache
normalization. Its output remains preserved; a narrowly bound alias correction
and fresh configuration are required before compilation. This foundation
does not qualify a wheel, installed runtime or fallback.


## Task 4.1 correction: ORC zlib target normalization

The exact Arrow 25 and Arrow 26 sources deliberately replace `ZLIB_LIBRARY`
with `ZLIB::ZLIB` during ORC setup. Both recipes still request exact private
archive/header paths. Only that observed C++ cache value may pass a new guard:
the immutable manifest family, commit and context must match; the entire
private source/provider closure and generated CMake metadata must verify; and
File API 2.9 must expose exactly one global imported abstract static target
whose sole artifact is the source-owned patched private `libz.a`. Other aliases,
foreign or changed archives, incomplete evidence and Python aliases fail closed.
The source transforms, provider schema and final archive-member guards remain
unchanged.

Root first reproduced two intended alias-positive failures in each family.
The corrected focused suites passed 82 tests in 1.10 seconds (Arrow 25) and
150 tests in 1.87 seconds (Arrow 26), with supervisor/runner exit 0 and no
interruption. A read-only probe also passed the strict full cache guard against
the real preserved generation-3 CMake output; its execution receipt is
`phase4/arrow-zlib-actual-cache-probe-execution-1/execution.json`, SHA-256
`328b6f189b5fd1d06d0d6308b5babed232d76f219f0c1100d899464dcdd99741`.
This does not resume the failed generation, compile Arrow or admit a runtime.
Fresh committed-source configuration remains required.

Independent sequential SPEC/QUALITY review approved the exact four-file source
change with no blockers. Its retained review SHA-256 is
`f03d5ed6aaf07a78da6dce647cf92d1c65e5071cea34a641cb1fc69a0fcd7efb`.
The signed-by-agent field identifies the reviewer; it is not a cryptographic
signature. A separate full dependency-capture probe remains distinct from this
source/cache review.


## Task 4.1 correction: compressed source verification

The first complete configured-provider probe timed out after 300.215 seconds,
with supervisor/runner exit 124, no report and confirmed descendant cleanup.
Its failed receipt remains retained. The selected Boost archive contains
62,227 regular members; filename sorting creates 30,878 backwards data-offset
jumps. Primary CPython 3.14.7 source confirms these seeks restart gzip
decompression readers and replay preceding data.

Both new recipes now read members in physical archive order, then sort archive
evidence before appending generated rows in their existing order. All layout,
type, symlink, content, transformation and generated-template checks remain.
The original Arrow 25 recipe, helper and compute patches are unchanged.
Root reproduced two repeated-rewind failures while all 16 malformed-source
cases passed. The complete corrected suites passed 250 tests in 2.88 seconds;
receipt `phase4/arrow-archive-order-focused-1/execution.json` has SHA-256
`9adaf25b23ca19df891951ad4238e16284f9cff60538cad80b01ee898e029afe`.

The real read-only configured-provider probe then completed in 84.383 seconds,
with supervisor/runner exit 0 and no interruption. It captured 469 targets,
18 native inputs, 12 static archives and complete selected source evidence,
including 62,228 Boost file/symlink rows. Its receipt is
`phase4/arrow-zlib-actual-selection-probe-execution-2/execution.json`, SHA-256
`d9aaea3c58c51b45e5eac4d6410a4f072751e2942e33cb05d13c40697ef2c767`.
This probe reads the preserved failed generation 3. It establishes the corrected
capture behavior, not a successful new build, wheel or runtime admission.

Independent sequential SPEC/QUALITY review approved the exact four-file change
with no blockers; review SHA-256
`848d4d672e9754e2c9fd0089e29fd671482a3c9b42cddeaea188a84c38bb1a3a`.
The verifier still scans all metadata, permits one initial decompressor rewind
and retains its existing per-file memory/hash behavior. This is not a constant
memory or universal elapsed-time claim.


## Task 4.3 followup: final recipe bindings

Only the two new-family recipe constants now bind the reviewed ORC alias and
physical-traversal corrections. Whole-source reversal to the previous committed
normalizer proves that all policy bodies and original Arrow 25 constants are
unchanged. Both actual Arrow raw receipt pins remain `None`.
The final normalizer/assembler suites passed 259 tests in 9.17 seconds, with
supervisor/runner exit 0 and no interruption. Execution receipt
`phase4/private-family-admission-final-recipe-pins-1/execution.json` has SHA-256
`b5078f90b8fa975ce8b3a78157f061d438794babe0979683a13f2904141f3ea9`.
This updates recipe identity only; it admits no compiled artifact.


## Actual configuration handshakes and failed generation 4

Fresh Arrow 26 generation 4 configured and captured providers in 203.747 seconds;
corrective Arrow 25 generation 1 completed in 252.156 seconds. Both supervisor
and runner exited 0, without interruption or budget abort, observing at most
one frontend. Their immutable configured inputs, first checkpoints and
byte-exact configured-raw snapshots are retained. Independent sequential
SPEC/QUALITY review approved each only for isolated bounded compilation.
The review hashes are `659937bb9f943225bea0f142c34a15bc2003a4b0f3d44f21971e0ce6f4df8d05`
(26) and `a5a29e91e756b4a5f5a2c913e7fb1a7010b1f4ffaefa73df19174f5d8ab71392`
(25). The snapshots preserve the configuration observations when a recipe later
replaces its mutable build receipt.

Arrow 26 compilation then failed in 294.16 seconds at 1,011 of 3,139 displayed
build steps. The C99 uriparser unity unit reported an implicit `reallocarray`
declaration and an integer-to-pointer return error. Supervisor/runner exited 1,
without interruption, cleanup error or budget abort; the observer saw at most
four frontends. This failure occurred before an attempted owned-process stop
inspection, so no signal was sent. No wheel was produced. Failed raw receipt
SHA-256 is `898fc46c9255cc171d346398221b8d63ad18b5e39510a17f79538c8dc0da9da4`;
the next checkpoint is preserved, but is not approval to resume changed inputs.
Observation `phase4/arrow26-generation4-failed-observation-1.json` has SHA-256
`24b9d12e0d2200b32e0025f74d7d17d285d8c0adcf679c38423de0af26840a94`.
Arrow 25 remains configured and has not been compiled.

Primary source and CMake documentation explain the uriparser failure: its
feature-test macros must precede libc headers; global unity inclusion changes
that ordering. An isolated real CMake reproduction fails with unity, then builds
the unchanged 114-file source with per-target unity disabled. C99 and
`HAVE_REALLOCARRAY` remain enabled, without suppressing diagnostics. The four
commands have the expected exits 0/1/0/0. The supervised diagnostic completed
in 3.347 seconds with exit 0, no interruption and at most one frontend.
Its execution SHA-256 is
`6382b82965bbedaa9c08444201cb5581d0dc5764b2e522862e5ea69bc164d0a9`.
This proves the bounded compatibility correction; recipe integration remains
pending alongside the SDK cohort changes.

## Newly established bundled SDK findings

The explicit provider map covered 18 native inputs, not the complete bundled
SDK closure. Current primary advisories prompted a separate review of retained
registered Arrow 25 sources and final maps. The exact historical wheel/profile
chain, two relevant ELF members, seven map hashes and source/archive records
bind actual linked implementations:

| Component | Selected version | Current finding scope |
|---|---|---|
| c-ares | 1.34.6 | Three identified July advisories; query completion and parser/process code is linked |
| s2n-tls | 1.7.0 | TLS record handling is linked; QUIC code is present but QUIC-server deployment applicability remains pending |
| AWS SDK C++ | 1.11.778 | Base64 decode implementation is linked; fixes also require an updated CRT cohort |
| gRPC C++ | 1.76.0 | Affected transport/decompression implementations are linked; maintainer GHSA has no CVE asserted |

`gRPC_CARES_PROVIDER=none` in the cache did not establish absence: the retained
wrapper calls `ares_gethostbyname`, which calls `ares_getaddrinfo`; the final
Flight map contains those implementations. No live exploit or deployment
trigger was exercised. Both registered roots remain disabled at mode `000`.
The independent applicability report has SHA-256
`f8a298fb86b8c17787994aa4e0d5cb3255389a4b4fb5d7ff686c0f5d968fbb81`.

The new intermediate security report preserves the previous eleven findings
unchanged, adds eight stable SDK IDs conservatively as open, and adds an open
bundled-closure review gap. It contains 19 findings, ten open, and two coverage
gap records. SHA-256 is `f1f7cb36f41e7aa62a34547b624508b95dab3cb7c53989e154a29fb3b4c50fd5`.
This is not a passing phase report. Further current SDK advisory review and a
coherent compatible replacement cohort are required before fresh full builds.
The intermediate capsule's inherited top-level counts were not recomputed;
the totals above count its actual finding/gap arrays. Preserve that capsule
unchanged and correct the counts in its successor. Eleven additional c-ares
GHSAs published on 20 October and Thrift C++ `CVE-2026-41607` are already
established pending additions; the nineteen-row snapshot is not exhaustive.
Phase 4 has not been pushed or completed.

The source-reviewed replacement cohort for both Arrow families is gRPC
1.83.1, Protobuf 35.1 (C++ 7.35.1), Abseil 20250512.1 and c-ares 1.34.9.
The exact gRPC submodule and Protobuf source pins support that cohort;
c-ares 1.34.7 and 1.34.8 remain affected by the eleven newer advisories.
All four archives are retained and hash-verified. Recommendation SHA-256 is
`c7c8b0f6debb723015d6c82cd10d25a7f25046e6f671b2d5c93088fb16e3deaf`.
Fresh configuration, generated-protocol compilation and runtime checks remain
required. Abseil's January hash-overflow fix is already present in the old
selected 20250127.0 source; it is not a new selected-source security finding.
Removed Abseil static target names are a latent static/cross-compilation issue,
not an established blocker for these native shared-library configurations.

A separate libiconv review recovered direct current NVD API access after
transport-specific web failures. Complete returned keyword snapshots contained
one Mutt-specific libiconv mention and zero libcharset matches; the CPE keyword
queries returned zero rows. These are bounded corpus observations, not proof
of universal advisory absence. Exact GNU 1.19 source confirms the known GB18030
allocation guard, and the existing source-object finding remains unchanged.
The scoped review SHA-256 is
`4a615b29e5739cf4c6f2579b09d3c0a4b4f39569e7f8178691b4694e98bd12fb`.

## Compatibility scope and corrective Arrow 25 dependency recipe

The user excluded further cybersecurity work and requested continued dependency
and Polars/Arrow compatibility implementation. The amended plan preserves
existing admission checks and contained environments. The additional RapidJSON
counter test source was not compiled or executed; no arithmetic patch was
implemented. Scope receipt SHA-256 is
`06ab19c8ff98922fdf7bfae22c9e6eb516b41f890568f033fbb7c2707cd0e534`.

The corrective Arrow 25 recipe now captures the reviewed 21-member source
cohort as explicit required archive inputs. It retains the original two ordered
compute patches, all fourteen capabilities and six codecs, encryption, both
allocators and the private provider prefix. Eight direct SDK packages require
bundled selection. Both version and checksum transformations reject missing,
changed or duplicate assignments, including different-valued duplicates that
would otherwise override an earlier assignment. Source policy is explicitly
`arrow25-two-compute-patches-private-network-v2`; historical recipes and raw
artifact admission pins are unchanged.

Parent-owned regression first observed four differing-duplicate cases fail in
0.19 seconds. After correction, both complete recipe suites passed 424 tests
in 6.11 seconds, supervisor/runner exit 0, no interruption. Final execution
receipt SHA-256 is
`fc19ad5c24172acdb14318417172a3df553cb534f674c3b9e3f43758fef4a139`.
This validates recipe orchestration with synthetic fixtures. Actual candidate
configuration, compilation, final source proof and runtime compatibility remain
separate work; no installed runtime is admitted by this task.

Arrow 26's equivalent cohort recipe was committed separately at `7f8001b`.
It uses source policy `arrow26-integrated-private-network-v3`, retains its
integrated compute corrections and eight ABI-2600 libraries, and disables
unity compilation only for uriparser. Its independent core review passed;
review SHA-256 is
`14c5e46bab2973c3261b8729d41abf7568cba91b98163aa84fe5ae28d6dcd06a`.
The common finite 21-member cohort matches retained input SHA-256
`584b47d20f2ec9ce3d3f98ed4e616c9b866de96d5033cd8f8406060b7d88a69f`.
Both recipes require fresh candidate generations and keep their existing
configuration/compilation approval boundary. Phase 4 remains in progress.

## RapidJSON target compatibility correction

Arrow 26 generation 5 failed configuration in 215.011 seconds with exit 1,
no interruption or cleanup error, and at most one compiler frontend. The newer
RapidJSON source provides an `INTERFACE` target named `RapidJSON`; Arrow's
wrapper attempted to create the same target again. ORC's download completed
and was not the cause. The failed generation and checkpoint remain intact;
raw receipt SHA-256 is
`c72cd5e6b6f1c414c7e169910c300f02589955b3b092621f242e922e3da078fc`.

GitHits inspection of both pinned Arrow wrappers and RapidJSON's exact source
showed a second requirement: the upstream target provides only an install
include and exports its metadata. Both recipes now preserve that target,
create the original imported fallback only when absent, and add the source
include through `BUILD_INTERFACE`. This makes headers available to Arrow
without embedding the build source directory in installed exports. The custom
target dependency and all other recipe code, source pins and capabilities are
unchanged. Arrow 26's correction is committed at `89ce72b`.

Parent-owned RED observed six failures and two passes in the eight new cases.
The corrected complete suites passed 432 tests in 6.68 seconds, exit 0 without
interruption; execution receipt SHA-256 is
`7767382db00e9f322021e9154664492c72e33927d0c92518ec485071c24c1510`.
The new CMake cases use `LANGUAGES NONE` to verify actual target reuse,
fallback, effective consumer includes and export generation without compiling.
Independent reviews passed for each recipe. New full candidate configurations
and compilation remain required; these tests do not admit a runtime.

## Finite cohort receipt adapters

The new Arrow 25 and 26 source policies now have separate finite receipt
adapters. Dispatch uses the explicit Arrow schema, retaining the seven-library
ABI-2500 and eight-library ABI-2600 contracts. A separate
`bundled_source_attestation` field binds supplemental source evidence without
rewriting raw build receipts. Normalization requires paired attestation path
and checksum arguments, preserves copied inputs, and rejects unsupported
legacy rebuild dispatch. Actual receipt, attestation, verifier and transformation
review pins remain closed. Historical source policies and their pins remain
unchanged.

Independent review found that retained source symlinks could still resolve
through historical directories. A regression reproduced that traversal and
three invalid-chain cases: four failures, two passes in 0.62 seconds. The
correction resolves recorded path components and link chains without historical
filesystem access, preserving direct-mode checks. Focused GitHits inspection
of CPython 3.14.7's `posixpath.realpath` informed the component ordering and
cycle handling; its filesystem implementation was not copied.

The complete adapter suites then passed 294 tests in 11.22 seconds, exit 0
without interruption. Execution receipt SHA-256 is
`94bf25598c2b527e3eaa914c930e5d750ff91cd8fcae49d78aad80a8a37d518b`.
The independent review addendum closes the relocation finding; SHA-256 is
`51f3bb4024325be55ee7329dbb4b1946a4b72f85ad1f904cb3661f65db1d3145`.
The legacy admission function and protected historical assignments remain
byte-exact. Actual final source attestation, native wheel qualification and
the application compatibility checks remain separate tasks. No runtime is
activated by this adapter change.

## Source-proof verification primitives

A standalone standard-library verifier now checks source archive membership,
current source bytes, exact generated-file derivations, the disjoint 21-member
supplemental cohort, and retained copies selected by checksum. The AWS header
derivation checks the captured static/default configuration. It performs no
recipe, generator, compiler or native-tool execution. Its completed-attestation
reader rejects unqualified ExternalProject rules; capture/CLI orchestration,
complete final-subject checks and additional generated-source rules remain
separate work against the actual completed builds.

The initial test run recorded 29 setup errors because the implementation was
intentionally absent, then all 29 primitive tests passed. Independent review
found that checking each symbolic-link target alone missed cycles and escapes
through directory links. Eight additional tests reproduced six failures and
two passes in 0.27 seconds. The inventory-only component resolver fixes those
cases in both original-source and retained-copy modes without accessing
historical directories.

All 37 primitive tests passed in 0.19 seconds, exit 0 without interruption;
execution receipt SHA-256 is
`edb3c58b4b1ee4c5b3d8fe824ba742aa7305cfa785e03d3c1286230118620f7f`.
Independent review passed within this primitive scope and closed its chain
finding; addendum SHA-256 is
`f705174d9aabeba38912edc8cd0d5f0decae8ab4091e5a1428be569af96deff1`.
No actual final attestation, wheel qualification, runtime admission or phase
completion is claimed by these synthetic tests.

## Azure static XML link-interface correction: Arrow 25

The completed Arrow 26 wheel failed its first native import with
`undefined symbol: libiconv_open`. Read-only inspection of the completed
corrective Arrow 25 library found the same three unresolved `libiconv` symbols.
Both builds had compiled successfully; compilation and a union of link maps
did not establish that each final library could load. The failed build
generations and candidate environment remain intact and unqualified.

Exact Azure and CMake source inspection through GitHits explains the missing
dependency: Azure storage-common privately links the raw `LIBXML2_LIBRARIES`
archive path, bypassing the existing `LibXml2::LibXml2` interface to Iconv, zlib
and the math library. The Arrow 25 recipe now adds that named dependency to
the source-owned static Azure target immediately after its FetchContent step.
The finite transformation rejects missing or duplicate anchors, aliases,
imported targets, shared libraries and foreign source directories. Windows
behavior, dependency source bytes, providers, capabilities and flags remain
unchanged. A direct Azure source patch and linker group flags were rejected
because the existing named interface expresses the missing dependency.

Root-owned RED recorded eight failures and one pass in 3.55 seconds. The
corrected complete Arrow 25/26 recipe suites passed 450 tests in 8.14 seconds,
exit 0 without interruption; execution receipt SHA-256 is
`b1d5d96e9985bd58cb9bc1c24ca5fdc3dc991083adb97468baa96c5be4c81f76`.
The tests include an actual tiny shared-library link that fails without the
interface and succeeds with it; `-z,defs` belongs only to this fixture.
Independent Arrow 25 source/test review passed; review SHA-256 is
`a1fbf36afbfa4c056528d2f262f2cd8f9a518a166192aadc7b51d4708b4e7de4`.
Fresh full builds and native imports remain required; this correction neither
activates a runtime nor completes phase 4.

## Azure static XML link-interface correction: Arrow 26

Arrow 26 now applies the equivalent finite, guarded Azure target correction.
Only one source-transformation pair changes; the selected cohort, build flags
and eight-library ABI-2600 capability contract remain unchanged. Root RED
recorded eight failures and one pass in 1.12 seconds. The same complete
450-test GREEN above binds both final recipe and test hashes. Independent
Arrow 26 source/test review passed; review SHA-256 is
`8d25a898d6f2ff936de0776533fb0306884481d04e2fc9011e19c495c9bb09a4`.

Both corrected recipes require fresh generations. Reuse of the existing
private provider prefix is appropriate because its iconv archive already
defines the missing symbols; the defect was the omitted link interface.
The earlier completed wheels remain runtime failures, and the phase remains
unpushed until its required native and application checks are complete.
