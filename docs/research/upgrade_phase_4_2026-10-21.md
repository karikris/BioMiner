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
