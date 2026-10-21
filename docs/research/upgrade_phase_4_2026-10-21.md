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

The qualified two-patch Arrow 25 recipe and artifacts remain unchanged for
fallback and four-runtime testing. Unrelated scientific work remains outside
this task. Arrow 26 installation, runtime qualification, phase completion and
production promotion are not established by preparatory research or builder
tooling checks.

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
