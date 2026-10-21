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
