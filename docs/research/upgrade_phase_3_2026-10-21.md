# Phase 3: Polars 2 compatibility

Phase starts on `main` at `e723354ec27016c17b0c5ad36b9caef01d88548d`.
The verified phase-2 push handshake is
`/home/toffe/.local/state/biominer/upgrades/20261021/phase-2-pushed.json`,
SHA-256 `be7e78758322b74adec6e5a7ea820350d0a3946d3111aa5e7b6e78023afa8885`.
The continuation gate returned `ready` for phase 3.

Unrelated scientific changes remain outside this task's commits and tests.
The frozen patched phase-2 fallback and active development environment are
preserved. No production scientific run or hosted-service qualification is
claimed here.

## Task 3.1: explicit list expansion

Both production `explode` calls now specify
`empty_as_null=True, keep_nulls=True`. Sampling still rejects empty, null,
mixed-null and unresolved query definitions. Original query IDs and duplicate
discovery hits remain intact; structured provenance ordering remains explicit.
Flickr reports retain duplicate label counts, lexical ties and observed query
hit distributions, while absent instrumentation remains distinct from zero.
Two test-only expansion calls now declare the same semantics.

GitHits inspected tagged Polars 1.44.2 implementation and warning tests and
the [Polars 2 migration guide](https://github.com/pola-rs/polars/blob/22a147de3d2bb2e44b97338a2510816c7105c9f2/docs/source/releases/upgrade/2.md).
The two explicit arguments preserve the approved row contract across the
default change. No upstream implementation was copied.

The new tests produced nine expected warning failures before the repair.
Afterwards, the four focused test modules passed with deprecations treated as
errors: **56 passed in 2.05 seconds** on Polars 1.44.2, and **56 passed in
2.56 seconds** on Polars 2.0.0 with custom Arrow
`25.0.1+biominer.f918dfd84c8c`. Both use Python 3.14.7.
Independent automated specification and quality review approved the six
changed files; no human review is claimed.

Receipts are retained beneath
`/home/toffe/.local/state/biominer/upgrades/20261021/phase3/task3.1/`:

| Artifact | SHA-256 |
|---|---|
| Expected failing test receipt | `6fd0677fc0844cf98cf0c4e573cd138ee9977fb20e484ae620e8d161c3ea09f0` |
| Polars 1 test receipt | `dcd336e344c084b45616890f960dc09a70796f8f5ca6595f9bbf3cad99b16d57` |
| Polars 2 test receipt | `a775d1d55c5e99626a67259e7e494a3dd8db90cf393683a091b5f9ddf7523d75` |
| Six-file source hash inventory | `0a2eeb17efc6204d67fc2c060fea4b33469564c031ba0f6f357357360b912fed` |

These are focused working-tree checks. The Polars 2 environment is exploratory,
not admitted for production. Task 3.2 resource/schema checks, task 3.3 locked
runtime qualification, clean-snapshot full regression and the phase push
handshake remain pending.
