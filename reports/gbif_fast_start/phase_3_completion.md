# Phase 3 completion — readiness and support-manifest refactor

Phase 3 is complete and regression-clean. All four task commits were pushed to
`origin/main`; the full repository suite passed 2,355 tests in 86.39 seconds.

## Immutable task ledger

| Task | Commit | Outcome |
|---|---|---|
| 3.1 | `e239b4c3694e466041f39445e7d276e63b251ed1` | One fail-closed evaluator for strict human and adaptive GBIF support paths |
| 3.2 | `053473526316890520bfe85a1ea00f1d92a23efc` | Explicit policy, provider, automated-QA, route, audit and rejection readiness checks |
| 3.3 | `7b81a954943a2fca46a13dfbd0e06b1a711439c7` | Narrow provisional screening and prototype permit |
| 3.4 | `e063d7e7dfa7745aa74f66cd5b6a3e65ce9fd3db` | Separate calibrated and provisional-nonparametric dependency chains |

## Capability boundary

| Capability | Strict ready | Ready provisional |
|---|---:|---:|
| Reference embeddings | Yes | Yes |
| Prototype creation | Yes | Yes |
| Provisional scoring | Yes | Yes |
| Calibrated scoring | Yes | No |
| Scientific release from this permit | Yes | No |
| Downstream Flickr review required | Yes | Yes |
| Reference-bank statistical audit required | Policy dependent | Yes |

The strict scientific-release capability concerns reference readiness only. It
does not bypass the separate mandatory human-review gate for every Flickr record
included in a final occurrence dataset.

## Admission and dependency evidence

Support eligibility now has two named evidence paths. The strict path requires a
completed human review and verified target identity. The adaptive path requires
GBIF provider assertion, all automated gates, compatible YOLOE route, canonical
deduplicated media, an explicit provisional declaration and a permitting policy.
A completed human rejection overrides every provider or automated signal.

The old `verified_support_only` check no longer exists. Adaptive readiness
instead reports admission-policy compliance, provider integrity, automated QA,
route separation, provisional declaration, statistical-audit planning and human
rejection enforcement. `strict_support_only` appears only in strict mode.

Classifier and calibrator artifacts remain mandatory for calibrated scoring.
They are not required for explicitly selected provisional nonparametric scoring,
whose persisted semantics are `uncalibrated_similarity_and_margin_not_probability`.
There is no silent fallback between these modes.

## Verification and limitations

- Full regression: 2,355 passed, 0 failed in 86.39 seconds.
- Strict and adaptive contract, permit, stale-fingerprint and dependency tests pass.
- All task SHAs were present on `origin/main` before this ledger was created.
- No live production GBIF bank, embeddings, prototypes or Flickr scores were
  produced in this phase.
- No accuracy, calibration, speed or manual-review savings are claimed yet.

Phase 4 must bind reference embeddings and prototypes to the admission identity,
add reference-bank diagnostics, and implement provisional ranking evidence.
