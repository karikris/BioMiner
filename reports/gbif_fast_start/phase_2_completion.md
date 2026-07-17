# Phase 2 completion — automated GBIF quality admission

Phase 2 is complete and regression-clean. All four task commits were pushed to
`origin/main`, the focused Phase 2 suite passed 120 tests, and the full repository
suite passed 2,348 tests in 85.62 seconds.

## Immutable task ledger

| Task | Commit | Outcome |
|---|---|---|
| 2.1 | `2bc8f146a4745230786a30ef0f23773c36f4c5c0` | Pure 24-gate GBIF provider-assertion evaluator with complete ordered outcomes |
| 2.2 | `3dbdb0ce118478e8a1eac16b1b50b0e76e39d5b4` | One-pass YOLOE reference routing with persisted detector, area, domain and shared full-frame evidence |
| 2.3 | `782c84c38725d025272acf5383569ed5f8108de1` | Deterministic observation, photographer, duplicate, view and class-balance selection |
| 2.4 | `b9d790baeba3b3597bf1973d83a5650883f5ec57` | Fail-closed compiler for decisions, provisional support and synchronized summaries |

## Phase acceptance

| Criterion | Evidence | Result |
|---|---|---:|
| Fixture-backed GBIF admission | `test_gbif_admission_eligibility.py`, `test_reference_admission_compiler.py` | Passed |
| YOLOE route behavior | `test_reference_yoloe_routing.py`, `test_full_frame_yoloe_routing.py` | Passed |
| Canonical duplicate behavior | `test_reference_deduplication.py`, `test_provisional_selection.py` | Passed |
| Deterministic independent selection | `test_provisional_selection.py` | Passed |
| Every task pushed | `origin/main` contained `b9d790baeba3b3597bf1973d83a5650883f5ec57` before this ledger | Passed |
| Focused regression | 120 passed, 0 failed | Passed |
| Full regression | 2,348 passed, 0 failed in 85.62 s | Passed |

## Required artifact contracts

The compiler implements and fixture-tests all five required artifacts:

- `reference_admission_decisions.parquet`
- `reference_provisional_support.parquet`
- `reference_admission_summary.parquet`
- `reference_admission_report.json`
- `reference_admission_summary.md`

These are contract and fixture results, not a claim that a live GBIF production
bank was populated in Phase 2.

## Transparency and evidence boundaries

Every compiled provisional-support row independently carries source provenance,
the GBIF provider assertion, all 24 automated gate outcomes, route and detector
evidence, duplicate resolution, selection evidence, admission-policy identity,
and provisional status.

- GBIF identity remains **provider-asserted and unreviewed**.
- `human_verified` remains `false`.
- YOLOE provides quality, visual-domain, subject-area and life-stage routing; it
  does not decide species identity.
- A coordinate-less image may support a global prototype but is explicitly not
  geographic-prototype eligible.
- Provisional support does not grant calibrated scoring or scientific-release
  permission.

## Remaining dependencies

Phase 3 must connect the provisional compiler to generalized support manifests,
readiness states, capability permits and downstream dependency invalidation.
Later phases must build embeddings and provisional prototypes, integrate staged
scoring, run statistical audits and human remediation, and measure real speed and
review-work savings. No such downstream outcome is claimed here.
