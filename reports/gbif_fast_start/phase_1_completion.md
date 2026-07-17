# Phase 1 completion — admission policy and schema contracts

Status: complete<br>
Evidence head: `c15016893462c93683185959ab1277e5393899b5`<br>
Generated: `2026-07-17T11:13:35Z`

## Immutable task ledger

| Task | Commit | Outcome |
|---|---|---|
| `gbif-fast-1.1` | `da7fb1944011dd2ee5e7e7aec53f123a9c589fc0` | Added the immutable, versioned admission policy and canonical fingerprint. |
| `gbif-fast-1.2` | `282696d8d094a1af8c2be63ea0afa6d972e73e47` | Added provider, human, provisional, QA, routing and audit evidence to support-manifest v3. |
| `gbif-fast-1.3` | `9170f05c6b79b9d3ef2c9b528c4d261f53369bc8` | Added summary v2, readiness v3, `ready_provisional`, and independent capability permits. |
| `gbif-fast-1.4` | `c15016893462c93683185959ab1277e5393899b5` | Added explicit v2 strict migration and policy-identity invalidation tests. |

All four commits are present on `origin/main` with their own GitHits records.

## Compatibility matrix

| Input or change | Result | Evidence meaning |
|---|---|---|
| Native v3 strict support | Pass | Existing human-verified behavior remains explicit and permit-capable. |
| Legacy v2 passed directly to v3 validation | Rejected | Old artifacts remain distinguishable. |
| Legacy v2 passed through explicit migration | Pass | Human verification is preserved; provider assertion is not invented. |
| Admission policy without mode | Rejected | Missing mode never becomes fast-start. |
| Admission mode or policy fingerprint changes | Stale identity rejected | Readiness and every downstream fingerprint chain must rebuild. |

`ready_provisional` permits reference embedding and provisional scoring. It
cannot permit calibrated scoring or scientific release, and no readiness state
bypasses mandatory human review for final Flickr occurrence output.

## Validation evidence

- Focused Phase 1 contract and downstream suites: 177 passed.
- First full-suite run: 2263 passed and 24 failed. Every failure used the same
  production-run test double that lacked the new embedding capability field.
- Remediated production-run module: 84 passed.
- Final locked full suite: 2287 passed in 84.22 seconds.
- Ruff checks, JSONL parsing and whitespace validation passed.

Phase 2 may begin. Automated GBIF eligibility, YOLOE routing, independence
selection, and admission compilation are not claimed by this phase.
