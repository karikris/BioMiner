# Geography-conditioned dynamic pooling — technical release verification

Verdict: **technical workflow verified with explicit live and secret-scan
limitations** at committed workflow SHA
`19fd744b1104c09dde75367bafb6b531ef4239a4`.

## Test gates

| Gate | Result |
|---|---:|
| Full regression | 3,233 passed in 169.62 s |
| Strict compatibility | 77 passed in 17.04 s |
| Adaptive mode | 68 passed in 7.56 s |
| Dynamic pooling | 428 passed in 60.94 s |
| CLI | 146 passed in 8.73 s |
| Configured schema/parity | 44 passed in 0.84 s |
| Downstream handoffs | 85 passed in 5.87 s |

Focused gates intentionally overlap the full regression. They are independent
release receipts for named surfaces, not additional unique tests.

The schema/parity gate uses exact committed TaxaLens
`e845dd98493979f37b04dbb6538e0d7b8758ca11` and ButterflyLens
`1cea643623f2f20a2bea72afc754c7b194db3278` contracts. The ButterflyLens parity
fixture covers 24 schemas, 20 valid cases, 20 invalid cases, 20 version checks
and 15 vocabulary checks. It does not execute a live database or modify a
consumer worktree.

## Static, supply-chain and artifact gates

All 867 tracked Python/repository files were evaluated in the relevant scope:

- Ruff passed across all tracked Python files.
- The locked all-extras graph resolves 44 packages under Python 3.14.5 and
  `uv` 0.11.19; `pip-audit` reports zero known vulnerabilities. The local,
  unpublished `biominer` package is explicitly skipped.
- No tracked file exceeds 1 MiB. The largest is the 339,686-byte append-only
  GitHits provenance ledger. The only tracked Parquet file is a bounded
  17,865-byte adaptive-release review-queue fixture; no source media or model
  weight is tracked.
- `git diff --check` passed.

The tracked secret scan reports 453 heuristic findings in 106 files: 424
semantic/content hashes, SHAs or fingerprints; 21 secret field names,
placeholders or security-report terms; six deliberately fake Basic Auth test
URLs; and two deliberately fake Base64 secret-loader values. A separate scan
found zero private-key blocks or common live-token prefixes. This is a
classified heuristic result, not a proof that every possible secret format is
absent.

## Evidence boundary

No GitHits call was made, per the user's directive; both required records have
`solution_id: null`, and GitHits contributed no code or architecture.

No live corpus, current source-bound human review set, production timing run or
MPS run was executed. Therefore reviewed accuracy, subgroup support, comparable
throughput, MPS peak memory, production-default selection and occurrence
release remain unavailable. A green technical gate does not authorize any of
those claims.
