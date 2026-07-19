# Dynamic-pooling production-selection ablation

Status: **24 variants reported; 0 variants eligible for production selection**.

The machine-readable table covers all combinations of three candidate
strategies, two pool variants, and four raw fusion methods. Every row carries
all nine preregistered selection criteria, including explicit unavailable
states and denominators. Row identities bind the table to the frozen plan,
score execution, and review-work plan.

## Criterion register

| Criterion | Status | Selection eligible |
|---|---|---:|
| Target/candidate recall | Available as fixture structural recall only | No |
| Reviewed precision and confidence bounds | Unavailable; no completed real reviews | No |
| Family and geographic subgroup behavior | Unavailable; no completed real reviews | No |
| Review workload | Seven representative and seven targeted fixture work items; zero completed | No |
| Computation | Observed cached-vector fixture execution | No |
| Embedding and matrix reuse | Observed for the complete shared execution | No |
| MPS memory | Unavailable; fixture execution did not run on MPS | No |
| Target-pruning regressions | Zero in the complete fixture union | No |
| Unsupported statistical claims | None in the validated table | No |

The family-first-safe schedule has structural fixture target recall of 1/7 at
rank 1 and 7/7 at rank 3. Geography-first and parallel-union have 7/7 at rank
1. Every schedule retains 7/7 targets at rank 5 and the complete five-taxon
union. These are candidate-order observations, not reviewed classification
accuracy.

The shared execution uses 14 cached-vector work items and seven unique query
vectors, records seven query-vector reuse events, 100 pool-matrix references,
35 unique pool matrices, 65 within-batch matrix reuses, and a maximum 2,240
bytes of pool matrices in its single batch. These metrics describe this whole
fixture execution, not per-variant incremental cost or avoided runtime.

Reviewed precision, confidence bounds, family and geographic subgroup
estimates, and MPS peak memory are unavailable. There are zero completed real
reviews, leaving the full 86-effective-review shortfall; the subgroup floor is
30 independent records. All raw scores remain non-probabilistic. Missing
geography is not absence. No production default or occurrence release is
authorized.

Machine-readable table:
`reports/geo_dynamic_pooling/pilot/production_selection_ablation.csv`.

Table fingerprint:
`sha256:7368a623b9fbd9a665e2ae135c7da24c2adf9ed3a87b8e36241a3a2f14a676ec`.
