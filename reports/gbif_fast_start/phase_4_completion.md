# Phase 4 completion — embeddings, prototypes and fast scoring

Phase 4 is complete. All four task commits were pushed and the full repository
suite passed 2,359 tests in 86.92 seconds.

| Task | Commit | Result |
|---|---|---|
| 4.1 | `fe882bd8f0b4749cd10f8cea28a6bbcc2f7dfbb3` | Admission-bound embeddings with content/model cache reuse |
| 4.2 | `d5103b5842bbb25555dd6f91b68c1962cd40322a` | Robust, balanced and clustered provisional prototypes |
| 4.3 | `df49d671cb3728172c9fce2fc4de029d1adeb21d` | Descriptive reference-quality diagnostics |
| 4.4 | `0a428d5ad3f5008e162f0a9a94da3dcbfede3797` | Nonparametric provisional candidate ranking |

Embedding rows persist admission mode/policy, identity basis, provisional state,
human-review status and quality flags. Review-only evidence changes alter artifact
identity but not the content-and-model vector cache key.

Robust prototypes record provisional, human-verified and outlier counts,
dispersion, method and seed. Diagnostics persist centroid, leave-one-out, nearest
same/competitor, margin, influence and route evidence while explicitly stating
that taxonomic misidentification is not assessed.

Provisional ranking exposes raw prototype and reference similarities, top-k
evidence, competitor margin, compatibility, candidate rank and required review.
`probability_available` is false and calibrated probability is null.

These are fixture-tested contracts, not a live production run or a measured
claim of accuracy, speed, calibration or saved review work.
