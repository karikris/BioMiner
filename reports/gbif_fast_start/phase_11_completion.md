# Phase 11 completion — testing and regression protection

Phase 11 is complete. All four planned task commits and one acceptance repair
were pushed to `main`. The final full repository run passed 2,515 tests in
98.57 seconds, and the expanded targeted suite passed 164 tests in 26.02
seconds.

| Task | Commit | Result |
|---|---|---|
| 11.1 | `883780a1841de80622bcdaf5dfacb92c29c28f91` | Thirteen explicit automated-admission and human-override cases exercise public production APIs |
| 11.2 | `23eb833c3fad0c2df74a5e6711fd7823bbfe2a12` | Ten readiness states cover strict, provisional, blocked, stale, mode-change, audit, route and rejection behavior |
| 11.3 | `2e807546c0d4cebe50c18f1a2b8df63aa9877d3d` | Ten lifecycle guarantees connect admission, audit, targeted review, bank revision, reuse, selective rescoring and final export |
| 11.4 | `ddc6e3ea036a953f1a9f7875feb54543204b532c` | A measured baseline guards first-score time, zero pre-score review, embedding reuse, selective rerun ratio and peak traced memory |

The lifecycle test proves that provider-asserted GBIF references can support
initial provisional scoring without reference review, while strict mode keeps
the prior human-verification requirement. A statistical audit flags only the
underperforming species; only its references enter targeted review; one bad
reference is excluded; reusable embeddings remain cached; and unrelated
Flickr scores are not recomputed.

Scientific evidence boundaries remain fail-closed. A provisional raw margin
is not a probability. It cannot authorize final release. An unreviewed Flickr
candidate may be scored for triage, but the final exporter rejects the entire
dataset if even one record has not completed source-bound human review.

The performance guard uses five-sample medians against a separately recorded
seven-sample baseline. Review count, reuse count, reuse ratio and selective
rerun ratio are exact correctness invariants. Only host-sensitive elapsed time
and traced memory receive documented relative-plus-absolute tolerances. The
baseline environment is Linux x86_64 under WSL2, Python 3.14.5 and Polars
1.41.2.

The first full regression at `ddc6e3e` passed 2,513 tests and exposed one
concurrency defect: a readiness publisher scheduled after the winning writer
could raise `FileExistsError` before recording its failure audit. Commit
`3962db990944bb03e2627472ba5bcb6a8ec7224d` moved that check inside the
existing audited transaction. The three publication tests then passed ten
consecutive runs, the readiness module passed 46 tests, and the final full
suite passed 2,515 tests.

These are deterministic fixture-backed contracts. They do not claim a live
Papilio demoleus workflow, production runtime or memory improvement, saved
human reviews, or improved model quality.
