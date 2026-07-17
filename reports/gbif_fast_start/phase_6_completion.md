# Phase 6 completion — mandatory Flickr verification

Phase 6 is complete. All four task commits were pushed and the final full
repository run passed 2,414 tests in 87.23 seconds.

| Task | Commit | Result |
|---|---|---|
| 6.1 | `5f38036c1ba4016e8b15ae9e4543ec244c676069` | Complete, source-hash-bound human release gate |
| 6.2 | `d40a600a461da1f15cd0cf63f72387650d63657c` | Scored candidates remain separate from release eligibility |
| 6.3 | `c2ff7cc722f8ebd8700db2bac86366c219198856` | Four provenance-preserving verification campaigns |
| 6.4 | `2643bf3c1caffe2f68ba837d99064a0c99192c7c` | Fail-closed atomic final Parquet export |

A final Flickr occurrence must have decisive human review bound to the current
source image hash, resolved duplication, supported target identity, suitable
domain and life stage, valid coordinate/date evidence, completed required second
review or adjudication, and release-policy permission.

Unreviewed candidates may retain route, embedding identity, candidate ranking,
provisional margin and review priority, but their release state is always
excluded. Four Polars/Parquet campaigns preserve inclusion probability, sampling
stratum, owner/observation/duplicate groups, geography, query tier, score band
and candidate competitors.

The final writer validates the complete frame before publishing. Unreviewed,
Skip, Can’t view, uncertain, conflicted, stale-hash, unsupported-claim or
otherwise ineligible rows block the entire export; valid rows are not silently
published as a partial batch.

Two initial full-suite runs exposed a pre-existing SQLite context lifecycle leak
under Python 3.14. Commit `096380ad86324102dee67e3f73cc71ce21798036`
preserved transaction semantics and added
deterministic close-on-exit. The subsequent full run passed cleanly.

These are fixture-tested software gates, not evidence that live human reviews or
released occurrence records are scientifically correct, and not a measured
claim of accuracy, speed, cost or reviewer-work savings.
