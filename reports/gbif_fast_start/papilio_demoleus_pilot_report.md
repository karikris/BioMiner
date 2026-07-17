# Papilio demoleus adaptive GBIF pilot

The fixture-backed adaptive path reached its first provisional score in a
measured 502.480375 ms with zero reference reviews before scoring. One
provisional reference was admitted in the integration fixture. The remediation
fixture reused two embeddings and selectively rescored one affected Flickr
record.

| Metric | Value | Evidence |
|---|---:|---|
| Time to first score | 502.480375 ms | Measured fixture |
| References admitted | 1 | Fixture |
| Reference reviews before first score | 0 | Fixture |
| Human-reviewed Flickr labels | 0 | Workspace evidence |
| Species legitimately flagged | 0 | Unavailable pending labels |
| References later reviewed | 0 | Unavailable pending labels |
| Embeddings reused | 2 | Fixture |
| Flickr records selectively rescored | 1 | Fixture |

Quality metrics are unavailable because no human-reviewed Flickr label artifact
exists. A representative 50-record review queue is ready; until those decisions
are complete, no live species flag, reference remediation or before/after
quality claim is permitted.

Earlier prototype-only manifests record 81 provider-supported references, 81
embeddings, 26 prototypes, 13,496 classified Flickr records and 81 reused
embeddings on resume. Their local artifacts are absent, so these values are
historical context rather than current adaptive results.

Strict mode rejects provider-only support and therefore requires reference
verification before scoring. Both modes retain source-bound human review for
final Flickr release, and provisional readiness never authorizes scientific
release.

No live quality improvement, production speedup, memory saving or human-review
saving is claimed.
