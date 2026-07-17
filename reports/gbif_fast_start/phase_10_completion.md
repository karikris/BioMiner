# Phase 10 completion — observability and reporting

Phase 10 is complete. All four task commits were pushed and the final full
repository run passed 2,490 tests in 95.24 seconds.

| Task | Commit | Result |
|---|---|---|
| 10.1 | `77c19cf478e6ff7efeb31029a7b9d729aa4c0a37` | Ten source-bound admission stages derive measured counts from exact media identities and preserve unavailable states |
| 10.2 | `66de5f5e6202be533bb704b39a131d5c72bd718d` | Nine efficiency dimensions retain evidence status, source fingerprint and unit; unlike work units are never summed |
| 10.3 | `3b51b86715aec6750fd185421998ee660eb59a10` | Six immutable maturity labels separate provider support, human evidence, raw scores, probabilities and release status |
| 10.4 | `83f25328c6b44b33d9c97ece1658fb622b2c73ea` | Seven blocker kinds expose selective resume actions, release impact, source evidence and human-input requirements |

The admission funnel covers candidates, downloads, decoded images,
deduplicated images, YOLOE routing, provisional admission, human verification,
exclusion, flags and later review. Each measured count is derived from sorted
unique media IDs. Missing stage evidence remains unavailable rather than zero,
and branching review outcomes are not forced into a false monotonic funnel.

Efficiency rows distinguish direct measurement, derivation, estimation,
fixtures and missing instrumentation. Time to first score, review work,
embedding and detector reuse, prototype rebuilds, selective rescoring,
full-rerun avoidance and peak memory remain source-bound. Full-rerun avoidance
may use more than one unit, but different units are never aggregated.

Every adaptive report now embeds the canonical maturity legend. Provider-
asserted provisional support is not human verified; human-verified reference
support is not a Flickr label; a raw score is not a probability; a calibrated
probability cannot authorize release; and final release status requires the
human-review rules to pass.

Blocker reports cover failed downloads, retryable media, invalid routes, stale
bank artifacts, incomplete audit samples, pending targeted review and pending
selective reruns. Each has a fixed resume action. Human-input blockers cannot
be automatically cleared, and selective retry preserves unrelated completed
work.

These are fixture-tested observability contracts. No production speedup,
memory reduction, review saving, quality gain or blocker clearance is claimed.
