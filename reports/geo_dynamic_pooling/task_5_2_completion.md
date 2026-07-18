# Task 5.2 completion — bounded cached uncertainty expansion

Task 5.2 is complete. Its three required implementation commits were pushed
directly to `origin/main` through
`d6e03450bd301bba2ae3ea1e6ffcadb05059d8f9`; that exact remote SHA was
verified at 2026-07-18T05:59:11Z.

BioMiner now represents expansion evidence as a first-class immutable
artifact. Eleven configured raw signals cover family/species ambiguity,
global/local and prototype/input disagreement, local support, subject area,
known competitors, no-geo fallback, out-of-distribution evidence and
route/domain compatibility. Every signal is observed or has an exact
unavailable reason. Missing evidence never triggers expansion on its own, and
the uncalibrated threshold policy is versioned and fingerprinted.

Expansion selects identities from the committed reference geography index. It
accepts no encoder or vector payload, retains the query embedding fingerprint
and all prior memberships, and creates an immutable replacement plan. Cache
reuse evidence records the retained and added embedding fingerprints, zero
encoder invocations and no vector materialization.

Every round is intersected with maximum rounds, stage capacity, total pool
capacity, configured candidate priority plus mandatory target/safety
eligibility, and a per-candidate addition increment. Applied rounds stop at a
mandatory rescore boundary. Clear signals, maximum rounds, stage or total
budget exhaustion, and absence of additional cached references each produce
an explicit stopped decision. No decision authorizes release.

The deterministic fixture expanded one two-member plan to five members,
retaining two round-zero identities and adding three round-one identities. It
recorded one `small_species_margin` trigger, one cache-reuse row with zero
encoder calls, and one `round_complete_rescore_required` decision. The JSON
report records all artifact fingerprints and adversarial stopping cases. This
is implementation evidence, not live scientific or performance evidence.

The expansion/cache gate passed 117 tests. The full regression passed 2,816
tests in 106.31 seconds, and repository-wide Ruff passed. All four Task 5.2
provenance entries record `skipped_user_directive` with null solution IDs. No
GitHits call was made and no external code, result or architecture is claimed.

No live Flickr score or reference pool was expanded, no calibrated probability
or statistical support was produced, no candidate strategy was selected, and
no human verification, occurrence release or deployment claim is made. Phase
6 Task 6.1 can now make detection and one-time full-frame embedding boundaries
explicit.
