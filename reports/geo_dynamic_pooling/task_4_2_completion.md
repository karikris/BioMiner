# Task 4.2 completion — evidence-bound candidate strategy evaluation

Task 4.2 is complete. Its three required subtask commits were pushed directly
to `origin/main` through `4735050bd1a8206d44a22d6032cfbbfa75b40635`;
that exact remote SHA was verified at 2026-07-18T04:57:54Z.

BioMiner now emits immutable per-plan/per-k strategy metrics for target,
reviewed-species and family candidate recall; candidate and evaluated-set size;
dot products; reference members; elapsed time; peak memory; cache reuse; and
explicit no-geography and wrong-family slices. Resource metrics must come from
supplied instrumentation. Missing or inconsistent work, time, memory, or cache
rows fail validation rather than being estimated.

The hard-family-pruning counterfactual is separate from production membership.
It counts a loss only when the reviewed correct species exists in the complete
union but would be absent from a hypothetical family-priority-only pool.
Reviewed species already missing from the complete union are reported
separately and excluded from the family-pruning loss denominator.

The intended candidate remains `parallel_family_geography_union`, but selection
is fail-closed. The configured gate requires all three strategies on identical
labels; overall, no-geo and wrong-family recall; resource and cache limits;
recall non-inferiority; a sufficient counterfactual denominator; and non-fixture
evidence when configured. Passing the gate makes a candidate eligible for the
next phase but does not mutate the production default or establish universal
superiority.

The deterministic two-label fixture produced 18 metric rows over k=1, 2, and
5. At k=1, family-first-safe lost its deliberately wrong-family correct species;
geography-first and parallel union retained it. The counterfactual recorded one
correct-species loss among two eligible labels (50%), including one of one in
the no-geo slice. These are implementation-fixture values, not live performance
estimates. The selection gate failed exactly `non_fixture_evidence`, so no
strategy was selected or made production-default eligible. The ablation report
fingerprint is
`sha256:73b47e70b182ac1fccc8118b95048228476c9f6d726a5b78d7e7d584286f3148`.

The strategy evaluation gate passed 93 tests. The full regression passed 2,766
tests in 104.06 seconds, and repository-wide Ruff passed. All four required
Task 4.2 provenance entries record `skipped_user_directive` with null solution
IDs. No GitHits call was made and no external repository, result, code, prose,
or architectural contribution is claimed.

Task 4.2 does not claim live accuracy or efficiency, empirical superiority,
production strategy selection, taxonomic identity, dynamic-pool completion,
calibration, statistical support, human verification, occurrence release, or
deployment. Phase 5 can now implement the dynamic reference-pool policy and
planner against this fail-closed evaluation boundary.
