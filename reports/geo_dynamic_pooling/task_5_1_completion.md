# Task 5.1 completion — dynamic reference-pool policy and planner

Task 5.1 is complete. Its four required subtask commits were pushed directly
to `origin/main` through `b1ae26d15e6b3c866ea57c5c8a972444a4860e0d`;
that exact remote SHA was verified at 2026-07-18T05:26:10Z.

BioMiner now has one immutable `DynamicReferencePoolPolicy` covering stage
limits, global/local quotas, the total budget, class balance, geographic
fallback, observer/locality/burst diversity, uncertainty expansion, target and
safety preservation, prohibited family/geography hard pruning, a deterministic
seed, and a complete semantic fingerprint. The default policy fingerprint is
`sha256:08a5983f4e3c9d92894b5bcca2fbb18dd7a6d74114fdc90523ad29fde654cdc5`.

The planner consumes the existing family/geography candidate set, reference
geography index, and global anchor artifacts. It selects biological reference
observations rather than media volume, prefers committed global anchors and
exact workload-cluster local evidence, reuses cached embedding fingerprints,
and emits the Phase 0 immutable plan/member/summary schemas. It never copies
embedding vectors or fabricates query distance. Plan-wide balancing caps
observer and locality reuse, accepts supplied burst identities, round-robins
candidate taxa under the stage budget, and retains one validated route/domain.

Coverage reporting materializes every plan candidate, including candidates
with zero selected references. Global, local, safety, and independent-support
shortfalls remain separate, with exact fallback and local-unavailable reasons.
No-geo is an evidence-availability state and never a claim of taxon absence.

The bounded fixture produced one plan, three members (two global and one
local), two pool summaries, and one complete coverage row. Their fingerprints
are recorded in the JSON report. Counterexamples also prove balanced 3/2 class
counts under a five-member budget, observer/locality caps, one slot per burst,
route/domain isolation, zero-member shortfall materialization, and explicit
global-only no-geo fallback. These are implementation fixtures, not live
reference-coverage or performance estimates.

The pool-planner gate passed 82 tests. The full regression passed 2,796 tests
in 103.97 seconds, and repository-wide Ruff passed. All five Task 5.1
provenance entries record `skipped_user_directive` with null solution IDs. No
GitHits call was made and no external repository, solution, code, prose, result,
or architectural contribution is claimed.

No live pool was planned or scored, no candidate strategy was selected or
production-defaulted, and no claim is made about accuracy, calibration,
statistical support, taxonomic identity, human verification, occurrence
release, or deployment. Task 5.2 can now add bounded uncertainty expansion
over cached reference embeddings.
