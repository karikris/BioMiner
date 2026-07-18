# Task 4.1 completion — candidate strategy schedules

Task 4.1 is complete. Its three required subtask commits were pushed directly
to `origin/main` through `b372ce18d6be62c1b66025b700d5c4e4a884428c`;
that exact remote SHA was verified at 2026-07-18T04:35:28Z.

BioMiner now implements geography-first, safe family-first, and parallel
family/geography union schedules over the existing immutable
`family_geo_candidate_sets` complete union. These are strategy-plan artifacts,
not competing candidate truth layers. Every source accepted taxon appears once
in every plan with its original row fingerprint and all active inclusion axes.

Geography-first schedules available regional evidence before required safety,
family expansion, and the complete remainder. No-geo inputs retain their
target and global safety evidence. Safe family-first uses family evidence only
to prioritize or partition work; a deliberately wrong-family target moves to
the required-safety stage but cannot disappear. Parallel union combines family,
geography, query, visual, target, and safety axes independently, deduplicating
only at accepted-taxon grain.

The strategy and target-preservation gate passed 82 tests. The full regression
passed 2,756 tests in 102.92 seconds, and repository-wide Ruff passed. A bounded
five-candidate fixture wrote and reloaded all three plans. Membership was
identical, target and complete-union preservation held, and the aggregate
fingerprint was
`sha256:b59f4f883e1359fe9a7ea8faa5a189771daa06a14562619164fff2a274ba8b9e`.

All four required GitHits calls timed out. They are recorded as unavailable;
no external solution, code, prose, or architecture is claimed. Task 4.1 was
determined by BioMiner's accepted family-as-index-not-gate ADR and committed
complete-union contracts.

No strategy is selected or made production default. Task 4.2 must measure
candidate recall, family recall, work, memory, reuse, no-geo behavior, and the
hard-family-pruning counterfactual using configured evaluation evidence. This
task does not claim live accuracy, performance, taxonomic identity, human
verification, statistical support, occurrence release, or deployment.
