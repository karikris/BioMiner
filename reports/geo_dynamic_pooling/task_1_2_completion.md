# Task 1.2 completion — downstream handoff schemas

Task 1.2 is complete. Its three required subtask commits and one mechanical
formatter support commit were pushed directly to `origin/main` through
`e1d12a73ff1e0b98e40a513f37b6330bb25a4aa6`; that exact remote SHA was
verified at 2026-07-18T02:40:12Z.

BioMiner now has deterministic, fail-closed TaxaLens and ButterflyLens product
manifests. Each pins producer and consumer commits, requires complete
role-specific artifact declarations, separates semantic from physical identity,
and preserves explicit unavailable states and evidence maturity. Raw scores are
not probabilities, zero reviews are not zero-valued quality, candidate-only map
cells are not occurrences, and missing baseline is not biological absence.

TaxaLens remains pinned to `c5e87ead4fdb26d5c5624bbb8d8d67e46d8eddbc`.
ButterflyLens advances from `fcee1a76886e37cb2f0d9badbe91b70a18a0e7c3`
to `1cea643623f2f20a2bea72afc754c7b194db3278` after an explicit compatibility
review. Its wire schemas remain compatible while its downstream adapter now
owns stricter repeated independent assignment, blind disclosure, authenticated
append-only review submission, reviewer identity, service-role writes, grants,
and RLS enforcement.

The cross-repository gate passed 33 tests, including reads from exact committed
sibling objects. The full regression at the formatted implementation SHA passed
2,660 tests in 100.67 seconds. Ruff format/lint, JSON and JSONL validation,
required trailers, generated-artifact inspection, secret-prefix inspection and
remote verification passed. All four required GitHits calls timed out and are
recorded as unavailable; no external solution, code or prose is claimed.

This task implements contracts and compatibility evidence only. It does not
claim a live product import, database mutation, reviewer assignment, human
review, calibration, performance superiority, occurrence release, publication,
or production deployment.
