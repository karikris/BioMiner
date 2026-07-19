# Reference-first work facade removal

Date: 2026-07-19

BioMiner removed `biominer.run.reference_work` and its package-level exports.
The module defined a second queue vocabulary, payload wrapper, lease wrapper,
and enqueue/claim facade for the retired `reference-first` workflow. Exact
symbol searches found no production caller after the alternate workflow
selector was removed; its only consumer was its own test module.

The generic SQLite/PostgreSQL `WorkStore`, claim fencing, stale-claim recovery,
and adaptive orchestrator remain current. Concrete applications may enqueue
their own versioned payloads through that shared workstore rather than routing
them through an unused ten-kind facade.

The duplicated `REFERENCE_FIRST_ARTIFACT_KEYS` subset was also removed. The
canonical artifact map and directory-key set already validate every local and
cloud path. `reference-first-run-artifacts-v1.0.0` remains the immutable
on-wire layout version for existing and current manifests; the historical
label is not a selectable workflow or fallback and is not reinterpreted.

The same audit removed five path entries with no remaining producer or
consumer: generic vision-stage metrics/summary, the superseded hierarchical
review queue and visual-QA findings, and the deleted cascade object-evidence
join. Current reference review, dynamic-pool review, target-aware scores, and
workflow reports have distinct declared paths.
