# Adaptive stage vocabulary cutover

Date: 2026-07-19

`RunStage` now contains exactly the 31 stages in
`ADAPTIVE_REFERENCE_PRODUCTION_STAGES`. Six labels that were never part of that
graph were removed:

- `reference_review`;
- `reference_readiness`;
- `classifier_training`;
- `classifier_calibration`;
- `target_aware_scoring`; and
- `evaluation`.

Concrete `biominer references` commands are application operations, not
independent production stages. Their option resolver no longer imports
`RunStage`, command specifications no longer carry a redundant stage field,
and reference-command dry-run JSON identifies the command without inventing a
production-stage membership. Support-preflight diagnostics now name the real
commands operators can execute.

The only automatic-completion-protected stages are the two human gates in the
adaptive graph: `flickr_human_verification` and `targeted_reference_review`.
Historical manifests containing removed enum values are intentionally rejected
by the current manifest loader; they remain recoverable from their originating
Git revision. There is no alias or fallback translation into the adaptive
graph.

GitHits was not called under the user's explicit directive. Provenance records
`githits_status: skipped_user_directive` and `solution_id: null`.
