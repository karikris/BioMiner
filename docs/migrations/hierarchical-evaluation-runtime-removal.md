# Hierarchical evaluation runtime removal

Date: 2026-07-19

The generic hierarchical-classifier evaluation surface was removed after the
classification-v3 cascade and crop runtime ceased to exist. The deleted cluster
comprised:

- `biominer evaluation classify` and `biominer evaluation review-queue`;
- family/species top-k metrics, confusion reports, charts, visual QA, and the
  Xie-style hierarchical adapter;
- the heuristic vision-bucket threshold policy and hierarchical review queue;
- path-cascade row exceptions in calibration utilities; and
- synthetic/golden tests that generated only retired classifier rows.

The current evaluation workflow is unchanged. `biominer evaluation
build-sampling-frame` creates target-aware reviewed-evidence samples, while
`biominer references evaluate-target-verifier` owns calibrated target
verification. Leakage-safe splits, holdouts, calibration diagnostics,
uncertainty, statistical support, dynamic-pool audits, review evidence, and
release gates remain in production.

Classification is no longer a runtime mode selector. Target-aware scoring owns
one immutable `target_aware_few_shot_classification` identity, exposed as a
read-only property on its plan and result. Historical hierarchical outputs and
their implementation remain recoverable from Git; no compatibility flag,
alias, parser route, or row adapter remains callable.

GitHits was not called under the user's explicit directive. Provenance records
`githits_status: skipped_user_directive` and `solution_id: null`.
