# Phase 15 Build Week prototype-mode open-source implementation check

Task 15.1 used GitHits solution
`ad0cbd0a-7015-48ec-8a8e-c94ae8031813`.

The useful implementation patterns were:

- make experimental behavior an explicit opt-in mode;
- require a versioned configuration manifest instead of inferring intent;
- preserve the existing default;
- reject missing, unknown, or conflicting configuration;
- persist deployment status and limitations with the output;
- fail closed instead of silently falling back to another classifier.

BioMiner applies those patterns with
`build_week_target_aware_prototype`. The mode is local-only for the current
Build Week demonstration, requires the frozen Papilio demoleus prototype
configuration and its SHA-256-pinned artifacts, and records `prototype` in
the run plan. It cannot dispatch through the legacy object-scoring stages.

This task does not relax scientific-readiness or calibration checks. That
execution policy remains the separately reviewable scope of Task 15.2.
