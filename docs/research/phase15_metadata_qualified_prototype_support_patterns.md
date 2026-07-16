# Phase 15 metadata-qualified prototype support implementation check

Task 15.2 attempted to query GitHits, but the GitHits MCP server was not
exposed in the active tool session. No new solution ID is claimed.

The implementation therefore reuses the fail-closed opt-in pattern recorded
for Task 15.1 in GitHits solution
`ad0cbd0a-7015-48ec-8a8e-c94ae8031813` and BioMiner's existing prototype
freeze and embedding validation contracts.

The narrow prototype permit:

- accepts zero independently human-verified labels;
- accepts a missing probability calibrator;
- requires explicit `prototype` and `prototype_uncalibrated` status;
- requires SHA-256-pinned local artifacts;
- validates licensing, attribution, image eligibility, target identity,
  duplicate removal, route separation, embedding identity, classifier
  fingerprint, target scoreability, no hierarchy pruning, and
  non-probability score semantics;
- remains unavailable to every non-prototype classification mode.

This is an execution permit for experimental screening evidence. It is not
scientific readiness, calibrated accuracy, or taxonomic validation.
