# Phase 14 Build Week report open-source implementation check

Task 14.6 used GitHits solution
`9ec7b692-eae4-423d-8d6f-1cd91819a8ce`. The useful open-source reporting
patterns were:

- publish paired machine-readable JSON and human-readable Markdown;
- pin dataset, configuration, model, and report provenance;
- distinguish measured performance and failure metrics from interpretation;
- report abstention and coverage separately from accuracy;
- keep limitations and failure cases prominent rather than burying them.

BioMiner applies those patterns with stricter evidence semantics. It does not
copy the example's placeholder accuracy, calibration, or memory values.
Classification accuracy and calibration error remain `null` because the
prototype reference bank has no independently reviewed taxonomic labels.
Raw-margin coverage, throughput, memory, and operational failures are reported
only where they were measured.
