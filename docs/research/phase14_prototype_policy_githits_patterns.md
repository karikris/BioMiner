# Phase 14 prototype-policy open-source implementation check

Task 14.5 used GitHits solution
`348946ee-80cf-43d4-85c7-df03df773aeb`. The result distilled MIT-licensed
selective-classification examples from several repositories.

The useful patterns were:

- keep frozen embeddings separate from lightweight policy selection;
- define abstention directly on a raw top-1/top-2 margin;
- keep model-selection, calibration, and final-test use explicit;
- persist the complete selected-policy identity and threshold provenance;
- prove that final-test changes cannot alter the selected policy.

BioMiner adapts those patterns with stricter evidence semantics. No threshold
or calibrator is fitted from provider-supported labels. The selected policy is
therefore `prototype_uncalibrated`, the `0.10` raw-margin threshold is
predeclared rather than optimized, calibration rows are used only for a
coverage audit, and final-test rows are not read for selection.
