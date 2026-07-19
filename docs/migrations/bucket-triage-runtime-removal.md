# Bucket triage runtime removal

Date: 2026-07-19

BioMiner removed the residual `bioclip.triage` runtime and its private
`BucketPolicy`. The function assigned Gold, Silver, Bronze, bin, or review
states from raw BioCLIP similarity, title matching, coordinates, dates, and
heuristic negative labels. After the cascade/bucket production cutover its only
caller was its dedicated test module.

Those assignments are not compatible with the adaptive evidence model: raw
similarity is not calibrated probability, title matching is not human review,
and coordinates or dates cannot authorize occurrence release. Current output
keeps raw component scores, explicit maturity, representative review,
calibration, risk controls, and downstream-owned release decisions separate.

Historical `occurrence_bin` fields and committed bucket reports remain
unchanged for audit. The generic source-record extractor may continue to emit
its non-authoritative historical default where required by that versioned
schema, but no runtime promotes or re-triages it. There is no compatibility
fallback and no row-level migration into current target-aware evidence.
