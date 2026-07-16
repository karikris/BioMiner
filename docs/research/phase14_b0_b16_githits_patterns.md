# Phase 14 B0-B16 open-source implementation check

Task 14.4.4 used GitHits solution
`74cdbb77-db59-4026-867d-f09227acc016`, which surfaced an MIT-licensed
frozen-embedding baseline comparison in `miquelt9/accents-cat`.

The useful implementation patterns were:

- keep the embedding model frozen while comparing centroid, nearest-neighbour,
  logistic-regression, and linear-SVM baselines;
- bind fitting, model selection, calibration, and final testing to separate
  deterministic partitions;
- persist per-record predictions and aggregate metrics as durable tabular
  artifacts;
- compare raw margins and retrieval behaviour without treating uncalibrated
  scores as probabilities.

BioMiner does not copy that implementation. The local benchmark uses Polars,
the frozen BioCLIP prototype schema, the Phase 14 split manifest, explicit
target-aware candidate unions, and BioMiner's evidence-only scientific
semantics. Provider-supported labels are used only for retrieval and internal
consistency because none of the 81 records is independently human verified.
