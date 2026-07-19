# Dynamic-pooling production-default decision

Decision: **insufficient evidence; runtime defaults unchanged**.

The frozen policy evaluated all nine selection criteria across the complete
24-variant table. Zero variants are eligible. No candidate strategy, pool
variant, fusion method, or selection-evidence fingerprint is set.

Three software/fixture gates pass: embedding and matrix reuse were observed,
the complete candidate union has no target-pruning regression, and the report
contains no unsupported statistical claim. Six criteria block selection:

- target/candidate recall is fixture structural evidence, not reviewed accuracy;
- reviewed precision and its lower confidence bound are unavailable;
- family and geographic subgroup estimates are unavailable;
- zero effective real reviews leaves the full 86-review shortfall;
- comparable runtime/throughput was not instrumented; and
- MPS peak memory was not measured.

This is not a rejection of measured production performance. The required
performance evidence does not exist yet.

The current and resulting settings fingerprints are identical:
`sha256:0fd197b2650a79d99970cada3dcbabe9980c5a265d9d71f929bbcf6f51e13e7d`.
Both settings states are `unselected`; candidate strategy, pool variant, and
fusion method remain null. The safe reference-pool policy fingerprint remains
`sha256:08a5983f4e3c9d92894b5bcca2fbb18dd7a6d74114fdc90523ad29fde654cdc5`.

Raw scores remain non-probabilistic. Missing geography is not biological
absence. Planned review work is not completed review. No production default or
occurrence release is authorized.
