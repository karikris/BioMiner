# Alternate workflow and Build Week runtime removal

BioMiner removed the `run --workflow legacy`, `run --workflow reference-first`,
and `build_week_target_aware_prototype` runtime surfaces on 2026-07-19.

The Build Week mode was a local, one-species prototype wrapper around pinned
pilot artifacts. Its configuration and metadata-qualified permit duplicated
checks now owned by the versioned adaptive reference, full-frame embedding,
scoring, review, and release contracts. The alternate workflow plans also let
callers select obsolete crop/cascade semantics beside the adaptive graph.

`biominer run` now plans only `ADAPTIVE_REFERENCE_PRODUCTION_STAGES`. Operators
may still select a bounded subset with `--stages`; `all` means the adaptive
graph. Historical Build Week reports and research documentation remain in Git
as evidence of the earlier experiment, not as executable production inputs.

There is no fallback. A removed mode or workflow argument is rejected by the
CLI. Current target-preserving work uses the adaptive full-frame contracts and
retains its explicit human-review and scientific-release gates.
