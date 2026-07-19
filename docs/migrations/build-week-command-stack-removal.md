# Build Week command-stack removal

Date: 2026-07-19

BioMiner removed the remaining executable Build Week benchmark/report stack:

- `biominer dev vision prototype-smoke[-five]`;
- `biominer dev vision prototype-build-embeddings`;
- `biominer dev vision prototype-staged-flickr`;
- `biominer dev vision prototype-benchmark-matrix`;
- `biominer dev vision prototype-select-policy`;
- `biominer bioclip prototype-evidence`; and
- the standalone historical reference-workflow baseline builder.

These commands formed a self-contained one-species experiment sequence. Their
modules were imported only by CLI forwarding functions and tests dedicated to
the same wrappers. They did not produce inputs consumed by the adaptive `run`
graph or the current `references` command implementations. Retaining them made
a completed July 2026 experiment appear to be a second supported runtime.

The removal does not delete the current reference acquisition, deduplication,
QA, freeze, full-frame embedding, prototype, dynamic-score, or evaluation
contracts. Those remain available through `biominer references` and the
adaptive orchestration boundary. Independent BioCLIP and YOLOE runtime checks
also remain under `biominer dev vision`.

Frozen reports, manifests, pilot configuration, and Git history remain as
historical evidence. Their four config/report-only regression modules were
also removed: they validated static Phase 14/15 files but no current parser,
schema validator, or production transformation. Commands recorded inside the
immutable artifacts are no longer parseable and must not be replayed as a
current production workflow. There is no compatibility fallback; callers must
use the current artifact owners and explicit human/release gates.
