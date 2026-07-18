# Dynamic global/local pool scoring

Status: bounded cached-vector fixture execution completed.

Fourteen encoder-free scoring work items cover seven cases under a global-only
control and dynamic global/local pooling. The production raw-component,
disagreement, four-method fusion, ranking, bounded matrix-batch, and validation
code produced 168 case/strategy/pool/method result rows. Candidate schedules
share identical membership, so each score result is reused across all three
schedules rather than recomputed.

This run uses deterministic eight-dimensional fixture vectors. It did not load
BioCLIP, decode an image, or invoke an encoder. The historical committed
BioCLIP manifest remains real prior execution evidence, but is not counted as
this run. The equal linear weights are fixture parameters, not weights fitted
on reviewed validation data.

## Reuse evidence

- Seven query vectors served 14 work items: seven observed reuse events.
- Family matrix cache: 14 requests, 13 hits, one materialization.
- Candidate matrix cache: 14 requests, seven hits, seven materializations.
- Pool matrix cache: 100 requests, 65 hits, 35 materializations.
- One bounded batch referenced 100 pool matrices but held 35 unique matrices,
  recording 65 within-batch reuse events.
- Peak batch pool storage was 2,240 bytes, below the preregistered 512 MiB cap.
- Encoder invocations and image materializations: zero.

Avoided encoder seconds, matrix seconds, cost, and energy remain unavailable
because they were not instrumented.

## Global/local comparison

The 72 located case/strategy/method pairs made local components available. Raw
target values changed in 36 pairs; the fixture top candidate did not change.
All 12 no-geography pairs exactly matched the global-only fallback in numeric
target score and top identity. Missing geography therefore remains an explicit
unavailable local-evidence state, not a zero score or absence claim.

Every fixture expected target ranked first under every raw fusion method. That
is a deterministic software-fixture property, not reviewed precision, recall,
or biological accuracy. Scores remain raw and uncalibrated; human review,
statistical support, production default selection, and occurrence release all
remain unavailable.
