# Task 8.1 completion — dynamic compute batching

Status: completed and pushed to `origin/main` through
`a6d90e8b11a8b1f7439ed1707ef89d9670a8232d`.

## Delivered

- Dynamic scoring now consumes only a validated persisted full-frame embedding
  and immutable vector matrices. Its API accepts no encoder, image, path or
  pixel input, and its result records zero encoder invocations and zero image
  materializations.
- Optional MPS-aware image batching derives a safe capacity from explicit
  allocator snapshots, configured target utilization and estimated bytes per
  image. Missing telemetry uses a clearly labeled fixed-batch path.
- Recognized memory errors shrink and retry only the uncommitted failed slice.
  Successful slices are never repeated; minimum-size and non-memory errors
  propagate.
- Pool scoring work is grouped deterministically under hard work-count,
  unique-matrix and unique Float32-byte ceilings. Shared matrices count once
  within a batch, while telemetry separates within-batch reuse from cross-batch
  reloads.
- Batch validation re-derives policy, working sets, orders, metrics and all
  semantic fingerprints without repeating matrix scoring. Canonical scientific
  result order remains independent of locality-oriented execution order.

## Measured gate

- MPS, memory, matrix and worker fixture suite: 92 passed in 0.98 seconds.
- Full regression: 2,908 passed in 110.43 seconds.
- Repository-wide Ruff lint: passed.
- Changed-file Ruff format check: passed. Whole-tree formatting remains an
  unconfigured baseline affecting 330 pre-existing files.
- Provenance: 156 JSONL records; all four Task 8.1 records state
  `skipped_user_directive`, `solution_id: null`, and no GitHits call.
- Remote verification: `origin/main` resolved to `a6d90e8b…` after push.

The memory-headroom fixture encoded five images as batches `[2, 2, 1]`. The
bounded-retry fixture attempted `[1,2,3,4]`, retried its failed slice as
`[1,2]`, and then encoded `[3,4]` and `[5]`, with no successful image repeated.
The matrix fixture grouped three work items into batches `[2,1]`, counted nine
matrix references over three unique matrices (seven rows, 56 Float32 bytes),
recorded three within-batch reuses and three cross-batch reloads, and performed
no encoder or image work.

## Claim boundary

The MPS snapshots and vectors are deterministic fakes, not live accelerator or
throughput evidence. Task 8.1 changes scheduling and observability, not score
semantics or production defaults. It does not establish memory savings,
scientific performance, taxonomy, human review, occurrence release or
deployment. Work-avoided instrumentation begins in Task 8.2.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
