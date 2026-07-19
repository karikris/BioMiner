# Dynamic-pool plan CLI removal

Status: command-surface simplification.

The seven plan-only `biominer dynamic-pooling` subcommands were removed on
2026-07-19. They never executed production work. Their input bindings were too
small to construct several closed artifacts, and at least one output declared
JSONL while the owning validated contract writes Parquet. Persisting those
plans created an appearance of runtime readiness without a valid execution
boundary.

Use `biominer run --dry-run` for the canonical stage plan and typed runtime
settings. Use `biominer references` for concrete reference acquisition,
validation, embedding, prototype, and audit operations. Product handoffs remain
separate immutable integration commands and cannot authorize occurrence
release.

Previously persisted dynamic command plans are non-authoritative diagnostics.
Do not reinterpret them as execution receipts or synthesize their intended
outputs. Preserve them with the prior Git revision if they are needed for
audit.
