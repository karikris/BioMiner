# Phase 0 completion — baseline and architecture decision

Status: complete<br>
Evidence head: `a39b05c8ca843ae368c29aaa6ab3ebe0e459111b`<br>
Generated: `2026-07-17T10:28:00Z`

## Task ledger

| Task | Immutable commit | Result |
|---|---|---|
| `gbif-fast-0.1` | `7ca83acb7d0ada7fd026bd4c7f343939c6b72293` | Strict workflow, environment, lockfile and test baseline recorded; 2264 full-suite and 46 focused tests passed. |
| `gbif-fast-0.2` | `247b42f3206d48bb79e2dbf97c5a92e4f207ae71` | Three-mode adaptive-admission ADR, 24 gates and scientific boundaries recorded; 48 focused tests passed. |
| `gbif-fast-0.3` | `a39b05c8ca843ae368c29aaa6ab3ebe0e459111b` | Deterministic benchmark contract and reports added; 18 focused tests passed. |

Each numbered task is present on `origin/main` and has a corresponding entry in
`provenance/githits.jsonl`.

## Performance evidence available at Phase 0

- 93 reference media rows selected and 81 frozen as provisional support.
- 81 references await human review; zero completed manual reviews were recorded.
- 81 reference embeddings were reused and zero recomputed on resume.
- One persistent BioCLIP load, six model-cache hits and 1,765,261,312 bytes peak RSS were recorded.
- Prototype scoring recorded 13,496 Flickr records and 634,312 candidate-score rows at 2.274524 records per second.

The last group is labelled prototype-only proxy evidence. Strict manual-review wait,
strict time to embeddings, strict time to prototypes and strict time to first Flickr
score were not measured. They remain unavailable or not instrumented, never zero.

## Acceptance

- The latest-main strict baseline is reproducible from committed evidence.
- The adaptive design, alternatives, gates, ownership and release boundaries are explicit.
- The performance schema rejects unsupported measured values and detects tampering.
- JSON, JSONL, formatting, lint and focused tests pass.

Phase 1 may begin. This completion record does not claim that adaptive policy or
readiness-contract implementation already exists.
