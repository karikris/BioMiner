# AGENTS.md — BioMiner

## Scope and precedence

Applies to the whole repository. Nested `AGENTS.md` files may add stricter
rules but must not weaken scientific, provenance, security, testing, or
active-goal safeguards.

Precedence:

1. Explicit current user goal.
2. Nearest `AGENTS.md`.
3. Accepted ADRs, versioned schemas, manifests, and tests.
4. Current implementation.
5. General docs.
6. Least destructive reversible choice.

Record material conflicts; never resolve them silently.

## Load instructions selectively

Read only what the task needs:

| Task | Document |
|---|---|
| New session, active goal, architecture status, stale-doc warnings | [CURRENT_STATE.md](docs/agents/CURRENT_STATE.md) |
| Git, commits, branches, pushes, provenance | [GIT_AND_PROVENANCE.md](docs/agents/GIT_AND_PROVENANCE.md) |
| GitHits, Valyu, MCP, skills, OSS dependency research | [TOOLS_AND_SKILLS.md](docs/agents/TOOLS_AND_SKILLS.md) |
| Registry, Flickr, geography, references, YOLOE, BioCLIP, review | [SCIENCE_AND_PIPELINE.md](docs/agents/SCIENCE_AND_PIPELINE.md) |
| Parquet, S3, PostgreSQL, queues, caches, workers, performance | [DATA_STORAGE_AND_PERFORMANCE.md](docs/agents/DATA_STORAGE_AND_PERFORMANCE.md) |
| Tests, live checks, phase completion, release | [TESTING_AND_RELEASE.md](docs/agents/TESTING_AND_RELEASE.md) |
| Plan/report format | [TASK_TEMPLATE.md](docs/agents/TASK_TEMPLATE.md) |

Index: [docs/agents/README.md](docs/agents/README.md).

## Active-goal safety

Another Codex session may be running. Before each task:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
```

If dirty, inspect ownership and current reports. Never reset, restore, clean,
rebase, merge, switch branches, mass-format, stage, commit, amend, or delete
another session's work. Avoid active output roots and do not start duplicate
Flickr, reference, YOLOE, BioCLIP, evaluation, or publication jobs. Stop on
unavoidable overlap.

For the active multi-phase geography-conditioned pooling goal, treat
[`CURRENT_STATE.md`](docs/agents/CURRENT_STATE.md) as the resumable handoff
ledger. Read it before resuming numbered work, update it at task boundaries,
and verify every recorded SHA and gate against local Git before relying on it.

## Core scientific invariants

- Registry artifacts and source snapshots define taxonomic identity.
- Search, metadata, comments, provider labels, geography, YOLOE, and BioCLIP
  are evidence—not truth.
- Deduplicate processing, never discovery provenance.
- Use `GBIF provider-asserted provisional support` for automatically admitted
  unreviewed GBIF references; never call it verified or ground truth.
- Unreviewed Flickr remains candidate evidence. Final inclusion requires
  source-bound decisive human review and all release gates.
- YOLOE routes and measures quality; it does not classify species.
- BioCLIP stays frozen unless an accepted goal explicitly changes that policy.
- Raw similarities, distances, detector scores, margins, and SVM outputs are
  not probabilities.
- Geography prioritizes evidence; missing source evidence is not biological
  absence.
- Keep adult, larval, pupal, specimen, artifact, and incompatible domains
  separate.
- Target-aware modes retain their candidate and full-frame contracts; never
  silently substitute legacy hierarchy pruning or crops.
- Downstream handoffs are immutable, versioned artifacts. Pin exact TaxaLens
  and ButterflyLens commits; never consume a dirty sibling worktree or move a
  compatibility pin silently.
- Keep candidate evidence, model evidence, human review, quality estimates,
  and release-ready occurrence-candidate maturity separate. Review alone is
  not occurrence release, and unavailable or unrun evidence is not false or
  zero evidence.
- Missing evidence remains explicit; release fails closed.

Current direction: adaptive GBIF fast-start → provisional scoring → mandatory
Flickr review → statistical audit → targeted remediation → selective rerun.
Legacy cascade/crop/bucket paths are removal targets, not compatibility or
adaptive authority. Flickr comment promotion, the disconnected dynamic-pooling
plan CLI, and the synthetic cascade benchmark have already been removed; do not
restore or wrap them. Preserve historical artifacts through Git and migration
documentation rather than callable fallbacks.

Current implementation and release evidence is summarized in
`reports/gbif_fast_start/final_report.json` and `final_report.md`; detailed gate
results are in `adaptive_release_verification.json` and `.md`. These reports
verify software and fixture behavior only. Their listed live-source steps and
human review remain required before scientific release claims.

The completed geography-conditioned dynamic-pooling software goal is summarized
in `reports/geo_dynamic_pooling/final_report.json` and `.md`; its technical and
scientific-semantics receipts remain separate. The production-selection outcome
is `insufficient_evidence`, not a selected default or scientific release.

## Task workflow

1. **Discover:** inspect status, current task evidence, code, tests, ADRs, and
   manifests; use GitHits for non-trivial precedent only when available and
   permitted by the active goal, and Valyu for current authoritative external
   facts.
2. **Plan:** define invariants, schemas, failures, compatibility, tests,
   cleanup, rollback, and active-file ownership.
3. **Implement:** smallest complete change; tests with behavior; remove old code
   only with migration coverage.
4. **Verify:** focused tests first; broader checks at the required boundary;
   inspect `git diff --check` and generated files.
5. **Commit/report:** follow the explicit goal's branch, commit, push, and
   provenance rules. Never fabricate evidence.

## Engineering defaults

- Python `>=3.14`; Polars, Parquet, DuckDB; small JSON only for configuration,
  manifests, checkpoints, and reports.
- Production defaults: S3-compatible storage and PostgreSQL work state.
  Filesystem/SQLite are explicit local modes.
- Work is bounded, resumable, idempotent, observable, and fingerprinted.
- Coordinators merge, sort, deduplicate, commit, and publish; manifests last.
- Delete temporary media only after verified durable commit.
- Never commit secrets, source media, model weights, raw API dumps, caches,
  generated registries, databases, or large runtime artifacts.

## Tool and Git summary

- GitHits: OSS patterns and dependency internals when available and permitted;
  follow active-goal overrides in `TOOLS_AND_SKILLS.md`.
- Valyu: current official docs, primary literature, APIs, terms, provenance.
- Skills: read the relevant `SKILL.md` before use.
- Headroom: compress large outputs when available.
- Morph: prohibited unless the user explicitly re-enables it.
- MCP/skills are developer assistance only, never production dependencies.
- Determine branch/commit/push policy from the current goal; do not assume.
- One coherent commit per required task boundary; no force-push or history
  rewrite; record exact tests, SHAs, and limitations.

## Done

Complete only when active work is preserved, implementation and migration are
complete, required checks pass, failures/unavailable evidence remain visible,
schemas/fingerprints/provenance are updated, no runtime secret/artifact is
staged, required commit/push evidence exists, and remaining human/live work is
reported.
