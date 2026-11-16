# EmbeddingGemma 2 routing implementation plan

> **For agentic workers:** Use Superpowers subagent-driven-development and test-driven-development. Preserve other sessions' work; work directly on main under the explicit repository instruction.

**Goal:** Replace the executable YOLOE26 visual-routing workflows with frozen, full-frame `google/embeddinggemma-2` image/text evidence while retaining the separate BioCLIP, review, and release gates.

**Architecture:** Introduce image-level routing evidence, distinct from object detections. Persist raw cosine scores and margins with immutable model/prompt/preprocessing identities. Missing localization, masks, object counts, and subject area remain unavailable, and incompatible checkpoints fail closed.

**Tech stack:** Python >=3.14.7, Sentence Transformers >=6.1, Transformers, Pillow, Polars/Parquet, existing local developer SQLite orchestration.

## Task 1: Replace the visual routing backend and its executable workflows

- Starting branch/SHA: `main`, `3457d07feeb0526e074ae08e14d30a891c1dfb11`.
- Ownership: existing dirty AGENTS.md, GBIF reporting modules/tests, and untracked GBIF scripts belong to prior work. Do not stage or commit them. The instruction to commit dirty main first conflicts with the explicit prohibition on committing another session's work; preserve that work and commit only this task's reviewed paths.
- Baseline: the only production lineage is the raw GBIF DWCA. No production/model/discovery job or historical artifact is resumed.
- Skills: adhd preflight (concrete requested substitution; broad divergent ideation is unnecessary), using-superpowers, writing-plans, test-driven-development, subagent-driven-development, verification-before-completion, requesting-code-review/receiving-code-review, headroom, GitHits MCP.
- External evidence: official Google model documentation and model card, retrieved 2026-11-17; model `google/embeddinggemma-2`, revision `f9972ac22e696a105c482202646979e33e4b5a1e` verified through the Hugging Face model API. GitHits Sentence Transformers source evidence is an older indexed snapshot and is used only for API precedent.

### 1. Adapter and evidence contract

- [x] Add failing deterministic tests for full-frame routing, signed similarities, domain separation, ambiguous margins, invalid vectors, immutable model identity, and absent spatial evidence.
- [x] Implement a lazy optional SentenceTransformer adapter with frozen text/vision weights, audio disabled, float32 or bfloat16, normalized 768D embeddings, cached text vectors, and one embedding per image.
- [x] Integrate image-level evidence through separate bounded embedding publication and versioned image-route scoring fingerprints without manufacturing boxes, masks, objectness, or counts. Keep measured-detection schemas separate.
- [x] Verify focused adapter, detection, routing, and full-frame tests.

### 2. Developer workflows and migration

- [x] Migrate sample/sharded configuration, runtime checks, prefetch, smoke commands, and orchestrator defaults to EmbeddingGemma 2.
- [x] Bind resume state to source snapshot, immutable model revision, prompts, preprocessing, score policy, and schema; reject old/unbound output directories.
- [x] Remove the YOLOE26 runtime implementation/setup entry points. Retain historical evidence readers only where compatibility requires them, with explicit naming and semantics.
- [x] Test CLI parsing, smoke execution with fake model/runtime, resumability, shard isolation, and mismatch rejection.

### 3. Reference safety, documentation, and completion

- [x] Preserve full-frame image identities for BioCLIP; no spatial attention variants from embedding evidence.
- [x] Make missing reference subject presence/size and localization explicit so automatic reference admission cannot bypass existing quality requirements.
- [x] Update current pipeline docs and add a migration/research report with the exact roles, source URLs, contracts changed, tests, limitations, and no live performance claims.
- [x] Run focused integration tests, full regression where the available environment permits, compile/lint checks, provenance validation, and `git diff --check`.
- [x] Review specification compliance, then code quality; resolve material findings.
- [ ] Stage only task-owned files, inspect the entire staged diff, commit on main, push without force, and verify the remote SHA. Record any rejection/environment limitation honestly.

## Verification outcome

Final affected/neighbor matrix: 351 passed in 7.90s across 22 modules. Full
regression: 4,487 passed, one pre-existing registry source-label check failed
in 182.80s. Both new system-interpreter cache regressions passed in the final
focused run. The unrelated source label was preserved. Backend specification
review and final integration code-quality review approved the change after
fixes. No weights were downloaded and no live inference or production job ran.

Lock consistency, focused Ruff F checks on all touched Python modules/tests,
compilation, shell syntax, and whitespace checks passed. Commit/push SHAs are
recorded in the completion receipt rather than self-referencing this plan.

## Invariants and stop conditions

Raw similarities and margins are evidence, not probabilities. EmbeddingGemma 2 supplies coarse visual-domain evidence, never taxonomic identity; BioCLIP remains frozen and independent. Adult/larval/pupal/specimen/artifact domains remain separate. Missing or failed evidence is never a biological negative. Human review and release gates remain fail closed.

Stop on unavoidable active-file overlap, incompatible output identity, non-fast-forward push, or evidence required for scientific admission that is unavailable. A deterministic fake-model test does not establish live biological accuracy. No model weights or source media enter Git.

## Rollback

Revert only this task's commit in a later authorized change; never reset the shared worktree. Existing runtime/model artifacts remain untouched. New outputs use new fingerprints and may not reuse YOLOE26 checkpoints or routes.
