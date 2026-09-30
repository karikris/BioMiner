# Monthly backend dependency upgrades implementation plan

> **For agentic workers:** Use superpowers:subagent-driven-development for the independent security regression and review tasks. Execute dependency resolution and publication sequentially on main.

**Goal:** Upgrade BioMiner's backend dependencies after reviewing their current release, security, and source evidence through GitHits; report residual OpenSSL exposure and TaxaLens/ButterflyLens tooling findings separately.

**Architecture:** Keep dependency selection in `pyproject.toml` and the reproducible graph in `uv.lock`. Verify in a private environment under `/tmp/biominer-dependencies-20261001-k3kxy0ga`, preserve the shared environment and ongoing GBIF work, and keep external research out of production code.

**Tech Stack:** Python 3.14, uv, pytest, GitHits, public upstream release and advisory evidence.

---

Starting branch/SHA: `main` / `0ba1784f673888ceff8da81c4eb3b6652011dbc4`.

Task ID: `backend-dependencies-20261001`.

Ownership: `AGENTS.md`, GBIF EDA/report scripts and their tests, and `veritaxa_codex_goal.md` were already dirty. Do not stage or change them. The instruction to first commit dirty main conflicts with the prohibition on committing another session's work; preserve that work and commit only this task's reviewed paths. The historical GitHits opt-out concerns the previous dynamic-pooling goal; the current explicit request authorizes GitHits. No branch, reset, merge, rebase, or active production job is part of this task.

Scientific invariants: frozen BioCLIP, immutable handoffs and consumer pins, source lineage and schemas, non-probability score semantics, and fail-closed review/release gates remain in force. Tests use fixtures, no source media, weights, live scientific providers, or production outputs.

### Task 1: Establish upgrade evidence

Files: read `pyproject.toml`, `uv.lock`, dependency call sites and tests; produce `docs/research/backend_dependency_review_2026-10-01.md` and one task-specific `provenance/githits.jsonl` record.

- [x] Review latest package health with GitHits `pkg_info` for direct and changed transitive packages.
- [x] Batch `pkg_upgrade_review` across the resolved changes and query `pkg_changelog`; check exact installed versions against current advisories.
- [x] Resolve missing changelogs and compatibility questions using `search`, explicit `search_status` continuations, `code_grep`, and bounded `read`. Record stale or unavailable evidence without treating it as an all-clear.
- [x] Inspect vendored OpenSSL versions and current official advisories. Audit the two frontend lockfiles without changing sibling files or handoff pins.

### Task 2: Add security behavior regressions

Files: create `tests/test_dependency_security.py`.

- [x] Exercise the public AnyIO TLS handshake with a Unicode hostname and a local capture stream; require IDNA 2008 SNI rather than the ambiguous IDNA 2003 hostname.
- [x] Exercise lxml's default parser with a fixture external parameter entity; require rejection of the external entity.
- [x] Exercise skops loading of a fixture archive declaring a future protocol; require fail-closed rejection. Keep this optional dependency test skippable in a base installation.
- [x] Run against the original environment and record the real failures before upgrading; then run against the candidate environment and require success.

### Task 3: Upgrade the reviewed graph

Files: modify `pyproject.toml`, regenerate `uv.lock`, create `.python-version`.

- [x] Raise direct dependency lower bounds to reviewed versions; add an AnyIO security floor so fresh unlocked installs cannot select the affected release.
- [x] Add the lxml security floor and pin CPython 3.14.7 after verifying the private Astral 20260929 build bundles OpenSSL 3.5.9. Raise `requires-python` to `>=3.14.7`; a custom interpreter still requires a separate OpenSSL inventory.
- [x] Move skops to `>=0.16,<0.17` after checking protocol/type-trust changes. Review the build-backend lower bound separately from installed runtime versions.
- [x] Resolve all extras with `uv lock --upgrade`; review every changed or newly added transitive package before installation.
- [x] Sync only the private environment with `UV_PROJECT_ENVIRONMENT=/tmp/biominer-dependencies-20261001-k3kxy0ga/venv uv sync --locked --all-extras --python /tmp/biominer-dependencies-20261001-k3kxy0ga/openssl/private-python/cpython-3.14.7-linux-x86_64-gnu/bin/python3.14`.

### Task 4: Verify and publish

- [x] Run focused security, HTTP, storage/Parquet, classifier persistence/training, and report tests, then the full offline suite. The first full run failed only on the absent DuckDB 1.5.6 Spatial extension; install/load its signed version, inspect its native inventory, and rerun: 2731 passed, 18 warnings.
- [x] Run `uv lock --check`, dependency consistency, package build and CLI smoke checks, and a fresh audit of the exact locked requirements. Distinguish registry audit coverage from vendored/native exposure.
- [x] Independently review security-test specification, dependency implementation and final staged narrative/provenance; final review passed with no substantive findings.
- [x] Validate final provenance JSONL, lock hashes, report evidence, and `git diff --check` for this task's files. Exactly one current-task record, all dependency fingerprints, the final JUnit receipt, document links and both sibling lock hashes were verified.
- [x] Stage only the nine owned paths and inspect the complete task diff; validate all lock artifact hashes and references. No secrets/runtime artifacts or unrelated files were staged.

Publication boundary: make one coherent commit on main, push without rewriting history, and verify remote SHA. Exact local/remote commit identifiers and the observed push result belong in the completion response. Record any rejection rather than merge another session's work.

Rollback: revert only this task's coherent commit if necessary; do not restore or delete another session's files. The baseline manifests are saved under the private temporary root. Existing immutable artifacts and the shared `.venv` are not migrated by tests.

Stop conditions: unavoidable file ownership overlap, an unsupported candidate with no safe alternative, a production contract regression that cannot be repaired within scope, or remote divergence requiring a merge. Evidence gaps and residual native advisories remain explicit even when package upgrades pass tests.
