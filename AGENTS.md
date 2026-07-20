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

|Task|Document|
|-|-|
|New session, active goal, architecture status, stale-doc warnings|[CURRENT\_STATE.md](docs/agents/CURRENT_STATE.md)|
|Git, commits, branches, pushes, provenance|[GIT\_AND\_PROVENANCE.md](docs/agents/GIT_AND_PROVENANCE.md)|
|GitHits, Valyu, MCP, skills, OSS dependency research|[TOOLS\_AND\_SKILLS.md](docs/agents/TOOLS_AND_SKILLS.md)|
|Registry, Flickr, geography, references, YOLOE, BioCLIP, review|[SCIENCE\_AND\_PIPELINE.md](docs/agents/SCIENCE_AND_PIPELINE.md)|
|Parquet, S3, PostgreSQL, queues, caches, workers, performance|[DATA\_STORAGE\_AND\_PERFORMANCE.md](docs/agents/DATA_STORAGE_AND_PERFORMANCE.md)|
|Tests, live checks, phase completion, release|[TESTING\_AND\_RELEASE.md](docs/agents/TESTING_AND_RELEASE.md)|
|Plan/report format|[TASK\_TEMPLATE.md](docs/agents/TASK_TEMPLATE.md)|

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

The active production baseline is the raw three-member GBIF DWCA intake. Treat
[`CURRENT\_STATE.md`](docs/agents/CURRENT_STATE.md) and the
[GBIF ground-zero pipeline](docs/PIPELINE_GROUND_ZERO.md) as the authoritative
scope: no earlier staging run, report, registry, Flickr corpus, model output,
or handoff may be resumed or cited as current evidence.

## Core scientific invariants

* Registry artifacts and source snapshots define taxonomic identity.
* Search, metadata, comments, provider labels, geography, YOLOE, and BioCLIP
are evidence—not truth.
* Deduplicate processing, never discovery provenance.
* Use `GBIF provider-asserted provisional support` for automatically admitted
unreviewed GBIF references; never call it verified or ground truth.
* Unreviewed Flickr remains candidate evidence. Final inclusion requires
source-bound decisive human review and all release gates.
* YOLOE routes and measures quality; it does not classify species.
* BioCLIP stays frozen unless an accepted goal explicitly changes that policy.
* Raw similarities, distances, detector scores, margins, and SVM outputs are
not probabilities.
* Geography prioritizes evidence; missing source evidence is not biological
absence.
* Keep adult, larval, pupal, specimen, artifact, and incompatible domains
separate.
* Target-aware modes retain their candidate and full-frame contracts; never
silently substitute legacy hierarchy pruning or crops.
* Downstream handoffs are immutable, versioned artifacts. Pin exact TaxaLens
and ButterflyLens commits; never consume a dirty sibling worktree or move a
compatibility pin silently.
* Keep candidate evidence, model evidence, human review, quality estimates,
and release-ready occurrence-candidate maturity separate. Review alone is
not occurrence release, and unavailable or unrun evidence is not false or
zero evidence.
* Missing evidence remains explicit; release fails closed.

Current direction: raw GBIF DWCA fingerprinting and validation → source-bound
taxonomic/name enrichment → species-first Flickr discovery → provenance-safe
media intake → YOLOE route optimization → hierarchical BioCLIP evidence →
separate review and release gates. The raw DWCA, its fingerprints, and newly
produced manifests define the only valid production lineage.

## Task workflow

1. **Discover:** Start with the adhd skill. Inspect repository status, task evidence,
code, tests, ADRs, manifests, generated files, and active-file ownership. Use
githits for every non-trivial task when available and permitted by the active
goal. Use valyu for statistics, probability, and current authoritative external
facts.
2. **Plan and Generate Tasks:** Use Superpowers writing-plans and using-superpowers
to define invariants, schemas, failure modes, compatibility, tests, cleanup,
rollback, and stop conditions, then decompose the strategy into concrete,
independently verifiable tasks.
3. **Implement:** Make the smallest complete change. Use Superpowers test-driven-development,
executing-plans, and subagent-driven-development. Add tests with behaviour,
preserve compatibility where required, and remove old code - if tests
fail do not bring old code back - refine the solution instead.
4. **Observe and Verify:** Verify every task with repository evidence and use githits
precedent where applicable. Use verification-before-completion, systematic-debugging,
requesting-code-review, and receiving-code-review. Run focused tests first,
broader checks at the required boundary, inspect git diff --check, and
validate generated files.
5. **Re-plan and Iterate:** Use verified results to correct failures, update the plan,
or advance to the next task until all acceptance criteria and stop conditions are
met. Use writing-skills to document decisions, evidence, and outcomes.
6. **Commit and Report:** Follow the active branch, commit to main, push, provenance,
and reporting rules exactly. Never fabricate tests, reviews, tool output, external
evidence, or completion status.

## Engineering defaults

* Python `>=3.14`; Polars, Parquet, DuckDB; small JSON only for configuration,
manifests, checkpoints, and reports.
* Production defaults: S3-compatible storage and PostgreSQL work state.
Filesystem/SQLite are explicit local modes.
* Work is bounded, resumable, idempotent, observable, and fingerprinted.
* Coordinators merge, sort, deduplicate, commit, and publish; manifests last.
* Delete temporary media only after verified durable commit.
* Never commit secrets, source media, model weights, raw API dumps, caches,
generated registries, databases, or large runtime artifacts.

## Tool and Git summary

* GitHits: OSS patterns and dependency internals when available and permitted;
follow active-goal overrides in `TOOLS\_AND\_SKILLS.md`.
* Valyu: current official docs, primary literature, APIs, terms, provenance.
* Skills: read the relevant `SKILL.md` before use.
* Headroom: compress large outputs when available.
* Morph: prohibited unless the user explicitly re-enables it.
* MCP/skills are developer assistance only, never production dependencies.
* Determine branch/commit/push policy from the current goal; do not assume.
* One coherent commit per required task boundary; no force-push or history
rewrite; record exact tests, SHAs, and limitations.

## Done

Complete only when active work is preserved, implementation and migration are
complete, required checks pass, failures/unavailable evidence remain visible,
schemas/fingerprints/provenance are updated, no runtime secret/artifact is
staged, required commit/push evidence exists, and remaining human/live work is
reported.
