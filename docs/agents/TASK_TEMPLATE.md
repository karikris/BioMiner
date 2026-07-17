# BioMiner task template

Use this template for a numbered task. Keep it concise and artifact-bound.

## Plan

```text
Task ID:
Objective:
Starting branch:
Starting SHA:
Active-goal overlap:
Relevant ADR/contracts:
Files expected:
GitHits query/status:
Valyu sources/status:
Skills:
Scientific invariants:
Compatibility requirements:
Failure states:
Tests:
Live/human inputs:
Commit boundary:
Push boundary:
Rollback:
```

## Implementation checklist

- [ ] Read root `AGENTS.md`.
- [ ] Read `CURRENT_STATE.md` and `GIT_AND_PROVENANCE.md`.
- [ ] Read only relevant topic documents.
- [ ] Inspect dirty worktree and active process state.
- [ ] Inspect current code, tests, schemas, reports, and ADRs.
- [ ] Record GitHits/Valyu evidence when required.
- [ ] Define or update tests with behavior.
- [ ] Implement the smallest complete change.
- [ ] Preserve legacy compatibility only where explicitly required.
- [ ] Preserve fingerprints and evidence maturity.
- [ ] Run focused tests.
- [ ] Run broader task checks.
- [ ] Run `git diff --check`.
- [ ] Inspect staged files for secrets/runtime artifacts.
- [ ] Commit and push at the required boundary.
- [ ] Record exact results and limitations.

## Completion report

```text
Task:
Status:
Starting branch/SHA:
Ending branch/SHA:
Remote SHA:
Active-goal files preserved:
Primary Codex model:
Reasoning effort:
Codex session:
GitHits:
Valyu:
Skills:
Files changed:
Contracts/schemas changed:
Policies/fingerprints changed:
Migrations:
Tests run:
Test results:
Live tests:
Human review:
Artifacts:
Performance evidence:
Provenance updates:
Scientific claims allowed:
Scientific claims blocked:
Known limitations:
Unexecuted work:
Next safe task:
```

Never report planned, fixture-only, simulated, unreviewed, unavailable, or
blocked work as live completion.
