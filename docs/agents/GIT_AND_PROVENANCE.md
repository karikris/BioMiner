# Git, commits, active sessions, and provenance

## Determine the current Git policy

Do not assume `main`, a feature branch, or a push cadence.

Read the explicit current goal. It may require:

- a named working branch;
- direct work on `main`;
- one commit per task or subtask;
- push after every task or after every phase;
- no merge;
- no history rewrite.

The explicit goal wins.

## Start of every task

Run:

```bash
git status --short
git branch --show-current
git rev-parse HEAD
git log -5 --oneline
git diff --stat
git diff --cached --stat
```

Record the starting SHA and branch in the task report.

If remote synchronization is required, do it only when:

- the goal authorizes it;
- the worktree is clean;
- no active session owns uncommitted work.

Never pull, rebase, merge, or switch branches merely to “get current” while
another session is active.

## Dirty-worktree protocol

A dirty worktree is evidence, not clutter.

1. Inspect changed paths and diffs.
2. Identify the active goal/task and process.
3. Check phase reports, GitHits records, logs, checkpoints, and recent commits.
4. Determine whether requested files overlap.
5. Work only in non-overlapping files when safe.
6. Stop and report unavoidable overlap.

Never:

```text
git reset --hard
git restore .
git checkout -- .
git clean -fd
git stash another session's work
```

Do not stage with `git add -A` or `git add .` unless every changed file belongs
to the current task and was reviewed.

## Commit boundaries

Default when the goal is silent:

- one coherent numbered task per commit;
- no mixed refactor, feature, generated data, and documentation commit;
- targeted tests before the commit;
- phase-level full regression before phase completion;
- push at the phase boundary.

When the goal specifies a different boundary, follow it.

Use focused conventional subjects such as:

```text
feat(references): ...
fix(readiness): ...
perf(bioclip): ...
test(run): ...
docs(architecture): ...
chore(release): ...
```

The historical `step0:`–`step4:` style is obsolete for the current staged
architecture.

## Commit trailers

Use the exact trailers required by the current authorized work. A suitable
pattern is:

```text
AI-Assistance: OpenAI Codex
AI-Primary-Model: exact-model-id
AI-Reasoning-Effort: exact-value
AI-Session: exact-session-id
Scope-Status: new | modified-existing
GitHits-Log: provenance/githits.jsonl#task-id
Human-Decision: concise decision
Human-Reviewed-By: Kris Kari
Tests: exact commands and results
```

Add origin repository/SHA when adapting external or sibling-repository work.

Never invent:

- a model;
- session ID;
- human review;
- test count;
- benchmark;
- source SHA;
- push result.

Do not use an AI model as Git co-author unless the user explicitly requires it.

## Before committing

Run the focused verification required by the task, then:

```bash
git diff --check
git status --short
git diff --stat
git diff --cached --stat
```

Inspect the complete staged diff.

Confirm no staged:

- `.env` or secrets;
- raw API payloads;
- downloaded media;
- model weights;
- caches;
- generated registries;
- large runtime Parquet;
- local databases;
- temporary logs;
- unreviewed unrelated changes.

## Push rules

- Push only at the boundary required by the goal.
- Do not force-push.
- Do not amend pushed commits.
- Do not rewrite history.
- Verify the remote SHA after pushing.
- If a push is rejected, do not bypass it. Report divergence and preserve local
  commits.
- Do not merge automatically unless explicitly directed.

## Active process and artifact safety

Git commits do not prove a live job has stopped.

Before changing output or workstore code:

- inspect running processes;
- inspect leases and worker IDs;
- inspect current output prefixes;
- inspect manifests and checkpoint state;
- avoid deleting or rewriting active artifacts.

A code task must not invalidate a running job silently. If schema or fingerprint
changes make active output stale, document the migration or stop condition.

## Provenance files

Use the current goal's provenance structure. Common locations include:

```text
provenance/githits.jsonl
docs/architecture/
run manifests
artifact manifests
stage completion JSON/Markdown
```

Every phase report should include:

- starting and ending SHA;
- task commits;
- branch and push SHAs;
- schemas or policies changed;
- exact test commands/results;
- artifacts and fingerprints;
- live steps not executed;
- human input still required;
- claims allowed and blocked.

## Completion checklist

A Git task is complete only when:

- active work was preserved;
- the correct branch was used;
- commit boundary matches the goal;
- tests and checks are recorded accurately;
- staged content is clean;
- provenance is updated;
- push or non-push state is explicit;
- limitations remain visible.
