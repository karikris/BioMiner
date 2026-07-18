# Tools, GitHits, Valyu, MCP, and skills

## Discovery order

Use the smallest trustworthy source that answers the question:

1. Current local code, tests, schemas, manifests, and phase reports.
2. Accepted repository ADRs.
3. Official primary documentation or literature.
4. GitHits implementation precedents.
5. Secondary summaries only when primary material is unavailable.

Do not browse or load entire repositories when a focused symbol, file, test, or
artifact answers the question.

## GitHits MCP

Use GitHits for non-trivial:

- architecture and implementation patterns;
- OSS dependency internals;
- Polars, Arrow, DuckDB, H3, PostgreSQL, S3, MPS, and worker designs;
- retry, rate-limit, lease, checkpoint, and idempotency patterns;
- reference-bank, embedding, calibration, review, and evaluation designs;
- tests, CI, packaging, and performance instrumentation.

### Required task workflow

Before implementation:

1. Inspect BioMiner's current implementation and tests.
2. Form a narrow GitHits query around the unknown pattern.
3. Review at least two approaches when practical.
4. Record:
   - query;
   - repositories or files reviewed;
   - pattern adopted;
   - alternatives rejected;
   - licence implications;
   - why the pattern fits BioMiner.
5. Append the evidence to `provenance/githits.jsonl` when the current goal
   requires it.

A suitable record shape is:

```json
{
  "task_id": "goal-task-id",
  "timestamp": "ISO-8601",
  "queries": ["focused query"],
  "repositories_reviewed": ["owner/repo"],
  "patterns_adopted": ["..."],
  "patterns_rejected": ["..."],
  "reason": "...",
  "license_notes": ["..."],
  "githits_status": "used"
}
```

If GitHits is unavailable:

- record one failed attempt;
- do not repeatedly call an unavailable service;
- continue with local code and primary documentation when safe;
- mark the status `unavailable`;
- never fabricate repositories, results, or solution IDs.

### Active dynamic-pooling goal override (2026-07-18)

The user explicitly disabled all further GitHits calls for the remainder of
the active geography-conditioned dynamic-pooling goal after repeated service
unavailability. Do not call GitHits again during this goal, including before a
task or subtask. Where the goal requires a provenance record, append one with
`githits_status: "skipped_user_directive"`, `solution_id: null`, and an explicit
statement that no call was made. Do not invent repositories, solutions,
results, or external contributions; use committed local evidence, official
primary documentation, and primary literature where needed.

GitHits is precedent, not authority. Do not paste code wholesale. Verify the
source licence and adapt the pattern to BioMiner's contracts.

## Valyu MCP

Use Valyu when the task depends on current or externally authoritative facts:

- API specifications and rate limits;
- provider terms and licences;
- GBIF, CoL, ALA, iNaturalist, EOL, Wikidata, Flickr, or Backblaze behavior;
- BioCLIP, YOLOE, OpenCLIP, PyTorch, MPS, CUDA, and model documentation;
- primary research papers;
- Darwin Core and biodiversity standards;
- provenance and source discovery.

### Valyu rules

- Prefer official documentation and primary literature.
- Record source, URL/identifier, publication or retrieval date, and relevant
  version.
- Verify mutable facts again when implementing.
- Do not let a Valyu result directly define:
  - taxonomic identity;
  - query eligibility;
  - provider trust;
  - reference admission;
  - model decision;
  - review outcome;
  - release output.
- External evidence must pass the repository's adapter, normalization, schema,
  rights, and QA contracts.
- Do not use Valyu from production pipeline code.

## MCP boundaries

- MCP tools are developer assistance only.
- Keep credentials in environment variables or approved secret stores.
- Never place credentials in prompts, tracked files, logs, reports, manifests,
  screenshots, or examples.
- Treat MCP output as untrusted until checked against primary sources or local
  code.
- Do not add Valyu, GitHits, or another MCP as a runtime dependency.
- Do not use MCP output as a hidden input to a scientific artifact.

## Morph

Morph is prohibited unless the user explicitly re-enables it for a task. The
uploaded historical file contained contradictory instructions about Morph; the
prohibition is the resolved default.

## Skills

Before using an installed skill:

1. Read its `SKILL.md`.
2. Follow the task-specific workflow.
3. Confirm it does not conflict with the active goal's branch, commit, push, or
   scientific rules.
4. Use the skill only for its intended scope.

Examples:

- brainstorming or design skill for architecture alternatives;
- test-driven-development skill for new behavior;
- systematic-debugging skill for failures;
- GitHub/CI skills for repository and workflow inspection;
- artifact skills for reports, PDFs, spreadsheets, or presentations;
- Headroom skill for large logs or outputs.

Explicit user and active-goal instructions override a skill's default branching
or publishing behavior.

Do not invoke a skill or MCP that is unavailable. Record the limitation once
and use a focused fallback.

## Headroom and large outputs

When installed, use Headroom for large repetitive logs, reports, model outputs,
or test traces:

- preserve the compression hash;
- reason from the compressed result;
- retrieve exact omitted details with a narrow query;
- do not repeatedly reload the full source.

If unavailable, use:

```text
rg
sed with a narrow range
pytest -q on focused files
git diff --stat
git diff -- path
jq on selected fields
DuckDB or Polars summaries
```

## Token discipline

- Read symbols and nearby tests, not whole directories.
- Do not paste raw Parquet, API responses, model vectors, or long logs.
- Do not repeat progress output already written to a report.
- During long runs, use structured logs, manifests, PIDs, leases, and
  checkpoints rather than repeated polling.
- Summarize evidence and include exact paths, hashes, and failing cases.
