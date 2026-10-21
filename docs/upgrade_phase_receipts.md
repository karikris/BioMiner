# Upgrade phase receipts, schema 1

`scripts/check_upgrade_phase.py` implements the gates in the approved
[six-phase plan](superpowers/plans/2026-10-21-security-polars-arrow.md).
It reads Git and executes reviewed commands, but never stages, commits,
pushes, changes branches, or modifies production workloads.

The operator manifest is reviewed input. This tool checks evidence consistency
and specific result semantics; it is not a hostile-command sandbox, a signed
attestation service, or proof of reviewer identity. Source scope and command
selection still need independent review. Commands and logs must contain no
credentials. Use the environment for provider access.

## Amended compatibility scope

The request without the cybersecurity component uses a separate external
`compatibility_phase.py` developer-assistance operator, SHA-256
`44ff68a2ef6b7542cadde614c07610f4dec60624696d9c465903b43af1859ee5`.
Its `biominer-upgrade-compatibility-phase/v1` receipts cover the remaining
candidate compatibility phases 4–6. The existing validator and admission
policies remain unchanged. See the [phase-4 report](research/upgrade_phase_4_2026-10-21.md)
for exact manifests, independent reviews and commands.

Publication requires all actual checks and digest-bound independent reviews.
A distinct documentation-only summary commit, matching local/remote `main`,
and create-only pushed receipt precede fresh continuation. These receipts
keep security passage, legacy phase passage, runtime admission, registration,
production readiness and production migration false. They provide no
production fallback qualification.

## Workflow

1. Complete task commits `N.1`, `N.2`, and `N.3` on `main`. Preserve unrelated
   work. `tested_commit` is exactly task `N.3`; all task SHAs must exist,
   differ, and form an ancestral chain after `base_commit`.
2. Prepare an external JSON manifest and independently authored review files.
   Bind source/configuration files to their committed bytes. Use a clean,
   isolated checkout for full-suite execution when main contains other work.
3. Run the fixed checks, preserving output in a new directory outside Git:

   ```sh
   python scripts/check_upgrade_phase.py run /external/plan.json --output /external/run-1
   python scripts/check_upgrade_phase.py validate /external/run-1/phase.json
   ```

4. Commit only the final phase summary and provenance. Between `tested_commit`
   and the final commit, only `docs/**/*.md`, `docs/**/*.json`, and
   `provenance/githits.jsonl` may change. Push through the normal coordinator
   workflow. The final commit is deliberately separate from the tested task
   commit, avoiding a self-referential receipt hash.
5. Observe the pushed commit and create the immutable handshake outside Git:

   ```sh
   python scripts/check_upgrade_phase.py finalize /external/run-1/phase.json \
     --final-commit FULL_SHA --output /external/phase-1-pushed.json
   python scripts/check_upgrade_phase.py continue /external/phase-1-pushed.json --phase 2
   ```

`finalize` requires local `main` HEAD and a fresh
`git ls-remote --exit-code origin refs/heads/main` observation to equal the final
commit. `continue` verifies every predecessor's manifests, committed sources,
artifacts, command logs and review hashes, and requires the immediate
predecessor to remain local and remote main. The next manifest references that
handshake and uses its final commit as its base. During a new phase's run, remote
main must still equal its predecessor; earlier receipts remain checked by
hash and Git ancestry once remote main advances.

Outputs use exclusive creation and cannot be overwritten by the tool. Failed
commands retain their exit code and output, but produce no passing candidate.
Credential-bearing output is rejected before writing. This is a bounded
pattern/environment-value check, not a universal secret detector. Keep the
external evidence directory: moving, deleting or changing referenced files
invalidates continuation. Historical validation checks committed source bytes
without requiring an old working checkout to remain at its historical commit.

## Manifest

All fields below are required. All file references are absolute
`{"path": "/external/file", "sha256": "<64 lowercase hex>"}` objects. All Git
commits use full 40-character lowercase hashes.

| Field | Contract |
|---|---|
| `schema_version`, `phase` | `1`; integer `1` through `6` |
| `repo` | Absolute path to the coordinator's Git checkout on `main` |
| `base_commit`, `tested_commit` | Start commit and final implementation-task commit |
| `tasks` | Exact mapping `{"N.1": SHA, "N.2": SHA, "N.3": SHA}` |
| `sources` | Repository-relative path → SHA-256 of committed tested bytes |
| `artifacts` | Nonempty list of hashed external evidence references |
| `implementer` | Nonempty implementation identity |
| `reviews` | Exactly `spec_review` and `quality_review`, each a JSON file reference |
| `checks` | Exact fixed check-name mapping from the table below |
| `blockers` | Explicit empty list required for advancement |
| `previous` | `null` for phase 1; prior pushed handshake reference otherwise |

Source hashes must cover every changed non-documentation task file and, when
tracked, `pyproject.toml`, `uv.lock`, `.python-version`, and this validator.
Independent review should include unchanged dependencies/configuration needed
by the selected tests. Every declared source must match both Git and the current
execution checkout. Deleted files remain recorded by the task commits.

A check is `{"argv": ["python", "-m", "pytest", "-q"], "checker": "pytest",
"receipt": FILE_REFERENCE}`. The result reference is omitted in a run plan.
Optional `cwd` selects an absolute execution checkout; optional
`timeout_seconds` defaults to 3600. Commands run as argv without a shell.
Full-suite/baseline commands accept only `-q`, `-ra`, `--strict-config`,
`--strict-markers`, and `--color=no`; selectors are forbidden. Their checkout
must contain no tracked code differences or untracked/ignored Python,
configuration, shell, or lock files. This includes pytest configuration and
`conftest.py`; keep environments outside the isolated execution checkout.
`PYTEST_ADDOPTS` is cleared by the runner.

The runner writes a command receipt with `schema_version: 1`, `check`, exact
`argv`, absolute `cwd`, integer `exit_code`, `tested_commit`, `source_digest`,
and hashed `log`. The source digest is SHA-256 of the `sources` object encoded
as UTF-8 JSON with sorted keys and separators `(",", ":")`. Exit code must be
zero. A pytest check requires a standard terminal summary with at least one
passed test and no failures, errors, skips, xfails, xpasses or deselections;
warnings may remain visible. Arbitrary success strings are rejected.

A review receipt contains `schema_version: 1`, `kind`, `reviewer`,
`implementer`, exact `task_commits`, `source_digest`, `verdict: "approved"`,
`findings: []`, and `report: FILE_REFERENCE` for the actual nonempty review
text. Each reviewer must differ from the implementer. A file hash binds the
review's exact contents; it does not turn an automated claim into human review.

## Fixed checks and structured reports

| Phase | Required checks |
|---|---|
| 1 | `inventory`, `baseline`, `security`, `phase_gate_tests` |
| 2 | `runtime_snapshot`, `migration`, `native_verify`, `security` |
| 3 | `polars_semantics`, `polars_resources`, `full_suite`, `security` |
| 4 | `arrow_build`, `arrow_api`, `native_verify`, `security` |
| 5 | `compatibility_matrix`, `live_postgres`, `live_s3`, `live_cleanup`, `full_suite`, `benchmark`, `security` |
| 6 | `canary`, `migration`, `recovery`, `retirement`, `native_verify`, `security` |

`baseline`, `phase_gate_tests`, `polars_semantics`, `polars_resources`,
`full_suite`, `arrow_api`, `canary`, and `recovery` use the `pytest` checker.
Every other check uses `report`: the command's combined stdout/stderr must be
one JSON object with `schema_version: 1`, exact `kind`, and nonempty hashed
`evidence`. Additional required fields follow:

| Kind | Required report fields |
|---|---|
| `inventory` | Nonempty `entries`; each has `id`, `owner`, and hashed `runtime` |
| `security` | Nonempty `scopes`, explicit `findings` and `coverage_gaps`; each entry has a unique stable `id` and declared `scope`. Phase 1 may record open findings/gaps. Later phases retain every predecessor ID/scope and require hashed resolution `evidence` with status `resolved` or `not_applicable`. Phase 2 additionally permits `contained` for findings, never coverage gaps. Developer/MCP tools outside BioMiner are not runtime scopes. |
| `runtime_snapshot` | `self_contained: true` plus hashed `launcher`, `configuration`, `lock`, `verifier`, `runtime` |
| `migration` | Nonnegative integer `eligible`, `migrated`, `blocked`; eligible equals migrated, blocked is zero. Explicit `remaining_exposures`, each with owner, `status: "contained"` and hashed evidence; phase 6 requires none. |
| `native_verify` | Pinned `runtime_profile` reference; `cases` exactly `artifacts`, `loaded_versions`, `tls`, `import_order`, `features` |
| `arrow_build` | `compiler_jobs` 1–4; custom `26.0.0+biominer.*` version; hashed `source`, `recipe`, `wheel`, `native_receipt`; `features` includes `s3`, `flight`, `encryption` |
| `compatibility_matrix` | Exactly four `combinations`: `p1a25`, `p1a26`, `p2a25`, `p2a26`. Each has a distinct pinned `runtime_profile`, hashed command `receipt`, hashed identical `fixture`, `logical_differences: []`, hashed `comparison`. |
| `live_postgres` | `execution: "live"`, task-owned `namespace` beginning `biominer_upgrade_`; cases exactly `authenticated_tls`, `transactions`, `rollback`, `crud`, `leases`, `idempotency` |
| `live_s3` | Same live/namespace contract; cases exactly `parquet_roundtrip`, `ranges`, `listing`, `retries`, `manifest_last`, `scoped_delete` |
| `live_cleanup` | `namespaces` exactly `postgres` and `s3`; each has the matching live `namespace`, integer `remaining: 0`, and hashed `inspection` |
| `benchmark` | Exactly three `samples`, each containing `baseline` and `target` positive finite `wall_seconds` and `peak_rss_bytes`, plus hashed `evidence`. Both target medians must be at most 1.15× baseline. Empty `budget_violations`, `resume_losses`, `provenance_losses` required. |
| `retirement` | Empty `affected_entrypoints` and `unsafe_fallbacks`; hashed `entrypoint_inventory` |

Every item in `cases` is `{"receipt": FILE_REFERENCE}`. That nested command
receipt has the same fields as a top-level command receipt, plus
`runtime_profile: FILE_REFERENCE`. Its `check` is `REPORT_KIND.CASE_NAME`;
its tested commit, source digest and cwd must match the enclosing manifest/check.
Its command must invoke pytest and its log must meet the strict summary
contract. A reused log without the exact case/source/runtime-bound receipt
cannot satisfy a named native/live check.

For `native_verify`, `live_postgres`, and `live_s3`, pin `runtime_profile` in
the operator's check entry and the produced report. For `compatibility_matrix`,
pin `runtime_profiles` in the check entry: a map from the four combination
names to four distinct hashed profiles. Each combination and nested receipt
must match its pinned profile. Report producers must retain real observations;
fixture-only results cannot be relabelled live.
