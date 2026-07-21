# Ground Zero test-suite review — 2026-07-21

## Decision

The default suite now treats executable package and CLI behavior as the source
of truth. Tests that validated deleted reports, deleted design documents,
ignored local data, another checkout, or an exact historical Git object were
removed. Historical artifacts were not restored to satisfy stale assertions.

Tests for production modules that are still imported by BioMiner or exposed by
its CLI remain. Their removal must follow a separate production-usage audit so
that reducing test count does not silently retain untested code.

## Evidence reviewed

The first post-merge run reported 2,457 passing tests, 175 failures, and one
skip. The failures were environmental or stale rather than regressions in a
public behavior:

| Failure cause | Cases | Ground Zero disposition |
| --- | ---: | --- |
| Paths deleted by the Ground Zero commit | 97 | Remove artifact/document tests |
| Missing ignored pilot registry inputs | 56 | Remove local-data-bound tests |
| Exact commits expected in a sibling checkout | 16 | Remove cross-repository tests |
| Stale file-content assertions | 5 | Remove implementation-history tests |
| Agent-pack byte/hash mismatch | 1 | Remove packaging-artifact test |

The reset removed 33 test modules, one machine-specific performance baseline,
and three orphaned fixture files. The removed performance test compared wall
clock and traced memory against one host while importing private fixtures from
other test modules; its deterministic count and ratio assertions were already
covered by unit tests.

## Retained suite contract

Default tests must be:

- isolated: temporary files, fake clients/classifiers, no credentials, live
  providers, generated repository state, sibling repositories, or test order;
- deterministic: fixed clocks and identifiers, stable ordering, no wall-clock
  or host-memory thresholds;
- behavioral: package and CLI outcomes, durable schemas, retry semantics,
  idempotence, and scientific fail-closed rules;
- focused: Arrange, Act, Assert; one observable behavior per test, with
  parameterization allowed for examples of the same validation rule;
- complete at boundaries: success, important edge cases, and retryable or
  terminal errors without requiring model downloads.

The Ground Zero EDA, TaxaLens publication, ButterflyLens handoff, and GBIF
reference-media tests were updated first because they crossed the merge and
were the highest-risk boundaries. CLI subprocess tests use the active Python
interpreter and isolated working directories. Publication and EDA tests were
split where one test exercised unrelated failure modes.

## Open structural debt

The remaining suite has pure fixture helpers imported between some test
modules, and a smaller set of tests that directly exercise private production
helpers. These tests are deterministic and green, but the coupling makes
refactoring more expensive. Address this incrementally by domain:

1. Move shared builders into `tests/factories.py` or a domain module under
   `tests/support/`, then make every consuming test arrange its own fresh value.
2. Replace private-helper assertions with public command, builder, validator,
   writer, or loader behavior. Where no suitable interface exists, decide
   whether the helper deserves a public typed boundary or the test is redundant.
3. Keep source-retirement and test-retirement in the same patch. Confirm CLI,
   import, and call-site absence before deleting either.
4. Add a lightweight static guard against new cross-test imports after the
   existing fixture imports have been migrated.

This is intentionally safer than deleting all adaptive or dynamic-pool tests:
those production packages still exist and cover scientific boundaries in the
future-state mission.

## Open-source practice check

GitHits was used to compare the approach with open-source pytest practice. The
adopted patterns were temporary-directory fixtures, explicit fake dependencies,
Arrange/Act/Assert structure, and invoking CLI subprocesses with the active
interpreter rather than a hard-coded virtual environment. No external code was
copied. Sources inspected:

- [Blockether/vis testing issue](https://github.com/Blockether/vis/issues/36)
- [EvalCraft CLI entry point](https://github.com/beyhangl/evalcraft/blob/ca622c7cdd970a6065597a900dd4a16ea2eb420d/evalcraft/cli/main.py)

## Required verification

Run from the repository root:

```bash
uv run pytest -q
uv run biominer --help
git diff --check
```

Observed on 2026-07-21 after merging `origin/main` at `e695807`:

| Command | Result |
| --- | --- |
| `uv run pytest -q` | 2,459 passed in 104.77 seconds |
| `uv run biominer --help` | Passed |
| `git diff --check` | Passed |
