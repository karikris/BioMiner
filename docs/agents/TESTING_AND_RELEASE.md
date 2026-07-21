# Testing, phase completion, and release

## Default environment

BioMiner requires Python `>=3.14`.

Typical setup:

```bash
uv sync --extra test
```

Use the extras required by the changed area:

```text
geo
ml
postgres
test
```

Do not add heavyweight optional dependencies to registry-only installations.

## Test order

1. Test the smallest changed behavior.
2. Run neighboring module tests.
3. Run integration tests across the changed contract.
4. Run CLI smoke tests for changed commands.
5. Run the full suite at the task/phase boundary required by the goal.
6. Run lint, type, schema, provenance, and generated-file checks configured by
   the repository.

Examples:

```bash
uv run pytest -q tests/test_reference_admission_policy.py
uv run pytest -q tests/test_support_admission.py tests/test_reference_readiness.py
uv run pytest -q
git diff --check
```

Do not rerun a large suite repeatedly when focused evidence is sufficient.

## Current observed baseline

The dynamic-pooling implementation baseline at
`30319fa7b7e9f24256556b293cf2e2db6e6ce2e7` was:

```text
complete pilot evaluation gate    56 passed
full regression                  3225 passed
```

These counts are historical evidence, not a target. Ground Zero deliberately
removed report-, documentation-, generated-data-, and sibling-repository
verification from the default suite. Use the current local count and record any
change. The reset rationale is in
`docs/research/ground_zero_test_suite_review_2026-07-21.md`.

At the 2026-07-21 Ground Zero reset on merged `origin/main` commit `e695807`:

```text
full regression                  2459 passed
```

## Default suite constraints

Default tests must:

- follow Arrange, Act, Assert with one observable behavior per test;
- exercise public package or CLI interfaces rather than private helpers or
  source-code layout;
- create inputs and outputs under `tmp_path` rather than depending on generated
  repository state;
- remain independent of execution order and shared mutable state;
- make no live Flickr, GBIF, CoL, iNaturalist, ALA, EOL, Wikidata, or OpenAI
  calls;
- require no real model weights;
- require no source images;
- require no cloud credentials;
- use deterministic fake clients, fixtures, clocks, seeds, and storage;
- verify retries, ordering, checkpoints, and idempotence;
- fail closed on stale fingerprints and incompatible schemas.

The default suite must not inspect deleted reports or documentation as a proxy
for runtime behavior, call `git show` for historical commits, open a sibling
checkout, use a hard-coded virtual-environment interpreter, or assert elapsed
time or process-memory ceilings. Put host performance measurement in an
explicit benchmark workflow. Invoke subprocess CLI tests with the active
interpreter and an isolated working directory.

Several malformed examples may be parameterized when they demonstrate the same
single rejection contract. Do not combine unrelated success, error, and
publication behaviors in one test.

## Required test classes

### Registry and queries

- accepted identity and lineage;
- source/version provenance;
- name normalization;
- collisions and homonyms;
- query eligibility;
- deterministic query IDs;
- duplicate physical request versus retained logical associations;
- checkpoint resume and incompatible checkpoint rejection.

### Flickr

- one application-wide rate ledger;
- reservation before request;
- retry/backoff;
- deterministic pagination and partitions;
- query-hit provenance;
- canonical photo deduplication;
- stale-claim recovery.

### Geography and candidates

- invalid/uncertain coordinates;
- dateline;
- resolution parents;
- no-geo and fallback;
- stable clusters;
- target always present;
- candidate reasons and fingerprints;
- absence never treated as negative truth.

### References

- automated admission matrix;
- strict/adaptive/flagged modes;
- licence/decode/dimension/route/duplicate gates;
- observation/observer independence;
- human rejection override;
- readiness permit matrix;
- create-only publication concurrency;
- stale policy invalidation.

### Vision and BioCLIP

- route separation;
- full-frame transform identity;
- no spatial-crop implementation or callable fallback;
- one raw embedding per image;
- cache invalidation;
- fixed-k and balanced prototypes;
- provisional score semantics;
- raw score not probability;
- model/reference fingerprint matching.

### Human review and evaluation

- source-hash-bound labels;
- append-only decisions;
- conflicts and adjudication;
- support/calibration/test leakage;
- representative versus targeted samples;
- weighted estimates;
- insufficient-sample states;
- final export blocks all non-decisive or stale records.

### Incremental remediation

- reference revision impact;
- unchanged embedding reuse;
- affected prototype/model rebuild;
- selective rescore;
- paired before/after evaluation;
- unrelated species untouched.

## Live tests

Live source/model tests are opt-in and must:

- use explicit credentials;
- have bounded calls, rows, images, time, and cost;
- identify provider terms and user agent;
- write a receipt;
- preserve source/model versions;
- never be required by the default suite;
- never fabricate success when credentials or human input are absent.

A live model test must record device, model revision, preprocessing, batch size,
memory, and exact artifact fingerprints.

## Failure policy

- Report the first exact failure and preserve its evidence.
- Do not weaken a production check to make a test pass.
- Do not restore obsolete code merely for a stale test.
- When intended behavior changes, update the contract and tests together.
- A flaky concurrency test requires root-cause repair and repeated evidence, not
  a retry decorator or skip.
- Unsupported metrics or live steps remain explicit.

## Phase completion

At a phase boundary:

1. Run all phase-relevant tests.
2. Run the full suite if the goal requires it.
3. Run configured lint/type/schema checks.
4. Verify provenance JSON/YAML.
5. Inspect secrets and generated files.
6. Inspect `git status --short`.
7. Record task commits and pushed SHAs.
8. Write phase JSON/Markdown with:
   - commands and results;
   - artifacts;
   - schemas/policies;
   - limitations;
   - live steps not executed;
   - human work still required;
   - claims allowed and blocked.

## Release gates

Release verification must confirm:

- adaptive default and strict compatibility behave as documented;
- provider assertions are not called verified;
- raw scores are not called probabilities;
- unreviewed Flickr records cannot enter final export;
- calibration/final-test labels exclude provider assertions;
- target-aware modes preserve their candidate/full-frame contracts;
- statistical audit requirements are present;
- targeted review and selective rerun are provenance-bound;
- all cloud/local outputs are resumable;
- rights and attribution are complete;
- no secret or source media is committed;
- documentation matches the selected mode.

For geography-conditioned dynamic pooling, release verification must also
confirm:

- family optimization never catastrophically prunes the target or complete
  candidate union;
- geography is evidence, never identity or biological absence;
- GBIF provider assertions remain provisional;
- compatible image embeddings are reused rather than recomputed per pool;
- raw family/global/local/fusion evidence is not labelled probability;
- calibrated probability, human verification, statistical support,
  release-readiness and publication maturity remain separate;
- unreviewed Flickr cannot enter an occurrence export;
- insufficient strata report estimates unavailable rather than zero;
- targeted failure-discovery work cannot support unweighted population claims;
  and
- TaxaLens and ButterflyLens handoffs preserve their pinned consumer maturity,
  rights, review, RLS and release boundaries.

Passing these software gates verifies implementation semantics only. It does
not fill the 86-effective-review shortfall, select a production strategy, claim
live biological performance, or authorize occurrence release.

Do not merge, tag, publish, or claim completion automatically unless the user
explicitly requests it.
