# Target-aware few-shot Phase 0 verification

Date: 2026-07-13. Confidence: high for repository and design-contract state;
unknown for biological performance beyond the explicitly historical candidate
counts in the baseline report.

Phase 0 established a reproducible baseline, audited the current classification
architecture, and fixed the artifact and migration contracts for the planned
target-aware few-shot workflow. It added documentation only. No reference media
was acquired, no model weights were loaded, and no Flickr candidate was treated
as reviewed truth.

The user directed work to continue directly on `main`. The starting commit was
`f73afe42f8b2c6c7773878e6d65c88ed6c9a7138`.

## Task commits

| Task | Commit | Output |
|-|-|-|
| 0.1 baseline | `a6f0301483fb3d71d5731fd049c7f5a1fe5b80d6` | `docs/baselines/target-classification-baseline.json`, `docs/baselines/target-classification-baseline.md` |
| 0.2 architecture audit | `06eed244a04ab201bbc0798aab816b2927e4e3df` | `docs/adr/target_aware_few_shot_classifier.md` |
| 0.3 schema and migration contract | `ad10516d1df344d9e31fabd7b4c8c1cef4b9bd1e` | `docs/schemas/target_aware_few_shot_contracts.md` |

Each numbered task is a separate commit with its requested commit message.

While Phase 0 was in progress, `origin/main` advanced by seven commits from
`f73afe4` to `526f45a`. The remote changes were merged without rewriting the
three task commits. The merged and validated phase head was:

`a5cb24534cbf5a1f6b8ab7bc0f55e2e444f23219`

It was pushed to `origin/main` and independently verified with
`git ls-remote origin refs/heads/main` before this report was written.

## Environment and dependencies

| Check | Observed result |
|-|-|
| Python | `3.14.5` |
| `uv lock --check` | passed; 35 packages in the lock operation |
| `pyproject.toml` SHA-256 | `be79d998a73a3d641e3d9196ea4659848f3965e470c31021b2baa1fe6283f3d0` |
| `uv.lock` SHA-256 | `4925abc898f1ea60f4795f4f894fa572a1dd13ab23b80eb40d644a06bd1776d4` |
| Configured lint/type checks | none found in `pyproject.toml`; no Ruff, mypy, or Pyright configuration file exists |

The phase did not change dependency files.

## Test evidence

| Point | Command/scope | Result |
|-|-|-|
| Starting baseline | `uv run pytest -q` | 966 passed in 37.23 seconds |
| Task 0.1 | focused baseline/evaluation suite | 60 passed in 5.54 seconds |
| Task 0.2 | focused architecture audit suite | 180 passed in 6.60 seconds |
| Task 0.3 | strict schema, detection, runtime-path, taxonomy-store, embedding-cache, classification-mode, and config-asset tests | 96 passed in 2.62 seconds |
| Task 0.3 contract scan | all minimum artifact names plus ASCII validation | 34 required artifacts found; passed |
| Pre-fetch phase tree | `uv run pytest -q` | 966 passed in 34.88 seconds |
| Final merged phase tree | `uv run pytest -q` | 974 passed in 33.51 seconds |
| Staged/final whitespace | `git diff --check` | passed |

The higher final test count comes from the seven fetched registry and vision
commits, not from weakened selection or skipped tests.

## Baseline result boundary

An existing historical multilingual Flickr/BioCLIP artifact was available for
the baseline. Its checksummed Parquet rows produced:

- records: 13,489;
- target species top one: 5,567;
- target species top five: 8,297;
- target species top ten: 9,311;
- mapped Papilionidae top one: 5,567;
- unmapped top-one family: 309.

The artifact did not persist genus top-20/top-three results, species top-20, a
reviewed truth set, the visual mode, or threshold configuration. Those values
remain null or unavailable in the baseline report. The counts describe a
candidate stream and classifier outputs, not verified recall, accuracy, or
taxonomic validation.

## Design conclusions

The accepted ADR records that:

- family top-one and genus top-three pruning are invalid for target
  verification;
- a second text-prompt pass is not reference-image reranking;
- raw cosine similarity is not probability;
- geography selects plausible candidates and support but does not certify
  labels;
- only manually verified, licence-compatible, deduplicated references may
  enter production support embeddings.

The schema contract defines all minimum geographic, clustering, regional
candidate, reference, embedding, prototype, classifier, calibrator, inference,
and evaluation artifact names. It also defines full SHA-256 identities,
non-executable JSON plus NPZ persistence, route separation, readiness gates,
and a fail-closed migration boundary that leaves legacy
`object_bioclip_scores.parquet` immutable.

## Tool and provenance notes

GitHits was queried fresh before each numbered task. Useful evidence was kept
only where it agreed with local source and repository invariants. Examples that
used arbitrary thresholds, content-blind short hashes, fake Parquet fallbacks,
fabricated default probabilities, or simulated verification results were
explicitly rejected through GitHits feedback.

Morph codebase search was attempted during discovery and design review but
returned HTTP 429. Focused local source, tests, and migration documentation were
used as the authoritative fallback. Headroom compressed the 2,608-line task
specification and field-level extracts before reasoning over them.

## Repository hygiene

Only the three Phase 0 task documents were present in their numbered commits.
The following pre-existing untracked paths were not staged, modified, or
committed:

- `config/papilio_demoleus_flickr_estimator.sh2`;
- `config/papilio_demoleus_multilingual_keywords.json`;
- `docs/superpowers/`;
- `duplicate_query_terms_skipped`;
- `logs/`;
- `query_terms_added`.

No source images, model files, API dumps, secrets, cache objects, Parquet build
outputs, or DuckDB databases were added by Phase 0.

## Unexecuted live work

No live GBIF, iNaturalist, Flickr, reference-download, CUDA, YOLOE, or BioCLIP
run was required or executed in this documentation-only phase. Reproduction was
limited to deterministic tests and recomputation from the existing historical
Parquet artifact. Biological performance remains unmeasured until later phases
produce reviewed reference and evaluation data.
