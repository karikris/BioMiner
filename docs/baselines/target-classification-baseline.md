# Target-classification baseline

Recorded: 2026-07-13 10:48:30 UTC

Task: Phase 0, Task 0.1

Target: *Papilio demoleus*

The machine-readable record is
[`target-classification-baseline.json`](target-classification-baseline.json).

## Reproducibility state

| Item | Recorded value |
|---|---|
| Branch | `main` |
| Starting commit | `f73afe42f8b2c6c7773878e6d65c88ed6c9a7138` |
| Python | `3.14.5` |
| `uv.lock` SHA-256 | `4925abc898f1ea60f4795f4f894fa572a1dd13ab23b80eb40d644a06bd1776d4` |
| Lock validation | `uv lock --check`: passed, 35 packages resolved |
| Full test suite | `uv run pytest -q`: 966 passed in 37.23 seconds |
| Tracked worktree at capture | Clean |

Untracked files already existed at capture time. They are enumerated in the
manifest and are not baseline inputs.

## Current classifier settings

The production default is `target_scope_object_screening`, not the optional
`hierarchical_butterfly_classification` mode. The default target-scope path:

1. Scores family, genus and species text labels.
2. Retains the global species top 20.
3. Filters that set to the family text top 1 when family metadata is present,
   falling back to the unfiltered top 20 when the filter is empty.
4. Scores all remaining candidates in a second text-prompt pass and retains
   the top 5.
5. Records genus scores as diagnostics; genus rank does not gate species in
   this mode.

This second pass is still text-only evidence. It is not support-image
reranking and it does not produce a calibrated target probability.

The optional classification-v3 hierarchy uses a fixed beam width of 3 across
`FAMILY -> SUBFAMILY -> TRIBE -> SUBTRIBE -> GENUS -> SPECIES`, followed by a
species top-20 text pass, a text rerank of all 20, a retained top 5 and a
reported top 3. The legacy seven-rank layout instead applies family top 1,
genus top 20 then top 3, with a strict `> 0.90` genus shortcut.

Detected non-hard-negative objects are normally scored from a detector crop
with 0.12 padding and a 336-pixel target. A no-detection fallback uses the
whole image; hard-negative detections are excluded. The configured bucket
thresholds are 0.70 high species score, 0.35 low species score, 0.05 minimum
species and family margins, and 0.80 high detector score. These raw BioCLIP
scores and margins are not calibrated probabilities.

## Existing Papilio demoleus evidence

The only substantial local Papilio demoleus BioCLIP artifact is the 2026-06-09
flat zero-shot run under
`data/live_runs/papilio_demoleus_global_multilingual_20260609_071759`. It
scored 13,489 Flickr search candidates against 2,000 species and stored ten
species candidates per image. It predates the current path-cascade output
schema.

| Requested measure | Reproduced value | Interpretation |
|---|---:|---|
| Records seen | 13,489 | Unreviewed Flickr search candidates |
| Papilionidae count | 5,567 | Top-1 predicted species mapped to Papilionidae; not verified images |
| Papilio genus top-20 recall | unavailable | No genus ranking and no reviewed truth labels |
| Papilio genus top-three survival | unavailable | No genus ranking |
| *Papilio demoleus* species top 20 | unavailable | Artifact stores top 10 only |
| *Papilio demoleus* species top 5 | 8,297 | Target contained in stored top five; not recall |
| *Papilio demoleus* species top 1 | 5,567 | Target assigned top one; not accuracy |
| *Papilio demoleus* species top 10 | 9,311 | Supplemental containment count |
| Visual mode | unavailable | Not recorded in the historical schema |
| Threshold configuration | unavailable | No versioned threshold policy in the run summary |

The retired model-free cascade benchmark is not a substitute: it used synthetic
taxonomy and cannot support biological target metrics. Its historical artifact
and verification record remain available without retaining a callable runtime.

## Baseline boundary

Flickr query matches are discovery evidence. Nothing in this baseline treats
the candidate stream, query terms, BioCLIP top-k containment or top-1 output as
verified species truth. Consequently the unavailable recall and accuracy
fields remain unavailable until reviewed labels exist.
