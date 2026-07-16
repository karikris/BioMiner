# Papilio demoleus Build Week few-shot prototype report

Status: **complete for prototype-only Phase 14 evidence**. This does not
authorize a scientific production release or a production-default change.
All report inputs and outputs are local; S3 was not used.

> This prototype demonstrates the architecture and few-shot retrieval
> workflow. Provider-supported reference labels have not all received
> independent human taxonomic verification.

## What the prototype demonstrates

The prototype freezes an 81-record support bank, embeds it once with BioCLIP
2.5 Huge, scores a fixed target-aware candidate union, routes images with
YOLOE, executes the B0-B16 benchmark contract, and selects a conservative
global-reference policy. BioCLIP and YOLOE are screening components only.
GBIF remains the taxonomic spine, Flickr metadata remains discovery evidence,
and geography remains candidate-prior evidence.

The selected policy is B13 global references with status
`prototype_uncalibrated`. The target is always scoreable, no hierarchy pruning
or spatial crop is permitted, and a raw top-1-minus-top-2 margin below 0.10
abstains. That threshold was predeclared; it was not fitted from labels and is
not a probability threshold.

## Reference bank

Ninety-three provider-supported media were selected. Eighty-one GBIF records
passed the local prototype freeze, two records were excluded by automated QA,
and ten Wikimedia downloads remain retryable operational failures. None of
the 81 frozen records is independently human taxonomically verified.

| Evidence | Count |
|---|---:|
| Frozen prototype support | 81 |
| Provider-supported | 81 |
| Human-verified | 0 |
| Target adult | 20 |
| Target caterpillar | 1 |
| Regional competitors | 31 |
| Reviewed false-winner species | 11 |
| Broader Papilionidae | 11 |
| Biological hard negatives | 7 |
| Visual-domain negatives | 0 |

Trust is R4 for all 81 records: R1 0, R2 0, R3 0, R4 81, R5 0. Geographic
layers are A 51, B 6, C 0, D 24, and E 0. The frozen source distribution is
GBIF 81 and Wikimedia Commons 0. Licence policy is 2 allowed and 79
research-only.

The split contains 26 support-train, 30 model-selection, 13 calibration, and
12 final-test records. Adult, larval, and specimen routes remain separate:
80 adult-field records, one larval record, and zero pinned specimens. The
larval record is in calibration rather than support-train, so no larval
prototype exists.

## Exact support shortfalls

The acquisition ledger records 553 missing selections across 34 shortfall
scopes.

| Reference group | Requested | Selected | Shortfall |
|---|---:|---:|---:|
| Target adult | 50 | 20 | 30 |
| Target caterpillar | 20 | 1 | 19 |
| Regional competitors | 100 | 32 | 68 |
| Reviewed false-winner genera | 100 | 11 | 89 |
| Historical false-winner species | 20 | 0 | 20 |
| Broader Papilionidae | 200 | 11 | 189 |
| Moth negatives | 45 | 1 | 44 |
| Other-insect negatives | 30 | 6 | 24 |
| Other-Lepidoptera negatives | 30 | 0 | 30 |
| Visual-domain negatives | 11 | 11 | 0 |
| Visual-neighbour candidates | 40 | 0 | 40 |

The exact scope-level ledger is
`runs/papilio_demoleus_pilot_prototype_acquisition_20260715/prototype_reference_shortfalls.parquet`
with SHA-256
`ac2ddccf0e410a332793c21150bf1859d1e459dc85c3d1062386e2ba795c3b5a`.

## B0-B16 benchmark

All 81 frozen records were scored with no skips, producing 1,539 prediction
rows and 12,874 candidate-score rows. The experiment summaries below cover
the 55 model-selection, calibration, and final-test records; support-train is
used for fitting rather than evaluation. Target retrieval and provider
consistency are diagnostic only—not classification accuracy.

| Experiment | Method | Target top-1 | Mean target rank | Mean raw margin | Abstention | Full input |
|---|---|---:|---:|---:|---:|---:|
| B0 | Current text-pruned | 0.90 | 1.0 | 0.048773 | 0.000000 | 1.000000 |
| B1 | Zero-shot, no pruning | 0.90 | 3.6 | 0.048773 | 0.000000 | 1.000000 |
| B2 | SimpleShot | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.981818 |
| B3 | Centered SimpleShot | 0.90 | 1.0 | 0.394124 | 0.018182 | 0.981818 |
| B4 | Top-five references | 0.90 | 1.0 | 0.160001 | 0.018182 | 0.981818 |
| B5 | Multi-prototype | 0.90 | 1.0 | 0.150996 | 0.018182 | 0.981818 |
| B6 | Logistic regression | 0.90 | 1.1 | 12.895471 | 0.000000 | 1.000000 |
| B7 | LinearSVC | 0.90 | 1.1 | 2.118011 | 0.000000 | 1.000000 |
| B8 | LinearSVC with features | 0.90 | 1.1 | 2.112395 | 0.000000 | 1.000000 |
| B9 | Calibrated-abstention contract | 0.90 | 1.1 | 2.118011 | 0.000000 | 1.000000 |
| B10 | Raw full frame | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.981818 |
| B11 | Raw plus focused | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.000000 |
| B12 | Raw plus focused plus masked | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.000000 |
| B13 | Global references | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.981818 |
| B14 regional | Regional only | 0.20 | 1.0 | 0.102569 | 0.418182 | 0.581818 |
| B14 global | Global only | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.981818 |
| B14 layered | Trust-first layered | 0.80 | 1.0 | 0.121533 | 0.018182 | 0.000000 |
| B15 | Text-image fusion | 0.90 | 1.0 | 0.133462 | 0.018182 | 0.981818 |
| B16 | Image-only | 0.90 | 1.0 | 0.157182 | 0.018182 | 0.981818 |

B11 and B12 retained the executable contract but reused raw full-frame
evidence because focused and masked embeddings were unavailable. B6-B9 use
different raw score scales and their margins must not be compared numerically
with cosine-similarity margins. B9 did not produce a scientifically valid
calibrator because reviewed labels do not exist.

## Policy selection

Selection used only the 30 model-selection records. Compared with B0, B13
increased target scoreability from 0.10 to 1.00, retained 1.00 target top-1
retrieval on the three target records, increased mean raw margin from 0.041302
to 0.163995, and reduced missing or sub-0.10 margins from 0.90 to 0.20. All 27
non-target model-selection records ranked a competitor above the target.

Regional-only target scoreability was 0.30; trust-first layered scoreability
was 0.333333. Both could remove the target in sparse conditioned clusters, so
global references were selected. The frozen 0.10 margin policy accepts 24 of
30 model-selection records and abstains on 6. The calibration partition was
used only for coverage auditing: it would accept 9 of 13 and abstain on 4.
Zero reviewed calibration labels were used, no calibrator was fitted, and
final-test was not used for model or threshold selection.

## Staged Flickr inference

The local staged run planned 13,501 Flickr records, classified 13,496, and
retained five download/decode failures as retryable. It wrote 634,312
candidate-score rows: 34 species plus two known-negative and 11 visual-domain
classes per classified record. The target was scored in all 13,496 results.

YOLOE routing produced:

| Route | Count |
|---|---:|
| Adult butterfly field | 2,802 |
| Caterpillar field | 400 |
| Pinned specimen | 31 |
| Pupa or chrysalis | 4,505 |
| Ambiguous visual domain | 4,068 |
| No relevant organism | 1,583 |
| Possible moth or other insect | 61 |
| Artwork/logo/tattoo/artifact | 46 |

The routing actions were 3,233 score, 3,725 review, and 6,538 exclude.
BioCLIP routes were 6,527 adult, 400 larval, 31 pinned-specimen, and 6,538 not
routed. Only the adult route had frozen reference support; the 400 larval and
31 pinned-specimen cases therefore remained abstained/reviewable rather than
being scored against an incompatible bank.

The target beat the best text competitor on 1,502 of 13,496 records. Among
the 6,527 adult-reference routes, it beat the best reference competitor on
4,345 records (66.57%). These are unlabelled inference distributions, not
accuracy or prevalence.

The staged runner's preselection policy abstained on 12,296 records and did
not abstain on 1,200. This is distinct from the frozen Task 14.5 policy:
the staged runner used its existing 0.02 margin plus route gates, whereas the
frozen policy uses 0.10.

P3 sustained 2.274524 records per second (0.439652 seconds per record) with
1,765,261,312 bytes peak RSS. MPS allocator metrics were not instrumented and
are not guessed. The five failed Flickr records are `5229667156`,
`22573167168`, `6016065489`, `51546333924`, and `7607829480`; all failed at
download/decode after three attempts and remain retryable, not biological
negatives.

## Dashboard-ready evidence examples

No raw image is included in this report. The machine-readable companion
contains four metadata-only examples for an accepted high-margin case, a
low-margin abstention, a review route, and an excluded route. Each includes
the Flickr photo ID, geographic cluster, YOLOE route, visual input, nearest
target reference identifier, best competitor, raw similarities, margin, and
abstention reason. Public display of reference thumbnails remains subject to
licence review.

## Known limitations

- No reference label is independently human taxonomically verified.
- Provider-supported retrieval consistency is not classification accuracy.
- The policy is uncalibrated and emits no probabilities.
- Seventy-nine of 81 references are research-only.
- Sparse regional and layered planners can remove the target.
- Larval, pinned-specimen, and visual-domain support is incomplete.
- Focused and masked visual-input ablations were unavailable.
- Five Flickr failures and ten reference-source failures remain retryable.
- YOLOE routing accuracy is not validated; pupa and ambiguous routing require
  targeted review.
- Two support records lack owner evidence, so leakage protection is not
  claimed complete.
- The staged distribution is not a population prevalence estimate.

## Post-hackathon human-review plan

1. Review target adult and caterpillar references first with attributable
   expert decisions.
2. Review competitors, false winners, broader-family taxa, and hard negatives
   with licence and attribution evidence visible.
3. Resolve retryable reference downloads without converting failures into
   biological negatives.
4. Acquire larval support-train, pinned-specimen, visual-domain-negative,
   historical-false-winner, other-Lepidoptera, and visual-neighbour evidence.
5. Rebuild leakage components and the four splits after review.
6. Fit calibration and thresholds only on reviewed calibration labels.
7. Evaluate the locked policy once on reviewed final-test labels.
8. Audit licence policy before exposing thumbnails or image copies.

## Verification

- Phase 14 report contract: 22 passed.
- Prototype policy plus Phase 14 compact contract: 25 passed.
- Full repository suite: 2,309 passed in 78.74 seconds.
- Ruff, JSON validation, CLI help, and `git diff --check`: passed.

## Phase 15 prototype entry

Phase 14's prototype report is complete, so Phase 15 prototype integration may
proceed to its explicit go/no-go audit. This does not authorize a production
default change or scientific release.
