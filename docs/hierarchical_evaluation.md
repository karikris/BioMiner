# Hierarchical Evaluation

Scope: Phase 5 evaluation, calibration, QA review, and regression testing for the YOLOE-26 plus BioCLIP 2.5 hierarchical butterfly workflow.

This document defines how BioMiner measures the visual-classification pipeline. It does not redefine taxonomy, detection policy, evidence buckets, or occurrence publication rules.

## Evaluation Contract

BioMiner evaluates the pipeline as a screening and classification workflow:

```text
Flickr image
-> YOLOE-26 butterfly_like detection
-> detector crop
-> BioCLIP family top 3
-> selected family
-> BioCLIP species top 20 within selected family
-> BioCLIP rerank of all top 20 into top 5
-> object/photo evidence
-> review queues and reports
```

The accepted GBIF-backed registry and derived classification tables define candidate scope. They are not evaluation truth. Human-reviewed labels, trusted synthetic fixtures, or other explicitly reviewed labels provide evaluation truth.

BioCLIP scores are visual screening and classification evidence. They are not formal taxonomic validation, and they do not by themselves create verified Darwin Core occurrences. Evaluation reports must describe score quality, failure modes, and review needs without implying taxonomic authority.

YOLOE and BioCLIP are measured separately:

- YOLOE is an object detector only. Its output is evaluated as object discovery and butterfly-gate evidence.
- BioCLIP is evaluated only on eligible detector crops. Production evaluation must not reward or require all-image BioCLIP scoring.

## Fixed Invariants

Evaluation must preserve these classifier semantics:

- `target_scope_object_screening` remains the default mode and is reported separately from hierarchical mode.
- `hierarchical_butterfly_classification` performs open butterfly classification, not target validation.
- YOLOE does not classify family, genus, or species.
- BioCLIP scores only detections with `detection_status=detected` and `detector_label=butterfly_like`.
- Hierarchical mode scores configured butterfly families and records family top 3.
- Species top 20 is constrained to the selected top family.
- Species top 5 is reranked from all first-pass top-20 species.
- Hierarchical mode does not inject the run target species.
- Prompt-template variants are aggregated by taxon, with mean aggregation as the default.
- Non-butterfly detections remain evidence rows and review signals, but they are not BioCLIP species-scored.

## Truth Data

Reviewed labels are evaluation data, not registry data. A reviewed label row must identify its source, photo, optional detection/crop, label level, butterfly/non-butterfly status, accepted taxon fields when applicable, reviewer, review time, confidence, and notes.

Real accuracy claims require human-reviewed labels or an explicitly trusted labelled dataset. Model-free synthetic fixtures are valid for regression testing and arithmetic verification, but they must not be presented as biological performance evidence.

Low-confidence labels should be retained with their confidence and notes. Evaluation may report them separately or warn, but it must not silently treat weak labels as high-confidence truth.

## Evaluation Levels

Image-level butterfly gate:
Measures whether an input photo produces any usable butterfly-like object evidence and whether negative photos avoid false butterfly promotion.

Object-level butterfly crop detection:
Measures detected object rows, eligible crop count, no-detection count, non-butterfly skip count, and false positives or false negatives relative to reviewed object labels.

Family top1/top3:
Measures whether the reviewed family is the first predicted family and whether it appears anywhere in family top 3.

Selected-family correctness:
Measures whether the selected top family matches the reviewed family. This is the boundary that determines the species candidate pool, so it is reported explicitly even when family top3 recall is good.

Species top1/top5/top20:
Measures exact reviewed species recovery at the first position, in reranked top 5, and in first-pass top 20. Species metrics apply only to hierarchical rows with butterfly-positive species labels.

Review queue quality:
Measures whether uncertain, conflicting, missing-score, hard-negative, and metadata-vision disagreement cases are captured for human review and ranked above clean confident rows.

Negative and non-butterfly handling:
Measures whether non-butterfly labels avoid BioCLIP species evidence and avoid promotion into confident species outputs.

End-to-end photo summary behavior:
Measures whether object-level predictions aggregate into conservative photo summaries, whether multi-object conflicts are retained, and whether open hierarchical predictions are not misrepresented as verified target positives.

## Required Metrics

Detection and gating:

- `detection_eligible_count`: detector rows eligible for BioCLIP crop scoring.
- `no_detection_count`: photos or objects with no usable detection.
- `non_butterfly_skip_count`: detector rows retained as evidence but skipped for BioCLIP species scoring.
- `missing_prediction_count`: reviewed labels without a matching prediction row.
- `missing_label_count`: prediction rows without a matching reviewed label.

Family metrics:

- `family_top1_accuracy`: fraction of butterfly-positive evaluated objects whose reviewed family is family top1.
- `family_top3_recall`: fraction whose reviewed family appears in family top3.
- `selected_family_accuracy`: fraction whose selected family matches the reviewed family.
- `top_k_accuracy_by_family`: top-k metrics grouped by reviewed family.
- `family_confusion_matrix`: reviewed family versus predicted family counts, including missing prediction and not-butterfly buckets.

Species metrics:

- `species_top1_accuracy`: fraction of butterfly-positive species labels whose reviewed species is species top1.
- `species_top5_recall`: fraction whose reviewed species appears in reranked species top5.
- `species_top20_recall`: fraction whose reviewed species appears in first-pass species top20.
- `species_mrr`: mean reciprocal rank of the reviewed species in the best available ordered species list, normally species top20.
- `species_confusion_matrix`: reviewed species versus predicted species counts, including missing prediction and not-butterfly buckets.

Negative handling:

- `negative_labels`: reviewed non-butterfly objects or photos.
- `negative_correct_count`: negative labels without butterfly species promotion.
- `false_positive_butterfly_count`: non-butterfly labels with butterfly-positive evidence or species output.
- `false_negative_butterfly_count`: butterfly-positive labels with no eligible detector/BioCLIP result.

Review and bucket metrics:

- `review_queue_capture_rate`: fraction of labelled error or uncertainty cases present in the review queue.
- `conservative_bucket_distribution`: output counts by Gold, Silver, Bronze, Bin, InReview, or the current equivalent bucket fields.
- `manual_review_promotion_count`: count of reviewed rows promoted by human/comment evidence when available.
- `manual_review_decline_count`: count of reviewed rows declined or retained for review when available.
- `review_priority_counts`: counts by priority band or exact priority where available.
- `top_review_reasons`: most common review reasons.

Mode separation:

- Hierarchical metrics are computed for `classification_mode == hierarchical_butterfly_classification`.
- Target-scope metrics are computed separately or skipped with an explicit skipped-row count.
- Mixed-mode inputs must not merge hierarchical open-classification accuracy with target-scope screening support.

Calibration and uncertainty:

- `species_top1_margin`: difference between the first and second species scores when available.
- `family_margin`: difference between family top1 and top2 scores when available.
- `species_top5_entropy`: entropy over species top5 scores.
- `expected_calibration_error`: heuristic candidate-set-relative calibration error when score columns and correctness columns are available.

Calibration metrics must state that BioCLIP scores are candidate-set-relative. They are useful for review prioritisation and threshold tuning, not absolute biological confidence.

## Reports And Artifacts

Phase 5 evaluation reports should be model-free once prediction artifacts exist. A report writer should be able to consume object scores or joined object evidence plus reviewed labels and write:

- `evaluation_metrics.json`
- `family_confusion_matrix.parquet`
- `species_confusion_matrix.parquet`
- `evaluation_summary.md`
- `calibration_bins.parquet`
- `review_error_examples.parquet`
- optional charts only when requested:
  - `family_confusion_matrix.png`
  - `species_accuracy_by_family.png`
  - `calibration_reliability.png`
  - `review_reason_counts.png`

Report metadata should include run id, classification mode, taxonomy table version, prompt variant version, BioCLIP model/checkpoint when present, input paths, reviewed-label schema version, row counts, metric counts, limitations, and artifact paths.

Local evaluation command:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical
```

Use `--object-scores` instead of `--object-evidence` when evaluating raw BioCLIP object-score output. Use `--write-charts` only for local report directories when PNG diagnostics are needed.

Cloud evaluation command shape:

```bash
uv run biominer evaluation classify \
  --object-evidence s3://biominer/biominer/runs/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels s3://biominer/biominer/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir s3://biominer/biominer/reports/evaluation/papilionoidea_hierarchical \
  --storage-backend s3 \
  --config config/production.toml
```

S3 evaluation writes JSON, Parquet, and Markdown artifacts. Charts are intentionally local-only until the storage layer grows binary image writes.

Standalone local review-queue command:

```bash
uv run biominer evaluation review-queue \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --photo-summary runs/local_debug/papilionoidea_hierarchical/photo_evidence_summary.parquet \
  --output reports/review_queue.parquet
```

The queue prioritises human review; it is not an accuracy label set and must not be treated as taxonomic truth. Production `summarize` writes this artifact automatically for run outputs.

Xie-style metrics command:

```bash
uv run biominer evaluation classify \
  --object-evidence runs/local_debug/papilionoidea_hierarchical/object_evidence_joined.parquet \
  --reviewed-labels data/reviewed/papilionoidea_reviewed_labels.parquet \
  --output-dir reports/evaluation/papilionoidea_hierarchical \
  --evaluation-profile xie_style_metrics_only
```

This adds `xie_style_metrics.json`. It is a metrics profile only and does not mutate classifier outputs, replace BioMiner architecture, or alter the family-first candidate-selection rules.

## Regression Fixtures

Model-free fixtures should cover:

- family top1 correct
- family top3 correct but top1 wrong
- species top1 correct
- species top5 correct but top1 wrong
- species top20 correct but top5 wrong
- selected-family mismatch
- missing prediction
- negative label
- mixed target-scope and hierarchical rows
- no target species injection in hierarchical mode
- BioCLIP scored only eligible butterfly-like detections

Golden tests should compare compact JSON metrics and small Parquet tables. They should round floats to fixed precision and avoid snapshotting large output tables.

## Non-Goals

Phase 5 does not:

- train or fine-tune the detector
- store reviewed boxes for detector training
- create a permanent Flickr image archive
- score all images with BioCLIP
- use YOLOE as a family or species classifier
- replace BioMiner with Xie or any other architecture
- claim verified Darwin Core occurrences from visual scores alone
- make GBIF classification tables into evaluation truth
- require CUDA, MPS, real BioCLIP weights, Flickr credentials, or network access in unit tests

Xie-style evaluation, when enabled, is a metrics profile only. It reports benchmark-style macro, micro, per-family, and top-k metrics for the BioMiner architecture:

```text
architecture = biominer_yoloe26_bioclip25_hierarchical
evaluation_profile = xie_style_metrics_only
```

It must not alter classifier outputs or candidate selection.
