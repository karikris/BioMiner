# BioCLIP 2.5 Butterfly Identification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Historical note: this plan predates the Phase 4/5 audit fixes. The current
species prompt aggregation contract is mean aggregation by taxon by default,
with best-label retained only as evidence metadata. The older max-based
aggregation sketch below is superseded and is not the active implementation
contract.

**Goal:** Improve BioMiner's BioCLIP 2.5 butterfly species screening by adding prompt ensembling, grouped triage probabilities, taxonomic consistency signals, richer diagnostics, and validation/report tooling.

**Architecture:** Keep the existing persistent worker/register-runner architecture. Add small focused modules around label/prompt construction and prediction diagnostics, then thread the new fields through `BioClipClassifier`, `register_runner`, `triage`, and reports without changing Flickr fetch behavior.

**Tech Stack:** Python, Polars, Parquet, DuckDB where query-style analytics are needed, pytest, BioCLIP 2.5/OpenCLIP through the existing persistent worker.

---

## File Structure

- Create: `src/biominer/bioclip/prompt_templates.py`
  - Owns prompt templates, prompt variants per taxon, group definitions, and prompt aggregation helpers.
- Modify: `src/biominer/bioclip/species_candidates.py`
  - Extend `SpeciesCandidate` with optional `common_names`; expose prompt labels and label-to-species mappings.
- Modify: `src/biominer/bioclip/bioclip.py`
  - Aggregate prompt-level scores into taxon-level scores; preserve raw top-k prompt evidence.
- Modify: `src/biominer/bioclip/register_runner.py`
  - Build enriched species/triage label sets and persist new diagnostics in success rows.
- Modify: `src/biominer/bioclip/triage.py`
  - Use grouped probabilities, top-1 margin, and taxonomic consistency in Gold/Silver/Bronze decisions.
- Create: `src/biominer/bioclip/diagnostics.py`
  - Compute top-k margin, entropy, grouped probability summaries, and taxonomic consistency fields.
- Create: `scripts/evaluate_bioclip_species_validation.py`
  - Evaluate a reviewed validation parquet with Polars/DuckDB and write compact metrics.
- Modify: `scripts/generate_bioclip_species_visual_report.py`
  - Add plots/tables for margin, entropy, grouped probabilities, and hierarchy consistency.
- Create/modify tests:
  - `tests/test_prompt_templates.py`
  - `tests/test_species_candidates.py`
  - `tests/test_bioclip_prediction.py`
  - `tests/test_register_runner.py`
  - `tests/test_image_triage.py`
  - `tests/test_bioclip_diagnostics.py`
  - `tests/test_bioclip_species_validation_report.py`

## Phase 1: Prompt Template Ensembling

### Task 1: Add Prompt Template Module

**Files:**
- Create: `src/biominer/bioclip/prompt_templates.py`
- Create: `tests/test_prompt_templates.py`

- [ ] **Step 1: Write failing tests**

```python
from biominer.bioclip.prompt_templates import (
    PromptVariant,
    build_species_prompt_variants,
    aggregate_prompt_scores,
)


def test_build_species_prompt_variants_includes_scientific_and_common_names() -> None:
    variants = build_species_prompt_variants(
        scientific_name="Papilio demoleus",
        common_names=("lime butterfly", "chequered swallowtail"),
    )

    labels = [variant.label for variant in variants]
    assert "a photo of Papilio demoleus" in labels
    assert "a field photo of Papilio demoleus adult butterfly" in labels
    assert "a photo of lime butterfly" in labels
    assert all(variant.taxon_key == "Papilio demoleus" for variant in variants)


def test_aggregate_prompt_scores_uses_mean_by_default_and_keeps_evidence() -> None:
    variants = [
        PromptVariant(label="a photo of Papilio demoleus", taxon_key="Papilio demoleus", prompt_kind="scientific"),
        PromptVariant(label="a photo of lime butterfly", taxon_key="Papilio demoleus", prompt_kind="common"),
        PromptVariant(label="a photo of Papilio machaon", taxon_key="Papilio machaon", prompt_kind="scientific"),
    ]

    result = aggregate_prompt_scores(
        scores={
            "a photo of Papilio demoleus": 0.72,
            "a photo of lime butterfly": 0.08,
            "a photo of Papilio machaon": 0.55,
        },
        variants=variants,
        top_k=2,
    )

    assert result[0]["taxon_key"] == "Papilio machaon"
    assert result[0]["score"] == 0.55
    assert result[1]["taxon_key"] == "Papilio demoleus"
    assert result[1]["score"] == 0.40
    assert result[1]["best_label"] == "a photo of Papilio demoleus"
    assert result[1]["prompt_scores"]["a photo of lime butterfly"] == 0.08
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_prompt_templates.py`

Expected: FAIL because `biominer.bioclip.prompt_templates` does not exist.

- [ ] **Step 3: Implement prompt module**

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class PromptVariant:
    label: str
    taxon_key: str
    prompt_kind: str


SPECIES_PROMPT_TEMPLATES = (
    ("scientific", "a photo of {name}"),
    ("field_adult", "a field photo of {name} adult butterfly"),
    ("close_adult", "a close-up photo of {name} butterfly"),
)

COMMON_NAME_PROMPT_TEMPLATES = (
    ("common", "a photo of {name}"),
    ("common_adult", "a field photo of {name} adult butterfly"),
)


def build_species_prompt_variants(
    *,
    scientific_name: str,
    common_names: Sequence[str] = (),
) -> list[PromptVariant]:
    variants: list[PromptVariant] = []
    seen: set[str] = set()
    for kind, template in SPECIES_PROMPT_TEMPLATES:
        label = template.format(name=scientific_name)
        if label not in seen:
            variants.append(PromptVariant(label=label, taxon_key=scientific_name, prompt_kind=kind))
            seen.add(label)
    for common_name in common_names:
        for kind, template in COMMON_NAME_PROMPT_TEMPLATES:
            label = template.format(name=common_name)
            if label not in seen:
                variants.append(PromptVariant(label=label, taxon_key=scientific_name, prompt_kind=kind))
                seen.add(label)
    return variants


def aggregate_prompt_scores(
    *,
    scores: Mapping[str, float],
    variants: Sequence[PromptVariant],
    top_k: int,
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for variant in variants:
        score = float(scores.get(variant.label, 0.0))
        current = grouped.setdefault(
            variant.taxon_key,
            {"taxon_key": variant.taxon_key, "score": 0.0, "best_label": None, "prompt_scores": {}},
        )
        prompt_scores = current["prompt_scores"]
        assert isinstance(prompt_scores, dict)
        prompt_scores[variant.label] = score
        if score >= float(current["score"]):
            current["score"] = score
            current["best_label"] = variant.label
    return sorted(grouped.values(), key=lambda row: float(row["score"]), reverse=True)[:top_k]
```

- [ ] **Step 4: Run tests to verify pass**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_prompt_templates.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add src/biominer/bioclip/prompt_templates.py tests/test_prompt_templates.py
git commit -m "step3: add bioclip species prompt templates"
```

## Phase 2: Candidate Prompt Labels and Common Names

### Task 2: Extend Species Candidates With Prompt Variants

**Files:**
- Modify: `src/biominer/bioclip/species_candidates.py`
- Modify: `tests/test_species_candidates.py`

- [ ] **Step 1: Write failing tests**

Add these tests to `tests/test_species_candidates.py`:

```python
from biominer.bioclip.species_candidates import species_prompt_variants


def test_species_candidates_read_common_names_and_build_prompt_variants(tmp_path) -> None:
    path = tmp_path / "candidates.csv"
    path.write_text(
        "scientific_name,rank,family,genus,common_names\n"
        "Papilio demoleus,species,Papilionidae,Papilio,lime butterfly|chequered swallowtail\n",
        encoding="utf-8",
    )

    candidates = load_species_candidates(path)
    assert candidates[0].common_names == ("lime butterfly", "chequered swallowtail")

    variants = species_prompt_variants(candidates)
    labels = [variant.label for variant in variants]
    assert "a photo of Papilio demoleus" in labels
    assert "a photo of lime butterfly" in labels
```

- [ ] **Step 2: Run focused tests to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_species_candidates.py`

Expected: FAIL because `common_names` and `species_prompt_variants` do not exist.

- [ ] **Step 3: Implement candidate common names**

Modify `SpeciesCandidate`:

```python
common_names: tuple[str, ...] = ()
```

In `_candidate_from_row`, parse common names:

```python
common_names=_split_common_names(_first_text(row, "common_names", "commonNames", "vernacular_names", "vernacularNames")),
```

Add helper:

```python
def _split_common_names(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    names = []
    for part in value.replace(";", "|").split("|"):
        cleaned = " ".join(part.strip().split())
        if cleaned and cleaned not in names:
            names.append(cleaned)
    return tuple(names)
```

Add prompt variant exports:

```python
from biominer.bioclip.prompt_templates import PromptVariant, build_species_prompt_variants


def species_prompt_variants(candidates: list[SpeciesCandidate]) -> list[PromptVariant]:
    variants: list[PromptVariant] = []
    for candidate in candidates:
        variants.extend(
            build_species_prompt_variants(
                scientific_name=candidate.scientific_name,
                common_names=candidate.common_names,
            )
        )
    return variants
```

Update pinned target construction to set:

```python
common_names=("lime butterfly", "chequered swallowtail", "citrus swallowtail"),
```

- [ ] **Step 4: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_species_candidates.py tests/test_prompt_templates.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add src/biominer/bioclip/species_candidates.py tests/test_species_candidates.py
git commit -m "step3: add species prompt variants"
```

## Phase 3: Taxon-Level Species Scores

### Task 3: Aggregate Prompt Scores in BioClipClassifier

**Files:**
- Modify: `src/biominer/bioclip/bioclip.py`
- Modify: `src/biominer/bioclip/register_runner.py`
- Modify: `tests/test_bioclip_prediction.py`
- Modify: `tests/test_register_runner.py`

- [ ] **Step 1: Write failing classifier test**

Add to `tests/test_bioclip_prediction.py`:

```python
def test_bioclip_classifier_aggregates_species_prompt_variants() -> None:
    from biominer.bioclip.prompt_templates import PromptVariant

    class FakeScorer:
        def score_label_sets_batch(self, image_paths, label_sets):  # noqa: ANN001
            return {
                "species": [{
                    "a photo of Papilio demoleus": 0.62,
                    "a photo of lime butterfly": 0.84,
                    "a photo of Papilio machaon": 0.21,
                }],
                "triage": [{"a photo of an adult butterfly": 0.91}],
            }

    classifier = BioClipClassifier(runtime=_runtime(), scorer=FakeScorer())
    records = classifier.classify_images_with_label_sets(
        [{
            "flickr_photo_id": "1",
            "image_path": "/tmp/1.jpg",
            "image_hash": "sha256:image",
            "image_url_used": "https://live.staticflickr.com/1.jpg",
            "resolved_scientific_name": "Papilio demoleus",
            "text_evidence_present": True,
        }],
        label_sets={
            "species": [
                "a photo of Papilio demoleus",
                "a photo of lime butterfly",
                "a photo of Papilio machaon",
            ],
            "triage": ["a photo of an adult butterfly"],
        },
        species_prompt_variants=[
            PromptVariant("a photo of Papilio demoleus", "Papilio demoleus", "scientific"),
            PromptVariant("a photo of lime butterfly", "Papilio demoleus", "common"),
            PromptVariant("a photo of Papilio machaon", "Papilio machaon", "scientific"),
        ],
    )

    record = records[0]
    assert record["species_top1_scientific_name"] == "Papilio demoleus"
    assert record["species_top1_score"] == 0.84
    assert record["species_top1_label"] == "a photo of lime butterfly"
    assert record["species_prompt_topk_json"][0]["prompt_scores"]["a photo of Papilio demoleus"] == 0.62
```

- [ ] **Step 2: Run focused test to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_prediction.py::test_bioclip_classifier_aggregates_species_prompt_variants`

Expected: FAIL because `classify_images_with_label_sets` lacks `species_prompt_variants`.

- [ ] **Step 3: Implement classifier aggregation**

Change `classify_images_with_label_sets` signature:

```python
species_prompt_variants: Sequence[PromptVariant] | None = None,
```

When building species top-k, use:

```python
if label_set_name == "species" and species_prompt_variants:
    aggregated = aggregate_prompt_scores(scores=scores, variants=species_prompt_variants, top_k=top_k)
    topk_by_label_set[label_set_name] = [
        (str(row["best_label"]), float(row["score"])) for row in aggregated
    ]
    prompt_topk_by_label_set[label_set_name] = aggregated
else:
    topk_by_label_set[label_set_name] = sorted(
        ((label, float(scores.get(label, 0.0))) for label in labels),
        key=lambda item: item[1],
        reverse=True,
    )[:top_k]
```

Update `build_label_set_prediction_record` to accept and persist:

```python
species_prompt_topk: Sequence[Mapping[str, object]] = ()
```

Return fields:

```python
"species_prompt_topk_json": [dict(row) for row in species_prompt_topk],
"species_top1_scientific_name": str(species_prompt_topk[0]["taxon_key"]) if species_prompt_topk else None,
```

- [ ] **Step 4: Update register runner to pass prompt variants**

In `register_runner.py` import `species_prompt_variants`.

Build:

```python
species_variants = species_prompt_variants(species_candidates)
label_sets = {
    "species": [variant.label for variant in species_variants],
    "triage": DEFAULT_TRIAGE_LABELS,
}
```

Pass:

```python
predictions = classifier.classify_images_with_label_sets(
    images,
    label_sets=label_sets,
    species_prompt_variants=species_variants,
)
```

Update the protocol signature accordingly.

- [ ] **Step 5: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_prediction.py tests/test_register_runner.py tests/test_species_candidates.py`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:
```bash
git add src/biominer/bioclip/bioclip.py src/biominer/bioclip/register_runner.py tests/test_bioclip_prediction.py tests/test_register_runner.py
git commit -m "step3: aggregate bioclip species prompt scores"
```

## Phase 4: Grouped Triage and Diagnostics

### Task 4: Add Prediction Diagnostics

**Files:**
- Create: `src/biominer/bioclip/diagnostics.py`
- Create: `tests/test_bioclip_diagnostics.py`
- Modify: `src/biominer/bioclip/bioclip.py`

- [ ] **Step 1: Write failing diagnostics tests**

```python
import pytest

from biominer.bioclip.diagnostics import (
    grouped_probability_summary,
    topk_margin,
    probability_entropy,
)


def test_topk_margin_returns_difference_between_top_two_scores() -> None:
    assert topk_margin([{"label": "a", "score": 0.8}, {"label": "b", "score": 0.3}]) == pytest.approx(0.5)


def test_probability_entropy_is_low_for_confident_prediction() -> None:
    assert probability_entropy([0.98, 0.01, 0.01]) < 0.12


def test_grouped_probability_summary_sums_groups_and_keeps_top_group() -> None:
    summary = grouped_probability_summary(
        scores={
            "a photo of an adult butterfly": 0.55,
            "a photo of a swallowtail butterfly": 0.25,
            "a photo of a moth": 0.10,
            "a photo of artwork or illustration": 0.10,
        },
        groups={
            "adult_butterfly": {"a photo of an adult butterfly", "a photo of a swallowtail butterfly"},
            "hard_negative": {"a photo of a moth", "a photo of artwork or illustration"},
        },
    )

    assert summary["top_group"] == "adult_butterfly"
    assert summary["group_scores"]["adult_butterfly"] == pytest.approx(0.80)
    assert summary["group_scores"]["hard_negative"] == pytest.approx(0.20)
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_diagnostics.py`

Expected: FAIL because diagnostics module does not exist.

- [ ] **Step 3: Implement diagnostics**

```python
from __future__ import annotations

from math import log
from typing import Mapping, Sequence


TRIAGE_LABEL_GROUPS = {
    "adult_butterfly": {
        "a photo of an adult butterfly",
        "a photo of a swallowtail butterfly",
        "a photo of a butterfly",
    },
    "life_stage": {
        "a photo of an egg",
        "a photo of a caterpillar",
        "a photo of a larva",
        "a photo of a pupa",
        "a photo of a chrysalis",
    },
    "hard_negative": {
        "a photo of a moth",
        "a photo of a pinned museum specimen",
        "a photo of artwork or illustration",
        "a photo of a tattoo",
        "an ai generated image",
        "a photo of a logo or brand",
        "a photo of an object",
        "a photo of a textile or pattern",
        "a photo of an insect that is not a butterfly or moth",
        "a photo that is not a lepidoptera",
    },
}


def topk_margin(topk: Sequence[Mapping[str, object]]) -> float | None:
    scores = [float(row["score"]) for row in topk if row.get("score") is not None]
    if len(scores) < 2:
        return None
    return scores[0] - scores[1]


def probability_entropy(scores: Sequence[float]) -> float:
    total = sum(max(0.0, float(score)) for score in scores)
    if total <= 0:
        return 0.0
    probabilities = [max(0.0, float(score)) / total for score in scores]
    return -sum(probability * log(probability) for probability in probabilities if probability > 0)


def grouped_probability_summary(
    *,
    scores: Mapping[str, float],
    groups: Mapping[str, set[str]],
) -> dict[str, object]:
    group_scores = {
        group_name: sum(float(scores.get(label, 0.0)) for label in labels)
        for group_name, labels in groups.items()
    }
    top_group = max(group_scores.items(), key=lambda item: item[1])[0] if group_scores else None
    return {"top_group": top_group, "group_scores": group_scores}
```

- [ ] **Step 4: Persist diagnostics in prediction records**

In `build_label_set_prediction_record`, compute:

```python
from biominer.bioclip.diagnostics import topk_margin, probability_entropy

species_scores = [float(row["score"]) for row in species_topk]
triage_scores = [float(row["score"]) for row in triage_topk]
```

Return:

```python
"species_top1_top2_margin": topk_margin(species_topk),
"triage_top1_top2_margin": topk_margin(triage_topk),
"species_topk_entropy": probability_entropy(species_scores),
"triage_topk_entropy": probability_entropy(triage_scores),
```

- [ ] **Step 5: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_diagnostics.py tests/test_bioclip_prediction.py`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:
```bash
git add src/biominer/bioclip/diagnostics.py src/biominer/bioclip/bioclip.py tests/test_bioclip_diagnostics.py tests/test_bioclip_prediction.py
git commit -m "step3: add bioclip prediction diagnostics"
```

### Task 5: Use Grouped Triage in Bucket Rules

**Files:**
- Modify: `src/biominer/bioclip/bioclip.py`
- Modify: `src/biominer/bioclip/triage.py`
- Modify: `tests/test_image_triage.py`
- Modify: `tests/test_bioclip_prediction.py`

- [ ] **Step 1: Write failing triage tests**

Add to `tests/test_image_triage.py`:

```python
def test_high_species_score_with_small_margin_goes_to_review() -> None:
    result = classify_bioclip_triage(
        record={
            "title": "Papilio demoleus",
            "image_url": "https://live.staticflickr.com/1.jpg",
            "latitude": "-27.0",
            "longitude": "153.0",
            "date_taken": "2024-05-06",
        },
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.92,
            "species_top1_top2_margin": 0.02,
            "triage_group_top": "adult_butterfly",
            "triage_group_scores": {"adult_butterfly": 0.91, "hard_negative": 0.04},
        },
    )

    assert result["occurrence_bin"] == "in_review"
    assert result["bin_reason"] == "ambiguous_species_margin"


def test_hard_negative_group_overrides_species_score() -> None:
    result = classify_bioclip_triage(
        record={"title": "Papilio demoleus", "image_url": "https://live.staticflickr.com/1.jpg"},
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.95,
            "species_top1_top2_margin": 0.50,
            "triage_group_top": "hard_negative",
            "triage_group_scores": {"adult_butterfly": 0.08, "hard_negative": 0.87},
        },
    )

    assert result["occurrence_bin"] == "bin"
    assert result["bin_reason"] == "hard_negative_group"
```

- [ ] **Step 2: Run tests to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_image_triage.py`

Expected: FAIL because grouped triage fields are ignored.

- [ ] **Step 3: Persist grouped triage fields**

In `build_label_set_prediction_record`, after receiving raw triage scores, call:

```python
from biominer.bioclip.diagnostics import TRIAGE_LABEL_GROUPS, grouped_probability_summary
```

Use the original triage score mapping from the scorer. Add fields:

```python
"triage_group_top": triage_group_summary["top_group"],
"triage_group_scores": triage_group_summary["group_scores"],
```

- [ ] **Step 4: Update triage rules**

At the start of `classify_bioclip_triage` after `species_top1_score`:

```python
species_margin = _optional_float(prediction.get("species_top1_top2_margin"))
triage_group_top = str(prediction.get("triage_group_top") or "")
triage_group_scores = prediction.get("triage_group_scores") if isinstance(prediction.get("triage_group_scores"), dict) else {}
hard_negative_score = _optional_float(triage_group_scores.get("hard_negative")) if isinstance(triage_group_scores, dict) else None
```

Before existing category logic:

```python
if triage_group_top == "hard_negative" and hard_negative_score is not None and hard_negative_score >= 0.70:
    return _bucket_result(category_defaults(), bucket="bin", reason="hard_negative_group", text_species_match=text_species_match, is_target_positive=False, is_negative_material=True)
if species_top1_score is not None and species_top1_score >= GOLD_SPECIES_CONFIDENCE_THRESHOLD and species_margin is not None and species_margin < 0.05:
    return _bucket_result(category_defaults(), bucket="in_review", reason="ambiguous_species_margin", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)
```

- [ ] **Step 5: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_image_triage.py tests/test_bioclip_prediction.py`

Expected: PASS.

- [ ] **Step 6: Commit**

Run:
```bash
git add src/biominer/bioclip/bioclip.py src/biominer/bioclip/triage.py tests/test_image_triage.py tests/test_bioclip_prediction.py
git commit -m "step3: use grouped triage evidence"
```

## Phase 5: Hierarchical Consistency

### Task 6: Add Family/Genus Consistency Fields

**Files:**
- Modify: `src/biominer/bioclip/species_candidates.py`
- Modify: `src/biominer/bioclip/register_runner.py`
- Modify: `src/biominer/bioclip/triage.py`
- Modify: `tests/test_register_runner.py`
- Modify: `tests/test_image_triage.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_image_triage.py`:

```python
def test_gold_requires_genus_consistency_when_available() -> None:
    result = classify_bioclip_triage(
        record={
            "title": "Papilio demoleus",
            "image_url": "https://live.staticflickr.com/1.jpg",
            "latitude": "-27.0",
            "longitude": "153.0",
            "date_taken": "2024-05-06",
        },
        prediction={
            "species_top1_label": "a photo of Papilio demoleus",
            "species_top1_scientific_name": "Papilio demoleus",
            "species_top1_score": 0.93,
            "species_top1_top2_margin": 0.30,
            "species_top1_genus": "Papilio",
            "species_top1_family": "Papilionidae",
            "genus_top1": "Danaus",
            "family_top1": "Nymphalidae",
        },
    )

    assert result["occurrence_bin"] == "in_review"
    assert result["bin_reason"] == "taxonomy_inconsistent"
```

- [ ] **Step 2: Run test to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_image_triage.py::test_gold_requires_genus_consistency_when_available`

Expected: FAIL because hierarchy fields are ignored.

- [ ] **Step 3: Add candidate lookups**

In `species_candidates.py`, add:

```python
def taxon_metadata_by_scientific_name(candidates: list[SpeciesCandidate]) -> dict[str, dict[str, str | None]]:
    return {
        candidate.scientific_name: {"genus": candidate.genus, "family": candidate.family}
        for candidate in candidates
    }
```

- [ ] **Step 4: Persist species genus/family**

In `register_runner.py`, build metadata lookup and enrich predictions in `_success_row`:

```python
metadata = taxon_metadata_by_scientific_name(species_candidates)
```

Pass metadata into `_success_row`. Add:

```python
species_name = species_by_label.get(str(prediction.get("species_top1_label") or ""))
taxon_metadata = taxon_metadata_by_name.get(species_name or "", {})
enriched_prediction = {
    **prediction,
    "species_top1_scientific_name": species_name,
    "species_top1_genus": taxon_metadata.get("genus"),
    "species_top1_family": taxon_metadata.get("family"),
}
```

- [ ] **Step 5: Add triage consistency check**

In `classify_bioclip_triage`, before Gold/Silver assignment:

```python
if _taxonomy_inconsistent(prediction):
    return _bucket_result(category, bucket="in_review", reason="taxonomy_inconsistent", text_species_match=text_species_match, is_target_positive=False, is_negative_material=False)
```

Add helper:

```python
def _taxonomy_inconsistent(prediction: dict[str, object]) -> bool:
    species_genus = str(prediction.get("species_top1_genus") or "")
    genus_top1 = str(prediction.get("genus_top1") or "")
    species_family = str(prediction.get("species_top1_family") or "")
    family_top1 = str(prediction.get("family_top1") or "")
    if species_genus and genus_top1 and species_genus != genus_top1:
        return True
    return bool(species_family and family_top1 and species_family != family_top1)
```

- [ ] **Step 6: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_image_triage.py tests/test_register_runner.py tests/test_species_candidates.py`

Expected: PASS.

- [ ] **Step 7: Commit**

Run:
```bash
git add src/biominer/bioclip/species_candidates.py src/biominer/bioclip/register_runner.py src/biominer/bioclip/triage.py tests/test_register_runner.py tests/test_image_triage.py tests/test_species_candidates.py
git commit -m "step3: add taxonomy consistency checks"
```

## Phase 6: Validation and Reporting

### Task 7: Add Validation Evaluation Script

**Files:**
- Create: `scripts/evaluate_bioclip_species_validation.py`
- Create: `tests/test_bioclip_species_validation_report.py`

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import polars as pl


def test_validation_report_counts_precision_by_bucket(tmp_path) -> None:
    from scripts.evaluate_bioclip_species_validation import evaluate_validation

    predictions = tmp_path / "predictions.parquet"
    reviewed = tmp_path / "reviewed.parquet"
    output = tmp_path / "metrics.json"

    pl.DataFrame([
        {"flickr_photo_id": "1", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.91},
        {"flickr_photo_id": "2", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio machaon", "species_top1_score": 0.88},
        {"flickr_photo_id": "3", "occurrence_bin": "bronze", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.42},
    ]).write_parquet(predictions)
    pl.DataFrame([
        {"flickr_photo_id": "1", "reviewed_species": "Papilio demoleus"},
        {"flickr_photo_id": "2", "reviewed_species": "Papilio demoleus"},
        {"flickr_photo_id": "3", "reviewed_species": "Papilio demoleus"},
    ]).write_parquet(reviewed)

    metrics = evaluate_validation(predictions_path=predictions, reviewed_path=reviewed, output_path=output)

    assert metrics["rows_evaluated"] == 3
    assert metrics["bucket_metrics"]["gold"]["rows"] == 2
    assert metrics["bucket_metrics"]["gold"]["species_precision"] == 0.5
    assert output.exists()
```

- [ ] **Step 2: Run test to verify failure**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_species_validation_report.py`

Expected: FAIL because script does not exist.

- [ ] **Step 3: Implement validation script**

Implement:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def evaluate_validation(*, predictions_path: Path, reviewed_path: Path, output_path: Path) -> dict[str, object]:
    predictions = pl.read_parquet(predictions_path)
    reviewed = pl.read_parquet(reviewed_path)
    joined = predictions.join(reviewed, on="flickr_photo_id", how="inner").with_columns(
        (pl.col("species_top1_scientific_name") == pl.col("reviewed_species")).alias("species_correct")
    )
    bucket_metrics = {}
    for row in (
        joined.group_by("occurrence_bin")
        .agg(
            pl.len().alias("rows"),
            pl.col("species_correct").mean().alias("species_precision"),
            pl.col("species_top1_score").mean().alias("mean_species_score"),
        )
        .to_dicts()
    ):
        bucket_metrics[str(row["occurrence_bin"])] = {
            "rows": int(row["rows"]),
            "species_precision": float(row["species_precision"]),
            "mean_species_score": float(row["mean_species_score"]) if row["mean_species_score"] is not None else None,
        }
    metrics = {
        "rows_evaluated": joined.height,
        "bucket_metrics": bucket_metrics,
    }
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BioCLIP species predictions against reviewed labels.")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--reviewed", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    metrics = evaluate_validation(predictions_path=args.predictions, reviewed_path=args.reviewed, output_path=args.output)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused tests**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_species_validation_report.py`

Expected: PASS.

- [ ] **Step 5: Commit**

Run:
```bash
git add scripts/evaluate_bioclip_species_validation.py tests/test_bioclip_species_validation_report.py
git commit -m "step3: add bioclip validation metrics"
```

### Task 8: Add Diagnostics to Visual Report

**Files:**
- Modify: `scripts/generate_bioclip_species_visual_report.py`
- Modify: `tests/test_bioclip_species_visual_report.py`

- [ ] **Step 1: Write failing tests**

Add to `tests/test_bioclip_species_visual_report.py`:

```python
def test_numeric_summary_handles_margin_and_entropy_columns() -> None:
    report = load_report_module()
    df = pl.DataFrame({
        "species_top1_top2_margin": [0.10, 0.50],
        "species_topk_entropy": [0.20, 1.20],
    })

    summary = {
        "species_margin_stats": report.numeric_summary(df["species_top1_top2_margin"]),
        "species_entropy_stats": report.numeric_summary(df["species_topk_entropy"]),
    }

    assert summary["species_margin_stats"]["median"] == pytest.approx(0.30)
    assert summary["species_entropy_stats"]["max"] == pytest.approx(1.20)
```

- [ ] **Step 2: Run focused test**

Run: `/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_species_visual_report.py`

Expected: FAIL until report summary includes diagnostics fields.

- [ ] **Step 3: Add summary fields**

In `write_tables`, add when columns exist:

```python
if "species_top1_top2_margin" in df.columns:
    summary["species_margin_stats"] = numeric_summary(df["species_top1_top2_margin"])
if "species_topk_entropy" in df.columns:
    summary["species_entropy_stats"] = numeric_summary(df["species_topk_entropy"])
if "triage_group_top" in df.columns:
    summary["triage_groups"] = value_counts_dict(df, "triage_group_top")
```

- [ ] **Step 4: Add grouped triage table**

Add to table map:

```python
if "triage_group_top" in df.columns:
    table_map["triage_group_counts.csv"] = value_count_table(df, "triage_group_top", "triage_group_top")
```

- [ ] **Step 5: Run focused tests and smoke**

Run:
```bash
/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q tests/test_bioclip_species_visual_report.py
/home/toffe/github/karikris/BioMiner/.venv/bin/python scripts/generate_bioclip_species_visual_report.py --predictions /tmp/biominer_visual_smoke/predictions.parquet --candidates /tmp/biominer_visual_smoke/candidates.parquet --output-dir /tmp/biominer_visual_smoke/report
```

Expected: tests PASS and smoke writes HTML/PDF.

- [ ] **Step 6: Commit**

Run:
```bash
git add scripts/generate_bioclip_species_visual_report.py tests/test_bioclip_species_visual_report.py
git commit -m "step3: report bioclip diagnostics"
```

## Phase 7: Final Verification

### Task 9: Full Test, Static Scan, Push

**Files:**
- No code files unless failures reveal required fixes.

- [ ] **Step 1: Run pandas/static scan**

Run:
```bash
rg "import pandas|from pandas|\\bpd\\.|\\.to_pandas\\(" src scripts tests -n
```

Expected: no new pandas usage. Existing intentional dependency-boundary matches must be reviewed and documented before proceeding.

- [ ] **Step 2: Run full tests**

Run:
```bash
/home/toffe/github/karikris/BioMiner/.venv/bin/pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect git status**

Run:
```bash
git status --short
```

Expected: only generated `logs/` and `reports/` artifacts remain untracked; no modified tracked files.

- [ ] **Step 4: Push**

Run:
```bash
git push
```

Expected: current branch pushes successfully.

## Self-Review

Spec coverage:
- Prompt ensembling: Tasks 1-3.
- Grouped triage probabilities: Tasks 4-5.
- Top-1 margin and entropy: Tasks 4-5.
- Hierarchical family/genus/species consistency: Task 6.
- Focused candidate improvements through common names and prompt variants: Tasks 1-3.
- Calibration/validation: Task 7.
- Richer diagnostics and reporting: Tasks 4, 7, 8.
- Frequent commits: every task has its own commit step.
- Polars preference: validation/reporting use Polars and Parquet.

Placeholder scan:
- No `TBD`, `TODO`, `implement later`, or unspecified “write tests” steps remain.

Type consistency:
- `PromptVariant`, `species_prompt_variants`, `aggregate_prompt_scores`, `topk_margin`, `probability_entropy`, and grouped triage field names are introduced before later tasks depend on them.
