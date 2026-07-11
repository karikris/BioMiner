"""Synthetic, model-free taxonomy inputs for path-cascade benchmarks.

The names and identifiers in this module are deliberately invented.  They are
not biological assertions, GBIF records, or a substitute for the reviewed
classification-v3 registry.  The fixture exists only to exercise branching,
optional-rank skips, and large species candidate sets deterministically.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.cascade_contract import DEFAULT_RANK_BEAM_WIDTH
from biominer.bioclip.path_cascade_classifier import (
    INTERMEDIATE_CLASSIFICATION_RANKS,
    PathCascadeResult,
    RankStepResult,
    classify_path_cascade,
)
from biominer.bioclip.path_taxonomy_store import (
    RANK_SCREEN_PROMPT_STAGE,
    SPECIES_RERANK_PROMPT_STAGE,
    PathTaxonomyStore,
)
from biominer.registry.classification_v3 import (
    ASSERTED_PARENT_EDGE,
    CLASSIFICATION_V3_VERSION,
    REVIEWED_RANK_SKIP_EDGE,
    ClassificationV3Frames,
    build_classification_v3_frames,
    build_classification_v3_manifest,
    classification_v3_fingerprint,
    classification_v3_qa_frame,
    hierarchy_fingerprint,
    validate_classification_v3,
)


BENCHMARK_FIXTURE_NOTICE = (
    "Synthetic developer benchmark only; names and identifiers are neither biological "
    "taxonomy nor GBIF authority."
)
BENCHMARK_REGISTRY_VERSION = "synthetic-seven-family-cascade-v1"
BENCHMARK_SOURCE_ID = "fixture:synthetic-taxonomy-v1"
BENCHMARK_REVIEW_DATE = "2026-07-11"
BENCHMARK_KIND = "path_cascade_model_free_global_beam"
BENCHMARK_VERSION = "path-cascade-model-free-v1"
FAMILY_COUNT = 7
SUBFAMILIES_PER_FAMILY = 2
TRIBES_PER_SUBFAMILY = 2
GENERA_PER_BRANCH = 2
SPECIES_PER_SELECTED_GENUS = 25
BENCHMARK_SELECTED_GENUS_NODE_IDS = (
    "fixture:genus:01:01:01:01",
    "fixture:genus:01:01:01:02",
    "fixture:genus:01:01:01:03",
)
_DIVERGENCE_NODE_RAW_SIMILARITIES = {
    "fixture:family:01": 0.10,
    "fixture:family:02": 0.99,
    "fixture:family:03": 0.80,
    "fixture:subfamily:01:01": 0.95,
    "fixture:subfamily:01:02": 0.10,
    "fixture:subfamily:02:01": 0.80,
    "fixture:subfamily:02:02": 0.79,
    "fixture:subfamily:03:01": 0.78,
    "fixture:subfamily:03:02": 0.00,
}


@dataclass(frozen=True)
class SevenFamilyPathCascadeFixture:
    """Validated classification-v3 frames and store for developer benchmarks."""

    taxa: pl.DataFrame
    frames: ClassificationV3Frames
    qa_findings: pl.DataFrame
    manifest: Mapping[str, object]
    taxonomy_store: PathTaxonomyStore


@dataclass(frozen=True)
class PathCascadeBenchmarkResult:
    """Artifacts and metrics from the model-free global-beam benchmark."""

    metrics: dict[str, Any]
    cascade_result: PathCascadeResult
    output_dir: Path
    metrics_path: Path
    summary_path: Path


@dataclass(frozen=True)
class _HistoricalCumulativeCandidate:
    node_id: str
    parent_node_id: str
    scientific_name: str
    parent_raw_similarity: float
    current_rank_raw_similarity: float
    cumulative_path_raw_similarity: float


class DeterministicRawSimilarityScorer:
    """Stable model-free scores derived only from synthetic fixture identities.

    First-pass scores prefer lexicographically earlier synthetic nodes.  The
    species rerank stage deliberately reverses the retained species-number
    order so the benchmark exercises a distinct second scoring pass.
    """

    model_id = "synthetic-path-cascade-scorer"
    model_checkpoint = "model-free-v1"

    def __init__(self, taxonomy_store: PathTaxonomyStore) -> None:
        self._score_by_label: dict[str, float] = {}
        for row in taxonomy_store.prompt_labels.iter_rows(named=True):
            label = str(row["label"])
            score = _deterministic_prompt_score(
                node_id=str(row["node_id"]),
                prompt_stage=str(row["prompt_stage"]),
            )
            existing = self._score_by_label.setdefault(label, score)
            if existing != score:
                raise ValueError(f"synthetic prompt label maps to conflicting scores: {label}")
        self.calls: list[tuple[str, ...]] = []

    def raw_similarities(
        self,
        item: dict[str, Any],
        labels: tuple[str, ...],
    ) -> Mapping[str, float]:
        del item
        requested = tuple(str(label) for label in labels)
        missing = [label for label in requested if label not in self._score_by_label]
        if missing:
            raise ValueError("synthetic scorer received unknown labels: " + ", ".join(missing))
        self.calls.append(requested)
        return {label: self._score_by_label[label] for label in requested}


class _SubfamilyDivergenceScorer(DeterministicRawSimilarityScorer):
    """Fixture-only scorer that makes rank-local and cumulative beams diverge."""

    def __init__(self, taxonomy_store: PathTaxonomyStore) -> None:
        super().__init__(taxonomy_store)
        divergence_prompts = taxonomy_store.prompt_labels.filter(
            (pl.col("prompt_stage") == RANK_SCREEN_PROMPT_STAGE)
            & pl.col("node_id").is_in(_DIVERGENCE_NODE_RAW_SIMILARITIES)
        )
        overridden_nodes = set(divergence_prompts["node_id"].to_list())
        if overridden_nodes != set(_DIVERGENCE_NODE_RAW_SIMILARITIES):
            raise AssertionError("synthetic divergence fixture is missing rank prompts")
        for prompt in divergence_prompts.iter_rows(named=True):
            self._score_by_label[str(prompt["label"])] = (
                _DIVERGENCE_NODE_RAW_SIMILARITIES[str(prompt["node_id"])]
            )


def build_seven_family_path_cascade_fixture() -> SevenFamilyPathCascadeFixture:
    """Build the small, deterministic seven-family path-cascade fixture.

    The first asserted-subtribe branch has three genera with 25 species each.
    Every other branch has two genera with one species each.  This keeps the
    fixture compact while supplying a realistic species top-20 candidate set.
    """
    taxa, source = _synthetic_source_and_taxa()
    frames = build_classification_v3_frames(taxa, source)
    findings = validate_classification_v3(frames, taxa=taxa)
    qa_findings = classification_v3_qa_frame(findings)
    manifest = build_classification_v3_manifest(
        frames,
        registry_version=BENCHMARK_REGISTRY_VERSION,
    )
    fatal_count = sum(finding["severity"] == "fatal" for finding in findings)
    warning_count = sum(finding["severity"] == "warning" for finding in findings)
    manifest.update(
        {
            "created_at": f"{BENCHMARK_REVIEW_DATE}T00:00:00Z",
            "classification_fingerprint": classification_v3_fingerprint(frames),
            "hierarchy_fingerprint": hierarchy_fingerprint(frames),
            "fatal_finding_count": fatal_count,
            "warning_finding_count": warning_count,
            "qa_status": "failed" if fatal_count else "passed",
            "benchmark_fixture": True,
            "authoritative_taxonomy": False,
            "gbif_authority": False,
            "fixture_notice": BENCHMARK_FIXTURE_NOTICE,
            "benchmark_selected_genus_node_ids": list(BENCHMARK_SELECTED_GENUS_NODE_IDS),
        }
    )
    taxonomy_store = PathTaxonomyStore.from_frames(
        sources=frames.sources,
        nodes=frames.nodes,
        edges=frames.edges,
        gbif_mappings=frames.gbif_mappings,
        leaf_paths=frames.leaf_paths,
        prompt_labels=frames.prompt_labels,
        qa_findings=qa_findings,
        manifest=manifest,
    )
    return SevenFamilyPathCascadeFixture(
        taxa=taxa,
        frames=frames,
        qa_findings=qa_findings,
        manifest=taxonomy_store.manifest,
        taxonomy_store=taxonomy_store,
    )


def run_path_cascade_benchmark(*, output_dir: str | Path) -> PathCascadeBenchmarkResult:
    """Run the deterministic global-beam classifier against the synthetic fixture."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    total_start = perf_counter()

    fixture_start = perf_counter()
    fixture = build_seven_family_path_cascade_fixture()
    fixture_seconds = _elapsed(fixture_start)

    classify_start = perf_counter()
    scorer = DeterministicRawSimilarityScorer(fixture.taxonomy_store)
    cascade_result = classify_path_cascade(
        item={"benchmark_item_id": "synthetic-path-cascade-item"},
        scorer=scorer,
        taxonomy_store=fixture.taxonomy_store,
    )
    classify_seconds = _elapsed(classify_start)

    comparison_start = perf_counter()
    subfamily_selection_comparison = _subfamily_selection_comparison(fixture)
    comparison_seconds = _elapsed(comparison_start)

    intermediate_steps = tuple(
        step
        for step in cascade_result.rank_steps
        if step.rank in INTERMEDIATE_CLASSIFICATION_RANKS
    )
    excessive_beam_steps = [
        step.rank
        for step in intermediate_steps
        if step.retained_count > DEFAULT_RANK_BEAM_WIDTH
    ]
    if excessive_beam_steps:
        raise AssertionError(
            "global cascade retained more than three intermediate nodes at: "
            + ", ".join(excessive_beam_steps)
        )

    expected_score_calls = len(cascade_result.rank_steps) + 1
    if len(scorer.calls) != expected_score_calls:
        raise AssertionError(
            "benchmark scorer call count does not match rank and rerank steps: "
            f"expected {expected_score_calls}, observed {len(scorer.calls)}"
        )

    genus_step = next(step for step in cascade_result.rank_steps if step.rank == "GENUS")
    selected_genus_paths = fixture.taxonomy_store.filter_paths_by_rank_nodes(
        fixture.taxonomy_store.enabled_paths(),
        "GENUS",
        genus_step.retained_node_ids,
    )
    species_nodes_beneath_genus_top3 = fixture.taxonomy_store.species_nodes_in_paths(
        selected_genus_paths
    )
    species_beneath_genus_top3 = species_nodes_beneath_genus_top3.height
    first_pass_step = next(
        step for step in cascade_result.rank_steps if step.rank == "SPECIES"
    )
    expected_species_node_ids = set(
        species_nodes_beneath_genus_top3["node_id"].to_list()
    )
    actual_species_node_ids = set(first_pass_step.candidate_node_ids)
    if expected_species_node_ids != actual_species_node_ids:
        raise AssertionError(
            "species first-pass candidates escaped the retained genus top-three paths"
        )
    species_counts_by_genus = {
        str(row["genus_node_id"]): int(row["species_count"])
        for row in selected_genus_paths.group_by("genus_node_id")
        .agg(pl.col("species_node_id").n_unique().alias("species_count"))
        .sort("genus_node_id")
        .iter_rows(named=True)
    }

    rank_metrics = [
        _rank_step_metrics(step=step, labels=scorer.calls[index])
        for index, step in enumerate(cascade_result.rank_steps)
    ]
    rerank_labels = scorer.calls[-1]
    rerank_metrics = _rank_step_metrics(
        step=cascade_result.species_rerank_step,
        labels=rerank_labels,
    )
    unique_labels = {label for call in scorer.calls for label in call}
    elapsed_seconds = _elapsed(total_start)
    metrics_path = output / "benchmark_metrics.json"
    summary_path = output / "benchmark_summary.md"
    metrics: dict[str, Any] = {
        "benchmark_kind": BENCHMARK_KIND,
        "benchmark_version": BENCHMARK_VERSION,
        "status": "ok",
        "benchmark_fixture": True,
        "synthetic_taxonomy": True,
        "authoritative_taxonomy": False,
        "gbif_authority": False,
        "fixture_notice": BENCHMARK_FIXTURE_NOTICE,
        "registry_version": BENCHMARK_REGISTRY_VERSION,
        "classification_version": cascade_result.classification_version,
        "classification_fingerprint": cascade_result.classification_fingerprint,
        "hierarchy_fingerprint": cascade_result.taxonomy_fingerprint,
        "prompt_version": cascade_result.prompt_version,
        "beam_strategy": cascade_result.beam_strategy,
        "rank_beam_width": cascade_result.rank_beam_width,
        "family_candidate_count": rank_metrics[0]["candidate_count"],
        "rank_steps": rank_metrics,
        "species_rerank_step": rerank_metrics,
        "subfamily_selection_comparison": subfamily_selection_comparison,
        "unique_labels_scored": len(unique_labels),
        "genus_top3_node_ids": list(genus_step.retained_node_ids),
        "species_candidates_beneath_genus_top3": species_beneath_genus_top3,
        "species_candidate_node_ids_beneath_genus_top3": list(
            first_pass_step.candidate_node_ids
        ),
        "species_candidate_counts_by_genus": species_counts_by_genus,
        "species_first_pass_candidate_count": first_pass_step.candidate_count,
        "species_first_pass_retained_count": first_pass_step.retained_count,
        "species_rerank_candidate_count": cascade_result.species_rerank_step.candidate_count,
        "species_rerank_retained_count": cascade_result.species_rerank_step.retained_count,
        "reported_species_count": len(cascade_result.species_top3),
        "elapsed_seconds": elapsed_seconds,
        "elapsed_seconds_by_stage": {
            "build_fixture": fixture_seconds,
            "classify": classify_seconds,
            "compare_subfamily_selection": comparison_seconds,
        },
        "artifacts": {
            "metrics": str(metrics_path),
            "summary": str(summary_path),
        },
    }
    _assert_nonnegative_elapsed_telemetry(metrics)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(_benchmark_summary_markdown(metrics), encoding="utf-8")
    return PathCascadeBenchmarkResult(
        metrics=metrics,
        cascade_result=cascade_result,
        output_dir=output,
        metrics_path=metrics_path,
        summary_path=summary_path,
    )


def _subfamily_selection_comparison(
    fixture: SevenFamilyPathCascadeFixture,
) -> dict[str, Any]:
    """Compare production rank-local pruning with the fixture-only historical rule."""
    scorer = _SubfamilyDivergenceScorer(fixture.taxonomy_store)
    production_result = classify_path_cascade(
        item={"benchmark_item_id": "synthetic-subfamily-selection-divergence"},
        scorer=scorer,
        taxonomy_store=fixture.taxonomy_store,
    )
    family_step = next(
        step for step in production_result.rank_steps if step.rank == "FAMILY"
    )
    subfamily_step = next(
        step for step in production_result.rank_steps if step.rank == "SUBFAMILY"
    )
    current_node_ids = subfamily_step.retained_node_ids
    historical_candidates = _historical_cumulative_subfamily_selection(
        fixture=fixture,
        family_step=family_step,
        subfamily_step=subfamily_step,
    )
    historical_node_ids = tuple(candidate.node_id for candidate in historical_candidates)
    if current_node_ids == historical_node_ids:
        raise AssertionError("comparison fixture no longer makes subfamily beams diverge")
    return {
        "fixture_only": True,
        "rank": "SUBFAMILY",
        "beam_width": DEFAULT_RANK_BEAM_WIDTH,
        "production_beam_strategy": production_result.beam_strategy,
        "production_score_basis": "current_rank_raw_similarity_only",
        "historical_score_basis": "mean_family_and_subfamily_raw_similarity",
        "production_selected_node_ids": list(current_node_ids),
        "historical_selected_node_ids": list(historical_node_ids),
        "production_candidates": [
            {
                "node_id": node_id,
                "current_rank_raw_similarity": raw_similarity,
            }
            for node_id, raw_similarity in zip(
                subfamily_step.candidate_node_ids,
                subfamily_step.candidate_raw_similarities,
                strict=True,
            )
        ],
        "historical_candidates": [
            {
                "node_id": candidate.node_id,
                "parent_node_id": candidate.parent_node_id,
                "scientific_name": candidate.scientific_name,
                "parent_raw_similarity": candidate.parent_raw_similarity,
                "current_rank_raw_similarity": candidate.current_rank_raw_similarity,
                "cumulative_path_raw_similarity": (
                    candidate.cumulative_path_raw_similarity
                ),
            }
            for candidate in historical_candidates
        ],
        "selections_differ": True,
    }


def _historical_cumulative_subfamily_selection(
    *,
    fixture: SevenFamilyPathCascadeFixture,
    family_step: RankStepResult,
    subfamily_step: RankStepResult,
) -> tuple[_HistoricalCumulativeCandidate, ...]:
    """Reproduce the retired cumulative-path SUBFAMILY beam for this fixture only."""
    if (
        fixture.manifest.get("benchmark_fixture") is not True
        or fixture.manifest.get("registry_version") != BENCHMARK_REGISTRY_VERSION
    ):
        raise ValueError("historical comparison is restricted to the synthetic benchmark fixture")
    if family_step.rank != "FAMILY" or subfamily_step.rank != "SUBFAMILY":
        raise ValueError("historical comparison requires FAMILY and SUBFAMILY rank steps")
    family_score_by_node = dict(
        zip(
            family_step.candidate_node_ids,
            family_step.candidate_raw_similarities,
            strict=True,
        )
    )
    subfamily_score_by_node = dict(
        zip(
            subfamily_step.candidate_node_ids,
            subfamily_step.candidate_raw_similarities,
            strict=True,
        )
    )
    subfamily_names = {
        str(row["node_id"]): str(row["scientific_name"])
        for row in fixture.frames.nodes.filter(pl.col("rank") == "SUBFAMILY").iter_rows(
            named=True
        )
    }
    selected_family_ids = set(family_step.retained_node_ids)
    family_subfamily_pairs = (
        fixture.taxonomy_store.enabled_paths()
        .filter(pl.col("family_node_id").is_in(selected_family_ids))
        .select("family_node_id", "subfamily_node_id")
        .unique()
        .sort("family_node_id", "subfamily_node_id")
    )
    paths: list[_HistoricalCumulativeCandidate] = []
    for row in family_subfamily_pairs.iter_rows(named=True):
        family_node_id = str(row["family_node_id"])
        subfamily_node_id = str(row["subfamily_node_id"])
        family_similarity = family_score_by_node[family_node_id]
        subfamily_similarity = subfamily_score_by_node[subfamily_node_id]
        paths.append(
            _HistoricalCumulativeCandidate(
                node_id=subfamily_node_id,
                parent_node_id=family_node_id,
                scientific_name=subfamily_names[subfamily_node_id],
                parent_raw_similarity=family_similarity,
                current_rank_raw_similarity=subfamily_similarity,
                cumulative_path_raw_similarity=(
                    family_similarity + subfamily_similarity
                )
                / 2.0,
            )
        )
    paths.sort(
        key=lambda candidate: (
            -candidate.cumulative_path_raw_similarity,
            candidate.scientific_name,
            candidate.node_id,
            candidate.parent_node_id,
        )
    )
    return tuple(paths[:DEFAULT_RANK_BEAM_WIDTH])


def _synthetic_source_and_taxa() -> tuple[pl.DataFrame, dict[str, object]]:
    nodes: list[dict[str, object]] = []
    edges: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    taxa: list[dict[str, object]] = []

    for family_number in range(1, FAMILY_COUNT + 1):
        family_id = f"fixture:family:{family_number:02d}"
        family_name = f"FixtureFamily{family_number:02d}idae"
        nodes.append(_node(family_id, "FAMILY", family_name))
        for subfamily_number in range(1, SUBFAMILIES_PER_FAMILY + 1):
            subfamily_id = f"fixture:subfamily:{family_number:02d}:{subfamily_number:02d}"
            subfamily_name = (
                f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}inae"
            )
            nodes.append(_node(subfamily_id, "SUBFAMILY", subfamily_name))
            edges.append(_asserted_edge(family_id, subfamily_id))
            for tribe_number in range(1, TRIBES_PER_SUBFAMILY + 1):
                tribe_id = (
                    f"fixture:tribe:{family_number:02d}:{subfamily_number:02d}:{tribe_number:02d}"
                )
                tribe_name = (
                    f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}"
                    f"Tribe{tribe_number:02d}ini"
                )
                nodes.append(_node(tribe_id, "TRIBE", tribe_name))
                edges.append(_asserted_edge(subfamily_id, tribe_id))

                genus_parent_id = tribe_id
                if tribe_number == 1:
                    subtribe_id = (
                        f"fixture:subtribe:{family_number:02d}:{subfamily_number:02d}:"
                        f"{tribe_number:02d}"
                    )
                    subtribe_name = (
                        f"FixtureFamily{family_number:02d}Subfamily{subfamily_number:02d}"
                        f"Tribe{tribe_number:02d}ina"
                    )
                    nodes.append(_node(subtribe_id, "SUBTRIBE", subtribe_name))
                    edges.append(_asserted_edge(tribe_id, subtribe_id))
                    genus_parent_id = subtribe_id

                genus_count = (
                    3
                    if (family_number, subfamily_number, tribe_number) == (1, 1, 1)
                    else GENERA_PER_BRANCH
                )
                for genus_number in range(1, genus_count + 1):
                    genus_id = (
                        f"fixture:genus:{family_number:02d}:{subfamily_number:02d}:"
                        f"{tribe_number:02d}:{genus_number:02d}"
                    )
                    genus_name = (
                        f"FixtureGenus{family_number:02d}{subfamily_number:02d}"
                        f"{tribe_number:02d}{genus_number:02d}"
                    )
                    nodes.append(_node(genus_id, "GENUS", genus_name))
                    if tribe_number == 1:
                        edges.append(_asserted_edge(genus_parent_id, genus_id))
                    else:
                        edges.append(_reviewed_subtribe_skip_edge(tribe_id, genus_id))

                    species_count = (
                        SPECIES_PER_SELECTED_GENUS
                        if genus_id in BENCHMARK_SELECTED_GENUS_NODE_IDS
                        else 1
                    )
                    for species_number in range(1, species_count + 1):
                        species_id = f"{genus_id.replace(':genus:', ':species:')}:{species_number:02d}"
                        species_name = f"{genus_name} specimen{species_number:02d}"
                        synthetic_key = (
                            f"synthetic-{family_number:02d}{subfamily_number:02d}"
                            f"{tribe_number:02d}{genus_number:02d}{species_number:02d}"
                        )
                        nodes.append(_node(species_id, "SPECIES", species_name))
                        edges.append(_asserted_edge(genus_id, species_id))
                        mappings.append(_mapping(synthetic_key, species_name, species_id))
                        taxa.append(
                            {
                                "registry_version": BENCHMARK_REGISTRY_VERSION,
                                "accepted_taxon_key": synthetic_key,
                                "species_key": synthetic_key,
                                "scientific_name": species_name,
                                "family": family_name,
                                "genus": genus_name,
                                "species": species_name,
                                "rank": "SPECIES",
                                "taxonomic_status": "ACCEPTED",
                            }
                        )

    return pl.DataFrame(taxa), {
        "classification_version": CLASSIFICATION_V3_VERSION,
        "sources": [
            {
                "source_id": BENCHMARK_SOURCE_ID,
                "authority": "Synthetic benchmark fixture; not biological or GBIF authority",
                "release": BENCHMARK_REGISTRY_VERSION,
                "citation": BENCHMARK_FIXTURE_NOTICE,
                "retrieved_at": BENCHMARK_REVIEW_DATE,
                "evidence_url": "https://example.invalid/biominer/synthetic-path-cascade",
                "evidence": BENCHMARK_FIXTURE_NOTICE,
            }
        ],
        "nodes": nodes,
        "edges": edges,
        "species_mappings": mappings,
    }


def _node(node_id: str, rank: str, scientific_name: str) -> dict[str, object]:
    return {
        "node_id": node_id,
        "rank": rank,
        "scientific_name": scientific_name,
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _asserted_edge(parent_node_id: str, child_node_id: str) -> dict[str, object]:
    return {
        "parent_node_id": parent_node_id,
        "child_node_id": child_node_id,
        "edge_type": ASSERTED_PARENT_EDGE,
        "skipped_ranks": [],
        "skip_reason": "",
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _reviewed_subtribe_skip_edge(parent_node_id: str, child_node_id: str) -> dict[str, object]:
    return {
        "parent_node_id": parent_node_id,
        "child_node_id": child_node_id,
        "edge_type": REVIEWED_RANK_SKIP_EDGE,
        "skipped_ranks": ["SUBTRIBE"],
        "skip_reason": "Synthetic branch explicitly exercises the optional SUBTRIBE skip contract.",
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": BENCHMARK_FIXTURE_NOTICE,
        **_review(),
    }


def _mapping(
    synthetic_key: str,
    accepted_scientific_name: str,
    species_node_id: str,
) -> dict[str, object]:
    return {
        "gbif_species_key": synthetic_key,
        "accepted_scientific_name": accepted_scientific_name,
        "species_node_id": species_node_id,
        "source_id": BENCHMARK_SOURCE_ID,
        "evidence": (
            "Synthetic identity mapping required by the classification-v3 test schema; "
            "not a GBIF identifier or assertion."
        ),
        **_review(),
    }


def _review() -> dict[str, object]:
    return {
        "reviewed": True,
        "review_status": "reviewed",
        "reviewed_by": "BioMiner synthetic benchmark builder",
        "reviewed_at": BENCHMARK_REVIEW_DATE,
        "enabled": True,
    }


def _deterministic_prompt_score(*, node_id: str, prompt_stage: str) -> float:
    try:
        identity_numbers = tuple(int(part) for part in node_id.split(":")[2:])
    except ValueError as exc:
        raise ValueError(f"synthetic benchmark node has a nonnumeric identity: {node_id}") from exc
    if not identity_numbers:
        raise ValueError(f"synthetic benchmark node has no numeric identity: {node_id}")
    if prompt_stage == SPECIES_RERANK_PROMPT_STAGE:
        score = float(identity_numbers[-1]) / 100.0
    else:
        encoded_identity = sum(
            value * (100 ** (len(identity_numbers) - index - 1))
            for index, value in enumerate(identity_numbers)
        )
        score = -float(encoded_identity) / 1_000_000_000.0
    if not isfinite(score):
        raise ValueError(f"synthetic benchmark score is not finite for node: {node_id}")
    return score


def _rank_step_metrics(
    *,
    step: RankStepResult,
    labels: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "rank": step.rank,
        "prompt_stage": step.prompt_stage,
        "candidate_count": step.candidate_count,
        "retained_node_count": step.retained_count,
        "active_path_count_before": step.active_path_count_before,
        "active_path_count_after": step.active_path_count_after,
        "labels_scored": len(labels),
        "unique_labels_scored": len(set(labels)),
        "reviewed_skip_path_count": step.reviewed_skip_path_count,
        "skipped": step.skipped,
        "parent_node_ids": list(step.parent_node_ids),
        "candidate_node_ids": list(step.candidate_node_ids),
        "candidate_raw_similarities": list(step.candidate_raw_similarities),
        "retained_node_ids": list(step.retained_node_ids),
        "pruned_node_ids": list(step.pruned_node_ids),
    }


def _assert_nonnegative_elapsed_telemetry(metrics: Mapping[str, Any]) -> None:
    elapsed_seconds = float(metrics["elapsed_seconds"])
    stage_seconds = metrics["elapsed_seconds_by_stage"]
    if elapsed_seconds < 0 or any(float(value) < 0 for value in stage_seconds.values()):
        raise AssertionError("benchmark elapsed telemetry must be nonnegative")


def _benchmark_summary_markdown(metrics: Mapping[str, Any]) -> str:
    comparison = metrics["subfamily_selection_comparison"]
    rows = [
        "| Rank/stage | Candidates | Retained | Paths before | Paths after | Labels scored |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for step in metrics["rank_steps"]:
        rows.append(
            f"| {step['rank']} / {step['prompt_stage']} | {step['candidate_count']} | "
            f"{step['retained_node_count']} | {step['active_path_count_before']} | "
            f"{step['active_path_count_after']} | {step['labels_scored']} |"
        )
    rerank = metrics["species_rerank_step"]
    rows.append(
        f"| {rerank['rank']} / {rerank['prompt_stage']} | {rerank['candidate_count']} | "
        f"{rerank['retained_node_count']} | {rerank['active_path_count_before']} | "
        f"{rerank['active_path_count_after']} | {rerank['labels_scored']} |"
    )
    return "\n".join(
        [
            "# Model-free global-beam cascade benchmark",
            "",
            f"> {metrics['fixture_notice']}",
            "",
            f"- Beam strategy: `{metrics['beam_strategy']}`",
            f"- Fixed intermediate width: {metrics['rank_beam_width']}",
            (
                "- Species candidates beneath genus top 3: "
                f"{metrics['species_candidates_beneath_genus_top3']}"
            ),
            (
                "- Species first pass: "
                f"{metrics['species_first_pass_candidate_count']} candidates, "
                f"{metrics['species_first_pass_retained_count']} retained"
            ),
            (
                "- Species rerank: "
                f"{metrics['species_rerank_candidate_count']} candidates, "
                f"{metrics['species_rerank_retained_count']} retained"
            ),
            f"- Unique labels scored: {metrics['unique_labels_scored']}",
            f"- Elapsed seconds: {metrics['elapsed_seconds']:.6f}",
            "",
            "## SUBFAMILY selection regression",
            "",
            (
                "- Production global current-rank top 3: `"
                + "`, `".join(comparison["production_selected_node_ids"])
                + "`"
            ),
            (
                "- Historical cumulative-path top 3: `"
                + "`, `".join(comparison["historical_selected_node_ids"])
                + "`"
            ),
            f"- Selections differ: {str(comparison['selections_differ']).casefold()}",
            "- Historical reproduction scope: synthetic benchmark fixture only",
            "",
            *rows,
            "",
        ]
    )


def _elapsed(started_at: float) -> float:
    return max(0.0, perf_counter() - started_at)
