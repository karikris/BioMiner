from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass, fields, replace
from itertools import chain

import polars as pl
import pytest

from biominer.bioclip.path_cascade_classifier import (
    GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
    PathCascadeClassificationError,
    RankCandidateScore,
    classify_path_cascade,
    score_rank_candidates,
)
from biominer.bioclip.path_taxonomy_store import PathTaxonomyStore
from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
    EDGE_SCHEMA,
    GBIF_MAPPING_SCHEMA,
    LEAF_PATH_SCHEMA,
    NODE_SCHEMA,
    PROMPT_LABEL_SCHEMA,
    QA_FINDING_SCHEMA,
    SOURCE_SCHEMA,
)


class _RawScorer:
    model_id = "fake-raw-bioclip"
    model_checkpoint = "fake-raw-checkpoint"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, ...]] = []

    def raw_similarities(self, item, labels):  # noqa: ANN001, ANN201 - protocol fake.
        self.calls.append(tuple(labels))
        return {label: self.scores[label] for label in labels}


def test_result_models_exclude_complete_path_selection_scores() -> None:
    names = {field.name for field in fields(RankCandidateScore)}
    assert {
        "node_id",
        "rank",
        "scientific_name",
        "raw_similarity",
        "best_label",
        "best_label_similarity",
        "label_count",
    } <= names
    assert "score" not in names
    assert not any("cumulative" in name for name in names)
    score = RankCandidateScore("n", "FAMILY", "Name", 0.5, "label", 0.5, 1)
    with pytest.raises(FrozenInstanceError):
        score.raw_similarity = 0.9  # type: ignore[misc]


def test_rank_scoring_means_prompt_variants_and_breaks_ties_by_stable_identity() -> None:
    store = _store()
    extra = store.prompt_labels.filter(pl.col("node_id") == "f:a").with_columns(
        pl.lit("variant::f:a").alias("label"),
        pl.lit("variant::{name}").alias("prompt_template"),
        pl.lit(2, dtype=pl.Int64).alias("sort_order"),
    )
    store = replace(store, prompt_labels=pl.concat([store.prompt_labels, extra]))
    candidates = store.rank_candidates("FAMILY")
    scorer = _scorer(
        store,
        {"f:a": 0.9, "f:b": 0.6, "f:c": -0.2, "f:d": -0.3},
        label_overrides={"variant::f:a": 0.1},
    )

    scores = score_rank_candidates(
        item={},
        scorer=scorer,
        taxonomy_store=store,
        candidates=candidates,
    )

    assert [score.node_id for score in scores[:2]] == ["f:b", "f:a"]
    alpha = next(score for score in scores if score.node_id == "f:a")
    assert alpha.raw_similarity == pytest.approx(0.5)
    assert alpha.best_label == _label("f:a")
    assert alpha.best_label_similarity == 0.9
    assert alpha.label_count == 2

    same_name = store.nodes.filter(pl.col("node_id").is_in(["g:a1b", "g:a2u"]))
    tied = score_rank_candidates(
        item={},
        scorer=_scorer(store, {"g:a1b": 0.4, "g:a2u": 0.4}),
        taxonomy_store=store,
        candidates=same_name.reverse(),
    )
    assert [score.node_id for score in tied] == ["g:a1b", "g:a2u"]


def test_global_subfamily_beam_can_retain_three_nodes_from_one_family() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.9,
            "f:b": 0.8,
            "f:c": 0.7,
            "f:d": 0.6,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "sf:a4": 0.96,
            "sf:b1": 0.95,
            "sf:b2": 0.94,
            "sf:c1": 0.93,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SUBFAMILY")

    assert step.candidate_count == 7
    assert step.retained_count == 3
    assert [score.node_id for score in step.top_candidates] == ["sf:a1", "sf:a2", "sf:a3"]
    assert result.beam_strategy == GLOBAL_RANK_TOP_K_BEAM_STRATEGY
    assert result.rank_beam_width == 3
    assert result.taxonomy_fingerprint == store.hierarchy_fingerprint


def test_current_rank_child_score_beats_higher_prior_branch_score() -> None:
    store = _store()
    result = classify_path_cascade(
        item={},
        taxonomy_store=store,
        scorer=_scorer(
            store,
            {
                "f:a": 0.10,
                "f:b": 0.99,
                "f:c": 0.80,
                "f:d": 0.01,
                "sf:a1": 0.95,
                "sf:b2": 0.80,
                "sf:b1": 0.79,
                "sf:c1": 0.78,
                "sf:a2": 0.20,
                "sf:a3": 0.19,
                "sf:a4": 0.18,
            },
        ),
    )

    assert [score.node_id for score in _step(result, "SUBFAMILY").top_candidates] == [
        "sf:a1",
        "sf:b2",
        "sf:b1",
    ]


def test_repeated_leaf_paths_score_each_ancestor_node_once() -> None:
    store = _only_species_prefix(_store(), "s:a1:")
    scorer = _scorer(store, {})

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)

    for rank in ("FAMILY", "SUBFAMILY", "TRIBE", "SUBTRIBE", "GENUS"):
        assert _step(result, rank).candidate_count == 1
    labels = list(chain.from_iterable(scorer.calls))
    for node_id in ("f:a", "sf:a1", "t:a1", "u:a1", "g:a1"):
        assert labels.count(_label(node_id)) == 1
    assert _step(result, "SPECIES").candidate_count == 25


def test_family_rank_three_preserves_true_branch_but_rank_four_cannot_reenter() -> None:
    store = _store()
    true_scores = {
        "f:a": 0.99,
        "f:b": 0.90,
        "f:c": 0.80,
        "f:d": 0.70,
        "sf:c1": 0.99,
        "sf:a1": 0.90,
        "sf:b1": 0.80,
        "t:c1": 0.99,
        "t:a1": 0.90,
        "t:b1": 0.80,
        "g:c1": 0.99,
        "g:a1": 0.90,
        "g:b1": 0.80,
        "s:c1": 0.99,
    }
    result = classify_path_cascade(
        item={},
        scorer=_scorer(store, true_scores),
        taxonomy_store=store,
    )

    assert _step(result, "FAMILY").top_candidates[0].node_id == "f:a"
    assert result.species_top1 is not None and result.species_top1.node_id == "s:c1"
    assert result.final_winning_path[0].node_id == "f:c"

    pruned_scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.90,
            "f:c": 0.80,
            "f:d": 0.70,
            "sf:d1": 1.0,
            "t:d1": 1.0,
            "g:d1": 1.0,
            "s:d1": 1.0,
        },
    )
    pruned = classify_path_cascade(item={}, scorer=pruned_scorer, taxonomy_store=store)
    called_labels = set(chain.from_iterable(pruned_scorer.calls[1:]))
    assert _label("sf:d1") not in called_labels
    assert _label("s:d1") not in called_labels
    assert all(score.node_id != "s:d1" for score in pruned.species_top20)


def test_genus_beam_is_three_and_excluded_genus_species_are_never_scored() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.20,
            "f:c": 0.19,
            "f:d": 0.18,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "t:a1": 0.99,
            "t:a2": 0.98,
            "t:a3": 0.97,
            "u:a1": 0.99,
            "u:a2": 0.98,
            "u:a3": 0.97,
            "g:a1": 0.90,
            "g:a1b": 0.80,
            "g:a2": 0.70,
            "g:a3": 0.60,
            "s:a3": 1.0,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    genus = _step(result, "GENUS")

    assert genus.candidate_count == 5
    assert genus.retained_count == 3
    assert [score.node_id for score in genus.top_candidates] == ["g:a1", "g:a1b", "g:a2"]
    assert _label("s:a3") not in set(scorer.calls[-1])
    assert all(score.node_id != "s:a3" for score in result.species_top20)


def test_reviewed_subtribe_skip_does_not_consume_a_beam_slot() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:a": 0.99,
            "f:b": 0.20,
            "f:c": 0.19,
            "f:d": 0.18,
            "sf:a1": 0.99,
            "sf:a2": 0.98,
            "sf:a3": 0.97,
            "t:a1": 0.99,
            "t:a2": 0.98,
            "t:a3": 0.97,
            "u:a1": 0.99,
            "u:a1-2": 0.98,
            "u:a1-3": 0.97,
            "u:a1-4": 0.20,
            "u:a2": 0.19,
            "u:a3": 0.18,
            "g:askip": 0.99,
            "g:a1": 0.90,
            "g:a2u": 0.80,
            "s:askip": 1.0,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    subtribe = _step(result, "SUBTRIBE")
    genus_call = next(call for call in scorer.calls if _label("g:askip") in call)

    assert subtribe.skipped is False
    assert subtribe.retained_count == 3
    assert [score.node_id for score in subtribe.top_candidates] == ["u:a1", "u:a1-2", "u:a1-3"]
    assert _label("g:askip") in genus_call
    assert _label("g:a4u") not in genus_call
    assert result.species_top1 is not None and result.species_top1.node_id == "s:askip"
    assert result.skipped_ranks == ("SUBTRIBE",)


def test_fully_skipped_subtribe_records_skip_without_scoring_placeholders() -> None:
    store = _store()
    scorer = _scorer(
        store,
        {
            "f:b": 0.99,
            "f:c": 0.98,
            "f:d": 0.97,
            "f:a": 0.10,
            "sf:b1": 0.99,
            "sf:c1": 0.98,
            "sf:d1": 0.97,
            "sf:b2": 0.10,
            "t:b1": 0.99,
            "t:c1": 0.98,
            "t:d1": 0.97,
        },
    )

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SUBTRIBE")

    assert step.skipped is True
    assert step.retained_count == 0
    assert step.active_path_count_before == step.active_path_count_after == 3
    assert step.skip_reason == "all_active_paths_reviewed_rank_skip"
    assert len(scorer.calls) == 5  # FAMILY, SUBFAMILY, TRIBE, GENUS, SPECIES


def test_species_top_twenty_uses_species_score_only_beneath_genus_top_three() -> None:
    store = _store()
    overrides = {
        "f:a": 0.99,
        "f:b": 0.20,
        "f:c": 0.19,
        "f:d": 0.18,
        "sf:a1": 0.99,
        "sf:a2": 0.98,
        "sf:a3": 0.97,
        "t:a1": 0.99,
        "t:a2": 0.98,
        "t:a3": 0.97,
        "u:a1": 0.99,
        "u:a2": 0.98,
        "u:a3": 0.97,
        "g:a1": 0.90,
        "g:a2": 0.80,
        "g:a3": 0.70,
        "g:a1b": 0.60,
        "s:a2": 1.0,
        "s:a3": 0.10,
    }
    overrides.update({f"s:a1:{index:02d}": 0.90 - index * 0.01 for index in range(25)})
    scorer = _scorer(store, overrides)

    result = classify_path_cascade(item={}, scorer=scorer, taxonomy_store=store)
    step = _step(result, "SPECIES")

    assert step.candidate_count == 27
    assert step.retained_count == 20
    assert result.species_top1 is not None and result.species_top1.node_id == "s:a2"
    assert result.species_top1.raw_similarity == 1.0
    assert result.species_top20[1].node_id == "s:a1:00"
    assert result.species_top5 == result.species_top20[:5]
    assert result.species_top3 == result.species_top20[:3]
    assert step.top1_margin == pytest.approx(0.10)
    assert len(scorer.calls[-1]) == 27


@pytest.mark.parametrize("mode", ["missing", "duplicate", "nonaccepted"])
def test_retained_species_requires_exactly_one_accepted_gbif_mapping(mode: str) -> None:
    store = _only_species_prefix(_store(), "s:a1:00")
    species_id = "s:a1:00"
    if mode == "missing":
        mappings = store.gbif_mappings.filter(pl.col("species_node_id") != species_id)
    elif mode == "duplicate":
        row = store.gbif_mappings.filter(pl.col("species_node_id") == species_id)
        mappings = pl.concat([store.gbif_mappings, row])
    else:
        mappings = store.gbif_mappings.with_columns(
            pl.when(pl.col("species_node_id") == species_id)
            .then(pl.lit("DOUBTFUL"))
            .otherwise(pl.col("taxonomic_status"))
            .alias("taxonomic_status")
        )
    store = replace(store, gbif_mappings=mappings)

    with pytest.raises(PathCascadeClassificationError) as captured:
        classify_path_cascade(item={}, scorer=_scorer(store, {}), taxonomy_store=store)

    assert captured.value.code == "invalid_species_mapping"
    assert captured.value.rank == "SPECIES"


def test_missing_mandatory_rank_candidates_raise_structured_failure() -> None:
    store = _store()
    nodes = store.nodes.with_columns(
        pl.when(pl.col("rank") == "SUBFAMILY")
        .then(pl.lit(False))
        .otherwise(pl.col("enabled"))
        .alias("enabled")
    )
    store = replace(store, nodes=nodes)

    with pytest.raises(PathCascadeClassificationError) as captured:
        classify_path_cascade(item={}, scorer=_scorer(store, {}), taxonomy_store=store)

    assert captured.value.as_dict() == {
        "code": "no_rank_candidates",
        "rank": "SUBFAMILY",
        "candidate_count": 0,
        "active_path_count": 36,
        "message": "mandatory rank has no candidates in the active path union",
    }


def _step(result, rank):  # noqa: ANN001, ANN202 - compact test helper.
    return next(step for step in result.rank_steps if step.rank == rank)


def _scorer(
    store: PathTaxonomyStore,
    overrides: dict[str, float],
    *,
    label_overrides: dict[str, float] | None = None,
) -> _RawScorer:
    scores = {
        str(row["label"]): float(overrides.get(str(row["node_id"]), 0.01))
        for row in store.prompt_labels.iter_rows(named=True)
    }
    scores.update(label_overrides or {})
    return _RawScorer(scores)


def _label(node_id: str) -> str:
    return f"prompt::{node_id}"


def _only_species_prefix(store: PathTaxonomyStore, prefix: str) -> PathTaxonomyStore:
    paths = store.leaf_paths.filter(pl.col("species_node_id").str.starts_with(prefix))
    return replace(
        store,
        leaf_paths=paths,
        _enabled_hierarchy_hashes=frozenset(paths["hierarchy_hash"].to_list()),
    )


@dataclass(frozen=True)
class _PathSpec:
    family: tuple[str, str]
    subfamily: tuple[str, str]
    tribe: tuple[str, str]
    subtribe: tuple[str, str] | None
    genus: tuple[str, str]
    species: tuple[str, str]
    gbif_key: str


def _store() -> PathTaxonomyStore:
    nodes: dict[str, dict[str, object]] = {}
    paths: list[dict[str, object]] = []
    mappings: list[dict[str, object]] = []
    for spec in _path_specs():
        ranked = [
            ("FAMILY", spec.family),
            ("SUBFAMILY", spec.subfamily),
            ("TRIBE", spec.tribe),
        ]
        if spec.subtribe is not None:
            ranked.append(("SUBTRIBE", spec.subtribe))
        ranked.extend([("GENUS", spec.genus), ("SPECIES", spec.species)])
        for rank, (node_id, name) in ranked:
            nodes.setdefault(
                node_id,
                {
                    "classification_version": CLASSIFICATION_V3_VERSION,
                    "node_id": node_id,
                    "rank": rank,
                    "scientific_name": name,
                    "source_id": "fixture:v1",
                    "source_release": "fixture-v1",
                    "citation": "Phase 3 fixture",
                    "retrieved_at": "2026-07-11",
                    "evidence": "reviewed synthetic hierarchy",
                    "reviewed": True,
                    "review_status": "reviewed",
                    "reviewed_by": "Phase 3 test",
                    "reviewed_at": "2026-07-11",
                    "enabled": True,
                    "disabled_reason": "",
                },
            )
        by_rank = {rank: value for rank, value in ranked}
        rank_path = [rank for rank, _value in ranked]
        rank_path_node_ids = [value[0] for _rank, value in ranked]
        skipped = ["SUBTRIBE"] if spec.subtribe is None else []
        row: dict[str, object] = {
            "classification_version": CLASSIFICATION_V3_VERSION,
            "accepted_taxon_key": f"gbif:{spec.gbif_key}",
            "gbif_species_key": spec.gbif_key,
            "rank_path": rank_path,
            "rank_path_node_ids": rank_path_node_ids,
            "skipped_ranks": skipped,
            "path_completeness": "reviewed_optional_skip" if skipped else "complete",
            "hierarchy_hash": f"sha256:{spec.species[0]}",
            "source_release": "fixture-v1",
            "enabled": True,
            "disabled_reason": "",
        }
        for rank in CLASSIFICATION_RANKS:
            value = by_rank.get(rank, ("", ""))
            row[f"{rank.casefold()}_node_id"] = value[0]
            row[rank.casefold()] = value[1]
        paths.append(row)
        mappings.append(
            {
                "classification_version": CLASSIFICATION_V3_VERSION,
                "accepted_taxon_key": f"gbif:{spec.gbif_key}",
                "gbif_species_key": spec.gbif_key,
                "accepted_scientific_name": spec.species[1],
                "species_node_id": spec.species[0],
                "taxonomic_status": "ACCEPTED",
                "source_id": "fixture:v1",
                "source_release": "fixture-v1",
                "citation": "Phase 3 fixture",
                "retrieved_at": "2026-07-11",
                "evidence": "exact accepted fixture mapping",
                "reviewed": True,
                "review_status": "reviewed",
                "reviewed_by": "Phase 3 test",
                "reviewed_at": "2026-07-11",
                "enabled": True,
                "disabled_reason": "",
            }
        )
    node_frame = pl.DataFrame(list(nodes.values()), schema=NODE_SCHEMA)
    prompt_rows = [
        {
            "classification_version": CLASSIFICATION_V3_VERSION,
            "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
            "node_id": row["node_id"],
            "rank": row["rank"],
            "scientific_name": row["scientific_name"],
            "label": _label(str(row["node_id"])),
            "prompt_template": "prompt::{name}",
            "sort_order": 1,
            "enabled": True,
        }
        for row in nodes.values()
    ]
    path_frame = pl.DataFrame(paths, schema=LEAF_PATH_SCHEMA)
    return PathTaxonomyStore(
        sources=pl.DataFrame(schema=SOURCE_SCHEMA),
        nodes=node_frame,
        edges=pl.DataFrame(schema=EDGE_SCHEMA),
        gbif_mappings=pl.DataFrame(mappings, schema=GBIF_MAPPING_SCHEMA),
        leaf_paths=path_frame,
        prompt_labels=pl.DataFrame(prompt_rows, schema=PROMPT_LABEL_SCHEMA),
        qa_findings=pl.DataFrame(schema=QA_FINDING_SCHEMA),
        manifest={
            "classification_version": CLASSIFICATION_V3_VERSION,
            "prompt_version": CLASSIFICATION_V3_PROMPT_VERSION,
            "hierarchy_fingerprint": "sha256:fixture-hierarchy",
            "classification_fingerprint": "sha256:fixture-classification",
        },
        _enabled_hierarchy_hashes=frozenset(path_frame["hierarchy_hash"].to_list()),
    )


def _path_specs() -> list[_PathSpec]:
    family_a = ("f:a", "Alphaidae")
    family_b = ("f:b", "Betaidae")
    family_c = ("f:c", "Gammaidae")
    family_d = ("f:d", "Deltaidae")
    specs = [
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1", "Alphaona", "g:a1b", "Duplicata", "s:a1b", "Trap alpha", "2001"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-2", "Alphatwoa", "g:a2u", "Duplicata", "s:a2u", "Alpha two", "2002"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-3", "Alphathreea", "g:a3u", "AlphaThree", "s:a3u", "Alpha three", "2003"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", "u:a1-4", "Alphafoura", "g:a4u", "AlphaFour", "s:a4u", "Alpha four", "2004"),
        _spec(family_a, "a1", "AlphaOneinae", "Alphaonini", None, "", "g:askip", "AlphaSkip", "s:askip", "Alpha skip", "2005"),
        _spec(family_a, "a2", "AlphaTwoinae", "Alphatwoini", "u:a2", "Alphatwoa", "g:a2", "AlphaTwo", "s:a2", "Alpha branch two", "2006"),
        _spec(family_a, "a3", "AlphaThreeinae", "Alphathreeini", "u:a3", "Alphathreea", "g:a3", "AlphaThree", "s:a3", "Alpha branch three", "2007"),
        _spec(family_a, "a4", "AlphaFourinae", "Alphafourini", "u:a4", "Alphafoura", "g:a4", "AlphaFour", "s:a4", "Alpha branch four", "2008"),
        _spec(family_b, "b1", "BetaOneinae", "Betaonini", None, "", "g:b1", "BetaOne", "s:b1", "Beta one", "2101"),
        _spec(family_b, "b2", "BetaTwoinae", "Betatwoini", "u:b2", "Betatwoa", "g:b2", "BetaTwo", "s:b2", "Beta two", "2102"),
        _spec(family_c, "c1", "GammaOneinae", "Gammaonini", None, "", "g:c1", "GammaOne", "s:c1", "Gamma one", "2201"),
        _spec(family_d, "d1", "DeltaOneinae", "Deltaonini", None, "", "g:d1", "DeltaOne", "s:d1", "Delta one", "2301"),
    ]
    for index in range(25):
        specs.append(
            _spec(
                family_a,
                "a1",
                "AlphaOneinae",
                "Alphaonini",
                "u:a1",
                "Alphaona",
                "g:a1",
                "AlphaMain",
                f"s:a1:{index:02d}",
                f"Alpha species {index:02d}",
                str(3000 + index),
            )
        )
    return specs


def _spec(
    family: tuple[str, str],
    branch: str,
    subfamily_name: str,
    tribe_name: str,
    subtribe_id: str | None,
    subtribe_name: str,
    genus_id: str,
    genus_name: str,
    species_id: str,
    species_name: str,
    gbif_key: str,
) -> _PathSpec:
    return _PathSpec(
        family=family,
        subfamily=(f"sf:{branch}", subfamily_name),
        tribe=(f"t:{branch}", tribe_name),
        subtribe=(subtribe_id, subtribe_name) if subtribe_id else None,
        genus=(genus_id, genus_name),
        species=(species_id, species_name),
        gbif_key=gbif_key,
    )
