from __future__ import annotations

from types import SimpleNamespace

import pytest

import biominer.bioclip.cloud_work as cloud_work
import biominer.bioclip.object_runner as object_runner
from biominer.bioclip.cascade_contract import GLOBAL_RANK_TOP_K_BEAM_STRATEGY
from biominer.bioclip.path_cascade_classifier import PathCascadeResult, RankStepResult
from biominer.bioclip.path_taxonomy_store import (
    RANK_SCREEN_PROMPT_STAGE,
    SPECIES_FIRST_PASS_PROMPT_STAGE,
    SPECIES_RERANK_PROMPT_STAGE,
)
from biominer.registry.classification_v3 import (
    CLASSIFICATION_RANKS,
    CLASSIFICATION_V3_PROMPT_VERSION,
    CLASSIFICATION_V3_VERSION,
)


def test_local_and_cloud_v3_dispatch_serialize_identical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _empty_cascade_result()
    item = {
        "source": "flickr",
        "flickr_photo_id": "photo-1",
        "detection_id": "detection-1",
        "crop_hash": "sha256:crop-1",
        "visual_input_id": "crop-1",
        "visual_input_kind": "detector_crop",
        "ablation_mode": "detector_crop",
        "detector_score": 0.91,
    }
    scorer = SimpleNamespace(
        model_id="fake-bioclip",
        model_version="test",
        model_checkpoint="fake-checkpoint",
        embed_image_items=lambda items: [[1.0, 0.0] for _item in items],
    )
    taxonomy_store = SimpleNamespace(hierarchy_fingerprint="sha256:hierarchy")
    embedding_index = SimpleNamespace(cache_fingerprint="sha256:embedding-cache")
    calls: list[tuple[object, object, object, tuple[dict[str, object], ...]]] = []

    def fake_classify_path_cascade_batch(
        *,
        items,
        embedding_scorer,
        taxonomy_store,
        taxonomy_text_embedding_index,
    ):  # noqa: ANN001, ANN201 - production dispatch boundary fake.
        calls.append(
            (
                embedding_scorer,
                taxonomy_store,
                taxonomy_text_embedding_index,
                tuple(items),
            )
        )
        return tuple(result for _item in items)

    monkeypatch.setattr(
        object_runner,
        "classify_path_cascade_batch",
        fake_classify_path_cascade_batch,
    )
    monkeypatch.setattr(
        cloud_work,
        "classify_path_cascade_batch",
        fake_classify_path_cascade_batch,
    )

    local_rows = object_runner._score_hierarchical_detection_batch(
        items=[item],
        scorer=scorer,  # type: ignore[arg-type]
        path_taxonomy_store=taxonomy_store,  # type: ignore[arg-type]
        taxonomy_text_embedding_index=embedding_index,  # type: ignore[arg-type]
    )
    cloud_rows = cloud_work._score_path_cascade_cloud_batch(
        items=[item],
        scorer=scorer,  # type: ignore[arg-type]
        taxonomy_store=taxonomy_store,  # type: ignore[arg-type]
        taxonomy_text_embedding_index=embedding_index,  # type: ignore[arg-type]
    )

    assert len(calls) == 2
    assert all(call[:3] == (scorer, taxonomy_store, embedding_index) for call in calls)
    assert all(call[3] == (item,) for call in calls)
    for row in (*local_rows, *cloud_rows):
        row.pop("classified_at")
    assert local_rows == cloud_rows
    assert local_rows[0]["classification_mode"] == "hierarchical_butterfly_classification"
    assert local_rows[0]["candidate_source"] == "reviewed_classification_v3"


def _empty_cascade_result() -> PathCascadeResult:
    rank_steps = tuple(
        _skipped_step(
            rank,
            prompt_stage=(
                SPECIES_FIRST_PASS_PROMPT_STAGE
                if rank == "SPECIES"
                else RANK_SCREEN_PROMPT_STAGE
            ),
        )
        for rank in CLASSIFICATION_RANKS
    )
    return PathCascadeResult(
        classification_version=CLASSIFICATION_V3_VERSION,
        prompt_version=CLASSIFICATION_V3_PROMPT_VERSION,
        taxonomy_fingerprint="sha256:hierarchy",
        classification_fingerprint="sha256:classification",
        embedding_cache_fingerprint="sha256:embedding-cache",
        beam_strategy=GLOBAL_RANK_TOP_K_BEAM_STRATEGY,
        rank_beam_width=3,
        rank_steps=rank_steps,
        species_rerank_step=_skipped_step(
            "SPECIES",
            prompt_stage=SPECIES_RERANK_PROMPT_STAGE,
        ),
        species_top20=(),
        species_reranked_top20=(),
        species_top5=(),
        species_top3=(),
        species_top1=None,
        final_winning_path=(),
        skipped_ranks=CLASSIFICATION_RANKS,
    )


def _skipped_step(rank: str, *, prompt_stage: str) -> RankStepResult:
    return RankStepResult(
        rank=rank,
        prompt_stage=prompt_stage,
        candidate_count=0,
        retained_count=0,
        active_path_count_before=0,
        active_path_count_after=0,
        top_candidates=(),
        top1_margin=None,
        parent_node_ids=(),
        candidate_node_ids=(),
        candidate_raw_similarities=(),
        retained_node_ids=(),
        pruned_node_ids=(),
        reviewed_skip_path_count=0,
        skipped=True,
        skip_reason="no_enabled_reviewed_candidates",
    )
