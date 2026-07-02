from __future__ import annotations

import polars as pl

from biominer.bioclip.resume import bioclip_resume_key, build_bioclip_work_items
from biominer.storage.local import LocalStorageBackend
from biominer.workstore.resume import prepare_resume_plan
from biominer.workstore.sqlite import SQLiteWorkStore


def test_bioclip_resume_key_uses_only_six_identity_fields() -> None:
    base = {
        "source": "flickr",
        "flickr_photo_id": "1",
        "image_url": "https://example.test/1.jpg",
        "model_id": "bioclip2_5",
        "model_version": "bioclip2_5_huge",
        "model_checkpoint": "checkpoint-a",
    }

    assert bioclip_resume_key(**base) == bioclip_resume_key(**{**base, "score": "ignored", "bucket": "ignored"})
    assert bioclip_resume_key(**base) != bioclip_resume_key(**{**base, "image_url": "https://example.test/2.jpg"})
    assert bioclip_resume_key(**base) != bioclip_resume_key(**{**base, "model_checkpoint": "checkpoint-b"})


def test_build_bioclip_work_items_deduplicates_identity_rows() -> None:
    frame = pl.DataFrame(
        {
            "source": ["flickr", "flickr"],
            "flickr_photo_id": ["1", "1"],
            "image_url": ["https://example.test/1.jpg", "https://example.test/1.jpg"],
            "bioclip_top1_score": [0.1, 0.9],
        }
    )

    items = build_bioclip_work_items(
        input_frame=frame,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )

    assert len(items) == 1
    assert items[0]["source"] == "flickr"
    assert items[0]["model_checkpoint"] == "checkpoint-a"


def test_bioclip_completed_filtering_with_resume_plan(tmp_path) -> None:
    store = SQLiteWorkStore(tmp_path / "work.sqlite")
    frame = pl.DataFrame(
        {
            "source": ["flickr", "flickr"],
            "flickr_photo_id": ["1", "2"],
            "image_url": ["https://example.test/1.jpg", "https://example.test/2.jpg"],
        }
    )
    items = build_bioclip_work_items(
        input_frame=frame,
        model_id="bioclip2_5",
        model_version="bioclip2_5_huge",
        model_checkpoint="checkpoint-a",
    )
    done_key = items[0]["work_key"]
    store.enqueue_work(job_name="bioclip_screen", stage="classified", registry_version="registry-v1", items=[items[0]])
    store.claim_next_batch(worker_id="old-worker", job_name="bioclip_screen", stage="classified", registry_version="registry-v1", limit=1)
    store.mark_completed(work_key=done_key, output_uri="classified.parquet", checksum=None, row_count=1)

    plan = prepare_resume_plan(
        workstore=store,
        storage=LocalStorageBackend(),
        job_name="bioclip_screen",
        stage="classified",
        run_id="run-1",
        registry_version="registry-v1",
        planned_items=items,
        worker_id="worker-1",
        stale_after_seconds=3600,
    )
    claimed = store.list_work_items(job_name="bioclip_screen", stage="classified", registry_version="registry-v1", statuses=["claimed"])

    assert plan.skipped_completed_count == 1
    assert plan.claimed_count == 1
    assert claimed[0]["work_key"] != done_key
