from __future__ import annotations

from typing import Any

import polars as pl

from biominer.workstore.keys import stable_work_key


BIOCLIP_IDENTITY_FIELDS = (
    "source",
    "flickr_photo_id",
    "image_url",
    "model_id",
    "model_version",
    "model_checkpoint",
)


def bioclip_resume_key(
    *,
    source: str,
    flickr_photo_id: str,
    image_url: str,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
    **_: Any,
) -> str:
    payload = {
        "source": str(source),
        "flickr_photo_id": str(flickr_photo_id),
        "image_url": str(image_url),
        "model_id": str(model_id),
        "model_version": str(model_version),
        "model_checkpoint": str(model_checkpoint),
    }
    return stable_work_key(payload, prefix="bioclip_screen")


def build_bioclip_work_items(
    *,
    input_frame: pl.DataFrame,
    model_id: str,
    model_version: str,
    model_checkpoint: str,
) -> list[dict[str, Any]]:
    required = {"source", "flickr_photo_id", "image_url"}
    missing = required.difference(input_frame.columns)
    if missing:
        raise ValueError(f"input_frame missing BioCLIP resume columns: {sorted(missing)}")

    items_by_key: dict[str, dict[str, Any]] = {}
    for row in input_frame.select(["source", "flickr_photo_id", "image_url"]).iter_rows(named=True):
        item = {
            "source": str(row["source"]),
            "flickr_photo_id": str(row["flickr_photo_id"]),
            "image_url": str(row["image_url"]),
            "model_id": str(model_id),
            "model_version": str(model_version),
            "model_checkpoint": str(model_checkpoint),
        }
        work_key = bioclip_resume_key(**item)
        item["work_key"] = work_key
        items_by_key.setdefault(work_key, item)
    return [items_by_key[key] for key in sorted(items_by_key)]
