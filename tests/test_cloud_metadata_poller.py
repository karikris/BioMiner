from __future__ import annotations

from pathlib import Path

import polars as pl

from biominer.flickr_fetch.cloud_poller import CloudMetadataPoller, flickr_query_work_item
from biominer.flickr_fetch.query_planner import FlickrQuery
from biominer.run.stages import RunStage
from biominer.workstore.sqlite import SQLiteWorkStore


def test_cloud_metadata_poller_claims_writes_registers_then_completes(tmp_path: Path) -> None:
    storage = _FakeCloudStorage()
    workstore = SQLiteWorkStore(tmp_path / "workstore.sqlite")
    query = FlickrQuery(
        term="Danaus plexippus",
        language="en",
        search_field="tags",
        lane="normal_page",
        page=1,
        per_page=500,
        has_geo=0,
        registry_version="registry-v1",
        query_definition_id="q-tags",
        accepted_taxon_key="gbif:5130",
        accepted_scientific_name="Danaus plexippus",
        family_key="gbif:7017",
        genus_key="gbif:5131",
        species_key="gbif:5130",
    )
    workstore.enqueue_work(
        "biominer_production_run",
        "registry-v1",
        [flickr_query_work_item(query, run_id="run-1")],
        stage=RunStage.POLL_FLICKR.value,
    )

    result = CloudMetadataPoller(
        storage=storage,
        workstore=workstore,
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="registry-v1",
        run_id="run-1",
        worker_id="worker-1",
        storage_prefix="s3://biominer/runs/run_id=run-1/staging",
        fetch_metadata=lambda _query: _flickr_payload("photo-1"),
        max_api_calls=5,
    ).run_once(claim_limit=1)

    assert result.work_items_claimed == 1
    assert result.raw_responses_written == 1
    assert result.evidence_rows_written == 1
    assert result.workstore_work_items_completed == 1
    assert result.source_record_shard_uris == tuple(storage.parquet_payloads)
    shard = storage.parquet_payloads[result.source_record_shard_uris[0]]
    row = shard.to_dicts()[0]
    assert row["source"] == "flickr"
    assert row["flickr_photo_id"] == "photo-1"
    assert row["tag_search_terms"] == ["Danaus plexippus"]
    assert row["query_definition_ids"] == ["q-tags"]
    assert row["discovery_accepted_taxon_keys"] == ["gbif:5130"]
    assert row["registry_versions"] == ["registry-v1"]

    work_items = workstore.list_work_items(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="registry-v1",
    )
    assert [item["status"] for item in work_items] == ["completed"]
    assert work_items[0]["output_uri"] == result.source_record_shard_uris[0]
    shards = workstore.list_committed_shards(
        job_name="biominer_production_run",
        stage=RunStage.POLL_FLICKR.value,
        registry_version="registry-v1",
        run_id="run-1",
    )
    assert [shard["uri"] for shard in shards] == [result.source_record_shard_uris[0]]


class _FakeCloudStorage:
    def __init__(self) -> None:
        self.parquet_payloads: dict[str, pl.DataFrame] = {}
        self.json_payloads: dict[str, dict[str, object]] = {}

    def write_json(self, uri: str, payload: dict[str, object]) -> str:
        self.json_payloads[uri] = payload
        return uri

    def write_parquet_shard(self, uri: str, frame: pl.DataFrame) -> str:
        self.parquet_payloads[uri] = frame
        return uri


def _flickr_payload(photo_id: str) -> dict[str, object]:
    return {
        "photos": {
            "total": "1",
            "pages": "1",
            "page": "1",
            "perpage": "500",
            "photo": [
                {
                    "id": photo_id,
                    "title": "Danaus plexippus on milkweed",
                    "url_l": f"https://live.staticflickr.com/{photo_id}.jpg",
                    "datetaken": "2025-03-01 10:30:00",
                }
            ],
        }
    }
