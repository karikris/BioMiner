from __future__ import annotations

import polars as pl
import importlib.util
from pathlib import Path
import sys

from flickr_bio_occurrence.flickr.butterfly_terms import (
    ButterflySearchTerm,
    estimate_minimum_fetch_hours,
    load_butterfly_dashboard_terms,
    safe_query_variant,
)


def _load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_butterfly_flickr_fetch.py"
    spec = importlib.util.spec_from_file_location("run_butterfly_flickr_fetch", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_butterfly_dashboard_terms_includes_species_common_names_and_common_words(tmp_path) -> None:
    data_dir = tmp_path / "dashboard" / "data"
    reference_dir = data_dir / "reference"
    reference_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "scientificName": ["Papilio demoleus", "Papilio demoleus sthenelus"],
            "species": ["Papilio demoleus", "Papilio demoleus"],
            "genus": ["Papilio", "Papilio"],
            "family": ["Papilionidae", "Papilionidae"],
        }
    ).write_parquet(data_dir / "butterfly_sa2_bins.parquet")
    (reference_dir / "butterfly_conservation_status.csv").write_text(
        "accepted_taxon,match_names,rank,common_name\n"
        "Papilio demoleus,Papilio demoleus,species,Lime Butterfly / Citrus Swallowtail\n",
        encoding="utf-8",
    )

    terms = load_butterfly_dashboard_terms(data_dir, common_terms=("butterfly",))
    values = {term.term: term.source for term in terms}

    assert values["Papilio demoleus"] == "scientificName"
    assert values["Papilio demoleus sthenelus"] == "scientificName"
    assert values["Papilio"] == "genus"
    assert values["Papilionidae"] == "family"
    assert values["Lime Butterfly"] == "common_name"
    assert values["Citrus Swallowtail"] == "common_name"
    assert values["butterfly"] == "common_word"


def test_safe_query_variant_is_stable_slug() -> None:
    assert safe_query_variant("Papilio (Eleppone) anactus") == "papilio_eleppone_anactus"


def test_estimate_minimum_fetch_hours_uses_hourly_api_cap() -> None:
    assert estimate_minimum_fetch_hours(planned_api_calls=7200, api_calls_per_hour=3600) == 2.0


def test_fetch_state_knows_when_term_reached_reported_final_page(tmp_path) -> None:
    module = _load_fetch_script()
    state = module.FetchState(tmp_path / "state.sqlite")

    state.mark_done(
        work_item_id="butterfly:1",
        term="butterfly",
        term_source="common_word",
        page=1,
        raw_path=tmp_path / "page1.json",
        returned_records=250,
        new_records=250,
        flickr_pages=2,
    )
    assert state.term_is_exhausted("butterfly") is False

    state.mark_done(
        work_item_id="butterfly:2",
        term="butterfly",
        term_source="common_word",
        page=2,
        raw_path=tmp_path / "page2.json",
        returned_records=10,
        new_records=10,
        flickr_pages=2,
    )
    assert state.term_is_exhausted("butterfly") is True


def test_fetch_state_knows_empty_page_exhausts_term(tmp_path) -> None:
    module = _load_fetch_script()
    state = module.FetchState(tmp_path / "state.sqlite")

    state.mark_done(
        work_item_id="rare:1",
        term="rare",
        term_source="scientificName",
        page=1,
        raw_path=tmp_path / "page1.json",
        returned_records=0,
        new_records=0,
        flickr_pages=10,
    )

    assert state.term_is_exhausted("rare") is True


def test_fetch_state_uses_latest_flickr_page_count_for_exhaustion(tmp_path) -> None:
    module = _load_fetch_script()
    state = module.FetchState(tmp_path / "state.sqlite")

    state.mark_done(
        work_item_id="butterfly:1",
        term="butterfly",
        term_source="common_word",
        page=1,
        raw_path=tmp_path / "page1.json",
        returned_records=250,
        new_records=250,
        flickr_pages=40,
    )
    state.mark_done(
        work_item_id="butterfly:37",
        term="butterfly",
        term_source="common_word",
        page=37,
        raw_path=tmp_path / "page37.json",
        returned_records=250,
        new_records=0,
        flickr_pages=37,
    )

    assert state.term_is_exhausted("butterfly") is True


def test_fetch_script_writes_page_evidence_shard_and_enqueues_job(tmp_path) -> None:
    module = _load_fetch_script()
    term = ButterflySearchTerm(term="Papilio demoleus", source="scientificName")
    config = module.FetchConfig(
        output_root=tmp_path / "fetch",
        dashboard_data_dir=tmp_path / "dashboard",
        api_key_env="FLICKR_API_KEY",
        evidence_root=tmp_path / "evidence",
        classification_queue_path=tmp_path / "queue.sqlite",
        bioclip_model_version="bioclip2_5_huge",
        max_pages_per_term=1,
        per_page=250,
        soft_api_calls_per_hour=3200,
        hard_api_calls_per_hour=3600,
        start_taken_date="1950-01-01",
        end_taken_date="2026-06-07",
    )
    queue = module.ClassificationJobQueue(config.classification_queue_path)
    payload = {
        "photos": {
            "photo": [
                {
                    "id": "1",
                    "title": "Papilio demoleus",
                    "url_l": "https://live.staticflickr.com/large.jpg",
                }
            ]
        }
    }

    path = module.write_page_evidence_shard_and_enqueue(
        config=config,
        queue=queue,
        term=term,
        page=1,
        payload=payload,
    )

    assert path == tmp_path / "evidence" / "flickr" / "photos_search" / "scientificName" / "papilio_demoleus" / "page=00001.parquet"
    assert pl.read_parquet(path)["flickr_photo_id"][0] == "1"
    jobs = queue.list_jobs()
    assert len(jobs) == 1
    assert jobs[0].evidence_parquet_path == path
    assert jobs[0].model_version == "bioclip2_5_huge"
