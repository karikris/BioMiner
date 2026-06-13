from __future__ import annotations

import json

import polars as pl

from biominer.cli import build_parser, run
from biominer.reports.name_evidence import accepted_name_terms_from_keyword_json, build_name_evidence_report


def test_accepted_name_terms_from_keyword_json_excludes_broad_discovery_terms(tmp_path) -> None:
    keywords = tmp_path / "keywords.json"
    keywords.write_text(
        json.dumps(
            {
                "dictionary_groups": {
                    "scientific_taxonomic": [
                        {"term": "Papilio demoleus", "term_type": "scientific_name"},
                        {"term": "Papilio", "term_type": "genus"},
                    ],
                    "english_common_names": [
                        {"term": "lime butterfly", "term_type": "common_name"},
                    ],
                    "broad_terms": [
                        {"term": "butterfly", "term_type": "broad_butterfly"},
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    assert accepted_name_terms_from_keyword_json(keywords) == ("Papilio demoleus", "lime butterfly")


def test_name_evidence_report_counts_dwc_candidates_and_broad_query_gaps(tmp_path) -> None:
    metadata = tmp_path / "metadata.parquet"
    bioclip = tmp_path / "bioclip.parquet"
    keywords = tmp_path / "keywords.json"
    keywords.write_text(
        json.dumps(
            {
                "dictionary_groups": {
                    "scientific_taxonomic": [{"term": "Papilio demoleus", "term_type": "scientific_name"}],
                    "english_common_names": [{"term": "lime butterfly", "term_type": "common_name"}],
                    "broad": [{"term": "Papilio", "term_type": "genus"}],
                }
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        [
            {
                "flickr_photo_id": "1",
                "raw_title": "Papilio demoleus in garden",
                "raw_description": "",
                "raw_tags": "",
                "machine_tags": "",
                "first_query_label": "text:Papilio demoleus",
                "all_query_labels": ["text:Papilio demoleus"],
                "all_query_terms": ["Papilio demoleus"],
            },
            {
                "flickr_photo_id": "2",
                "raw_title": "lime butterfly",
                "raw_description": "",
                "raw_tags": "",
                "machine_tags": "",
                "first_query_label": "text:lime butterfly",
                "all_query_labels": ["text:lime butterfly", "tags:Papilio"],
                "all_query_terms": ["lime butterfly", "Papilio"],
            },
            {
                "flickr_photo_id": "3",
                "raw_title": "swallowtail",
                "raw_description": "",
                "raw_tags": "Papilio",
                "machine_tags": "",
                "first_query_label": "text:Papilio",
                "all_query_labels": ["text:Papilio", "tags:Papilio"],
                "all_query_terms": ["Papilio"],
            },
            {
                "flickr_photo_id": "4",
                "raw_title": "Papilio demoleus weak score",
                "raw_description": "",
                "raw_tags": "",
                "machine_tags": "",
                "first_query_label": "text:Papilio demoleus",
                "all_query_labels": ["text:Papilio demoleus"],
                "all_query_terms": ["Papilio demoleus"],
            },
        ]
    ).write_parquet(metadata)
    pl.DataFrame(
        [
            {"flickr_photo_id": "1", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.95},
            {"flickr_photo_id": "2", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.96},
            {"flickr_photo_id": "3", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.97},
            {"flickr_photo_id": "4", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.70},
            {"flickr_photo_id": "5", "occurrence_bin": "bronze", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.99},
        ]
    ).write_parquet(bioclip)

    report = build_name_evidence_report(
        metadata_path=metadata,
        bioclip_output_path=bioclip,
        keywords_json=keywords,
        target_species="Papilio demoleus",
        score_threshold=0.9,
    )

    assert report["gold_records"] == 4
    assert report["gold_target_species_score_gt_threshold"] == 3
    assert report["accepted_name_in_any_metadata_text"] == 2
    assert report["accepted_name_in_any_query"] == 2
    assert report["candidate_dwc_tier_count"] == 2
    assert report["records_without_accepted_name_count"] == 1
    assert report["top_query_labels_for_records_without_accepted_name"] == [
        {"query_label": "tags:Papilio", "records": 1},
        {"query_label": "text:Papilio", "records": 1},
    ]


def test_name_evidence_report_falls_back_to_single_query_term_columns(tmp_path) -> None:
    metadata = tmp_path / "metadata.parquet"
    bioclip = tmp_path / "bioclip.parquet"
    keywords = tmp_path / "keywords.json"
    keywords.write_text(
        json.dumps({"dictionary_groups": {"scientific_taxonomic": [{"term": "Papilio demoleus", "term_type": "scientific_name"}]}}),
        encoding="utf-8",
    )
    pl.DataFrame(
        [
            {
                "flickr_photo_id": "1",
                "raw_title": "Papilio demoleus",
                "query_field": "text",
                "query_term": "Papilio demoleus",
            }
        ]
    ).write_parquet(metadata)
    pl.DataFrame(
        [{"flickr_photo_id": "1", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.95}]
    ).write_parquet(bioclip)

    report = build_name_evidence_report(
        metadata_path=metadata,
        bioclip_output_path=bioclip,
        keywords_json=keywords,
        target_species="Papilio demoleus",
        score_threshold=0.9,
    )

    assert report["accepted_name_in_text_search_row"] == 1
    assert report["accepted_name_in_any_query"] == 1


def test_report_name_evidence_cli_writes_json(tmp_path, capsys) -> None:
    metadata = tmp_path / "metadata.parquet"
    bioclip = tmp_path / "bioclip.parquet"
    keywords = tmp_path / "keywords.json"
    output = tmp_path / "name_evidence.json"
    keywords.write_text(
        json.dumps({"dictionary_groups": {"scientific_taxonomic": [{"term": "Papilio demoleus", "term_type": "scientific_name"}]}}),
        encoding="utf-8",
    )
    pl.DataFrame(
        [{"flickr_photo_id": "1", "raw_title": "Papilio demoleus", "all_query_labels": ["text:Papilio demoleus"], "all_query_terms": ["Papilio demoleus"]}]
    ).write_parquet(metadata)
    pl.DataFrame(
        [{"flickr_photo_id": "1", "occurrence_bin": "gold", "species_top1_scientific_name": "Papilio demoleus", "species_top1_score": 0.95}]
    ).write_parquet(bioclip)

    args = build_parser().parse_args(
        [
            "report-name-evidence",
            "--metadata-output",
            str(metadata),
            "--bioclip-output",
            str(bioclip),
            "--keywords-json",
            str(keywords),
            "--target-species",
            "Papilio demoleus",
            "--score-threshold",
            "0.9",
            "--output",
            str(output),
        ]
    )
    assert run(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output)
    assert json.loads(output.read_text(encoding="utf-8"))["candidate_dwc_tier_count"] == 1
