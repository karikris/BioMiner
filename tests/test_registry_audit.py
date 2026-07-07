from __future__ import annotations

import json

import polars as pl

from biominer.registry.audit import audit_registry


def test_registry_audit_writes_language_target_and_curated_gap_reports(tmp_path) -> None:
    registry = tmp_path / "registry"
    registry.mkdir()
    _write_audit_fixture(registry)

    result = audit_registry(registry, report_dir=tmp_path / "reports")

    language_report = tmp_path / "reports" / "language_target_coverage_2026-07-papilio-demo.json"
    language_markdown = tmp_path / "reports" / "language_target_coverage_2026-07-papilio-demo.md"
    gap_report = tmp_path / "reports" / "curated_vernacular_gap_report_2026-07-papilio-demo.json"
    gap_markdown = tmp_path / "reports" / "curated_vernacular_gap_report_2026-07-papilio-demo.md"
    assert result["language_target_coverage_report"] == str(language_report)
    assert result["curated_vernacular_gap_report"] == str(gap_report)
    assert language_markdown.exists()
    assert gap_markdown.exists()

    language_payload = json.loads(language_report.read_text(encoding="utf-8"))
    assert language_payload["occurrence_countries_by_range_status"] == {
        "native_or_long_established": 1,
        "taxonomically_cautionary": 1,
    }
    assert {
        (row["source"], row["language"], row["region"], row["count"])
        for row in language_payload["curated_names_by_source_language_region"]
    } == {("Butterflies of India", "eng", "IN", 1)}
    assert {
        (row["language_code"], row["region"])
        for row in language_payload["languages_with_no_curated_source_found"]
    } >= {("hin", "South Asia"), ("eng", "Australia/New Guinea taxonomic caution")}
    assert language_payload["languages_with_only_generated_candidates"] == [
        {"accepted_taxon_key": "gbif:100", "language_code": "hin", "region": "South Asia"}
    ]
    assert language_payload["names_disabled_due_to_ambiguity"] == [
        {"accepted_taxon_key": "gbif:100", "display_name": "Lime", "language": "eng", "region": "IN", "source": "Butterflies of India"}
    ]
    assert language_payload["names_disabled_due_to_taxonomic_caution"] == [
        {
            "accepted_taxon_key": "gbif:100",
            "display_name": "Caution Lime",
            "language": "eng",
            "region": "AU",
            "source": "Cautionary Checklist",
        }
    ]
    assert language_payload["papilio_demoleus_regional_coverage_summary"]["South Asia"]["missing_curated_languages"] == ["hin"]

    gap_payload = json.loads(gap_report.read_text(encoding="utf-8"))
    assert gap_payload["gap_count"] == 2
    assert ("hin", "South Asia") in {
        (row["language_code"], row["region"])
        for row in gap_payload["curated_vernacular_gaps"]
    }
    assert "Papilio demoleus" in language_markdown.read_text(encoding="utf-8")
    assert "Languages With Only Generated Candidates" in gap_markdown.read_text(encoding="utf-8")


def _write_audit_fixture(registry) -> None:
    (registry / "manifest.json").write_text(
        json.dumps({"registry_version": "2026-07-papilio-demo"}),
        encoding="utf-8",
    )
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "rank": "SPECIES",
                "family": "Papilionidae",
            }
        ]
    ).write_parquet(registry / "taxa.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Papilio demoleus",
                "language": "la",
                "name_class": "accepted_scientific",
                "source": "GBIF",
                "enabled": True,
            }
        ]
    ).write_parquet(registry / "names.parquet")
    pl.DataFrame([{"search_field": "tags", "enabled": True}]).write_parquet(registry / "flickr_query_definitions.parquet")
    pl.DataFrame([{"severity": "warning"}]).write_parquet(registry / "qa_findings.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "country_code": "IN",
                "country_name": "India",
                "admin1_code": "",
                "admin1_name": "",
                "range_status": "native_or_long_established",
                "taxonomic_caution": False,
                "region": "South Asia",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "country_code": "AU",
                "country_name": "Australia",
                "admin1_code": "",
                "admin1_name": "",
                "range_status": "taxonomically_cautionary",
                "taxonomic_caution": True,
                "region": "Australia/New Guinea taxonomic caution",
            },
        ]
    ).write_parquet(registry / "range_countries.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "country_code": "IN",
                "country_name": "India",
                "admin1_code": "",
                "admin1_name": "",
                "language_code": "eng",
                "language_name": "English",
                "script": "Latn",
                "region": "South Asia",
                "priority": 10,
                "reason": "fixture",
                "source": "fixture",
                "source_version": "fixture",
                "enabled": True,
                "disabled_reason": "",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "country_code": "IN",
                "country_name": "India",
                "admin1_code": "",
                "admin1_name": "",
                "language_code": "hin",
                "language_name": "Hindi",
                "script": "Deva",
                "region": "South Asia",
                "priority": 20,
                "reason": "fixture",
                "source": "fixture",
                "source_version": "fixture",
                "enabled": True,
                "disabled_reason": "",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "scientific_name": "Papilio demoleus",
                "country_code": "AU",
                "country_name": "Australia",
                "admin1_code": "",
                "admin1_name": "",
                "language_code": "eng",
                "language_name": "English",
                "script": "Latn",
                "region": "Australia/New Guinea taxonomic caution",
                "priority": 90,
                "reason": "fixture",
                "source": "fixture",
                "source_version": "fixture",
                "enabled": False,
                "disabled_reason": "taxonomic_caution_range",
            },
        ]
    ).write_parquet(registry / "country_language_targets.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime Swallowtail",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "source": "Butterflies of India",
                "name_class": "vernacular",
                "trust_tier": "T2",
                "enabled": True,
                "disabled_reason": "",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Caution Lime",
                "language": "eng",
                "script": "Latn",
                "region": "AU",
                "source": "Cautionary Checklist",
                "name_class": "vernacular",
                "trust_tier": "T2",
                "enabled": False,
                "disabled_reason": "taxonomic_caution:demoleus_sthenelus_unresolved",
            },
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime",
                "language": "eng",
                "script": "Latn",
                "region": "IN",
                "source": "Butterflies of India",
                "name_class": "vernacular",
                "trust_tier": "T2",
                "enabled": False,
                "disabled_reason": "ambiguous_common_name",
            },
        ]
    ).write_parquet(registry / "source_name_assertions.parquet")
    pl.DataFrame(
        [
            {
                "candidate_id": "translation:hin",
                "source": "MyMemory",
                "source_record_id": "mymemory:hin",
                "source_language": "eng",
                "target_language": "hin",
                "source_name": "Lime Swallowtail",
                "translated_name": "Generated Hindi Candidate",
                "accepted_taxon_key": "gbif:100",
                "trust_tier": "T5",
                "enabled": False,
                "disabled_reason": "translation_name_requires_review",
                "review_state": "candidate",
                "confidence": "low",
                "precision_tier": "low",
                "corroborated": False,
            }
        ]
    ).write_parquet(registry / "translation_candidates.parquet")
    pl.DataFrame(
        [
            {
                "accepted_taxon_key": "gbif:100",
                "display_name": "Lime",
                "language": "eng",
                "region": "IN",
                "source": "Butterflies of India",
                "disabled_reason": "ambiguous_common_name",
            }
        ]
    ).write_parquet(registry / "name_candidates.parquet")
