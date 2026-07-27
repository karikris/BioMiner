from pathlib import Path


ADR = Path("docs/adr/gbif_media_quality_v4.md")
PLAN = Path("docs/gbif_media_quality_v4_plan.md")


def test_v4_adr_preserves_scope_and_source_assertions() -> None:
    text = " ".join(ADR.read_text(encoding="utf-8").split())

    for required in (
        "all 16,612,063 v3 media rows",
        "all 18,680,565 raw multimedia assertions",
        "75,352,491 occurrence rows",
        "never rewrites them",
        "occurrence quality table",
        "media assertion quality table",
        "media resource table",
        "URL equality is not content identity",
    ):
        assert required in text


def test_v4_adr_freezes_semantic_null_and_result_contracts() -> None:
    text = " ".join(ADR.read_text(encoding="utf-8").split())

    for status in (
        "`PASS`",
        "`FAIL`",
        "`UNKNOWN`",
        "`NOT_APPLICABLE`",
        "`WITHHELD`",
        "`GENERALIZED`",
        "`CONFLICT`",
        "`NOT_TESTED`",
    ):
        assert status in text
    assert "No aggregate quality score" in text
    assert "are not completeness failures" in text


def test_v4_adr_requires_full_evidence_and_network_gates() -> None:
    text = " ".join(ADR.read_text(encoding="utf-8").split())

    for required in (
        "`source_row_id`",
        "target field",
        "original and derived values",
        "rule version",
        "categorical confidence",
        "reviewer status",
        "823-row pilot",
        "before a 126,634-row network run can start",
        "4,055-row difference is an explicit rights block",
        "manifest is written last",
    ):
        assert required in text


def test_v4_plan_maps_runtime_and_source_control_ownership() -> None:
    text = " ".join(PLAN.read_text(encoding="utf-8").split())

    for required in (
        "data/derived/gbif_media_database/v4/",
        "reports/gbif_media_database/v4/",
        "src/biominer/gbif_quality/",
        "src/biominer/gbif_media_resolution/",
        "src/biominer/workstore/",
        "42 global acceptance",
        "20 required reports",
    ):
        assert required in text
