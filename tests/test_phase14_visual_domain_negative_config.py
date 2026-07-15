from __future__ import annotations

import json
from pathlib import Path
import re


CONFIG_PATH = (
    Path(__file__).parents[1]
    / "config"
    / "pilot"
    / "papilio_demoleus_visual_domain_negative_candidates.json"
)
EXPECTED_CATEGORIES = {
    "artwork",
    "logos",
    "tattoos",
    "pinned_specimens",
    "partial_wings",
    "dead_specimens",
    "flowers",
    "fruit_closeups",
    "garden_scenes",
    "clutter",
    "printed_butterfly_images",
}
CONTEXTUAL_NEGATIVES = {
    "flowers",
    "fruit_closeups",
    "garden_scenes",
}
BIOLOGICAL_HARD_NEGATIVES = {
    "pinned_specimens",
    "partial_wings",
    "dead_specimens",
}
ALLOWED_LICENCES = {"CC0", "CC BY-SA 4.0", "Public domain"}
SHA1_PATTERN = re.compile(r"sha1:[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _candidates() -> list[dict[str, object]]:
    return _config()["candidates"]


def test_visual_negative_candidate_config_has_exact_prototype_contract() -> None:
    config = _config()

    assert config["schema_version"] == (
        "prototype-visual-domain-negative-candidates-v1.0.0"
    )
    assert config["target_accepted_taxon_key"] == "gbif:1938069"
    assert config["deployment_scope"] == "taxalensdemo_production"
    assert config["media_storage_policy"] == ("persistent_local_demo_cache_untracked")
    assert config["prototype_policy"] == {
        "provider_supported_metadata_is_sufficient": True,
        "agent_screening_is_human_verification": False,
        "human_review_required_for_production": True,
        "support_bank_enablement_deferred_to_phase": "14.3",
    }


def test_all_required_visual_domains_have_one_distinct_candidate() -> None:
    candidates = _candidates()

    assert len(candidates) == len(EXPECTED_CATEGORIES)
    assert {row["visual_domain_category"] for row in candidates} == (
        EXPECTED_CATEGORIES
    )
    assert len({row["candidate_id"] for row in candidates}) == len(candidates)
    assert len({row["source_record_id"] for row in candidates}) == len(candidates)
    assert len({row["media_uri"] for row in candidates}) == len(candidates)
    assert len({row["provider_source_sha1"] for row in candidates}) == len(candidates)


def test_candidates_preserve_remote_and_local_raster_media_provenance() -> None:
    candidates = _candidates()

    assert len({row["local_media_path"] for row in candidates}) == len(candidates)
    assert len({row["local_media_sha256"] for row in candidates}) == len(candidates)
    for row in candidates:
        assert row["source"] == "Wikimedia Commons"
        assert row["source_record_uri"].startswith(
            "https://commons.wikimedia.org/wiki/File:"
        )
        assert row["media_uri"].startswith(
            "https://upload.wikimedia.org/wikipedia/commons/"
        )
        assert row["mime_type"] in {"image/jpeg", "image/png"}
        assert row["width"] > 0
        assert row["height"] > 0
        assert SHA1_PATTERN.fullmatch(row["provider_source_sha1"])
        assert row["local_media_path"].startswith(
            "data/cache/taxalensdemo/reference_negatives/commons_"
        )
        assert SHA256_PATTERN.fullmatch(row["local_media_sha256"])
        assert row["local_media_variant"] == "provider_raster_preview"
        assert row["duplicate_check_status"] == "distinct_source_and_sha1"


def test_rights_licence_and_attribution_are_complete_and_allowed() -> None:
    for row in _candidates():
        assert row["licence"] in ALLOWED_LICENCES
        assert row["licence_check_status"] == "allowed"
        assert row["attribution"].strip()
        assert row["rights_evidence_uri"] == row["source_record_uri"]
        if row["licence"] == "Public domain":
            assert row["licence_uri"] is None
            assert row["licence_uri_status"] == ("provider_not_supplied_public_domain")
            assert row["rights_holder"] is None
        else:
            assert row["licence_uri"].startswith(("http://", "https://"))
            assert row["licence_uri_status"] == "provider_supplied"
            assert row["rights_holder"].strip()


def test_provider_support_never_claims_human_verification() -> None:
    for row in _candidates():
        assert row["verification_status"] == "provider_supported"
        assert row["verification_actor"] == "wikimedia_commons"
        assert row["agent_screening_status"] == "passed"
        assert row["agent_screened_by"] == "codex"
        assert row["human_review_status"] == "pending"
        assert row["human_verified"] is False
        assert row["reviewed_by"] is None
        assert row["reviewed_at"] is None


def test_candidates_are_prototype_only_until_support_bank_qa() -> None:
    for row in _candidates():
        assert row["prototype_eligible"] is True
        assert row["production_eligible"] is False
        assert row["support_bank_enabled"] is False
        assert row["category_evidence"].strip()


def test_contextual_negatives_are_target_absent_full_frames() -> None:
    by_category = {row["visual_domain_category"]: row for row in _candidates()}

    for category in CONTEXTUAL_NEGATIVES:
        row = by_category[category]
        assert row["target_presence"] == "absent"
        assert row["contains_biological_butterfly"] is False
        assert row["contains_butterfly_visual"] is False


def test_biological_hard_negatives_remain_separate_from_context() -> None:
    by_category = {row["visual_domain_category"]: row for row in _candidates()}

    for category in BIOLOGICAL_HARD_NEGATIVES:
        row = by_category[category]
        assert row["contains_biological_butterfly"] is True
        assert row["contains_butterfly_visual"] is True
