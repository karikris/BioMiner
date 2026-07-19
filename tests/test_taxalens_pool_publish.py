"""End-to-end contract tests for the pinned TaxaLens handoff consumer."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import pytest

from biominer.integration.taxalens_pool_handoff import (
    TAXALENS_PINNED_COMMIT,
    TAXALENS_POOL_HANDOFF_FILE,
    TAXALENS_TARGET_CONTRACTS,
)
from biominer.integration.taxalens_pool_publish import (
    publish_taxalens_pool_handoff,
    validate_published_taxalens_pool_handoff,
)
from biominer.storage.handoff import verify_handoff_bundle
from helpers.dynamic_pool_handoff_fixture import (
    build_dynamic_pool_handoff_fixture,
    build_quality_report_fixture,
    build_review_selection_fixture,
)


ROOT = Path(__file__).parents[1]
TAXALENS_REPOSITORY = ROOT.parent / "taxalens"
PRODUCER_COMMIT = "1" * 40


def test_complete_publication_is_deterministic_and_pinned_consumer_verifies_it(
    tmp_path: Path,
) -> None:
    first = _publish(tmp_path / "first", quality_report=build_quality_report_fixture())
    second = _publish(
        tmp_path / "second",
        quality_report=build_quality_report_fixture(),
    )

    assert first.manifest == second.manifest
    assert first.bundle.sha256 == second.bundle.sha256
    assert (
        first.bundle.archive_path.read_bytes()
        == second.bundle.archive_path.read_bytes()
    )
    assert first.bundle.file_count == 9
    assert first.manifest["consumer"] == {
        "repository": "karikris/taxalens",
        "commit": TAXALENS_PINNED_COMMIT,
    }
    assert first.manifest["target_contracts"] == TAXALENS_TARGET_CONTRACTS
    assert first.manifest["evidence_maturity"]["quality_estimate"]["status"] == (
        "available"
    )
    assert first.manifest["evidence_maturity"]["release"]["release_ready"] is False
    geographic = next(
        row for row in first.manifest["artifacts"] if row["role"] == "geographic_cells"
    )
    assert geographic["availability"] == "unavailable"
    verification = verify_handoff_bundle(
        first.bundle.archive_path,
        expected_sha256=first.bundle.sha256,
    )
    assert verification.source_git_sha == PRODUCER_COMMIT
    assert {item.relative_path for item in verification.files} == {
        "artifacts/dynamic_pool_candidate_scores.parquet",
        "artifacts/dynamic_pool_photo_summary.parquet",
        "artifacts/dynamic_reference_pool_members.parquet",
        "artifacts/dynamic_reference_pool_plans.parquet",
        "artifacts/dynamic_reference_pool_summary.parquet",
        "artifacts/family_geo_candidate_sets.parquet",
        "artifacts/review/taxalens_quality_sidecar.json",
        "artifacts/review/taxalens_review_sampling_frame.parquet",
        TAXALENS_POOL_HANDOFF_FILE,
    }
    consumer = _load_pinned_taxalens_consumer(tmp_path)
    downstream = consumer.verify_biominer_handoff_archive(
        first.bundle.archive_path,
        expected_sha256=first.bundle.sha256,
    )
    assert (
        consumer.HANDOFF_INVENTORY_SCHEMA_VERSION
        == (TAXALENS_TARGET_CONTRACTS["storage_handoff_inventory"])
    )
    assert (
        consumer.INTERNAL_INVENTORY_PATH
        == (TAXALENS_TARGET_CONTRACTS["storage_handoff_inventory_path"])
    )
    assert downstream.inventory.source_git_sha == PRODUCER_COMMIT
    assert len(downstream.inventory.files) == first.bundle.file_count


def test_publication_without_review_outcomes_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    published = _publish(tmp_path / "no-quality", quality_report=None)

    quality = next(
        row
        for row in published.manifest["artifacts"]
        if row["role"] == "quality_sidecar"
    )
    assert quality["availability"] == "unavailable"
    assert published.bundle.file_count == 8
    assert not (
        published.root / "artifacts/review/taxalens_quality_sidecar.json"
    ).exists()
    assert published.manifest["evidence_maturity"]["human_review"]["status"] == (
        "not_evaluated"
    )
    assert published.manifest["evidence_maturity"]["quality_estimate"] == {
        "status": "unavailable",
        "label": None,
        "zero_review_is_zero_quality": False,
        "release_authorizing": False,
        "unavailable_reason": "no validated reviewed quality report supplied",
    }
    validate_published_taxalens_pool_handoff(
        published.root,
        published.manifest,
    )


def test_publication_is_create_only_and_keeps_archives_outside_root(
    tmp_path: Path,
) -> None:
    base = tmp_path / "create-only"
    published = _publish(base, quality_report=None)

    with pytest.raises(FileExistsError, match="create-only"):
        _publish(base, quality_report=None)
    assert published.manifest_path.is_file()

    selection, policy = build_review_selection_fixture()
    nested_root = tmp_path / "nested"
    with pytest.raises(ValueError, match="outside the publication root"):
        publish_taxalens_pool_handoff(
            **_publication_inputs(
                selection=selection,
                policy=policy,
                quality_report=None,
            ),
            output_root=nested_root,
            archive_output_dir=nested_root / "archives",
        )
    assert not nested_root.exists()


def test_directory_validator_detects_artifact_tampering(tmp_path: Path) -> None:
    published = _publish(tmp_path / "tamper", quality_report=None)
    score_path = published.root / "artifacts/dynamic_pool_candidate_scores.parquet"
    with score_path.open("ab") as output:
        output.write(b"tampered")

    with pytest.raises(ValueError, match="byte count differs"):
        validate_published_taxalens_pool_handoff(
            published.root,
            published.manifest,
        )


def _publish(base: Path, *, quality_report):
    selection, policy = build_review_selection_fixture()
    return publish_taxalens_pool_handoff(
        **_publication_inputs(
            selection=selection,
            policy=policy,
            quality_report=quality_report,
        ),
        output_root=base / "handoff",
        archive_output_dir=base / "archives",
    )


def _publication_inputs(*, selection, policy, quality_report) -> dict[str, object]:
    return {
        "score_pool_frames": build_dynamic_pool_handoff_fixture(),
        "review_selection": selection,
        "review_sampling_policy": policy,
        "quality_report": quality_report,
        "producer_commit": PRODUCER_COMMIT,
        "created_at": "2026-07-18T12:00:00+10:00",
        "run_id": "run-papilio-demoleus-001",
        "registry_version": "registry-2026-07-18",
        "source_snapshot_fingerprints": ["sha256:" + "2" * 64],
        "model_fingerprint": "sha256:" + "3" * 64,
        "preprocessing_fingerprint": "sha256:" + "4" * 64,
    }


def _load_pinned_taxalens_consumer(tmp_path: Path) -> ModuleType:
    if not (TAXALENS_REPOSITORY / ".git").exists():
        pytest.skip("TaxaLens sibling checkout is unavailable")
    committed = subprocess.run(
        [
            "git",
            "show",
            f"{TAXALENS_PINNED_COMMIT}:taxalens/product/biominer_handoff.py",
        ],
        cwd=TAXALENS_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module_path = tmp_path / "pinned_taxalens_biominer_handoff.py"
    module_path.write_text(committed, encoding="utf-8")
    module_name = "pinned_taxalens_biominer_handoff"
    specification = importlib.util.spec_from_file_location(module_name, module_path)
    if specification is None or specification.loader is None:
        raise RuntimeError("could not load pinned TaxaLens handoff consumer")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module
