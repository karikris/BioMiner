from __future__ import annotations

import pytest

from biominer.cli import _parse_run_stages, build_parser
from biominer.run import ADAPTIVE_REFERENCE_PRODUCTION_STAGES, ProductionRunRequest
from biominer.run.adaptive_config import REFERENCE_ADMISSION_MODES


def _required_run_args() -> list[str]:
    return [
        "run",
        "--taxon",
        "Papilio demoleus",
        "--registry-dir",
        "registry",
        "--output-prefix",
        "runs",
    ]


def test_production_cli_defaults_to_adaptive_gbif_references() -> None:
    args = build_parser().parse_args(_required_run_args())

    assert args.reference_admission_mode == "adaptive_gbif_fast_start"
    assert args.reference_source == "gbif"
    assert args.initial_scoring_mode == "provisional_reference_ranking"
    assert args.flickr_release_requires_human_review is True
    assert args.statistical_reference_audit is True
    assert _parse_run_stages(None) == ADAPTIVE_REFERENCE_PRODUCTION_STAGES


@pytest.mark.parametrize(
    "removed_args",
    (
        ("--workflow", "legacy"),
        ("--workflow", "reference-first"),
        ("--classification-mode", "build-week-prototype"),
        ("--classification-config", "prototype.json"),
    ),
)
def test_production_cli_rejects_removed_alternate_workflows(
    removed_args: tuple[str, str],
) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([*_required_run_args(), *removed_args])


def test_production_request_records_the_same_safe_defaults() -> None:
    request = ProductionRunRequest(taxon="Papilio demoleus")

    assert request.reference_admission_mode == "adaptive_gbif_fast_start"
    assert request.reference_source == "gbif"
    assert request.initial_scoring_mode == "provisional_reference_ranking"
    assert request.flickr_release_requires_human_review is True
    assert request.statistical_reference_audit is True
    assert request.stages == ADAPTIVE_REFERENCE_PRODUCTION_STAGES


def test_reference_admission_mode_exposes_all_explicit_contracts() -> None:
    parser = build_parser()

    for mode in REFERENCE_ADMISSION_MODES:
        args = parser.parse_args(
            [*_required_run_args(), "--reference-admission-mode", mode]
        )
        assert args.reference_admission_mode == mode


def test_reference_admission_help_explains_adaptive_and_strict_modes() -> None:
    help_text = build_parser()._subparsers._group_actions[0].choices["run"].format_help()

    assert "adaptive_gbif_fast_start" in help_text
    assert "human_verified_strict" in help_text
    assert "human_verified_flagged_only" in help_text
    assert "production default" in help_text
