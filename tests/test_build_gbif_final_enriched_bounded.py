from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts"
    / "build_gbif_final_enriched_bounded.py"
)


def test_bounded_builder_script_imports_and_exposes_required_cli() -> None:
    specification = importlib.util.spec_from_file_location(
        "build_gbif_final_enriched_bounded",
        SCRIPT,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)

    parser = module._parser()
    destinations = {action.dest for action in parser._actions}
    assert {
        "temporal_parquet",
        "pre_temporal_parquet",
        "temporal_audit",
        "registry_dir",
        "quality_dir",
        "state_dir",
        "output_dir",
        "producer_git_sha",
        "run_id",
        "telemetry_dir",
        "peak_rss_target_bytes",
    }.issubset(destinations)


def test_bounded_builder_help_is_runnable() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "restartable source-ordinal windows" in completed.stdout
