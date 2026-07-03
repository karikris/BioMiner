from __future__ import annotations

import ast
from pathlib import Path

from biominer.common.status import CLAIMED, COMPLETED, FAILED, PENDING, RUN_COMPLETED, RUN_FAILED, RUN_PLANNED, RUN_RUNNING


def test_status_constants_have_expected_wire_values() -> None:
    assert (PENDING, CLAIMED, COMPLETED, FAILED) == ("pending", "claimed", "completed", "failed")
    assert (RUN_PLANNED, RUN_RUNNING, RUN_COMPLETED, RUN_FAILED) == ("planned", "running", "completed", "failed")


def test_status_constants_are_not_redefined_in_stateful_modules() -> None:
    repo = Path(__file__).resolve().parents[1]
    forbidden = {
        "PENDING",
        "CLAIMED",
        "COMPLETED",
        "FAILED",
        "RUN_PLANNED",
        "RUN_RUNNING",
        "RUN_COMPLETED",
        "RUN_FAILED",
    }
    checked = (
        "src/biominer/workstore/sqlite.py",
        "src/biominer/workstore/postgres.py",
        "src/biominer/flickr_fetch/metadata_poller.py",
        "src/biominer/flickr_comments/comments_enrichment.py",
    )
    offenders: list[str] = []
    for relative in checked:
        tree = ast.parse((repo / relative).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in forbidden:
                    offenders.append(f"{relative}:{target.id}")

    assert offenders == []
