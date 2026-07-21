from __future__ import annotations

from biominer.common.status import CLAIMED, COMPLETED, FAILED, PENDING, RUN_COMPLETED, RUN_FAILED, RUN_PLANNED, RUN_RUNNING


def test_status_constants_have_expected_wire_values() -> None:
    assert (PENDING, CLAIMED, COMPLETED, FAILED) == ("pending", "claimed", "completed", "failed")
    assert (RUN_PLANNED, RUN_RUNNING, RUN_COMPLETED, RUN_FAILED) == ("planned", "running", "completed", "failed")
