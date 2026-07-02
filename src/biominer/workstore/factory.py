from __future__ import annotations

from biominer.config import WorkStoreConfig, create_workstore
from biominer.workstore.base import WorkStore


def create_work_store(config: WorkStoreConfig | None = None) -> WorkStore:
    return create_workstore(config)


__all__ = ["WorkStore", "WorkStoreConfig", "create_work_store", "create_workstore"]
