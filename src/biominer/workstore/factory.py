from __future__ import annotations

from biominer.storage.config import WorkStoreConfig
from biominer.workstore.base import WorkStore
from biominer.workstore.postgres import PostgresWorkStore
from biominer.workstore.sqlite import SQLiteWorkStore


def create_work_store(config: WorkStoreConfig | None = None) -> WorkStore:
    selected = config or WorkStoreConfig()
    backend = selected.backend.lower()
    if backend == "sqlite":
        return SQLiteWorkStore(selected.sqlite_path)
    if backend == "postgres":
        if not selected.dsn:
            raise ValueError("BIOMINER_WORKSTORE_DSN is required for postgres workstore")
        return PostgresWorkStore(selected.dsn)
    raise ValueError(f"unsupported workstore backend: {selected.backend}")
