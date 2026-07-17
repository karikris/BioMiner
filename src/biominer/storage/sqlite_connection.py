"""SQLite connection factory with transactional close-on-exit semantics."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Any


class ClosingSQLiteConnection(sqlite3.Connection):
    """Preserve sqlite3 commit/rollback context behavior and then close."""

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> bool:
        try:
            suppress = super().__exit__(exc_type, exc_value, traceback)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise
        self.close()
        return suppress


def connect_closing(
    database: str | Path,
    **kwargs: Any,
) -> ClosingSQLiteConnection:
    """Open a connection whose ``with`` block also releases the database."""

    return sqlite3.connect(
        database,
        factory=ClosingSQLiteConnection,
        **kwargs,
    )


__all__ = ["ClosingSQLiteConnection", "connect_closing"]
