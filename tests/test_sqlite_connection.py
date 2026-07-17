from __future__ import annotations

import sqlite3

import pytest

from biominer.storage.sqlite_connection import connect_closing


def test_closing_connection_commits_and_releases_on_success(tmp_path) -> None:
    database = tmp_path / "state.sqlite"

    with connect_closing(database) as connection:
        connection.execute("CREATE TABLE evidence (value TEXT NOT NULL)")
        connection.execute("INSERT INTO evidence VALUES ('committed')")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with connect_closing(database) as verification:
        assert verification.execute("SELECT value FROM evidence").fetchall() == [
            ("committed",)
        ]


def test_closing_connection_rolls_back_and_releases_on_failure(tmp_path) -> None:
    database = tmp_path / "state.sqlite"
    with connect_closing(database) as setup:
        setup.execute("CREATE TABLE evidence (value TEXT NOT NULL)")

    with pytest.raises(RuntimeError, match="abort"):
        with connect_closing(database) as connection:
            connection.execute("INSERT INTO evidence VALUES ('rolled back')")
            raise RuntimeError("abort")

    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        connection.execute("SELECT 1")
    with connect_closing(database) as verification:
        assert verification.execute("SELECT value FROM evidence").fetchall() == []
