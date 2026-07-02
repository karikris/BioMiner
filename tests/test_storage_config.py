from __future__ import annotations

import pytest

from biominer.storage.config import StorageConfig, WorkStoreConfig, load_storage_config_from_env, load_workstore_config_from_env
from biominer.storage.factory import create_storage_backend
from biominer.workstore.factory import create_work_store
from biominer.workstore.sqlite import SQLiteWorkStore


def test_storage_config_defaults_to_local(monkeypatch) -> None:
    for key in (
        "BIOMINER_STORAGE_BACKEND",
        "BIOMINER_S3_ENDPOINT_URL",
        "BIOMINER_S3_ACCESS_KEY_ID",
        "BIOMINER_S3_SECRET_ACCESS_KEY",
        "BIOMINER_S3_REGION",
        "BIOMINER_S3_BUCKET",
        "BIOMINER_S3_PREFIX",
    ):
        monkeypatch.delenv(key, raising=False)

    config = load_storage_config_from_env()

    assert config == StorageConfig()
    assert create_storage_backend(config).__class__.__name__ == "LocalStorageBackend"


def test_storage_config_reads_s3_env_without_requiring_credentials_at_import(monkeypatch) -> None:
    monkeypatch.setenv("BIOMINER_STORAGE_BACKEND", "s3")
    monkeypatch.setenv("BIOMINER_S3_ENDPOINT_URL", "https://s3.us-west-004.backblazeb2.com")
    monkeypatch.setenv("BIOMINER_S3_ACCESS_KEY_ID", "key-id")
    monkeypatch.setenv("BIOMINER_S3_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("BIOMINER_S3_REGION", "us-west-004")
    monkeypatch.setenv("BIOMINER_S3_BUCKET", "biominer")
    monkeypatch.setenv("BIOMINER_S3_PREFIX", "prod")

    config = load_storage_config_from_env()
    backend = create_storage_backend(config)

    assert config.backend == "s3"
    assert config.endpoint_url == "https://s3.us-west-004.backblazeb2.com"
    assert config.bucket == "biominer"
    assert backend.base_uri == "s3://biominer/prod"


def test_s3_factory_requires_bucket_for_s3_backend() -> None:
    with pytest.raises(ValueError, match="bucket"):
        create_storage_backend(StorageConfig(backend="s3", endpoint_url="https://example.test"))


def test_workstore_config_defaults_to_sqlite(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("BIOMINER_WORKSTORE_BACKEND", raising=False)
    monkeypatch.delenv("BIOMINER_WORKSTORE_DSN", raising=False)
    monkeypatch.setenv("BIOMINER_WORKSTORE_SQLITE_PATH", str(tmp_path / "state.sqlite"))

    config = load_workstore_config_from_env()
    store = create_work_store(config)

    assert config == WorkStoreConfig(sqlite_path=str(tmp_path / "state.sqlite"))
    assert isinstance(store, SQLiteWorkStore)


def test_postgres_factory_requires_dsn() -> None:
    with pytest.raises(ValueError, match="BIOMINER_WORKSTORE_DSN"):
        create_work_store(WorkStoreConfig(backend="postgres", dsn=None))
