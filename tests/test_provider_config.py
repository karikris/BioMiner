from __future__ import annotations

import pytest

from biominer.config import (
    BatchPolicy,
    BioMinerConfig,
    ConfigError,
    RuntimeConfig,
    StorageConfig,
    WorkStoreConfig,
    create_storage_backend,
    create_workstore,
    load_biominer_config,
    redact_config,
    validate_config,
)
from biominer.storage.local import LocalStorageBackend
from biominer.workstore.sqlite import SQLiteWorkStore


def test_load_defaults_without_file() -> None:
    config = load_biominer_config(None, env={})

    assert config.storage.backend == "local"
    assert config.storage.prefix == "."
    assert config.workstore.backend == "sqlite"
    assert config.workstore.sqlite_path == "data/state/biominer.sqlite"
    assert config.runtime.default_batch_rows == 50000
    assert config.runtime.target_parquet_mb == 64


def test_load_cloud_toml_with_env_resolution(tmp_path) -> None:
    path = tmp_path / "biominer.toml"
    path.write_text(
        """
        [biominer.storage]
        backend = "s3"
        bucket = "biominer"
        prefix = "biominer"
        endpoint_url_env = "BIOMINER_S3_ENDPOINT_URL"
        access_key_id_env = "BIOMINER_S3_ACCESS_KEY_ID"
        secret_access_key_env = "BIOMINER_S3_SECRET_ACCESS_KEY"
        region = "auto"

        [biominer.workstore]
        backend = "postgres"
        dsn_env = "BIOMINER_WORKSTORE_DSN"

        [biominer.runtime]
        worker_id_env = "BIOMINER_WORKER_ID"
        default_batch_rows = 50000
        target_parquet_mb = 128
        """,
        encoding="utf-8",
    )
    env = {
        "BIOMINER_S3_ENDPOINT_URL": "https://s3.us-west-004.backblazeb2.com",
        "BIOMINER_S3_ACCESS_KEY_ID": "fake-key-id",
        "BIOMINER_S3_SECRET_ACCESS_KEY": "fake-secret",
        "BIOMINER_WORKSTORE_DSN": "postgresql://user:password@example.test:5432/postgres",
        "BIOMINER_WORKER_ID": "worker-001",
    }

    config = load_biominer_config(path, env=env)

    assert config.storage.endpoint_url == "https://s3.us-west-004.backblazeb2.com"
    assert config.storage.access_key_id == "fake-key-id"
    assert config.storage.secret_access_key == "fake-secret"
    assert config.workstore.dsn == "postgresql://user:password@example.test:5432/postgres"
    assert config.runtime.worker_id == "worker-001"


def test_missing_cloud_env_raises_on_validation(tmp_path) -> None:
    path = tmp_path / "biominer.toml"
    path.write_text(
        """
        [biominer.storage]
        backend = "s3"
        bucket = "biominer"
        endpoint_url_env = "BIOMINER_S3_ENDPOINT_URL"
        access_key_id_env = "BIOMINER_S3_ACCESS_KEY_ID"
        secret_access_key_env = "BIOMINER_S3_SECRET_ACCESS_KEY"

        [biominer.workstore]
        backend = "postgres"
        dsn_env = "BIOMINER_WORKSTORE_DSN"
        """,
        encoding="utf-8",
    )
    config = load_biominer_config(path, env={})

    with pytest.raises(ConfigError, match="BIOMINER_S3_ENDPOINT_URL"):
        validate_config(config, require_cloud_credentials=True)


def test_local_config_does_not_require_cloud_env() -> None:
    config = BioMinerConfig(storage=StorageConfig(), workstore=WorkStoreConfig(), runtime=RuntimeConfig())

    validate_config(config, require_cloud_credentials=True)


def test_invalid_backend_rejected() -> None:
    config = BioMinerConfig(storage=StorageConfig(backend="dropbox"), workstore=WorkStoreConfig(), runtime=RuntimeConfig())

    with pytest.raises(ConfigError, match="storage.backend"):
        validate_config(config)


def test_batch_policy_row_flush() -> None:
    policy = BatchPolicy(max_rows=50000, target_mb=128)

    assert not policy.should_flush(row_count=49999)
    assert policy.should_flush(row_count=50000)


def test_batch_policy_byte_flush() -> None:
    policy = BatchPolicy(max_rows=50000, target_mb=128)

    assert policy.should_flush(row_count=1, estimated_bytes=128 * 1024 * 1024)


def test_redaction_hides_secrets() -> None:
    config = BioMinerConfig(
        storage=StorageConfig(
            backend="s3",
            bucket="biominer",
            prefix="biominer",
            endpoint_url="https://s3.us-west-004.backblazeb2.com",
            access_key_id="fake-key-id",
            secret_access_key="super-secret",
        ),
        workstore=WorkStoreConfig(
            backend="postgres",
            dsn="postgresql://user:password@example.test:5432/postgres",
        ),
        runtime=RuntimeConfig(worker_id="worker-001"),
    )

    redacted = redact_config(config)
    rendered = repr(redacted)

    assert "super-secret" not in rendered
    assert "password" not in rendered
    assert redacted["storage"]["secret_access_key"] == "<redacted>"
    assert redacted["workstore"]["dsn"] == "<redacted>"


def test_factories_local(tmp_path) -> None:
    config = BioMinerConfig(
        storage=StorageConfig(prefix=str(tmp_path)),
        workstore=WorkStoreConfig(sqlite_path=str(tmp_path / "state.sqlite")),
        runtime=RuntimeConfig(),
    )

    storage = create_storage_backend(config.storage)
    store = create_workstore(config.workstore)

    assert isinstance(storage, LocalStorageBackend)
    assert storage.prefix == str(tmp_path)
    assert isinstance(store, SQLiteWorkStore)


def test_factory_s3_missing_env_is_clear() -> None:
    config = StorageConfig(backend="s3", bucket="biominer", prefix="biominer")

    with pytest.raises(ConfigError, match="endpoint"):
        create_storage_backend(config)


def test_cli_accepts_config_option_if_wired(tmp_path) -> None:
    from biominer.cli import build_parser

    path = tmp_path / "biominer.local.toml"
    path.write_text(
        """
        [biominer.storage]
        backend = "local"
        prefix = "."
        """,
        encoding="utf-8",
    )

    args = build_parser().parse_args(["--config", str(path), "--version"])

    assert args.config == str(path)
