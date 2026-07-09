from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Any
import os
import tomllib
from urllib.parse import urlsplit

from biominer.storage.cloud import CloudStorage
from biominer.storage.local import LocalStorageBackend
from biominer.storage.s3 import S3StorageBackend
from biominer.workstore.base import WorkStore
from biominer.workstore.postgres import PostgresWorkStore
from biominer.workstore.sqlite import SQLiteWorkStore


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "s3"
    bucket: str | None = None
    prefix: str = ""
    endpoint_url_env: str | None = "BIOMINER_S3_ENDPOINT_URL"
    access_key_id_env: str | None = "BIOMINER_S3_ACCESS_KEY_ID"
    secret_access_key_env: str | None = "BIOMINER_S3_SECRET_ACCESS_KEY"
    region: str = "auto"
    endpoint_url: str | None = None
    access_key_id: str | None = field(default=None, repr=False)
    secret_access_key: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class WorkStoreConfig:
    backend: str = "postgres"
    sqlite_path: str = "data/state/biominer.sqlite"
    dsn_env: str | None = "BIOMINER_WORKSTORE_DSN"
    dsn: str | None = field(default=None, repr=False)


@dataclass(frozen=True)
class RuntimeConfig:
    worker_id_env: str = "BIOMINER_WORKER_ID"
    worker_id: str = "local"
    default_batch_rows: int = 50000
    target_parquet_mb: int = 64


@dataclass(frozen=True)
class BioMinerConfig:
    storage: StorageConfig
    workstore: WorkStoreConfig
    runtime: RuntimeConfig


@dataclass(frozen=True)
class BatchPolicy:
    max_rows: int
    target_mb: int

    def should_flush(self, *, row_count: int, estimated_bytes: int | None = None) -> bool:
        if row_count >= self.max_rows:
            return True
        if estimated_bytes is None:
            return False
        return estimated_bytes >= self.target_mb * 1024 * 1024


def load_biominer_config(path: str | Path | None = None, *, env: Mapping[str, str] | None = None) -> BioMinerConfig:
    selected_env = env if env is not None else os.environ
    raw: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    biominer = raw.get("biominer", {})
    if not isinstance(biominer, dict):
        raise ConfigError("TOML config must contain a [biominer] table")
    storage = _load_storage_config(biominer.get("storage", {}), selected_env)
    workstore = _load_workstore_config(biominer.get("workstore", {}), selected_env)
    runtime = _load_runtime_config(biominer.get("runtime", {}), selected_env)
    return BioMinerConfig(storage=storage, workstore=workstore, runtime=runtime)


def load_storage_config_from_env(env: Mapping[str, str] | None = None) -> StorageConfig:
    selected_env = env if env is not None else os.environ
    return _load_storage_config({}, selected_env)


def load_workstore_config_from_env(env: Mapping[str, str] | None = None) -> WorkStoreConfig:
    selected_env = env if env is not None else os.environ
    return _load_workstore_config({}, selected_env)


def validate_config(
    config: BioMinerConfig,
    *,
    require_cloud_credentials: bool = False,
    allow_local_backends: bool = False,
) -> None:
    if config.storage.backend not in {"local", "s3"}:
        raise ConfigError("storage.backend must be 'local' or 's3'")
    if config.workstore.backend not in {"sqlite", "postgres"}:
        raise ConfigError("workstore.backend must be 'sqlite' or 'postgres'")
    if config.runtime.default_batch_rows <= 0:
        raise ConfigError("runtime.default_batch_rows must be positive")
    if config.runtime.target_parquet_mb <= 0:
        raise ConfigError("runtime.target_parquet_mb must be positive")
    if not allow_local_backends and config.storage.backend == "local":
        raise ConfigError("storage.backend=local is allowed only with an explicit dev/test override")
    if not allow_local_backends and config.workstore.backend == "sqlite":
        raise ConfigError("workstore.backend=sqlite is allowed only with an explicit dev/test override")
    if require_cloud_credentials:
        missing = production_missing_config_variables(config)
        if missing:
            raise ConfigError("missing production config values: " + ", ".join(missing))
    if config.storage.backend == "s3":
        if not config.storage.bucket:
            raise ConfigError("storage.bucket is required for s3 storage")
    if config.workstore.backend == "postgres":
        if not config.workstore.dsn_env and not config.workstore.dsn:
            raise ConfigError("workstore.dsn_env or resolved dsn is required for postgres workstore")


def production_missing_config_variables(config: BioMinerConfig) -> list[str]:
    missing: list[str] = []
    if config.storage.backend == "s3":
        _append_missing(missing, config.storage.bucket, "BIOMINER_S3_BUCKET")
        _append_missing(missing, config.storage.prefix, "BIOMINER_S3_PREFIX")
        _append_missing(missing, config.storage.endpoint_url, config.storage.endpoint_url_env or "BIOMINER_S3_ENDPOINT_URL")
        _append_missing(missing, config.storage.access_key_id, config.storage.access_key_id_env or "BIOMINER_S3_ACCESS_KEY_ID")
        _append_missing(missing, config.storage.secret_access_key, config.storage.secret_access_key_env or "BIOMINER_S3_SECRET_ACCESS_KEY")
        _append_missing(missing, config.storage.region, "BIOMINER_S3_REGION")
    if config.workstore.backend == "postgres":
        _append_missing(missing, config.workstore.dsn, config.workstore.dsn_env or "BIOMINER_WORKSTORE_DSN")
        _append_missing(missing, _production_worker_id(config.runtime.worker_id), config.runtime.worker_id_env)
    return missing


def create_storage_backend(config: StorageConfig | None = None) -> CloudStorage:
    selected = config or StorageConfig()
    backend = selected.backend.lower()
    if backend == "local":
        return LocalStorageBackend(prefix=selected.prefix)
    if backend == "s3":
        cloud_config = BioMinerConfig(storage=selected, workstore=WorkStoreConfig(backend="sqlite", dsn_env=None), runtime=RuntimeConfig())
        validate_config(cloud_config, require_cloud_credentials=True, allow_local_backends=True)
        return S3StorageBackend(
            bucket=str(selected.bucket),
            prefix=selected.prefix,
            endpoint_url=selected.endpoint_url,
            access_key_id=selected.access_key_id,
            secret_access_key=selected.secret_access_key,
            region=selected.region,
        )
    raise ConfigError(f"unsupported storage backend: {selected.backend}")


def create_workstore(config: WorkStoreConfig | None = None) -> WorkStore:
    selected = config or WorkStoreConfig()
    backend = selected.backend.lower()
    if backend == "sqlite":
        return SQLiteWorkStore(selected.sqlite_path)
    if backend == "postgres":
        if not selected.dsn:
            raise ConfigError("BIOMINER_WORKSTORE_DSN or configured dsn_env is required for postgres workstore")
        return PostgresWorkStore(selected.dsn)
    raise ConfigError(f"unsupported workstore backend: {selected.backend}")


def redact_config(config: BioMinerConfig) -> dict[str, Any]:
    return {
        "storage": {
            "backend": config.storage.backend,
            "bucket": config.storage.bucket,
            "prefix": config.storage.prefix,
            "endpoint_url_env": config.storage.endpoint_url_env,
            "access_key_id_env": config.storage.access_key_id_env,
            "secret_access_key_env": config.storage.secret_access_key_env,
            "region": config.storage.region,
            "endpoint_url": config.storage.endpoint_url,
            "access_key_id": _redact_access_key(config.storage.access_key_id),
            "secret_access_key": "<redacted>" if config.storage.secret_access_key else None,
        },
        "workstore": {
            "backend": config.workstore.backend,
            "sqlite_path": config.workstore.sqlite_path,
            "dsn_env": config.workstore.dsn_env,
            "dsn": "<redacted>" if config.workstore.dsn else None,
        },
        "runtime": {
            "worker_id_env": config.runtime.worker_id_env,
            "worker_id": config.runtime.worker_id,
            "default_batch_rows": config.runtime.default_batch_rows,
            "target_parquet_mb": config.runtime.target_parquet_mb,
        },
    }


def redact_text(text: str, config: BioMinerConfig) -> str:
    redacted = text
    sensitive_values = {
        config.storage.access_key_id,
        config.storage.secret_access_key,
        config.workstore.dsn,
        _dsn_password(config.workstore.dsn),
    }
    for value in sorted((item for item in sensitive_values if item and len(item) >= 4), key=len, reverse=True):
        redacted = redacted.replace(value, "<redacted>")
    return redacted


def _load_storage_config(raw: Any, env: Mapping[str, str]) -> StorageConfig:
    values = _as_table(raw, "biominer.storage")
    backend = str(values.get("backend", env.get("BIOMINER_STORAGE_BACKEND", "s3"))).lower()
    prefix_default = "." if backend == "local" else env.get("BIOMINER_S3_PREFIX", "")
    endpoint_url_env = values.get("endpoint_url_env")
    access_key_id_env = values.get("access_key_id_env")
    secret_access_key_env = values.get("secret_access_key_env")
    if backend == "s3":
        endpoint_url_env = str(endpoint_url_env or "BIOMINER_S3_ENDPOINT_URL")
        access_key_id_env = str(access_key_id_env or "BIOMINER_S3_ACCESS_KEY_ID")
        secret_access_key_env = str(secret_access_key_env or "BIOMINER_S3_SECRET_ACCESS_KEY")
    if backend == "local":
        return StorageConfig(
            backend="local",
            bucket=None,
            prefix=str(values.get("prefix", prefix_default)),
            endpoint_url_env=None,
            access_key_id_env=None,
            secret_access_key_env=None,
            region="",
            endpoint_url=None,
            access_key_id=None,
            secret_access_key=None,
        )
    return StorageConfig(
        backend=backend,
        bucket=_optional_str(values.get("bucket", env.get("BIOMINER_S3_BUCKET"))),
        prefix=str(values.get("prefix", prefix_default)),
        endpoint_url_env=_optional_str(endpoint_url_env),
        access_key_id_env=_optional_str(access_key_id_env),
        secret_access_key_env=_optional_str(secret_access_key_env),
        region=str(values.get("region", env.get("BIOMINER_S3_REGION", ""))),
        endpoint_url=_optional_str(values.get("endpoint_url") or (env.get(str(endpoint_url_env)) if endpoint_url_env else env.get("BIOMINER_S3_ENDPOINT_URL"))),
        access_key_id=_optional_str(values.get("access_key_id") or (env.get(str(access_key_id_env)) if access_key_id_env else env.get("BIOMINER_S3_ACCESS_KEY_ID"))),
        secret_access_key=_optional_str(
            values.get("secret_access_key") or (env.get(str(secret_access_key_env)) if secret_access_key_env else env.get("BIOMINER_S3_SECRET_ACCESS_KEY"))
        ),
    )


def _load_workstore_config(raw: Any, env: Mapping[str, str]) -> WorkStoreConfig:
    values = _as_table(raw, "biominer.workstore")
    backend = str(values.get("backend", env.get("BIOMINER_WORKSTORE_BACKEND", "postgres"))).lower()
    dsn_env = values.get("dsn_env")
    if backend == "postgres":
        dsn_env = str(dsn_env or "BIOMINER_WORKSTORE_DSN")
    return WorkStoreConfig(
        backend=backend,
        sqlite_path=str(values.get("sqlite_path", env.get("BIOMINER_WORKSTORE_SQLITE_PATH", "data/state/biominer.sqlite"))),
        dsn_env=_optional_str(dsn_env),
        dsn=_optional_str(values.get("dsn") or (env.get(str(dsn_env)) if dsn_env else env.get("BIOMINER_WORKSTORE_DSN"))),
    )


def _load_runtime_config(raw: Any, env: Mapping[str, str]) -> RuntimeConfig:
    values = _as_table(raw, "biominer.runtime")
    worker_id_env = str(values.get("worker_id_env", "BIOMINER_WORKER_ID"))
    return RuntimeConfig(
        worker_id_env=worker_id_env,
        worker_id=str(values.get("worker_id", env.get(worker_id_env, ""))),
        default_batch_rows=int(values.get("default_batch_rows", 50000)),
        target_parquet_mb=int(values.get("target_parquet_mb", 64)),
    )


def _as_table(raw: Any, name: str) -> dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{name} must be a TOML table")
    return raw


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _append_missing(missing: list[str], value: str | None, env_name: str | None) -> None:
    if value:
        return
    missing.append(str(env_name or "<unknown>"))


def _production_worker_id(value: str | None) -> str:
    cleaned = str(value or "").strip()
    return "" if cleaned.casefold() == "local" else cleaned


def _redact_access_key(value: str | None) -> str | None:
    if not value:
        return None
    return "<redacted>"


def _dsn_password(dsn: str | None) -> str | None:
    if not dsn:
        return None
    try:
        return urlsplit(dsn).password
    except ValueError:
        return None
