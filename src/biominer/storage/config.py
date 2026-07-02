from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class StorageConfig:
    backend: str = "local"
    bucket: str | None = None
    prefix: str = ""
    endpoint_url: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    region: str = "auto"


@dataclass(frozen=True)
class WorkStoreConfig:
    backend: str = "sqlite"
    dsn: str | None = None
    sqlite_path: str = "data/state/biominer.sqlite"


def load_storage_config_from_env() -> StorageConfig:
    return StorageConfig(
        backend=os.environ.get("BIOMINER_STORAGE_BACKEND", "local").lower(),
        bucket=_env_optional("BIOMINER_S3_BUCKET"),
        prefix=os.environ.get("BIOMINER_S3_PREFIX", ""),
        endpoint_url=_env_optional("BIOMINER_S3_ENDPOINT_URL"),
        access_key_id=_env_optional("BIOMINER_S3_ACCESS_KEY_ID"),
        secret_access_key=_env_optional("BIOMINER_S3_SECRET_ACCESS_KEY"),
        region=os.environ.get("BIOMINER_S3_REGION", "auto"),
    )


def load_workstore_config_from_env() -> WorkStoreConfig:
    return WorkStoreConfig(
        backend=os.environ.get("BIOMINER_WORKSTORE_BACKEND", "sqlite").lower(),
        dsn=_env_optional("BIOMINER_WORKSTORE_DSN"),
        sqlite_path=os.environ.get("BIOMINER_WORKSTORE_SQLITE_PATH", "data/state/biominer.sqlite"),
    )


def _env_optional(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None
