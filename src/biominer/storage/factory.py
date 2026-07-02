from __future__ import annotations

from biominer.storage.cloud import CloudStorage
from biominer.storage.config import StorageConfig
from biominer.storage.local import LocalStorageBackend
from biominer.storage.s3 import S3StorageBackend


def create_storage_backend(config: StorageConfig | None = None) -> CloudStorage:
    selected = config or StorageConfig()
    backend = selected.backend.lower()
    if backend == "local":
        return LocalStorageBackend()
    if backend == "s3":
        if not selected.bucket:
            raise ValueError("StorageConfig.bucket or BIOMINER_S3_BUCKET is required for s3 storage")
        return S3StorageBackend(
            bucket=selected.bucket,
            prefix=selected.prefix,
            endpoint_url=selected.endpoint_url,
            access_key_id=selected.access_key_id,
            secret_access_key=selected.secret_access_key,
            region=selected.region,
        )
    raise ValueError(f"unsupported storage backend: {selected.backend}")
