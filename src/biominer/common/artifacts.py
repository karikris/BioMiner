from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl

from biominer.storage.parquet import write_parquet


@dataclass(frozen=True)
class ArtifactMetadata:
    path: str
    file_name: str
    size_bytes: int
    sha256: str
    rows: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def write_json_artifact(path: str | Path, payload: dict[str, Any]) -> ArtifactMetadata:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(output)
    return artifact_metadata(output)


def write_parquet_artifact(path: str | Path, frame: pl.DataFrame) -> ArtifactMetadata:
    output = write_parquet(frame, path)
    return artifact_metadata(output, rows=frame.height)


def artifact_metadata(path: str | Path, *, rows: int | None = None) -> ArtifactMetadata:
    source = Path(path)
    return ArtifactMetadata(
        path=str(source),
        file_name=source.name,
        size_bytes=source.stat().st_size,
        sha256=f"sha256:{hashlib.sha256(source.read_bytes()).hexdigest()}",
        rows=rows,
    )
