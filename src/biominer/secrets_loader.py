from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex


DEFAULT_SECRETS_ENV = Path("/Applications/secrets/secrets.env")
SECRETS_ENV_PATH_VAR = "BIOMINER_SECRETS_ENV"
SECRETS_LOAD_DISABLE_VAR = "BIOMINER_LOAD_SECRETS_ENV"


@dataclass(frozen=True)
class SecretsLoadResult:
    path: Path
    exists: bool
    loaded_names: tuple[str, ...] = ()
    skipped_existing_names: tuple[str, ...] = ()
    skipped_invalid_lines: int = 0
    error: str | None = None


def load_runtime_secrets_env(path: str | Path | None = None, *, override: bool = False) -> SecretsLoadResult:
    target = _secrets_path(path)
    if str(os.environ.get(SECRETS_LOAD_DISABLE_VAR, "")).casefold() in {"0", "false", "no", "off"}:
        return SecretsLoadResult(path=target, exists=target.exists())
    try:
        if not target.exists():
            return SecretsLoadResult(path=target, exists=False)
        loaded: list[str] = []
        skipped_existing: list[str] = []
        skipped_invalid = 0
        for line in target.read_text(encoding="utf-8").splitlines():
            item = _parse_env_line(line)
            if item is None:
                if _looks_like_assignment(line):
                    skipped_invalid += 1
                continue
            name, value = item
            if not override and name in os.environ:
                skipped_existing.append(name)
                continue
            os.environ[name] = value
            loaded.append(name)
        return SecretsLoadResult(
            path=target,
            exists=True,
            loaded_names=tuple(loaded),
            skipped_existing_names=tuple(skipped_existing),
            skipped_invalid_lines=skipped_invalid,
        )
    except OSError as exc:
        return SecretsLoadResult(path=target, exists=True, error=str(exc))


def _secrets_path(path: str | Path | None) -> Path:
    if path:
        return Path(path).expanduser()
    configured = os.environ.get(SECRETS_ENV_PATH_VAR)
    if configured:
        return Path(configured).expanduser()
    for candidate in _default_secret_paths():
        if candidate.exists():
            return candidate
    return DEFAULT_SECRETS_ENV


def _default_secret_paths() -> tuple[Path, ...]:
    sibling = _runtime_base_path() / "secrets" / "secrets.env"
    return (
        DEFAULT_SECRETS_ENV,
        sibling,
        Path("/mnt/c/Applications/secrets/secrets.env"),
    )


def _runtime_base_path() -> Path:
    try:
        from biominer.runtime_paths import resolve_runtime_base_path
    except Exception:  # pragma: no cover - defensive startup fallback.
        return Path.cwd()
    return resolve_runtime_base_path()


def _parse_env_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        parts = shlex.split(stripped, comments=True, posix=True)
    except ValueError:
        return None
    if not parts:
        return None
    if parts[0].casefold() == "export":
        parts = parts[1:]
    if len(parts) == 3 and parts[1] == "=":
        parts = [f"{parts[0]}={parts[2]}"]
    if len(parts) != 1 or "=" not in parts[0]:
        return None
    name, value = parts[0].split("=", 1)
    if not name.isidentifier():
        return None
    return name, value


def _looks_like_assignment(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped and not stripped.startswith("#") and "=" in stripped)
