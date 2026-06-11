from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json_config(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("BioMiner JSON config must contain a top-level object")
    return data


def load_term_list(path: str | Path, *, key: str = "terms") -> tuple[str, ...]:
    data = load_json_config(path)
    values = data.get(key, data)
    if isinstance(values, dict):
        terms = [term for group in values.values() if isinstance(group, list) for term in group]
    elif isinstance(values, list):
        terms = values
    else:
        raise ValueError(f"Expected {key!r} to be a list or grouped object")
    return tuple(str(term).strip() for term in terms if str(term).strip())
