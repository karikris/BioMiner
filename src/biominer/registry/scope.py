from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class ButterflyScope:
    scope_id: str
    root_scientific_name: str
    root_rank: str
    included_families: tuple[str, ...]
    root_taxon_key: str = ""
    family_taxon_keys: dict[str, str] = field(default_factory=dict)


def load_scope(path: str | Path = "config/butterfly_scope.json") -> ButterflyScope:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    root = payload["root"]
    families = tuple(str(item) for item in payload["included_families"])
    return ButterflyScope(
        scope_id=str(payload["scope_id"]),
        root_scientific_name=str(root["scientific_name"]),
        root_rank=str(root["rank"]),
        included_families=families,
        root_taxon_key=str(root.get("gbif_taxon_key") or ""),
        family_taxon_keys={
            str(name): str(key)
            for name, key in payload.get("gbif_family_taxon_keys", {}).items()
        },
    )
