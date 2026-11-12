"""Parse a fingerprinted discovery vocabulary, retaining its overlapping groups."""

from __future__ import annotations

import re
from pathlib import Path

from biominer.common.semantic_hash import canonical_semantic_fingerprint


def load_labels(path: Path, *, expected_count: int = 392) -> list[dict[str, str]]:
    labels = []
    group = None
    declared = {}
    for line in Path(path).read_text().splitlines():
        heading = re.fullmatch(r"## (.+) \((\d+)\)", line.strip())
        if heading:
            group = heading[1]
            declared[group] = int(heading[2])
        entry = re.fullmatch(r"- `([a-z0-9_]+)` — (.+)", line.strip())
        if entry:
            if group is None:
                raise ValueError("Label has no group")
            labels.append({"label_id": entry[1], "group": group, "text": entry[2]})
    if len({label["label_id"] for label in labels}) != len(labels):
        raise ValueError("Label bank contains duplicate identifiers")
    if len(labels) != expected_count:
        raise ValueError(f"Expected {expected_count} labels, found {len(labels)}")
    for group, count in declared.items():
        if sum(label["group"] == group for label in labels) != count:
            raise ValueError(f"Declared label count differs for {group}")
    return labels


def label_fingerprint(labels: list[dict[str, str]]) -> str:
    return canonical_semantic_fingerprint(
        {"labels": labels, "prompt": "Classification", "version": 1}
    )


def evidence_family(label: dict[str, str]) -> str | None:
    """Candidate evidence groups only; these are not exclusive image classes."""
    key, group = label["label_id"], label["group"]
    if group == "Visual Medium":
        return None
    if (
        key.startswith("real_butterfly")
        and key
        not in {
            "real_butterfly_egg",
            "real_butterfly_larva",
            "real_butterfly_pupa",
            "real_butterfly_wing",
        }
    ) or key in {"real_butterflies_multiple", "butterfly_in_jar"}:
        return "butterfly"
    if group in {
        "Butterfly Subject",
        "Other Invertebrates",
        "Other Animals",
        "Plants Fungi",
        "People Activities",
    }:
        return "other_biological"
    if key in {
        "mixed_insects",
        "flower_without_insect",
        "people_without_insect",
        "specimen_collection",
        "insect_collection",
        "seashell",
    }:
        return "other_biological"
    return "non_biological"
