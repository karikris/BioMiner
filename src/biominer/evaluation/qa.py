from __future__ import annotations

import json
from typing import Any, Mapping

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.detection.policy import DetectionPolicy, detection_is_bioclip_eligible
from biominer.evaluation.review_queue import build_hierarchical_review_queue
from biominer.evaluation.thresholds import VisionBucketPolicy, load_vision_bucket_policy


VISUAL_QA_FINDINGS_SCHEMA: dict[str, pl.DataType] = {
    "finding_id": pl.String,
    "severity": pl.String,
    "finding_type": pl.String,
    "source": pl.String,
    "flickr_photo_id": pl.String,
    "detection_id": pl.String,
    "message": pl.String,
    "suggested_action": pl.String,
    "classification_mode": pl.String,
}

HIGH_PRIORITY_REVIEW_THRESHOLD = 80


def build_visual_qa_findings(
    *,
    object_evidence: pl.DataFrame,
    photo_summary: pl.DataFrame | None = None,
    reviewed_labels: pl.DataFrame | None = None,
    policy: VisionBucketPolicy | None = None,
) -> pl.DataFrame:
    """Build deterministic QA findings for visual classification artifacts."""

    del reviewed_labels  # reserved for later label-aware QA without changing the public signature.
    active_policy = policy or load_vision_bucket_policy()
    findings: list[dict[str, str]] = []
    rows = object_evidence.to_dicts() if not object_evidence.is_empty() else []
    for row in rows:
        _append_row_findings(findings, row, policy=active_policy)
    _append_photo_summary_findings(findings, photo_summary)
    _append_multi_object_conflict_findings(findings, rows)
    _append_high_priority_review_findings(
        findings,
        object_evidence=object_evidence,
        photo_summary=photo_summary,
        policy=active_policy,
    )
    return _findings_frame(findings)


def empty_visual_qa_findings_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=VISUAL_QA_FINDINGS_SCHEMA)


def _append_row_findings(findings: list[dict[str, str]], row: Mapping[str, Any], *, policy: VisionBucketPolicy) -> None:
    if _is_hierarchical(row) and not _selected_family(row):
        findings.append(
            _finding(
                "fatal",
                "hierarchical_missing_selected_family",
                row,
                "hierarchical classifier row is missing selected family",
                "rerun hierarchical scoring and verify family top-k output",
            )
        )
    if not _is_path_cascade_row(row) and _species_top20_outside_selected_family(row):
        findings.append(
            _finding(
                "fatal",
                "species_top20_outside_selected_family",
                row,
                "species_top20 contains a species outside the selected family",
                "rebuild the classification table and rerun family-first species filtering",
            )
        )
    if _is_path_cascade_row(row):
        _append_path_cascade_findings(findings, row)
    if _species_top5_not_subset_top20(row):
        findings.append(
            _finding(
                "fatal",
                "species_top5_not_subset_species_top20",
                row,
                "species_top5 contains candidates not present in species_top20",
                "rerun species reranking from the full constrained top20 candidate set",
            )
        )
    if _has_bioclip_score_for_noneligible_detection(row):
        findings.append(
            _finding(
                "fatal",
                "bioclip_score_for_noneligible_detection",
                row,
                "BioCLIP score exists for a detector label that is not BioCLIP-eligible",
                "skip BioCLIP scoring for non-eligible detections and regenerate object evidence",
            )
        )
    if _is_butterfly_like_detection(row) and not _has_bioclip_score(row):
        findings.append(
            _finding(
                "warning",
                "butterfly_like_missing_bioclip_score",
                row,
                "butterfly_like detection has no BioCLIP score",
                "inspect scorer logs and rerun missing butterfly_like crops",
            )
        )
    if _is_hierarchical(row) and not _string_list(row.get("family_top3")):
        findings.append(
            _finding(
                "warning",
                "empty_family_top3",
                row,
                "hierarchical classifier row has empty family_top3",
                "verify family prompt scoring and taxonomy label availability",
            )
        )
    if _is_hierarchical(row) and not _string_list(row.get("species_top5")):
        findings.append(
            _finding(
                "warning",
                "empty_species_top5",
                row,
                "hierarchical classifier row has empty species_top5",
                "verify family-constrained species candidates and rerank output",
            )
        )
    margin = _species_margin(row)
    if margin is not None and margin <= policy.minimum_species_margin:
        findings.append(
            _finding(
                "warning",
                "very_low_species_margin",
                row,
                "species top1 margin is below the visual QA review threshold",
                "prioritize this row for manual review",
            )
        )
    if _metadata_species_conflict(row):
        findings.append(
            _finding(
                "warning",
                "metadata_vision_species_conflict",
                row,
                "metadata species evidence conflicts with BioCLIP top species",
                "review Flickr text/comment evidence against visual classifier output",
            )
        )


def _append_photo_summary_findings(findings: list[dict[str, str]], photo_summary: pl.DataFrame | None) -> None:
    if photo_summary is None or photo_summary.is_empty():
        return
    for row in photo_summary.to_dicts():
        if _truthy(row.get("photo_multi_object_conflict")):
            findings.append(
                _finding(
                    "warning",
                    "multi_object_species_conflict",
                    row,
                    "photo summary reports multiple object-level species predictions",
                    "review all butterfly detections for this photo together",
                )
            )


def _append_multi_object_conflict_findings(findings: list[dict[str, str]], rows: list[dict[str, Any]]) -> None:
    species_by_photo: dict[tuple[str, str], set[str]] = {}
    row_by_photo: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        species = _species_name(row)
        if not species:
            continue
        key = (_text(row.get("source")), _text(row.get("flickr_photo_id")))
        species_by_photo.setdefault(key, set()).add(species.casefold())
        row_by_photo.setdefault(key, row)
    for key, species in species_by_photo.items():
        if len(species) <= 1:
            continue
        findings.append(
            _finding(
                "warning",
                "multi_object_species_conflict",
                row_by_photo[key],
                "photo has multiple butterfly detections with different species top1 predictions",
                "review all butterfly detections for this photo together",
                detection_id="",
            )
        )


def _append_high_priority_review_findings(
    findings: list[dict[str, str]],
    *,
    object_evidence: pl.DataFrame,
    photo_summary: pl.DataFrame | None,
    policy: VisionBucketPolicy,
) -> None:
    if object_evidence.is_empty():
        return
    queue = build_hierarchical_review_queue(
        object_evidence=object_evidence,
        photo_summary=photo_summary,
        policy=policy,
    )
    for row in queue.to_dicts():
        priority = _optional_int(row.get("review_priority"))
        if priority is None or priority < HIGH_PRIORITY_REVIEW_THRESHOLD:
            continue
        findings.append(
            _finding(
                "info",
                "high_priority_review_row",
                row,
                f"review queue priority is {priority}",
                "send this row to the front of the manual visual QA queue",
            )
        )


def _findings_frame(findings: list[dict[str, str]]) -> pl.DataFrame:
    if not findings:
        return empty_visual_qa_findings_frame()
    findings = sorted(
        findings,
        key=lambda row: (
            _severity_rank(row["severity"]),
            row["finding_type"],
            row["source"],
            row["flickr_photo_id"],
            row["detection_id"],
            row["message"],
        ),
    )
    rows = [{**row, "finding_id": f"visual_qa_{index:06d}"} for index, row in enumerate(findings, start=1)]
    return pl.DataFrame(rows, schema=VISUAL_QA_FINDINGS_SCHEMA)


def _finding(
    severity: str,
    finding_type: str,
    row: Mapping[str, Any],
    message: str,
    suggested_action: str,
    *,
    detection_id: str | None = None,
) -> dict[str, str]:
    return {
        "finding_id": "",
        "severity": severity,
        "finding_type": finding_type,
        "source": _text(row.get("source")),
        "flickr_photo_id": _text(row.get("flickr_photo_id")),
        "detection_id": _text(row.get("detection_id")) if detection_id is None else detection_id,
        "message": message,
        "suggested_action": suggested_action,
        "classification_mode": _text(row.get("classification_mode")),
    }


def _species_top20_outside_selected_family(row: Mapping[str, Any]) -> bool:
    selected_key = _first_text(row, "selected_family_key", "family_top1_accepted_taxon_key")
    family_keys = _string_list(row.get("species_top20_family_keys") or row.get("species_top20_candidate_family_keys"))
    if selected_key and family_keys and any(key != selected_key for key in family_keys):
        return True
    selected_family = _selected_family(row).casefold()
    families = [family.casefold() for family in _string_list(row.get("species_top20_families"))]
    if selected_family and families and any(family != selected_family for family in families):
        return True
    species_candidate_family_key = _first_text(row, "species_candidate_family_key", "species_top1_family_key")
    if selected_key and species_candidate_family_key and selected_key != species_candidate_family_key:
        return True
    species_candidate_family = _first_text(row, "species_candidate_family", "species_top1_family").casefold()
    return bool(selected_family and species_candidate_family and selected_family != species_candidate_family)


def _append_path_cascade_findings(
    findings: list[dict[str, str]],
    row: Mapping[str, Any],
) -> None:
    for prefix in ("family", "subfamily", "tribe", "genus"):
        if not _string_list(row.get(f"{prefix}_top3")) or not _text(
            row.get(f"selected_{prefix}")
        ):
            findings.append(
                _finding(
                    "fatal",
                    "hierarchical_missing_rank_result",
                    row,
                    f"global cascade output is missing required {prefix.upper()} results",
                    "rebuild classification-v3 artifacts and rerun the global cascade",
                )
            )
    skipped = set(_string_list(row.get("skipped_ranks")))
    if not _text(row.get("selected_subtribe")) and "SUBTRIBE" not in skipped:
        findings.append(
            _finding(
                "fatal",
                "hierarchical_missing_subtribe_or_skip",
                row,
                "global cascade output has neither a selected SUBTRIBE nor reviewed skip",
                "repair reviewed SUBTRIBE path evidence and rerun classification",
            )
        )
    for prefix in ("family", "subfamily", "tribe", "subtribe", "genus"):
        lengths = {
            len(_list_like(row.get(f"{prefix}_top3"))),
            len(_list_like(row.get(f"{prefix}_top3_node_ids"))),
            len(_list_like(row.get(f"{prefix}_top3_scores"))),
        }
        if len(lengths) != 1:
            findings.append(
                _finding(
                    "fatal",
                    "cascade_top3_arrays_misaligned",
                    row,
                    f"{prefix.upper()} top-three names, node IDs, and scores are misaligned",
                    "regenerate the versioned cascade output row",
                )
            )
    overlay_ids = {
        _text(value)
        for prefix in ("family", "subfamily", "tribe", "subtribe", "genus")
        for value in _list_like(row.get(f"{prefix}_top3_node_ids"))
        if _text(value)
    }
    accepted_keys = {
        _text(value)
        for column in (
            "species_top20_accepted_taxon_keys",
            "species_top5_accepted_taxon_keys",
            "species_top3_accepted_taxon_keys",
        )
        for value in _list_like(row.get(column))
        if _text(value)
    }
    if overlay_ids & accepted_keys:
        findings.append(
            _finding(
                "fatal",
                "overlay_node_id_in_accepted_taxon_keys",
                row,
                "reviewed classification node ID appears in a GBIF accepted-key field",
                "regenerate outputs with species mappings and node-ID fields separated",
            )
        )


def _species_top5_not_subset_top20(row: Mapping[str, Any]) -> bool:
    top5_names = {_norm(value) for value in _string_list(row.get("species_top5"))}
    top20_names = {_norm(value) for value in _string_list(row.get("species_top20"))}
    top5_keys = {_norm(value) for value in _string_list(row.get("species_top5_accepted_taxon_keys"))}
    top20_keys = {_norm(value) for value in _string_list(row.get("species_top20_accepted_taxon_keys"))}
    return bool(
        (top5_names and top20_names and not top5_names <= top20_names)
        or (top5_keys and top20_keys and not top5_keys <= top20_keys)
    )


def _metadata_species_conflict(row: Mapping[str, Any]) -> bool:
    if _truthy(row.get("bioclip_tag_conflict")) or _truthy(row.get("metadata_species_conflict")):
        return True
    predicted = _species_name(row).casefold()
    if not predicted:
        return False
    for field in (
        "flickr_text_species_candidate",
        "metadata_species_candidate",
        "text_species_candidate",
        "flickr_tag_species_candidate",
        "comment_species_candidate",
    ):
        candidate = _text(row.get(field)).casefold()
        if candidate and candidate != predicted:
            return True
    return False


def _is_hierarchical(row: Mapping[str, Any]) -> bool:
    return _text(row.get("classification_mode")) == HIERARCHICAL_BUTTERFLY_CLASSIFICATION


def _has_bioclip_score(row: Mapping[str, Any]) -> bool:
    return (
        _optional_float(row.get("species_top1_score")) is not None
        or _optional_float(row.get("target_species_score")) is not None
        or bool(_species_name(row))
    )


def _has_bioclip_score_for_noneligible_detection(row: Mapping[str, Any]) -> bool:
    if not _has_bioclip_score(row):
        return False
    label = _text(row.get("detector_label"))
    status = _text(row.get("detection_status"))
    if not label and not status:
        return False
    if status and status != "detected":
        return True
    if not label:
        return False
    return label not in set(DetectionPolicy().bioclip_eligible_labels)


def _is_butterfly_like_detection(row: Mapping[str, Any]) -> bool:
    if detection_is_bioclip_eligible(dict(row)):
        return True
    label = _text(row.get("detector_label")).casefold()
    return label in {"butterfly_like", "butterfly", "adult_butterfly"}


def _selected_family(row: Mapping[str, Any]) -> str:
    selected = _text(row.get("selected_family"))
    if _is_path_cascade_row(row):
        return selected
    return selected or _text(row.get("family_top1"))


def _is_path_cascade_row(row: Mapping[str, Any]) -> bool:
    return _text(row.get("classifier_schema_version")).startswith(
        "butterfly-cascade-output-"
    )


def _species_name(row: Mapping[str, Any]) -> str:
    return _first_text(row, "species_top1_scientific_name", "species_top1")


def _species_margin(row: Mapping[str, Any]) -> float | None:
    explicit = _optional_float(row.get("species_top1_margin"))
    if explicit is not None:
        return explicit
    scores = _float_list(
        row.get("species_top5_rerank_scores") or row.get("species_top5_scores")
    )
    if len(scores) < 2:
        return None
    return float(scores[0] - scores[1])


def _first_text(row: Mapping[str, Any], *columns: str) -> str:
    for column in columns:
        text = _text(row.get(column))
        if text:
            return text
    return ""


def _string_list(value: object) -> list[str]:
    return [_text(item) for item in _list_like(value) if _text(item)]


def _float_list(value: object) -> list[float]:
    return [number for item in _list_like(value) if (number := _optional_float(item)) is not None]


def _list_like(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, pl.Series):
        return value.to_list()
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return [value]


def _optional_float(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).casefold()
    return text in {"1", "true", "yes", "y"}


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _norm(value: object) -> str:
    return _text(value).casefold()


def _severity_rank(severity: str) -> int:
    return {"fatal": 0, "warning": 1, "info": 2}.get(severity, 99)


__all__ = [
    "HIGH_PRIORITY_REVIEW_THRESHOLD",
    "VISUAL_QA_FINDINGS_SCHEMA",
    "build_visual_qa_findings",
    "empty_visual_qa_findings_frame",
]
