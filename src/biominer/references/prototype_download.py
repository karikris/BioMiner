from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime

import polars as pl

from biominer.references.licensing import ReferenceLicencePolicy
from biominer.references.prototype_acquisition import (
    make_prototype_visual_reference_ids,
    validate_prototype_reference_selections,
)
from biominer.references.schemas import (
    REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
    reference_media_candidates_frame,
    validate_reference_media_candidates,
)


PROTOTYPE_DOWNLOAD_INPUT_POLICY_VERSION = "prototype-download-inputs-v1.0.0"


def compile_prototype_download_candidates(
    *,
    selections: pl.DataFrame,
    biological_media_candidates: Sequence[pl.DataFrame],
    visual_domain_manifest: Mapping[str, object],
    licence_policy: ReferenceLicencePolicy,
) -> pl.DataFrame:
    validate_prototype_reference_selections(selections)
    if not isinstance(licence_policy, ReferenceLicencePolicy):
        raise TypeError("licence_policy must be a ReferenceLicencePolicy")
    biological = _combined_biological_candidates(biological_media_candidates)
    biological_by_id = {
        str(row["reference_media_id"]): row
        for row in biological.iter_rows(named=True)
    }
    visual_by_id = _visual_candidates_by_media_id(visual_domain_manifest)
    rows: list[dict[str, object]] = []
    for selection in selections.iter_rows(named=True):
        scope_type = str(selection["candidate_scope_type"])
        if scope_type == "accepted_taxon":
            media_id = str(selection["reference_media_id"])
            source = biological_by_id.get(media_id)
            if source is None:
                raise ValueError(
                    f"prototype biological selection lacks media metadata: {media_id}"
                )
            row = dict(source)
        elif scope_type == "visual_domain":
            media_id = str(selection["reference_media_id"])
            source = visual_by_id.get(media_id)
            if source is None:
                raise ValueError(
                    f"prototype visual selection lacks curated metadata: {media_id}"
                )
            row = _visual_media_candidate(source, selection)
        else:  # guarded by the selection validator
            raise ValueError(f"unsupported prototype scope type: {scope_type}")
        _validate_selection_candidate_identity(selection, row)
        decision = licence_policy.evaluate(
            media_licence=row["licence"],
            licence_uri=row["licence_uri"],
            attribution=row["attribution"],
        )
        if decision.status not in {"allowed", "research_only"}:
            raise ValueError(
                "prototype selection is not download-licence eligible: "
                f"{media_id} ({decision.reason or decision.status})"
            )
        if decision.status != selection["licence_policy_status"]:
            raise ValueError(
                f"prototype selection has stale licence status: {media_id}"
            )
        row["licence_policy_status"] = decision.status
        row["download_status"] = "pending"
        row["exclusion_reason"] = None
        rows.append(row)
    candidates = reference_media_candidates_frame(rows)
    validate_prototype_download_inputs(selections, candidates)
    return candidates


def validate_prototype_download_inputs(
    selections: pl.DataFrame,
    media_candidates: pl.DataFrame,
) -> None:
    validate_prototype_reference_selections(selections)
    validate_reference_media_candidates(media_candidates)
    if selections.height != media_candidates.height:
        raise ValueError(
            "prototype selections and download candidates differ in count"
        )
    candidates = {
        str(row["reference_media_id"]): row
        for row in media_candidates.iter_rows(named=True)
    }
    if set(selections["reference_media_id"]) != set(candidates):
        raise ValueError(
            "prototype selections and download candidates differ in identity"
        )
    for selection in selections.iter_rows(named=True):
        _validate_selection_candidate_identity(
            selection,
            candidates[str(selection["reference_media_id"])],
        )


def _combined_biological_candidates(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        raise ValueError("biological media candidate inputs must not be empty")
    for frame in frames:
        validate_reference_media_candidates(frame)
    combined = pl.concat(list(frames), how="vertical").sort(
        ["source", "provider_media_id", "reference_observation_id"]
    )
    validate_reference_media_candidates(combined)
    return combined


def _visual_candidates_by_media_id(
    manifest: Mapping[str, object],
) -> dict[str, Mapping[str, object]]:
    raw = manifest.get("candidates")
    if not isinstance(raw, list):
        raise ValueError("visual-domain manifest candidates must be an array")
    snapshot = str(manifest.get("source_snapshot_version") or "").strip()
    if not snapshot:
        raise ValueError("visual-domain manifest lacks source_snapshot_version")
    rows: dict[str, Mapping[str, object]] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("visual-domain candidate must be an object")
        _observation_id, media_id = make_prototype_visual_reference_ids(item)
        if media_id in rows:
            raise ValueError(f"duplicate visual-domain media identity: {media_id}")
        value = dict(item)
        value["source_snapshot_version"] = snapshot
        rows[media_id] = value
    return rows


def _visual_media_candidate(
    row: Mapping[str, object],
    selection: Mapping[str, object],
) -> dict[str, object]:
    observation_id, media_id = make_prototype_visual_reference_ids(row)
    checksum = str(row.get("provider_source_sha1") or "").strip() or None
    selected_at = selection["selected_at"]
    if not isinstance(selected_at, datetime):
        raise ValueError("prototype visual selection selected_at is invalid")
    return {
        "schema_version": REFERENCE_MEDIA_CANDIDATES_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "reference_observation_id": observation_id,
        "provider_media_id": row["provider_media_id"],
        "source": row["source"],
        "media_identifier": row["media_uri"],
        "media_type": "StillImage",
        "width": row["width"],
        "height": row["height"],
        "creator": row.get("creator"),
        "rights_holder": row.get("rights_holder"),
        "licence": row["licence"],
        "licence_uri": row.get("licence_uri"),
        "attribution": row["attribution"],
        "occurrence_licence": None,
        "original_provider": row["source"],
        "media_position": 0,
        "source_checksum": checksum,
        "source_checksum_algorithm": "sha1" if checksum else None,
        "download_status": "pending",
        "verification_status": "unreviewed",
        "exclusion_reason": None,
        "licence_policy_status": selection["licence_policy_status"],
        "retrieved_at": selected_at,
        "source_snapshot_version": row["source_snapshot_version"],
    }


def _validate_selection_candidate_identity(
    selection: Mapping[str, object],
    candidate: Mapping[str, object],
) -> None:
    media_id = str(selection["reference_media_id"])
    for field in (
        "reference_media_id",
        "reference_observation_id",
        "source",
        "provider_media_id",
        "media_identifier",
        "licence",
        "attribution",
    ):
        if selection[field] != candidate[field]:
            raise ValueError(
                f"prototype selection has stale {field}: {media_id}"
            )


__all__ = [
    "PROTOTYPE_DOWNLOAD_INPUT_POLICY_VERSION",
    "compile_prototype_download_candidates",
    "validate_prototype_download_inputs",
]
