"""Version and selectively invalidate an adaptively reviewed support bank."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.readiness import (
    REFERENCE_SUPPORT_MANIFEST_FILE,
    reference_support_manifest_fingerprint,
    reference_support_manifest_frame,
    reference_support_manifest_schema,
    validate_reference_support_manifest,
)
from biominer.references.targeted_review import (
    validate_targeted_reference_review_queue,
)
from biominer.references.targeted_review_decisions import (
    TargetedReferenceReviewResult,
    targeted_reference_review_decision_binding_schema,
    validate_targeted_reference_review_decisions,
    validate_targeted_reference_review_result,
)
from biominer.storage.parquet import write_parquet


REFERENCE_BANK_CHANGE_MANIFEST_FILE = "reference_bank_change_manifest.parquet"
REFERENCE_BANK_INVALIDATION_MANIFEST_FILE = (
    "reference_bank_invalidation_manifest.parquet"
)
REFERENCE_BANK_CHANGE_SCHEMA_VERSION = "reference-bank-change-manifest-v1.0.0"
REFERENCE_BANK_INVALIDATION_SCHEMA_VERSION = (
    "reference-bank-invalidation-manifest-v1.0.0"
)
REFERENCE_BANK_REVISION_SCHEMA_VERSION = "adaptive-reference-bank-revision-v1.0.0"
_VERSION_PATTERN = re.compile(r"(?P<prefix>.+-v)(?P<number>[0-9]+)\Z")
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")


REFERENCE_DOWNSTREAM_DEPENDENCY_SCHEMA = {
    "artifact_id": pl.String,
    "artifact_type": pl.String,
    "artifact_fingerprint": pl.String,
    "reference_bank_fingerprint": pl.String,
    "reference_media_ids": pl.List(pl.String),
}


def reference_bank_change_manifest_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "accepted_taxon_key": pl.String,
        "scientific_name": pl.String,
        "old_route": pl.String,
        "new_route": pl.String,
        "was_targeted": pl.Boolean,
        "change_type": pl.String,
        "old_support_eligible": pl.Boolean,
        "new_support_eligible": pl.Boolean,
        "old_provisional_support": pl.Boolean,
        "new_provisional_support": pl.Boolean,
        "old_identity_evidence_basis": pl.String,
        "new_identity_evidence_basis": pl.String,
        "effective_decision_ids": pl.List(pl.String),
        "effective_reviewer_ids": pl.List(pl.String),
        "changed_fields": pl.List(pl.String),
        "old_reference_bank_version": pl.String,
        "new_reference_bank_version": pl.String,
        "old_reference_bank_fingerprint": pl.String,
        "new_reference_bank_fingerprint": pl.String,
        "old_support_row_fingerprint": pl.String,
        "new_support_row_fingerprint": pl.String,
        "change_fingerprint": pl.String,
    }


def reference_bank_invalidation_manifest_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "artifact_id": pl.String,
        "artifact_type": pl.String,
        "artifact_fingerprint": pl.String,
        "invalidated": pl.Boolean,
        "affected_reference_media_ids": pl.List(pl.String),
        "change_types": pl.List(pl.String),
        "old_reference_bank_fingerprint": pl.String,
        "new_reference_bank_fingerprint": pl.String,
        "invalidation_fingerprint": pl.String,
    }


@dataclass(frozen=True, slots=True)
class AdaptiveSupportBankRevision:
    revised_support_manifest: pl.DataFrame
    change_manifest: pl.DataFrame
    invalidation_manifest: pl.DataFrame
    old_reference_bank_version: str
    new_reference_bank_version: str
    old_reference_bank_fingerprint: str
    new_reference_bank_fingerprint: str
    old_support_manifest_fingerprint: str
    new_support_manifest_fingerprint: str
    revision_fingerprint: str


def reference_downstream_dependencies_frame(
    rows: Sequence[Mapping[str, object]] | None = None,
) -> pl.DataFrame:
    frame = pl.DataFrame(
        rows or [],
        schema=REFERENCE_DOWNSTREAM_DEPENDENCY_SCHEMA,
        orient="row",
        strict=True,
    ).sort("artifact_id")
    if frame["artifact_id"].n_unique() != frame.height:
        raise ValueError("downstream dependencies repeat an artifact ID")
    for row in frame.iter_rows(named=True):
        _required_text(row["artifact_id"], field="artifact_id")
        _required_text(row["artifact_type"], field="artifact_type")
        _full_sha256(row["artifact_fingerprint"], field="artifact_fingerprint")
        _full_sha256(
            row["reference_bank_fingerprint"],
            field="reference_bank_fingerprint",
        )
        reference_ids = row["reference_media_ids"]
        if not reference_ids or reference_ids != sorted(set(reference_ids)):
            raise ValueError(
                "downstream dependency reference IDs must be nonempty and canonical"
            )
        for reference_id in reference_ids:
            _required_text(reference_id, field="reference_media_id")
    return frame


def revise_adaptive_support_bank(
    current_support_manifest: pl.DataFrame,
    review: TargetedReferenceReviewResult,
    downstream_dependencies: pl.DataFrame,
    *,
    new_reference_bank_version: str | None = None,
) -> AdaptiveSupportBankRevision:
    """Apply reviewed dispositions and preserve an exact old-to-new ledger."""

    validate_reference_support_manifest(current_support_manifest)
    if current_support_manifest.is_empty():
        raise ValueError("adaptive support bank revision requires existing rows")
    validate_targeted_reference_review_queue(review.targeted_queue)
    validate_targeted_reference_review_decisions(review.targeted_decisions)
    validate_targeted_reference_review_result(review)
    if (
        review.decision_bindings.schema
        != targeted_reference_review_decision_binding_schema()
    ):
        raise ValueError("targeted decision binding schema mismatch")
    dependencies = reference_downstream_dependencies_frame(
        downstream_dependencies.to_dicts()
    )

    old_versions = current_support_manifest["reference_bank_version"].unique()
    old_fingerprints = current_support_manifest[
        "reference_bank_fingerprint"
    ].unique()
    if len(old_versions) != 1 or len(old_fingerprints) != 1:
        raise ValueError("current support manifest spans multiple bank identities")
    old_version = str(old_versions.item())
    old_fingerprint = str(old_fingerprints.item())
    expected_version = _incremented_version(old_version)
    new_version = new_reference_bank_version or expected_version
    if new_version != expected_version:
        raise ValueError(
            f"new reference bank version must increment to {expected_version}"
        )
    if dependencies.filter(
        pl.col("reference_bank_fingerprint") != old_fingerprint
    ).height:
        raise ValueError("downstream dependency bank fingerprint is stale")

    support_by_id = {
        str(row["reference_media_id"]): row
        for row in current_support_manifest.iter_rows(named=True)
    }
    targeted_by_id = {
        str(row["reference_media_id"]): row
        for row in review.targeted_queue.iter_rows(named=True)
    }
    unknown_targeted = sorted(set(targeted_by_id) - set(support_by_id))
    if unknown_targeted:
        raise ValueError(
            "targeted queue references media outside the support bank: "
            + ", ".join(unknown_targeted)
        )
    for media_id, targeted in targeted_by_id.items():
        support = support_by_id[media_id]
        if (
            not support["provisional_support"]
            or support["identity_evidence_basis"] != "gbif_provider_asserted"
        ):
            raise ValueError(
                "adaptive revision targets only provisional GBIF support: "
                + media_id
            )
        if targeted["scientific_name"] != support["scientific_name"]:
            raise ValueError("targeted queue and support-bank species disagree")
    dependency_ids = {
        str(reference_id)
        for row in dependencies.iter_rows(named=True)
        for reference_id in row["reference_media_ids"]
    }
    unknown_dependencies = sorted(dependency_ids - set(support_by_id))
    if unknown_dependencies:
        raise ValueError(
            "downstream dependencies reference unknown media: "
            + ", ".join(unknown_dependencies)
        )

    outcomes_by_id = {
        str(row["reference_media_id"]): row
        for row in review.workflow.outcomes.iter_rows(named=True)
    }
    if set(outcomes_by_id) != set(targeted_by_id):
        raise ValueError("review outcomes must cover the complete targeted queue")
    binding_decision_ids = set(review.decision_bindings["review_decision_id"])
    effective_decision_ids = {
        str(decision_id)
        for outcome in outcomes_by_id.values()
        for decision_id in outcome["effective_decision_ids"]
    }
    if not effective_decision_ids.issubset(binding_decision_ids):
        raise ValueError("effective review decisions lack targeted provenance bindings")

    old_manifest_fingerprint = reference_support_manifest_fingerprint(
        current_support_manifest
    )
    new_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_BANK_REVISION_SCHEMA_VERSION,
            "old_reference_bank_version": old_version,
            "new_reference_bank_version": new_version,
            "old_reference_bank_fingerprint": old_fingerprint,
            "old_support_manifest_fingerprint": old_manifest_fingerprint,
            "targeting_fingerprints": sorted(
                str(value)
                for value in review.targeted_queue["targeting_fingerprint"]
            ),
            "decision_binding_fingerprints": sorted(
                str(value)
                for value in review.decision_bindings["binding_fingerprint"]
            ),
            "review_outcomes": [
                {
                    "reference_media_id": row["reference_media_id"],
                    "review_status": row["review_status"],
                    "effective_decision_ids": row["effective_decision_ids"],
                    "resolved_verification_status": row[
                        "resolved_verification_status"
                    ],
                    "target_identity_verified": row[
                        "target_identity_verified"
                    ],
                    "life_stage": row["life_stage"],
                    "visual_domain": row["visual_domain"],
                    "view": row["view"],
                }
                for row in sorted(
                    outcomes_by_id.values(),
                    key=lambda item: str(item["reference_media_id"]),
                )
            ],
        }
    )

    revised_rows: list[dict[str, object]] = []
    change_context: dict[str, dict[str, object]] = {}
    for old in current_support_manifest.iter_rows(named=True):
        media_id = str(old["reference_media_id"])
        row = dict(old)
        outcome = outcomes_by_id.get(media_id)
        if outcome is None:
            change_type = (
                "unchanged_provisional"
                if old["provisional_support"]
                else "unchanged"
            )
            decision_ids: list[str] = []
            reviewer_ids: list[str] = []
        elif (
            outcome["review_status"] == "completed"
            and outcome["resolved_verification_status"] == "verified"
            and outcome["target_identity_verified"] is True
        ):
            if not outcome["support_eligible"]:
                raise ValueError(
                    "verified targeted reference is not support eligible: "
                    + media_id
                )
            _promote_verified(row, outcome)
            change_type = "promoted_verified"
            decision_ids = list(outcome["effective_decision_ids"])
            reviewer_ids = list(outcome["effective_reviewer_ids"])
        elif (
            outcome["review_status"] == "completed"
            and outcome["resolved_verification_status"] == "excluded"
        ):
            _exclude_rejected(row, outcome, review.workflow.excluded)
            change_type = "excluded_after_review"
            decision_ids = list(outcome["effective_decision_ids"])
            reviewer_ids = list(outcome["effective_reviewer_ids"])
        else:
            _quarantine_pending(row, outcome)
            change_type = "flagged_review_pending"
            decision_ids = list(outcome["effective_decision_ids"])
            reviewer_ids = list(outcome["effective_reviewer_ids"])
        revised_rows.append(row)
        change_context[media_id] = {
            "change_type": change_type,
            "effective_decision_ids": sorted(decision_ids),
            "effective_reviewer_ids": sorted(reviewer_ids),
        }

    revised = reference_support_manifest_frame(
        revised_rows,
        reference_bank_version=new_version,
        reference_bank_fingerprint=new_fingerprint,
    )
    revised_by_id = {
        str(row["reference_media_id"]): row
        for row in revised.iter_rows(named=True)
    }
    change_rows = [
        _change_row(
            old,
            revised_by_id[str(old["reference_media_id"])],
            context=change_context[str(old["reference_media_id"])],
            old_version=old_version,
            new_version=new_version,
            old_fingerprint=old_fingerprint,
            new_fingerprint=new_fingerprint,
            was_targeted=str(old["reference_media_id"]) in targeted_by_id,
        )
        for old in current_support_manifest.iter_rows(named=True)
    ]
    changes = pl.DataFrame(
        change_rows,
        schema=reference_bank_change_manifest_schema(),
        orient="row",
        strict=True,
    ).sort("reference_media_id")
    invalidations = _build_invalidations(
        dependencies,
        changes,
        old_fingerprint=old_fingerprint,
        new_fingerprint=new_fingerprint,
    )
    new_manifest_fingerprint = reference_support_manifest_fingerprint(revised)
    revision_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_BANK_REVISION_SCHEMA_VERSION,
            "old_support_manifest_fingerprint": old_manifest_fingerprint,
            "new_support_manifest_fingerprint": new_manifest_fingerprint,
            "change_fingerprints": changes["change_fingerprint"].to_list(),
            "invalidation_fingerprints": invalidations[
                "invalidation_fingerprint"
            ].to_list(),
        }
    )
    result = AdaptiveSupportBankRevision(
        revised_support_manifest=revised,
        change_manifest=changes,
        invalidation_manifest=invalidations,
        old_reference_bank_version=old_version,
        new_reference_bank_version=new_version,
        old_reference_bank_fingerprint=old_fingerprint,
        new_reference_bank_fingerprint=new_fingerprint,
        old_support_manifest_fingerprint=old_manifest_fingerprint,
        new_support_manifest_fingerprint=new_manifest_fingerprint,
        revision_fingerprint=revision_fingerprint,
    )
    validate_adaptive_support_bank_revision(result)
    return result


def validate_adaptive_support_bank_revision(
    result: AdaptiveSupportBankRevision,
) -> None:
    validate_reference_support_manifest(result.revised_support_manifest)
    if result.change_manifest.schema != reference_bank_change_manifest_schema():
        raise ValueError("reference bank change manifest schema mismatch")
    if (
        result.invalidation_manifest.schema
        != reference_bank_invalidation_manifest_schema()
    ):
        raise ValueError("reference bank invalidation manifest schema mismatch")
    if result.change_manifest["reference_media_id"].n_unique() != (
        result.change_manifest.height
    ):
        raise ValueError("reference bank change manifest repeats reference media")
    for row in result.change_manifest.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_BANK_CHANGE_SCHEMA_VERSION:
            raise ValueError("unsupported reference bank change manifest version")
        payload = dict(row)
        fingerprint = payload.pop("change_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("reference bank change fingerprint mismatch")
    for row in result.invalidation_manifest.iter_rows(named=True):
        if row["schema_version"] != REFERENCE_BANK_INVALIDATION_SCHEMA_VERSION:
            raise ValueError("unsupported reference invalidation manifest version")
        if bool(row["invalidated"]) != bool(row["affected_reference_media_ids"]):
            raise ValueError("reference invalidation state lacks affected references")
        payload = dict(row)
        fingerprint = payload.pop("invalidation_fingerprint")
        if fingerprint != canonical_semantic_fingerprint(payload):
            raise ValueError("reference invalidation fingerprint mismatch")
    if reference_support_manifest_fingerprint(
        result.revised_support_manifest
    ) != result.new_support_manifest_fingerprint:
        raise ValueError("revised support manifest fingerprint mismatch")
    expected_revision_fingerprint = canonical_semantic_fingerprint(
        {
            "schema_version": REFERENCE_BANK_REVISION_SCHEMA_VERSION,
            "old_support_manifest_fingerprint": (
                result.old_support_manifest_fingerprint
            ),
            "new_support_manifest_fingerprint": (
                result.new_support_manifest_fingerprint
            ),
            "change_fingerprints": result.change_manifest[
                "change_fingerprint"
            ].to_list(),
            "invalidation_fingerprints": result.invalidation_manifest[
                "invalidation_fingerprint"
            ].to_list(),
        }
    )
    if result.revision_fingerprint != expected_revision_fingerprint:
        raise ValueError("adaptive support bank revision fingerprint mismatch")


def write_adaptive_support_bank_revision(
    result: AdaptiveSupportBankRevision,
    output_dir: str | Path,
) -> dict[str, Path]:
    validate_adaptive_support_bank_revision(result)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    return {
        "support_manifest": write_parquet(
            result.revised_support_manifest,
            root / REFERENCE_SUPPORT_MANIFEST_FILE,
        ),
        "change_manifest": write_parquet(
            result.change_manifest,
            root / REFERENCE_BANK_CHANGE_MANIFEST_FILE,
        ),
        "invalidation_manifest": write_parquet(
            result.invalidation_manifest,
            root / REFERENCE_BANK_INVALIDATION_MANIFEST_FILE,
        ),
    }


def _promote_verified(row: dict[str, object], outcome: Mapping[str, object]) -> None:
    row.update(
        {
            "identity_evidence_basis": "human_verified",
            "human_review_status": "completed",
            "human_verified_identity": True,
            "provisional_support": False,
            "statistical_audit_required": False,
            "admission_status": "admitted",
            "admission_reasons": _union(
                row["admission_reasons"],
                "targeted_human_review_verified",
            ),
            "reference_quality_flags": _union(
                row["reference_quality_flags"],
                "targeted_statistical_review_complete",
            ),
            "route_evidence_basis": "human_verified_review",
            "review_status": "completed",
            "verification_status": "verified",
            "target_identity_verified": True,
            "life_stage": outcome["life_stage"],
            "visual_domain": outcome["visual_domain"],
            "view": outcome["view"],
            "route": _route(outcome["life_stage"], outcome["visual_domain"]),
            "review_decision_ids": list(outcome["effective_decision_ids"]),
            "reviewer_ids": list(outcome["effective_reviewer_ids"]),
            "support_eligible": True,
            "exclusion_reasons": [],
        }
    )
    row["geographic_prototype_eligible"] = bool(
        row["geo_cluster_id"]
        and row["latitude"] is not None
        and row["longitude"] is not None
    )


def _exclude_rejected(
    row: dict[str, object],
    outcome: Mapping[str, object],
    excluded: pl.DataFrame,
) -> None:
    matched = excluded.filter(
        pl.col("review_request_id") == outcome["review_request_id"]
    )
    reasons = (
        matched["resolved_exclusion_reasons"].item()
        if matched.height == 1
        else []
    )
    corrected_route = _optional_route(
        outcome["life_stage"],
        outcome["visual_domain"],
    )
    row.update(
        {
            "identity_evidence_basis": "none",
            "human_review_status": "rejected",
            "human_verified_identity": False,
            "provisional_support": False,
            "statistical_audit_required": False,
            "admission_status": "excluded",
            "admission_reasons": _union(
                row["admission_reasons"],
                "targeted_human_review_excluded",
            ),
            "reference_quality_flags": _union(
                row["reference_quality_flags"],
                "targeted_human_review_excluded",
            ),
            "route_evidence_basis": "none",
            "geographic_prototype_eligible": False,
            "review_status": "completed",
            "verification_status": "excluded",
            "target_identity_verified": False,
            "life_stage": (
                outcome["life_stage"]
                if corrected_route is not None
                else row["life_stage"]
            ),
            "visual_domain": (
                outcome["visual_domain"]
                if corrected_route is not None
                else row["visual_domain"]
            ),
            "view": outcome["view"] if corrected_route is not None else row["view"],
            "route": corrected_route or row["route"],
            "review_decision_ids": list(outcome["effective_decision_ids"]),
            "reviewer_ids": list(outcome["effective_reviewer_ids"]),
            "support_split": None,
            "support_eligible": False,
            "exclusion_reasons": sorted(
                {*reasons, "targeted_human_review_excluded"}
            ),
            "split_assignment_fingerprint": None,
        }
    )


def _quarantine_pending(
    row: dict[str, object],
    outcome: Mapping[str, object],
) -> None:
    status = str(outcome["review_status"])
    human_status = (
        "conflict"
        if status == "conflict"
        else "pending"
        if status == "pending"
        else "in_review"
    )
    row.update(
        {
            "human_review_status": human_status,
            "admission_status": "review_required",
            "admission_reasons": _union(
                row["admission_reasons"],
                "targeted_human_review_pending",
            ),
            "reference_quality_flags": _union(
                row["reference_quality_flags"],
                "targeted_human_review_pending",
            ),
            "geographic_prototype_eligible": False,
            "review_status": status,
            "review_decision_ids": list(outcome["effective_decision_ids"]),
            "reviewer_ids": list(outcome["effective_reviewer_ids"]),
            "support_split": None,
            "support_eligible": False,
            "exclusion_reasons": _union(
                row["exclusion_reasons"],
                "targeted_human_review_pending",
            ),
            "split_assignment_fingerprint": None,
        }
    )


def _change_row(
    old: Mapping[str, object],
    new: Mapping[str, object],
    *,
    context: Mapping[str, object],
    old_version: str,
    new_version: str,
    old_fingerprint: str,
    new_fingerprint: str,
    was_targeted: bool,
) -> dict[str, object]:
    ignored = {
        "reference_bank_version",
        "reference_bank_fingerprint",
        "support_row_fingerprint",
    }
    changed_fields = sorted(
        field
        for field in reference_support_manifest_schema()
        if field not in ignored and old[field] != new[field]
    )
    row = {
        "schema_version": REFERENCE_BANK_CHANGE_SCHEMA_VERSION,
        "reference_media_id": old["reference_media_id"],
        "accepted_taxon_key": old["accepted_taxon_key"],
        "scientific_name": old["scientific_name"],
        "old_route": old["route"],
        "new_route": new["route"],
        "was_targeted": was_targeted,
        "change_type": context["change_type"],
        "old_support_eligible": old["support_eligible"],
        "new_support_eligible": new["support_eligible"],
        "old_provisional_support": old["provisional_support"],
        "new_provisional_support": new["provisional_support"],
        "old_identity_evidence_basis": old["identity_evidence_basis"],
        "new_identity_evidence_basis": new["identity_evidence_basis"],
        "effective_decision_ids": context["effective_decision_ids"],
        "effective_reviewer_ids": context["effective_reviewer_ids"],
        "changed_fields": changed_fields,
        "old_reference_bank_version": old_version,
        "new_reference_bank_version": new_version,
        "old_reference_bank_fingerprint": old_fingerprint,
        "new_reference_bank_fingerprint": new_fingerprint,
        "old_support_row_fingerprint": old["support_row_fingerprint"],
        "new_support_row_fingerprint": new["support_row_fingerprint"],
        "change_fingerprint": "",
    }
    payload = dict(row)
    payload.pop("change_fingerprint")
    row["change_fingerprint"] = canonical_semantic_fingerprint(payload)
    return row


def _build_invalidations(
    dependencies: pl.DataFrame,
    changes: pl.DataFrame,
    *,
    old_fingerprint: str,
    new_fingerprint: str,
) -> pl.DataFrame:
    changed = {
        str(row["reference_media_id"]): str(row["change_type"])
        for row in changes.iter_rows(named=True)
        if not str(row["change_type"]).startswith("unchanged")
    }
    rows: list[dict[str, object]] = []
    for dependency in dependencies.iter_rows(named=True):
        affected = sorted(set(dependency["reference_media_ids"]) & set(changed))
        row = {
            "schema_version": REFERENCE_BANK_INVALIDATION_SCHEMA_VERSION,
            "artifact_id": dependency["artifact_id"],
            "artifact_type": dependency["artifact_type"],
            "artifact_fingerprint": dependency["artifact_fingerprint"],
            "invalidated": bool(affected),
            "affected_reference_media_ids": affected,
            "change_types": sorted({changed[media_id] for media_id in affected}),
            "old_reference_bank_fingerprint": old_fingerprint,
            "new_reference_bank_fingerprint": new_fingerprint,
            "invalidation_fingerprint": "",
        }
        payload = dict(row)
        payload.pop("invalidation_fingerprint")
        row["invalidation_fingerprint"] = canonical_semantic_fingerprint(payload)
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema=reference_bank_invalidation_manifest_schema(),
        orient="row",
        strict=True,
    ).sort("artifact_id")


def _route(life_stage: object, visual_domain: object) -> str:
    stage = str(life_stage)
    domain = str(visual_domain)
    if domain == "pinned_specimen":
        return "pinned_specimen"
    if domain != "live_field":
        raise ValueError("reviewed visual domain is not support-bank eligible")
    try:
        return {
            "adult": "adult_field",
            "larva": "larval",
            "pupa": "pupal",
            "egg": "egg",
        }[stage]
    except KeyError as exc:
        raise ValueError("reviewed life stage has no support route") from exc


def _optional_route(life_stage: object, visual_domain: object) -> str | None:
    try:
        return _route(life_stage, visual_domain)
    except ValueError:
        return None


def _incremented_version(version: str) -> str:
    match = _VERSION_PATTERN.fullmatch(version)
    if match is None:
        raise ValueError("reference bank version must end in -v<integer>")
    return f"{match.group('prefix')}{int(match.group('number')) + 1}"


def _union(values: object, value: str) -> list[str]:
    return sorted({*(str(item) for item in values), value})  # type: ignore[union-attr]


def _required_text(value: object, *, field: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{field} must be nonblank")
    return text


def _full_sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full sha256 fingerprint")
    return text


__all__ = [
    "REFERENCE_BANK_CHANGE_MANIFEST_FILE",
    "REFERENCE_BANK_CHANGE_SCHEMA_VERSION",
    "REFERENCE_BANK_INVALIDATION_MANIFEST_FILE",
    "REFERENCE_BANK_INVALIDATION_SCHEMA_VERSION",
    "REFERENCE_BANK_REVISION_SCHEMA_VERSION",
    "REFERENCE_DOWNSTREAM_DEPENDENCY_SCHEMA",
    "AdaptiveSupportBankRevision",
    "reference_bank_change_manifest_schema",
    "reference_bank_invalidation_manifest_schema",
    "reference_downstream_dependencies_frame",
    "revise_adaptive_support_bank",
    "validate_adaptive_support_bank_revision",
    "write_adaptive_support_bank_revision",
]
