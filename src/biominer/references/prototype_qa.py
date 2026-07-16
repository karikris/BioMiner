from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from PIL import Image
import polars as pl

from biominer.references.prototype_acquisition import (
    validate_prototype_reference_selections,
)
from biominer.references.schemas import validate_reference_media_objects
from biominer.storage.cloud import CloudStorage
from biominer.storage.paths import build_report_uri, safe_path_component
from biominer.storage.uri import join_uri


PROTOTYPE_QA_VERSION = "prototype-support-qa-v1.0.0"
PROTOTYPE_QA_ROWS_SCHEMA_VERSION = "prototype-support-qa-rows-v1.0.0"
PROTOTYPE_QA_FILE = "prototype_support_qa.parquet"
PROTOTYPE_QA_REPORT_FILE = "prototype_support_qa_report.json"
PROTOTYPE_QA_SUMMARY_FILE = "prototype_support_qa_summary.md"


@dataclass(frozen=True, slots=True)
class PrototypeQAConfig:
    policy_version: str = "prototype-support-qa-policy-v1"
    analysis_size: int = 256
    exclude_min_dimension: int = 128
    review_min_dimension: int = 512
    exclude_entropy_bits: float = 1.5
    review_entropy_bits: float = 3.0
    review_gradient_mean: float = 0.008
    exclude_clipped_fraction: float = 0.98
    review_clipped_fraction: float = 0.60

    def __post_init__(self) -> None:
        if not self.policy_version.strip():
            raise ValueError("prototype QA policy_version must be nonblank")
        if self.analysis_size < 32:
            raise ValueError("analysis_size must be at least 32")
        if not 0 < self.exclude_min_dimension <= self.review_min_dimension:
            raise ValueError("prototype QA dimension thresholds are invalid")
        if not 0 <= self.exclude_entropy_bits <= self.review_entropy_bits <= 8:
            raise ValueError("prototype QA entropy thresholds are invalid")
        if not 0 <= self.review_gradient_mean <= 1:
            raise ValueError("review_gradient_mean must be in [0, 1]")
        if not 0 <= self.review_clipped_fraction <= self.exclude_clipped_fraction <= 1:
            raise ValueError("prototype QA clipping thresholds are invalid")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PrototypeQAResult:
    qualifications: pl.DataFrame
    report: dict[str, Any]
    markdown: str


def prototype_qa_schema() -> dict[str, pl.DataType]:
    return {
        "schema_version": pl.String,
        "reference_media_id": pl.String,
        "reference_observation_id": pl.String,
        "source": pl.String,
        "candidate_scope_type": pl.String,
        "candidate_scope_id": pl.String,
        "planned_route": pl.String,
        "planned_life_stage": pl.String,
        "planned_visual_domain": pl.String,
        "source_object_uri": pl.String,
        "qa_disposition": pl.String,
        "qa_reason": pl.String,
        "provisional_visual_domain": pl.String,
        "provisional_life_stage": pl.String,
        "adult_larva_check": pl.String,
        "pinned_field_check": pl.String,
        "artifact_biological_check": pl.String,
        "subject_presence_check": pl.String,
        "subject_presence_evidence": pl.String,
        "subject_size_check": pl.String,
        "subject_area_ratio": pl.Float64,
        "image_quality_check": pl.String,
        "image_quality_flags": pl.List(pl.String),
        "decoded_width": pl.UInt32,
        "decoded_height": pl.UInt32,
        "luminance_mean": pl.Float64,
        "luminance_std": pl.Float64,
        "entropy_bits": pl.Float64,
        "gradient_mean": pl.Float64,
        "dark_clipped_fraction": pl.Float64,
        "bright_clipped_fraction": pl.Float64,
        "metadata_disagreement_check": pl.String,
        "metadata_disagreement_flags": pl.List(pl.String),
        "licence_completeness_check": pl.String,
        "attribution_completeness_check": pl.String,
        "detector_evidence_status": pl.String,
        "human_taxonomic_verification": pl.Boolean,
        "operational_failure_retryable": pl.Boolean,
        "qa_policy_version": pl.String,
        "qa_policy_fingerprint": pl.String,
        "qa_fingerprint": pl.String,
    }


def qualify_prototype_support_bank(
    *,
    selections: pl.DataFrame,
    media_objects: pl.DataFrame,
    identity_groups: pl.DataFrame,
    biological_observations: Sequence[pl.DataFrame],
    visual_domain_manifest: Mapping[str, object],
    config: PrototypeQAConfig | None = None,
    detector_evidence: pl.DataFrame | None = None,
    generated_at: str | datetime | None = None,
) -> PrototypeQAResult:
    validate_prototype_reference_selections(selections)
    validate_reference_media_objects(media_objects)
    active = config or PrototypeQAConfig()
    timestamp = _utc_datetime(generated_at or datetime.now(UTC))
    _validate_identity_inputs(selections, media_objects, identity_groups)
    observations = _observation_lookup(biological_observations)
    visual = _visual_lookup(visual_domain_manifest)
    detections = _detector_lookup(detector_evidence)
    objects = {
        str(row["reference_media_id"]): row
        for row in media_objects.iter_rows(named=True)
    }
    identities = {
        str(row["reference_media_id"]): row
        for row in identity_groups.iter_rows(named=True)
    }

    rows = []
    for selection in selections.sort("reference_media_id").iter_rows(named=True):
        media_id = str(selection["reference_media_id"])
        row = _qualify_one(
            selection=selection,
            media_object=objects[media_id],
            identity=identities[media_id],
            observation=observations.get(str(selection["reference_observation_id"])),
            visual=visual.get(str(selection["provider_media_id"])),
            detector=detections.get(media_id),
            config=active,
        )
        row["qa_fingerprint"] = _row_fingerprint(row)
        rows.append(row)
    qualifications = pl.DataFrame(rows, schema=prototype_qa_schema()).sort(
        "reference_media_id"
    )
    report = _report(qualifications, active, timestamp)
    result = PrototypeQAResult(qualifications, report, _markdown(report))
    validate_prototype_qa_result(result)
    return result


def validate_prototype_qa_result(result: PrototypeQAResult) -> None:
    if result.qualifications.schema != prototype_qa_schema():
        raise ValueError("prototype QA schema is invalid")
    if (
        result.qualifications["reference_media_id"].n_unique()
        != result.qualifications.height
    ):
        raise ValueError("prototype QA rows contain duplicate media identities")
    if result.qualifications.filter(pl.col("human_taxonomic_verification")).height:
        raise ValueError("automated prototype QA cannot claim human verification")
    if result.report.get("schema_version") != PROTOTYPE_QA_VERSION:
        raise ValueError("prototype QA report schema is invalid")
    if (
        result.report.get("semantics", {}).get("human_taxonomic_verification")
        is not False
    ):
        raise ValueError("prototype QA report overstates verification")


def publish_prototype_qa_result(
    result: PrototypeQAResult,
    *,
    storage: CloudStorage,
    output_prefix: str,
    run_id: str | None = None,
    settings_fingerprint: str | None = None,
) -> dict[str, str]:
    validate_prototype_qa_result(result)
    prefix = str(output_prefix).strip().rstrip("/")
    if not prefix:
        raise ValueError("prototype QA output_prefix must be nonblank")
    effective_run_id = str(run_id or "").strip() or (
        "prototype-qa-"
        + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ-")
        + uuid4().hex[:12]
    )
    component = safe_path_component(effective_run_id)
    artifact_prefix = join_uri(prefix, "qualification", f"run_id={component}")
    uris = {
        "qualifications": join_uri(artifact_prefix, PROTOTYPE_QA_FILE),
        "summary": build_report_uri(
            prefix,
            run_id=component,
            report_name=PROTOTYPE_QA_SUMMARY_FILE.removesuffix(".md"),
            suffix="md",
        ),
        "report": build_report_uri(
            prefix,
            run_id=component,
            report_name=PROTOTYPE_QA_REPORT_FILE.removesuffix(".json"),
        ),
    }
    if any(storage.exists(uri) for uri in uris.values()):
        raise FileExistsError("prototype QA run already exists")
    storage.write_parquet_shard(
        uris["qualifications"], result.qualifications, overwrite=False
    )
    storage.write_text(uris["summary"], result.markdown)
    report = json.loads(json.dumps(result.report))
    report["run_id"] = effective_run_id
    report["settings_fingerprint"] = settings_fingerprint
    report["artifacts"] = {
        name: {
            "uri": uri,
            "byte_count": storage.file_size(uri),
            "sha256": storage.file_sha256(uri),
        }
        for name, uri in sorted(uris.items())
        if name != "report"
    }
    storage.write_json(uris["report"], report)
    return uris


def _qualify_one(
    *, selection, media_object, identity, observation, visual, detector, config
):  # noqa: ANN001
    media_id = str(selection["reference_media_id"])
    valid = (
        media_object.get("decode_status") == "valid"
        and identity.get("support_disposition") == "eligible"
    )
    base = {
        "schema_version": PROTOTYPE_QA_ROWS_SCHEMA_VERSION,
        "reference_media_id": media_id,
        "reference_observation_id": selection["reference_observation_id"],
        "source": selection["source"],
        "candidate_scope_type": selection["candidate_scope_type"],
        "candidate_scope_id": selection["candidate_scope_id"],
        "planned_route": selection["route"],
        "planned_life_stage": selection["life_stage"],
        "planned_visual_domain": selection["visual_domain"],
        "source_object_uri": media_object.get("source_object_uri"),
        "human_taxonomic_verification": False,
        "qa_policy_version": config.policy_version,
        "qa_policy_fingerprint": config.fingerprint,
    }
    if not valid:
        return {
            **base,
            "qa_disposition": "operational_failure",
            "qa_reason": str(
                media_object.get("quarantine_reason")
                or identity.get("support_disposition")
                or "unavailable_media"
            ),
            "provisional_visual_domain": None,
            "provisional_life_stage": None,
            **{
                name: "not_evaluated"
                for name in (
                    "adult_larva_check",
                    "pinned_field_check",
                    "artifact_biological_check",
                    "subject_presence_check",
                    "subject_size_check",
                    "image_quality_check",
                    "metadata_disagreement_check",
                    "licence_completeness_check",
                    "attribution_completeness_check",
                )
            },
            "subject_presence_evidence": "unavailable_media",
            "subject_area_ratio": None,
            "image_quality_flags": [],
            "decoded_width": media_object.get("decoded_width"),
            "decoded_height": media_object.get("decoded_height"),
            "luminance_mean": None,
            "luminance_std": None,
            "entropy_bits": None,
            "gradient_mean": None,
            "dark_clipped_fraction": None,
            "bright_clipped_fraction": None,
            "metadata_disagreement_flags": [],
            "detector_evidence_status": "not_evaluated",
            "operational_failure_retryable": True,
        }

    metrics = _image_metrics(Path(str(media_object["source_object_uri"])), config)
    quality_check, quality_flags = _quality_check(metrics, config)
    metadata_flags = _metadata_flags(selection, observation, visual)
    metadata_check = "review" if metadata_flags else "pass"
    route, life_stage = _provisional_domain(selection, visual, detector)
    detector_status = "available" if detector is not None else "not_instrumented"
    subject_presence, presence_evidence, subject_area = _subject_evidence(
        visual, detector
    )
    subject_size = (
        "pass"
        if subject_area is not None and subject_area >= 0.01
        else ("review" if subject_area is None else "exclude")
    )
    route_check = (
        "review" if "route_visual_domain_conflict" in metadata_flags else "pass"
    )
    adult_larva = "review" if "life_stage_route_conflict" in metadata_flags else "pass"
    artifact_biological = (
        "review" if "scope_visual_domain_conflict" in metadata_flags else "pass"
    )
    licence_check = (
        "pass"
        if _complete(selection.get("licence"))
        and _complete(selection.get("licence_uri"))
        else "exclude"
    )
    attribution_check = (
        "pass"
        if selection.get("attribution_complete") is True
        and _complete(selection.get("attribution"))
        else "exclude"
    )

    checks = {
        "image_quality": quality_check,
        "metadata": metadata_check,
        "adult_larva": adult_larva,
        "pinned_field": route_check,
        "artifact_biological": artifact_biological,
        "subject_presence": subject_presence,
        "subject_size": subject_size,
        "licence": licence_check,
        "attribution": attribution_check,
    }
    if "exclude" in checks.values():
        disposition = "excluded"
        reason = next(name for name, status in checks.items() if status == "exclude")
    elif "review" in checks.values():
        disposition = "needs_review"
        reason = ",".join(name for name, status in checks.items() if status == "review")
    else:
        disposition = "provisionally_qualified"
        reason = "all_automated_checks_passed"
    return {
        **base,
        "qa_disposition": disposition,
        "qa_reason": reason,
        "provisional_visual_domain": route,
        "provisional_life_stage": life_stage,
        "adult_larva_check": adult_larva,
        "pinned_field_check": route_check,
        "artifact_biological_check": artifact_biological,
        "subject_presence_check": subject_presence,
        "subject_presence_evidence": presence_evidence,
        "subject_size_check": subject_size,
        "subject_area_ratio": subject_area,
        "image_quality_check": quality_check,
        "image_quality_flags": quality_flags,
        "decoded_width": metrics["width"],
        "decoded_height": metrics["height"],
        "luminance_mean": metrics["mean"],
        "luminance_std": metrics["std"],
        "entropy_bits": metrics["entropy"],
        "gradient_mean": metrics["gradient"],
        "dark_clipped_fraction": metrics["dark"],
        "bright_clipped_fraction": metrics["bright"],
        "metadata_disagreement_check": metadata_check,
        "metadata_disagreement_flags": metadata_flags,
        "licence_completeness_check": licence_check,
        "attribution_completeness_check": attribution_check,
        "detector_evidence_status": detector_status,
        "operational_failure_retryable": False,
    }


def _image_metrics(path: Path, config: PrototypeQAConfig) -> dict[str, float | int]:
    with Image.open(path) as image:
        width, height = image.size
        gray = image.convert("L")
        gray.thumbnail((config.analysis_size, config.analysis_size))
        analysis_width, analysis_height = gray.size
        pixels = list(gray.get_flattened_data())
        histogram = gray.histogram()
    pixel_count = len(pixels)
    mean = sum(pixels) / pixel_count
    variance = sum((value - mean) ** 2 for value in pixels) / pixel_count
    entropy = -sum(
        (count / pixel_count) * math.log2(count / pixel_count)
        for count in histogram
        if count
    )
    dx_count = analysis_height * max(analysis_width - 1, 0)
    dy_count = max(analysis_height - 1, 0) * analysis_width
    dx = (
        sum(
            abs(pixels[offset + column + 1] - pixels[offset + column])
            for offset in range(0, pixel_count, analysis_width)
            for column in range(analysis_width - 1)
        )
        / dx_count
        / 255.0
        if dx_count
        else 0.0
    )
    dy = (
        sum(
            abs(pixels[index + analysis_width] - pixels[index])
            for index in range(pixel_count - analysis_width)
        )
        / dy_count
        / 255.0
        if dy_count
        else 0.0
    )
    return {
        "width": width,
        "height": height,
        "mean": mean / 255.0,
        "std": math.sqrt(variance) / 255.0,
        "entropy": entropy,
        "gradient": float((dx + dy) / 2),
        "dark": sum(value <= 8 for value in pixels) / pixel_count,
        "bright": sum(value >= 247 for value in pixels) / pixel_count,
    }


def _quality_check(metrics, config):  # noqa: ANN001
    flags = []
    minimum = min(int(metrics["width"]), int(metrics["height"]))
    clipped = max(float(metrics["dark"]), float(metrics["bright"]))
    if minimum < config.exclude_min_dimension:
        flags.append("very_low_resolution")
    elif minimum < config.review_min_dimension:
        flags.append("low_resolution")
    if float(metrics["entropy"]) < config.exclude_entropy_bits:
        flags.append("very_low_entropy")
    elif float(metrics["entropy"]) < config.review_entropy_bits:
        flags.append("low_entropy")
    if clipped >= config.exclude_clipped_fraction:
        flags.append("severe_exposure_clipping")
    elif clipped >= config.review_clipped_fraction:
        flags.append("exposure_clipping")
    if float(metrics["gradient"]) < config.review_gradient_mean:
        flags.append("low_gradient_detail")
    if any(
        flag.startswith("very_") or flag == "severe_exposure_clipping" for flag in flags
    ):
        return "exclude", flags
    return ("review" if flags else "pass"), flags


def _metadata_flags(selection, observation, visual):  # noqa: ANN001
    flags = []
    route, stage, domain, scope = (
        str(selection.get(k) or "")
        for k in ("route", "life_stage", "visual_domain", "candidate_scope_type")
    )
    if (route == "larval") != (stage in {"larva", "caterpillar"}):
        flags.append("life_stage_route_conflict")
    if (route == "pinned_specimen") != (domain == "pinned_specimen"):
        flags.append("route_visual_domain_conflict")
    if scope == "visual_domain" and domain == "live_field":
        flags.append("scope_visual_domain_conflict")
    if observation and observation.get("identification_disagreement") is True:
        flags.append("provider_identification_disagreement")
    if observation and observation.get("uncertain_taxon_match") is True:
        flags.append("uncertain_taxon_match")
    if (
        observation
        and observation.get("preserved_specimen") is True
        and route != "pinned_specimen"
    ):
        flags.append("preserved_specimen_route_conflict")
    if visual and visual.get("human_verified") is True:
        flags.append("unexpected_human_verified_marker")
    return sorted(flags)


def _provisional_domain(selection, visual, detector):  # noqa: ANN001
    if detector and detector.get("provisional_visual_domain"):
        return str(detector["provisional_visual_domain"]), detector.get(
            "provisional_life_stage"
        )
    if visual:
        return str(
            visual.get("visual_domain_category") or selection.get("visual_domain")
        ), selection.get("life_stage")
    return str(selection.get("visual_domain") or "unknown"), selection.get("life_stage")


def _subject_evidence(visual, detector):  # noqa: ANN001
    if detector is not None:
        present = detector.get("subject_present")
        area = detector.get("subject_area_ratio")
        return (
            (
                "pass"
                if present is True
                else "exclude"
                if present is False
                else "review"
            ),
            "detector",
            (float(area) if area is not None else None),
        )
    if visual is not None and isinstance(visual.get("contains_butterfly_visual"), bool):
        return (
            ("pass" if visual["contains_butterfly_visual"] else "exclude"),
            "curated_source_metadata",
            None,
        )
    return "review", "not_instrumented", None


def _validate_identity_inputs(selections, media_objects, identity_groups):  # noqa: ANN001
    expected = set(selections["reference_media_id"].to_list())
    for name, frame in (
        ("media objects", media_objects),
        ("identity groups", identity_groups),
    ):
        if (
            "reference_media_id" not in frame.columns
            or set(frame["reference_media_id"].to_list()) != expected
        ):
            raise ValueError(f"prototype QA {name} differ from selections")


def _observation_lookup(frames):  # noqa: ANN001
    rows = {}
    for frame in frames:
        for row in frame.iter_rows(named=True):
            key = str(row["reference_observation_id"])
            if key in rows:
                raise ValueError(f"duplicate biological observation identity: {key}")
            rows[key] = row
    return rows


def _visual_lookup(manifest):  # noqa: ANN001
    result = {}
    for raw in manifest.get("candidates", []):
        if not isinstance(raw, Mapping):
            continue
        provider_id = str(raw.get("provider_media_id") or "")
        result[provider_id] = raw
    return result


def _detector_lookup(frame):  # noqa: ANN001
    if frame is None:
        return {}
    required = {
        "reference_media_id",
        "subject_present",
        "subject_area_ratio",
        "provisional_visual_domain",
        "provisional_life_stage",
    }
    if not required.issubset(frame.columns):
        raise ValueError("detector evidence schema is incomplete")
    return {str(row["reference_media_id"]): row for row in frame.iter_rows(named=True)}


def _report(
    frame: pl.DataFrame, config: PrototypeQAConfig, generated_at: datetime
) -> dict[str, Any]:
    def distribution(column):
        return dict(
            sorted(Counter(str(value) for value in frame[column].to_list()).items())
        )

    return {
        "schema_version": PROTOTYPE_QA_VERSION,
        "status": "complete",
        "prototype_only": True,
        "generated_at": generated_at.isoformat(),
        "policy_version": config.policy_version,
        "policy_fingerprint": config.fingerprint,
        "counts": {"selected": frame.height, **distribution("qa_disposition")},
        "distributions": {
            name: distribution(name)
            for name in (
                "provisional_visual_domain",
                "provisional_life_stage",
                "image_quality_check",
                "subject_presence_check",
                "subject_size_check",
                "licence_completeness_check",
                "attribution_completeness_check",
                "detector_evidence_status",
            )
        },
        "semantics": {
            "human_taxonomic_verification": False,
            "model_output_is_taxonomic_validation": False,
            "operational_failures_are_biological_negatives": False,
            "unmeasured_visual_evidence_is_guessed": False,
        },
    }


def _markdown(report):  # noqa: ANN001
    counts = report["counts"]
    return "\n".join(
        [
            "# Prototype support QA",
            "",
            "Automated prototype screening only; this is not human taxonomic verification.",
            "",
            f"- Selected: {counts['selected']}",
            f"- Provisionally qualified: {counts.get('provisionally_qualified', 0)}",
            f"- Needs review: {counts.get('needs_review', 0)}",
            f"- Excluded: {counts.get('excluded', 0)}",
            f"- Retryable operational failures: {counts.get('operational_failure', 0)}",
            "",
        ]
    )


def _complete(value):  # noqa: ANN001
    return bool(str(value or "").strip())


def _row_fingerprint(row):  # noqa: ANN001
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _utc_datetime(value: str | datetime) -> datetime:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


__all__ = [
    "PROTOTYPE_QA_FILE",
    "PrototypeQAConfig",
    "PrototypeQAResult",
    "prototype_qa_schema",
    "publish_prototype_qa_result",
    "qualify_prototype_support_bank",
    "validate_prototype_qa_result",
]
