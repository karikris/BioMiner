"""Dashboard-ready, licence-aware evidence for the Build Week prototype."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import polars as pl

from biominer.storage.parquet import write_parquet


PROTOTYPE_EVIDENCE_CONFIG_VERSION = "prototype-evidence-report-config-v1.0.0"
PROTOTYPE_EVIDENCE_REPORT_VERSION = "prototype-evidence-report-v1.0.0"
PROTOTYPE_EVIDENCE_DASHBOARD_FILE = "prototype_evidence_dashboard.parquet"
PROTOTYPE_EVIDENCE_COMPETITORS_FILE = "prototype_evidence_regional_competitors.parquet"
PROTOTYPE_EVIDENCE_REFERENCES_FILE = "prototype_evidence_nearest_references.parquet"
PROTOTYPE_EVIDENCE_REPORT_FILE = "prototype_evidence_report.json"
PROTOTYPE_EVIDENCE_SUMMARY_FILE = "prototype_evidence_report.md"

_SECRET_PATTERN = re.compile(
    r"(?i)(?:api[_-]?key|authorization|bearer|password|secret|token)\s*[:=]"
)
_FORBIDDEN_PUBLIC_COLUMNS = {
    "decoded_image_sha256",
    "image_url",
    "source_image_sha256",
    "source_object_uri",
    "text_prompt",
}
_NEAREST_REFERENCE_DTYPE = pl.List(
    pl.Struct(
        {
            "accepted_taxon_key": pl.String,
            "reference_group": pl.String,
            "reference_media_id": pl.String,
            "scientific_name": pl.String,
            "similarity": pl.Float64,
        }
    )
)


@dataclass(frozen=True, slots=True)
class PrototypeEvidenceConfig:
    target_accepted_taxon_key: str
    target_scientific_name: str
    classifications: Path
    classifications_sha256: str
    candidate_scores: Path
    candidate_scores_sha256: str
    support_manifest: Path
    support_manifest_sha256: str
    output_dir: Path
    verification_limitations: tuple[str, ...]
    storage_backend: str = "local"
    s3_permitted: bool = False
    target_reference_limit: int = 3
    competitor_reference_limit: int = 3
    regional_competitor_preview_limit: int = 5
    report_example_limit: int = 5

    def __post_init__(self) -> None:
        for field_name in (
            "classifications",
            "candidate_scores",
            "support_manifest",
            "output_dir",
        ):
            raw = getattr(self, field_name)
            if "://" in str(raw):
                raise ValueError(f"{field_name} must be a local path")
            object.__setattr__(self, field_name, Path(raw).expanduser())
        for field_name in (
            "classifications_sha256",
            "candidate_scores_sha256",
            "support_manifest_sha256",
        ):
            _require_sha256(getattr(self, field_name), field=field_name)
        if self.storage_backend != "local" or self.s3_permitted:
            raise ValueError("prototype evidence reporting must remain local-only")
        for field_name in (
            "target_reference_limit",
            "competitor_reference_limit",
            "regional_competitor_preview_limit",
            "report_example_limit",
        ):
            if int(getattr(self, field_name)) <= 0:
                raise ValueError(f"{field_name} must be positive")
        limitations = tuple(
            _required_text(item, field="verification_limitations[]")
            for item in self.verification_limitations
        )
        if not limitations or len(limitations) != len(set(limitations)):
            raise ValueError("verification_limitations must be non-empty and unique")
        object.__setattr__(self, "verification_limitations", limitations)

    @classmethod
    def read_json(cls, path: str | Path) -> PrototypeEvidenceConfig:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise TypeError("prototype evidence config must be a JSON object")
        values = dict(payload)
        if values.pop("schema_version", None) != PROTOTYPE_EVIDENCE_CONFIG_VERSION:
            raise ValueError("unsupported prototype evidence config schema")
        unknown = set(values) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown prototype evidence fields: {sorted(unknown)}")
        if isinstance(values.get("verification_limitations"), list):
            values["verification_limitations"] = tuple(
                values["verification_limitations"]
            )
        return cls(**values)  # type: ignore[arg-type]

    def verify_inputs(self) -> None:
        for path, expected in (
            (self.classifications, self.classifications_sha256),
            (self.candidate_scores, self.candidate_scores_sha256),
            (self.support_manifest, self.support_manifest_sha256),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            actual = _file_sha256(path)
            if actual != expected:
                raise ValueError(
                    f"SHA-256 mismatch for {path}: expected {expected}, got {actual}"
                )


@dataclass(frozen=True, slots=True)
class PrototypeEvidenceResult:
    dashboard_path: Path
    competitors_path: Path
    references_path: Path
    report_path: Path
    summary_path: Path
    report: dict[str, Any]


def build_prototype_evidence_outputs(
    config: PrototypeEvidenceConfig,
) -> PrototypeEvidenceResult:
    config.verify_inputs()
    classifications = pl.read_parquet(config.classifications)
    candidates = pl.read_parquet(config.candidate_scores)
    support = pl.read_parquet(config.support_manifest)
    _validate_inputs(
        classifications,
        candidates,
        support,
        config=config,
    )
    references = _nearest_reference_evidence(
        classifications,
        support,
        config=config,
    )
    competitors = _regional_competitor_evidence(candidates, config=config)
    dashboard = _dashboard(
        classifications,
        references,
        competitors,
        config=config,
    )
    _validate_public_output(dashboard, artifact="dashboard")
    _validate_public_output(references, artifact="nearest references")
    _validate_public_output(competitors, artifact="regional competitors")

    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    dashboard_path = write_parquet(
        dashboard,
        output / PROTOTYPE_EVIDENCE_DASHBOARD_FILE,
    )
    competitors_path = write_parquet(
        competitors,
        output / PROTOTYPE_EVIDENCE_COMPETITORS_FILE,
    )
    references_path = write_parquet(
        references,
        output / PROTOTYPE_EVIDENCE_REFERENCES_FILE,
    )
    artifact_records = {
        "dashboard": _artifact_record(dashboard_path, dashboard.height),
        "regional_competitors": _artifact_record(
            competitors_path,
            competitors.height,
        ),
        "nearest_references": _artifact_record(references_path, references.height),
    }
    report = _report(
        dashboard,
        references,
        competitors,
        artifact_records=artifact_records,
        config=config,
    )
    report_path = output / PROTOTYPE_EVIDENCE_REPORT_FILE
    summary_path = output / PROTOTYPE_EVIDENCE_SUMMARY_FILE
    report_text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    summary_text = prototype_evidence_markdown(report)
    _assert_no_secrets(report_text)
    _assert_no_secrets(summary_text)
    report_path.write_text(report_text, encoding="utf-8")
    summary_path.write_text(summary_text, encoding="utf-8")
    return PrototypeEvidenceResult(
        dashboard_path=dashboard_path,
        competitors_path=competitors_path,
        references_path=references_path,
        report_path=report_path,
        summary_path=summary_path,
        report=report,
    )


def prototype_evidence_markdown(report: Mapping[str, Any]) -> str:
    counts = dict(report["counts"])
    lines = [
        "# Build Week prototype reference evidence",
        "",
        "> Prototype screening evidence only. Scores are uncalibrated and are "
        "not probabilities or taxonomic validation.",
        "",
        "## Dashboard coverage",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Flickr records | {counts['dashboard_rows']} |",
        f"| Regional competitor rows | {counts['regional_competitor_rows']} |",
        f"| Nearest reference rows | {counts['nearest_reference_rows']} |",
        f"| Abstained records | {counts['abstained_records']} |",
        f"| Target scored records | {counts['target_scored_records']} |",
        "",
        "## Exposed evidence",
        "",
        *[f"- `{item}`" for item in report["exposed_fields"]],
        "",
        "## Why this image was ranked",
        "",
    ]
    for example in report["evidence_panel_examples"]:
        panel = dict(example["why_this_image_was_ranked"])
        lines.extend(
            [
                f"### Flickr photo `{example['flickr_photo_id']}`",
                "",
                f"- Target: `{panel['target_species']}`",
                f"- Target similarity: `{panel['target_similarity']}`",
                f"- Best competitor: `{panel['best_competitor_name']}`",
                f"- Best competitor similarity: "
                f"`{panel['best_competitor_similarity']}`",
                f"- Competitor margin: `{panel['competitor_margin']}`",
                f"- Text score: `{panel['text_score']}`",
                f"- Geographic evidence: `{panel['geographic_evidence']}`",
                f"- Visual input: `{panel['visual_input_used']}`",
                f"- YOLOE route: `{panel['yoloe_route']}`",
                f"- Abstention: `{panel['abstention']}`",
                f"- Abstention reason: `{panel['abstention_reason']}`",
                "",
                "Closest target references:",
                "",
                *_reference_lines(panel["closest_target_references"]),
                "",
                "Closest competitor references:",
                "",
                *_reference_lines(panel["closest_competitor_references"]),
                "",
            ]
        )
    lines.extend(
        [
            "## Verification limitations",
            "",
            *[f"- {item}" for item in report["verification_limitations"]],
            "",
            "## Publication safety",
            "",
            "- No credentials are included.",
            "- No reference image bytes or source-object URIs are included.",
            "- Reference examples retain licence and attribution metadata.",
            "",
        ]
    )
    return "\n".join(lines)


def _nearest_reference_evidence(
    classifications: pl.DataFrame,
    support: pl.DataFrame,
    *,
    config: PrototypeEvidenceConfig,
) -> pl.DataFrame:
    public_support = support.select(
        "reference_media_id",
        "provider_media_id",
        "accepted_taxon_key",
        "scientific_name",
        "trust_level",
        "verification_status",
        "geographic_layer",
        "geo_cluster_id",
        "route",
        "reference_group",
        "licence",
        "licence_policy_status",
        "attribution",
    )
    return (
        classifications.select(
            "order_index",
            "flickr_photo_id",
            pl.col("nearest_references_json")
            .str.json_decode(dtype=_NEAREST_REFERENCE_DTYPE)
            .alias("nearest_reference"),
        )
        .explode("nearest_reference")
        .unnest("nearest_reference")
        .join(
            public_support,
            on=["reference_media_id", "accepted_taxon_key", "scientific_name"],
            how="left",
            validate="m:1",
        )
        .with_columns(
            (
                pl.col("accepted_taxon_key") == pl.lit(config.target_accepted_taxon_key)
            ).alias("target_reference"),
            pl.lit("reference_identifier_only_no_image_copy").alias(
                "media_exposure_policy"
            ),
        )
        .select(
            "order_index",
            "flickr_photo_id",
            "target_reference",
            "accepted_taxon_key",
            "scientific_name",
            "reference_media_id",
            "provider_media_id",
            "similarity",
            "trust_level",
            "verification_status",
            "geographic_layer",
            "geo_cluster_id",
            "route",
            "reference_group",
            "licence",
            "licence_policy_status",
            "attribution",
            "media_exposure_policy",
        )
        .sort(
            [
                "order_index",
                "target_reference",
                "similarity",
                "reference_media_id",
            ],
            descending=[False, True, True, False],
        )
    )


def _regional_competitor_evidence(
    candidates: pl.DataFrame,
    *,
    config: PrototypeEvidenceConfig,
) -> pl.DataFrame:
    return (
        candidates.filter(
            (pl.col("class_kind") == "species")
            & ~pl.col("target_candidate")
            & (pl.col("accepted_taxon_key") != pl.lit(config.target_accepted_taxon_key))
        )
        .select(
            "order_index",
            "flickr_photo_id",
            pl.col("accepted_taxon_key").alias("competitor_accepted_taxon_key"),
            pl.col("display_name").alias("competitor_scientific_name"),
            "candidate_reason",
            "text_similarity",
            "reference_prototype_similarity",
            pl.col("prototype_score").alias("competitor_score"),
            pl.col("rank").alias("competitor_rank"),
            "score_semantics",
            "experimental_screening_evidence",
        )
        .sort(
            [
                "order_index",
                "competitor_rank",
                "competitor_accepted_taxon_key",
            ]
        )
    )


def _dashboard(
    classifications: pl.DataFrame,
    references: pl.DataFrame,
    competitors: pl.DataFrame,
    *,
    config: PrototypeEvidenceConfig,
) -> pl.DataFrame:
    reference_fields = [
        "reference_media_id",
        "provider_media_id",
        "accepted_taxon_key",
        "scientific_name",
        "similarity",
        "trust_level",
        "verification_status",
        "geographic_layer",
        "reference_group",
        "route",
        "licence",
        "licence_policy_status",
        "attribution",
        "media_exposure_policy",
    ]
    target_references = (
        references.filter(pl.col("target_reference"))
        .group_by("flickr_photo_id")
        .agg(
            pl.struct(reference_fields)
            .head(config.target_reference_limit)
            .alias("closest_target_references")
        )
    )
    competitor_references = (
        references.filter(~pl.col("target_reference"))
        .group_by("flickr_photo_id")
        .agg(
            pl.struct(reference_fields)
            .head(config.competitor_reference_limit)
            .alias("closest_competitor_references")
        )
    )
    target_trust = (
        references.filter(pl.col("target_reference"))
        .group_by("flickr_photo_id")
        .agg(
            pl.col("trust_level")
            .drop_nulls()
            .unique()
            .sort()
            .alias("target_reference_trust_levels")
        )
    )
    reference_layers = (
        references.filter(pl.col("target_reference"))
        .group_by("flickr_photo_id")
        .agg(
            pl.col("geographic_layer").first().alias("geographic_layer"),
            pl.col("reference_group").first().alias("reference_layer"),
        )
    )
    competitor_preview_fields = [
        "competitor_accepted_taxon_key",
        "competitor_scientific_name",
        "candidate_reason",
        "text_similarity",
        "reference_prototype_similarity",
        "competitor_score",
        "competitor_rank",
    ]
    competitor_preview = competitors.group_by("flickr_photo_id").agg(
        pl.struct(competitor_preview_fields)
        .head(config.regional_competitor_preview_limit)
        .alias("regional_competitors")
    )
    base = classifications.select(
        "order_index",
        "flickr_photo_id",
        pl.col("target_scientific_name").alias("target_species"),
        "target_accepted_taxon_key",
        "target_scored",
        "regional_candidate_count",
        "regional_scored_count",
        "geo_cluster_id",
        "coordinate_quality",
        pl.col("detection_route").alias("yoloe_route"),
        "bioclip_route",
        "reference_route_used",
        "visual_input",
        "complete_canvas_retained",
        "spatial_crop_applied",
        "higher_rank_pruning_applied",
        pl.col("nearest_target_reference_similarity").alias("target_similarity"),
        pl.col("best_competitor_reference_similarity").alias(
            "best_competitor_similarity"
        ),
        "best_competitor_reference_key",
        "best_competitor_reference_name",
        pl.col("target_competitor_reference_margin").alias("competitor_margin"),
        pl.col("target_text_similarity").alias("text_score"),
        "prototype_score",
        "uncalibrated_margin",
        pl.col("abstain").alias("abstention"),
        "abstention_reason",
        "prototype_status",
        "score_semantics",
        "experimental_screening_evidence",
        "flickr_query_match_is_label",
    )
    joined = (
        base.join(target_references, on="flickr_photo_id", how="left")
        .join(competitor_references, on="flickr_photo_id", how="left")
        .join(target_trust, on="flickr_photo_id", how="left")
        .join(reference_layers, on="flickr_photo_id", how="left")
        .join(competitor_preview, on="flickr_photo_id", how="left")
        .with_columns(
            pl.lit(list(config.verification_limitations)).alias(
                "verification_limitations"
            ),
            pl.lit("prototype").alias("deployment_status"),
        )
    )
    return joined.with_columns(
        pl.struct(
            pl.col("target_species"),
            pl.col("closest_target_references"),
            pl.col("closest_competitor_references"),
            pl.col("regional_competitors"),
            pl.struct(
                "geo_cluster_id",
                "coordinate_quality",
                "geographic_layer",
            ).alias("geographic_evidence"),
            pl.col("visual_input").alias("visual_input_used"),
            pl.col("target_similarity"),
            pl.col("best_competitor_reference_name").alias("best_competitor_name"),
            pl.col("best_competitor_similarity"),
            pl.col("competitor_margin"),
            pl.col("text_score"),
            pl.col("yoloe_route"),
            pl.col("abstention"),
            pl.col("abstention_reason"),
        )
        .struct.json_encode()
        .alias("why_this_image_was_ranked_json")
    ).sort("order_index")


def _validate_inputs(
    classifications: pl.DataFrame,
    candidates: pl.DataFrame,
    support: pl.DataFrame,
    *,
    config: PrototypeEvidenceConfig,
) -> None:
    required_classification = {
        "flickr_photo_id",
        "nearest_references_json",
        "target_scientific_name",
        "target_accepted_taxon_key",
        "target_scored",
        "regional_candidate_count",
        "regional_scored_count",
        "higher_rank_pruning_applied",
        "spatial_crop_applied",
        "visual_input",
        "score_semantics",
        "prototype_status",
        "flickr_query_match_is_label",
    }
    required_candidates = {
        "flickr_photo_id",
        "class_kind",
        "accepted_taxon_key",
        "target_candidate",
        "display_name",
        "candidate_reason",
        "prototype_score",
    }
    required_support = {
        "reference_media_id",
        "provider_media_id",
        "accepted_taxon_key",
        "scientific_name",
        "trust_level",
        "verification_status",
        "geographic_layer",
        "route",
        "reference_group",
        "licence",
        "licence_policy_status",
        "attribution",
    }
    for frame, required, label in (
        (classifications, required_classification, "classifications"),
        (candidates, required_candidates, "candidate scores"),
        (support, required_support, "support manifest"),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} omit columns: {missing}")
        if frame.is_empty():
            raise ValueError(f"{label} must not be empty")
    target_rows = classifications.filter(
        (pl.col("target_accepted_taxon_key") == config.target_accepted_taxon_key)
        & (pl.col("target_scientific_name") == config.target_scientific_name)
    )
    if target_rows.height != classifications.height:
        raise ValueError("classification target identity is inconsistent")
    if classifications.filter(
        ~pl.col("target_scored")
        | (pl.col("regional_candidate_count") != pl.col("regional_scored_count"))
        | pl.col("higher_rank_pruning_applied")
        | pl.col("spatial_crop_applied")
        | (pl.col("visual_input") != "raw_full_image")
        | pl.col("flickr_query_match_is_label")
    ).height:
        raise ValueError("prototype classification evidence invariants failed")
    if support.filter(pl.col("human_verified")).height:
        raise ValueError(
            "provider-supported references cannot be called human verified"
        )


def _validate_public_output(frame: pl.DataFrame, *, artifact: str) -> None:
    forbidden = sorted(_FORBIDDEN_PUBLIC_COLUMNS.intersection(frame.columns))
    if forbidden:
        raise ValueError(f"{artifact} exposes forbidden fields: {forbidden}")
    _assert_no_secrets(
        json.dumps(
            frame.head(100).to_dicts(),
            sort_keys=True,
            default=str,
        )
    )


def _report(
    dashboard: pl.DataFrame,
    references: pl.DataFrame,
    competitors: pl.DataFrame,
    *,
    artifact_records: dict[str, Any],
    config: PrototypeEvidenceConfig,
) -> dict[str, Any]:
    samples = []
    for row in dashboard.head(config.report_example_limit).to_dicts():
        samples.append(
            {
                "flickr_photo_id": row["flickr_photo_id"],
                "why_this_image_was_ranked": json.loads(
                    row["why_this_image_was_ranked_json"]
                ),
            }
        )
    return {
        "schema_version": PROTOTYPE_EVIDENCE_REPORT_VERSION,
        "status": "passed",
        "deployment_status": "prototype",
        "storage_backend": "local",
        "s3_used": False,
        "target": {
            "accepted_taxon_key": config.target_accepted_taxon_key,
            "scientific_name": config.target_scientific_name,
        },
        "counts": {
            "dashboard_rows": dashboard.height,
            "regional_competitor_rows": competitors.height,
            "nearest_reference_rows": references.height,
            "target_scored_records": dashboard.filter(pl.col("target_scored")).height,
            "abstained_records": dashboard.filter(pl.col("abstention")).height,
        },
        "exposed_fields": [
            "target species",
            "target reference examples",
            "target reference trust levels",
            "regional competitors",
            "nearest reference images by safe identifier",
            "target similarity",
            "best competitor similarity",
            "competitor margin",
            "text score",
            "geographic layer",
            "reference layer",
            "YOLOE route",
            "abstention",
            "prototype status",
            "verification limitations",
            "Why this image was ranked",
        ],
        "evidence_panel_examples": samples,
        "verification_limitations": list(config.verification_limitations),
        "publication_safety": {
            "credentials_exposed": False,
            "source_image_bytes_exposed": False,
            "source_object_uris_exposed": False,
            "reference_image_urls_exposed": False,
            "licence_metadata_retained": True,
            "attribution_retained": True,
        },
        "artifacts": artifact_records,
    }


def _artifact_record(path: Path, rows: int) -> dict[str, Any]:
    return {
        "path": str(path),
        "row_count": rows,
        "byte_count": path.stat().st_size,
        "sha256": _file_sha256(path),
    }


def _reference_lines(values: object) -> list[str]:
    items = values if isinstance(values, list) else []
    if not items:
        return ["- none"]
    return [
        "- "
        f"`{item['reference_media_id']}` — {item['scientific_name']}; "
        f"similarity `{item['similarity']}`; trust `{item['trust_level']}`; "
        f"layer `{item['geographic_layer']}`; licence `{item['licence']}`; "
        f"attribution `{item['attribution']}`"
        for item in items
    ]


def _assert_no_secrets(value: str) -> None:
    if _SECRET_PATTERN.search(value):
        raise ValueError("prototype evidence output contains secret-like text")


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _require_sha256(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"sha256:[0-9a-f]{64}", value):
        raise ValueError(f"{field} must be sha256:<64 lowercase hex>")
    return value


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value.strip()


__all__ = [
    "PROTOTYPE_EVIDENCE_COMPETITORS_FILE",
    "PROTOTYPE_EVIDENCE_CONFIG_VERSION",
    "PROTOTYPE_EVIDENCE_DASHBOARD_FILE",
    "PROTOTYPE_EVIDENCE_REFERENCES_FILE",
    "PROTOTYPE_EVIDENCE_REPORT_FILE",
    "PROTOTYPE_EVIDENCE_REPORT_VERSION",
    "PROTOTYPE_EVIDENCE_SUMMARY_FILE",
    "PrototypeEvidenceConfig",
    "PrototypeEvidenceResult",
    "build_prototype_evidence_outputs",
    "prototype_evidence_markdown",
]
