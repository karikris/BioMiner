from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from biominer.storage.parquet import write_parquet
from biominer.storage.uri import is_cloud_uri, join_uri


CLASSIFICATION_TABLE_VERSION = "gbif-butterfly-classification-v1"
PROMPT_VARIANT_VERSION = "butterfly-hierarchical-prompts-v1"

BUTTERFLY_CLASSIFICATION_TAXA_FILE = "butterfly_classification_taxa.parquet"
BUTTERFLY_FAMILY_LABELS_FILE = "butterfly_family_labels.parquet"
BUTTERFLY_SPECIES_LABELS_FILE = "butterfly_species_labels.parquet"
BUTTERFLY_CLASSIFICATION_MANIFEST_FILE = "butterfly_classification_manifest.json"
BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE = "butterfly_classification_qa_findings.parquet"

FAMILY_PROMPT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("family_prompt", "a photo of a butterfly in the family {family}"),
    ("family_prompt", "a close-up photo of a butterfly in the family {family}"),
    ("family_prompt", "a field photo of a butterfly in the family {family}"),
)

SPECIES_PROMPT_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("species_prompt", "a photo of {scientific_name}"),
    ("species_prompt", "a close-up photo of the butterfly species {scientific_name}"),
    ("species_prompt", "a field photo of the butterfly species {scientific_name}"),
)

CLASSIFICATION_TAXA_SCHEMA: dict[str, pl.DataType] = {
    "registry_version": pl.String,
    "classification_table_version": pl.String,
    "source": pl.String,
    "source_version": pl.String,
    "retrieved_at": pl.String,
    "scope_id": pl.String,
    "accepted_taxon_key": pl.String,
    "gbif_species_key": pl.String,
    "scientific_name": pl.String,
    "canonical_name": pl.String,
    "rank": pl.String,
    "taxonomic_status": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "species_key": pl.String,
    "species": pl.String,
    "species_epithet": pl.String,
    "in_scope": pl.Boolean,
    "classification_enabled": pl.Boolean,
    "classification_disabled_reason": pl.String,
}

FAMILY_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "registry_version": pl.String,
    "classification_table_version": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "label": pl.String,
    "label_kind": pl.String,
    "prompt_template": pl.String,
    "prompt_variant_version": pl.String,
    "enabled": pl.Boolean,
    "sort_order": pl.Int64,
}

SPECIES_LABEL_SCHEMA: dict[str, pl.DataType] = {
    "registry_version": pl.String,
    "classification_table_version": pl.String,
    "accepted_taxon_key": pl.String,
    "gbif_species_key": pl.String,
    "family_key": pl.String,
    "family": pl.String,
    "genus_key": pl.String,
    "genus": pl.String,
    "scientific_name": pl.String,
    "canonical_name": pl.String,
    "label": pl.String,
    "label_kind": pl.String,
    "prompt_template": pl.String,
    "prompt_variant_version": pl.String,
    "enabled": pl.Boolean,
    "sort_order": pl.Int64,
}

CLASSIFICATION_QA_SCHEMA: dict[str, pl.DataType] = {
    "severity": pl.String,
    "code": pl.String,
    "table": pl.String,
    "message": pl.String,
    "details_json": pl.String,
}

EXPECTED_CLASSIFICATION_ARTIFACTS = (
    BUTTERFLY_CLASSIFICATION_TAXA_FILE,
    BUTTERFLY_FAMILY_LABELS_FILE,
    BUTTERFLY_SPECIES_LABELS_FILE,
    BUTTERFLY_CLASSIFICATION_MANIFEST_FILE,
    BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE,
)

_MIN_SPECIES_PER_FAMILY_WARNING_THRESHOLD = 2


def normalize_rank(value: object) -> str:
    return " ".join(str(value or "").strip().split()).upper()


def bare_gbif_key(value: object) -> str:
    text = str(value or "").strip()
    if text.casefold().startswith("gbif:"):
        text = text.split(":", 1)[1]
    return text.strip()


def derive_species_epithet(value: object) -> str:
    parts = [part.strip() for part in str(value or "").replace(" x ", " ").split() if part.strip()]
    return parts[1] if len(parts) >= 2 else ""


def source_metadata_from_manifest_or_snapshots(
    *,
    registry_manifest: dict[str, object] | None = None,
    source_snapshots: pl.DataFrame | None = None,
) -> dict[str, str]:
    manifest = registry_manifest or {}
    source = str(manifest.get("source") or "")
    source_version = str(manifest.get("source_version") or "")
    retrieved_at = str(manifest.get("retrieved_at") or manifest.get("start") or "")
    if source_snapshots is not None and not source_snapshots.is_empty():
        rows = source_snapshots.to_dicts()
        selected = next((row for row in rows if str(row.get("source") or "").casefold() == "gbif"), rows[0])
        source = str(selected.get("source") or source or "")
        source_version = str(selected.get("source_version") or source_version or "")
        retrieved_at = str(selected.get("retrieved_at") or retrieved_at or "")
    return {
        "source": source or "GBIF",
        "source_version": source_version,
        "retrieved_at": retrieved_at,
    }


def build_classification_taxa_frame(
    taxa: pl.DataFrame,
    *,
    registry_manifest: dict[str, object] | None = None,
    source_snapshots: pl.DataFrame | None = None,
) -> pl.DataFrame:
    _require_columns(taxa, ("accepted_taxon_key", "scientific_name", "rank"), table="taxa")
    manifest = registry_manifest or {}
    metadata = source_metadata_from_manifest_or_snapshots(
        registry_manifest=registry_manifest,
        source_snapshots=source_snapshots,
    )
    registry_version = str(manifest.get("registry_version") or _first_column_value(taxa, "registry_version") or "")
    rows: list[dict[str, object]] = []
    for index, row in enumerate(taxa.to_dicts()):
        if normalize_rank(row.get("rank")).casefold() != "species":
            continue
        in_scope = _bool_value(row.get("in_scope"), default=True)
        if not in_scope:
            continue
        accepted_taxon_key = _text(row.get("accepted_taxon_key"))
        scientific_name = _text(row.get("scientific_name"))
        species = _text(row.get("species")) or scientific_name
        canonical_name = species or scientific_name
        species_key = _text(row.get("species_key")) or accepted_taxon_key
        family_key = _text(row.get("family_key"))
        family = _text(row.get("family"))
        genus_key = _text(row.get("genus_key"))
        genus = _text(row.get("genus"))
        disabled_reasons = []
        if not accepted_taxon_key:
            disabled_reasons.append("missing_accepted_taxon_key")
        if not scientific_name:
            disabled_reasons.append("missing_scientific_name")
        if not family_key:
            disabled_reasons.append("missing_family_key")
        if not family:
            disabled_reasons.append("missing_family")
        rows.append(
            {
                "registry_version": registry_version,
                "classification_table_version": CLASSIFICATION_TABLE_VERSION,
                "source": metadata["source"],
                "source_version": metadata["source_version"],
                "retrieved_at": metadata["retrieved_at"],
                "scope_id": _text(row.get("scope_id")) or str(manifest.get("scope_id") or ""),
                "accepted_taxon_key": accepted_taxon_key,
                "gbif_species_key": bare_gbif_key(accepted_taxon_key or species_key),
                "scientific_name": scientific_name,
                "canonical_name": canonical_name,
                "rank": "SPECIES",
                "taxonomic_status": _text(row.get("taxonomic_status"), row.get("status")) or "accepted",
                "family_key": family_key,
                "family": family,
                "genus_key": genus_key,
                "genus": genus,
                "species_key": species_key,
                "species": species,
                "species_epithet": derive_species_epithet(canonical_name),
                "in_scope": True,
                "classification_enabled": not disabled_reasons,
                "classification_disabled_reason": ",".join(disabled_reasons),
                "_source_row_index": index,
            }
        )
    if not rows:
        return pl.DataFrame(schema=CLASSIFICATION_TAXA_SCHEMA)
    frame = pl.DataFrame(rows).sort(["family", "genus", "scientific_name", "accepted_taxon_key", "_source_row_index"])
    frame = _dedupe_classification_taxa(frame)
    return ensure_classification_taxa_schema(frame.drop("_source_row_index", strict=False))


def ensure_classification_taxa_schema(frame: pl.DataFrame) -> pl.DataFrame:
    return _ensure_schema(frame, CLASSIFICATION_TAXA_SCHEMA)


def build_family_label_frame(classification_taxa: pl.DataFrame) -> pl.DataFrame:
    taxa = ensure_classification_taxa_schema(classification_taxa)
    if taxa.is_empty():
        return pl.DataFrame(schema=FAMILY_LABEL_SCHEMA)
    family_rows = (
        taxa.filter(pl.col("classification_enabled"))
        .select(["registry_version", "classification_table_version", "family_key", "family"])
        .filter((pl.col("family_key") != "") & (pl.col("family") != ""))
        .sort(["family", "family_key"])
        .unique(subset=["family_key"], keep="first")
        .to_dicts()
    )
    rows: list[dict[str, object]] = []
    for family_row in family_rows:
        for index, (label_kind, template) in enumerate(FAMILY_PROMPT_TEMPLATES, start=1):
            rows.append(
                {
                    "registry_version": family_row["registry_version"],
                    "classification_table_version": family_row["classification_table_version"],
                    "family_key": family_row["family_key"],
                    "family": family_row["family"],
                    "label": template.format(family=family_row["family"]),
                    "label_kind": label_kind,
                    "prompt_template": template,
                    "prompt_variant_version": PROMPT_VARIANT_VERSION,
                    "enabled": True,
                    "sort_order": index,
                }
            )
    return _ensure_schema(pl.DataFrame(rows), FAMILY_LABEL_SCHEMA).sort(["family", "sort_order", "label"])


def build_species_label_frame(classification_taxa: pl.DataFrame) -> pl.DataFrame:
    taxa = ensure_classification_taxa_schema(classification_taxa)
    if taxa.is_empty():
        return pl.DataFrame(schema=SPECIES_LABEL_SCHEMA)
    species_rows = (
        taxa.filter(pl.col("classification_enabled"))
        .select(
            [
                "registry_version",
                "classification_table_version",
                "accepted_taxon_key",
                "gbif_species_key",
                "family_key",
                "family",
                "genus_key",
                "genus",
                "scientific_name",
                "canonical_name",
            ]
        )
        .sort(["family", "genus", "scientific_name", "accepted_taxon_key"])
        .to_dicts()
    )
    rows: list[dict[str, object]] = []
    for species_row in species_rows:
        for index, (label_kind, template) in enumerate(SPECIES_PROMPT_TEMPLATES, start=1):
            rows.append(
                {
                    **species_row,
                    "label": template.format(scientific_name=species_row["scientific_name"]),
                    "label_kind": label_kind,
                    "prompt_template": template,
                    "prompt_variant_version": PROMPT_VARIANT_VERSION,
                    "enabled": True,
                    "sort_order": index,
                }
            )
    return _ensure_schema(pl.DataFrame(rows), SPECIES_LABEL_SCHEMA).sort(["family", "genus", "scientific_name", "sort_order", "label"])


def build_classification_artifact_frames(
    taxa: pl.DataFrame,
    *,
    registry_manifest: dict[str, object] | None = None,
    source_snapshots: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, object]]:
    classification_taxa = build_classification_taxa_frame(
        taxa,
        registry_manifest=registry_manifest,
        source_snapshots=source_snapshots,
    )
    family_labels = build_family_label_frame(classification_taxa)
    species_labels = build_species_label_frame(classification_taxa)
    findings = validate_classification_tables(classification_taxa, family_labels, species_labels)
    qa_findings = classification_qa_findings_frame(findings)
    manifest = build_classification_table_manifest(
        registry_manifest=registry_manifest or {},
        classification_taxa=classification_taxa,
        family_labels=family_labels,
        species_labels=species_labels,
        findings=findings,
    )
    return classification_taxa, family_labels, species_labels, qa_findings, manifest


def build_classification_table_manifest(
    *,
    registry_manifest: dict[str, object],
    classification_taxa: pl.DataFrame,
    family_labels: pl.DataFrame,
    species_labels: pl.DataFrame,
    findings: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    taxa = ensure_classification_taxa_schema(classification_taxa)
    enabled = taxa.filter(pl.col("classification_enabled"))
    finding_rows = findings if findings is not None else validate_classification_tables(taxa, family_labels, species_labels)
    return {
        "classification_table_version": CLASSIFICATION_TABLE_VERSION,
        "prompt_variant_version": PROMPT_VARIANT_VERSION,
        "registry_version": str(registry_manifest.get("registry_version") or _first_column_value(taxa, "registry_version") or ""),
        "source_registry_qa_status": registry_manifest.get("qa_status"),
        "source_registry_manifest_hash": _stable_json_hash(registry_manifest) if registry_manifest else None,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "species_count": enabled.height,
        "family_count": enabled.select("family_key").unique().height if "family_key" in enabled.columns else 0,
        "genus_count": enabled.filter(pl.col("genus") != "").select("genus").unique().height if "genus" in enabled.columns else 0,
        "family_label_count": family_labels.height,
        "species_label_count": species_labels.height,
        "disabled_species_count": taxa.filter(~pl.col("classification_enabled")).height,
        "fatal_finding_count": sum(1 for finding in finding_rows if finding.get("severity") == "fatal"),
        "warning_finding_count": sum(1 for finding in finding_rows if finding.get("severity") == "warning"),
        "expected_artifacts": list(EXPECTED_CLASSIFICATION_ARTIFACTS),
        "classification_taxa_rows": taxa.height,
    }


def validate_classification_tables(
    classification_taxa: pl.DataFrame,
    family_labels: pl.DataFrame,
    species_labels: pl.DataFrame,
) -> list[dict[str, object]]:
    taxa = ensure_classification_taxa_schema(classification_taxa)
    families = _ensure_schema(family_labels, FAMILY_LABEL_SCHEMA)
    species = _ensure_schema(species_labels, SPECIES_LABEL_SCHEMA)
    findings: list[dict[str, object]] = []
    enabled = taxa.filter(pl.col("classification_enabled"))
    if enabled.is_empty():
        findings.append(_finding("fatal", "no_enabled_species", "butterfly_classification_taxa", "classification table has no enabled species"))
    if families.filter(pl.col("enabled")).is_empty():
        findings.append(_finding("fatal", "no_family_labels", "butterfly_family_labels", "family label table has no enabled labels"))
    if species.filter(pl.col("enabled")).is_empty():
        findings.append(_finding("fatal", "no_species_labels", "butterfly_species_labels", "species label table has no enabled labels"))
    _append_duplicate_enabled_taxa_findings(findings, enabled)
    _append_missing_enabled_taxa_findings(findings, enabled)
    _append_reference_findings(findings, enabled, families, species)
    _append_label_count_findings(findings, enabled, families, species)
    _append_family_size_findings(findings, enabled)
    if not enabled.filter(pl.col("genus") == "").is_empty():
        findings.append(_finding("warning", "enabled_species_missing_genus", "butterfly_classification_taxa", "one or more enabled species is missing genus"))
    return findings


def classification_qa_findings_frame(findings: Sequence[dict[str, object]]) -> pl.DataFrame:
    rows = [
        {
            "severity": str(finding.get("severity") or ""),
            "code": str(finding.get("code") or ""),
            "table": str(finding.get("table") or ""),
            "message": str(finding.get("message") or ""),
            "details_json": json.dumps(finding.get("details") or {}, sort_keys=True),
        }
        for finding in findings
    ]
    return _ensure_schema(pl.DataFrame(rows), CLASSIFICATION_QA_SCHEMA)


def build_classification_tables_from_registry_dir(
    registry_dir: str | Path,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    registry = Path(registry_dir)
    output = Path(output_dir) if output_dir is not None else registry
    taxa_path = registry / "taxa.parquet"
    if not taxa_path.exists():
        raise FileNotFoundError(f"missing required registry artifact: {taxa_path}")
    registry_manifest = _read_json_optional(registry / "manifest.json")
    source_snapshots_path = registry / "source_snapshots.parquet"
    source_snapshots = pl.read_parquet(source_snapshots_path) if source_snapshots_path.exists() else None
    classification_taxa, family_labels, species_labels, qa_findings, manifest = build_classification_artifact_frames(
        pl.read_parquet(taxa_path),
        registry_manifest=registry_manifest,
        source_snapshots=source_snapshots,
    )
    output.mkdir(parents=True, exist_ok=True)
    paths = classification_artifact_paths(output)
    write_parquet(classification_taxa, paths["classification_taxa"])
    write_parquet(family_labels, paths["family_labels"])
    write_parquet(species_labels, paths["species_labels"])
    write_parquet(qa_findings, paths["qa_findings"])
    file_sizes = _local_file_sizes(paths)
    manifest = {
        **manifest,
        "artifact_file_sizes": file_sizes,
        "estimated_metadata_only_size_mb": _bytes_to_mb(file_sizes.get("classification_taxa", 0)),
    }
    Path(paths["manifest"]).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return _classification_summary(
        registry_dir=registry,
        output_dir=output,
        paths=paths,
        manifest=manifest,
        classification_taxa=classification_taxa,
        family_labels=family_labels,
        species_labels=species_labels,
        qa_findings=qa_findings,
    )


def classification_artifact_paths(root: str | Path) -> dict[str, Path]:
    root_path = Path(root)
    base = root_path.parent if root_path.suffix == ".parquet" else root_path
    classification_taxa = root_path if root_path.suffix == ".parquet" else base / BUTTERFLY_CLASSIFICATION_TAXA_FILE
    return {
        "classification_taxa": classification_taxa,
        "family_labels": base / BUTTERFLY_FAMILY_LABELS_FILE,
        "species_labels": base / BUTTERFLY_SPECIES_LABELS_FILE,
        "manifest": base / BUTTERFLY_CLASSIFICATION_MANIFEST_FILE,
        "qa_findings": base / BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE,
    }


def classification_artifact_uris(root: str | Path) -> dict[str, str]:
    text = str(root)
    base = text.rsplit("/", 1)[0] if text.endswith(".parquet") else text.rstrip("/")
    classification_taxa = text if text.endswith(".parquet") else join_uri(base, BUTTERFLY_CLASSIFICATION_TAXA_FILE)
    return {
        "classification_taxa": classification_taxa,
        "family_labels": join_uri(base, BUTTERFLY_FAMILY_LABELS_FILE),
        "species_labels": join_uri(base, BUTTERFLY_SPECIES_LABELS_FILE),
        "manifest": join_uri(base, BUTTERFLY_CLASSIFICATION_MANIFEST_FILE),
        "qa_findings": join_uri(base, BUTTERFLY_CLASSIFICATION_QA_FINDINGS_FILE),
    }


@dataclass(frozen=True)
class ButterflyTaxonomyStore:
    classification_taxa: pl.DataFrame
    family_labels: pl.DataFrame
    species_labels: pl.DataFrame
    manifest: dict[str, object] | None = None

    @classmethod
    def read(cls, root: str | Path) -> ButterflyTaxonomyStore:
        if is_cloud_uri(str(root)):
            raise ValueError("ButterflyTaxonomyStore.read currently supports local paths only")
        paths = classification_artifact_paths(root)
        missing = [str(path) for key, path in paths.items() if key != "qa_findings" and key != "manifest" and not path.exists()]
        if missing:
            raise FileNotFoundError("missing butterfly classification artifacts: " + ", ".join(missing))
        manifest = _read_json_optional(paths["manifest"])
        store = cls(
            classification_taxa=pl.read_parquet(paths["classification_taxa"]),
            family_labels=pl.read_parquet(paths["family_labels"]),
            species_labels=pl.read_parquet(paths["species_labels"]),
            manifest=manifest,
        )
        fatal = [finding for finding in store.validation_findings() if finding.get("severity") == "fatal"]
        if fatal:
            codes = ", ".join(str(finding.get("code")) for finding in fatal)
            raise ValueError(f"invalid butterfly classification artifacts: {codes}")
        return store

    def validation_findings(self) -> list[dict[str, object]]:
        return validate_classification_tables(self.classification_taxa, self.family_labels, self.species_labels)

    def family_candidates(self) -> pl.DataFrame:
        taxa = ensure_classification_taxa_schema(self.classification_taxa)
        return (
            taxa.filter(pl.col("classification_enabled"))
            .select(["family_key", "family"])
            .unique(subset=["family_key"], keep="first")
            .sort(["family", "family_key"])
        )

    def family_prompt_labels(self) -> tuple[str, ...]:
        return tuple(
            _ensure_schema(self.family_labels, FAMILY_LABEL_SCHEMA)
            .filter(pl.col("enabled"))
            .sort(["family", "sort_order", "label"])
            .select("label")
            .to_series()
            .to_list()
        )

    def species_for_family(self, family_key: str) -> pl.DataFrame:
        family_key_text = str(family_key or "").strip()
        taxa = ensure_classification_taxa_schema(self.classification_taxa)
        known = set(self.family_candidates().select("family_key").to_series().to_list())
        if family_key_text not in known:
            raise KeyError(f"unknown family_key: {family_key_text}")
        return taxa.filter(pl.col("classification_enabled") & (pl.col("family_key") == family_key_text)).sort(
            ["family", "genus", "scientific_name", "accepted_taxon_key"]
        )

    def species_prompt_labels_for_family(self, family_key: str) -> tuple[str, ...]:
        self.species_for_family(family_key)
        return tuple(
            _ensure_schema(self.species_labels, SPECIES_LABEL_SCHEMA)
            .filter(pl.col("enabled") & (pl.col("family_key") == str(family_key or "").strip()))
            .sort(["family", "genus", "scientific_name", "sort_order", "label"])
            .select("label")
            .to_series()
            .to_list()
        )

    def species_labels_for_taxa(self, accepted_taxon_keys: Sequence[str]) -> pl.DataFrame:
        keys = [str(key) for key in accepted_taxon_keys]
        return (
            _ensure_schema(self.species_labels, SPECIES_LABEL_SCHEMA)
            .filter(pl.col("enabled") & pl.col("accepted_taxon_key").is_in(keys))
            .sort(["family", "genus", "scientific_name", "sort_order", "label"])
        )


def _classification_summary(
    *,
    registry_dir: Path,
    output_dir: Path,
    paths: dict[str, Path],
    manifest: dict[str, object],
    classification_taxa: pl.DataFrame,
    family_labels: pl.DataFrame,
    species_labels: pl.DataFrame,
    qa_findings: pl.DataFrame,
) -> dict[str, object]:
    enabled = classification_taxa.filter(pl.col("classification_enabled")) if "classification_enabled" in classification_taxa.columns else pl.DataFrame()
    file_sizes = dict(manifest.get("artifact_file_sizes") or {})
    return {
        "classification_table_version": CLASSIFICATION_TABLE_VERSION,
        "prompt_variant_version": PROMPT_VARIANT_VERSION,
        "registry_dir": str(registry_dir),
        "output_dir": str(output_dir),
        "outputs": {key: str(path) for key, path in paths.items()},
        "classification_taxa_rows": classification_taxa.height,
        "family_label_rows": family_labels.height,
        "species_label_rows": species_labels.height,
        "species_count": int(manifest.get("species_count") or enabled.height),
        "family_count": int(manifest.get("family_count") or 0),
        "genus_count": int(manifest.get("genus_count") or 0),
        "enabled_species_count": enabled.height,
        "disabled_species_count": int(manifest.get("disabled_species_count") or 0),
        "fatal_findings": int(manifest.get("fatal_finding_count") or 0),
        "warning_findings": int(manifest.get("warning_finding_count") or 0),
        "qa_finding_rows": qa_findings.height,
        "artifact_file_sizes": file_sizes,
        "local_file_sizes_mb": {key: _bytes_to_mb(value) for key, value in file_sizes.items()},
    }


def _require_columns(frame: pl.DataFrame, columns: Sequence[str], *, table: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")


def _ensure_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty() and not frame.columns:
        return pl.DataFrame(schema=schema)
    expressions = []
    for column, dtype in schema.items():
        if column in frame.columns:
            expressions.append(pl.col(column).cast(dtype).alias(column))
        else:
            expressions.append(pl.lit(_default_for_dtype(dtype)).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(schema))


def _dedupe_classification_taxa(frame: pl.DataFrame) -> pl.DataFrame:
    keyed = frame.filter(pl.col("accepted_taxon_key") != "").unique(subset=["accepted_taxon_key"], keep="first")
    unkeyed = frame.filter(pl.col("accepted_taxon_key") == "")
    return pl.concat([keyed, unkeyed], how="diagonal_relaxed").sort(["family", "genus", "scientific_name", "accepted_taxon_key", "_source_row_index"])


def _append_duplicate_enabled_taxa_findings(findings: list[dict[str, object]], enabled: pl.DataFrame) -> None:
    duplicates = (
        enabled.filter(pl.col("accepted_taxon_key") != "")
        .group_by("accepted_taxon_key")
        .len()
        .filter(pl.col("len") > 1)
        .sort("accepted_taxon_key")
    )
    if not duplicates.is_empty():
        findings.append(
            _finding(
                "fatal",
                "duplicate_enabled_accepted_taxon_key",
                "butterfly_classification_taxa",
                "enabled species contains duplicate accepted_taxon_key values",
                {"accepted_taxon_keys": duplicates["accepted_taxon_key"].to_list()},
            )
        )


def _append_missing_enabled_taxa_findings(findings: list[dict[str, object]], enabled: pl.DataFrame) -> None:
    checks = (
        ("enabled_species_missing_family_key", "family_key"),
        ("enabled_species_missing_family", "family"),
        ("enabled_species_missing_scientific_name", "scientific_name"),
    )
    for code, column in checks:
        if not enabled.filter(pl.col(column) == "").is_empty():
            findings.append(_finding("fatal", code, "butterfly_classification_taxa", f"one or more enabled species is missing {column}"))


def _append_reference_findings(findings: list[dict[str, object]], enabled: pl.DataFrame, family_labels: pl.DataFrame, species_labels: pl.DataFrame) -> None:
    known_species = set(enabled["accepted_taxon_key"].to_list()) if "accepted_taxon_key" in enabled.columns else set()
    known_families = set(enabled["family_key"].to_list()) if "family_key" in enabled.columns else set()
    family_refs = set(family_labels.filter(pl.col("enabled"))["family_key"].to_list()) if "family_key" in family_labels.columns else set()
    species_refs = set(species_labels.filter(pl.col("enabled"))["accepted_taxon_key"].to_list()) if "accepted_taxon_key" in species_labels.columns else set()
    unknown_family_refs = sorted(ref for ref in family_refs if ref and ref not in known_families)
    unknown_species_refs = sorted(ref for ref in species_refs if ref and ref not in known_species)
    if unknown_family_refs:
        findings.append(
            _finding(
                "fatal",
                "family_label_unknown_family_key",
                "butterfly_family_labels",
                "family label references an unknown family_key",
                {"family_keys": unknown_family_refs},
            )
        )
    if unknown_species_refs:
        findings.append(
            _finding(
                "fatal",
                "species_label_unknown_accepted_taxon_key",
                "butterfly_species_labels",
                "species label references an unknown accepted_taxon_key",
                {"accepted_taxon_keys": unknown_species_refs},
            )
        )


def _append_label_count_findings(findings: list[dict[str, object]], enabled: pl.DataFrame, family_labels: pl.DataFrame, species_labels: pl.DataFrame) -> None:
    family_count = enabled.select("family_key").unique().height if not enabled.is_empty() else 0
    species_count = enabled.select("accepted_taxon_key").unique().height if not enabled.is_empty() else 0
    expected_family_labels = family_count * len(FAMILY_PROMPT_TEMPLATES)
    expected_species_labels = species_count * len(SPECIES_PROMPT_TEMPLATES)
    if family_labels.filter(pl.col("enabled")).height != expected_family_labels:
        findings.append(
            _finding(
                "warning",
                "family_label_count_mismatch",
                "butterfly_family_labels",
                "enabled family label count does not match expected prompt count",
                {"expected": expected_family_labels, "actual": family_labels.filter(pl.col("enabled")).height},
            )
        )
    if species_labels.filter(pl.col("enabled")).height != expected_species_labels:
        findings.append(
            _finding(
                "warning",
                "species_label_count_mismatch",
                "butterfly_species_labels",
                "enabled species label count does not match expected prompt count",
                {"expected": expected_species_labels, "actual": species_labels.filter(pl.col("enabled")).height},
            )
        )


def _append_family_size_findings(findings: list[dict[str, object]], enabled: pl.DataFrame) -> None:
    if enabled.is_empty():
        return
    small = (
        enabled.group_by(["family_key", "family"])
        .len()
        .filter(pl.col("len") < _MIN_SPECIES_PER_FAMILY_WARNING_THRESHOLD)
        .sort(["family", "family_key"])
    )
    if not small.is_empty():
        findings.append(
            _finding(
                "warning",
                "family_has_few_species",
                "butterfly_classification_taxa",
                "one or more families has fewer species than the warning threshold",
                {"families": small.to_dicts(), "threshold": _MIN_SPECIES_PER_FAMILY_WARNING_THRESHOLD},
            )
        )


def _finding(severity: str, code: str, table: str, message: str, details: dict[str, object] | None = None) -> dict[str, object]:
    return {
        "severity": severity,
        "code": code,
        "table": table,
        "message": message,
        "details": dict(details or {}),
    }


def _read_json_optional(path: Path) -> dict[str, object]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _local_file_sizes(paths: dict[str, Path]) -> dict[str, int]:
    return {key: path.stat().st_size for key, path in paths.items() if path.exists() and path.is_file()}


def _bytes_to_mb(value: object) -> float:
    return round(float(value or 0) / (1024 * 1024), 6)


def _stable_json_hash(payload: dict[str, object]) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _first_column_value(frame: pl.DataFrame, column: str) -> object:
    if column not in frame.columns or frame.is_empty():
        return None
    values = frame.select(column).to_series().drop_nulls().to_list()
    return values[0] if values else None


def _text(*values: object) -> str:
    for value in values:
        if value is None:
            continue
        text = " ".join(str(value).strip().split())
        if text:
            return text
    return ""


def _bool_value(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in {"1", "true", "yes", "y"}:
        return True
    if text in {"0", "false", "no", "n"}:
        return False
    return default


def _default_for_dtype(dtype: pl.DataType) -> object:
    if dtype == pl.Boolean:
        return False
    if dtype in {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64}:
        return 0
    if dtype in {pl.Float32, pl.Float64}:
        return 0.0
    return ""
