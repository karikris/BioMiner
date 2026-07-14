"""Fail-closed identity isolation for reference and evaluation artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import json
import re

import polars as pl

from biominer.common.semantic_hash import canonical_semantic_fingerprint


EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION = "evaluation-leakage-register-v1.0.0"
EVALUATION_LEAKAGE_FINDING_SCHEMA_VERSION = "evaluation-leakage-finding-v1.0.0"
EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION = (
    "evaluation-identity-component-v1.0.0"
)
EVALUATION_LEAKAGE_REGISTER_FILE = "evaluation_leakage_register.parquet"
EVALUATION_BOOTSTRAP_COMPONENT_FILE = "evaluation_bootstrap_components.parquet"

BALANCED_CHALLENGE_PARTITION = "balanced_challenge"
NATURAL_STREAM_PARTITION = "natural_stream"
REFERENCE_PARTITIONS = frozenset(
    {
        "support_train",
        "model_selection",
        "calibration",
        "threshold_selection",
        "reference_final_test",
    }
)
EVALUATION_PARTITIONS = frozenset(
    {BALANCED_CHALLENGE_PARTITION, NATURAL_STREAM_PARTITION}
)
LEAKAGE_PARTITIONS = REFERENCE_PARTITIONS | EVALUATION_PARTITIONS

BALANCED_CHALLENGE_ARTIFACT_KIND = "balanced_challenge_holdout"
NATURAL_STREAM_ARTIFACT_KIND = "natural_stream_holdout"

EVALUATION_LEAKAGE_REGISTER_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "register_version": pl.String,
    "register_fingerprint": pl.String,
    "item_fingerprint": pl.String,
    "item_id": pl.String,
    "partition": pl.String,
    "source_artifact_kind": pl.String,
    "source_artifact_fingerprint": pl.String,
    "source": pl.String,
    "source_observation_id": pl.String,
    "visual_content_sha256": pl.String,
    "duplicate_group_id": pl.String,
    "perceptual_duplicate_group_id": pl.String,
    "observer_owner_group_id": pl.String,
    "photographer_id": pl.String,
    "flickr_owner_id": pl.String,
    "provider_mirror_group_id": pl.String,
    "geographic_burst_group_id": pl.String,
}

EVALUATION_LEAKAGE_FINDING_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "finding_fingerprint": pl.String,
    "identity_dimension": pl.String,
    "identity_value": pl.String,
    "partitions": pl.List(pl.String),
    "item_ids": pl.List(pl.String),
}

EVALUATION_IDENTITY_COMPONENT_SCHEMA: dict[str, pl.DataType] = {
    "schema_version": pl.String,
    "register_fingerprint": pl.String,
    "partition": pl.String,
    "bootstrap_component_id": pl.String,
    "component_size": pl.UInt32,
    "item_id": pl.String,
}

_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_OPTIONAL_IDENTITY_FIELDS = (
    "duplicate_group_id",
    "observer_owner_group_id",
    "photographer_id",
    "flickr_owner_id",
    "provider_mirror_group_id",
    "geographic_burst_group_id",
)
_COVERAGE_FIELDS = (
    "visual_content_sha256",
    "perceptual_duplicate_group_id",
    "source_observation_id",
    "photographer_id",
    "flickr_owner_id",
    "provider_mirror_group_id",
    "geographic_burst_group_id",
)


class EvaluationLeakageError(ValueError):
    """Raised when identities cross reference or evaluation partitions."""

    def __init__(self, message: str, findings: pl.DataFrame) -> None:
        super().__init__(message)
        self.findings = findings


@dataclass(frozen=True, slots=True)
class EvaluationLeakageIdentity:
    """One visual item and every identity available for leakage detection."""

    item_id: str
    partition: str
    source_artifact_kind: str
    source_artifact_fingerprint: str
    source: str
    source_observation_id: str
    visual_content_sha256: str
    perceptual_duplicate_group_id: str
    duplicate_group_id: str | None = None
    observer_owner_group_id: str | None = None
    photographer_id: str | None = None
    flickr_owner_id: str | None = None
    provider_mirror_group_id: str | None = None
    geographic_burst_group_id: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "item_id",
            "source_artifact_kind",
            "source_observation_id",
            "perceptual_duplicate_group_id",
        ):
            object.__setattr__(
                self,
                field,
                _required_text(getattr(self, field), field=field),
            )
        partition = _required_text(self.partition, field="partition")
        if partition not in LEAKAGE_PARTITIONS:
            raise ValueError(f"unsupported leakage partition: {partition}")
        object.__setattr__(self, "partition", partition)
        object.__setattr__(
            self,
            "source_artifact_fingerprint",
            _sha256(
                self.source_artifact_fingerprint,
                field="source_artifact_fingerprint",
            ),
        )
        source = _required_text(self.source, field="source").casefold()
        object.__setattr__(self, "source", source)
        object.__setattr__(
            self,
            "visual_content_sha256",
            _sha256(self.visual_content_sha256, field="visual_content_sha256"),
        )
        for field in _OPTIONAL_IDENTITY_FIELDS:
            object.__setattr__(
                self,
                field,
                _optional_text(getattr(self, field), field=field),
            )
        if source.startswith("flickr") and self.flickr_owner_id is None:
            raise ValueError("Flickr leakage identities require flickr_owner_id")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(
            {
                "schema_version": EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION,
                "item": {
                    "item_id": self.item_id,
                    "partition": self.partition,
                    "source_artifact_kind": self.source_artifact_kind,
                    "source_artifact_fingerprint": (self.source_artifact_fingerprint),
                    "source": self.source,
                    "source_observation_id": self.source_observation_id,
                    "visual_content_sha256": self.visual_content_sha256,
                    "perceptual_duplicate_group_id": (
                        self.perceptual_duplicate_group_id
                    ),
                    **{
                        field: getattr(self, field)
                        for field in _OPTIONAL_IDENTITY_FIELDS
                    },
                },
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluationLeakageAudit:
    register_fingerprint: str
    register_item_count: int
    reference_item_count: int
    balanced_challenge_item_count: int
    natural_stream_item_count: int
    coverage_by_dimension: tuple[tuple[str, int], ...]


def empty_evaluation_leakage_register() -> pl.DataFrame:
    return pl.DataFrame(schema=EVALUATION_LEAKAGE_REGISTER_SCHEMA)


def empty_evaluation_leakage_findings() -> pl.DataFrame:
    return pl.DataFrame(schema=EVALUATION_LEAKAGE_FINDING_SCHEMA)


def build_evaluation_leakage_register(
    identities: Sequence[EvaluationLeakageIdentity],
    *,
    register_version: str,
) -> pl.DataFrame:
    """Build a deterministic register and reject cross-partition identities."""

    version = _required_text(register_version, field="register_version")
    normalized = tuple(identities)
    if not normalized:
        raise ValueError("evaluation leakage identities must not be empty")
    if any(not isinstance(item, EvaluationLeakageIdentity) for item in normalized):
        raise TypeError("identities must contain EvaluationLeakageIdentity values")
    ordered = tuple(sorted(normalized, key=lambda item: item.item_id))
    item_ids = [item.item_id for item in ordered]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("evaluation leakage item_id values must be unique")
    rows = [_identity_row(item, register_version=version) for item in ordered]
    register_fingerprint = _register_fingerprint(version, rows)
    for row in rows:
        row["register_fingerprint"] = register_fingerprint
    frame = pl.DataFrame(rows, schema=EVALUATION_LEAKAGE_REGISTER_SCHEMA).sort(
        "item_id"
    )
    validate_evaluation_leakage_register(frame)
    return frame


def build_evaluation_identity_components(register: pl.DataFrame) -> pl.DataFrame:
    """Collapse all within-partition identity links into bootstrap components."""

    if not isinstance(register, pl.DataFrame):
        raise TypeError("register must be a Polars DataFrame")
    normalized = register.sort("item_id")
    validate_evaluation_leakage_register(normalized)
    evaluation = normalized.filter(pl.col("partition").is_in(EVALUATION_PARTITIONS))
    if evaluation.is_empty():
        return pl.DataFrame(schema=EVALUATION_IDENTITY_COMPONENT_SCHEMA)
    register_fingerprint = _single_text(normalized, "register_fingerprint")
    parents = {
        str(item_id): str(item_id) for item_id in evaluation["item_id"].to_list()
    }

    def find(item_id: str) -> str:
        parent = parents[item_id]
        while parent != parents[parent]:
            parent = parents[parent]
        while item_id != parent:
            next_item = parents[item_id]
            parents[item_id] = parent
            item_id = next_item
        return parent

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        first, second = sorted((left_root, right_root))
        parents[second] = first

    token_owner: dict[tuple[str, str, str], str] = {}
    for row in evaluation.iter_rows(named=True):
        item_id = str(row["item_id"])
        partition = str(row["partition"])
        for dimension, identity_value in _identity_tokens(row):
            token = (partition, dimension, identity_value)
            owner = token_owner.setdefault(token, item_id)
            union(item_id, owner)

    members: dict[tuple[str, str], list[str]] = defaultdict(list)
    partition_by_item = {
        str(row["item_id"]): str(row["partition"])
        for row in evaluation.iter_rows(named=True)
    }
    for item_id in sorted(parents):
        members[(partition_by_item[item_id], find(item_id))].append(item_id)
    rows: list[dict[str, object]] = []
    for (partition, _root), item_ids in sorted(members.items()):
        component_id = canonical_semantic_fingerprint(
            {
                "schema_version": EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
                "register_fingerprint": register_fingerprint,
                "partition": partition,
                "item_ids": item_ids,
            }
        )
        rows.extend(
            {
                "schema_version": EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION,
                "register_fingerprint": register_fingerprint,
                "partition": partition,
                "bootstrap_component_id": component_id,
                "component_size": len(item_ids),
                "item_id": item_id,
            }
            for item_id in item_ids
        )
    return pl.DataFrame(
        rows,
        schema=EVALUATION_IDENTITY_COMPONENT_SCHEMA,
        orient="row",
    ).sort("partition", "bootstrap_component_id", "item_id")


def find_evaluation_leakage(register: pl.DataFrame) -> pl.DataFrame:
    """Return deterministic cross-partition identity findings."""

    _validate_register_structure(register)
    assignments: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in register.iter_rows(named=True):
        item_id = str(row["item_id"])
        partition = str(row["partition"])
        for dimension, identity_value in _identity_tokens(row):
            assignments[(dimension, identity_value)][partition].add(item_id)
    rows: list[dict[str, object]] = []
    for (dimension, identity_value), by_partition in sorted(assignments.items()):
        if len(by_partition) <= 1:
            continue
        partitions = sorted(by_partition)
        item_ids = sorted(
            item_id for values in by_partition.values() for item_id in values
        )
        finding_fingerprint = canonical_semantic_fingerprint(
            {
                "schema_version": EVALUATION_LEAKAGE_FINDING_SCHEMA_VERSION,
                "identity_dimension": dimension,
                "identity_value": identity_value,
                "partitions": partitions,
                "item_ids": item_ids,
            }
        )
        rows.append(
            {
                "schema_version": EVALUATION_LEAKAGE_FINDING_SCHEMA_VERSION,
                "finding_fingerprint": finding_fingerprint,
                "identity_dimension": dimension,
                "identity_value": identity_value,
                "partitions": partitions,
                "item_ids": item_ids,
            }
        )
    return pl.DataFrame(
        rows,
        schema=EVALUATION_LEAKAGE_FINDING_SCHEMA,
    ).sort("identity_dimension", "identity_value")


def validate_evaluation_leakage_register(register: pl.DataFrame) -> None:
    """Fail if the register is malformed or any identity crosses partitions."""

    findings = find_evaluation_leakage(register)
    if findings.is_empty():
        return
    details = "; ".join(
        f"{row['identity_dimension']}={row['identity_value']!r} "
        f"partitions={row['partitions']} items={row['item_ids']}"
        for row in findings.head(10).iter_rows(named=True)
    )
    raise EvaluationLeakageError(
        f"reference/evaluation leakage detected: {details}",
        findings,
    )


def validate_reference_and_holdout_leakage(
    register: pl.DataFrame,
    balanced_challenge: pl.DataFrame,
    natural_stream: pl.DataFrame,
) -> EvaluationLeakageAudit:
    """Verify register completeness, artifact identity, and group isolation."""

    from biominer.evaluation.holdouts import (
        validate_evaluation_holdouts_disjoint,
    )

    validate_evaluation_holdouts_disjoint(balanced_challenge, natural_stream)
    validate_evaluation_leakage_register(register)
    reference = register.filter(pl.col("partition").is_in(REFERENCE_PARTITIONS))
    if reference.is_empty():
        raise ValueError(
            "evaluation leakage register must include reference/training items"
        )
    _validate_holdout_register_partition(
        register,
        balanced_challenge,
        partition=BALANCED_CHALLENGE_PARTITION,
        artifact_kind=BALANCED_CHALLENGE_ARTIFACT_KIND,
    )
    _validate_holdout_register_partition(
        register,
        natural_stream,
        partition=NATURAL_STREAM_PARTITION,
        artifact_kind=NATURAL_STREAM_ARTIFACT_KIND,
    )
    coverage = tuple(
        (
            field,
            register.height - register[field].null_count(),
        )
        for field in _COVERAGE_FIELDS
    )
    return EvaluationLeakageAudit(
        register_fingerprint=_single_text(register, "register_fingerprint"),
        register_item_count=register.height,
        reference_item_count=reference.height,
        balanced_challenge_item_count=balanced_challenge.height,
        natural_stream_item_count=natural_stream.height,
        coverage_by_dimension=coverage,
    )


def _validate_holdout_register_partition(
    register: pl.DataFrame,
    holdout: pl.DataFrame,
    *,
    partition: str,
    artifact_kind: str,
) -> None:
    registered = register.filter(pl.col("partition") == partition)
    expected_ids = set(holdout["evaluation_item_id"].to_list())
    registered_ids = set(registered["item_id"].to_list())
    if registered_ids != expected_ids:
        missing = sorted(expected_ids - registered_ids)
        unexpected = sorted(registered_ids - expected_ids)
        raise ValueError(
            f"{partition} leakage identity coverage mismatch: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    holdout_fingerprint = _single_text(holdout, "holdout_fingerprint")
    registered_by_id = {
        str(row["item_id"]): row for row in registered.iter_rows(named=True)
    }
    for holdout_row in holdout.iter_rows(named=True):
        item_id = str(holdout_row["evaluation_item_id"])
        identity = registered_by_id[item_id]
        if identity["source_artifact_kind"] != artifact_kind:
            raise ValueError(
                f"{partition} item {item_id} has the wrong source artifact kind"
            )
        if identity["source_artifact_fingerprint"] != holdout_fingerprint:
            raise ValueError(
                f"{partition} item {item_id} has the wrong holdout fingerprint"
            )
        source = str(holdout_row["source"]).casefold()
        if identity["source"] != source:
            raise ValueError(f"{partition} item {item_id} has a source mismatch")
        if identity["source_observation_id"] != holdout_row["flickr_photo_id"]:
            raise ValueError(f"{partition} item {item_id} has an observation mismatch")
        _match_optional_holdout_identity(
            identity,
            holdout_row,
            register_field="duplicate_group_id",
            holdout_field="duplicate_group_id",
            item_id=item_id,
        )
        _match_optional_holdout_identity(
            identity,
            holdout_row,
            register_field="observer_owner_group_id",
            holdout_field="observer_owner_group_id",
            item_id=item_id,
        )
        crop_hash = holdout_row["crop_hash"]
        if crop_hash is not None:
            if (
                _sha256(crop_hash, field="crop_hash")
                != identity["visual_content_sha256"]
            ):
                raise ValueError(
                    f"{partition} item {item_id} has a content hash mismatch"
                )


def _match_optional_holdout_identity(
    identity: Mapping[str, object],
    holdout_row: Mapping[str, object],
    *,
    register_field: str,
    holdout_field: str,
    item_id: str,
) -> None:
    expected = holdout_row[holdout_field]
    if expected is not None and identity[register_field] != expected:
        raise ValueError(f"evaluation item {item_id} has a {register_field} mismatch")


def _identity_row(
    item: EvaluationLeakageIdentity,
    *,
    register_version: str,
) -> dict[str, object]:
    return {
        "schema_version": EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION,
        "register_version": register_version,
        "register_fingerprint": "sha256:" + "0" * 64,
        "item_fingerprint": item.fingerprint,
        "item_id": item.item_id,
        "partition": item.partition,
        "source_artifact_kind": item.source_artifact_kind,
        "source_artifact_fingerprint": item.source_artifact_fingerprint,
        "source": item.source,
        "source_observation_id": item.source_observation_id,
        "visual_content_sha256": item.visual_content_sha256,
        "duplicate_group_id": item.duplicate_group_id,
        "perceptual_duplicate_group_id": item.perceptual_duplicate_group_id,
        "observer_owner_group_id": item.observer_owner_group_id,
        "photographer_id": item.photographer_id,
        "flickr_owner_id": item.flickr_owner_id,
        "provider_mirror_group_id": item.provider_mirror_group_id,
        "geographic_burst_group_id": item.geographic_burst_group_id,
    }


def _validate_register_structure(register: pl.DataFrame) -> None:
    if not isinstance(register, pl.DataFrame):
        raise TypeError("evaluation leakage register must be a Polars DataFrame")
    if dict(register.schema) != EVALUATION_LEAKAGE_REGISTER_SCHEMA:
        raise ValueError("evaluation leakage register physical schema mismatch")
    if register.is_empty():
        raise ValueError("evaluation leakage register must not be empty")
    if not register.equals(register.sort("item_id")):
        raise ValueError("evaluation leakage register is not sorted by item_id")
    if register["item_id"].n_unique() != register.height:
        raise ValueError("evaluation leakage register item_id values are not unique")
    if (
        _single_text(register, "schema_version")
        != EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION
    ):
        raise ValueError("evaluation leakage register schema is incompatible")
    register_version = _single_text(register, "register_version")
    for row in register.iter_rows(named=True):
        item = EvaluationLeakageIdentity(
            item_id=_required_text(row["item_id"], field="item_id"),
            partition=_required_text(row["partition"], field="partition"),
            source_artifact_kind=_required_text(
                row["source_artifact_kind"],
                field="source_artifact_kind",
            ),
            source_artifact_fingerprint=_required_text(
                row["source_artifact_fingerprint"],
                field="source_artifact_fingerprint",
            ),
            source=_required_text(row["source"], field="source"),
            source_observation_id=_required_text(
                row["source_observation_id"],
                field="source_observation_id",
            ),
            visual_content_sha256=_required_text(
                row["visual_content_sha256"],
                field="visual_content_sha256",
            ),
            perceptual_duplicate_group_id=_required_text(
                row["perceptual_duplicate_group_id"],
                field="perceptual_duplicate_group_id",
            ),
            **{
                field: _optional_row_text(row[field])
                for field in _OPTIONAL_IDENTITY_FIELDS
            },
        )
        if item.fingerprint != row["item_fingerprint"]:
            raise ValueError(f"invalid item_fingerprint for {item.item_id}")
    register_fingerprint = _single_text(register, "register_fingerprint")
    _sha256(register_fingerprint, field="register_fingerprint")
    if register_fingerprint != _register_fingerprint(
        register_version,
        register.iter_rows(named=True),
    ):
        raise ValueError("evaluation leakage register_fingerprint is invalid")


def _register_fingerprint(
    register_version: str,
    rows: Iterable[Mapping[str, object]],
) -> str:
    materialized = list(rows)
    return canonical_semantic_fingerprint(
        {
            "schema_version": EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION,
            "register_version": register_version,
            "items": [
                {
                    "item_id": str(row["item_id"]),
                    "item_fingerprint": str(row["item_fingerprint"]),
                }
                for row in sorted(
                    materialized,
                    key=lambda value: str(value["item_id"]),
                )
            ],
        }
    )


def _identity_tokens(
    row: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    source = str(row["source"])
    tokens = [
        ("exact_hash", str(row["visual_content_sha256"])),
        (
            "perceptual_duplicate_group",
            str(row["perceptual_duplicate_group_id"]),
        ),
        (
            "source_observation",
            _scoped_identity(source, str(row["source_observation_id"])),
        ),
    ]
    optional = (
        ("duplicate_group", "duplicate_group_id", False),
        ("observer_owner", "observer_owner_group_id", True),
        ("photographer", "photographer_id", True),
        ("flickr_owner", "flickr_owner_id", True),
        ("provider_mirror", "provider_mirror_group_id", False),
        ("geographic_burst", "geographic_burst_group_id", True),
    )
    for dimension, field, source_scoped in optional:
        value = row[field]
        if value is None:
            continue
        identity = _scoped_identity(source, str(value)) if source_scoped else str(value)
        tokens.append((dimension, identity))
        if field in {
            "observer_owner_group_id",
            "photographer_id",
            "flickr_owner_id",
        }:
            tokens.append(("person_or_owner", _scoped_identity(source, str(value))))
    return tuple(tokens)


def _scoped_identity(source: str, value: str) -> str:
    return json.dumps([source, value], separators=(",", ":"))


def _single_text(frame: pl.DataFrame, field: str) -> str:
    values = frame[field].unique().to_list()
    if len(values) != 1:
        raise ValueError(f"{field} must have one consistent value")
    return _required_text(values[0], field=field)


def _required_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{field} must be non-empty canonical text")
    return value


def _optional_text(value: object, *, field: str) -> str | None:
    return None if value is None else _required_text(value, field=field)


def _optional_row_text(value: object) -> str | None:
    return None if value is None else str(value)


def _sha256(value: object, *, field: str) -> str:
    text = _required_text(value, field=field)
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise ValueError(f"{field} must be a full lowercase sha256 fingerprint")
    return text


__all__ = [
    "BALANCED_CHALLENGE_ARTIFACT_KIND",
    "BALANCED_CHALLENGE_PARTITION",
    "EVALUATION_BOOTSTRAP_COMPONENT_FILE",
    "EVALUATION_IDENTITY_COMPONENT_SCHEMA",
    "EVALUATION_IDENTITY_COMPONENT_SCHEMA_VERSION",
    "EVALUATION_LEAKAGE_FINDING_SCHEMA",
    "EVALUATION_LEAKAGE_FINDING_SCHEMA_VERSION",
    "EVALUATION_LEAKAGE_REGISTER_FILE",
    "EVALUATION_LEAKAGE_REGISTER_SCHEMA",
    "EVALUATION_LEAKAGE_REGISTER_SCHEMA_VERSION",
    "EVALUATION_PARTITIONS",
    "EvaluationLeakageAudit",
    "EvaluationLeakageError",
    "EvaluationLeakageIdentity",
    "LEAKAGE_PARTITIONS",
    "NATURAL_STREAM_ARTIFACT_KIND",
    "NATURAL_STREAM_PARTITION",
    "REFERENCE_PARTITIONS",
    "build_evaluation_identity_components",
    "build_evaluation_leakage_register",
    "empty_evaluation_leakage_findings",
    "empty_evaluation_leakage_register",
    "find_evaluation_leakage",
    "validate_evaluation_leakage_register",
    "validate_reference_and_holdout_leakage",
]
