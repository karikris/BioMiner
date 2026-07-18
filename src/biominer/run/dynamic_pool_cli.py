"""Plan-first CLI contracts for the dynamic-pooling workflow."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
from pathlib import Path
from urllib.parse import urlparse

from biominer.bioclip.dynamic_pool_fusion import RAW_FUSION_METHODS
from biominer.candidates.strategy_ablation import CANDIDATE_STRATEGIES
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.run.dynamic_pool_config import load_dynamic_pooling_settings
from biominer.run.stages import RunStage
from biominer.storage.uri import join_uri


DYNAMIC_POOL_COMMAND_PLAN_VERSION = "dynamic-pool-command-plan-v1.0.0"


@dataclass(frozen=True, slots=True)
class DynamicPoolOperationSpec:
    """The exact artifact boundary for one public workflow command."""

    name: str
    stages: tuple[RunStage, ...]
    required_inputs: tuple[str, ...]
    intended_outputs: tuple[str, ...]
    required_selections: tuple[str, ...] = ()


DYNAMIC_POOL_OPERATION_SPECS: Mapping[str, DynamicPoolOperationSpec] = {
    spec.name: spec
    for spec in (
        DynamicPoolOperationSpec(
            name="build-reference-geography-index",
            stages=(RunStage.REFERENCE_GEOGRAPHY_INDEX,),
            required_inputs=(
                "normalized_reference_geography",
                "reference_embeddings",
                "reference_support_manifest",
            ),
            intended_outputs=(
                "reference_geography_index.jsonl",
                "reference_geography_index_manifest.json",
            ),
        ),
        DynamicPoolOperationSpec(
            name="plan-pools",
            stages=(
                RunStage.FLICKR_GEO_TAXON_PARTITIONING,
                RunStage.FAMILY_ROUTING,
                RunStage.DYNAMIC_POOL_PLANNING,
            ),
            required_inputs=(
                "candidate_plans",
                "flickr_partitions",
                "reference_geography_index",
            ),
            intended_outputs=(
                "dynamic_pool_plans.jsonl",
                "dynamic_pool_plan_manifest.json",
            ),
            required_selections=("candidate_strategy",),
        ),
        DynamicPoolOperationSpec(
            name="score-pools",
            stages=(RunStage.DYNAMIC_POOL_SCORING,),
            required_inputs=(
                "dynamic_pool_plans",
                "flickr_embeddings",
                "reference_embeddings",
            ),
            intended_outputs=(
                "dynamic_pool_scores.parquet",
                "dynamic_pool_score_manifest.json",
            ),
            required_selections=("fusion_method",),
        ),
        DynamicPoolOperationSpec(
            name="build-review-sample",
            stages=(RunStage.REVIEW_SAMPLE_PLANNING,),
            required_inputs=("dynamic_pool_scores",),
            intended_outputs=(
                "representative_review_sample.parquet",
                "targeted_review_queue.parquet",
                "review_sample_manifest.json",
            ),
        ),
        DynamicPoolOperationSpec(
            name="audit-quality",
            stages=(RunStage.RISK_CONTROLLED_AUDIT,),
            required_inputs=("review_sample", "reviewed_labels"),
            intended_outputs=(
                "dynamic_pool_quality_audit.json",
                "dynamic_pool_quality_audit_summary.md",
            ),
        ),
        DynamicPoolOperationSpec(
            name="selective-rerun",
            stages=(
                RunStage.AFFECTED_REFERENCE_REBUILD,
                RunStage.AFFECTED_RECORD_RESCORE,
            ),
            required_inputs=(
                "flickr_embedding_cache",
                "matrix_dependencies",
                "pool_dependencies",
                "reference_embedding_cache",
                "reference_revision",
                "scoring_dependencies",
            ),
            intended_outputs=(
                "selective_rerun_plan.json",
                "selective_rerun_receipt.json",
            ),
        ),
        DynamicPoolOperationSpec(
            name="export-handoff",
            stages=(RunStage.FINAL_QUALITY_GATE,),
            required_inputs=(
                "dynamic_pool_plans",
                "dynamic_pool_scores",
                "quality_report",
                "review_queue",
            ),
            intended_outputs=(
                "dynamic_pool_handoff_manifest.json",
                "dynamic_pool_handoff_summary.md",
            ),
            required_selections=("candidate_strategy", "fusion_method"),
        ),
    )
}


@dataclass(frozen=True, slots=True)
class DynamicPoolCommandPlan:
    """A deterministic description of work; never scientific authority."""

    operation: str
    stages: tuple[str, ...]
    settings_fingerprint: str
    settings_selection_status: str
    candidate_strategy: str | None
    fusion_method: str | None
    input_bindings: tuple[tuple[str, str], ...]
    output_root: str
    output_bindings: tuple[tuple[str, str], ...]
    execution_mode: str = "dry_run"
    schema_version: str = DYNAMIC_POOL_COMMAND_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != DYNAMIC_POOL_COMMAND_PLAN_VERSION:
            raise ValueError("unsupported dynamic-pool command plan version")
        try:
            spec = DYNAMIC_POOL_OPERATION_SPECS[self.operation]
        except KeyError as exc:
            raise ValueError(
                f"unsupported dynamic-pooling operation: {self.operation!r}"
            ) from exc
        expected_stages = tuple(stage.value for stage in spec.stages)
        if self.stages != expected_stages:
            raise ValueError("dynamic-pooling plan stages do not match operation")
        _sha256(self.settings_fingerprint, field="settings_fingerprint")
        if self.candidate_strategy is not None:
            if self.candidate_strategy not in CANDIDATE_STRATEGIES:
                raise ValueError(
                    "dynamic-pooling plan candidate strategy is unsupported"
                )
        if self.fusion_method is not None:
            if self.fusion_method not in RAW_FUSION_METHODS:
                raise ValueError("dynamic-pooling plan fusion method is unsupported")
        if self.settings_selection_status != _selection_status(
            self.candidate_strategy,
            self.fusion_method,
        ):
            raise ValueError("dynamic-pooling plan selection status mismatch")
        inputs = dict(self.input_bindings)
        if len(inputs) != len(self.input_bindings):
            raise ValueError("dynamic-pooling plan input names must be unique")
        if tuple(sorted(inputs.items())) != self.input_bindings:
            raise ValueError("dynamic-pooling plan inputs must use canonical order")
        if set(inputs) != set(spec.required_inputs):
            raise ValueError("dynamic-pooling plan inputs do not match operation")
        for name, uri in inputs.items():
            _artifact_uri(uri, field=f"input {name!r} URI")
        output_root = _artifact_uri(self.output_root, field="output_root")
        expected_outputs = tuple(
            (name, join_uri(output_root, name)) for name in spec.intended_outputs
        )
        if self.output_bindings != expected_outputs:
            raise ValueError("dynamic-pooling plan outputs do not match operation")
        if self.execution_mode != "dry_run":
            raise ValueError("dynamic-pooling plan execution mode must be dry_run")

    @property
    def fingerprint(self) -> str:
        return canonical_semantic_fingerprint(self.identity_payload())

    def identity_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "operation": self.operation,
            "stages": list(self.stages),
            "settings_fingerprint": self.settings_fingerprint,
            "settings_selection_status": self.settings_selection_status,
            "candidate_strategy": self.candidate_strategy,
            "fusion_method": self.fusion_method,
            "input_bindings": dict(self.input_bindings),
            "output_root": self.output_root,
            "output_bindings": dict(self.output_bindings),
            "execution_mode": self.execution_mode,
            "adapter_status": "not_connected",
            "structural_validation_status": "valid",
            "selection_requirements": list(self.selection_requirements),
            "selection_requirements_satisfied": not self.selection_validation_issues,
            "selection_validation_issues": list(self.selection_validation_issues),
            "scientific_authority": {
                "calibration": False,
                "human_verification": False,
                "occurrence_release": False,
                "statistical_support": False,
            },
        }

    def to_dict(self) -> dict[str, object]:
        return {
            **self.identity_payload(),
            "plan_fingerprint": self.fingerprint,
        }

    @property
    def selection_requirements(self) -> tuple[str, ...]:
        return DYNAMIC_POOL_OPERATION_SPECS[self.operation].required_selections

    @property
    def selection_validation_issues(self) -> tuple[str, ...]:
        return tuple(
            f"{name} is unselected"
            for name in self.selection_requirements
            if getattr(self, name) is None
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> DynamicPoolCommandPlan:
        if not isinstance(value, Mapping):
            raise TypeError("dynamic-pooling command plan must be a mapping")
        required_fields = {
            "schema_version",
            "operation",
            "stages",
            "settings_fingerprint",
            "settings_selection_status",
            "candidate_strategy",
            "fusion_method",
            "input_bindings",
            "output_root",
            "output_bindings",
            "execution_mode",
            "adapter_status",
            "structural_validation_status",
            "selection_requirements",
            "selection_requirements_satisfied",
            "selection_validation_issues",
            "scientific_authority",
            "plan_fingerprint",
        }
        if set(value) != required_fields:
            raise ValueError("dynamic-pooling command plan fields do not match")
        plan = cls(
            schema_version=_nonblank(value["schema_version"], field="schema_version"),
            operation=_nonblank(value["operation"], field="operation"),
            stages=_text_sequence(value["stages"], field="stages"),
            settings_fingerprint=_nonblank(
                value["settings_fingerprint"], field="settings_fingerprint"
            ),
            settings_selection_status=_nonblank(
                value["settings_selection_status"],
                field="settings_selection_status",
            ),
            candidate_strategy=_optional_text(value["candidate_strategy"]),
            fusion_method=_optional_text(value["fusion_method"]),
            input_bindings=_binding_items(
                value["input_bindings"], field="input_bindings"
            ),
            output_root=_nonblank(value["output_root"], field="output_root"),
            output_bindings=_binding_items(
                value["output_bindings"], field="output_bindings", sort=False
            ),
            execution_mode=_nonblank(value["execution_mode"], field="execution_mode"),
        )
        if value["plan_fingerprint"] != plan.fingerprint:
            raise ValueError("dynamic-pooling command plan fingerprint mismatch")
        if dict(value) != plan.to_dict():
            raise ValueError("dynamic-pooling command plan derived fields mismatch")
        return plan


def add_dynamic_pooling_parsers(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> None:
    """Add the seven stable dynamic-pooling operation parsers."""

    for spec in DYNAMIC_POOL_OPERATION_SPECS.values():
        parser = subparsers.add_parser(spec.name)
        parser.add_argument("--settings", required=True)
        parser.add_argument(
            "--input",
            action="append",
            default=[],
            metavar="NAME=URI",
            help="bind one exact named input artifact; repeat for every required input",
        )
        parser.add_argument("--output-root", required=True)
        parser.add_argument("--plan-output")
        parser.add_argument("--dry-run", action="store_true")


def build_dynamic_pool_command_plan(args: argparse.Namespace) -> DynamicPoolCommandPlan:
    """Resolve one CLI namespace to an immutable artifact plan."""

    operation = str(getattr(args, "dynamic_pooling_command", "") or "").strip()
    try:
        spec = DYNAMIC_POOL_OPERATION_SPECS[operation]
    except KeyError as exc:
        raise ValueError(
            f"unsupported dynamic-pooling operation: {operation!r}"
        ) from exc
    settings = load_dynamic_pooling_settings(args.settings)
    inputs = _parse_input_bindings(args.input, required=spec.required_inputs)
    output_root = _artifact_uri(args.output_root, field="output_root")
    outputs = tuple(
        (name, join_uri(output_root, name)) for name in spec.intended_outputs
    )
    return DynamicPoolCommandPlan(
        operation=operation,
        stages=tuple(stage.value for stage in spec.stages),
        settings_fingerprint=settings.fingerprint,
        settings_selection_status=settings.selection_status,
        candidate_strategy=settings.candidate_strategy,
        fusion_method=settings.fusion_method,
        input_bindings=tuple(sorted(inputs.items())),
        output_root=output_root,
        output_bindings=outputs,
    )


def run_dynamic_pooling_command(args: argparse.Namespace) -> int:
    """Print or persist a validated plan; live adapters fail closed."""

    try:
        plan = build_dynamic_pool_command_plan(args)
        if not args.dry_run:
            raise ValueError(
                "dynamic-pooling production adapters are not connected; "
                "use --dry-run to inspect the validated execution plan"
            )
        plan_output = _write_plan(plan, args.plan_output) if args.plan_output else None
    except (FileNotFoundError, TypeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
    payload = plan.to_dict()
    payload["plan_output"] = str(plan_output) if plan_output is not None else None
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _parse_input_bindings(
    values: Sequence[str],
    *,
    required: Sequence[str],
) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for raw in values:
        if not isinstance(raw, str) or "=" not in raw:
            raise ValueError("each --input must use NAME=URI")
        name, uri = raw.split("=", 1)
        name = _nonblank(name, field="input name")
        uri = _artifact_uri(uri, field=f"input {name!r} URI")
        if name in bindings:
            raise ValueError(f"duplicate input binding: {name}")
        bindings[name] = uri
    required_names = set(required)
    provided_names = set(bindings)
    missing = sorted(required_names - provided_names)
    extra = sorted(provided_names - required_names)
    if missing or extra:
        raise ValueError(
            "dynamic-pooling input bindings do not match operation contract: "
            f"missing={missing}, extra={extra}"
        )
    return bindings


def _write_plan(plan: DynamicPoolCommandPlan, output: str) -> Path:
    destination = Path(_nonblank(output, field="plan_output"))
    if destination.suffix.casefold() != ".json":
        raise ValueError("plan_output must be a JSON file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_dynamic_pool_command_plan(path: str | Path) -> DynamicPoolCommandPlan:
    source = Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("dynamic-pooling command plan JSON is invalid") from exc
    if not isinstance(value, Mapping):
        raise ValueError("dynamic-pooling command plan JSON must contain an object")
    return DynamicPoolCommandPlan.from_mapping(value)


def _artifact_uri(value: object, *, field: str) -> str:
    uri = _nonblank(value, field=field)
    if any(ord(character) < 32 for character in uri):
        raise ValueError(f"{field} must not contain control characters")
    parsed = urlparse(uri)
    if parsed.scheme not in {"", "file", "s3"}:
        raise ValueError(f"{field} must be a local, file, or s3 artifact URI")
    if parsed.scheme == "s3" and (not parsed.netloc or not parsed.path.strip("/")):
        raise ValueError(f"{field} s3 URI must include a bucket and artifact path")
    if parsed.scheme == "file" and parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"{field} file URI host is unsupported")
    return uri


def _binding_items(
    value: object,
    *,
    field: str,
    sort: bool = True,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field} must be a mapping")
    items = tuple(
        (
            _nonblank(name, field=f"{field} name"),
            _nonblank(uri, field=f"{field} URI"),
        )
        for name, uri in value.items()
    )
    return tuple(sorted(items)) if sort else items


def _text_sequence(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return tuple(_nonblank(item, field=field) for item in value)


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return _nonblank(value, field="optional value")


def _selection_status(candidate_strategy: str | None, fusion_method: str | None) -> str:
    selected_count = sum(
        value is not None for value in (candidate_strategy, fusion_method)
    )
    if selected_count == 0:
        return "unselected"
    if selected_count == 1:
        return "partially_selected_with_bound_evidence"
    return "selected_with_bound_evidence"


def _sha256(value: object, *, field: str) -> str:
    text = _nonblank(value, field=field).casefold()
    if not text.startswith("sha256:") or len(text) != 71:
        raise ValueError(f"{field} must be a sha256 fingerprint")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ValueError(f"{field} must be a sha256 fingerprint") from exc
    return text


def _nonblank(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be nonblank text")
    return value.strip()


__all__ = [
    "DYNAMIC_POOL_COMMAND_PLAN_VERSION",
    "DYNAMIC_POOL_OPERATION_SPECS",
    "DynamicPoolCommandPlan",
    "DynamicPoolOperationSpec",
    "add_dynamic_pooling_parsers",
    "build_dynamic_pool_command_plan",
    "load_dynamic_pool_command_plan",
    "run_dynamic_pooling_command",
]
