"""Reproducible strict-reference performance baseline with explicit gaps."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from biominer.common.semantic_hash import canonical_semantic_fingerprint


REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_SCHEMA_VERSION = (
    "reference-workflow-performance-baseline-v1.0.0"
)
REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_JSON = (
    "strict_workflow_performance_baseline.json"
)
REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_MARKDOWN = (
    "strict_workflow_performance_baseline.md"
)

_METRIC_STATUSES = frozenset({"measured", "derived", "unavailable", "not_instrumented"})
_SHA256_PREFIX = "sha256:"


@dataclass(frozen=True, slots=True)
class BaselineMetric:
    status: str
    unit: str
    value: int | float | None = None
    source: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        status = _required_text(self.status, field="status")
        if status not in _METRIC_STATUSES:
            raise ValueError(f"unsupported baseline metric status: {status}")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "unit", _required_text(self.unit, field="unit"))
        if status in {"measured", "derived"}:
            if (
                isinstance(self.value, bool)
                or not isinstance(self.value, int | float)
                or self.value < 0
            ):
                raise ValueError(
                    f"{status} baseline metric requires a nonnegative value"
                )
            object.__setattr__(
                self, "source", _required_text(self.source, field="source")
            )
            if self.reason is not None:
                raise ValueError(f"{status} baseline metric must not have a reason")
        else:
            if self.value is not None or self.source is not None:
                raise ValueError(
                    f"{status} baseline metric must not have a value or source"
                )
            object.__setattr__(
                self, "reason", _required_text(self.reason, field="reason")
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class ReferenceWorkflowPerformanceBaseline:
    report: dict[str, Any]

    def __post_init__(self) -> None:
        validate_reference_workflow_performance_baseline(self.report)


def build_reference_workflow_performance_baseline(
    phase14_report: Mapping[str, object],
    phase15_verification: Mapping[str, object],
    *,
    source_git_sha: str,
    generated_at: str | datetime,
) -> ReferenceWorkflowPerformanceBaseline:
    """Compile known prototype evidence without inventing a strict live run."""

    _require_schema(
        phase14_report,
        "papilio-demoleus-build-week-prototype-report-v1.0.0",
        artifact="Phase 14 prototype report",
    )
    _require_schema(
        phase15_verification,
        "biominer-phase15-prototype-final-verification-v1.0.0",
        artifact="Phase 15 verification report",
    )
    reference_bank = _required_mapping(phase14_report, "reference_bank")
    staged = _required_mapping(phase14_report, "staged_flickr_inference")
    performance = _required_mapping(staged, "performance")
    resume = _required_mapping(phase15_verification, "resume_and_cache")

    selected_media = _nonnegative_int(reference_bank, "selected_media")
    support_count = _nonnegative_int(reference_bank, "prototype_support")
    human_verified = _nonnegative_int(reference_bank, "human_verified")
    classified = _nonnegative_int(staged, "classified")
    score_rows = _nonnegative_int(staged, "candidate_score_rows")
    awaiting_review = support_count - human_verified
    if awaiting_review < 0:
        raise ValueError("human-verified reference count exceeds support count")

    metrics = {
        "reference_candidates_acquired": _unavailable(
            "rows",
            "committed reports expose selected media but not the total acquired-candidate count",
        ),
        "reference_media_selected": _measured(
            selected_media,
            "rows",
            "phase14.reference_bank.selected_media",
        ),
        "reference_media_downloaded": _unavailable(
            "rows",
            "selected-media count does not prove a complete download-attempt denominator",
        ),
        "provisional_support_frozen": _measured(
            support_count,
            "rows",
            "phase14.reference_bank.prototype_support",
        ),
        "references_awaiting_human_review": _derived(
            awaiting_review,
            "rows",
            "phase14.reference_bank.prototype_support - phase14.reference_bank.human_verified",
        ),
        "manual_reference_reviews_completed": _measured(
            human_verified,
            "rows",
            "phase14.reference_bank.human_verified",
        ),
        "strict_time_blocked_before_readiness": _unavailable(
            "seconds",
            "no committed end-to-end strict reference run records manual-wait start and completion",
        ),
        "strict_time_to_reference_embeddings": _not_instrumented(
            "seconds",
            "the committed prototype report records embedding counts but not elapsed time from run start",
        ),
        "strict_time_to_prototypes": _not_instrumented(
            "seconds",
            "the committed prototype report does not record prototype-ready elapsed time",
        ),
        "strict_time_to_first_flickr_score": _unavailable(
            "seconds",
            "no committed strict run reached a first Flickr score",
        ),
        "flickr_records_scored": _measured(
            classified,
            "rows",
            "phase14.staged_flickr_inference.classified",
        ),
        "flickr_candidate_score_rows": _measured(
            score_rows,
            "rows",
            "phase14.staged_flickr_inference.candidate_score_rows",
        ),
        "bioclip_persistent_model_loads": _measured(
            _nonnegative_int(resume, "bioclip_persistent_model_loads"),
            "loads",
            "phase15.resume_and_cache.bioclip_persistent_model_loads",
        ),
        "bioclip_model_cache_hits": _measured(
            _nonnegative_int(resume, "bioclip_model_cache_hits"),
            "hits",
            "phase15.resume_and_cache.bioclip_model_cache_hits",
        ),
        "reference_embedding_cache_hits": _measured(
            _nonnegative_int(resume, "support_embedding_resume_reused"),
            "rows",
            "phase15.resume_and_cache.support_embedding_resume_reused",
        ),
        "reference_embeddings_recomputed_on_resume": _measured(
            _nonnegative_int(resume, "support_embedding_resume_recomputed"),
            "rows",
            "phase15.resume_and_cache.support_embedding_resume_recomputed",
        ),
        "peak_rss_memory": _measured(
            _nonnegative_int(performance, "rss_peak_memory_bytes"),
            "bytes",
            "phase14.staged_flickr_inference.performance.rss_peak_memory_bytes",
        ),
        "flickr_records_per_second": _measured(
            _nonnegative_number(performance, "records_per_second"),
            "rows_per_second",
            "phase14.staged_flickr_inference.performance.records_per_second",
        ),
        "selective_rerun_records": _unavailable(
            "rows",
            "the baseline predates reference-revision impact analysis and selective rescoring",
        ),
        "full_rerun_work_avoided": _not_instrumented(
            "rows",
            "the current resume report records reuse but not a common work-unit denominator",
        ),
    }
    metric_payload = {key: value.to_dict() for key, value in sorted(metrics.items())}
    report: dict[str, Any] = {
        "schema_version": REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_SCHEMA_VERSION,
        "benchmark_id": "strict-reference-workflow-baseline-20260717",
        "generated_at": _utc_text(generated_at),
        "source_git_sha": _required_text(source_git_sha, field="source_git_sha"),
        "status": "strict_live_baseline_unavailable_with_prototype_proxy_evidence",
        "workflow": "human_verified_strict",
        "strict_live_run_executed": False,
        "comparison_evidence": "committed_prototype_only_reports",
        "metrics": metric_payload,
        "resume": {
            "completed_resume_without_stage_work": _required_bool(
                resume,
                "completed_resume_without_stage_work",
            ),
            "support_embedding_resume_reused": _nonnegative_int(
                resume,
                "support_embedding_resume_reused",
            ),
            "support_embedding_resume_recomputed": _nonnegative_int(
                resume,
                "support_embedding_resume_recomputed",
            ),
        },
        "evidence_sources": {
            "phase14_report": "reports/phase14/papilio_demoleus_build_week_prototype_report.json",
            "phase15_verification": "reports/phase15/prototype_final_verification.json",
        },
        "semantics": {
            "prototype_timings_are_strict_baseline": False,
            "unavailable_values_are_zero": False,
            "raw_scores_are_probabilities": False,
            "provider_supported_is_human_verified": False,
        },
    }
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    return ReferenceWorkflowPerformanceBaseline(report=report)


def validate_reference_workflow_performance_baseline(
    report: Mapping[str, object],
) -> None:
    if (
        report.get("schema_version")
        != REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_SCHEMA_VERSION
    ):
        raise ValueError("unsupported reference workflow performance baseline schema")
    if report.get("strict_live_run_executed") is not False:
        raise ValueError("strict baseline must not claim an unrecorded live run")
    if report.get("workflow") != "human_verified_strict":
        raise ValueError("strict baseline workflow is invalid")
    metrics = _required_mapping(report, "metrics")
    required_metrics = {
        "reference_candidates_acquired",
        "reference_media_downloaded",
        "references_awaiting_human_review",
        "strict_time_blocked_before_readiness",
        "strict_time_to_reference_embeddings",
        "strict_time_to_prototypes",
        "strict_time_to_first_flickr_score",
        "manual_reference_reviews_completed",
        "bioclip_persistent_model_loads",
        "reference_embedding_cache_hits",
        "peak_rss_memory",
        "selective_rerun_records",
        "full_rerun_work_avoided",
    }
    missing = required_metrics - set(metrics)
    if missing:
        raise ValueError(f"strict baseline omits required metrics: {sorted(missing)}")
    for name, raw in metrics.items():
        if not isinstance(raw, Mapping):
            raise ValueError(f"baseline metric {name} must be an object")
        BaselineMetric(
            status=raw.get("status"),  # type: ignore[arg-type]
            value=raw.get("value"),  # type: ignore[arg-type]
            unit=raw.get("unit"),  # type: ignore[arg-type]
            source=raw.get("source"),  # type: ignore[arg-type]
            reason=raw.get("reason"),  # type: ignore[arg-type]
        )
    fingerprint = _required_text(
        report.get("report_fingerprint"), field="report_fingerprint"
    )
    if not fingerprint.startswith(_SHA256_PREFIX):
        raise ValueError("reference workflow performance fingerprint is invalid")
    unsigned = {
        key: value for key, value in report.items() if key != "report_fingerprint"
    }
    if fingerprint != canonical_semantic_fingerprint(unsigned):
        raise ValueError("reference workflow performance fingerprint mismatch")


def reference_workflow_performance_markdown(report: Mapping[str, object]) -> str:
    validate_reference_workflow_performance_baseline(report)
    metrics = _required_mapping(report, "metrics")
    lines = [
        "# Strict reference workflow performance baseline",
        "",
        f"- Status: `{report['status']}`",
        f"- Source SHA: `{report['source_git_sha']}`",
        f"- Generated: `{report['generated_at']}`",
        f"- Strict live run executed: `{str(report['strict_live_run_executed']).lower()}`",
        f"- Fingerprint: `{report['report_fingerprint']}`",
        "",
        "The committed evidence is a prototype-only proxy. It does not provide a strict",
        "time-to-first-score result. Missing values remain unavailable or not instrumented.",
        "",
        "| Metric | Status | Value | Unit | Evidence or reason |",
        "|---|---|---:|---|---|",
    ]
    for name, raw in sorted(metrics.items()):
        assert isinstance(raw, Mapping)
        value = raw.get("value")
        evidence = raw.get("source") or raw.get("reason")
        lines.append(
            f"| `{name}` | `{raw.get('status')}` | "
            f"{value if value is not None else '—'} | `{raw.get('unit')}` | {evidence} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Prototype evidence proves persistent model and embedding reuse mechanisms.",
            "- It does not measure the strict manual-review wait or strict time to first score.",
            "- Adaptive and strict paired benchmarks must use this same metric contract.",
            "- Unavailable values are not zero and must not be used in speedup calculations.",
            "",
        ]
    )
    return "\n".join(lines)


def write_reference_workflow_performance_baseline(
    baseline: ReferenceWorkflowPerformanceBaseline,
    output_dir: str | Path,
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_JSON
    markdown_path = output / REFERENCE_WORKFLOW_PERFORMANCE_BASELINE_MARKDOWN
    _atomic_write_text(
        json_path,
        json.dumps(baseline.report, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(
        markdown_path,
        reference_workflow_performance_markdown(baseline.report),
    )
    return {"json": json_path, "markdown": markdown_path}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase14-report", type=Path, required=True)
    parser.add_argument("--phase15-verification", type=Path, required=True)
    parser.add_argument("--source-git-sha", required=True)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    baseline = build_reference_workflow_performance_baseline(
        _read_json(args.phase14_report),
        _read_json(args.phase15_verification),
        source_git_sha=args.source_git_sha,
        generated_at=args.generated_at,
    )
    paths = write_reference_workflow_performance_baseline(baseline, args.output_dir)
    print(json.dumps({key: str(value) for key, value in sorted(paths.items())}))
    return 0


def _measured(value: int | float, unit: str, source: str) -> BaselineMetric:
    return BaselineMetric(status="measured", value=value, unit=unit, source=source)


def _derived(value: int | float, unit: str, source: str) -> BaselineMetric:
    return BaselineMetric(status="derived", value=value, unit=unit, source=source)


def _unavailable(unit: str, reason: str) -> BaselineMetric:
    return BaselineMetric(status="unavailable", unit=unit, reason=reason)


def _not_instrumented(unit: str, reason: str) -> BaselineMetric:
    return BaselineMetric(status="not_instrumented", unit=unit, reason=reason)


def _require_schema(
    report: Mapping[str, object],
    expected: str,
    *,
    artifact: str,
) -> None:
    if report.get("schema_version") != expected:
        raise ValueError(f"unsupported {artifact} schema")


def _required_mapping(value: Mapping[str, object], field: str) -> Mapping[str, object]:
    result = value.get(field)
    if not isinstance(result, Mapping):
        raise ValueError(f"{field} must be an object")
    return result


def _nonnegative_int(value: Mapping[str, object], field: str) -> int:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError(f"{field} must be a nonnegative integer")
    return result


def _nonnegative_number(value: Mapping[str, object], field: str) -> int | float:
    result = value.get(field)
    if isinstance(result, bool) or not isinstance(result, int | float) or result < 0:
        raise ValueError(f"{field} must be a nonnegative number")
    return result


def _required_bool(value: Mapping[str, object], field: str) -> bool:
    result = value.get(field)
    if not isinstance(result, bool):
        raise ValueError(f"{field} must be Boolean")
    return result


def _required_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be nonblank")
    return result


def _utc_text(value: str | datetime) -> str:
    parsed = (
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, str)
        else value
    )
    if parsed.tzinfo is None or parsed.utcoffset() != UTC.utcoffset(parsed):
        raise ValueError("generated_at must be an aware UTC timestamp")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON must contain an object: {path}")
    return payload


def _atomic_write_text(path: Path, value: str) -> None:
    with NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    os.replace(temporary, path)


if __name__ == "__main__":
    raise SystemExit(main())
