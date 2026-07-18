"""Cached-vector global/local scoring execution for the bounded pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from math import fsum, isclose, sqrt
from pathlib import Path
import re

import polars as pl

from biominer.bioclip.dynamic_pool_compute import (
    DynamicVectorScoringWork,
    PoolMatrixBatchPolicy,
    PoolMatrixBatchResult,
    build_dynamic_vector_scoring_work,
    execute_dynamic_vector_scoring_batches,
    validate_pool_matrix_batch_result,
)
from biominer.bioclip.dynamic_pool_fusion import (
    FUSION_COMPONENTS,
    GLOBAL_FUSION_COMPONENTS,
    RAW_FUSION_METHODS,
    ValidationLinearFusionParameters,
)
from biominer.bioclip.dynamic_pool_scoring import (
    GlobalReferencePoolInput,
    LocalReferencePoolInput,
)
from biominer.bioclip.matrix_cache import (
    CandidatePrototypeVector,
    DynamicPoolMatrixCache,
    DynamicPoolMatrixCacheMetrics,
    FamilyPrototypeMatrixCache,
    FamilyPrototypeVector,
    MatrixCacheMetrics,
    PoolReferenceVector,
)
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.evaluation.dynamic_pool_pilot_ablation import (
    build_dynamic_pool_pilot_candidate_ablation,
    validate_dynamic_pool_pilot_candidate_ablation,
)
from biominer.evaluation.dynamic_pool_pilot_plan import (
    PILOT_CANDIDATE_STRATEGIES,
    PILOT_CASE_EVIDENCE_BASIS,
    PILOT_POOL_VARIANTS,
    validate_dynamic_pool_pilot_plan,
)
from biominer.vision.full_frame_attention import RAW_FULL_IMAGE_KIND
from biominer.vision.target_full_frame import RawFullFrameEmbedding


DYNAMIC_POOL_PILOT_SCORING_VERSION = "dynamic-pool-pilot-scoring-v1.0.0"
DYNAMIC_POOL_PILOT_SCORING_REPORT_VERSION = "dynamic-pool-pilot-scoring-report-v1.0.0"
DYNAMIC_POOL_PILOT_SCORING_REPORT_FILE = "dynamic_pool_scoring.json"
PILOT_FIXTURE_VECTOR_DIMENSION = 8
PILOT_SCORING_ROUTE = "adult_field"

_SORT = ("case_id", "candidate_strategy", "pool_variant", "fusion_method")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class DynamicPoolPilotScoringExecution:
    """One deterministic batch execution and its observed reuse evidence."""

    works: tuple[DynamicVectorScoringWork, ...]
    batch_result: PoolMatrixBatchResult
    family_cache_metrics: MatrixCacheMetrics
    dynamic_cache_metrics: DynamicPoolMatrixCacheMetrics
    results: pl.DataFrame
    execution_fingerprint: str


def dynamic_pool_pilot_scoring_schema() -> dict[str, pl.DataType]:
    """Return the canonical 24-variant-per-case result projection."""

    return {
        "schema_version": pl.String,
        "pilot_id": pl.String,
        "plan_fingerprint": pl.String,
        "case_id": pl.String,
        "fixture_media_id": pl.String,
        "country_code": pl.String,
        "region_id": pl.String,
        "no_geo": pl.Boolean,
        "candidate_strategy": pl.String,
        "pool_variant": pl.String,
        "fusion_method": pl.String,
        "source_work_fingerprint": pl.String,
        "source_result_fingerprint": pl.String,
        "query_embedding_id": pl.String,
        "query_embedding_fingerprint": pl.String,
        "candidate_matrix_signature": pl.String,
        "pool_matrix_set_fingerprint": pl.String,
        "fusion_score_set_fingerprint": pl.String,
        "ranking_fingerprint": pl.String,
        "target_accepted_taxon_key": pl.String,
        "top_candidate_accepted_taxon_key": pl.String,
        "target_rank": pl.UInt32,
        "top_raw_fusion_score": pl.Float64,
        "target_raw_fusion_score": pl.Float64,
        "top_margin_raw": pl.Float64,
        "local_evidence_status": pl.String,
        "fixture_expected_target_at_1": pl.Boolean,
        "expected_label_basis": pl.String,
        "score_work_reused_across_candidate_strategies": pl.Boolean,
        "encoder_invocations": pl.UInt32,
        "image_materializations": pl.UInt32,
        "raw_score_is_probability": pl.Boolean,
        "probability_availability": pl.String,
        "human_review_status": pl.String,
        "production_default_eligible": pl.Boolean,
        "result_fingerprint": pl.String,
    }


def execute_dynamic_pool_pilot_scoring(
    plan: Mapping[str, object],
) -> DynamicPoolPilotScoringExecution:
    """Score global and global/local fixture pools without images or an encoder."""

    validate_dynamic_pool_pilot_plan(plan)
    candidate_ablation = build_dynamic_pool_pilot_candidate_ablation(plan)
    validate_dynamic_pool_pilot_candidate_ablation(candidate_ablation, plan)
    candidate_sets = {
        str(case_id): str(group["candidate_set_fingerprint"][0])
        for (case_id,), group in candidate_ablation.group_by("case_id")
    }
    catalog = {
        str(taxon["accepted_taxon_key"]): taxon for taxon in plan["taxon_catalog"]
    }
    family_cache = FamilyPrototypeMatrixCache()
    dynamic_cache = DynamicPoolMatrixCache()
    linear_parameters = _fixture_linear_parameters(plan)
    model_fingerprint = _fixture_model_fingerprint(plan)
    work_context: dict[str, tuple[Mapping[str, object], str]] = {}
    works: list[DynamicVectorScoringWork] = []
    for case in plan["cases"]:
        source_embedding = _fixture_query_embedding(
            plan=plan,
            case=case,
            catalog_keys=list(catalog),
            model_fingerprint=model_fingerprint,
        )
        for pool_variant in PILOT_POOL_VARIANTS:
            family_matrix = family_cache.get_or_build(
                route=PILOT_SCORING_ROUTE,
                visual_input_kind=RAW_FULL_IMAGE_KIND,
                family_partition="pilot-papilionidae",
                model_fingerprint=model_fingerprint,
                family_prototype_set_fingerprint=_family_source_fingerprint(plan),
                prototypes=_family_prototypes(plan),
            )
            candidate_matrix = dynamic_cache.get_candidate_matrix(
                route=PILOT_SCORING_ROUTE,
                visual_input_kind=RAW_FULL_IMAGE_KIND,
                family_partition="pilot-papilionidae",
                model_fingerprint=model_fingerprint,
                candidate_set_fingerprint=candidate_sets[str(case["case_id"])],
                reference_prototype_artifact_fingerprint=(
                    _candidate_prototype_source_fingerprint(plan)
                ),
                candidates=_candidate_prototypes(plan),
            )
            global_pools = _global_pool_inputs(
                plan=plan,
                catalog=catalog,
                cache=dynamic_cache,
                model_fingerprint=model_fingerprint,
            )
            local_pools = _local_pool_inputs(
                plan=plan,
                case=case,
                pool_variant=pool_variant,
                catalog=catalog,
                cache=dynamic_cache,
                model_fingerprint=model_fingerprint,
            )
            work = build_dynamic_vector_scoring_work(
                source_embedding,
                query_id=f"pilot-score-query:{case['case_id']}:{pool_variant}",
                route=PILOT_SCORING_ROUTE,
                family_matrix=family_matrix,
                candidate_matrix=candidate_matrix,
                global_pools=global_pools,
                local_pools=local_pools,
                linear_parameters=linear_parameters,
            )
            works.append(work)
            work_context[work.work_fingerprint] = (case, pool_variant)

    batch_result = execute_dynamic_vector_scoring_batches(
        works,
        policy=PoolMatrixBatchPolicy(
            maximum_work_items_per_batch=64,
            maximum_unique_pool_matrices_per_batch=256,
            maximum_pool_matrix_bytes_per_batch=int(
                plan["execution_limits"]["maximum_matrix_cache_bytes"]
            ),
        ),
    )
    result_rows = _project_scoring_results(
        plan=plan,
        batch_result=batch_result,
        work_context=work_context,
    )
    execution = DynamicPoolPilotScoringExecution(
        works=tuple(works),
        batch_result=batch_result,
        family_cache_metrics=family_cache.cache_metrics(),
        dynamic_cache_metrics=dynamic_cache.cache_metrics(),
        results=result_rows,
        execution_fingerprint="",
    )
    execution = DynamicPoolPilotScoringExecution(
        works=execution.works,
        batch_result=execution.batch_result,
        family_cache_metrics=execution.family_cache_metrics,
        dynamic_cache_metrics=execution.dynamic_cache_metrics,
        results=execution.results,
        execution_fingerprint=canonical_semantic_fingerprint(
            _execution_identity(execution)
        ),
    )
    validate_dynamic_pool_pilot_scoring_execution(execution, plan)
    return execution


def validate_dynamic_pool_pilot_scoring_execution(
    execution: DynamicPoolPilotScoringExecution,
    plan: Mapping[str, object],
) -> None:
    """Validate batch, cache, projection, and no-encoder evidence together."""

    validate_dynamic_pool_pilot_plan(plan)
    if not isinstance(execution, DynamicPoolPilotScoringExecution):
        raise TypeError("pilot scoring execution has the wrong type")
    if len(execution.works) != len(plan["cases"]) * len(PILOT_POOL_VARIANTS):
        raise ValueError("pilot scoring work coverage is incomplete")
    if len({work.work_fingerprint for work in execution.works}) != len(execution.works):
        raise ValueError("pilot scoring work identities are not unique")
    validate_pool_matrix_batch_result(execution.batch_result)
    if execution.batch_result.metrics.encoder_invocations != 0:
        raise ValueError("pilot scoring crossed the encoder-free boundary")
    if execution.batch_result.metrics.image_materializations != 0:
        raise ValueError("pilot scoring materialized an image")
    _validate_cache_metrics(execution.family_cache_metrics)
    _validate_cache_metrics(execution.dynamic_cache_metrics.candidate)
    _validate_cache_metrics(execution.dynamic_cache_metrics.pool)
    validate_dynamic_pool_pilot_scoring_results(execution.results, plan)
    expected = canonical_semantic_fingerprint(_execution_identity(execution))
    if execution.execution_fingerprint != expected:
        raise ValueError("pilot scoring execution fingerprint differs")


def validate_dynamic_pool_pilot_scoring_results(
    frame: pl.DataFrame,
    plan: Mapping[str, object],
) -> None:
    """Validate all case/strategy/pool/fusion projections and maturity fields."""

    validate_dynamic_pool_pilot_plan(plan)
    if not isinstance(frame, pl.DataFrame):
        raise TypeError("pilot scoring results must be a Polars DataFrame")
    if frame.schema != dynamic_pool_pilot_scoring_schema():
        raise ValueError("pilot scoring result schema mismatch")
    if not frame.equals(frame.sort(*_SORT)):
        raise ValueError("pilot scoring results are not canonically sorted")
    expected_rows = (
        len(plan["cases"])
        * len(PILOT_CANDIDATE_STRATEGIES)
        * len(PILOT_POOL_VARIANTS)
        * len(RAW_FUSION_METHODS)
    )
    if frame.height != expected_rows:
        raise ValueError("pilot scoring result coverage is incomplete")
    if frame.select(*_SORT).n_unique() != frame.height:
        raise ValueError("pilot scoring result grain is not unique")
    if set(frame["case_id"]) != {str(case["case_id"]) for case in plan["cases"]}:
        raise ValueError("pilot scoring result cases differ from the plan")
    if set(frame["candidate_strategy"]) != set(PILOT_CANDIDATE_STRATEGIES):
        raise ValueError("pilot scoring candidate strategies differ from the plan")
    if set(frame["pool_variant"]) != set(PILOT_POOL_VARIANTS):
        raise ValueError("pilot scoring pool variants differ from the plan")
    if set(frame["fusion_method"]) != set(RAW_FUSION_METHODS):
        raise ValueError("pilot scoring fusion methods differ from production")
    for row in frame.to_dicts():
        if row["schema_version"] != DYNAMIC_POOL_PILOT_SCORING_VERSION:
            raise ValueError("unsupported pilot scoring result version")
        if (
            row["pilot_id"] != plan["pilot_id"]
            or row["plan_fingerprint"] != plan["plan_fingerprint"]
        ):
            raise ValueError("pilot scoring result plan identity differs")
        for field in (
            "plan_fingerprint",
            "source_work_fingerprint",
            "source_result_fingerprint",
            "query_embedding_id",
            "query_embedding_fingerprint",
            "candidate_matrix_signature",
            "pool_matrix_set_fingerprint",
            "fusion_score_set_fingerprint",
            "ranking_fingerprint",
            "result_fingerprint",
        ):
            if not _is_sha256(row[field]):
                raise ValueError(f"pilot scoring {field} is invalid")
        if int(row["target_rank"]) < 1:
            raise ValueError("pilot scoring target rank must be positive")
        if row["fixture_expected_target_at_1"] != (row["target_rank"] == 1):
            raise ValueError("pilot fixture target-at-one field is inconsistent")
        if row["expected_label_basis"] != PILOT_CASE_EVIDENCE_BASIS:
            raise ValueError("pilot scoring promoted fixture labels")
        if row["score_work_reused_across_candidate_strategies"] is not True:
            raise ValueError("pilot scoring did not reuse identical score work")
        if row["encoder_invocations"] != 0 or row["image_materializations"] != 0:
            raise ValueError("pilot scoring crossed the cached-vector boundary")
        if row["raw_score_is_probability"] is not False:
            raise ValueError("pilot scoring promoted a raw score to probability")
        if row["probability_availability"] != "unavailable_fixture_uncalibrated":
            raise ValueError("pilot scoring fabricated probability availability")
        if row["human_review_status"] != "unavailable_not_run":
            raise ValueError("pilot scoring fabricated human review")
        if row["production_default_eligible"] is not False:
            raise ValueError("pilot scoring authorized a production default")
        expected_local = (
            "available"
            if row["pool_variant"] == "dynamic_global_local" and not row["no_geo"]
            else "unavailable"
        )
        if row["local_evidence_status"] != expected_local:
            raise ValueError("pilot scoring local evidence state is inconsistent")
        payload = {
            key: value for key, value in row.items() if key != "result_fingerprint"
        }
        if row["result_fingerprint"] != canonical_semantic_fingerprint(payload):
            raise ValueError("pilot scoring result fingerprint differs")
    for keys, group in frame.group_by(
        "case_id", "pool_variant", "fusion_method", maintain_order=True
    ):
        if group["source_result_fingerprint"].n_unique() != 1:
            raise ValueError(
                "pilot candidate schedules repeated identical vector scoring"
            )
        if group["query_embedding_fingerprint"].n_unique() != 1:
            raise ValueError("pilot candidate schedules changed query embeddings")
        if group.height != len(PILOT_CANDIDATE_STRATEGIES):
            raise ValueError(f"pilot scoring strategy projection is incomplete: {keys}")


def build_dynamic_pool_pilot_scoring_report(
    plan: Mapping[str, object],
    execution: DynamicPoolPilotScoringExecution,
) -> dict[str, object]:
    """Summarize raw scoring and measured reuse without accuracy authority."""

    validate_dynamic_pool_pilot_scoring_execution(execution, plan)
    report = _scoring_report_payload(plan, execution)
    report["report_fingerprint"] = canonical_semantic_fingerprint(report)
    validate_dynamic_pool_pilot_scoring_report(report, plan, execution)
    return report


def validate_dynamic_pool_pilot_scoring_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    execution: DynamicPoolPilotScoringExecution,
) -> None:
    """Require the report to equal a fresh execution-derived summary."""

    if report.get("schema_version") != DYNAMIC_POOL_PILOT_SCORING_REPORT_VERSION:
        raise ValueError("unsupported pilot scoring report version")
    validate_dynamic_pool_pilot_scoring_execution(execution, plan)
    expected = _scoring_report_payload(plan, execution)
    expected["report_fingerprint"] = canonical_semantic_fingerprint(expected)
    if dict(report) != expected:
        raise ValueError("pilot scoring report differs from its execution")


def write_dynamic_pool_pilot_scoring_report(
    report: Mapping[str, object],
    plan: Mapping[str, object],
    execution: DynamicPoolPilotScoringExecution,
    output: str | Path,
) -> Path:
    """Atomically write one validated pilot scoring report."""

    validate_dynamic_pool_pilot_scoring_report(report, plan, execution)
    destination = Path(output)
    if destination.suffix.casefold() != ".json":
        destination /= DYNAMIC_POOL_PILOT_SCORING_REPORT_FILE
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def _project_scoring_results(
    *,
    plan: Mapping[str, object],
    batch_result: PoolMatrixBatchResult,
    work_context: Mapping[str, tuple[Mapping[str, object], str]],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for result in batch_result.canonical_results:
        case, pool_variant = work_context[result.work_fingerprint]
        pool_signatures = sorted(
            pool.pool_matrix.matrix_signature
            for pool in (*result.work.global_pools, *result.work.local_pools)
            if pool.pool_matrix is not None
        )
        for ranking in result.rankings.method_rankings:
            target = next(
                candidate
                for candidate in ranking.candidates
                if candidate.candidate_accepted_taxon_key == case["accepted_taxon_key"]
            )
            target_score = next(
                score
                for score in result.fusion_scores.scores
                if score.method == ranking.method
                and score.candidate_accepted_taxon_key == case["accepted_taxon_key"]
            )
            for strategy in PILOT_CANDIDATE_STRATEGIES:
                base: dict[str, object] = {
                    "schema_version": DYNAMIC_POOL_PILOT_SCORING_VERSION,
                    "pilot_id": plan["pilot_id"],
                    "plan_fingerprint": plan["plan_fingerprint"],
                    "case_id": case["case_id"],
                    "fixture_media_id": case["fixture_media_id"],
                    "country_code": case["country_code"],
                    "region_id": case["region_id"],
                    "no_geo": case["geographic_evidence_status"]
                    == "missing_source_geography",
                    "candidate_strategy": strategy,
                    "pool_variant": pool_variant,
                    "fusion_method": ranking.method,
                    "source_work_fingerprint": result.work_fingerprint,
                    "source_result_fingerprint": result.result_fingerprint,
                    "query_embedding_id": result.source_embedding_id,
                    "query_embedding_fingerprint": result.source_embedding_fingerprint,
                    "candidate_matrix_signature": (
                        result.work.candidate_matrix.matrix_signature
                    ),
                    "pool_matrix_set_fingerprint": canonical_semantic_fingerprint(
                        pool_signatures
                    ),
                    "fusion_score_set_fingerprint": (
                        result.fusion_scores.score_set_fingerprint
                    ),
                    "ranking_fingerprint": ranking.ranking_fingerprint,
                    "target_accepted_taxon_key": case["accepted_taxon_key"],
                    "top_candidate_accepted_taxon_key": (
                        ranking.top_candidate_accepted_taxon_key
                    ),
                    "target_rank": target.candidate_rank,
                    "top_raw_fusion_score": ranking.top_raw_fusion_score,
                    "target_raw_fusion_score": target.raw_fusion_score,
                    "top_margin_raw": ranking.top_margin_raw,
                    "local_evidence_status": target_score.local_evidence_status,
                    "fixture_expected_target_at_1": target.candidate_rank == 1,
                    "expected_label_basis": case["expected_label_basis"],
                    "score_work_reused_across_candidate_strategies": True,
                    "encoder_invocations": result.encoder_invocations,
                    "image_materializations": result.image_materializations,
                    "raw_score_is_probability": False,
                    "probability_availability": "unavailable_fixture_uncalibrated",
                    "human_review_status": "unavailable_not_run",
                    "production_default_eligible": False,
                }
                rows.append(
                    {
                        **base,
                        "result_fingerprint": canonical_semantic_fingerprint(base),
                    }
                )
    return pl.DataFrame(
        rows,
        schema=dynamic_pool_pilot_scoring_schema(),
        orient="row",
        strict=True,
    ).sort(*_SORT)


def _scoring_report_payload(
    plan: Mapping[str, object], execution: DynamicPoolPilotScoringExecution
) -> dict[str, object]:
    frame = execution.results
    matrix = execution.dynamic_cache_metrics
    comparison = _global_local_comparison(frame)
    return {
        "schema_version": DYNAMIC_POOL_PILOT_SCORING_REPORT_VERSION,
        "pilot_id": plan["pilot_id"],
        "plan_fingerprint": plan["plan_fingerprint"],
        "execution_fingerprint": execution.execution_fingerprint,
        "evidence_basis": "deterministic_fixture_vectors_through_production_scoring",
        "model_execution": {
            "bioclip_image_encoder_run": False,
            "synthetic_fixture_vectors": True,
            "fixture_vector_dimension": PILOT_FIXTURE_VECTOR_DIMENSION,
            "historical_bioclip_manifest_counted_as_current_run": False,
            "linear_parameters": "fixture_equal_weights_not_validation_fitted",
        },
        "coverage": {
            "case_count": len(plan["cases"]),
            "candidate_strategy_count": len(PILOT_CANDIDATE_STRATEGIES),
            "pool_variant_count": len(PILOT_POOL_VARIANTS),
            "fusion_method_count": len(RAW_FUSION_METHODS),
            "variant_count_per_case": int(frame.height / len(plan["cases"])),
            "result_row_count": frame.height,
            "score_work_item_count": len(execution.works),
        },
        "embedding_reuse": {
            "unique_query_embedding_count": frame[
                "query_embedding_fingerprint"
            ].n_unique(),
            "query_embedding_consumption_count": sum(
                result.cached_query_vectors_consumed
                for result in execution.batch_result.canonical_results
            ),
            "query_embedding_reuse_event_count": (
                len(execution.works) - frame["query_embedding_fingerprint"].n_unique()
            ),
            "encoder_invocations": execution.batch_result.metrics.encoder_invocations,
            "image_materializations": (
                execution.batch_result.metrics.image_materializations
            ),
            "avoided_encoder_seconds": None,
            "avoided_encoder_seconds_status": "not_instrumented",
        },
        "matrix_reuse": {
            "family": _matrix_metrics(execution.family_cache_metrics),
            "candidate": _matrix_metrics(matrix.candidate),
            "pool": _matrix_metrics(matrix.pool),
            "batch": asdict(execution.batch_result.metrics),
            "avoided_matrix_seconds": None,
            "avoided_matrix_seconds_status": "not_instrumented",
        },
        "variant_metrics": _variant_metrics(frame),
        "global_local_comparison": comparison,
        "results_fingerprint": canonical_semantic_fingerprint(
            frame["result_fingerprint"].to_list()
        ),
        "selection": {
            "status": "insufficient_evidence",
            "selected_candidate_strategy": None,
            "selected_pool_variant": None,
            "selected_fusion_method": None,
            "production_default_eligible": False,
            "reason": "fixture_vectors_and_expected_taxa_cannot_select_a_default",
        },
        "scientific_claims": {
            "raw_scores_are_probabilities": False,
            "fixture_target_at_one_is_reviewed_accuracy": False,
            "missing_geography_is_biological_absence": False,
            "human_review_completed": False,
            "statistical_support_available": False,
            "occurrence_release_authorized": False,
        },
    }


def _variant_metrics(frame: pl.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool_variant in PILOT_POOL_VARIANTS:
        for method in RAW_FUSION_METHODS:
            group = frame.filter(
                (pl.col("pool_variant") == pool_variant)
                & (pl.col("fusion_method") == method)
            )
            rows.append(
                {
                    "pool_variant": pool_variant,
                    "fusion_method": method,
                    "result_row_count": group.height,
                    "distinct_score_work_count": group[
                        "source_result_fingerprint"
                    ].n_unique(),
                    "fixture_expected_target_at_1_count": int(
                        group["fixture_expected_target_at_1"].sum()
                    ),
                    "fixture_expected_target_at_1_fraction": float(
                        group["fixture_expected_target_at_1"].mean()
                    ),
                    "mean_target_rank": float(group["target_rank"].mean()),
                    "mean_target_raw_fusion_score": float(
                        group["target_raw_fusion_score"].mean()
                    ),
                    "local_evidence_available_count": group.filter(
                        pl.col("local_evidence_status") == "available"
                    ).height,
                    "metric_interpretation": "fixture_raw_structural_not_accuracy",
                }
            )
    return rows


def _global_local_comparison(frame: pl.DataFrame) -> dict[str, object]:
    global_rows = frame.filter(pl.col("pool_variant") == "global_only_control")
    dynamic_rows = frame.filter(pl.col("pool_variant") == "dynamic_global_local")
    keys = ["case_id", "candidate_strategy", "fusion_method"]
    pairs = global_rows.join(
        dynamic_rows,
        on=keys,
        how="inner",
        suffix="_dynamic",
        validate="1:1",
    )
    located = pairs.filter(~pl.col("no_geo"))
    no_geo = pairs.filter(pl.col("no_geo"))
    located_score_changes = sum(
        not isclose(left, right, abs_tol=1e-12)
        for left, right in zip(
            located["target_raw_fusion_score"],
            located["target_raw_fusion_score_dynamic"],
            strict=True,
        )
    )
    no_geo_parity = sum(
        isclose(left, right, abs_tol=1e-12) and top == top_dynamic
        for left, right, top, top_dynamic in zip(
            no_geo["target_raw_fusion_score"],
            no_geo["target_raw_fusion_score_dynamic"],
            no_geo["top_candidate_accepted_taxon_key"],
            no_geo["top_candidate_accepted_taxon_key_dynamic"],
            strict=True,
        )
    )
    return {
        "located_pair_count": located.height,
        "located_target_raw_score_changed_count": located_score_changes,
        "located_top_candidate_changed_count": located.filter(
            pl.col("top_candidate_accepted_taxon_key")
            != pl.col("top_candidate_accepted_taxon_key_dynamic")
        ).height,
        "no_geo_pair_count": no_geo.height,
        "no_geo_global_fallback_parity_count": no_geo_parity,
        "comparison_is_accuracy_claim": False,
    }


def _fixture_query_embedding(
    *,
    plan: Mapping[str, object],
    case: Mapping[str, object],
    catalog_keys: Sequence[str],
    model_fingerprint: str,
) -> RawFullFrameEmbedding:
    target_index = catalog_keys.index(str(case["accepted_taxon_key"]))
    competitor_index = (target_index + 1) % len(catalog_keys)
    vector = _unit_vector(target_index, competitor_index, admixture=0.22)
    embedding_id = canonical_semantic_fingerprint(
        {
            "schema_version": "pilot-fixture-query-embedding-id-v1.0.0",
            "plan_fingerprint": plan["plan_fingerprint"],
            "fixture_media_id": case["fixture_media_id"],
        }
    )
    embedding_version = "pilot-fixture-full-frame-embedding-v1.0.0"
    embedding_fingerprint = canonical_semantic_fingerprint(
        {
            "embedding": vector,
            "embedding_id": embedding_id,
            "embedding_version": embedding_version,
        }
    )
    return RawFullFrameEmbedding(
        embedding_id=embedding_id,
        embedding_version=embedding_version,
        embedding_fingerprint=embedding_fingerprint,
        visual_input_id=canonical_semantic_fingerprint(
            ["pilot-fixture-visual-input", case["fixture_media_id"]]
        ),
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        raw_image_content_hash=canonical_semantic_fingerprint(
            ["no-source-bytes-fixture", case["fixture_media_id"]]
        ),
        transformation_fingerprint=canonical_semantic_fingerprint(
            ["identity-full-frame-fixture", case["fixture_media_id"]]
        ),
        model_fingerprint=model_fingerprint,
        image_resize_mode="fixture_no_image",
        preprocessing_contract_fingerprint=canonical_semantic_fingerprint(
            ["pilot-fixture-preprocessing-contract-v1"]
        ),
        preprocessing_fingerprint=canonical_semantic_fingerprint(
            ["pilot-fixture-preprocessing-v1"]
        ),
        embedding_dimension=len(vector),
        embedding=vector,
        embedding_norm=sqrt(fsum(value * value for value in vector)),
    )


def _family_prototypes(
    plan: Mapping[str, object],
) -> tuple[FamilyPrototypeVector, ...]:
    vector = tuple(1 / sqrt(5) if index < 5 else 0.0 for index in range(8))
    return (
        FamilyPrototypeVector(
            family_key="gbif:9417",
            family_name="Papilionidae",
            prototype_fingerprint=canonical_semantic_fingerprint(
                [plan["plan_fingerprint"], "fixture-family-prototype", list(vector)]
            ),
            embedding=vector,
        ),
    )


def _candidate_prototypes(
    plan: Mapping[str, object],
) -> tuple[CandidatePrototypeVector, ...]:
    return tuple(
        CandidatePrototypeVector(
            accepted_taxon_key=str(taxon["accepted_taxon_key"]),
            scientific_name=str(taxon["scientific_name"]),
            prototype_fingerprint=canonical_semantic_fingerprint(
                [
                    plan["plan_fingerprint"],
                    "fixture-candidate-prototype",
                    taxon["accepted_taxon_key"],
                    list(_unit_vector(index)),
                ]
            ),
            embedding=_unit_vector(index),
        )
        for index, taxon in enumerate(plan["taxon_catalog"])
    )


def _global_pool_inputs(
    *,
    plan: Mapping[str, object],
    catalog: Mapping[str, Mapping[str, object]],
    cache: DynamicPoolMatrixCache,
    model_fingerprint: str,
) -> tuple[GlobalReferencePoolInput, ...]:
    keys = list(catalog)
    output: list[GlobalReferencePoolInput] = []
    for index, key in enumerate(keys):
        references = _reference_vectors(
            plan=plan,
            candidate_key=key,
            scope="global",
            region_id=None,
            primary_vector=_unit_vector(index),
            secondary_vector=_unit_vector(index, (index + 1) % len(keys), 0.12),
        )
        membership = canonical_semantic_fingerprint(
            [reference.member_fingerprint for reference in references]
        )
        matrix = cache.get_pool_matrix(
            route=PILOT_SCORING_ROUTE,
            visual_input_kind=RAW_FULL_IMAGE_KIND,
            geographic_scope="global",
            candidate_accepted_taxon_key=key,
            model_fingerprint=model_fingerprint,
            reference_embedding_artifact_fingerprint=(
                _reference_embedding_source_fingerprint(plan)
            ),
            pool_membership_fingerprint=membership,
            pool_ids=(f"pilot-global-pool:{key}",),
            references=references,
        )
        output.append(
            GlobalReferencePoolInput(
                candidate_accepted_taxon_key=key,
                candidate_scientific_name=str(catalog[key]["scientific_name"]),
                pool_matrix=matrix,
                configured_reference_count=2,
                configured_top_k=2,
            )
        )
    return tuple(output)


def _local_pool_inputs(
    *,
    plan: Mapping[str, object],
    case: Mapping[str, object],
    pool_variant: str,
    catalog: Mapping[str, Mapping[str, object]],
    cache: DynamicPoolMatrixCache,
    model_fingerprint: str,
) -> tuple[LocalReferencePoolInput, ...]:
    available = (
        pool_variant == "dynamic_global_local"
        and case["geographic_evidence_status"] == "located_fixture_context"
    )
    output: list[LocalReferencePoolInput] = []
    keys = list(catalog)
    for index, key in enumerate(keys):
        if not available:
            reason = (
                "global_only_control"
                if pool_variant == "global_only_control"
                else "missing_source_geography_global_fallback"
            )
            output.append(
                LocalReferencePoolInput(
                    candidate_accepted_taxon_key=key,
                    candidate_scientific_name=str(catalog[key]["scientific_name"]),
                    local_pool_status="unavailable",
                    local_pool_unavailable_reason=reason,
                    pool_matrix=None,
                    configured_reference_count=0,
                    configured_top_k=2,
                )
            )
            continue
        references = _reference_vectors(
            plan=plan,
            candidate_key=key,
            scope="exact_local_cell",
            region_id=str(case["region_id"]),
            primary_vector=_unit_vector(index, (index + 2) % len(keys), 0.05),
            secondary_vector=_unit_vector(index, (index + 1) % len(keys), 0.08),
        )
        membership = canonical_semantic_fingerprint(
            [reference.member_fingerprint for reference in references]
        )
        matrix = cache.get_pool_matrix(
            route=PILOT_SCORING_ROUTE,
            visual_input_kind=RAW_FULL_IMAGE_KIND,
            geographic_scope="exact_local_cell",
            candidate_accepted_taxon_key=key,
            model_fingerprint=model_fingerprint,
            reference_embedding_artifact_fingerprint=(
                _reference_embedding_source_fingerprint(plan)
            ),
            pool_membership_fingerprint=membership,
            pool_ids=(f"pilot-local-pool:{case['region_id']}:{key}",),
            references=references,
        )
        output.append(
            LocalReferencePoolInput(
                candidate_accepted_taxon_key=key,
                candidate_scientific_name=str(catalog[key]["scientific_name"]),
                local_pool_status="available",
                local_pool_unavailable_reason=None,
                pool_matrix=matrix,
                configured_reference_count=2,
                configured_top_k=2,
            )
        )
    return tuple(output)


def _reference_vectors(
    *,
    plan: Mapping[str, object],
    candidate_key: str,
    scope: str,
    region_id: str | None,
    primary_vector: tuple[float, ...],
    secondary_vector: tuple[float, ...],
) -> tuple[PoolReferenceVector, ...]:
    rows: list[PoolReferenceVector] = []
    for index, vector in enumerate((primary_vector, secondary_vector), start=1):
        identity = {
            "schema_version": "pilot-fixture-reference-vector-v1.0.0",
            "plan_fingerprint": plan["plan_fingerprint"],
            "candidate_key": candidate_key,
            "scope": scope,
            "region_id": region_id,
            "reference_index": index,
            "embedding": list(vector),
        }
        fingerprint = canonical_semantic_fingerprint(identity)
        rows.append(
            PoolReferenceVector(
                reference_media_id=f"pilot-reference-media:{fingerprint}",
                reference_observation_id=f"pilot-reference-observation:{fingerprint}",
                member_fingerprint=canonical_semantic_fingerprint(
                    ["pilot-member", fingerprint]
                ),
                reference_embedding_fingerprint=canonical_semantic_fingerprint(
                    ["pilot-reference-embedding", fingerprint]
                ),
                embedding=vector,
            )
        )
    return tuple(rows)


def _fixture_linear_parameters(
    plan: Mapping[str, object],
) -> ValidationLinearFusionParameters:
    return ValidationLinearFusionParameters(
        validation_artifact_fingerprint=canonical_semantic_fingerprint(
            [plan["plan_fingerprint"], "fixture-equal-linear-parameters-not-validation"]
        ),
        full_weights=tuple(1 / len(FUSION_COMPONENTS) for _ in FUSION_COMPONENTS),
        global_only_weights=tuple(
            1 / len(GLOBAL_FUSION_COMPONENTS) for _ in GLOBAL_FUSION_COMPONENTS
        ),
    )


def _unit_vector(
    primary_index: int,
    secondary_index: int | None = None,
    admixture: float = 0.0,
) -> tuple[float, ...]:
    values = [0.0] * PILOT_FIXTURE_VECTOR_DIMENSION
    values[primary_index] = 1.0
    if secondary_index is not None:
        values[secondary_index] = admixture
    norm = sqrt(fsum(value * value for value in values))
    return tuple(value / norm for value in values)


def _fixture_model_fingerprint(plan: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        [
            plan["plan_fingerprint"],
            "synthetic-fixture-vectors-not-bioclip-model-execution",
            PILOT_FIXTURE_VECTOR_DIMENSION,
        ]
    )


def _family_source_fingerprint(plan: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        [plan["plan_fingerprint"], "pilot-family-prototype-source"]
    )


def _candidate_prototype_source_fingerprint(plan: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        [plan["plan_fingerprint"], "pilot-candidate-prototype-source"]
    )


def _reference_embedding_source_fingerprint(plan: Mapping[str, object]) -> str:
    return canonical_semantic_fingerprint(
        [plan["plan_fingerprint"], "pilot-reference-embedding-source"]
    )


def _execution_identity(
    execution: DynamicPoolPilotScoringExecution,
) -> dict[str, object]:
    return {
        "schema_version": "dynamic-pool-pilot-scoring-execution-v1.0.0",
        "work_fingerprints": [work.work_fingerprint for work in execution.works],
        "batch_result_fingerprint": execution.batch_result.result_fingerprint,
        "family_cache_metrics": _matrix_metrics(execution.family_cache_metrics),
        "dynamic_cache_metrics": {
            "candidate": _matrix_metrics(execution.dynamic_cache_metrics.candidate),
            "pool": _matrix_metrics(execution.dynamic_cache_metrics.pool),
        },
        "result_fingerprints": execution.results["result_fingerprint"].to_list(),
    }


def _matrix_metrics(metrics: MatrixCacheMetrics) -> dict[str, int | float | None]:
    return {
        "requests": metrics.requests,
        "hits": metrics.hits,
        "misses": metrics.misses,
        "materializations": metrics.materializations,
        "entries": metrics.entries,
        "rows_materialized": metrics.rows_materialized,
        "bytes_materialized": metrics.bytes_materialized,
        "evictions": metrics.evictions,
        "hit_rate": metrics.hit_rate,
    }


def _validate_cache_metrics(metrics: MatrixCacheMetrics) -> None:
    if metrics.requests != metrics.hits + metrics.misses:
        raise ValueError("pilot matrix cache request accounting differs")
    if metrics.misses != metrics.materializations:
        raise ValueError("pilot matrix cache materialization accounting differs")
    if metrics.entries + metrics.evictions != metrics.materializations:
        raise ValueError("pilot matrix cache entry accounting differs")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


__all__ = [
    "DYNAMIC_POOL_PILOT_SCORING_REPORT_FILE",
    "DYNAMIC_POOL_PILOT_SCORING_REPORT_VERSION",
    "DYNAMIC_POOL_PILOT_SCORING_VERSION",
    "DynamicPoolPilotScoringExecution",
    "build_dynamic_pool_pilot_scoring_report",
    "dynamic_pool_pilot_scoring_schema",
    "execute_dynamic_pool_pilot_scoring",
    "validate_dynamic_pool_pilot_scoring_execution",
    "validate_dynamic_pool_pilot_scoring_report",
    "validate_dynamic_pool_pilot_scoring_results",
    "write_dynamic_pool_pilot_scoring_report",
]
