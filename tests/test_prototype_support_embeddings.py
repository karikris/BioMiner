from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image
import polars as pl

from biominer.benchmarks.prototype_support_embeddings import (
    PROTOTYPE_EMBEDDING_FAILURES_FILE,
    PROTOTYPE_REFERENCE_EMBEDDINGS_FILE,
    PROTOTYPE_REFERENCE_PROTOTYPES_FILE,
    PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE,
    PrototypeSupportEmbeddingConfig,
    run_prototype_support_embedding_job,
)
from biominer.bioclip.bioclip_worker import decoded_rgb_image_content_hash
from biominer.common.semantic_hash import canonical_semantic_fingerprint
from biominer.references.prototype_freeze import (
    PROTOTYPE_READINESS_SCHEMA_VERSION,
    PROTOTYPE_SUPPORT_SCHEMA_VERSION,
    prototype_support_schema,
)


MODEL = "imageomics/bioclip-2.5-vith14"
REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"


class FakeScorer:
    model_id = MODEL
    model_revision = REVISION
    model_weights_sha256 = "sha256:" + "a" * 64
    open_clip_version = "3.3.0"
    open_clip_config_sha256 = "sha256:" + "b" * 64
    preprocessing_version = "openclip-preprocessing-attestation-v2"
    preprocessing_fingerprint = "sha256:" + "c" * 64
    image_resize_mode = "longest"
    effective_image_resize_mode = "longest"
    device = "mps"
    gpu_name = "Apple MPS fixture"

    def __init__(
        self, vectors: dict[str, list[float]], *, fail_name: str | None = None
    ):
        self.vectors = vectors
        self.fail_name = fail_name
        self.calls: list[tuple[str, ...]] = []
        self.attestation_calls = 0
        self.last_image_content_hashes: tuple[str, ...] = ()

    @property
    def cache_metrics(self) -> dict[str, int]:
        return {
            "bioclip_worker_process_starts": 1,
            "bioclip_model_loads": 1,
            "bioclip_worker_requests": len(self.calls),
        }

    def ensure_model_attestation(self) -> None:
        self.attestation_calls += 1

    def embed_image_paths(self, paths: list[Path]) -> list[list[float]]:
        names = tuple(path.name for path in paths)
        self.calls.append(names)
        if self.fail_name is not None and self.fail_name in names:
            raise RuntimeError(f"fixture failure: {self.fail_name}")
        hashes = []
        for path in paths:
            with Image.open(path) as image:
                hashes.append(decoded_rgb_image_content_hash(image.convert("RGB")))
        self.last_image_content_hashes = tuple(hashes)
        return [self.vectors[name] for name in names]


def test_builds_frozen_embeddings_prototypes_and_route_local_neighbours(
    tmp_path: Path,
) -> None:
    config, rows = _fixture_job(tmp_path)
    scorer = FakeScorer(
        {
            row["path"].name: [float(index + 1), float(5 - index), 1.0]
            for index, row in enumerate(rows)
        }
    )

    result = run_prototype_support_embedding_job(config, scorer=scorer)

    assert result.report["status"] == "complete"
    assert result.embeddings_path.name == PROTOTYPE_REFERENCE_EMBEDDINGS_FILE
    assert result.prototypes_path.name == PROTOTYPE_REFERENCE_PROTOTYPES_FILE
    assert result.visual_neighbours_path is not None
    assert result.visual_neighbours_path.name == PROTOTYPE_VISUAL_NEIGHBOUR_SPECIES_FILE
    assert result.failures_path is None
    embeddings = pl.read_parquet(result.embeddings_path)
    prototypes = pl.read_parquet(result.prototypes_path)
    neighbours = pl.read_parquet(result.visual_neighbours_path)
    assert embeddings.height == 5
    assert set(embeddings["verification_status"]) == {"provider_supported"}
    assert embeddings["human_verified"].to_list() == [False] * 5
    assert set(prototypes["route"]) == {"adult_field", "larval"}
    assert set(neighbours["route"]) == {"adult_field"}
    assert neighbours.height == 2
    assert set(neighbours["subject_accepted_taxon_key"]) == {"gbif:1", "gbif:2"}
    assert scorer.attestation_calls == 1
    assert len(scorer.calls) == 3


def test_one_failed_record_is_quarantined_and_remaining_records_progress(
    tmp_path: Path,
) -> None:
    config, rows = _fixture_job(tmp_path)
    failed_name = rows[3]["path"].name
    scorer = FakeScorer(
        {
            row["path"].name: [float(index + 1), 1.0, 0.5]
            for index, row in enumerate(rows)
        },
        fail_name=failed_name,
    )

    result = run_prototype_support_embedding_job(config, scorer=scorer)

    assert result.report["status"] == "complete_with_retryable_failures"
    assert result.report["counts"]["embedded"] == 4
    assert result.report["counts"]["retryable_failures"] == 1
    assert result.failures_path is not None
    assert result.failures_path.name == PROTOTYPE_EMBEDDING_FAILURES_FILE
    failures = pl.read_parquet(result.failures_path)
    assert failures.height == 1
    assert failures["retryable"].item() is True
    assert failures["error_type"].item() == "RuntimeError"
    assert failed_name in failures["error_message"].item()
    assert pl.read_parquet(result.embeddings_path).height == 4
    assert len(scorer.calls) == 5  # failed batch is isolated, then later work continues


def test_explicit_skip_is_a_retryable_audited_partition(tmp_path: Path) -> None:
    config, rows = _fixture_job(tmp_path)
    skipped_media_id = str(rows[-1]["support"]["reference_media_id"])
    config = PrototypeSupportEmbeddingConfig(
        **{
            field: getattr(config, field)
            for field in config.__dataclass_fields__
            if field != "skip_records"
        },
        skip_records=((skipped_media_id, "temporary provider record issue"),),
    )
    scorer = FakeScorer(
        {
            row["path"].name: [float(index + 1), 1.0, 0.5]
            for index, row in enumerate(rows)
        }
    )

    result = run_prototype_support_embedding_job(config, scorer=scorer)

    assert result.report["counts"]["operator_skips"] == 1
    failures = pl.read_parquet(result.failures_path)
    assert failures["reference_media_id"].item() == skipped_media_id
    assert failures["error_type"].item() == "operator_skipped"
    assert skipped_media_id not in set(
        pl.read_parquet(result.embeddings_path)["reference_media_id"]
    )


def test_completed_embeddings_resume_without_model_recomputation(
    tmp_path: Path,
) -> None:
    config, rows = _fixture_job(tmp_path)
    vectors = {
        row["path"].name: [float(index + 1), 1.0, 0.5] for index, row in enumerate(rows)
    }
    first_scorer = FakeScorer(vectors)
    first = run_prototype_support_embedding_job(config, scorer=first_scorer)
    first_embeddings = pl.read_parquet(first.embeddings_path)
    resume_scorer = FakeScorer(vectors)

    resumed = run_prototype_support_embedding_job(config, scorer=resume_scorer)

    assert resumed.report["counts"]["resumed_embeddings"] == 5
    assert resume_scorer.calls == []
    assert resume_scorer.attestation_calls == 0
    assert resumed.report["model"]["model_execution_performed_this_run"] is False
    assert resumed.report["model"]["resumed_runtime_attestation"] is True
    assert resumed.report["model"]["gpu_name"] == "Apple MPS fixture"
    assert pl.read_parquet(resumed.embeddings_path).equals(first_embeddings)


def test_pilot_config_pins_external_full_bank_job() -> None:
    config = PrototypeSupportEmbeddingConfig.read_json(
        "config/pilot/papilio_demoleus_prototype_embeddings.json"
    )

    assert config.model_name == MODEL
    assert config.model_revision == REVISION
    assert config.device == "mps"
    assert config.skip_records == ()
    assert config.output_dir.parts[0] == "runs"
    assert config.batch_size == 16


def _fixture_job(
    tmp_path: Path,
) -> tuple[PrototypeSupportEmbeddingConfig, list[dict[str, object]]]:
    definitions = [
        ("gbif:1", "Species one", "adult_field", "adult", "support_train"),
        ("gbif:1", "Species one", "adult_field", "adult", "support_train"),
        ("gbif:2", "Species two", "adult_field", "adult", "support_train"),
        ("gbif:1", "Species one", "larval", "larva", "support_train"),
        ("gbif:1", "Species one", "adult_field", "adult", "model_selection"),
    ]
    fixture_rows: list[dict[str, object]] = []
    support_rows = []
    for index, (taxon_key, name, route, life_stage, split) in enumerate(definitions):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32 + index, 24 + index), (20 * index, 90, 160)).save(path)
        base = _support_row(
            index=index,
            path=path,
            taxon_key=taxon_key,
            scientific_name=name,
            route=route,
            life_stage=life_stage,
            split=split,
        )
        base["support_row_fingerprint"] = canonical_semantic_fingerprint(
            {
                key: value
                for key, value in base.items()
                if key != "support_row_fingerprint"
            }
        )
        support_rows.append(base)
        fixture_rows.append({"path": path, "support": base})
    support = pl.DataFrame(
        support_rows, schema=prototype_support_schema(), orient="row", strict=True
    ).sort("reference_media_id")
    support_path = tmp_path / "prototype_support_manifest.parquet"
    support.write_parquet(support_path)
    support_fingerprint = canonical_semantic_fingerprint(support.to_dicts())
    readiness = {
        "schema_version": PROTOTYPE_READINESS_SCHEMA_VERSION,
        "prototype_readiness_status": "prototype_ready_with_shortfalls",
        "classification_authorised": True,
        "bank_status": "prototype_only",
        "reference_bank_version": "prototype-bank-fixture-v1",
        "human_verification_complete": False,
        "support_manifest_fingerprint": support_fingerprint,
    }
    readiness_path = tmp_path / "reference_bank_prototype_readiness.json"
    readiness_path.write_text(json.dumps(readiness), encoding="utf-8")
    return (
        PrototypeSupportEmbeddingConfig(
            support_manifest=support_path,
            support_manifest_sha256=_file_sha256(support_path),
            readiness=readiness_path,
            readiness_sha256=_file_sha256(readiness_path),
            output_dir=tmp_path / "output",
            runtime_python=tmp_path / "runtime" / "bin" / "python",
            hf_cache_dir=tmp_path / "hf",
            model_name=MODEL,
            model_revision=REVISION,
            open_clip_version="3.3.0",
            device="mps",
            batch_size=2,
        ),
        fixture_rows,
    )


def _support_row(
    *,
    index: int,
    path: Path,
    taxon_key: str,
    scientific_name: str,
    route: str,
    life_stage: str,
    split: str,
) -> dict[str, object]:
    fingerprint = _file_sha256(path)
    return {
        "schema_version": PROTOTYPE_SUPPORT_SCHEMA_VERSION,
        "reference_bank_version": "prototype-bank-fixture-v1",
        "reference_media_id": f"reference-media:{index:064x}",
        "reference_observation_id": f"reference-observation:{index:064x}",
        "candidate_scope_type": "accepted_taxon",
        "accepted_taxon_key": taxon_key,
        "scientific_name": scientific_name,
        "source": "GBIF",
        "source_snapshot_version": "fixture-v1",
        "provider_media_id": f"media-{index}",
        "trust_level": "R4",
        "verification_status": "provider_supported",
        "human_verified": False,
        "geographic_layer": "A",
        "geo_cluster_id": "cluster-a",
        "route": route,
        "life_stage": life_stage,
        "visual_domain": "live_field",
        "reference_group": f"fixture:{taxon_key}",
        "licence": "CC BY 4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "licence_policy_status": "allowed",
        "attribution": "Fixture creator / CC BY 4.0",
        "attribution_complete": True,
        "source_object_uri": str(path),
        "source_image_sha256": fingerprint,
        "source_object_fingerprint": _sha(f"object-{index}"),
        "duplicate_group_id": f"duplicate-{index}",
        "exact_hash_group_id": f"exact-{index}",
        "perceptual_duplicate_group_id": f"perceptual-{index}",
        "observation_group_id": f"observation-{index}",
        "burst_group_id": f"burst-{index}",
        "owner_group_id": f"owner-{index}",
        "photographer_group_id": f"photographer-{index}",
        "provider_mirror_group_id": f"provider-mirror-{index}",
        "qa_disposition": "review",
        "image_quality_check": "pass",
        "subject_presence_check": "review",
        "subject_size_check": "review",
        "detector_evidence_status": "not_instrumented",
        "dataset_split": split,
        "leakage_component_id": f"leakage-{index}",
        "leakage_component_size": 1,
        "split_fingerprint": _sha(f"split-{index}"),
        "prototype_only": True,
        "support_row_fingerprint": "",
    }


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
