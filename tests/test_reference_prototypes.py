from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
import hashlib
from math import isclose, sqrt
from pathlib import Path
import struct
from urllib.parse import unquote, urlsplit

from PIL import Image
import polars as pl
from polars.testing import assert_frame_equal
import pytest

import biominer.bioclip.reference_embeddings as reference_embeddings_module
from biominer.bioclip.reference_embeddings import (
    ReferenceVisualInput,
    build_reference_embeddings,
    decoded_image_file_content_hash,
    reference_embeddings_artifact_fingerprint,
)
import biominer.bioclip.reference_prototypes as reference_prototypes_module
from biominer.bioclip.reference_prototypes import (
    PROTOTYPE_METHOD_NORMALIZED_MEAN,
    PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    PROTOTYPE_SCOPE_GLOBAL,
    PROTOTYPE_SCOPE_REGIONAL,
    REFERENCE_PROTOTYPES_FILE,
    ReferenceCenteringContext,
    ReferenceObservationEmbedding,
    build_reference_centering_contexts,
    build_reference_prototypes,
    load_reference_prototypes,
    mean_center_query_embedding,
    reference_prototypes_artifact_fingerprint,
    reference_prototypes_schema,
    validate_reference_prototypes,
    write_reference_prototypes,
)
from biominer.detection.detector_base import DecodedImage
import biominer.references.readiness as readiness_module
from biominer.references.readiness import (
    REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
    ReferenceBankReadinessPermit,
    ReferenceModelInputIdentity,
    make_reference_split_assignment_fingerprint,
    reference_support_manifest_fingerprint,
    reference_support_manifest_schema,
)
from biominer.vision.full_frame_attention import (
    FULL_FRAME_VISUAL_INPUT_VERSION,
    RAW_FULL_IMAGE_KIND,
    TargetPreprocessingContract,
    raw_full_frame_visual_input,
)


NOW = datetime(2026, 7, 14, 9, 0, tzinfo=UTC)
TARGET = "gbif:1938069"
COMPETITOR = "gbif:1938070"
REVISION = "191d741545e4c741cdef4b22c6eb69c945c1e592"
WEIGHTS_SHA256 = "sha256:" + "f" * 64
OPEN_CLIP_CONFIG_SHA256 = "sha256:" + "e" * 64
PREPROCESSING_ATTESTATION_VERSION = "openclip-preprocessing-attestation-v2"
PREPROCESSING_CONFIG = {
    "fill_color": 0,
    "interpolation": "bicubic",
    "mean": [0.48145466, 0.4578275, 0.40821073],
    "mode": "RGB",
    "resize_mode": "longest",
    "size": [224, 224],
    "std": [0.26862954, 0.26130258, 0.27577711],
}


@dataclass(frozen=True, slots=True)
class _EmbeddingSpec:
    media_id: str
    observation_id: str
    taxon_key: str
    scientific_name: str
    geo_cluster_id: str
    vector: tuple[float, float, float]
    support_split: str = "support_train"
    life_stage: str = "adult"
    visual_domain: str = "live_field"
    route: str = "adult_field"


class _FakeScorer:
    model_id = "imageomics/bioclip-2.5-vith14"
    model_revision = REVISION
    model_weights_sha256 = WEIGHTS_SHA256
    image_resize_mode = "longest"
    effective_image_resize_mode = "longest"
    open_clip_version = "3.3.0"
    open_clip_config_sha256 = OPEN_CLIP_CONFIG_SHA256
    preprocessing_version = PREPROCESSING_ATTESTATION_VERSION
    preprocessing_config = PREPROCESSING_CONFIG

    def __init__(self, vectors: dict[str, tuple[float, float, float]]) -> None:
        self._vectors = vectors
        self.last_image_content_hashes: tuple[str, ...] = ()
        self.preprocessing_config = dict(PREPROCESSING_CONFIG)
        self.preprocessing_fingerprint = (
            reference_embeddings_module._preprocessing_attestation_fingerprint(  # noqa: SLF001 - fixture must use the production identity contract.
                open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
                open_clip_version=self.open_clip_version,
                preprocessing_config=self.preprocessing_config,
                preprocessing_version=PREPROCESSING_ATTESTATION_VERSION,
            )
        )

    def embed_image_paths(self, image_paths: list[Path]) -> list[list[float]]:
        self.last_image_content_hashes = tuple(
            decoded_image_file_content_hash(path) for path in image_paths
        )
        return [list(self._vectors[path.name]) for path in image_paths]

    def ensure_model_attestation(self) -> None:
        return None


def test_builds_normalized_global_and_regional_species_prototypes(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "target-b",
                "target-observation-b",
                TARGET,
                "Papilio demoleus",
                "cluster-b",
                (0.8, 0.6, 0.0),
            ),
            _spec(
                "competitor-a",
                "competitor-observation-a",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
            _spec(
                "competitor-b",
                "competitor-observation-b",
                COMPETITOR,
                "Papilio polytes",
                "cluster-b",
                (-0.6, 0.8, 0.0),
            ),
        ),
    )

    prototypes = build_reference_prototypes(
        embeddings,
        balanced_sampling_seed=17,
    )

    assert prototypes.schema == reference_prototypes_schema(3)
    assert prototypes.height == 12
    assert set(prototypes["prototype_method"]) == {
        PROTOTYPE_METHOD_NORMALIZED_MEAN,
        PROTOTYPE_METHOD_SIMPLESHOT_MEAN_CENTERED,
    }
    assert set(prototypes["cluster_scope_type"]) == {
        PROTOTYPE_SCOPE_GLOBAL,
        PROTOTYPE_SCOPE_REGIONAL,
    }
    assert set(prototypes["accepted_taxon_key"]) == {TARGET, COMPETITOR}
    assert set(prototypes["route"]) == {"adult_field"}
    assert set(prototypes["life_stage"]) == {"adult"}
    assert set(prototypes["visual_domain"]) == {"live_field"}
    assert set(prototypes["visual_input_kind"]) == {RAW_FULL_IMAGE_KIND}
    assert all(
        isclose(value, 1.0, abs_tol=1e-6) for value in prototypes["embedding_norm"]
    )
    assert prototypes["prototype_id"].n_unique() == prototypes.height
    assert prototypes["prototype_fingerprint"].n_unique() == prototypes.height

    target_global = _prototype_row(
        prototypes,
        taxon_key=TARGET,
        scope=PROTOTYPE_SCOPE_GLOBAL,
        method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
    )
    expected = _unit((1.8, 0.6, 0.0))
    assert target_global["geo_cluster_id"] == "all"
    assert target_global["view"] == "all"
    assert target_global["reference_count"] == 2
    assert target_global["independent_observation_count"] == 2
    assert target_global["balanced_sampling_seed"] is None
    assert target_global["centering_fingerprint"] is None
    assert target_global["embedding"] == pytest.approx(expected, abs=1e-6)

    regional = prototypes.filter(
        pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_REGIONAL
    )
    assert regional.height == 8
    assert set(regional["geo_cluster_id"]) == {"cluster-a", "cluster-b"}
    assert set(regional["reference_count"]) == {1}
    assert set(regional["independent_observation_count"]) == {1}


def test_support_train_only_and_independent_observation_weighting(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-view-a",
                "target-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "target-view-b",
                "target-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
            _spec(
                "calibration-only",
                "calibration-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.0, 0.0, 1.0),
                support_split="calibration",
            ),
        ),
    )

    prototypes = build_reference_prototypes(embeddings)
    global_row = _prototype_row(
        prototypes,
        taxon_key=TARGET,
        scope=PROTOTYPE_SCOPE_GLOBAL,
        method=PROTOTYPE_METHOD_NORMALIZED_MEAN,
    )

    assert set(prototypes["prototype_method"]) == {PROTOTYPE_METHOD_NORMALIZED_MEAN}
    assert global_row["reference_count"] == 2
    assert global_row["independent_observation_count"] == 1
    assert global_row["embedding"] == pytest.approx(_unit((1.0, 1.0, 0.0)), abs=1e-6)
    assert global_row["embedding"][2] == 0.0


def test_balanced_centering_context_and_query_transform_are_deterministic(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "target-b",
                "target-observation-b",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.8, 0.6, 0.0),
            ),
            _spec(
                "target-c",
                "target-observation-c",
                TARGET,
                "Papilio demoleus",
                "cluster-b",
                (0.6, 0.8, 0.0),
            ),
            _spec(
                "competitor-a",
                "competitor-observation-a",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
        ),
    )

    first = build_reference_centering_contexts(
        embeddings,
        balanced_sampling_seed=23,
    )
    second = build_reference_centering_contexts(
        embeddings,
        balanced_sampling_seed=23,
    )

    assert first == second
    assert len(first) == 1
    context = first[0]
    assert context.route == "adult_field"
    assert context.visual_input_kind == RAW_FULL_IMAGE_KIND
    assert context.species_count == 2
    assert context.independent_observation_count == 2
    assert context.reference_count == 2
    assert context.balanced_sampling_seed == 23
    assert len(context.selected_observation_ids) == 2
    assert "competitor-observation-a" in context.selected_observation_ids
    assert sum(value * value for value in context.mean_embedding) > 0.0
    assert context.centering_fingerprint.startswith("sha256:")
    with pytest.raises(ValueError, match="does not match"):
        replace(context, centering_fingerprint=_sha("wrong-centering"))
    vector_by_observation = {
        "target-observation-a": _unit((1.0, 0.0, 0.0)),
        "target-observation-b": _unit((0.8, 0.6, 0.0)),
        "target-observation-c": _unit((0.6, 0.8, 0.0)),
        "competitor-observation-a": _unit((0.0, 1.0, 0.0)),
    }
    expected_mean = tuple(
        sum(
            vector_by_observation[item][index]
            for item in context.selected_observation_ids
        )
        / len(context.selected_observation_ids)
        for index in range(3)
    )
    assert context.mean_embedding == pytest.approx(expected_mean, abs=1e-12)

    transformed = mean_center_query_embedding((1.0, 0.0, 0.0), context)
    assert len(transformed) == 3
    assert isclose(sqrt(sum(value * value for value in transformed)), 1.0, abs_tol=1e-6)
    assert transformed == mean_center_query_embedding((1.0, 0.0, 0.0), context)


def test_query_equal_to_centering_mean_fails_without_a_direction() -> None:
    context = ReferenceCenteringContext(
        route="adult_field",
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        embedding_dimension=3,
        balanced_sampling_seed=42,
        species_count=2,
        reference_count=2,
        independent_observation_count=2,
        selected_observation_ids=("observation-a", "observation-b"),
        selected_observation_fingerprints=(
            _sha("observation-a"),
            _sha("observation-b"),
        ),
        mean_embedding=(1.0, 0.0, 0.0),
    )

    with pytest.raises(ValueError, match="non-zero norm"):
        mean_center_query_embedding((1.0, 0.0, 0.0), context)


def test_raw_prototypes_do_not_depend_on_balanced_centering_seed(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "target-b",
                "target-observation-b",
                TARGET,
                "Papilio demoleus",
                "cluster-b",
                (0.8, 0.6, 0.0),
            ),
            _spec(
                "competitor-a",
                "competitor-observation-a",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
        ),
    )

    first = build_reference_prototypes(embeddings, balanced_sampling_seed=1)
    second = build_reference_prototypes(embeddings, balanced_sampling_seed=2)
    first_raw = first.filter(
        pl.col("prototype_method") == PROTOTYPE_METHOD_NORMALIZED_MEAN
    )
    second_raw = second.filter(
        pl.col("prototype_method") == PROTOTYPE_METHOD_NORMALIZED_MEAN
    )

    assert_frame_equal(first_raw, second_raw)


def test_balanced_selection_rank_does_not_depend_on_embedding_content() -> None:
    item = ReferenceObservationEmbedding(
        accepted_taxon_key=TARGET,
        scientific_name="Papilio demoleus",
        geo_cluster_id="cluster-a",
        life_stage="adult",
        visual_domain="live_field",
        route="adult_field",
        visual_input_kind=RAW_FULL_IMAGE_KIND,
        reference_observation_id="stable-observation",
        duplicate_group_ids=("duplicate:stable-observation",),
        reference_count=1,
        embedding=(1.0, 0.0, 0.0),
        contributor_fingerprints=(_sha("contributor-a"),),
        observation_fingerprint=_sha("embedding-version-a"),
    )
    changed_embedding = replace(
        item,
        embedding=(0.0, 1.0, 0.0),
        contributor_fingerprints=(_sha("contributor-b"),),
        observation_fingerprint=_sha("embedding-version-b"),
    )

    assert reference_prototypes_module._balanced_observation_rank(  # noqa: SLF001
        item,
        balanced_sampling_seed=42,
    ) == reference_prototypes_module._balanced_observation_rank(  # noqa: SLF001
        changed_embedding,
        balanced_sampling_seed=42,
    )


def test_mean_centered_method_is_omitted_without_a_competitor(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "target-b",
                "target-observation-b",
                TARGET,
                "Papilio demoleus",
                "cluster-b",
                (0.8, 0.6, 0.0),
            ),
        ),
    )

    prototypes = build_reference_prototypes(embeddings)

    assert set(prototypes["prototype_method"]) == {PROTOTYPE_METHOD_NORMALIZED_MEAN}
    assert prototypes["mean_centered"].to_list() == [False, False, False]


def test_adult_and_larval_banks_have_independent_contexts_and_prototypes(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-adult",
                "target-adult-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "competitor-adult",
                "competitor-adult-observation",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
            _spec(
                "target-larva",
                "target-larva-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.0, 0.0, 1.0),
                life_stage="larva",
                route="larval",
            ),
            _spec(
                "competitor-larva",
                "competitor-larva-observation",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                _unit((1.0, 1.0, 0.0)),
                life_stage="larva",
                route="larval",
            ),
        ),
    )

    contexts = build_reference_centering_contexts(embeddings)
    prototypes = build_reference_prototypes(embeddings)

    assert {context.route for context in contexts} == {"adult_field", "larval"}
    assert contexts[0].centering_fingerprint != contexts[1].centering_fingerprint
    assert set(prototypes["route"]) == {"adult_field", "larval"}
    adult_target = prototypes.filter(
        (pl.col("accepted_taxon_key") == TARGET)
        & (pl.col("route") == "adult_field")
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
        & (pl.col("prototype_method") == PROTOTYPE_METHOD_NORMALIZED_MEAN)
    ).row(0, named=True)
    larval_target = prototypes.filter(
        (pl.col("accepted_taxon_key") == TARGET)
        & (pl.col("route") == "larval")
        & (pl.col("cluster_scope_type") == PROTOTYPE_SCOPE_GLOBAL)
        & (pl.col("prototype_method") == PROTOTYPE_METHOD_NORMALIZED_MEAN)
    ).row(0, named=True)
    assert adult_target["embedding"] == pytest.approx((1.0, 0.0, 0.0))
    assert larval_target["embedding"] == pytest.approx((0.0, 0.0, 1.0))


def test_no_support_train_rows_fail_closed(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "calibration-a",
                "calibration-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
                support_split="calibration",
            ),
        ),
    )

    with pytest.raises(ValueError, match="support_train"):
        build_reference_prototypes(embeddings)


def test_calibration_embeddings_do_not_enter_prototype_identity(
    tmp_path: Path,
) -> None:
    shared = (
        _spec(
            "target-support",
            "target-support-observation",
            TARGET,
            "Papilio demoleus",
            "cluster-a",
            (1.0, 0.0, 0.0),
        ),
    )
    first_embeddings = _embedding_artifact(
        tmp_path / "first",
        (
            *shared,
            _spec(
                "target-calibration",
                "target-calibration-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.0, 1.0, 0.0),
                support_split="calibration",
            ),
        ),
    )
    second_embeddings = _embedding_artifact(
        tmp_path / "second",
        (
            *shared,
            _spec(
                "target-calibration",
                "target-calibration-observation",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (0.0, 0.0, 1.0),
                support_split="calibration",
            ),
        ),
    )

    assert reference_embeddings_artifact_fingerprint(first_embeddings) != (
        reference_embeddings_artifact_fingerprint(second_embeddings)
    )
    assert_frame_equal(
        build_reference_prototypes(first_embeddings),
        build_reference_prototypes(second_embeddings),
    )


def test_write_load_and_artifact_fingerprint_round_trip(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
            _spec(
                "competitor-a",
                "competitor-observation-a",
                COMPETITOR,
                "Papilio polytes",
                "cluster-a",
                (0.0, 1.0, 0.0),
            ),
        ),
    )
    prototypes = build_reference_prototypes(embeddings, balanced_sampling_seed=31)

    path = write_reference_prototypes(prototypes, tmp_path / "published")
    loaded = load_reference_prototypes(
        tmp_path / "published",
        expected_model_fingerprint=str(prototypes["model_fingerprint"][0]),
        expected_reference_embedding_fingerprint=(
            reference_embeddings_artifact_fingerprint(
                embeddings.filter(pl.col("support_split") == "support_train")
            )
        ),
        expected_support_manifest_fingerprint=str(
            prototypes["support_manifest_fingerprint"][0]
        ),
    )

    assert path == tmp_path / "published" / REFERENCE_PROTOTYPES_FILE
    assert_frame_equal(loaded, prototypes)
    assert reference_prototypes_artifact_fingerprint(loaded) == (
        reference_prototypes_artifact_fingerprint(prototypes)
    )
    with pytest.raises(FileExistsError):
        write_reference_prototypes(prototypes, path)


def test_prototype_fingerprint_pins_little_endian_float_encoding(
    tmp_path: Path,
) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
        ),
    )
    prototypes = build_reference_prototypes(embeddings)
    row = prototypes.row(0, named=True)

    preimage = reference_prototypes_module._prototype_fingerprint_preimage(row)  # noqa: SLF001 - verifies the normative binary identity.
    little = struct.pack("<d", float(row["embedding_norm"])) + b"".join(
        struct.pack("<f", float(value)) for value in row["embedding"]
    )
    big = struct.pack(">d", float(row["embedding_norm"])) + b"".join(
        struct.pack(">f", float(value)) for value in row["embedding"]
    )

    assert preimage.endswith(little)
    assert not preimage.endswith(big)
    assert row["prototype_fingerprint"] == (
        "sha256:" + hashlib.sha256(preimage).hexdigest()
    )


def test_validation_rejects_tampered_vector(tmp_path: Path) -> None:
    embeddings = _embedding_artifact(
        tmp_path,
        (
            _spec(
                "target-a",
                "target-observation-a",
                TARGET,
                "Papilio demoleus",
                "cluster-a",
                (1.0, 0.0, 0.0),
            ),
        ),
    )
    prototypes = build_reference_prototypes(embeddings)
    rows = prototypes.to_dicts()
    rows[0]["embedding"] = [0.0, 0.0, 0.0]
    tampered = pl.DataFrame(
        rows,
        schema=reference_prototypes_schema(3),
        orient="row",
        strict=True,
    )

    with pytest.raises(ValueError, match="embedding_norm|unit-normalized"):
        validate_reference_prototypes(tampered)


def _spec(
    media_id: str,
    observation_id: str,
    taxon_key: str,
    scientific_name: str,
    geo_cluster_id: str,
    vector: tuple[float, float, float],
    *,
    support_split: str = "support_train",
    life_stage: str = "adult",
    visual_domain: str = "live_field",
    route: str = "adult_field",
) -> _EmbeddingSpec:
    normalized = _unit(vector)
    return _EmbeddingSpec(
        media_id=media_id,
        observation_id=observation_id,
        taxon_key=taxon_key,
        scientific_name=scientific_name,
        geo_cluster_id=geo_cluster_id,
        vector=(normalized[0], normalized[1], normalized[2]),
        support_split=support_split,
        life_stage=life_stage,
        visual_domain=visual_domain,
        route=route,
    )


def _embedding_artifact(
    tmp_path: Path,
    specs: tuple[_EmbeddingSpec, ...],
) -> pl.DataFrame:
    support_rows: list[dict[str, object]] = []
    vectors: dict[str, tuple[float, float, float]] = {}
    for index, spec in enumerate(specs, start=1):
        source_path = _image(
            tmp_path / "source" / f"{spec.media_id}.png",
            (index % 255, (index + 1) % 255, (index + 2) % 255),
        )
        support_rows.append(_support_row(spec, source_path=source_path))
        vectors[f"{spec.media_id}.png"] = spec.vector
    manifest = pl.DataFrame(
        support_rows,
        schema=reference_support_manifest_schema(),
        orient="row",
        strict=True,
    ).sort(
        [
            "accepted_taxon_key",
            "geo_cluster_id",
            "route",
            "support_split",
            "reference_media_id",
        ]
    )
    visual_inputs: list[ReferenceVisualInput] = []
    for row in manifest.iter_rows(named=True):
        media_id = str(row["reference_media_id"])
        source_path = Path(unquote(urlsplit(str(row["source_object_uri"])).path))
        staged_path = tmp_path / f"{media_id}.png"
        staged_path.write_bytes(source_path.read_bytes())
        visual_inputs.append(
            ReferenceVisualInput.from_variant(
                reference_media_id=media_id,
                source_image_path=source_path,
                image_path=staged_path,
                variant=raw_full_frame_visual_input(_decoded_image(staged_path)),
            )
        )
    return build_reference_embeddings(
        manifest,
        visual_inputs,
        readiness_permit=_permit(manifest),
        scorer=_FakeScorer(vectors),
        embedding_created_at=NOW,
    )


def _support_row(
    spec: _EmbeddingSpec,
    *,
    source_path: Path,
) -> dict[str, object]:
    assigned_at = NOW - timedelta(days=1)
    row: dict[str, object] = {
        "schema_version": REFERENCE_SUPPORT_MANIFEST_SCHEMA_VERSION,
        "reference_bank_version": "reference-bank-v1",
        "registry_version": "butterflies-v1",
        "reference_media_id": spec.media_id,
        "canonical_reference_media_id": spec.media_id,
        "reference_observation_id": spec.observation_id,
        "review_request_id": f"review:{spec.media_id}",
        "review_decision_ids": [f"decision:{spec.media_id}"],
        "reviewer_ids": ["reviewer:fixture"],
        "source": "inaturalist",
        "source_observation_id": f"source:{spec.observation_id}",
        "provider_media_id": spec.media_id,
        "source_record_url": f"https://example.test/observations/{spec.observation_id}",
        "source_snapshot_version": "fixture-v1",
        "source_dataset_key": "dataset:fixture",
        "accepted_taxon_key": spec.taxon_key,
        "scientific_name": spec.scientific_name,
        "target_candidate": spec.taxon_key == TARGET,
        "geo_cluster_id": spec.geo_cluster_id,
        "observer_id": f"observer:{spec.observation_id}",
        "observed_at": NOW - timedelta(days=2),
        "latitude": -33.86,
        "longitude": 151.21,
        "source_object_uri": source_path.resolve().as_uri(),
        "image_sha256": _file_sha256(source_path),
        "perceptual_hash": f"{int(hashlib.sha256(spec.media_id.encode()).hexdigest()[:16], 16):016x}",
        "object_fingerprint": _sha(f"object:{spec.media_id}"),
        "duplicate_group_id": f"duplicate:{spec.media_id}",
        "duplicate_type": "exact",
        "creator": "Fixture Creator",
        "rights_holder": "Fixture Creator",
        "licence": "CC BY 4.0",
        "licence_uri": "https://creativecommons.org/licenses/by/4.0/",
        "licence_policy_status": "accepted",
        "attribution": "Fixture Creator / CC BY 4.0",
        "review_status": "completed",
        "verification_status": "verified",
        "target_identity_verified": True,
        "life_stage": spec.life_stage,
        "visual_domain": spec.visual_domain,
        "view": "dorsal",
        "route": spec.route,
        "support_split": spec.support_split,
        "support_eligible": True,
        "exclusion_reasons": [],
        "split_assignment_fingerprint": make_reference_split_assignment_fingerprint(
            reference_media_id=spec.media_id,
            split_version="split-v1",
            support_split=spec.support_split,
            included=True,
            exclusion_reason=None,
            assigned_by="fixture",
            assigned_at=assigned_at,
        ),
        "reference_bank_fingerprint": _sha("bank"),
    }
    row["support_row_fingerprint"] = readiness_module._support_row_fingerprint(  # noqa: SLF001 - fixture mirrors the persisted contract.
        row
    )
    return row


def _permit(manifest: pl.DataFrame) -> ReferenceBankReadinessPermit:
    contract = TargetPreprocessingContract()
    attestation = reference_embeddings_module._preprocessing_attestation_fingerprint(  # noqa: SLF001 - fixture must match production attestation.
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        open_clip_version=_FakeScorer.open_clip_version,
        preprocessing_config=PREPROCESSING_CONFIG,
        preprocessing_version=PREPROCESSING_ATTESTATION_VERSION,
    )
    identity = ReferenceModelInputIdentity(
        model_name=_FakeScorer.model_id,
        model_version="bioclip-2.5-vith14",
        model_revision=REVISION,
        checkpoint_uri="s3://models/bioclip-2.5-vith14/model.safetensors",
        checkpoint_sha256=WEIGHTS_SHA256,
        open_clip_version=_FakeScorer.open_clip_version,
        open_clip_config_sha256=OPEN_CLIP_CONFIG_SHA256,
        preprocessing_version=contract.version,
        preprocessing_contract_fingerprint=contract.fingerprint,
        preprocessing_attestation_fingerprint=attestation,
        input_contract_version=FULL_FRAME_VISUAL_INPUT_VERSION,
    )
    return ReferenceBankReadinessPermit(
        status="ready",
        registry_version="butterflies-v1",
        reference_bank_version="reference-bank-v1",
        target_accepted_taxon_key=TARGET,
        policy_fingerprint=_sha("policy"),
        bank_fingerprint=_sha("bank"),
        support_manifest_fingerprint=reference_support_manifest_fingerprint(manifest),
        summary_fingerprint=_sha("summary"),
        split_assignments_fingerprint=_sha("splits"),
        model_input_schema_version=identity.schema_version,
        model_name=identity.model_name,
        model_version=identity.model_version,
        model_revision=identity.model_revision,
        checkpoint_uri=identity.checkpoint_uri,
        checkpoint_sha256=identity.checkpoint_sha256,
        open_clip_version=identity.open_clip_version,
        open_clip_config_sha256=identity.open_clip_config_sha256,
        preprocessing_version=identity.preprocessing_version,
        preprocessing_contract_fingerprint=identity.preprocessing_contract_fingerprint,
        preprocessing_attestation_fingerprint=(
            identity.preprocessing_attestation_fingerprint
        ),
        input_contract_version=identity.input_contract_version,
        model_input_fingerprint=identity.fingerprint,
        readiness_sha256=_sha("readiness"),
        support_manifest_sha256=_sha("support-manifest-file"),
        summary_sha256=_sha("summary-file"),
    )


def _prototype_row(
    frame: pl.DataFrame,
    *,
    taxon_key: str,
    scope: str,
    method: str,
) -> dict[str, object]:
    selected = frame.filter(
        (pl.col("accepted_taxon_key") == taxon_key)
        & (pl.col("cluster_scope_type") == scope)
        & (pl.col("prototype_method") == method)
    )
    assert selected.height == 1
    return selected.row(0, named=True)


def _image(path: Path, colour: tuple[int, int, int]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 3), colour).save(path, format="PNG")
    return path


def _decoded_image(path: Path) -> DecodedImage:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        return DecodedImage(
            width=rgb.width,
            height=rgb.height,
            mode="RGB",
            data=rgb.tobytes(),
            source_uri=str(path),
        )


def _unit(values: tuple[float, float, float]) -> tuple[float, float, float]:
    norm = sqrt(sum(value * value for value in values))
    if norm <= 0.0:
        raise ValueError("fixture vector must be non-zero")
    return tuple(value / norm for value in values)  # type: ignore[return-value]


def _sha(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
