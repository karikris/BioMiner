from __future__ import annotations

from datetime import UTC, datetime
import json
from math import isfinite
from pathlib import Path
import re
import subprocess
from typing import IO, Any, Callable, Mapping, Sequence

from biominer.bioclip.diagnostics import (
    TRIAGE_LABEL_GROUPS,
    grouped_probability_summary,
    probability_entropy,
    topk_margin,
)
from biominer.bioclip.bioclip_worker import (
    OPENCLIP_PREPROCESSING_ATTESTATION_VERSION,
    canonical_preprocessing_config,
    normalize_image_resize_mode,
    preprocessing_attestation_fingerprint,
)
from biominer.bioclip.model_registry import BioClipRuntime
from biominer.bioclip.prompt_templates import (
    SPECIES_PROMPT_AGGREGATION_DEFAULT,
    PromptVariant,
    aggregate_prompt_scores,
)

BioClipScorer = Callable[[Path, Sequence[str]], Mapping[str, float]]
BioClipBatchScorer = Callable[
    [Sequence[Path], Sequence[str]], Sequence[Mapping[str, float]]
]
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]
LabelSets = Mapping[str, Sequence[str]]
_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}\Z")
_PREPROCESSING_ATTESTATION_FIELDS = frozenset(
    {
        "open_clip_version",
        "open_clip_config_sha256",
        "preprocessing_version",
        "preprocessing_config",
        "preprocessing_fingerprint",
    }
)


DEFAULT_TRIAGE_LABELS = (
    "a photo of an adult butterfly",
    "a photo of a swallowtail butterfly",
    "a photo of a butterfly",
    "a photo of a moth",
    "a photo of an egg",
    "a photo of a caterpillar",
    "a photo of a larva",
    "a photo of a pupa",
    "a photo of a chrysalis",
    "a photo of a pinned museum specimen",
    "a photo of artwork or illustration",
    "a photo of a tattoo",
    "an ai generated image",
    "a photo of a logo or brand",
    "a photo of an object",
    "a photo of a textile or pattern",
    "a photo of an insect that is not a butterfly or moth",
    "a photo that is not a lepidoptera",
)

DEFAULT_BIOCLIP_LABELS = DEFAULT_TRIAGE_LABELS

SWALLOWTAIL_VISUAL_LABELS = {
    "a photo of a swallowtail butterfly",
}

BUTTERFLY_VISUAL_LABELS = {
    *SWALLOWTAIL_VISUAL_LABELS,
    "a photo of a butterfly",
}

NON_WILD_OR_CONFLICT_LABELS = {
    "a photo of a moth",
    "a photo of an egg",
    "a photo of a caterpillar",
    "a photo of a larva",
    "a photo of a pupa",
    "a photo of a chrysalis",
    "a photo of a pinned museum specimen",
    "a photo of artwork or illustration",
    "a photo of a tattoo",
}


class BioClipClassifier:
    def __init__(
        self, *, runtime: BioClipRuntime, scorer: BioClipScorer | None = None
    ) -> None:
        self.runtime = runtime
        self._scorer = scorer

    def classify_image(
        self,
        *,
        flickr_photo_id: str,
        image_path: str | Path,
        image_hash: str,
        image_url_used: str,
        resolved_scientific_name: str,
        text_evidence_present: bool,
        labels: Sequence[str] = DEFAULT_BIOCLIP_LABELS,
        top_k: int = 10,
    ) -> dict[str, Any]:
        if not self.runtime.available:
            raise RuntimeError(
                f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}"
            )
        scores = self._score(Path(image_path), labels)
        topk = sorted(
            ((label, float(scores.get(label, 0.0))) for label in labels),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        return build_vision_prediction_record(
            flickr_photo_id=flickr_photo_id,
            runtime=self.runtime,
            image_hash=image_hash,
            image_url_used=image_url_used,
            resolved_scientific_name=resolved_scientific_name,
            text_evidence_present=text_evidence_present,
            topk=topk,
        )

    def classify_images(
        self,
        images: Sequence[dict[str, object]],
        *,
        labels: Sequence[str] = DEFAULT_BIOCLIP_LABELS,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        if not self.runtime.available:
            raise RuntimeError(
                f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}"
            )
        image_paths = [Path(str(image["image_path"])) for image in images]
        scores_by_image = self._score_batch(image_paths, labels)
        records: list[dict[str, Any]] = []
        for image, scores in zip(images, scores_by_image, strict=True):
            topk = sorted(
                ((label, float(scores.get(label, 0.0))) for label in labels),
                key=lambda item: item[1],
                reverse=True,
            )[:top_k]
            records.append(
                build_vision_prediction_record(
                    flickr_photo_id=str(image["flickr_photo_id"]),
                    runtime=self.runtime,
                    image_hash=str(image["image_hash"]),
                    image_url_used=str(image["image_url_used"]),
                    resolved_scientific_name=str(
                        image.get("resolved_scientific_name") or ""
                    ),
                    text_evidence_present=bool(image.get("text_evidence_present")),
                    topk=topk,
                )
            )
        return records

    def classify_images_with_label_sets(
        self,
        images: Sequence[dict[str, object]],
        *,
        label_sets: LabelSets,
        top_k: int = 10,
        species_prompt_variants: Sequence[PromptVariant] | None = None,
    ) -> list[dict[str, Any]]:
        if not self.runtime.available:
            raise RuntimeError(
                f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}"
            )
        image_paths = [Path(str(image["image_path"])) for image in images]
        scores_by_label_set = self._score_label_sets_batch(image_paths, label_sets)
        records: list[dict[str, Any]] = []
        for index, image in enumerate(images):
            topk_by_label_set: dict[str, list[tuple[str, float]]] = {}
            prompt_topk_by_label_set: dict[str, list[dict[str, object]]] = {}
            raw_scores_by_label_set: dict[str, Mapping[str, float]] = {}
            for label_set_name, labels in label_sets.items():
                scores = scores_by_label_set[label_set_name][index]
                raw_scores_by_label_set[label_set_name] = scores
                if label_set_name == "species" and species_prompt_variants:
                    aggregated = aggregate_prompt_scores(
                        scores=scores,
                        variants=species_prompt_variants,
                        top_k=top_k,
                        aggregation=SPECIES_PROMPT_AGGREGATION_DEFAULT,
                    )
                    topk_by_label_set[label_set_name] = [
                        (str(row["best_label"]), float(row["score"]))
                        for row in aggregated
                    ]
                    prompt_topk_by_label_set[label_set_name] = aggregated
                else:
                    topk_by_label_set[label_set_name] = sorted(
                        ((label, float(scores.get(label, 0.0))) for label in labels),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:top_k]
            records.append(
                build_label_set_prediction_record(
                    flickr_photo_id=str(image["flickr_photo_id"]),
                    runtime=self.runtime,
                    image_hash=str(image["image_hash"]),
                    image_url_used=str(image["image_url_used"]),
                    resolved_scientific_name=str(
                        image.get("resolved_scientific_name") or ""
                    ),
                    text_evidence_present=bool(image.get("text_evidence_present")),
                    topk_by_label_set=topk_by_label_set,
                    species_prompt_topk=prompt_topk_by_label_set.get("species", []),
                    triage_scores_by_label=raw_scores_by_label_set.get("triage", {}),
                )
            )
        return records

    def _score(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        if self._scorer is not None:
            return self._scorer(image_path, labels)
        return _score_with_open_clip(image_path, labels, self.runtime)

    def _score_batch(
        self, image_paths: Sequence[Path], labels: Sequence[str]
    ) -> Sequence[Mapping[str, float]]:
        if self._scorer is not None and hasattr(self._scorer, "score_batch"):
            return self._scorer.score_batch(image_paths, labels)  # type: ignore[attr-defined]
        if self._scorer is None:
            with PersistentBioClipScorer(runtime=self.runtime) as scorer:
                return scorer.score_batch(image_paths, labels)
        return [self._score(image_path, labels) for image_path in image_paths]

    def _score_label_sets_batch(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> dict[str, list[Mapping[str, float]]]:
        if self._scorer is not None and hasattr(self._scorer, "score_label_sets_batch"):
            return self._scorer.score_label_sets_batch(image_paths, label_sets)  # type: ignore[attr-defined]
        if self._scorer is None:
            with PersistentBioClipScorer(runtime=self.runtime) as scorer:
                return scorer.score_label_sets_batch(image_paths, label_sets)
        return {
            name: [self._score(image_path, labels) for image_path in image_paths]
            for name, labels in label_sets.items()
        }


class ExternalBioClipScorer:
    def __init__(
        self,
        *,
        runtime: BioClipRuntime,
        worker_script: str | Path | None = None,
        hf_cache_dir: str | Path = "data/cache/huggingface",
        runner: SubprocessRunner = subprocess.run,
        device: str = "auto",
        require_cuda: bool | None = None,
        image_resize_mode: str | None = None,
        preprocess_workers: int = 1,
    ) -> None:
        self.runtime = runtime
        self.worker_script = (
            Path(worker_script)
            if worker_script is not None
            else _default_worker_script()
        )
        self.hf_cache_dir = Path(hf_cache_dir)
        self.runner = runner
        self.device = _coerce_worker_device(device=device, require_cuda=require_cuda)
        self.image_resize_mode = normalize_image_resize_mode(image_resize_mode)
        self.preprocess_workers = _positive_preprocess_workers(preprocess_workers)

    def __call__(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        return self.score_batch([image_path], labels)[0]

    def score_batch(
        self, image_paths: Sequence[Path], labels: Sequence[str]
    ) -> list[Mapping[str, float]]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "labels": list(labels),
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.device,
            "preprocess_workers": self.preprocess_workers,
        }
        if self.image_resize_mode is not None:
            request["image_resize_mode"] = self.image_resize_mode
        result = self.runner(
            [str(self.runtime.venv_python), str(self.worker_script)],
            input=json.dumps(request),
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"BioCLIP worker failed: {result.stderr.strip()}")
        payload = json.loads(result.stdout)
        _validated_worker_image_resize_mode(payload, self.image_resize_mode)
        if "scores_by_image" in payload:
            return [
                {str(label): float(score) for label, score in scores.items()}
                for scores in payload["scores_by_image"]
            ]
        return [
            {str(label): float(score) for label, score in payload["scores"].items()}
        ]

    def score_label_sets_batch(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> dict[str, list[Mapping[str, float]]]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "label_sets": {name: list(labels) for name, labels in label_sets.items()},
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.device,
            "preprocess_workers": self.preprocess_workers,
        }
        if self.image_resize_mode is not None:
            request["image_resize_mode"] = self.image_resize_mode
        result = self.runner(
            [str(self.runtime.venv_python), str(self.worker_script)],
            input=json.dumps(request),
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"BioCLIP worker failed: {result.stderr.strip()}")
        payload = json.loads(result.stdout)
        _validated_worker_image_resize_mode(payload, self.image_resize_mode)
        return _coerce_label_set_scores(payload["scores_by_image_by_label_set"])


def classify_species_agreement(
    *,
    resolved_scientific_name: str,
    topk_labels: list[str],
    text_evidence_present: bool,
) -> str:
    if not topk_labels:
        return "text_only" if text_evidence_present else "uncertain"

    normalized_labels = {_normalize_label(label) for label in topk_labels}
    species_label = _normalize_label(f"a photo of {resolved_scientific_name}")
    if species_label in normalized_labels:
        return "exact_species_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {
        _normalize_label(label) for label in BUTTERFLY_VISUAL_LABELS
    }:
        return "same_family_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {
        _normalize_label(label) for label in NON_WILD_OR_CONFLICT_LABELS
    }:
        return "text_vision_conflict" if text_evidence_present else "non_butterfly"

    return "text_vision_conflict" if text_evidence_present else "uncertain"


def build_vision_prediction_record(
    *,
    flickr_photo_id: str,
    runtime: BioClipRuntime,
    image_hash: str,
    image_url_used: str,
    resolved_scientific_name: str,
    text_evidence_present: bool,
    topk: list[tuple[str, float]],
) -> dict[str, Any]:
    topk_json = [{"label": label, "score": float(score)} for label, score in topk]
    topk_labels = [item["label"] for item in topk_json]
    agreement_status = classify_species_agreement(
        resolved_scientific_name=resolved_scientific_name,
        topk_labels=topk_labels,
        text_evidence_present=text_evidence_present,
    )
    return {
        "flickr_photo_id": flickr_photo_id,
        "model_family": "bioclip",
        "model_name": runtime.model.model_name,
        "model_version": runtime.model.model_id,
        "model_checkpoint": runtime.model.checkpoint,
        "model_hash": runtime.model.model_hash,
        "runtime_package_version": runtime.package_version,
        "image_hash": image_hash,
        "image_url_used": image_url_used,
        "top1_label": topk_json[0]["label"] if topk_json else None,
        "top1_score": topk_json[0]["score"] if topk_json else None,
        "topk_json": topk_json,
        "species_agreement_status": agreement_status,
        "vision_review_required": _vision_review_required(agreement_status),
        "created_at": datetime.now(UTC).isoformat(),
    }


def build_label_set_prediction_record(
    *,
    flickr_photo_id: str,
    runtime: BioClipRuntime,
    image_hash: str,
    image_url_used: str,
    resolved_scientific_name: str,
    text_evidence_present: bool,
    topk_by_label_set: Mapping[str, list[tuple[str, float]]],
    species_prompt_topk: Sequence[Mapping[str, object]] = (),
    triage_scores_by_label: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    species_topk = _topk_json(topk_by_label_set.get("species", []))
    triage_topk = _topk_json(topk_by_label_set.get("triage", []))
    triage_group_summary = grouped_probability_summary(
        scores=triage_scores_by_label or {},
        groups=TRIAGE_LABEL_GROUPS,
    )
    species_scores = [float(row["score"]) for row in species_topk]
    triage_scores = [float(row["score"]) for row in triage_topk]
    compatibility_topk = species_topk or triage_topk
    agreement_status = classify_species_agreement(
        resolved_scientific_name=resolved_scientific_name,
        topk_labels=[str(item["label"]) for item in species_topk],
        text_evidence_present=text_evidence_present,
    )
    return {
        "flickr_photo_id": flickr_photo_id,
        "model_family": "bioclip",
        "model_name": runtime.model.model_name,
        "model_version": runtime.model.model_id,
        "model_checkpoint": runtime.model.checkpoint,
        "model_hash": runtime.model.model_hash,
        "runtime_package_version": runtime.package_version,
        "image_hash": image_hash,
        "image_url_used": image_url_used,
        "species_top1_label": species_topk[0]["label"] if species_topk else None,
        "species_top1_scientific_name": str(species_prompt_topk[0]["taxon_key"])
        if species_prompt_topk
        else None,
        "species_top1_score": species_topk[0]["score"] if species_topk else None,
        "species_topk_json": species_topk,
        "species_prompt_topk_json": [dict(row) for row in species_prompt_topk],
        "species_top1_top2_margin": topk_margin(species_topk),
        "species_topk_entropy": probability_entropy(species_scores),
        "triage_top1_label": triage_topk[0]["label"] if triage_topk else None,
        "triage_top1_score": triage_topk[0]["score"] if triage_topk else None,
        "triage_topk_json": triage_topk,
        "triage_top1_top2_margin": topk_margin(triage_topk),
        "triage_topk_entropy": probability_entropy(triage_scores),
        "triage_group_top": triage_group_summary["top_group"],
        "triage_group_scores": triage_group_summary["group_scores"],
        "top1_label": compatibility_topk[0]["label"] if compatibility_topk else None,
        "top1_score": compatibility_topk[0]["score"] if compatibility_topk else None,
        "topk_json": compatibility_topk,
        "species_agreement_status": agreement_status,
        "vision_review_required": _vision_review_required(agreement_status),
        "created_at": datetime.now(UTC).isoformat(),
    }


def _topk_json(topk: Sequence[tuple[str, float]]) -> list[dict[str, float | str]]:
    return [{"label": label, "score": float(score)} for label, score in topk]


def _vision_review_required(agreement_status: str) -> bool:
    return agreement_status not in {"exact_species_agreement", "same_genus_agreement"}


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _score_with_open_clip(
    image_path: Path, labels: Sequence[str], runtime: BioClipRuntime
) -> Mapping[str, float]:
    with PersistentBioClipScorer(runtime=runtime) as scorer:
        return scorer(image_path, labels)


def _default_worker_script() -> Path:
    return Path(__file__).with_name("bioclip_worker.py")


class PersistentBioClipScorer:
    def __init__(
        self,
        *,
        runtime: BioClipRuntime,
        worker_script: str | Path | None = None,
        hf_cache_dir: str | Path = "data/cache/huggingface",
        popen: PopenFactory = subprocess.Popen,
        device: str = "auto",
        require_cuda: bool | None = None,
        image_resize_mode: str | None = None,
        preprocess_workers: int = 1,
    ) -> None:
        self.runtime = runtime
        self.worker_script = (
            Path(worker_script)
            if worker_script is not None
            else _default_worker_script()
        )
        self.hf_cache_dir = Path(hf_cache_dir)
        self.popen = popen
        self.requested_device = _coerce_worker_device(
            device=device, require_cuda=require_cuda
        )
        self.image_resize_mode = normalize_image_resize_mode(image_resize_mode)
        self.preprocess_workers = _positive_preprocess_workers(preprocess_workers)
        self._process: subprocess.Popen[str] | None = None
        self._stdin: IO[str] | None = None
        self._stdout: IO[str] | None = None
        self.device: str | None = None
        self.gpu_name: str | None = None
        self.effective_image_resize_mode: str | None = None
        self.model_weights_sha256: str | None = None
        self.open_clip_version: str | None = None
        self.open_clip_config_sha256: str | None = None
        self.preprocessing_version: str | None = None
        self.preprocessing_config: dict[str, object] | None = None
        self.preprocessing_fingerprint: str | None = None
        self.last_image_content_hashes: list[str] | None = None
        self._pinned_reference_model_identity: dict[str, str] | None = None

    @property
    def model_id(self) -> str:
        return self.runtime.model.model_name.removeprefix("hf-hub:")

    @property
    def model_revision(self) -> str:
        return self.runtime.model.checkpoint

    def __enter__(self) -> "PersistentBioClipScorer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001 - context manager protocol.
        self.close()

    def __call__(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        return self.score_batch([image_path], labels)[0]

    def score_batch(
        self, image_paths: Sequence[Path], labels: Sequence[str]
    ) -> list[Mapping[str, float]]:
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "labels": list(labels),
            **self._worker_request_metadata(),
        }
        try:
            payload = self._request_payload(
                request,
                response_fields=("scores_by_image", "scores"),
                response_description="scores",
                require_model_attestation=True,
            )
            if "scores_by_image" in payload:
                return [
                    {str(label): float(score) for label, score in scores.items()}
                    for scores in payload["scores_by_image"]
                ]
            return [
                {str(label): float(score) for label, score in payload["scores"].items()}
            ]
        except Exception:
            self._discard_process()
            raise

    def score_label_sets_batch(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> dict[str, list[Mapping[str, float]]]:
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "label_sets": {name: list(labels) for name, labels in label_sets.items()},
            **self._worker_request_metadata(),
        }
        try:
            payload = self._request_payload(
                request,
                response_fields=("scores_by_image_by_label_set",),
                response_description="label-set scores",
                require_model_attestation=True,
            )
            return _coerce_label_set_scores(payload["scores_by_image_by_label_set"])
        except Exception:
            self._discard_process()
            raise

    def embed_text_labels(self, labels: Sequence[str]) -> list[list[float]]:
        request = {
            "text_labels": list(labels),
            **self._worker_request_metadata(),
        }
        try:
            payload = self._request_payload(
                request,
                response_fields=("text_embeddings",),
                response_description="text embeddings",
                require_model_attestation=True,
            )
            return [
                [float(value) for value in embedding]
                for embedding in payload["text_embeddings"]
            ]
        except Exception:
            self._discard_process()
            raise

    def ensure_model_attestation(self) -> None:
        request = {
            "probe": True,
            **self._worker_request_metadata(),
        }
        try:
            self._request_payload(
                request,
                response_fields=("probed",),
                response_description="model attestation probe",
                require_model_attestation=True,
            )
        except Exception:
            self._discard_process()
            raise

    def pin_reference_model_identity(
        self,
        *,
        model_weights_sha256: str,
        open_clip_version: str,
        open_clip_config_sha256: str,
        preprocessing_fingerprint: str,
        image_resize_mode: str,
    ) -> None:
        expected = {
            "model_weights_sha256": _required_fingerprint(
                model_weights_sha256,
                field="reference model weights SHA-256",
            ),
            "open_clip_version": _required_runtime_text(
                open_clip_version,
                field="reference OpenCLIP version",
            ),
            "open_clip_config_sha256": _required_fingerprint(
                open_clip_config_sha256,
                field="reference OpenCLIP config SHA-256",
            ),
            "preprocessing_fingerprint": _required_fingerprint(
                preprocessing_fingerprint,
                field="reference preprocessing fingerprint",
            ),
            "image_resize_mode": _required_runtime_text(
                image_resize_mode,
                field="reference image resize mode",
            ),
        }
        previous = self._pinned_reference_model_identity
        if previous is not None and previous != expected:
            raise RuntimeError("BioCLIP reference model identity is already pinned")
        self._pinned_reference_model_identity = expected
        self._validate_pinned_reference_model_identity()

    def embed_image_paths(self, image_paths: Sequence[Path]) -> list[list[float]]:
        if not image_paths:
            raise ValueError("at least one image path is required for embedding")
        request = {
            "image_embedding_paths": [str(image_path) for image_path in image_paths],
            **self._worker_request_metadata(),
        }
        try:
            payload = self._request_payload(
                request,
                response_fields=("image_embeddings",),
                response_description="image embeddings",
                require_model_attestation=True,
            )
            embeddings = _validated_image_embeddings(
                payload,
                expected_count=len(image_paths),
            )
            image_content_hashes = _validated_image_content_hashes(
                payload,
                expected_count=len(image_paths),
            )
            self.last_image_content_hashes = image_content_hashes
            return embeddings
        except Exception:
            self._discard_process()
            raise

    def _worker_request_metadata(self) -> dict[str, object]:
        metadata: dict[str, object] = {
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.requested_device,
            "preprocess_workers": self.preprocess_workers,
        }
        if self.image_resize_mode is not None:
            metadata["image_resize_mode"] = self.image_resize_mode
        return metadata

    def _request_payload(
        self,
        request: Mapping[str, object],
        *,
        response_fields: Sequence[str],
        response_description: str,
        require_model_attestation: bool = False,
    ) -> Mapping[str, object]:
        process = self._ensure_process()
        if process.poll() is not None:
            raise RuntimeError(
                "BioCLIP persistent worker exited early with code "
                f"{getattr(process, 'returncode', None)}"
            )
        if self._stdin is None or self._stdout is None:
            raise RuntimeError(
                "BioCLIP persistent worker did not expose stdin/stdout pipes"
            )
        self._stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._stdin.flush()
        ready_seen = False
        while True:
            line = self._stdout.readline()
            if not line:
                raise RuntimeError(
                    "BioCLIP persistent worker closed stdout before returning "
                    f"{response_description}"
                )
            decoded = json.loads(line)
            if not isinstance(decoded, Mapping):
                raise RuntimeError("BioCLIP worker response must be a JSON object")
            payload = dict(decoded)
            if "error" in payload:
                raise RuntimeError(f"BioCLIP worker failed: {payload['error']}")
            self._record_worker_metadata(payload)
            if require_model_attestation:
                self._require_frozen_worker_model_identity(payload)
            if payload.get("ready") is True:
                if ready_seen:
                    raise RuntimeError(
                        "BioCLIP worker returned duplicate ready responses"
                    )
                ready_seen = True
                continue
            if any(field in payload for field in response_fields):
                return payload
            raise RuntimeError(
                f"BioCLIP worker response did not include {response_description}"
            )

    def _record_worker_metadata(self, payload: Mapping[str, object]) -> None:
        if "device" in payload:
            self.device = str(payload.get("device") or "")
            self.gpu_name = str(payload.get("gpu_name") or "")
        self.effective_image_resize_mode = _validated_worker_image_resize_mode(
            payload,
            self.image_resize_mode,
            previous=self.effective_image_resize_mode,
        )
        if "model_id" in payload:
            model_id = str(payload.get("model_id") or "")
            if model_id != self.model_id:
                raise RuntimeError(
                    "BioCLIP worker model ID mismatch: "
                    f"requested {self.model_id!r}, got {model_id!r}"
                )
        if "model_revision" in payload:
            revision = str(payload.get("model_revision") or "")
            if revision != self.model_revision:
                raise RuntimeError(
                    "BioCLIP worker model revision mismatch: "
                    f"requested {self.model_revision!r}, got {revision!r}"
                )
        if "model_weights_sha256" in payload:
            raw_weights_sha256 = payload.get("model_weights_sha256")
            if raw_weights_sha256 is not None:
                weights_sha256 = str(raw_weights_sha256)
                if _SHA256_PATTERN.fullmatch(weights_sha256) is None:
                    raise RuntimeError(
                        "BioCLIP worker did not report a valid model weights SHA-256"
                    )
                if (
                    self.model_weights_sha256 is not None
                    and weights_sha256 != self.model_weights_sha256
                ):
                    raise RuntimeError("BioCLIP worker model weights SHA-256 changed")
                self.model_weights_sha256 = weights_sha256

        attestation_fields = _PREPROCESSING_ATTESTATION_FIELDS.intersection(payload)
        if not attestation_fields:
            self._validate_pinned_reference_model_identity()
            return
        missing_fields = _PREPROCESSING_ATTESTATION_FIELDS.difference(payload)
        if missing_fields:
            raise RuntimeError(
                "BioCLIP worker did not report complete preprocessing attestation: "
                + ", ".join(sorted(missing_fields))
            )
        open_clip_version = str(payload.get("open_clip_version") or "").strip()
        if not open_clip_version:
            raise RuntimeError("BioCLIP worker OpenCLIP version must be non-empty")
        if open_clip_version != self.runtime.package_version:
            raise RuntimeError(
                "BioCLIP worker OpenCLIP version mismatch: "
                f"expected {self.runtime.package_version!r}, got {open_clip_version!r}"
            )
        raw_config_sha256 = payload.get("open_clip_config_sha256")
        config_sha256 = None if raw_config_sha256 is None else str(raw_config_sha256)
        if (
            config_sha256 is not None
            and _SHA256_PATTERN.fullmatch(config_sha256) is None
        ):
            raise RuntimeError(
                "BioCLIP worker did not report a valid OpenCLIP config SHA-256"
            )
        preprocessing_version = str(payload.get("preprocessing_version") or "").strip()
        if preprocessing_version != OPENCLIP_PREPROCESSING_ATTESTATION_VERSION:
            raise RuntimeError(
                "BioCLIP worker preprocessing attestation version mismatch: "
                f"expected {OPENCLIP_PREPROCESSING_ATTESTATION_VERSION!r}, "
                f"got {preprocessing_version!r}"
            )
        try:
            preprocessing_config = canonical_preprocessing_config(
                payload.get("preprocessing_config")
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"BioCLIP worker preprocessing config is invalid: {exc}"
            ) from exc
        preprocessing_fingerprint = str(payload.get("preprocessing_fingerprint") or "")
        if _SHA256_PATTERN.fullmatch(preprocessing_fingerprint) is None:
            raise RuntimeError(
                "BioCLIP worker did not report a valid preprocessing fingerprint"
            )
        expected_fingerprint = preprocessing_attestation_fingerprint(
            open_clip_config_sha256=config_sha256,
            open_clip_version=open_clip_version,
            preprocessing_config=preprocessing_config,
            preprocessing_version=preprocessing_version,
        )
        if preprocessing_fingerprint != expected_fingerprint:
            raise RuntimeError("BioCLIP worker preprocessing fingerprint mismatch")

        if self.preprocessing_fingerprint is not None:
            attestation_values = (
                (
                    "OpenCLIP version",
                    self.open_clip_version,
                    open_clip_version,
                ),
                (
                    "OpenCLIP config SHA-256",
                    self.open_clip_config_sha256,
                    config_sha256,
                ),
                (
                    "preprocessing version",
                    self.preprocessing_version,
                    preprocessing_version,
                ),
                (
                    "preprocessing config",
                    self.preprocessing_config,
                    preprocessing_config,
                ),
                (
                    "preprocessing fingerprint",
                    self.preprocessing_fingerprint,
                    preprocessing_fingerprint,
                ),
            )
            for field, previous, current in attestation_values:
                if previous != current:
                    raise RuntimeError(f"BioCLIP worker {field} changed")

        self.open_clip_version = open_clip_version
        self.open_clip_config_sha256 = config_sha256
        self.preprocessing_version = preprocessing_version
        self.preprocessing_config = preprocessing_config
        self.preprocessing_fingerprint = preprocessing_fingerprint
        self._validate_pinned_reference_model_identity()

    def _validate_pinned_reference_model_identity(self) -> None:
        expected = self._pinned_reference_model_identity
        if expected is None:
            return
        actual = {
            "model_weights_sha256": self.model_weights_sha256,
            "open_clip_version": self.open_clip_version,
            "open_clip_config_sha256": self.open_clip_config_sha256,
            "preprocessing_fingerprint": self.preprocessing_fingerprint,
            "image_resize_mode": self.effective_image_resize_mode,
        }
        mismatches = sorted(
            field for field, value in actual.items() if value != expected[field]
        )
        if mismatches:
            raise RuntimeError(
                "BioCLIP worker no longer matches pinned reference model identity: "
                + ", ".join(mismatches)
            )

    def _require_frozen_worker_model_identity(
        self,
        payload: Mapping[str, object],
    ) -> None:
        if "model_id" not in payload or "model_revision" not in payload:
            raise RuntimeError(
                "BioCLIP worker did not report complete frozen model identity"
            )
        model_name = self.runtime.model.model_name
        is_hugging_face_model = model_name.startswith("hf-hub:") or (
            "/" in model_name and not model_name.startswith("local-dir:")
        )
        if is_hugging_face_model and (
            "model_weights_sha256" not in payload
            or payload.get("model_weights_sha256") is None
        ):
            raise RuntimeError(
                "BioCLIP worker did not report frozen Hugging Face model weights"
            )
        if not _PREPROCESSING_ATTESTATION_FIELDS.issubset(payload):
            raise RuntimeError(
                "BioCLIP worker did not report complete preprocessing attestation"
            )

    def _discard_process(self) -> None:
        process = self._process
        try:
            self._force_stop_process(process)
        finally:
            self._close_process_pipes(process)
            self._process = None
            self._stdin = None
            self._stdout = None
            self.device = None
            self.gpu_name = None
            self.effective_image_resize_mode = None
            self.model_weights_sha256 = None
            self.open_clip_version = None
            self.open_clip_config_sha256 = None
            self.preprocessing_version = None
            self.preprocessing_config = None
            self.preprocessing_fingerprint = None
            self.last_image_content_hashes = None

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        graceful = False
        try:
            running = process.poll() is None
        except Exception:  # noqa: BLE001 - cleanup must remain best effort.
            running = True
        if running and self._stdin is not None:
            try:
                self._stdin.write(json.dumps({"shutdown": True}, sort_keys=True) + "\n")
                self._stdin.flush()
                process.wait(timeout=10)
                graceful = True
            except Exception:  # noqa: BLE001 - shutdown must not mask caller errors.
                graceful = False
        elif not running:
            try:
                process.wait(timeout=10)
                graceful = True
            except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                graceful = False
        if not graceful:
            self._force_stop_process(process)
        self._close_process_pipes(process)
        self._process = None
        self._stdin = None
        self._stdout = None

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        if self._process is None:
            process = self.popen(
                [
                    str(self.runtime.venv_python),
                    str(self.worker_script),
                    "--persistent",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            self._process = process
            self._stdin = process.stdin
            self._stdout = process.stdout
            if process.stdin is None or process.stdout is None:
                self._discard_process()
                raise RuntimeError(
                    "BioCLIP persistent worker did not expose stdin/stdout pipes"
                )
        return self._process

    @staticmethod
    def _force_stop_process(process: object | None) -> None:
        if process is None:
            return
        try:
            running = process.poll() is None  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - cleanup must remain best effort.
            running = True
        if not running:
            try:
                process.wait(timeout=10)  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                pass
            return

        terminate_succeeded = False
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
                terminate_succeeded = True
            except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                pass
        if terminate_succeeded:
            try:
                process.wait(timeout=10)  # type: ignore[attr-defined]
                return
            except Exception:  # noqa: BLE001 - escalate to kill below.
                pass

        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                pass
        wait = getattr(process, "wait", None)
        if callable(wait):
            try:
                wait(timeout=10)
            except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                pass

    def _close_process_pipes(self, process: object | None) -> None:
        pipes = [self._stdin, self._stdout]
        if process is not None:
            pipes.extend(
                [
                    getattr(process, "stdin", None),
                    getattr(process, "stdout", None),
                    getattr(process, "stderr", None),
                ]
            )
        closed: set[int] = set()
        for pipe in pipes:
            if pipe is None or id(pipe) in closed:
                continue
            closed.add(id(pipe))
            close = getattr(pipe, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - cleanup must remain best effort.
                    pass


def _coerce_label_set_scores(
    payload: Mapping[str, Sequence[Mapping[str, object]]],
) -> dict[str, list[Mapping[str, float]]]:
    return {
        str(label_set_name): [
            {str(label): float(score) for label, score in scores.items()}
            for scores in scores_by_image
        ]
        for label_set_name, scores_by_image in payload.items()
    }


def _validated_image_embeddings(
    payload: Mapping[str, object],
    *,
    expected_count: int,
) -> list[list[float]]:
    raw_embeddings = payload.get("image_embeddings")
    if not isinstance(raw_embeddings, list):
        raise RuntimeError("BioCLIP worker image embeddings must be a list")
    if len(raw_embeddings) != expected_count:
        raise RuntimeError(
            f"BioCLIP worker returned {len(raw_embeddings)} rows for "
            f"{expected_count} images"
        )
    reported_dimension = payload.get("embedding_dim")
    if (
        isinstance(reported_dimension, bool)
        or not isinstance(reported_dimension, int)
        or reported_dimension <= 0
    ):
        raise RuntimeError(
            "BioCLIP worker reported embedding dimension must be a positive integer"
        )
    result: list[list[float]] = []
    row_dimensions: set[int] = set()
    for raw_embedding in raw_embeddings:
        if not isinstance(raw_embedding, list):
            raise RuntimeError("BioCLIP worker embedding rows must be lists")
        values: list[float] = []
        for raw_value in raw_embedding:
            if isinstance(raw_value, bool):
                raise RuntimeError(
                    "BioCLIP worker embedding vectors must contain finite values"
                )
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise RuntimeError(
                    "BioCLIP worker embedding vectors must contain finite values"
                ) from exc
            if not isfinite(value):
                raise RuntimeError(
                    "BioCLIP worker embedding vectors must contain finite values"
                )
            values.append(value)
        row_dimensions.add(len(values))
        result.append(values)
    if len(row_dimensions) != 1:
        raise RuntimeError("BioCLIP worker embedding dimension mismatch between rows")
    actual_dimension = next(iter(row_dimensions), 0)
    if actual_dimension != reported_dimension:
        raise RuntimeError(
            "BioCLIP worker reported embedding dimension does not match vector width"
        )
    return result


def _validated_image_content_hashes(
    payload: Mapping[str, object],
    *,
    expected_count: int,
) -> list[str]:
    raw_hashes = payload.get("image_content_hashes")
    if not isinstance(raw_hashes, list):
        raise RuntimeError("BioCLIP worker image content hashes must be a list")
    if len(raw_hashes) != expected_count:
        raise RuntimeError(
            f"BioCLIP worker returned {len(raw_hashes)} image content hashes for "
            f"{expected_count} images"
        )
    hashes: list[str] = []
    for raw_hash in raw_hashes:
        if not isinstance(raw_hash, str) or _SHA256_PATTERN.fullmatch(raw_hash) is None:
            raise RuntimeError(
                "BioCLIP worker image content hashes must be valid SHA-256 values"
            )
        hashes.append(raw_hash)
    return hashes


def _validated_worker_image_resize_mode(
    payload: Mapping[str, object],
    requested: str | None,
    *,
    previous: str | None = None,
) -> str | None:
    if "image_resize_mode" not in payload:
        if requested is not None:
            raise RuntimeError(
                "BioCLIP worker did not report the requested image resize mode"
            )
        return previous

    value = payload.get("image_resize_mode")
    effective = normalize_image_resize_mode(None if value is None else str(value))
    if requested is not None and effective != requested:
        raise RuntimeError(
            "BioCLIP worker image resize mode mismatch: "
            f"requested {requested!r}, got {effective!r}"
        )
    return effective


def _coerce_worker_device(*, device: str, require_cuda: bool | None) -> str:
    if require_cuda is True and device == "auto":
        return "cuda"
    normalized = device.casefold().strip()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported BioCLIP device {device!r}")
    return normalized


def _positive_preprocess_workers(value: int) -> int:
    workers = int(value)
    if workers <= 0:
        raise ValueError("preprocess_workers must be positive")
    return workers


def _required_runtime_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise ValueError(f"{field} must be non-empty")
    return result


def _required_fingerprint(value: object, *, field: str) -> str:
    result = str(value or "")
    if _SHA256_PATTERN.fullmatch(result) is None:
        raise ValueError(f"{field} must be a valid SHA-256 fingerprint")
    return result
