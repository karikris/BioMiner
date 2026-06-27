from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import subprocess
from typing import IO, Any, Callable, Mapping, Sequence

from biominer.bioclip.diagnostics import TRIAGE_LABEL_GROUPS, grouped_probability_summary, probability_entropy, topk_margin
from biominer.bioclip.model_registry import BioClipRuntime
from biominer.bioclip.prompt_templates import PromptVariant, aggregate_prompt_scores

BioClipScorer = Callable[[Path, Sequence[str]], Mapping[str, float]]
BioClipBatchScorer = Callable[[Sequence[Path], Sequence[str]], Sequence[Mapping[str, float]]]
SubprocessRunner = Callable[..., subprocess.CompletedProcess[str]]
PopenFactory = Callable[..., subprocess.Popen[str]]
LabelSets = Mapping[str, Sequence[str]]


DEFAULT_BIOCLIP_LABELS = (
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
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


PAPILIO_DEMOLEUS_VISUAL_LABELS = {
    "a photo of Papilio demoleus",
    "a photo of lime butterfly",
    "a photo of chequered swallowtail",
    "a photo of citrus swallowtail",
}

SWALLOWTAIL_VISUAL_LABELS = {
    *PAPILIO_DEMOLEUS_VISUAL_LABELS,
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
    def __init__(self, *, runtime: BioClipRuntime, scorer: BioClipScorer | None = None) -> None:
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
            raise RuntimeError(f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}")
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
            raise RuntimeError(f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}")
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
                    resolved_scientific_name=str(image.get("resolved_scientific_name") or ""),
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
        return_image_embeddings: bool = False,
    ) -> list[dict[str, Any]]:
        if not self.runtime.available:
            raise RuntimeError(f"BioCLIP runtime is not available: {self.runtime.unavailable_reason}")
        image_paths = [Path(str(image["image_path"])) for image in images]
        image_embeddings: Sequence[Sequence[float]] = ()
        if return_image_embeddings:
            scores_by_label_set, image_embeddings = self._score_label_sets_batch_with_embeddings(image_paths, label_sets)
        else:
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
                    aggregated = aggregate_prompt_scores(scores=scores, variants=species_prompt_variants, top_k=top_k)
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
                    resolved_scientific_name=str(image.get("resolved_scientific_name") or ""),
                    text_evidence_present=bool(image.get("text_evidence_present")),
                    topk_by_label_set=topk_by_label_set,
                    species_prompt_topk=prompt_topk_by_label_set.get("species", []),
                    triage_scores_by_label=raw_scores_by_label_set.get("triage", {}),
                    image_embedding=image_embeddings[index] if index < len(image_embeddings) else None,
                )
            )
        return records

    def _score(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        if self._scorer is not None:
            return self._scorer(image_path, labels)
        return _score_with_open_clip(image_path, labels, self.runtime)

    def _score_batch(self, image_paths: Sequence[Path], labels: Sequence[str]) -> Sequence[Mapping[str, float]]:
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

    def _score_label_sets_batch_with_embeddings(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> tuple[dict[str, list[Mapping[str, float]]], Sequence[Sequence[float]]]:
        if self._scorer is not None and hasattr(self._scorer, "score_label_sets_batch_with_embeddings"):
            return self._scorer.score_label_sets_batch_with_embeddings(image_paths, label_sets)  # type: ignore[attr-defined]
        return self._score_label_sets_batch(image_paths, label_sets), ()


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
    ) -> None:
        self.runtime = runtime
        self.worker_script = Path(worker_script) if worker_script is not None else _default_worker_script()
        self.hf_cache_dir = Path(hf_cache_dir)
        self.runner = runner
        self.device = _coerce_worker_device(device=device, require_cuda=require_cuda)

    def __call__(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        return self.score_batch([image_path], labels)[0]

    def score_batch(self, image_paths: Sequence[Path], labels: Sequence[str]) -> list[Mapping[str, float]]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "labels": list(labels),
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.device,
        }
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
        if "scores_by_image" in payload:
            return [
                {str(label): float(score) for label, score in scores.items()}
                for scores in payload["scores_by_image"]
            ]
        return [{str(label): float(score) for label, score in payload["scores"].items()}]

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
        }
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
        return _coerce_label_set_scores(payload["scores_by_image_by_label_set"])

    def score_label_sets_batch_with_embeddings(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> tuple[dict[str, list[Mapping[str, float]]], Sequence[Sequence[float]]]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "label_sets": {name: list(labels) for name, labels in label_sets.items()},
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.device,
            "return_image_embeddings": True,
        }
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
        return _coerce_label_set_scores(payload["scores_by_image_by_label_set"]), _coerce_embeddings(payload.get("image_embeddings", []))


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

    if resolved_scientific_name == "Papilio demoleus":
        if normalized_labels & {_normalize_label(label) for label in PAPILIO_DEMOLEUS_VISUAL_LABELS}:
            return "exact_species_agreement" if text_evidence_present else "vision_only"
        if normalized_labels & {_normalize_label(label) for label in SWALLOWTAIL_VISUAL_LABELS}:
            return "same_family_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {_normalize_label(label) for label in BUTTERFLY_VISUAL_LABELS}:
        return "same_family_agreement" if text_evidence_present else "vision_only"

    if normalized_labels & {_normalize_label(label) for label in NON_WILD_OR_CONFLICT_LABELS}:
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
    image_embedding: Sequence[float] | None = None,
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
    record = {
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
        "species_top1_scientific_name": str(species_prompt_topk[0]["taxon_key"]) if species_prompt_topk else None,
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
    if image_embedding is not None:
        record["image_embedding"] = [float(value) for value in image_embedding]
        record["embedding_dimension"] = len(image_embedding)
        record["preprocessing_version"] = "open_clip_default"
    return record


def _topk_json(topk: Sequence[tuple[str, float]]) -> list[dict[str, float | str]]:
    return [{"label": label, "score": float(score)} for label, score in topk]


def _vision_review_required(agreement_status: str) -> bool:
    return agreement_status not in {"exact_species_agreement", "same_genus_agreement"}


def _normalize_label(value: str) -> str:
    return " ".join(value.casefold().split())


def _score_with_open_clip(image_path: Path, labels: Sequence[str], runtime: BioClipRuntime) -> Mapping[str, float]:
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
    ) -> None:
        self.runtime = runtime
        self.worker_script = Path(worker_script) if worker_script is not None else _default_worker_script()
        self.hf_cache_dir = Path(hf_cache_dir)
        self.popen = popen
        self.requested_device = _coerce_worker_device(device=device, require_cuda=require_cuda)
        self._process: subprocess.Popen[str] | None = None
        self._stdin: IO[str] | None = None
        self._stdout: IO[str] | None = None
        self.device: str | None = None
        self.gpu_name: str | None = None

    def __enter__(self) -> "PersistentBioClipScorer":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:  # noqa: ANN001 - context manager protocol.
        self.close()

    def __call__(self, image_path: Path, labels: Sequence[str]) -> Mapping[str, float]:
        return self.score_batch([image_path], labels)[0]

    def score_batch(self, image_paths: Sequence[Path], labels: Sequence[str]) -> list[Mapping[str, float]]:
        process = self._ensure_process()
        if process.poll() is not None:
            raise RuntimeError(f"BioCLIP persistent worker exited early with code {process.returncode}")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "labels": list(labels),
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.requested_device,
        }
        assert self._stdin is not None
        assert self._stdout is not None
        self._stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._stdin.flush()
        while True:
            line = self._stdout.readline()
            if not line:
                raise RuntimeError("BioCLIP persistent worker closed stdout before returning scores")
            payload = json.loads(line)
            if "error" in payload:
                raise RuntimeError(f"BioCLIP worker failed: {payload['error']}")
            if payload.get("ready"):
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
                continue
            if "device" in payload:
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
            if "scores_by_image" in payload:
                return [
                    {str(label): float(score) for label, score in scores.items()}
                    for scores in payload["scores_by_image"]
                ]
            return [{str(label): float(score) for label, score in payload["scores"].items()}]

    def score_label_sets_batch(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> dict[str, list[Mapping[str, float]]]:
        process = self._ensure_process()
        if process.poll() is not None:
            raise RuntimeError(f"BioCLIP persistent worker exited early with code {process.returncode}")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "label_sets": {name: list(labels) for name, labels in label_sets.items()},
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.requested_device,
        }
        assert self._stdin is not None
        assert self._stdout is not None
        self._stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._stdin.flush()
        while True:
            line = self._stdout.readline()
            if not line:
                raise RuntimeError("BioCLIP persistent worker closed stdout before returning scores")
            payload = json.loads(line)
            if "error" in payload:
                raise RuntimeError(f"BioCLIP worker failed: {payload['error']}")
            if payload.get("ready"):
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
                continue
            if "device" in payload:
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
            if "scores_by_image_by_label_set" in payload:
                return _coerce_label_set_scores(payload["scores_by_image_by_label_set"])
            raise RuntimeError("BioCLIP worker response did not include label-set scores")

    def score_label_sets_batch_with_embeddings(
        self,
        image_paths: Sequence[Path],
        label_sets: LabelSets,
    ) -> tuple[dict[str, list[Mapping[str, float]]], Sequence[Sequence[float]]]:
        process = self._ensure_process()
        if process.poll() is not None:
            raise RuntimeError(f"BioCLIP persistent worker exited early with code {process.returncode}")
        request = {
            "image_paths": [str(image_path) for image_path in image_paths],
            "label_sets": {name: list(labels) for name, labels in label_sets.items()},
            "model_name": self.runtime.model.model_name,
            "checkpoint": self.runtime.model.checkpoint,
            "hf_cache_dir": str(self.hf_cache_dir),
            "device": self.requested_device,
            "return_image_embeddings": True,
        }
        assert self._stdin is not None
        assert self._stdout is not None
        self._stdin.write(json.dumps(request, sort_keys=True) + "\n")
        self._stdin.flush()
        while True:
            line = self._stdout.readline()
            if not line:
                raise RuntimeError("BioCLIP persistent worker closed stdout before returning scores")
            payload = json.loads(line)
            if "error" in payload:
                raise RuntimeError(f"BioCLIP worker failed: {payload['error']}")
            if payload.get("ready"):
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
                continue
            if "device" in payload:
                self.device = str(payload.get("device") or "")
                self.gpu_name = str(payload.get("gpu_name") or "")
            if "scores_by_image_by_label_set" in payload:
                return _coerce_label_set_scores(payload["scores_by_image_by_label_set"]), _coerce_embeddings(payload.get("image_embeddings", []))
            raise RuntimeError("BioCLIP worker response did not include label-set scores")

    def close(self) -> None:
        process = self._process
        if process is None:
            return
        if process.poll() is None and self._stdin is not None:
            try:
                self._stdin.write(json.dumps({"shutdown": True}, sort_keys=True) + "\n")
                self._stdin.flush()
                process.wait(timeout=10)
            except Exception:  # noqa: BLE001 - shutdown must not mask caller errors.
                process.terminate()
                process.wait(timeout=10)
        self._process = None
        self._stdin = None
        self._stdout = None

    def _ensure_process(self) -> subprocess.Popen[str]:
        if self.runtime.venv_python is None:
            raise RuntimeError("BioCLIP runtime does not define a Python executable")
        if self._process is None:
            process = self.popen(
                [str(self.runtime.venv_python), str(self.worker_script), "--persistent"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
            )
            if process.stdin is None or process.stdout is None:
                process.terminate()
                raise RuntimeError("BioCLIP persistent worker did not expose stdin/stdout pipes")
            self._process = process
            self._stdin = process.stdin
            self._stdout = process.stdout
        return self._process


def _coerce_label_set_scores(payload: Mapping[str, Sequence[Mapping[str, object]]]) -> dict[str, list[Mapping[str, float]]]:
    return {
        str(label_set_name): [
            {str(label): float(score) for label, score in scores.items()}
            for scores in scores_by_image
        ]
        for label_set_name, scores_by_image in payload.items()
    }


def _coerce_embeddings(payload: object) -> list[list[float]]:
    if not isinstance(payload, list):
        return []
    embeddings: list[list[float]] = []
    for row in payload:
        if isinstance(row, list):
            embeddings.append([float(value) for value in row])
    return embeddings


def _coerce_worker_device(*, device: str, require_cuda: bool | None) -> str:
    if require_cuda is True and device == "auto":
        return "cuda"
    normalized = device.casefold().strip()
    if normalized not in {"auto", "cuda", "mps", "cpu"}:
        raise ValueError(f"Unsupported BioCLIP device {device!r}")
    return normalized
