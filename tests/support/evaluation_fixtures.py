"""Synthetic fixtures used exclusively by evaluation tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from biominer.bioclip.classification_modes import HIERARCHICAL_BUTTERFLY_CLASSIFICATION
from biominer.evaluation.labels import REVIEWED_LABEL_SCHEMA_VERSION
from biominer.storage.parquet import write_parquet


SYNTHETIC_EVALUATION_FIXTURE_VERSION = "evaluation-synthetic-v1"


@dataclass(frozen=True)
class SyntheticEvaluationFixture:
    classification_taxa: pl.DataFrame
    object_detections: pl.DataFrame
    object_scores: pl.DataFrame
    reviewed_labels: pl.DataFrame

    def to_frames(self) -> dict[str, pl.DataFrame]:
        return {
            "classification_taxa": self.classification_taxa,
            "object_detections": self.object_detections,
            "object_scores": self.object_scores,
            "reviewed_labels": self.reviewed_labels,
        }


def build_synthetic_evaluation_fixture() -> SyntheticEvaluationFixture:
    taxa = _classification_taxa()
    detections = _object_detections()
    scores = _object_scores()
    labels = _reviewed_labels()
    return SyntheticEvaluationFixture(
        classification_taxa=taxa,
        object_detections=detections,
        object_scores=scores,
        reviewed_labels=labels,
    )


def write_synthetic_evaluation_fixture(output_dir: str | Path) -> dict[str, str]:
    output = Path(output_dir)
    fixture = build_synthetic_evaluation_fixture()
    paths = {
        "classification_taxa": output / "evaluation_classification_taxa.parquet",
        "object_detections": output / "object_detections.parquet",
        "object_scores": output / "object_scores.parquet",
        "reviewed_labels": output / "reviewed_labels.parquet",
    }
    for name, path in paths.items():
        write_parquet(fixture.to_frames()[name], path)
    return {name: str(path) for name, path in paths.items()}


def _classification_taxa() -> pl.DataFrame:
    family_specs = (
        ("gbif:9001", "Papilionidae", "gbif:9100", "Papilio"),
        ("gbif:9002", "Nymphalidae", "gbif:9200", "Danaus"),
        ("gbif:9003", "Pieridae", "gbif:9300", "Pieris"),
    )
    rows: list[dict[str, object]] = []
    for family_key, family, genus_key, genus in family_specs:
        rows.append(
            {
                "fixture_version": SYNTHETIC_EVALUATION_FIXTURE_VERSION,
                "accepted_taxon_key": family_key,
                "scientific_name": family,
                "rank": "FAMILY",
                "family_key": family_key,
                "family": family,
                "genus_key": None,
                "genus": None,
                "classification_enabled": True,
            }
        )
        for index in range(1, 11):
            species_key = f"{family_key}:sp{index:02d}"
            rows.append(
                {
                    "fixture_version": SYNTHETIC_EVALUATION_FIXTURE_VERSION,
                    "accepted_taxon_key": species_key,
                    "scientific_name": f"{genus} synthetic{index:02d}",
                    "rank": "SPECIES",
                    "family_key": family_key,
                    "family": family,
                    "genus_key": genus_key,
                    "genus": genus,
                    "classification_enabled": True,
                }
            )
    return pl.DataFrame(rows).sort(["rank", "family", "scientific_name"])


def _object_detections() -> pl.DataFrame:
    rows = [
        _detection_row("photo-top1", "det-top1", "butterfly_like", 0.94, "sha256:crop-top1"),
        _detection_row("photo-top5", "det-top5", "butterfly_like", 0.93, "sha256:crop-top5"),
        _detection_row("photo-top20", "det-top20", "butterfly_like", 0.92, "sha256:crop-top20"),
        _detection_row("photo-wrong", "det-wrong", "butterfly_like", 0.91, "sha256:crop-wrong"),
        _detection_row("photo-negative", "det-negative", "hard_negative", 0.88, None),
    ]
    return pl.DataFrame(rows)


def _object_scores() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _score_row(
                photo_id="photo-top1",
                detection_id="det-top1",
                crop_hash="sha256:crop-top1",
                family="Papilionidae",
                family_key="gbif:9001",
                species_top20=["Papilio synthetic01", "Papilio synthetic02", "Papilio synthetic03"],
                species_top20_keys=["gbif:9001:sp01", "gbif:9001:sp02", "gbif:9001:sp03"],
                species_top5=["Papilio synthetic01", "Papilio synthetic02", "Papilio synthetic03"],
                species_top5_keys=["gbif:9001:sp01", "gbif:9001:sp02", "gbif:9001:sp03"],
                score=0.91,
                margin=0.40,
            ),
            _score_row(
                photo_id="photo-top5",
                detection_id="det-top5",
                crop_hash="sha256:crop-top5",
                family="Papilionidae",
                family_key="gbif:9001",
                species_top20=["Papilio synthetic03", "Papilio synthetic02", "Papilio synthetic04"],
                species_top20_keys=["gbif:9001:sp03", "gbif:9001:sp02", "gbif:9001:sp04"],
                species_top5=["Papilio synthetic03", "Papilio synthetic02", "Papilio synthetic04"],
                species_top5_keys=["gbif:9001:sp03", "gbif:9001:sp02", "gbif:9001:sp04"],
                score=0.56,
                margin=0.01,
            ),
            _score_row(
                photo_id="photo-top20",
                detection_id="det-top20",
                crop_hash="sha256:crop-top20",
                family="Nymphalidae",
                family_key="gbif:9002",
                species_top20=[
                    "Danaus synthetic02",
                    "Danaus synthetic03",
                    "Danaus synthetic04",
                    "Danaus synthetic05",
                    "Danaus synthetic06",
                    "Danaus synthetic07",
                    "Danaus synthetic08",
                    "Danaus synthetic09",
                    "Danaus synthetic10",
                    "Danaus synthetic01",
                ],
                species_top20_keys=[
                    "gbif:9002:sp02",
                    "gbif:9002:sp03",
                    "gbif:9002:sp04",
                    "gbif:9002:sp05",
                    "gbif:9002:sp06",
                    "gbif:9002:sp07",
                    "gbif:9002:sp08",
                    "gbif:9002:sp09",
                    "gbif:9002:sp10",
                    "gbif:9002:sp01",
                ],
                species_top5=[
                    "Danaus synthetic02",
                    "Danaus synthetic03",
                    "Danaus synthetic04",
                    "Danaus synthetic05",
                    "Danaus synthetic06",
                ],
                species_top5_keys=[
                    "gbif:9002:sp02",
                    "gbif:9002:sp03",
                    "gbif:9002:sp04",
                    "gbif:9002:sp05",
                    "gbif:9002:sp06",
                ],
                score=0.44,
                margin=0.08,
            ),
            _score_row(
                photo_id="photo-wrong",
                detection_id="det-wrong",
                crop_hash="sha256:crop-wrong",
                family="Pieridae",
                family_key="gbif:9003",
                family_top3=["Pieridae", "Nymphalidae", "Papilionidae"],
                family_top3_keys=["gbif:9003", "gbif:9002", "gbif:9001"],
                species_top20=["Pieris synthetic02", "Pieris synthetic03", "Pieris synthetic04"],
                species_top20_keys=["gbif:9003:sp02", "gbif:9003:sp03", "gbif:9003:sp04"],
                species_top5=["Pieris synthetic02", "Pieris synthetic03", "Pieris synthetic04"],
                species_top5_keys=["gbif:9003:sp02", "gbif:9003:sp03", "gbif:9003:sp04"],
                score=0.39,
                margin=0.04,
                text_candidate="Danaus synthetic02",
            ),
        ]
    )


def _reviewed_labels() -> pl.DataFrame:
    return pl.DataFrame(
        [
            _current_reviewed_label(
                _label_row(
                    "photo-top1",
                    "det-top1",
                    "sha256:crop-top1",
                    "gbif:9001:sp01",
                    "Papilio synthetic01",
                    "gbif:9001",
                    "Papilionidae",
                    "gbif:9100",
                    "Papilio",
                )
            ),
            _current_reviewed_label(
                _label_row(
                    "photo-top5",
                    "det-top5",
                    "sha256:crop-top5",
                    "gbif:9001:sp02",
                    "Papilio synthetic02",
                    "gbif:9001",
                    "Papilionidae",
                    "gbif:9100",
                    "Papilio",
                )
            ),
            _current_reviewed_label(
                _label_row(
                    "photo-top20",
                    "det-top20",
                    "sha256:crop-top20",
                    "gbif:9002:sp01",
                    "Danaus synthetic01",
                    "gbif:9002",
                    "Nymphalidae",
                    "gbif:9200",
                    "Danaus",
                )
            ),
            _current_reviewed_label(
                _label_row(
                    "photo-wrong",
                    "det-wrong",
                    "sha256:crop-wrong",
                    "gbif:9002:sp02",
                    "Danaus synthetic02",
                    "gbif:9002",
                    "Nymphalidae",
                    "gbif:9200",
                    "Danaus",
                )
            ),
            _current_reviewed_label(
                {
                    "source": "flickr",
                    "flickr_photo_id": "photo-negative",
                    "detection_id": "det-negative",
                    "crop_hash": "",
                    "label_level": "negative",
                    "is_butterfly": False,
                    "accepted_taxon_key": "",
                    "scientific_name": "",
                    "family_key": "",
                    "family": "",
                    "genus_key": "",
                    "genus": "",
                    "label_source": "synthetic_fixture",
                    "reviewer_id": "fixture",
                    "reviewed_at": "2026-07-10T00:00:00Z",
                    "review_confidence": "high",
                    "review_notes": "synthetic non-butterfly detector row",
                }
            ),
        ]
    )


def _current_reviewed_label(row: dict[str, object]) -> dict[str, object]:
    is_butterfly = row["is_butterfly"] is True
    is_species = row["label_level"] == "species"
    taxonomy_complete = bool(row["accepted_taxon_key"]) and bool(
        row["scientific_name"]
    )
    return {
        "schema_version": REVIEWED_LABEL_SCHEMA_VERSION,
        **row,
        "target_present": False if not is_butterfly else None,
        "label_certainty": row["review_confidence"],
        "life_stage": "unknown",
        "visual_domain": "ambiguous",
        "view": "unknown",
        "route": None,
        "geo_cluster_id": None,
        "source_query_tier": None,
        "source_query_term": None,
        "duplicate_group_id": None,
        "observer_owner_group_id": None,
        "dataset_split": "unassigned",
        "second_review_status": "unknown",
        "ambiguity_reason": "synthetic fixture",
        "unsuitable_for_species_identification": (
            False if is_butterfly and is_species and taxonomy_complete else None
        ),
    }


def _detection_row(
    photo_id: str,
    detection_id: str,
    label: str,
    score: float,
    crop_hash: str | None,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "detection_id": detection_id,
        "crop_hash": crop_hash,
        "detector_label": label,
        "detector_score": score,
        "objectness_score": score,
        "detection_status": "detected",
        "backend": "synthetic",
        "model_id": "synthetic-yoloe26",
        "model_version": SYNTHETIC_EVALUATION_FIXTURE_VERSION,
        "checkpoint": "none",
    }


def _score_row(
    *,
    photo_id: str,
    detection_id: str,
    crop_hash: str,
    family: str,
    family_key: str,
    species_top20: list[str],
    species_top20_keys: list[str],
    species_top5: list[str],
    species_top5_keys: list[str],
    score: float,
    margin: float,
    family_top3: list[str] | None = None,
    family_top3_keys: list[str] | None = None,
    text_candidate: str = "",
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "detection_id": detection_id,
        "crop_hash": crop_hash,
        "classification_mode": HIERARCHICAL_BUTTERFLY_CLASSIFICATION,
        "taxonomy_table_version": SYNTHETIC_EVALUATION_FIXTURE_VERSION,
        "family_top3": family_top3 or [family, "Nymphalidae", "Pieridae"],
        "family_top3_accepted_taxon_keys": family_top3_keys or [family_key, "gbif:9002", "gbif:9003"],
        "family_top3_scores": [0.90, 0.08, 0.02],
        "selected_family": family,
        "selected_family_key": family_key,
        "species_top20": species_top20,
        "species_top20_accepted_taxon_keys": species_top20_keys,
        "species_top5": species_top5,
        "species_top5_accepted_taxon_keys": species_top5_keys,
        "species_top5_scores": [score, max(0.0, score - margin), 0.10],
        "species_top1": species_top20[0],
        "species_top1_scientific_name": species_top20[0],
        "species_top1_accepted_taxon_key": species_top20_keys[0],
        "accepted_taxon_key": species_top20_keys[0],
        "species_top1_score": score,
        "species_top1_margin": margin,
        "detector_label": "butterfly_like",
        "detector_score": 0.90,
        "flickr_text_species_candidate": text_candidate,
        "occurrence_bin": "in_review",
        "bin_reason": "synthetic_evaluation_fixture",
    }


def _label_row(
    photo_id: str,
    detection_id: str,
    crop_hash: str,
    taxon_key: str,
    scientific_name: str,
    family_key: str,
    family: str,
    genus_key: str,
    genus: str,
) -> dict[str, object]:
    return {
        "source": "flickr",
        "flickr_photo_id": photo_id,
        "detection_id": detection_id,
        "crop_hash": crop_hash,
        "label_level": "species",
        "is_butterfly": True,
        "accepted_taxon_key": taxon_key,
        "scientific_name": scientific_name,
        "family_key": family_key,
        "family": family,
        "genus_key": genus_key,
        "genus": genus,
        "label_source": "synthetic_fixture",
        "reviewer_id": "fixture",
        "reviewed_at": "2026-07-10T00:00:00Z",
        "review_confidence": "high",
        "review_notes": "synthetic butterfly object label",
    }


__all__ = [
    "SYNTHETIC_EVALUATION_FIXTURE_VERSION",
    "SyntheticEvaluationFixture",
    "build_synthetic_evaluation_fixture",
    "write_synthetic_evaluation_fixture",
]
