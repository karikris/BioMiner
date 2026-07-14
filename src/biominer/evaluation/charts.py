from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


FAMILY_CONFUSION_CHART_FILE = "family_confusion_matrix.png"
SPECIES_ACCURACY_BY_FAMILY_CHART_FILE = "species_accuracy_by_family.png"
CALIBRATION_RELIABILITY_CHART_FILE = "calibration_reliability.png"
REVIEW_REASON_COUNTS_CHART_FILE = "review_reason_counts.png"


def write_evaluation_charts(
    *,
    family_confusion: pl.DataFrame,
    species_accuracy_by_family: pl.DataFrame,
    review_error_examples: pl.DataFrame,
    output_dir: str | Path,
    calibration_bins: pl.DataFrame | None = None,
) -> dict[str, str]:
    """Write optional PNG charts for local classification evaluation reports."""
    plt = _pyplot()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths = {
        "family_confusion_chart": output / FAMILY_CONFUSION_CHART_FILE,
        "species_accuracy_by_family_chart": output / SPECIES_ACCURACY_BY_FAMILY_CHART_FILE,
        "review_reason_counts_chart": output / REVIEW_REASON_COUNTS_CHART_FILE,
    }
    _write_confusion_matrix_chart(
        plt,
        family_confusion,
        paths["family_confusion_chart"],
        title="Family Confusion Matrix",
    )
    _write_species_accuracy_by_family_chart(
        plt,
        species_accuracy_by_family,
        paths["species_accuracy_by_family_chart"],
    )
    _write_review_reason_counts_chart(
        plt,
        review_error_examples,
        paths["review_reason_counts_chart"],
    )

    if calibration_bins is not None and not calibration_bins.is_empty():
        calibration_path = output / CALIBRATION_RELIABILITY_CHART_FILE
        _write_calibration_reliability_chart(plt, calibration_bins, calibration_path)
        paths["calibration_reliability_chart"] = calibration_path

    return {key: str(path) for key, path in paths.items()}


def _pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - covered only when optional dep is absent.
        raise RuntimeError("matplotlib is required for --write-charts") from exc
    return plt


def _write_confusion_matrix_chart(plt: Any, frame: pl.DataFrame, path: Path, *, title: str) -> None:
    rows = _confusion_matrix_rows(frame)
    if not rows:
        _write_empty_chart(plt, path, title=title, message="no confusion rows")
        return

    true_labels = sorted({row["true_name"] for row in rows})
    predicted_labels = sorted({row["predicted_name"] for row in rows})
    counts = {
        (row["true_name"], row["predicted_name"]): int(row["count"])
        for row in rows
    }
    matrix = [
        [counts.get((true_label, predicted_label), 0) for predicted_label in predicted_labels]
        for true_label in true_labels
    ]
    width = max(5.0, min(12.0, 1.0 + 0.55 * len(predicted_labels)))
    height = max(4.0, min(10.0, 1.0 + 0.45 * len(true_labels)))
    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix)
    ax.set_title(title)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("Reviewed family")
    ax.set_xticks(range(len(predicted_labels)), labels=predicted_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(true_labels)), labels=true_labels)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    if len(true_labels) * len(predicted_labels) <= 100:
        for y_index, row in enumerate(matrix):
            for x_index, value in enumerate(row):
                if value:
                    ax.text(x_index, y_index, str(value), ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_species_accuracy_by_family_chart(plt: Any, frame: pl.DataFrame, path: Path) -> None:
    if frame.is_empty():
        _write_empty_chart(
            plt,
            path,
            title="Species Accuracy By Family",
            message="no matched species labels",
        )
        return

    rows = frame.sort(["total", "family"], descending=[True, False]).to_dicts()
    labels = [_text(row.get("family")) or _text(row.get("family_key")) for row in rows]
    values = [float(row.get("accuracy") or 0.0) for row in rows]
    totals = [int(row.get("total") or 0) for row in rows]
    height = max(3.0, 1.2 + 0.38 * len(rows))
    fig, ax = plt.subplots(figsize=(8.0, height))
    positions = list(range(len(rows)))
    ax.barh(positions, values)
    ax.set_title("Species Accuracy By Family")
    ax.set_xlabel("Species top1 accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    for position, value, total in zip(positions, values, totals, strict=True):
        ax.text(min(1.0, value + 0.02), position, f"{value:.2f} (n={total})", va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_calibration_reliability_chart(plt: Any, frame: pl.DataFrame, path: Path) -> None:
    rows = frame.sort("bin_index").to_dicts()
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    confidence = [float(row.get("avg_confidence") or 0.0) for row in rows]
    accuracy = [float(row.get("accuracy") or 0.0) for row in rows]
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="ideal")
    ax.plot(confidence, accuracy, marker="o", label="observed")
    ax.set_title("Calibration Reliability")
    ax.set_xlabel("Average confidence")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_review_reason_counts_chart(plt: Any, frame: pl.DataFrame, path: Path) -> None:
    rows = _review_reason_rows(frame)
    if not rows:
        _write_empty_chart(
            plt,
            path,
            title="Review Reason Counts",
            message="no review reasons",
        )
        return

    labels = [row["review_reason"] for row in rows]
    counts = [row["count"] for row in rows]
    height = max(3.0, 1.2 + 0.35 * len(rows))
    fig, ax = plt.subplots(figsize=(7.0, height))
    positions = list(range(len(rows)))
    ax.barh(positions, counts)
    ax.set_title("Review Reason Counts")
    ax.set_xlabel("Count")
    ax.set_yticks(positions, labels=labels)
    ax.invert_yaxis()
    for position, count in zip(positions, counts, strict=True):
        ax.text(count + 0.05, position, str(count), va="center")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_empty_chart(plt: Any, path: Path, *, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _confusion_matrix_rows(frame: pl.DataFrame) -> list[dict[str, object]]:
    if frame.is_empty():
        return []
    rows = []
    for row in frame.to_dicts():
        true_name = _text(row.get("true_name"))
        predicted_name = _text(row.get("predicted_name"))
        count = int(row.get("count") or 0)
        if true_name and predicted_name and count > 0:
            rows.append({"true_name": true_name, "predicted_name": predicted_name, "count": count})
    return rows


def _review_reason_rows(frame: pl.DataFrame) -> list[dict[str, object]]:
    if frame.is_empty() or "error_type" not in frame.columns:
        return []
    counts: dict[str, int] = {}
    for row in frame.to_dicts():
        reason = _text(row.get("error_type"))
        if reason:
            counts[reason] = counts.get(reason, 0) + 1
    return [
        {"review_reason": reason, "count": count}
        for reason, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    ]


def _text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


__all__ = [
    "CALIBRATION_RELIABILITY_CHART_FILE",
    "FAMILY_CONFUSION_CHART_FILE",
    "REVIEW_REASON_COUNTS_CHART_FILE",
    "SPECIES_ACCURACY_BY_FAMILY_CHART_FILE",
    "write_evaluation_charts",
]
