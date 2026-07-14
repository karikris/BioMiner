from __future__ import annotations

import json
from pathlib import Path
import socket

import polars as pl
import pytest

from biominer.cli import build_parser, run


def test_reference_review_commands_parse_exact_public_surface() -> None:
    parser = build_parser()

    export = parser.parse_args(
        [
            "references",
            "export-review-queue",
            "--acquisition-selections",
            "selections.parquet",
            "--observations",
            "observations.parquet",
            "--media-candidates",
            "candidates.parquet",
            "--media-objects",
            "objects.parquet",
            "--duplicate-relationships",
            "relationships.parquet",
            "--deduplication-report",
            "deduplication-report.json",
            "--reference-bank-version",
            "bank-v1",
            "--output-dir",
            "review-export",
            "--history-head",
            "export-history-head.json",
            "--run-id",
            "export-run",
            "--include-research-only",
        ]
    )
    imported = parser.parse_args(
        [
            "references",
            "import-review-decisions",
            "--review-queue",
            "queue.parquet",
            "--queue-provenance",
            "queue-provenance.parquet",
            "--decisions",
            "decisions.parquet",
            "--existing-decisions",
            "existing.parquet",
            "--prior-review-report",
            "prior-review-report.json",
            "--history-head",
            "import-history-head.json",
            "--output-dir",
            "review-import",
            "--run-id",
            "import-run",
        ]
    )

    assert export.command == "references"
    assert export.references_command == "export-review-queue"
    assert export.acquisition_selections == "selections.parquet"
    assert export.observations == "observations.parquet"
    assert export.media_candidates == "candidates.parquet"
    assert export.media_objects == "objects.parquet"
    assert export.duplicate_relationships == "relationships.parquet"
    assert export.deduplication_report == "deduplication-report.json"
    assert export.reference_bank_version == "bank-v1"
    assert export.output_dir == "review-export"
    assert export.history_head == "export-history-head.json"
    assert export.run_id == "export-run"
    assert export.include_research_only is True
    assert imported.command == "references"
    assert imported.references_command == "import-review-decisions"
    assert imported.review_queue == "queue.parquet"
    assert imported.queue_provenance == "queue-provenance.parquet"
    assert imported.decisions == "decisions.parquet"
    assert imported.existing_decisions == "existing.parquet"
    assert imported.prior_review_report == "prior-review-report.json"
    assert imported.history_head == "import-history-head.json"
    assert imported.output_dir == "review-import"
    assert imported.run_id == "import-run"


def test_reference_review_parser_defaults_optional_run_id_to_none() -> None:
    imported = build_parser().parse_args(
        [
            "references",
            "import-review-decisions",
            "--review-queue",
            "queue.parquet",
            "--queue-provenance",
            "queue-provenance.parquet",
            "--decisions",
            "decisions.parquet",
            "--existing-decisions",
            "existing.parquet",
            "--prior-review-report",
            "prior-review-report.json",
            "--history-head",
            "history-head.json",
            "--output-dir",
            "review-import",
        ]
    )

    assert imported.run_id is None


def test_export_reference_review_queue_runs_from_local_parquet_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = {
        name: _write_marker_frame(tmp_path / f"{name}.parquet", name)
        for name in (
            "selections",
            "objects",
            "candidates",
            "observations",
            "relationships",
        )
    }
    deduplication_report = _write_marker_json(
        tmp_path / "deduplication-report.json",
        "deduplication-report",
    )
    output_dir = tmp_path / "export"
    history_head = tmp_path / "history-head.json"
    sentinel = object()
    calls: dict[str, object] = {}

    def fake_build(
        selections: pl.DataFrame,
        media_objects: pl.DataFrame,
        media_candidates: pl.DataFrame,
        observations: pl.DataFrame,
        duplicate_relationships: pl.DataFrame,
        *,
        deduplication_report: dict[str, object],
        reference_bank_version: str,
        include_research_only: bool,
    ) -> object:
        calls["markers"] = [
            selections["marker"].item(),
            media_objects["marker"].item(),
            media_candidates["marker"].item(),
            observations["marker"].item(),
            duplicate_relationships["marker"].item(),
        ]
        calls["deduplication_report"] = deduplication_report["marker"]
        calls["reference_bank_version"] = reference_bank_version
        calls["include_research_only"] = include_research_only
        return sentinel

    def fake_write(
        result: object,
        destination: Path,
        *,
        run_id: str | None,
    ) -> dict[str, Path]:
        assert result is sentinel
        calls["run_id"] = run_id
        destination.mkdir()
        queue = destination / "reference_review_queue.parquet"
        report = destination / "reference_review_export_report.json"
        _write_marker_frame(queue, "queue")
        _write_marker_json(report, "export-report")
        return {"queue": queue, "report": report}

    def fake_initialize(head_path: str | Path, report_path: str | Path) -> None:
        calls["initialize_history"] = [str(head_path), str(report_path)]

    def fake_validate_destination(
        head_path: str | Path,
        packet_directory: str | Path,
    ) -> None:
        calls["validate_history_destination"] = [
            str(head_path),
            str(packet_directory),
        ]

    monkeypatch.setattr(socket, "create_connection", _unexpected_network)
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_history_head_destination",
        fake_validate_destination,
    )
    monkeypatch.setattr("biominer.cli.build_reference_review_queue", fake_build)
    monkeypatch.setattr("biominer.cli.write_reference_review_export", fake_write)
    monkeypatch.setattr(
        "biominer.cli.initialize_reference_review_history_head",
        fake_initialize,
    )

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "export-review-queue",
                "--acquisition-selections",
                str(paths["selections"]),
                "--observations",
                str(paths["observations"]),
                "--media-candidates",
                str(paths["candidates"]),
                "--media-objects",
                str(paths["objects"]),
                "--duplicate-relationships",
                str(paths["relationships"]),
                "--deduplication-report",
                str(deduplication_report),
                "--reference-bank-version",
                "bank-v1",
                "--output-dir",
                str(output_dir),
                "--history-head",
                str(history_head),
                "--run-id",
                "export-run",
            ]
        )
    )
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    assert rc == 0
    assert "\n" not in stdout
    assert calls == {
        "validate_history_destination": [str(history_head), str(output_dir)],
        "markers": [
            "selections",
            "objects",
            "candidates",
            "observations",
            "relationships",
        ],
        "deduplication_report": "deduplication-report",
        "reference_bank_version": "bank-v1",
        "include_research_only": False,
        "run_id": "export-run",
        "initialize_history": [
            str(history_head),
            str(output_dir / "reference_review_export_report.json"),
        ],
    }
    assert payload == {
        "artifacts": {
            "queue": str(output_dir / "reference_review_queue.parquet"),
            "report": str(output_dir / "reference_review_export_report.json"),
        },
        "command": "references export-review-queue",
        "output_dir": str(output_dir),
        "status": "complete",
    }


def test_export_rejects_history_head_inside_output_before_build_or_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "export"
    history_head = output_dir / "history-head.json"
    calls: list[str] = []

    def unexpected_build(*_args: object, **_kwargs: object) -> object:
        calls.append("build")
        raise AssertionError("queue build must follow history destination preflight")

    def unexpected_write(*_args: object, **_kwargs: object) -> dict[str, Path]:
        calls.append("write")
        raise AssertionError("packet write must follow history destination preflight")

    monkeypatch.setattr("biominer.cli.build_reference_review_queue", unexpected_build)
    monkeypatch.setattr("biominer.cli.write_reference_review_export", unexpected_write)

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "export-review-queue",
                "--acquisition-selections",
                str(tmp_path / "missing-selections.parquet"),
                "--observations",
                str(tmp_path / "missing-observations.parquet"),
                "--media-candidates",
                str(tmp_path / "missing-candidates.parquet"),
                "--media-objects",
                str(tmp_path / "missing-objects.parquet"),
                "--duplicate-relationships",
                str(tmp_path / "missing-relationships.parquet"),
                "--deduplication-report",
                str(tmp_path / "missing-deduplication-report.json"),
                "--reference-bank-version",
                "bank-v1",
                "--output-dir",
                str(output_dir),
                "--history-head",
                str(history_head),
            ]
        )
    )
    stdout = capsys.readouterr().out.strip()

    assert rc == 2
    assert "\n" not in stdout
    assert "outside immutable packet directories" in json.loads(stdout)["error"]
    assert calls == []
    assert not output_dir.exists()


def test_import_reference_review_decisions_runs_from_local_parquet_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queue = _write_marker_frame(tmp_path / "queue.parquet", "queue")
    provenance = _write_marker_frame(
        tmp_path / "queue-provenance.parquet",
        "queue-provenance",
    )
    decisions = _write_marker_frame(tmp_path / "decisions.parquet", "incoming")
    existing = _write_marker_frame(tmp_path / "existing.parquet", "existing")
    prior_report_path = _write_marker_json(
        tmp_path / "prior-review-report.json",
        "prior-review-report",
    )
    history_head = _write_marker_json(
        tmp_path / "history-head.json",
        "history-head",
    )
    output_dir = tmp_path / "import"
    sentinel = object()
    trusted_prior_report = {"marker": "trusted-prior-report"}
    trusted_prior_digest = "sha256:" + "a" * 64
    calls: dict[str, object] = {}

    def fake_validate_head(
        head_path: str | Path,
        report_path: str | Path,
    ) -> tuple[dict[str, object], str]:
        calls["validate_history"] = [str(head_path), str(report_path)]
        return trusted_prior_report, trusted_prior_digest

    def fake_validate_artifact(
        report: dict[str, object],
        logical_name: str,
        path: str | Path,
    ) -> None:
        assert report is trusted_prior_report
        calls.setdefault("validated_artifacts", []).append([logical_name, str(path)])

    def fake_import(
        raw: pl.DataFrame,
        *,
        queue: pl.DataFrame,
        queue_provenance: pl.DataFrame,
        existing_decisions: pl.DataFrame,
        prior_report: dict[str, object],
        prior_report_sha256: str,
    ) -> object:
        calls["markers"] = [
            raw["marker"].item(),
            queue["marker"].item(),
            queue_provenance["marker"].item(),
            existing_decisions["marker"].item(),
        ]
        calls["trusted_prior_report"] = prior_report is trusted_prior_report
        calls["trusted_prior_digest"] = prior_report_sha256
        return sentinel

    def fake_write(
        result: object,
        destination: Path,
        *,
        run_id: str | None,
    ) -> dict[str, Path]:
        assert result is sentinel
        calls["run_id"] = run_id
        destination.mkdir()
        ledger = destination / "reference_review_decisions.parquet"
        report = destination / "reference_review_import_report.json"
        _write_marker_frame(ledger, "ledger")
        _write_marker_json(report, "import-report")
        return {"decisions": ledger, "report": report}

    def fake_advance(
        head_path: str | Path,
        *,
        prior_report_path: str | Path,
        next_report_path: str | Path,
    ) -> None:
        calls["advance_history"] = [
            str(head_path),
            str(prior_report_path),
            str(next_report_path),
        ]

    monkeypatch.setattr(socket, "create_connection", _unexpected_network)
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_history_head",
        fake_validate_head,
    )
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_packet_artifact",
        fake_validate_artifact,
    )
    monkeypatch.setattr("biominer.cli.import_reference_review_decisions", fake_import)
    monkeypatch.setattr("biominer.cli.write_reference_review_import", fake_write)
    monkeypatch.setattr(
        "biominer.cli.advance_reference_review_history_head",
        fake_advance,
    )

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "import-review-decisions",
                "--review-queue",
                str(queue),
                "--queue-provenance",
                str(provenance),
                "--decisions",
                str(decisions),
                "--existing-decisions",
                str(existing),
                "--prior-review-report",
                str(prior_report_path),
                "--history-head",
                str(history_head),
                "--output-dir",
                str(output_dir),
                "--run-id",
                "import-run",
            ]
        )
    )
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    assert rc == 0
    assert "\n" not in stdout
    assert calls == {
        "validate_history": [str(history_head), str(prior_report_path)],
        "validated_artifacts": [
            ["queue", str(queue)],
            ["queue_provenance", str(provenance)],
            ["decisions", str(existing)],
        ],
        "markers": ["incoming", "queue", "queue-provenance", "existing"],
        "trusted_prior_report": True,
        "trusted_prior_digest": trusted_prior_digest,
        "run_id": "import-run",
        "advance_history": [
            str(history_head),
            str(prior_report_path),
            str(output_dir / "reference_review_import_report.json"),
        ],
    }
    assert payload == {
        "artifacts": {
            "decisions": str(output_dir / "reference_review_decisions.parquet"),
            "report": str(output_dir / "reference_review_import_report.json"),
        },
        "command": "references import-review-decisions",
        "output_dir": str(output_dir),
        "status": "complete",
    }


def test_reference_review_command_reports_missing_local_input_as_compact_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    provenance = _write_marker_frame(
        tmp_path / "queue-provenance.parquet",
        "queue-provenance",
    )
    existing = _write_marker_frame(
        tmp_path / "existing.parquet",
        "existing",
    )
    prior_report = _write_marker_json(
        tmp_path / "prior-review-report.json",
        "prior-review-report",
    )
    history_head = _write_marker_json(
        tmp_path / "history-head.json",
        "history-head",
    )
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_history_head",
        lambda *_args: ({"marker": "trusted-prior-report"}, "sha256:" + "a" * 64),
    )
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_packet_artifact",
        lambda *_args: None,
    )
    args = build_parser().parse_args(
        [
            "references",
            "import-review-decisions",
            "--review-queue",
            str(tmp_path / "missing-queue.parquet"),
            "--queue-provenance",
            str(provenance),
            "--decisions",
            str(tmp_path / "missing-decisions.parquet"),
            "--existing-decisions",
            str(existing),
            "--prior-review-report",
            str(prior_report),
            "--history-head",
            str(history_head),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    rc = run(args)
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    assert rc == 2
    assert "\n" not in stdout
    assert "decisions path does not exist" in payload["error"]
    assert not (tmp_path / "output").exists()


def test_reference_review_commands_report_filesystem_failures_as_compact_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = {
        name: _write_marker_frame(tmp_path / f"{name}.parquet", name)
        for name in (
            "selections",
            "objects",
            "candidates",
            "observations",
            "relationships",
            "queue",
            "queue-provenance",
            "decisions",
            "existing",
        )
    }
    deduplication_report = _write_marker_json(
        tmp_path / "deduplication-report.json",
        "deduplication-report",
    )
    prior_report = _write_marker_json(
        tmp_path / "prior-review-report.json",
        "prior-review-report",
    )
    history_head = _write_marker_json(
        tmp_path / "history-head.json",
        "history-head",
    )

    def denied_export(*_args: object, **_kwargs: object) -> dict[str, Path]:
        raise PermissionError("review export destination is read-only")

    def full_import(*_args: object, **_kwargs: object) -> dict[str, Path]:
        raise OSError("review import destination has no free space")

    monkeypatch.setattr(socket, "create_connection", _unexpected_network)
    monkeypatch.setattr(
        "biominer.cli.build_reference_review_queue",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "biominer.cli.import_reference_review_decisions",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_history_head",
        lambda *_args: ({"marker": "trusted-prior-report"}, "sha256:" + "a" * 64),
    )
    monkeypatch.setattr(
        "biominer.cli.validate_reference_review_packet_artifact",
        lambda *_args: None,
    )
    monkeypatch.setattr("biominer.cli.write_reference_review_export", denied_export)
    monkeypatch.setattr("biominer.cli.write_reference_review_import", full_import)

    export_rc = run(
        build_parser().parse_args(
            [
                "references",
                "export-review-queue",
                "--acquisition-selections",
                str(paths["selections"]),
                "--observations",
                str(paths["observations"]),
                "--media-candidates",
                str(paths["candidates"]),
                "--media-objects",
                str(paths["objects"]),
                "--duplicate-relationships",
                str(paths["relationships"]),
                "--deduplication-report",
                str(deduplication_report),
                "--reference-bank-version",
                "bank-v1",
                "--output-dir",
                str(tmp_path / "export"),
                "--history-head",
                str(tmp_path / "new-history-head.json"),
            ]
        )
    )
    export_stdout = capsys.readouterr().out.strip()

    import_rc = run(
        build_parser().parse_args(
            [
                "references",
                "import-review-decisions",
                "--review-queue",
                str(paths["queue"]),
                "--queue-provenance",
                str(paths["queue-provenance"]),
                "--decisions",
                str(paths["decisions"]),
                "--existing-decisions",
                str(paths["existing"]),
                "--prior-review-report",
                str(prior_report),
                "--history-head",
                str(history_head),
                "--output-dir",
                str(tmp_path / "import"),
            ]
        )
    )
    import_stdout = capsys.readouterr().out.strip()

    assert export_rc == 2
    assert "\n" not in export_stdout
    assert json.loads(export_stdout) == {
        "error": "review export destination is read-only"
    }
    assert import_rc == 2
    assert "\n" not in import_stdout
    assert json.loads(import_stdout) == {
        "error": "review import destination has no free space"
    }


def _write_marker_frame(path: Path, marker: str) -> Path:
    pl.DataFrame({"marker": [marker]}).write_parquet(path)
    return path


def _write_marker_json(path: Path, marker: str) -> Path:
    path.write_text(json.dumps({"marker": marker}), encoding="utf-8")
    return path


def _unexpected_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("reference review CLI must not access the network")
