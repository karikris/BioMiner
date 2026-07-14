from __future__ import annotations

import json
from pathlib import Path
import socket

import pytest

from biominer.cli import build_parser, run


def test_visual_domain_negative_command_parses_exact_public_surface() -> None:
    args = build_parser().parse_args(
        [
            "references",
            "compile-visual-domain-negatives",
            "--source-manifest",
            "negatives.json",
            "--output-dir",
            "compiled-negatives",
            "--run-id",
            "negative-run",
        ]
    )

    assert args.command == "references"
    assert args.references_command == "compile-visual-domain-negatives"
    assert args.source_manifest == "negatives.json"
    assert args.output_dir == "compiled-negatives"
    assert args.run_id == "negative-run"


def test_visual_domain_negative_command_defaults_run_id_to_none() -> None:
    args = build_parser().parse_args(
        [
            "references",
            "compile-visual-domain-negatives",
            "--source-manifest",
            "negatives.json",
            "--output-dir",
            "compiled-negatives",
        ]
    )

    assert args.run_id is None


def test_visual_domain_negative_command_is_local_and_reports_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "negatives.json"
    source.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "compiled"
    calls: list[tuple[str, str, str | None]] = []

    def unexpected_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("negative manifest compilation must not use the network")

    def fake_publish(
        source_manifest: str | Path,
        output_dir: str | Path,
        *,
        run_id: str | None = None,
    ) -> dict[str, Path]:
        calls.append((str(source_manifest), str(output_dir), run_id))
        return {
            "manifest": Path(output_dir)
            / "reference_visual_domain_negative_manifest.parquet",
            "report": Path(output_dir)
            / "reference_visual_domain_negative_manifest_report.json",
        }

    monkeypatch.setattr(socket, "create_connection", unexpected_network)
    monkeypatch.setattr(
        "biominer.cli.publish_curated_visual_domain_negative_manifest",
        fake_publish,
    )

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "compile-visual-domain-negatives",
                "--source-manifest",
                str(source),
                "--output-dir",
                str(output),
                "--run-id",
                "negative-run",
            ]
        )
    )
    stdout = capsys.readouterr().out.strip()

    assert rc == 0
    assert "\n" not in stdout
    assert calls == [(str(source), str(output), "negative-run")]
    assert json.loads(stdout) == {
        "artifacts": {
            "manifest": str(
                output / "reference_visual_domain_negative_manifest.parquet"
            ),
            "report": str(
                output / "reference_visual_domain_negative_manifest_report.json"
            ),
        },
        "command": "references compile-visual-domain-negatives",
        "output_dir": str(output),
        "status": "complete",
    }


def test_visual_domain_negative_command_returns_compact_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def reject(*_args: object, **_kwargs: object) -> dict[str, Path]:
        raise TypeError("rights evidence must be a string")

    monkeypatch.setattr(
        "biominer.cli.publish_curated_visual_domain_negative_manifest",
        reject,
    )

    rc = run(
        build_parser().parse_args(
            [
                "references",
                "compile-visual-domain-negatives",
                "--source-manifest",
                str(tmp_path / "negatives.json"),
                "--output-dir",
                str(tmp_path / "compiled"),
            ]
        )
    )
    stdout = capsys.readouterr().out.strip()

    assert rc == 2
    assert "\n" not in stdout
    assert json.loads(stdout) == {"error": "rights evidence must be a string"}
