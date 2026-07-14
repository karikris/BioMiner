from __future__ import annotations

import hashlib
import json
from pathlib import Path
import socket
from types import SimpleNamespace

import polars as pl
import pytest

from biominer.cli import build_parser, run


_PARQUET_ARGUMENTS = (
    ("candidate_species", "candidate-species"),
    ("acquisition_plan", "acquisition-plan"),
    ("acquisition_selections", "acquisition-selections"),
    ("observations", "observations"),
    ("media_candidates", "media-candidates"),
    ("media_objects", "media-objects"),
    ("duplicate_relationships", "duplicate-relationships"),
    ("review_queue", "review-queue"),
    ("queue_provenance", "queue-provenance"),
    ("review_decisions", "review-decisions"),
    ("split_assignments", "split-assignments"),
)


def test_reference_readiness_command_parses_exact_public_surface() -> None:
    args = build_parser().parse_args(
        [
            "references",
            "validate-readiness",
            "--candidate-species",
            "candidate-species.parquet",
            "--acquisition-plan",
            "acquisition-plan.parquet",
            "--acquisition-selections",
            "acquisition-selections.parquet",
            "--observations",
            "observations.parquet",
            "--media-candidates",
            "media-candidates.parquet",
            "--media-objects",
            "media-objects.parquet",
            "--duplicate-relationships",
            "duplicate-relationships.parquet",
            "--deduplication-report",
            "deduplication-report.json",
            "--review-queue",
            "review-queue.parquet",
            "--queue-provenance",
            "queue-provenance.parquet",
            "--review-decisions",
            "review-decisions.parquet",
            "--split-assignments",
            "split-assignments.parquet",
            "--readiness-policy",
            "readiness-policy.json",
            "--model-identity",
            "model-identity.json",
            "--registry-version",
            "registry-v1",
            "--reference-bank-version",
            "bank-v1",
            "--output-dir",
            "readiness",
            "--run-id",
            "readiness-run",
        ]
    )

    assert vars(args) == {
        "acquisition_plan": "acquisition-plan.parquet",
        "acquisition_selections": "acquisition-selections.parquet",
        "candidate_species": "candidate-species.parquet",
        "command": "references",
        "config": None,
        "deduplication_report": "deduplication-report.json",
        "duplicate_relationships": "duplicate-relationships.parquet",
        "media_candidates": "media-candidates.parquet",
        "media_objects": "media-objects.parquet",
        "model_identity": "model-identity.json",
        "observations": "observations.parquet",
        "output_dir": "readiness",
        "queue_provenance": "queue-provenance.parquet",
        "readiness_policy": "readiness-policy.json",
        "reference_bank_version": "bank-v1",
        "references_command": "validate-readiness",
        "registry_version": "registry-v1",
        "review_decisions": "review-decisions.parquet",
        "review_queue": "review-queue.parquet",
        "run_id": "readiness-run",
        "split_assignments": "split-assignments.parquet",
        "version": False,
    }


def test_reference_readiness_command_defaults_run_id_to_none() -> None:
    args = build_parser().parse_args(
        [
            "references",
            "validate-readiness",
            "--candidate-species",
            "candidate-species.parquet",
            "--acquisition-plan",
            "acquisition-plan.parquet",
            "--acquisition-selections",
            "acquisition-selections.parquet",
            "--observations",
            "observations.parquet",
            "--media-candidates",
            "media-candidates.parquet",
            "--media-objects",
            "media-objects.parquet",
            "--duplicate-relationships",
            "duplicate-relationships.parquet",
            "--deduplication-report",
            "deduplication-report.json",
            "--review-queue",
            "review-queue.parquet",
            "--queue-provenance",
            "queue-provenance.parquet",
            "--review-decisions",
            "review-decisions.parquet",
            "--split-assignments",
            "split-assignments.parquet",
            "--readiness-policy",
            "readiness-policy.json",
            "--model-identity",
            "model-identity.json",
            "--registry-version",
            "registry-v1",
            "--reference-bank-version",
            "bank-v1",
            "--output-dir",
            "readiness",
        ]
    )

    assert args.run_id is None


@pytest.mark.parametrize(
    ("readiness", "expected_rc", "expected_vision_permitted"),
    [
        ("ready", 0, True),
        ("blocked_missing_target_support", 2, False),
    ],
)
def test_reference_readiness_command_publishes_permitting_and_blocked_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    readiness: str,
    expected_rc: int,
    expected_vision_permitted: bool,
) -> None:
    paths = _write_readiness_inputs(tmp_path)
    output_dir = tmp_path / "readiness-output"
    parsed_policy = object()
    parsed_model_identity = object()
    readiness_payload = {"status": readiness}
    compiled = SimpleNamespace(readiness=readiness_payload)
    calls: dict[str, object] = {}

    class FakePolicy:
        @classmethod
        def from_mapping(cls, payload: dict[str, object]) -> object:
            calls["policy_payload"] = payload
            return parsed_policy

    class FakeModelIdentity:
        @classmethod
        def from_mapping(cls, payload: dict[str, object]) -> object:
            calls["model_identity_payload"] = payload
            return parsed_model_identity

    def fake_build(**kwargs: object) -> object:
        calls["table_markers"] = {
            name: value["marker"].item()
            for name, value in kwargs.items()
            if isinstance(value, pl.DataFrame)
        }
        calls["deduplication_report"] = kwargs["deduplication_report"]
        calls["policy"] = kwargs["policy"]
        calls["model_identity"] = kwargs["model_identity"]
        calls["registry_version"] = kwargs["registry_version"]
        calls["reference_bank_version"] = kwargs["reference_bank_version"]
        return compiled

    def fake_publish(
        result: object,
        destination: str | Path,
        *,
        run_id: str | None,
    ) -> dict[str, Path]:
        calls["publish"] = [result, str(destination), run_id]
        destination_path = Path(destination)
        destination_path.mkdir(parents=True)
        (destination_path / "reference_bank_readiness.json").write_bytes(
            b"published-readiness"
        )
        return {
            "readiness": Path(destination) / "reference_bank_readiness.json",
            "summary": Path(destination) / "reference_bank_summary.parquet",
            "support_manifest": Path(destination)
            / "reference_support_manifest.parquet",
        }

    monkeypatch.setattr(socket, "create_connection", _unexpected_network)
    monkeypatch.setattr("biominer.cli.ReferenceBankReadinessPolicy", FakePolicy)
    monkeypatch.setattr("biominer.cli.ReferenceModelInputIdentity", FakeModelIdentity)
    monkeypatch.setattr("biominer.cli.build_reference_bank_readiness", fake_build)
    monkeypatch.setattr("biominer.cli.publish_reference_bank_readiness", fake_publish)
    monkeypatch.setattr(
        "biominer.cli.reference_readiness_allows_vision",
        lambda value: value["status"] in {"ready", "ready_with_documented_shortfalls"},
    )

    rc = run(
        build_parser().parse_args(
            _readiness_command(paths, output_dir=output_dir, run_id="readiness-run")
        )
    )
    stdout = capsys.readouterr().out.strip()
    payload = json.loads(stdout)

    assert rc == expected_rc
    assert "\n" not in stdout
    assert calls == {
        "policy_payload": {"marker": "readiness-policy"},
        "model_identity_payload": {"marker": "model-identity"},
        "table_markers": {name: marker for name, marker in _PARQUET_ARGUMENTS},
        "deduplication_report": {"marker": "deduplication-report"},
        "policy": parsed_policy,
        "model_identity": parsed_model_identity,
        "registry_version": "registry-v1",
        "reference_bank_version": "bank-v1",
        "publish": [compiled, str(output_dir), "readiness-run"],
    }
    assert payload == {
        "artifacts": {
            "readiness": str(output_dir / "reference_bank_readiness.json"),
            "summary": str(output_dir / "reference_bank_summary.parquet"),
            "support_manifest": str(output_dir / "reference_support_manifest.parquet"),
        },
        "command": "references validate-readiness",
        "output_dir": str(output_dir),
            "readiness": readiness,
            "readiness_sha256": "sha256:"
            + hashlib.sha256(b"published-readiness").hexdigest(),
            "status": "complete",
        "vision_permitted": expected_vision_permitted,
    }


def test_reference_readiness_command_rejects_malformed_json_before_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_readiness_inputs(tmp_path)
    paths["readiness_policy"].write_text("{", encoding="utf-8")
    calls: list[str] = []

    monkeypatch.setattr(
        "biominer.cli.build_reference_bank_readiness",
        lambda **_kwargs: calls.append("build"),
    )
    monkeypatch.setattr(
        "biominer.cli.publish_reference_bank_readiness",
        lambda *_args, **_kwargs: calls.append("publish"),
    )

    rc = run(
        build_parser().parse_args(
            _readiness_command(paths, output_dir=tmp_path / "output")
        )
    )
    stdout = capsys.readouterr().out.strip()

    assert rc == 2
    assert "\n" not in stdout
    assert "Expecting property name" in json.loads(stdout)["error"]
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_reference_readiness_command_strictly_rejects_invalid_policy_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_readiness_inputs(tmp_path)
    calls: list[str] = []

    monkeypatch.setattr(
        "biominer.cli.build_reference_bank_readiness",
        lambda **_kwargs: calls.append("build"),
    )
    monkeypatch.setattr(
        "biominer.cli.publish_reference_bank_readiness",
        lambda *_args, **_kwargs: calls.append("publish"),
    )

    rc = run(
        build_parser().parse_args(
            _readiness_command(paths, output_dir=tmp_path / "output")
        )
    )
    stdout = capsys.readouterr().out.strip()
    error = json.loads(stdout)["error"]

    assert rc == 2
    assert "\n" not in stdout
    assert "reference bank readiness policy fields are invalid" in error
    assert "unknown: marker" in error
    assert calls == []
    assert not (tmp_path / "output").exists()


def test_reference_readiness_command_rejects_missing_table_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = _write_readiness_inputs(tmp_path)
    paths["candidate_species"].unlink()
    calls: list[str] = []

    class FakeConfig:
        @classmethod
        def from_mapping(cls, _payload: dict[str, object]) -> object:
            return object()

    monkeypatch.setattr("biominer.cli.ReferenceBankReadinessPolicy", FakeConfig)
    monkeypatch.setattr("biominer.cli.ReferenceModelInputIdentity", FakeConfig)
    monkeypatch.setattr(
        "biominer.cli.build_reference_bank_readiness",
        lambda **_kwargs: calls.append("build"),
    )
    monkeypatch.setattr(
        "biominer.cli.publish_reference_bank_readiness",
        lambda *_args, **_kwargs: calls.append("publish"),
    )

    rc = run(
        build_parser().parse_args(
            _readiness_command(paths, output_dir=tmp_path / "output")
        )
    )
    stdout = capsys.readouterr().out.strip()

    assert rc == 2
    assert "\n" not in stdout
    assert "candidate species path does not exist" in json.loads(stdout)["error"]
    assert calls == []
    assert not (tmp_path / "output").exists()


def _write_readiness_inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        name: _write_marker_frame(tmp_path / f"{marker}.parquet", marker)
        for name, marker in _PARQUET_ARGUMENTS
    }
    paths["deduplication_report"] = _write_marker_json(
        tmp_path / "deduplication-report.json",
        "deduplication-report",
    )
    paths["readiness_policy"] = _write_marker_json(
        tmp_path / "readiness-policy.json",
        "readiness-policy",
    )
    paths["model_identity"] = _write_marker_json(
        tmp_path / "model-identity.json",
        "model-identity",
    )
    return paths


def _readiness_command(
    paths: dict[str, Path],
    *,
    output_dir: Path,
    run_id: str | None = None,
) -> list[str]:
    command = ["references", "validate-readiness"]
    for name, _marker in _PARQUET_ARGUMENTS:
        command.extend(["--" + name.replace("_", "-"), str(paths[name])])
    command.extend(
        [
            "--deduplication-report",
            str(paths["deduplication_report"]),
            "--readiness-policy",
            str(paths["readiness_policy"]),
            "--model-identity",
            str(paths["model_identity"]),
            "--registry-version",
            "registry-v1",
            "--reference-bank-version",
            "bank-v1",
            "--output-dir",
            str(output_dir),
        ]
    )
    if run_id is not None:
        command.extend(["--run-id", run_id])
    return command


def _write_marker_frame(path: Path, marker: str) -> Path:
    pl.DataFrame({"marker": [marker]}).write_parquet(path)
    return path


def _write_marker_json(path: Path, marker: str) -> Path:
    path.write_text(json.dumps({"marker": marker}), encoding="utf-8")
    return path


def _unexpected_network(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("reference readiness CLI must not access the network")
