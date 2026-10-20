"""Opt-in paired Polars qualification using two freshly verified test runtimes."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess


_spec = importlib.util.spec_from_file_location(
    "polars_qualification", Path(__file__).resolve().parents[1] / "qualify_native_runtime.py",
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
CASE_MODULE = "scripts/upgrade_checks/test_polars_semantics_cases.py"


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _checked(reference):
    assert _reference(reference["path"]) == reference, "qualification input changed"
    return json.loads(Path(reference["path"]).read_bytes())


def _command(argv, output, name, cwd):
    # Keep provider credentials and accidental Python/pytest configuration out.
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output / "home"),
        "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PY_COLORS": "0",
        "POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "UV_NO_CACHE": "1",
    }
    result = native.run_test(argv, cwd, env, 900)
    log = output / f"{name}.log"
    with log.open("xb") as stream:
        stream.write(result.stdout)
    receipt = {"argv": argv, "cwd": str(cwd), "exit_code": result.returncode, "log": _reference(log)}
    with (output / f"{name}.json").open("x") as stream:
        json.dump(receipt, stream, sort_keys=True)
    assert result.returncode == 0, f"{name} failed; retained {log}"
    return result.stdout


def test_paired_polars_versions_and_public_artifacts():
    context_path = os.environ.get("BIOMINER_POLARS_SEMANTICS_CONTEXT")
    assert context_path, "explicit paired Polars qualification context required"
    context = json.loads(Path(context_path).read_bytes())
    assert set(context["runtimes"]) == {"p1a25", "p2a25"}
    assert len({r["profile"]["sha256"] for r in context["runtimes"].values()}) == 2
    output = Path(context["output"])
    output.mkdir(mode=0o700)
    cwd = native.validate_context(context)
    required = {
        "scripts/probe_upgrade_compatibility.py", "scripts/development_runtime.py",
        "tests/test_reference_downloader.py", "tests/test_frozen_embedding_classifiers.py",
        "tests/test_few_shot_training_features.py", CASE_MODULE,
        "tests/test_reference_prototypes.py", "tests/test_reference_scoring.py",
    }
    production = subprocess.check_output([
        "git", "-C", context["repo"], "ls-tree", "-r", "--name-only",
        context["tested_commit"], "--", "src/biominer",
    ], text=True, timeout=60).splitlines()
    required.update(name for name in production if name.endswith((".py", ".json")))
    assert required <= context["sources"].keys()
    assert Path(__file__).resolve() == cwd / CASE_MODULE

    profiles = {label: _checked(runtime["profile"]) for label, runtime in context["runtimes"].items()}
    arrow_identities = {native.arrow25_identity(profile) for profile in profiles.values()}
    assert len(arrow_identities) == 1, "paired runtimes must use the exact same Arrow wheel"
    arrow_version, arrow_wheel_sha256 = arrow_identities.pop()
    observations = {}
    for label, expected_polars in (("p1a25", "1.44.2"), ("p2a25", "2.0.0")):
        runtime = context["runtimes"][label]
        profile = profiles[label]
        development = _checked(runtime["development_receipt"])
        assert development["profile_sha256"] == runtime["profile"]["sha256"]
        assert development["production_ready"] is False
        checkout = Path(development["source_checkout"]).resolve()
        # The paired overlays may differ only in dependency configuration.
        for name, expected in context["sources"].items():
            if name not in {"pyproject.toml", "uv.lock"}:
                assert hashlib.sha256((checkout / name).read_bytes()).hexdigest() == expected
        python = str(Path(development["root"]) / "env/bin/python")
        verification = _command([
            profile["python"]["path"], "-I", "-B", str(cwd / "scripts/development_runtime.py"),
            "verify", "--root", development["root"],
            "--receipt-sha256", runtime["development_receipt"]["sha256"],
        ], output, f"{label}-verify", cwd)
        assert json.loads(verification)["production_ready"] is False
        identity = json.loads(_command([
            python, "-I", "-B", "-c",
            "import biominer,importlib.metadata as m,json,polars,pyarrow,sys;"
            "print(json.dumps({'python':sys.executable,'prefix':sys.prefix,"
            "'application':biominer.__file__,'versions':{n:m.version(n) "
            "for n in ('polars','polars-runtime-32','pyarrow')},"
            "'imported':{'polars':polars.__version__,'pyarrow':pyarrow.__version__},"
            "'installed':{d.metadata['Name'].lower().replace('_','-'):d.version "
            "for d in m.distributions()}}))",
        ], output, f"{label}-identity", cwd))
        assert identity["python"] == python
        assert Path(identity["prefix"]) == Path(development["root"]) / "env"
        assert Path(identity["application"]).resolve().is_relative_to(checkout / "src/biominer")
        assert identity["versions"] == {
            "polars": expected_polars, "polars-runtime-32": expected_polars,
            "pyarrow": arrow_version,
        }
        assert identity["imported"] == {"polars": expected_polars, "pyarrow": arrow_version}
        arguments = [
            python, "-I", "-B", str(cwd / "scripts/probe_upgrade_compatibility.py"),
            "--root", str(output / "paired"), "--label", label,
            "--source-sha", context["tested_commit"],
        ]
        if label == "p1a25":
            arguments += ["--create-fixtures"]
        _command(arguments, output, f"{label}-behavior", cwd)
        result_path = output / "paired/receipts" / f"{label}-behavior.json"
        result = json.loads(result_path.read_bytes())
        physical = {
            name: result["parquet"][name].pop("physical_sha256")
            for name in ("single", "batches")
        }
        assert result["label"] == label
        assert result["versions"]["polars"] == expected_polars
        result.pop("label")
        versions = result.pop("versions")
        observations[label] = {
            "identity": identity, "profile": runtime["profile"],
            "development_receipt": runtime["development_receipt"],
            "result": _reference(result_path), "physical_parquet_hashes": physical,
            "versions": versions, "logical": result,
        }

    # Physical encoding hashes stay in the original receipts. Only the explicit
    # physical Parquet fields and environment labels/versions are excluded here.
    assert {
        key: value for key, value in observations["p1a25"]["versions"].items() if key != "polars"
    } == {
        key: value for key, value in observations["p2a25"]["versions"].items() if key != "polars"
    }, "paired environments differ beyond the intended Polars upgrade"
    assert {
        key: value for key, value in observations["p1a25"]["identity"]["installed"].items()
        if key not in {"polars", "polars-runtime-32"}
    } == {
        key: value for key, value in observations["p2a25"]["identity"]["installed"].items()
        if key not in {"polars", "polars-runtime-32"}
    }, "installed distributions differ beyond the intended Polars upgrade"
    assert observations["p1a25"]["logical"] == observations["p2a25"]["logical"]
    native.validate_context(context)
    for runtime in context["runtimes"].values():
        _checked(runtime["profile"])
        development = _checked(runtime["development_receipt"])
        for name, expected in context["sources"].items():
            if name not in {"pyproject.toml", "uv.lock"}:
                assert hashlib.sha256(
                    (Path(development["source_checkout"]) / name).read_bytes(),
                ).hexdigest() == expected
    report = {
        "tested_commit": context["tested_commit"], "observations": observations,
        "arrow_wheel": {"version": arrow_version, "sha256": arrow_wheel_sha256},
        "logical_differences": [],
        "fixture": [_reference(p) for p in sorted((output / "paired/fixtures").iterdir())],
    }
    with (output / "comparison.json").open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
    print(json.dumps({"comparison": _reference(output / "comparison.json"), "logical_differences": []}))
