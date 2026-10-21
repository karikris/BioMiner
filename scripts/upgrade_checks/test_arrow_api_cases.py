"""Opt-in Arrow 26 depth qualification in an approved frozen generation.

Missing context fails. The shared storage suite remains usable with Arrow 25;
this module alone exercises Arrow 26's schema_depth_limit and default guard.
The same committed file supplies the offline worker; no external probe script
or pytest-environment Arrow package substitutes for the admitted runtime.
"""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys


_spec = importlib.util.spec_from_file_location(
    "arrow_api_qualification", Path(__file__).resolve().parents[1] / "qualify_native_runtime.py",
)
native = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(native)
CASE_MODULE = "scripts/upgrade_checks/test_arrow_api_cases.py"
SNAPSHOT_SOURCES = {
    "scripts/native_runtime.py", "scripts/freeze_native_profile.py",
    "scripts/check_native_runtime.py",
}


def _reference(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _checked(reference):
    assert _reference(reference["path"]) == reference, "Arrow qualification input changed"
    return json.loads(Path(reference["path"]).read_bytes())


def _save(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _validate_sources(context):
    cwd = native.validate_context(context)
    production = subprocess.check_output([
        "git", "-C", context["repo"], "ls-tree", "-r", "--name-only",
        context["tested_commit"], "--", "src/biominer",
    ], text=True, timeout=60).splitlines()
    required = {CASE_MODULE, "tests/test_storage_backends.py", *SNAPSHOT_SOURCES}
    required.update(name for name in production if name.endswith((".py", ".json")))
    assert required <= context["sources"].keys(), "Arrow qualification source closure incomplete"
    assert Path(__file__).resolve() == cwd / CASE_MODULE
    assert _reference(__file__)["sha256"] == context["sources"][CASE_MODULE]
    return cwd


def _runtime_inputs(context):
    runtime = context["runtime"]
    actual = native.runtime_configuration(
        runtime["snapshot"], runtime["snapshot_manifest"]["sha256"], runtime["generation"],
    )
    assert actual == runtime, "Arrow qualification runtime context mismatch"
    manifest = _checked(runtime["snapshot_manifest"])
    for name in SNAPSHOT_SOURCES:
        assert manifest["files"][name] == context["sources"][name], (
            "snapshot runtime policy differs from committed qualification source"
        )
    profile = _checked(runtime["runtime_profile"])
    artifacts = [item for item in profile["artifacts"] if item.get("name") == "pyarrow"]
    assert len(artifacts) == 1, "one exact custom Arrow 26 artifact required"
    artifact = artifacts[0]
    assert re.fullmatch(r"26\.0\.0\+biominer\.[0-9a-f]{12}", artifact.get("version", "")), (
        "Arrow 26 qualification requires the admitted custom Arrow 26 version"
    )
    assert re.fullmatch(r"[0-9a-f]{64}", artifact.get("sha256", ""))
    return runtime, {key: artifact[key] for key in ("version", "sha256")}


def _environment(output):
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(output / "home"),
        "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1", "PY_COLORS": "0",
        "POLARS_MAX_THREADS": "4", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
        "POLARS_OOC_SPILL_DIR": str(output / "scratch"),
        "TMPDIR": str(output / "supervision"), "UV_NO_CACHE": "1",
    }


def _command(runtime, arguments, output, name, cwd):
    argv = [
        runtime["approved_python"]["path"], "-I", "-B", runtime["launcher"]["path"],
        "run", runtime["snapshot"], *arguments,
    ]
    result = native.run_test(argv, cwd, _environment(output), 900)
    log = output / f"{name}.log"
    with log.open("xb") as stream:
        stream.write(result.stdout)
    receipt = output / f"{name}-command.json"
    _save(receipt, {
        "argv": argv, "cwd": str(cwd), "exit_code": result.returncode,
        "timeout_seconds": 900, "log": _reference(log),
    })
    assert result.returncode == 0, f"{name} failed; retained {log}"
    return result.stdout, _reference(receipt)


def _verify(runtime, output, name, cwd):
    raw, receipt = _command(
        runtime, ["verify", "--generation", runtime["generation"]], output, name, cwd,
    )
    result = json.loads(raw)
    assert result["status"] == "verified" and result["ready"] is True
    assert result["profile_sha256"] == runtime["runtime_profile"]["sha256"]
    assert result["generation"] == runtime["generation"]
    return receipt


def _exception(error):
    return {"type": f"{type(error).__module__}.{type(error).__name__}", "message": str(error)}


def _depth_rejection(operation):
    try:
        operation()
    except OSError as error:
        assert "Parquet schema too deeply nested" in str(error), str(error)
        return _exception(error)
    raise AssertionError("expected actual Parquet schema-depth rejection")


def _fixture(pa, pq, directory, list_levels):
    value, dtype = 42, pa.int64()
    for _ in range(list_levels):
        value = [value]
        # Match the compliant Parquet list child name for exact schema equality.
        dtype = pa.list_(pa.field("element", dtype))
    table = pa.table({"nested": pa.array([value], type=dtype)})
    path = directory / f"list-levels-{list_levels}.parquet"
    with path.open("xb") as stream:
        pq.write_table(table, stream)
    facts = {
        "file": _reference(path), "rows": table.num_rows, "list_levels": list_levels,
        "expected_parquet_schema_depth": 2 * list_levels + 2,
        "arrow_schema": str(table.schema),
        "arrow_schema_sha256": hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest(),
        "values": table.to_pydict(),
    }
    return path, table, facts


def _worker(context_reference):
    context = _checked(context_reference)
    _validate_sources(context)
    runtime, wheel = _runtime_inputs(context)
    import importlib.metadata as metadata
    import biominer
    import pyarrow as pa
    import pyarrow.parquet as pq
    from biominer.storage.local import LocalStorageBackend
    from biominer.storage.s3 import S3StorageBackend

    generation = Path(runtime["generation"])
    assert sys.executable == str(generation / "env/bin/python")
    assert Path(sys.prefix).resolve() == generation / "env"
    assert Path(pa.__file__).resolve().is_relative_to(generation / "env")
    assert pa.__version__ == metadata.version("pyarrow") == wheel["version"]
    package = Path(biominer.__file__).resolve().parent
    assert package.is_relative_to(generation / "env")
    installed_sources = {}
    for name, expected in context["sources"].items():
        if name.startswith("src/biominer/"):
            installed = package / Path(name).relative_to("src/biominer")
            reference = _reference(installed)
            assert reference["sha256"] == expected, f"installed source differs: {name}"
            installed_sources[name] = reference
    output = Path(context["output"])
    fixtures = output / "fixtures"
    fixtures.mkdir(mode=0o700)
    observation = {
        "schema": "biominer-arrow-api-observation/v1", "status": "started",
        "context": context_reference, "probe": _reference(__file__),
        "runtime_profile": runtime["runtime_profile"], "arrow_wheel": wheel,
        "python": sys.executable, "prefix": sys.prefix, "installed_sources": installed_sources,
        "versions": {name: metadata.version(name) for name in ("pyarrow", "polars", "polars-runtime-32")},
        "fixtures": {}, "direct_readers": {}, "public_readers": {},
    }
    try:
        shallow, table, facts = _fixture(pa, pq, fixtures, 10)
        observation["fixtures"]["depth22"] = facts

        def parquet_file_read(**options):
            with pq.ParquetFile(shallow, **options) as reader:
                return reader.read()

        for name, reader in (
            ("read_table", lambda **options: pq.read_table(shallow, **options)),
            ("ParquetFile.read", parquet_file_read),
        ):
            observed = {"limit21": _depth_rejection(lambda: reader(schema_depth_limit=21))}
            observation["direct_readers"][name] = observed
            for label, options in (("limit22", {"schema_depth_limit": 22}), ("default100", {})):
                actual = reader(**options)
                assert actual.schema.equals(table.schema, check_metadata=True)
                assert actual.equals(table, check_metadata=True)
                assert actual.to_pydict() == table.to_pydict()
                observed[label] = {"rows": actual.num_rows, "schema_and_values_equal": True}
            assert _reference(shallow) == facts["file"]

        deep, _, deep_facts = _fixture(pa, pq, fixtures, 50)
        observation["fixtures"]["depth102"] = deep_facts
        payload = deep.read_bytes()

        class InputOnlyS3Filesystem:
            def __init__(self):
                self.streams = []

            def open_input_file(self, path):
                assert path == "offline/depth102.parquet"
                stream = io.BytesIO(payload)
                self.streams.append(stream)
                return stream

        filesystem = InputOnlyS3Filesystem()
        s3 = S3StorageBackend(bucket="offline")

        def offline_path(uri):
            assert uri == "s3://offline/depth102.parquet"
            return filesystem, "offline/depth102.parquet"

        s3._filesystem_and_path = offline_path
        for name, backend, uri in (
            ("local", LocalStorageBackend(), deep),
            ("s3", s3, "s3://offline/depth102.parquet"),
        ):
            yielded = []

            def consume():
                for batch in backend.iter_parquet_batches(uri, batch_size=2):
                    yielded.append(batch)

            error = _depth_rejection(consume)
            assert yielded == []
            assert _reference(deep) == deep_facts["file"]
            observation["public_readers"][name] = {
                "default_limit": 100, "exception": error, "yielded_batches": 0,
                "input_unchanged": True,
            }
        assert len(filesystem.streams) == 1 and all(stream.closed for stream in filesystem.streams)
        assert set(fixtures.iterdir()) == {shallow, deep}
        observation.update(status="passed", cleanup={
            "fake_s3_input_streams_closed": True, "fixture_files_preserved": 2,
            "unexpected_fixture_outputs": [], "network_calls": "none: input-only fake S3 transport",
        })
    except Exception as error:
        observation.update(status="failed", error=_exception(error))
        raise
    finally:
        _save(output / "observation.json", observation)
    _checked(context_reference)


def test_arrow26_schema_depth_limits():
    context_path = os.environ.get("BIOMINER_ARROW_API_QUALIFICATION_CONTEXT")
    assert context_path, "explicit Arrow API qualification context required"
    context_reference = _reference(context_path)
    context = _checked(context_reference)
    cwd = _validate_sources(context)
    runtime, wheel = _runtime_inputs(context)
    output = native.gate.outside_git(context["output"], context["repo"])
    assert str(output) == context["output"], "canonical output path required"
    output.mkdir(mode=0o700)
    for name in ("scratch", "supervision"):
        (output / name).mkdir(mode=0o700)
    commands = [_verify(runtime, output, "before-verify", cwd)]
    _, receipt = _command(runtime, [
        "exec", "--generation", runtime["generation"], "--", str(cwd / CASE_MODULE),
        "--worker", json.dumps(context_reference, sort_keys=True),
    ], output, "depth-worker", cwd)
    commands.append(receipt)
    observed_reference = _reference(output / "observation.json")
    observed = _checked(observed_reference)
    assert observed["status"] == "passed"
    assert observed["context"] == context_reference
    assert observed["runtime_profile"] == runtime["runtime_profile"]
    assert observed["arrow_wheel"] == wheel
    assert observed["probe"] == _reference(cwd / CASE_MODULE)
    commands.append(_verify(runtime, output, "after-verify", cwd))
    reports = sorted((output / "supervision").glob("*.report/finished.json"))
    assert len(reports) == 1, "one supervised Arrow worker report required"
    finished = json.loads(reports[0].read_bytes())
    assert finished["status"] == "passed" and finished["exit_code"] == 0
    assert finished["cleanup"] == {"status": "removed"}
    assert list((output / "scratch").iterdir()) == []
    _checked(context_reference)
    _validate_sources(context)
    assert _runtime_inputs(context) == (runtime, wheel)
    _checked(observed_reference)
    for fixture in observed["fixtures"].values():
        assert _reference(fixture["file"]["path"]) == fixture["file"]
    report = output / "qualification.json"
    _save(report, {
        "schema": "biominer-arrow-api-qualification/v1", "status": "passed",
        "tested_commit": context["tested_commit"], "context": context_reference,
        "sources": context["sources"], "runtime": runtime, "arrow_wheel": wheel,
        "observation": observed_reference, "commands": commands,
        "supervisor_report": _reference(reports[0]),
        "limits": "Synthetic local and fake-S3 reader behavior; no hosted service or arbitrary-corruption coverage",
    })
    print(json.dumps({"qualification": _reference(report)}, sort_keys=True))


if __name__ == "__main__":
    assert len(sys.argv) == 3 and sys.argv[1] == "--worker", "private qualification worker only"
    _worker(json.loads(sys.argv[2]))
