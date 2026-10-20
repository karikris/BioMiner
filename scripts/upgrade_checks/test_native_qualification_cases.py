"""Opt-in real frozen-runtime probes; excluded by the default tests/ testpaths.

The producer supplies a private, hash-bound runtime context. Missing context
fails rather than skips. Every native operation uses the preserved launcher;
no system or pytest-environment package substitutes for the native generation.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


def _checked(reference):
    data = Path(reference["path"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == reference["sha256"], "runtime input changed"
    return data


def _context():
    path = os.environ.get("BIOMINER_NATIVE_QUALIFICATION_CONTEXT")
    assert path, "explicit native qualification context required"
    context = json.loads(Path(path).read_bytes())
    for key in ("snapshot_manifest", "runtime_profile", "launcher", "approved_python", "generation_receipt"):
        _checked(context[key])
    manifest = json.loads(_checked(context["snapshot_manifest"]))
    admission_name = "scripts/native_runtime.py"
    admission_hash = hashlib.sha256(
        Path(__file__).resolve().parents[1].joinpath("native_runtime.py").read_bytes()
    ).hexdigest()
    assert manifest["files"].get(admission_name) == admission_hash, (
        "snapshot admission policy differs from tested source"
    )
    _checked({"path": str(Path(context["snapshot"]) / admission_name), "sha256": admission_hash})
    return context


def _run(context, arguments, expected_exit=0):
    command = [
        context["approved_python"]["path"], "-I", "-B", context["launcher"]["path"],
        "run", context["snapshot"], *arguments,
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=300)
    assert result.returncode == expected_exit, (
        f"frozen command exited {result.returncode}: {result.stderr}\n{result.stdout}"
    )
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def _native(context, name):
    script = str(Path(context["snapshot"]) / "scripts/check_native_runtime.py")
    result = _run(context, [
        "exec", "--generation", context["generation"], "--", script,
        "--check", name, "--profile", context["runtime_profile"]["path"],
        "--generation", context["generation"],
    ])
    profile = json.loads(_checked(context["runtime_profile"]))
    assert result.get("status") == "passed"
    assert result.get("check") == name
    assert result.get("profile_sha256") == context["runtime_profile"]["sha256"]
    assert result.get("policy_id") == profile["extension_policy"]["policy_id"]
    return result


def _record(context, case, details):
    evidence = {
        "case": case, "runtime_profile": context["runtime_profile"],
        "generation_receipt": context["generation_receipt"], "details": details,
    }
    path = Path(context["evidence_directory"]) / f"{case}.json"
    with path.open("x") as target:
        json.dump(evidence, target, sort_keys=True)
        target.write("\n")
    print(json.dumps(evidence, sort_keys=True))


def test_artifacts():
    context = _context()
    artifacts = _run(context, ["artifacts"])
    assert artifacts.get("status") == "candidate" and artifacts.get("ready") is False
    assert artifacts.get("profile_sha256") == context["runtime_profile"]["sha256"]
    assert set(artifacts["artifacts"]) == {"biominer", "psycopg-c", "pyarrow", "duckdb", "lxml"}
    verified = _run(context, ["verify", "--generation", context["generation"]])
    assert verified.get("status") == "verified" and verified.get("ready") is True
    assert verified.get("profile_sha256") == context["runtime_profile"]["sha256"]
    assert verified.get("generation") == context["generation"]
    _record(context, "artifacts", {"artifacts": artifacts, "fresh_verification": verified})


def test_loaded_versions():
    context = _context()
    result = _native(context, "effective_versions")
    profile = json.loads(_checked(context["runtime_profile"]))
    versions = result["effective_versions"]
    assert versions["packages"] == {item["name"]: item["version"] for item in profile["artifacts"]}
    assert versions["python_openssl"].split()[1] == profile["python"]["openssl_version"]
    assert versions["lxml_native"]["lxml_runtime"] == versions["packages"]["lxml"]
    _record(context, "loaded_versions", result)


def test_tls():
    context = _context()
    result = _native(context, "tls")
    expected = {
        f"{library}_{case}" for library in ("libpq", "arrow")
        for case in ("valid_certificate", "wrong_hostname_rejected", "untrusted_issuer_rejected")
    }
    assert set(result["tls"]) == expected
    assert all(result["tls"][name] is True for name in expected)
    _record(context, "tls", result)


def test_import_order():
    context = _context()
    result = _native(context, "native_inventory")
    forward = ["pyarrow.flight", "pyarrow.parquet.encryption", "psycopg.pq", "duckdb", "lxml.etree"]
    assert result["import_orders"] == [forward, list(reversed(forward))]
    _record(context, "import_order", result)


IDENTITY = """
import importlib.metadata as m, json, sys
print(json.dumps({'executable': sys.executable, 'prefix': sys.prefix,
                  'versions': {n: m.version(n) for n in
                               ('biominer', 'psycopg-c', 'pyarrow', 'duckdb', 'lxml')}}))
"""

PARQUET_RESUME = """
import hashlib, importlib.metadata as m, json, pathlib, subprocess, sys
import pyarrow as pa
import pyarrow.parquet as pq
work, profile_hash, stage = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
identity = json.loads(subprocess.check_output(
    [sys.executable, '-I', '-B', '-c', sys.argv[4]], text=True))
parent = {'executable': sys.executable, 'prefix': sys.prefix,
          'versions': {n: m.version(n) for n in identity['versions']}}
assert identity == parent
source_hash = hashlib.sha256(b'bounded-native-resume-fixture-v1').hexdigest()
original = work / 'part-0.parquet'
checkpoint = work / 'checkpoint-1.json'
digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
if stage == 'interrupt':
    work.mkdir()
    pq.write_table(pa.table({'source_id': ['native-fixture'] * 2, 'ordinal': [0, 1]}), original)
    state = {'source_sha256': source_hash, 'profile_sha256': profile_hash,
             'part_sha256': digest(original), 'next_ordinal': 2}
    with checkpoint.open('x') as target:
        json.dump(state, target, sort_keys=True)
    print(json.dumps({'stage': 'checkpointed', 'identity': identity, 'state': state}))
    raise SystemExit(75)
assert stage == 'resume'
state = json.loads(checkpoint.read_bytes())
assert state == {'source_sha256': source_hash, 'profile_sha256': profile_hash,
                 'part_sha256': digest(original), 'next_ordinal': 2}
first = pq.read_table(original)
assert first.to_pydict() == {'source_id': ['native-fixture'] * 2, 'ordinal': [0, 1]}
combined = pa.concat_tables([first, pa.table({'source_id': ['native-fixture'] * 2, 'ordinal': [2, 3]})])
final = work / 'resumed.parquet'
assert not final.exists()
pq.write_table(combined, final)
assert pq.read_table(final).to_pydict() == {'source_id': ['native-fixture'] * 4, 'ordinal': [0, 1, 2, 3]}
assert digest(original) == state['part_sha256']
result = {'stage': 'resumed', 'identity': identity, 'rows': 4,
          'source_sha256': source_hash, 'profile_sha256': profile_hash,
          'original_sha256': digest(original), 'resumed_sha256': digest(final),
          'checkpoint_sha256': digest(checkpoint)}
with (work / 'checkpoint-2.json').open('x') as target:
    json.dump(result, target, sort_keys=True)
print(json.dumps(result))
"""

ARROW_BOOLEAN_ALLOCATION = """
import hashlib, json
import pyarrow as pa
import pyarrow.compute as pc

observations = []

def expected_fill(values, forward):
    result = list(values)
    indices = range(len(result)) if forward else range(len(result) - 1, -1, -1)
    last = None
    for index in indices:
        if result[index] is None:
            result[index] = last
        else:
            last = result[index]
    return result

def observe(name, result, expected, chunk_lengths):
    assert isinstance(result, pa.ChunkedArray)
    result.validate(full=True)
    assert result.to_pylist() == expected, name
    assert [len(chunk) for chunk in result.chunks] == chunk_lengths, name
    buffers = []
    for chunk in result.chunks:
        buffer = chunk.buffers()[1]
        minimum = ((chunk.offset + len(chunk)) * result.type.bit_width + 7) // 8
        if len(chunk):
            assert buffer is not None and buffer.size >= minimum, (name, minimum)
        buffers.append(None if buffer is None else buffer.size)
    observations.append({
        'case': name, 'type': str(result.type), 'length': len(result),
        'chunk_lengths': chunk_lengths, 'null_count': result.null_count,
        'data_buffer_bytes': buffers,
        'values_sha256': hashlib.sha256(json.dumps(expected).encode()).hexdigest(),
    })

def check_fills(name, values):
    original = values.to_pylist()
    lengths = [len(chunk) for chunk in values.chunks]
    observe(name + '.forward', pc.fill_null_forward(values),
            expected_fill(original, True), lengths)
    observe(name + '.backward', pc.fill_null_backward(values),
            expected_fill(original, False), lengths)
    assert values.to_pylist() == original

def check_replace(name, values, mask, replacement):
    original = values.to_pylist()
    flags = mask.to_pylist() if isinstance(mask, pa.Array) else [mask.as_py()] * len(values)
    expected = [
        None if flag is None else replacement.as_py() if flag else value
        for value, flag in zip(original, flags, strict=True)
    ]
    observe(name + '.replace', pc.replace_with_mask(values, mask, replacement),
            expected, [len(chunk) for chunk in values.chunks if len(chunk)])
    assert values.to_pylist() == original

# Bounded regressions for all three upstream GH-45086 allocation sites.
check_fills('large_boolean', pa.chunked_array(
    [[None] * 4096, [True], [None] * 4096], type=pa.bool_()))
check_replace('large_boolean', pa.chunked_array(
    [[False] * 4096] * 3, type=pa.bool_()), pa.scalar(True), pa.scalar(True))

sliced = pa.chunked_array([
    pa.array([True, None, False, None, True, False, None, True, None, False, True]).slice(1, 9),
    pa.array([False, None, True, None]).slice(1, 3),
], type=pa.bool_())
check_fills('sliced_boolean', sliced)
check_replace('sliced_boolean', sliced, pa.array([True, False, None] * 4), pa.scalar(True))

empty = pa.chunked_array([[], []], type=pa.bool_())
check_fills('empty_boolean', empty)
check_replace('empty_boolean', empty, pa.scalar(True), pa.scalar(True))
nulls = pa.chunked_array([[None] * 9, []], type=pa.bool_())
check_fills('null_boolean', nulls)
check_replace('null_boolean', nulls, pa.scalar(True), pa.scalar(False))

numeric = pa.chunked_array([[None, 3, None], [], [None, -2, None]], type=pa.int16())
check_fills('numeric_control', numeric)
check_replace('numeric_control', numeric, pa.scalar(True), pa.scalar(9, type=pa.int16()))
assert len(observations) == 15
print(json.dumps({'pyarrow': pa.__version__, 'cases': observations}, sort_keys=True))
"""


def test_features():
    context = _context()
    capability = _native(context, "geometry")
    policy = _native(context, "extension_policy")
    assert capability["capabilities"]["status"] == "capabilities_passed"
    assert capability["capabilities"]["live_postgres"] == "unrun"
    assert capability["capabilities"]["live_s3"] == "unrun"
    work = str(Path(context["evidence_directory"]) / "parquet-resume")
    arguments = ["exec", "--generation", context["generation"], "--", "-c",
                 PARQUET_RESUME, work, context["runtime_profile"]["sha256"]]
    interrupted = _run(context, [*arguments, "interrupt", IDENTITY], expected_exit=75)
    assert interrupted["stage"] == "checkpointed"
    resumed = _run(context, [*arguments, "resume", IDENTITY])
    assert resumed["stage"] == "resumed" and resumed["rows"] == 4
    assert resumed["original_sha256"] == interrupted["state"]["part_sha256"]
    assert resumed["identity"] == interrupted["identity"]
    identity = resumed["identity"]
    generation = Path(context["generation"])
    assert Path(identity["executable"]).absolute() == generation / "env/bin/python"
    assert Path(identity["prefix"]).absolute() == generation / "env"
    profile = json.loads(_checked(context["runtime_profile"]))
    assert identity["versions"] == {item["name"]: item["version"] for item in profile["artifacts"]}
    allocation = _run(context, [
        "exec", "--generation", context["generation"], "--", "-c", ARROW_BOOLEAN_ALLOCATION,
    ])
    assert allocation["pyarrow"] == identity["versions"]["pyarrow"]
    assert len(allocation["cases"]) == 15
    _record(context, "features", {
        "capability": capability, "compiled_policy": policy,
        "controlled_stop_exit": 75, "checkpoint": interrupted, "resume": resumed,
        "arrow_boolean_allocation": allocation,
    })
