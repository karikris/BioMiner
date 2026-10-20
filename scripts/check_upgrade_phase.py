"""Fail-closed, local evidence gates for the approved six-phase upgrade.

The manifest is reviewed operator input, not an adversarial execution sandbox.
Commands run without a shell. Reviews attest human/agent judgment; hashes prove
which review was used, not the reviewer's identity. No command writes Git state.
See docs/upgrade_phase_receipts.md for the versioned receipt contract.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PHASE_CHECKS = {
    1: ("inventory", "baseline", "security", "phase_gate_tests"),
    2: ("runtime_snapshot", "migration", "native_verify", "security"),
    3: ("polars_semantics", "polars_resources", "full_suite", "security"),
    4: ("arrow_build", "arrow_api", "native_verify", "security"),
    5: (
        "compatibility_matrix", "live_postgres", "live_s3", "live_cleanup",
        "full_suite", "benchmark", "security",
    ),
    6: ("canary", "migration", "recovery", "retirement", "native_verify", "security"),
}
PYTEST_CHECKS = {
    "baseline", "phase_gate_tests", "polars_semantics", "polars_resources",
    "full_suite", "arrow_api", "canary", "recovery",
}
CASE_SETS = {
    "native_verify": {"artifacts", "loaded_versions", "tls", "import_order", "features"},
    "live_postgres": {"authenticated_tls", "transactions", "rollback", "crud", "leases", "idempotency"},
    "live_s3": {"parquet_roundtrip", "ranges", "listing", "retries", "manifest_last", "scoped_delete"},
}
SHA = re.compile(r"[a-f0-9]{64}\Z")
COMMIT = re.compile(r"[a-f0-9]{40}\Z")
SECRET = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@"
    r"|(?:password|passwd|api[_-]?key|access[_-]?token|client[_-]?secret)"
    r"""["']?\s*[:=]\s*["']?[^\s"' ,}]{8,}"""
    r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    r"|AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,})"
)


class GateError(ValueError):
    """Evidence is missing, stale, unsafe, or does not satisfy a phase gate."""


def require(condition, message):
    if not condition:
        raise GateError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def scan_credentials(data):
    text = data.decode("utf-8", errors="replace")
    sensitive_values = (
        value for name, value in os.environ.items()
        if re.search(r"(?:PASSWORD|PASSWD|TOKEN|SECRET|(?:^|_)KEY(?:_|$)|DSN|CONNECTION_STRING)", name)
        and len(value) >= 8
    )
    require(
        not SECRET.search(text) and not any(value in text for value in sensitive_values),
        "credential material detected; evidence was not published",
    )


def read_bytes(path):
    path = Path(path)
    require(path.is_file(), "evidence file is missing")
    data = path.read_bytes()
    scan_credentials(data)
    return data


def load_json(path):
    try:
        value = json.loads(read_bytes(path))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError("invalid JSON evidence") from exc
    require(isinstance(value, dict), "JSON evidence must be an object")
    return value


def file_ref(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": digest(read_bytes(path))}


def verify_ref(ref):
    require(isinstance(ref, dict), "invalid evidence reference")
    require(set(ref) == {"path", "sha256"}, "invalid evidence reference fields")
    require(isinstance(ref["path"], str) and Path(ref["path"]).is_absolute(), "evidence path must be absolute")
    require(isinstance(ref["sha256"], str) and SHA.fullmatch(ref["sha256"]), "invalid evidence hash")
    data = read_bytes(ref["path"])
    require(digest(data) == ref["sha256"], "evidence hash mismatch")
    return data


def ref_json(ref):
    try:
        value = json.loads(verify_ref(ref))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise GateError("invalid referenced JSON evidence") from exc
    require(isinstance(value, dict), "referenced JSON must be an object")
    return value


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, timeout=60,
    )
    require(result.returncode == 0, "Git evidence command failed")
    return result.stdout.decode().strip()


def ancestor(repo, older, newer):
    git(repo, "merge-base", "--is-ancestor", older, newer)


def source_digest(sources):
    return digest(canonical(sources))


def documentation(path):
    return (
        (path.startswith("docs/") and Path(path).suffix in {".md", ".json"})
        or path == "provenance/githits.jsonl"
    )


def verify_sources(manifest, *, worktree):
    repo = Path(manifest["repo"]).resolve()
    require(git(repo, "branch", "--show-current") == "main", "phase must use main")
    sources = manifest["sources"]
    require(isinstance(sources, dict) and sources, "tested source/config hashes are required")
    for key in ("base_commit", "tested_commit"):
        require(isinstance(manifest[key], str) and COMMIT.fullmatch(manifest[key]), "full Git commit required")
        require(git(repo, "rev-parse", f"{manifest[key]}^{{commit}}") == manifest[key], "commit missing")
    tasks = manifest["tasks"]
    require(isinstance(tasks, dict), "task commits required")
    expected = [f"{manifest['phase']}.{i}" for i in range(1, 4)]
    require(set(tasks) == set(expected), "fixed phase task IDs required")
    previous = manifest["base_commit"]
    for task in expected:
        sha = tasks[task]
        require(isinstance(sha, str) and COMMIT.fullmatch(sha), "full task commit required")
        require(sha != previous, "each task needs a distinct commit")
        ancestor(repo, previous, sha)
        previous = sha
    require(previous == manifest["tested_commit"], "tested commit must be final implementation task commit")
    git(repo, "diff", "--check", manifest["base_commit"], previous)
    changed = git(repo, "diff", "--name-only", "--diff-filter=ACMRT", manifest["base_commit"], previous).splitlines()
    tracked = set(git(repo, "ls-tree", "-r", "--name-only", previous).splitlines())
    mandatory = {p for p in changed if not documentation(p)}
    mandatory |= tracked & {"pyproject.toml", "uv.lock", ".python-version", "scripts/check_upgrade_phase.py"}
    require(mandatory <= sources.keys(), "task source/config omitted from tested hashes")
    for path in changed:
        if documentation(path):
            scan_credentials(git(repo, "show", f"{previous}:{path}").encode())
    for path, expected_hash in sources.items():
        require(
            isinstance(path, str) and path in tracked
            and not Path(path).is_absolute() and ".." not in Path(path).parts,
            "invalid tested source path",
        )
        require(isinstance(expected_hash, str) and SHA.fullmatch(expected_hash), "invalid source hash")
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{previous}:{path}"],
            capture_output=True, timeout=60,
        )
        require(result.returncode == 0 and digest(result.stdout) == expected_hash, "committed source hash mismatch")
        scan_credentials(result.stdout)
        if worktree:
            require(digest(read_bytes(repo / path)) == expected_hash, "working source hash mismatch")
    return repo


def pytest_outcomes(log):
    # Parse the final complete pytest summary, not a substring saying "passed".
    lines = log.decode("utf-8").splitlines()
    pattern = r"(?P<counts>(?:\d+ (?:passed|failed|errors?|skipped|xfailed|xpassed|deselected|warnings?)(?:, )?)+) in \d+(?:\.\d+)?s(?: \([^)]*\))?"
    matches = [re.fullmatch(pattern, line.strip("= ")) for line in lines]
    summaries = [match for match in matches if match]
    require(summaries, "pytest summary missing or unsupported")
    counts = dict(
        (noun.rstrip("s") if noun in {"errors", "warnings"} else noun, int(number))
        for number, noun in re.findall(r"(\d+) ([a-z]+)", summaries[-1]["counts"])
    )
    require(counts.get("passed", 0) > 0, "pytest has no passed tests")
    require(
        all(counts.get(kind, 0) == 0 for kind in ("failed", "error", "skipped", "xfailed", "xpassed", "deselected")),
        "pytest failed, skipped, or incomplete checks",
    )
    return counts


def verify_nested(reference, case, context, check, profile):
    require(context is not None and check is not None, "nested receipt validation context required")
    receipt = ref_json(reference)
    require(receipt.get("schema_version") == 1 and receipt.get("check") == case, "nested case receipt mismatch")
    require(type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0, "nested case exit failure")
    require(receipt.get("tested_commit") == context["tested_commit"] and receipt.get("source_digest") == source_digest(context["sources"]), "nested tested source mismatch")
    require(receipt.get("cwd") == str(Path(check.get("cwd", context["repo"])).resolve()), "nested execution directory mismatch")
    require(receipt.get("runtime_profile") == profile, "nested runtime profile mismatch")
    verify_ref(profile)
    verify_command("phase_gate_tests", {"checker": "pytest", "argv": receipt.get("argv")})
    pytest_outcomes(verify_ref(receipt.get("log")))


def verify_cases(report, kind, context, check):
    cases = report.get("cases")
    require(isinstance(cases, dict) and set(cases) == CASE_SETS[kind], f"{kind} mandatory cases missing")
    require(context is not None and check is not None, "nested receipt validation context required")
    profile = check.get("runtime_profile")
    verify_ref(profile)
    require(report.get("runtime_profile") == profile, "report runtime profile mismatch")
    for name, item in cases.items():
        require(isinstance(item, dict), "nested case receipt required")
        verify_nested(item.get("receipt"), f"{kind}.{name}", context, check, profile)


def verify_report(data, kind, phase, *, context=None, check=None, previous_security=None):
    try:
        report = json.loads(data)
    except (ValueError, UnicodeDecodeError) as exc:
        raise GateError(f"{kind} requires structured report evidence") from exc
    require(
        isinstance(report, dict) and report.get("schema_version") == 1 and report.get("kind") == kind,
        f"{kind} report schema mismatch",
    )
    evidence = report.get("evidence")
    require(isinstance(evidence, list) and evidence, f"{kind} evidence required")
    for item in evidence:
        verify_ref(item)
    if kind == "inventory":
        entries = report.get("entries")
        require(isinstance(entries, list) and entries, "runtime inventory is empty")
        for item in entries:
            require(item.get("id") and item.get("owner"), "inventory ownership missing")
            verify_ref(item.get("runtime"))
    elif kind == "security":
        require(isinstance(report.get("scopes"), list) and report["scopes"] and isinstance(report.get("findings"), list), "security scopes/findings required")
        require(isinstance(report.get("coverage_gaps"), list), "security coverage gaps must be explicit")
        for category in ("findings", "coverage_gaps"):
            current = {}
            for finding in report[category]:
                require(isinstance(finding, dict) and isinstance(finding.get("id"), str) and finding["id"], "stable security finding/gap ID required")
                require(finding["id"] not in current and finding.get("scope") in report["scopes"], "duplicate or unscoped security finding")
                current[finding["id"]] = finding
                if phase > 1:
                    allowed = {"resolved", "not_applicable"}
                    if phase == 2 and category == "findings":
                        allowed.add("contained")
                    require(finding.get("status") in allowed, "applicable security finding or coverage gap remains open")
                    verify_ref(finding.get("evidence"))
            if previous_security is not None:
                for prior in previous_security[category]:
                    require(prior["id"] in current and current[prior["id"]]["scope"] == prior["scope"], "predecessor security finding/gap omitted or rescoped")
    elif kind == "runtime_snapshot":
        require(report.get("self_contained") is True, "fallback snapshot must be self-contained")
        for key in ("launcher", "configuration", "lock", "verifier", "runtime"):
            verify_ref(report.get(key))
    elif kind == "migration":
        counts = [report.get(k) for k in ("eligible", "migrated", "blocked")]
        require(all(type(n) is int and n >= 0 for n in counts), "migration counts missing")
        require(counts[0] == counts[1] and counts[2] == 0, "eligible migration incomplete")
        remaining = report.get("remaining_exposures")
        require(isinstance(remaining, list), "remaining migration exposure must be explicit")
        if phase == 6:
            require(not remaining, "final migration has remaining exposure")
        for item in remaining:
            require(item.get("status") == "contained" and item.get("owner"), "remaining exposure not contained")
            verify_ref(item.get("evidence"))
    elif kind in CASE_SETS:
        verify_cases(report, kind, context, check)
        if kind.startswith("live_"):
            require(report.get("execution") == "live", "live evidence required")
            require(isinstance(report.get("namespace"), str) and report["namespace"].startswith("biominer_upgrade_"), "task-owned live namespace required")
    elif kind == "arrow_build":
        require(type(report.get("compiler_jobs")) is int and 1 <= report["compiler_jobs"] <= 4, "Arrow build exceeds compiler budget")
        require(report.get("version", "").startswith("26.0.0+biominer."), "custom Arrow 26 build required")
        for key in ("source", "recipe", "wheel", "native_receipt"):
            verify_ref(report.get(key))
        require(set(report.get("features", [])) >= {"s3", "flight", "encryption"}, "Arrow capabilities missing")
    elif kind == "compatibility_matrix":
        combos = report.get("combinations")
        require(isinstance(combos, dict) and set(combos) == {"p1a25", "p1a26", "p2a25", "p2a26"}, "four compatibility combinations required")
        require(context is not None and check is not None, "nested receipt validation context required")
        profiles = check.get("runtime_profiles")
        require(isinstance(profiles, dict) and set(profiles) == set(combos), "four pinned runtime profiles required")
        for profile in profiles.values():
            verify_ref(profile)
        require(len({profile["sha256"] for profile in profiles.values()}) == 4, "four distinct runtime profiles required")
        fixture = None
        for name, item in combos.items():
            require(item.get("runtime_profile") == profiles[name], "matrix runtime profile mismatch")
            verify_nested(item.get("receipt"), f"{kind}.{name}", context, check, profiles[name])
            verify_ref(item.get("fixture"))
            fixture = fixture or item["fixture"]["sha256"]
            require(item["fixture"]["sha256"] == fixture, "compatibility fixtures differ")
            require(item.get("logical_differences") == [], "unexplained logical output differences")
            verify_ref(item.get("comparison"))
    elif kind == "live_cleanup":
        require(set(report.get("namespaces", {})) == {"postgres", "s3"}, "both live cleanup namespaces required")
        for item in report["namespaces"].values():
            require(item.get("namespace", "").startswith("biominer_upgrade_"), "cleanup namespace not task owned")
            require(type(item.get("remaining")) is int and item["remaining"] == 0, "live cleanup incomplete")
            verify_ref(item.get("inspection"))
    elif kind == "benchmark":
        samples = report.get("samples")
        require(isinstance(samples, list) and len(samples) == 3, "three paired benchmark samples required")
        for metric in ("wall_seconds", "peak_rss_bytes"):
            base, target = [], []
            for sample in samples:
                for side, values in (("baseline", base), ("target", target)):
                    value = sample.get(side, {}).get(metric)
                    require(type(value) in {float, int} and 0 < value < float("inf"), "benchmark metric invalid")
                    values.append(value)
                verify_ref(sample.get("evidence"))
            require(statistics.median(target) <= 1.15 * statistics.median(base), "benchmark regression exceeds 15 percent")
        require(report.get("budget_violations") == [] and report.get("resume_losses") == [] and report.get("provenance_losses") == [], "benchmark budget or correctness violation")
    elif kind == "retirement":
        require(report.get("affected_entrypoints") == [] and report.get("unsafe_fallbacks") == [], "affected entrypoints remain")
        verify_ref(report.get("entrypoint_inventory"))
    else:
        raise GateError("unsupported report kind")
    return report


def verify_command(name, item):
    expected = "pytest" if name in PYTEST_CHECKS else "report"
    require(item.get("checker") == expected, f"{name} requires {expected} checker")
    argv = item.get("argv")
    require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv), "command argv required")
    scan_credentials(canonical(argv))
    if expected == "pytest":
        require(len(argv) >= 3 and argv[1:3] == ["-m", "pytest"], "pytest command required")
        require(Path(argv[0]).name.startswith("python"), "pytest Python executable required")
        if name in {"full_suite", "baseline"}:
            require(all(x in {"-q", "-ra", "--strict-config", "--strict-markers", "--color=no"} for x in argv[3:]), "full pytest suite cannot be filtered")


def execution_directory(manifest, name, item, *, worktree):
    path = Path(item.get("cwd", manifest["repo"]))
    require(path.is_absolute(), "execution directory must be absolute")
    path = path.resolve()
    if worktree:
        for source, expected in manifest["sources"].items():
            require(digest(read_bytes(path / source)) == expected, "execution snapshot source hash mismatch")
        if name in {"baseline", "full_suite"}:
            changes = git(path, "diff", "--name-only", manifest["tested_commit"]).splitlines()
            unknown = git(path, "ls-files", "--others", "--exclude-standard").splitlines()
            unknown += git(path, "ls-files", "--others", "--ignored", "--exclude-standard").splitlines()
            require(
                all(documentation(p) for p in changes)
                and not any(Path(p).suffix in {".py", ".toml", ".lock", ".ini", ".cfg", ".yaml", ".yml", ".json", ".sh"} for p in unknown),
                "full-suite snapshot contains unrelated or untested code",
            )
    return path


def verify_reviews(manifest):
    reviews = manifest["reviews"]
    require(isinstance(reviews, dict) and set(reviews) == {"spec_review", "quality_review"}, "both independent reviews required")
    require(isinstance(manifest["implementer"], str) and manifest["implementer"], "implementer required")
    for kind, reference in reviews.items():
        item = ref_json(reference)
        require(item.get("schema_version") == 1 and item.get("kind") == kind, "review schema mismatch")
        require(item.get("reviewer") and item["reviewer"] != manifest["implementer"], "independent reviewer required")
        require(item.get("implementer") == manifest["implementer"], "review implementer mismatch")
        require(item.get("task_commits") == manifest["tasks"], "review task commits are stale")
        require(item.get("source_digest") == source_digest(manifest["sources"]), "review source hash is stale")
        require(item.get("verdict") == "approved" and item.get("findings") == [], "review findings block advancement")
        require(bool(verify_ref(item.get("report")).strip()), "actual review report required")


def security_predecessor(manifest):
    if manifest["previous"] is None:
        return None
    handshake = ref_json(manifest["previous"])
    previous = ref_json(handshake["candidate"])
    command = ref_json(previous["checks"]["security"]["receipt"])
    return json.loads(verify_ref(command["log"]))


def validate_phase(path, *, worktree=True, require_results=True):
    manifest = load_json(path)
    fields = {
        "schema_version", "phase", "repo", "base_commit", "tested_commit", "tasks",
        "sources", "artifacts", "implementer", "reviews", "checks", "blockers", "previous",
    }
    require(fields <= manifest.keys(), "phase manifest fields missing")
    phase = manifest["phase"]
    require(manifest["schema_version"] == 1 and type(phase) is int and phase in PHASE_CHECKS, "unsupported phase/schema")
    require(manifest["blockers"] == [], "phase has blockers")
    checks = manifest["checks"]
    require(isinstance(checks, dict) and set(checks) == set(PHASE_CHECKS[phase]), "fixed mandatory phase checks required")
    repo = verify_sources(manifest, worktree=worktree)
    require(isinstance(manifest["artifacts"], list) and manifest["artifacts"], "phase artifacts required")
    for item in manifest["artifacts"]:
        verify_ref(item)
    if phase == 1:
        require(manifest["previous"] is None, "phase 1 cannot have a predecessor")
    else:
        previous = verify_handshake(manifest["previous"], expected_phase=phase - 1)
        require(previous["final_commit"] == manifest["base_commit"], "stale predecessor base commit")
        require(Path(previous["repo"]).resolve() == repo, "predecessor repository mismatch")
    verify_reviews(manifest)
    results = {}
    for name, item in checks.items():
        require(isinstance(item, dict), "check must be an object")
        verify_command(name, item)
        cwd = execution_directory(manifest, name, item, worktree=worktree)
        if not require_results:
            continue
        receipt = ref_json(item.get("receipt"))
        require(receipt.get("schema_version") == 1 and receipt.get("check") == name, "check receipt schema mismatch")
        require(receipt.get("argv") == item["argv"], "check command mismatch")
        require(receipt.get("cwd") == str(cwd), "check working directory mismatch")
        require(type(receipt.get("exit_code")) is int and receipt["exit_code"] == 0, "check exit code failed")
        require(receipt.get("tested_commit") == manifest["tested_commit"] and receipt.get("source_digest") == source_digest(manifest["sources"]), "check tested source is stale")
        data = verify_ref(receipt.get("log"))
        results[name] = pytest_outcomes(data) if item["checker"] == "pytest" else verify_report(
            data, name, phase, context=manifest, check=item,
            previous_security=security_predecessor(manifest) if name == "security" else None,
        )
    # Join live and cleanup namespaces: an empty unrelated prefix is not cleanup.
    if phase == 5 and require_results:
        for service in ("postgres", "s3"):
            require(results[f"live_{service}"]["namespace"] == results["live_cleanup"]["namespaces"][service]["namespace"], "live cleanup namespace mismatch")
    return {"state": "validated", "phase": phase, "manifest": manifest}


def outside_git(path, repo):
    path = Path(path).resolve()
    require(not path.is_relative_to(Path(repo).resolve()), "receipt output must be outside Git")
    require(path.parent.is_dir(), "receipt parent directory must exist")
    probe = subprocess.run(
        ["git", "-C", str(path.parent), "rev-parse", "--show-toplevel"],
        capture_output=True, timeout=60,
    )
    require(probe.returncode != 0, "receipt output must be outside Git")
    return path


def create_file(path, data):
    scan_credentials(data)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise GateError("immutable output already exists") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def run_phase(path, output):
    result = validate_phase(path, require_results=False)
    manifest = result["manifest"]
    repo = Path(manifest["repo"]).resolve()
    output = outside_git(output, repo)
    require(not output.exists(), "immutable run output already exists")
    if manifest["phase"] > 1:
        previous = verify_handshake(manifest["previous"], expected_phase=manifest["phase"] - 1)
        # Before execution, the predecessor must still be the latest pushed phase.
        require(remote_main(repo) == previous["final_commit"], "remote predecessor changed")
    output.mkdir(mode=0o700)
    for name in PHASE_CHECKS[manifest["phase"]]:
        item = manifest["checks"][name]
        verify_sources(manifest, worktree=True)
        cwd = execution_directory(manifest, name, item, worktree=True)
        child = subprocess.run(
            item["argv"], cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=item.get("timeout_seconds", 3600),
            env={**os.environ, "PYTEST_ADDOPTS": "", "PY_COLORS": "0"},
        )
        log = output / f"{name}.log"
        create_file(log, child.stdout)
        receipt = {
            "schema_version": 1, "check": name, "argv": item["argv"], "cwd": str(cwd),
            "exit_code": child.returncode, "tested_commit": manifest["tested_commit"],
            "source_digest": source_digest(manifest["sources"]), "log": file_ref(log),
        }
        receipt_path = output / f"{name}.result.json"
        create_file(receipt_path, canonical(receipt))
        item["receipt"] = file_ref(receipt_path)
        require(child.returncode == 0, f"{name} exit code failed; evidence preserved")
        if item["checker"] == "pytest":
            pytest_outcomes(child.stdout)
        else:
            verify_report(
                child.stdout, name, manifest["phase"], context=manifest, check=item,
                previous_security=security_predecessor(manifest) if name == "security" else None,
            )
        verify_sources(manifest, worktree=True)
        execution_directory(manifest, name, item, worktree=True)
    candidate = output / "phase.json"
    create_file(candidate, canonical(manifest))
    validate_phase(candidate)
    return {"state": "validated", "receipt": file_ref(candidate)}


def remote_main(repo):
    output = git(repo, "ls-remote", "--exit-code", "origin", "refs/heads/main")
    rows = [row.split() for row in output.splitlines()]
    require(len(rows) == 1 and len(rows[0]) == 2 and rows[0][1] == "refs/heads/main", "remote main observation missing")
    require(COMMIT.fullmatch(rows[0][0]), "remote main SHA invalid")
    return rows[0][0]


def verify_final_commit(repo, tested, final):
    require(isinstance(final, str) and COMMIT.fullmatch(final) and final != tested, "distinct final phase summary commit required")
    ancestor(repo, tested, final)
    changed = git(repo, "diff", "--name-only", tested, final).splitlines()
    require(changed and all(documentation(p) for p in changed), "final phase commit may change only summary documentation/provenance")
    git(repo, "diff", "--check", tested, final)
    present = set(git(repo, "ls-tree", "-r", "--name-only", final).splitlines())
    for path in set(changed) & present:
        scan_credentials(git(repo, "show", f"{final}:{path}").encode())


def finalize_phase(path, final_commit, output):
    manifest = validate_phase(path)["manifest"]
    repo = Path(manifest["repo"]).resolve()
    output = outside_git(output, repo)
    verify_final_commit(repo, manifest["tested_commit"], final_commit)
    require(git(repo, "rev-parse", "HEAD") == final_commit, "final commit is not local main HEAD")
    observed = remote_main(repo)
    require(observed == final_commit, "remote main does not match final phase commit")
    receipt = {
        "schema_version": 1, "state": "pushed", "phase": manifest["phase"],
        "repo": str(repo), "candidate": file_ref(path),
        "tested_commit": manifest["tested_commit"], "final_commit": final_commit,
        "observed_remote_sha": observed, "remote_ref": "refs/heads/main",
        "observed_at": datetime.now(timezone.utc).isoformat(),
    }
    create_file(output, canonical(receipt))
    return receipt


def verify_handshake(reference, *, expected_phase):
    receipt = ref_json(reference)
    require(receipt.get("schema_version") == 1 and receipt.get("state") == "pushed", "actual pushed predecessor receipt required")
    require(receipt.get("phase") == expected_phase, "predecessor does not authorize the next phase")
    verify_ref(receipt.get("candidate"))
    candidate = validate_phase(receipt["candidate"]["path"], worktree=False)["manifest"]
    require(candidate["phase"] == expected_phase and candidate["tested_commit"] == receipt.get("tested_commit"), "predecessor tested commit mismatch")
    require(Path(candidate["repo"]).resolve() == Path(receipt["repo"]).resolve(), "predecessor repository mismatch")
    require(receipt.get("remote_ref") == "refs/heads/main" and receipt.get("observed_remote_sha") == receipt.get("final_commit"), "predecessor push evidence mismatch")
    verify_final_commit(receipt["repo"], receipt["tested_commit"], receipt["final_commit"])
    return receipt


def continue_phase(path, phase):
    require(type(phase) is int and phase in range(2, 7), "next phase must be 2 through 6")
    receipt = verify_handshake(file_ref(path), expected_phase=phase - 1)
    require(receipt["phase"] + 1 == phase, "only immediate next phase may continue")
    require(remote_main(receipt["repo"]) == receipt["final_commit"], "remote predecessor changed")
    require(git(receipt["repo"], "rev-parse", "HEAD") == receipt["final_commit"], "local main differs from predecessor")
    return {"state": "ready", "phase": phase, "previous": file_ref(path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run", "finalize", "continue"):
        sub = commands.add_parser(name)
        sub.add_argument("receipt", type=Path)
        if name in {"run", "finalize"}:
            sub.add_argument("--output", type=Path, required=True)
        if name == "finalize":
            sub.add_argument("--final-commit", required=True)
        if name == "continue":
            sub.add_argument("--phase", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_phase(args.receipt)
            result.pop("manifest")
        elif args.command == "run":
            result = run_phase(args.receipt, args.output)
        elif args.command == "finalize":
            result = finalize_phase(args.receipt, args.final_commit, args.output)
        else:
            result = continue_phase(args.receipt, args.phase)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (ValueError, OSError, KeyError, TypeError, AttributeError, subprocess.SubprocessError) as exc:
        # Never echo command output/argv, paths, or malformed secret-bearing values.
        message = str(exc) if isinstance(exc, GateError) else "malformed or unavailable phase evidence"
        print(json.dumps({"state": "blocked", "reason": message}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
