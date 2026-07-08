from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_vision_control_modules_import_without_model_runtime_dependencies() -> None:
    script = """
from __future__ import annotations

import importlib.abc
import importlib.util
import sys

blocked_roots = {"open_clip", "torch", "ultralytics"}


class BlockOptionalVisionDeps(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):  # noqa: ANN001, ANN201
        if fullname.split(".", 1)[0] in blocked_roots:
            raise ModuleNotFoundError(f"blocked optional vision dependency: {fullname}")
        return None


sys.meta_path.insert(0, BlockOptionalVisionDeps())
for module_name in (
    "biominer.cli",
    "biominer.detection.yoloe26_detector",
    "biominer.bioclip.bioclip_worker",
):
    __import__(module_name)
print("ok")
"""
    env = os.environ.copy()
    source_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = source_path if not env.get("PYTHONPATH") else f"{source_path}{os.pathsep}{env['PYTHONPATH']}"

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
