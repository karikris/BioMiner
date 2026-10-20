"""Frozen application build admission."""
import importlib.util
from pathlib import Path
import sys
import tarfile

import pytest

from scripts import build_native_project as builder


def test_project_build_rejects_archive_without_exact_git_revision(tmp_path):
    script = Path(__file__).parents[1] / 'scripts/build_native_project.py'
    assert script.exists(), 'frozen application builder missing'
    spec = importlib.util.spec_from_file_location('native_project', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    archive = tmp_path / 'source.tar'
    with tarfile.open(archive, 'w'):
        pass
    with pytest.raises(ValueError, match='Git revision'):
        module.source_revision(archive)


def test_project_build_rerun_preserves_every_existing_output_byte(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'pyproject.toml').write_text('[project]\nname="biominer"\n')
    work = tmp_path / 'build'
    work.mkdir()
    revision = 'a' * 40
    source = work / f'source-{revision}-{builder.record(project / "pyproject.toml")["sha256"][:12]}'
    source.mkdir()
    (source / 'pyproject.toml').write_bytes(b'previous overlay')
    (work / f'source-{revision}.tar').write_bytes(b'previous source archive')
    (work / 'receipt.json').write_bytes(b'previous receipt')
    (work / 'build.log').write_bytes(b'previous build log')
    before = {str(path.relative_to(work)): path.read_bytes() for path in work.rglob('*') if path.is_file()}
    monkeypatch.setattr(sys, 'argv', ['build_native_project.py', '--root', str(project),
                                    '--output', str(work), '--python', sys.executable])
    monkeypatch.setattr(builder.subprocess, 'check_output', lambda *args, **kwargs: revision)
    def archive(command, **kwargs):
        Path(command[command.index('--output') + 1]).write_bytes(b'rewritten source archive')
    monkeypatch.setattr(builder.subprocess, 'run', archive)
    monkeypatch.setattr(builder, 'source_revision', lambda path: revision)
    with pytest.raises((ValueError, FileExistsError), match='exist'):
        builder.main()
    after = {str(path.relative_to(work)): path.read_bytes() for path in work.rglob('*') if path.is_file()}
    assert after == before


def test_project_build_rejects_even_empty_existing_output_before_git(tmp_path, monkeypatch):
    work = tmp_path / 'build'
    work.mkdir()
    monkeypatch.setattr(sys, 'argv', ['build_native_project.py', '--root', str(tmp_path),
                                    '--output', str(work), '--python', sys.executable])
    def unexpected(*args, **kwargs):
        pytest.fail('existing generation must be rejected before Git or build commands')
    monkeypatch.setattr(builder.subprocess, 'check_output', unexpected)
    monkeypatch.setattr(builder.subprocess, 'run', unexpected)
    with pytest.raises(FileExistsError, match='exist'):
        builder.main()
    assert list(work.iterdir()) == []
