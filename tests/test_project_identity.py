from __future__ import annotations

import biominer
from biominer.cli import build_parser, run


def test_package_identity_is_biominer() -> None:
    assert biominer.__version__ == "0.1.0"
    assert "biominer" in build_parser().prog


def test_cli_version_reports_biominer(capsys) -> None:
    assert run(build_parser().parse_args(["--version"])) == 0

    assert capsys.readouterr().out.strip() == "biominer 0.1.0"
