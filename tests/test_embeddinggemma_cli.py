from biominer.cli import build_parser, run


def test_cli_exposes_experimental_workflow_without_loading_model():
    args = build_parser().parse_args(
        [
            "vision",
            "embeddinggemma",
            "prepare",
            "--input",
            "example.parquet",
            "--label-bank",
            "labels.md",
            "--output-dir",
            "experiment",
        ]
    )
    assert args.embeddinggemma_command == "prepare"


def test_model_free_comparison_requires_explicit_inputs(tmp_path, capsys):
    args = build_parser().parse_args(
        ["vision", "embeddinggemma", "compare", "--output-dir", str(tmp_path)]
    )
    assert run(args) == 2
    assert "configuration" in capsys.readouterr().err.lower()


def test_runtime_check_is_explicit_about_device_and_dtype():
    args = build_parser().parse_args(
        ["vision", "embeddinggemma", "runtime-check", "--output-dir", "receipt"]
    )
    assert args.device == "mps"
    assert args.dtype == "float32"


def test_second_run_accepts_ordered_configurations_without_loading_model():
    configurations = ["1120-256", "560-256", "1120-768", "560-768"]
    args = build_parser().parse_args(
        [
            "vision",
            "embeddinggemma",
            "run",
            "--input",
            "eligible.parquet",
            "--label-bank",
            "labels.md",
            "--output-dir",
            "experiment-v2",
            "--binary",
            *[part for value in configurations for part in ("--configuration", value)],
        ]
    )
    assert args.configuration == configurations
    assert args.binary is True
    assert args.dtype == "float32"


def test_standalone_runtime_check_accepts_new_budget():
    args = build_parser().parse_args(
        [
            "vision",
            "embeddinggemma",
            "runtime-check",
            "--output-dir",
            "receipt",
            "--budget",
            "560",
            "--budget",
            "1120",
        ]
    )
    assert args.budget == [560, 1120]
