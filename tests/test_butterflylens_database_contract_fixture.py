"""Exact-pin ButterflyLens wire-schema and database-boundary fixture checks."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import textwrap

import pytest

from biominer.integration.butterflylens_geographic_export import (
    butterflylens_geographic_cell_documents,
    export_butterflylens_geographic_impact,
)
from biominer.integration.butterflylens_model_export import (
    export_butterflylens_model_evidence,
)
from biominer.integration.butterflylens_pool_handoff import (
    BUTTERFLYLENS_PINNED_COMMIT,
)
from biominer.integration.butterflylens_review_export import (
    butterflylens_campaign_document,
    butterflylens_maturity_documents,
    export_butterflylens_review_evidence,
)
from helpers.butterflylens_handoff_fixture import (
    build_butterflylens_complete_fixture,
)


ROOT = Path(__file__).parents[1]
BUTTERFLYLENS = ROOT.parent / "ButterflyLens"
SCHEMAS = (
    "common.schema.json",
    "verification-contracts.schema.json",
    "verification-campaign.schema.json",
    "geographic-impact-contracts.schema.json",
    "geographic-impact-cell.schema.json",
    "classification-maturity.schema.json",
)
COMPATIBILITY_FIXTURE = ROOT / "tests/fixtures/product_pool_handoff_compatibility.json"


def _git_show(path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{BUTTERFLYLENS_PINNED_COMMIT}:{path}"],
        cwd=BUTTERFLYLENS,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def test_target_documents_validate_against_exact_committed_json_schemas(
    tmp_path: Path,
) -> None:
    python = BUTTERFLYLENS / ".venv/bin/python"
    if not python.is_file():
        pytest.skip("ButterflyLens contract-validation environment is unavailable")
    schema_directory = tmp_path / "schemas"
    schema_directory.mkdir()
    for filename in SCHEMAS:
        content = _git_show(f"packages/contracts/schemas/{filename}")
        (schema_directory / filename).write_text(content, encoding="utf-8")

    fixture = build_butterflylens_complete_fixture()
    documents = [
        {
            "schema_id": "urn:butterflylens:schema:verification-campaign:v1.0.0",
            "document": butterflylens_campaign_document(fixture["review"].campaign),
        },
        *(
            {
                "schema_id": ("urn:butterflylens:schema:geographic-impact-cell:v1.0.0"),
                "document": document,
            }
            for document in butterflylens_geographic_cell_documents(
                fixture["geographic"]
            )
        ),
        *(
            {
                "schema_id": (
                    "urn:butterflylens:schema:classification-maturity:v1.0.0"
                ),
                "document": document,
            }
            for document in butterflylens_maturity_documents(
                fixture["review"].classification_maturity
            )
        ),
    ]
    runner = textwrap.dedent(
        r"""
        import json
        from pathlib import Path
        import sys
        from jsonschema import Draft202012Validator, FormatChecker
        from referencing import Registry, Resource

        directory = Path(sys.argv[1])
        schemas = {}
        registry = Registry()
        for path in sorted(directory.glob("*.schema.json")):
            schema = json.loads(path.read_text())
            Draft202012Validator.check_schema(schema)
            schemas[schema["$id"]] = schema
            registry = registry.with_resource(
                schema["$id"], Resource.from_contents(schema)
            )
        failures = []
        for case in json.load(sys.stdin):
            validator = Draft202012Validator(
                schemas[case["schema_id"]],
                registry=registry,
                format_checker=FormatChecker(),
            )
            errors = sorted(
                validator.iter_errors(case["document"]),
                key=lambda error: list(error.path),
            )
            failures.extend(
                f"{case['schema_id']}:{list(error.path)}:{error.message}"
                for error in errors
            )
        if failures:
            raise SystemExit("\n".join(failures))
        """
    )
    result = subprocess.run(
        [str(python), "-c", runner, str(schema_directory)],
        input=json.dumps(documents),
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert len(documents) == 4


def test_exact_committed_database_fixtures_enforce_fail_closed_ingestion() -> None:
    map_fixture = _git_show("supabase/tests/database/005_map_impact_schema.test.sql")
    review_fixture = _git_show("supabase/tests/database/004_review_schema.test.sql")
    repeated = _git_show(
        "supabase/migrations/20260718013600_repeated_independent_assignments.sql"
    )
    blind = _git_show("supabase/migrations/20260718014600_blind_review_disclosure.sql")
    append_only = _git_show(
        "supabase/migrations/20260718015500_append_only_review_submission.sql"
    )
    rls = _git_show("supabase/migrations/20260717215002_rls_role_policies.sql")

    for value in (
        "unavailable Flickr evidence cannot be encoded as zero",
        "eligible release candidate requires every gate",
        "blocked release candidate requires blockers",
        "approval cannot bypass gates or qualified authority",
        "available model count requires a measured value",
        "impact cells are append-only",
    ):
        assert value in map_fixture
    for value in (
        "same reviewer cannot receive the same item twice",
        "review events are append-only",
        "event identity must match its assignment",
        "model agreement cannot define reviewer truth",
    ):
        assert value in review_fixture
    for value in (
        "repeated-independent-v1",
        "assignments_enforce_repeated_independence",
        "assignment identity and policy fields are immutable",
        "assignment sequence must be the next independent round",
    ):
        assert value in repeated
    for value in (
        "verification_campaigns_enforce_blind_review",
        "blind_review_assignments",
        "community review campaign must preserve the blind evidence boundary",
    ):
        assert value in blind
    for value in (
        "review_events_enforce_append_lineage",
        "review correction crosses assignment identity",
        "review context does not preserve assignment and blind lineage",
    ):
        assert value in append_only
    for value in (
        "Worker credentials and service_role remain server-only",
        "Review events are insert-only",
        "candidate_state in ('approved', 'exported') and all_release_gates_passed",
    ):
        assert value in rls


def test_frozen_butterflylens_descriptors_match_all_ten_exported_roles(
    tmp_path: Path,
) -> None:
    fixture = build_butterflylens_complete_fixture()
    model = export_butterflylens_model_evidence(
        project=fixture["project"],
        run=fixture["run"],
        layer=fixture["layer"],
        output_root=tmp_path,
    )
    geographic = export_butterflylens_geographic_impact(
        frame=fixture["geographic"], output_root=tmp_path
    )
    review = export_butterflylens_review_evidence(
        layer=fixture["review"], output_root=tmp_path
    )
    observed = [*model.artifacts, geographic.artifact, *review.artifacts]
    frozen = json.loads(COMPATIBILITY_FIXTURE.read_text(encoding="utf-8"))[
        "butterflylens"
    ]["artifacts"]

    assert observed == frozen
