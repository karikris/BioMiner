# Ground Zero EDA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce an immutable, reproducible, non-mutating EDA run for the raw GBIF Ground Zero occurrence Parquet, with an analytical work Parquet, Excel-friendly CSV exports, a visual PowerPoint deck, and a final manifest.

**Architecture:** A small report module will query the source Parquet directly with DuckDB and materialize only deterministic aggregate tables. It will keep the source untouched, write the work Parquet before compatibility CSVs and charts, then write the manifest last after hashing every output. The PowerPoint deck will be rendered from those aggregates so it introduces no independent analytical state.

**Tech Stack:** Python 3.14, DuckDB, Polars, PyArrow, Matplotlib, Pillow, python-pptx, pytest.

---

### Task 1: Define and test deterministic EDA artifact generation

**Files:**
- Create: `tests/test_ground_zero_eda.py`
- Create: `src/biominer/reports/ground_zero_eda.py`

- [ ] **Step 1: Write failing tests**

Create a small Parquet fixture with required GBIF occurrence columns and assert that `build_ground_zero_eda_run()` creates `eda_work.parquet`, formula-safe UTF-8-BOM CSVs, a `.pptx` deck, charts, and a final manifest. Assert the summary includes a `NULL` month category and that rerunning into the same directory raises `FileExistsError`.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ground_zero_eda.py -q`

Expected: FAIL because `biominer.reports.ground_zero_eda` does not yet exist.

- [ ] **Step 3: Implement the minimal generator**

Implement `build_ground_zero_eda_run(source_path, output_dir, source_manifest_path=None, top_n=20)` to validate its schema, refuse output overwrite, aggregate selected completeness, quality, temporal, geographic, taxonomic, and contributor measures with DuckDB, write the aggregate work Parquet plus Excel-safe CSVs, render charts and a PowerPoint deck, and write an output-hashed manifest last.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ground_zero_eda.py -q`

Expected: PASS.

### Task 2: Add a reproducible command surface

**Files:**
- Create: `scripts/run_ground_zero_eda.py`
- Modify: `README.md` only if a stable reporting command is documented there.

- [ ] **Step 1: Write a failing command-surface test**

Add a test that invokes `scripts/run_ground_zero_eda.py` with a fixture source and output directory and asserts that it reports the manifest path and exits successfully.

- [ ] **Step 2: Run the focused test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_ground_zero_eda.py -q`

Expected: FAIL because the command script does not exist.

- [ ] **Step 3: Implement the command**

Add an argparse command with explicit `--source`, `--source-manifest`, `--output`, and `--top-n` options. It must pass those values directly to `build_ground_zero_eda_run()` and print the final manifest as JSON.

- [ ] **Step 4: Run the focused test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_ground_zero_eda.py -q`

Expected: PASS.

### Task 3: Run and validate the Ground Zero EDA

**Files:**
- Create: `reports/ground_zero_eda/run_id=<UTC timestamp>/...` (runtime artifacts, not source-controlled)

- [ ] **Step 1: Run against the active source**

Run `scripts/run_ground_zero_eda.py` with `data/reference/gbif_global_papilionoidea_parquet/occurrence.parquet`, its DWCA receipt, and a new timestamped directory beneath `reports/ground_zero_eda/`.

- [ ] **Step 2: Validate artifact structure and provenance**

Check that the working Parquet and each CSV are readable, the deck opens as a PowerPoint package, chart files are non-empty, CSVs begin with a UTF-8 BOM, and the manifest is written after listing SHA-256 checksums for every created artifact.

- [ ] **Step 3: Run targeted and broader checks**

Run: `.venv/bin/python -m pytest tests/test_ground_zero_eda.py -q`

Run: `.venv/bin/python -m pytest tests/test_build_gbif_occurrences_parquet.py tests/test_build_dwca_members_parquet.py -q`

Expected: all selected checks pass without modifying the Ground Zero source.
