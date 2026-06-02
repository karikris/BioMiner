from __future__ import annotations

from pathlib import Path

import duckdb


def create_qa_views(*, db_path: str | Path, data_root: str | Path) -> Path:
    db = Path(db_path)
    root = Path(data_root)
    db.parent.mkdir(parents=True, exist_ok=True)
    view_patterns = {
        "raw_photos": root / "bronze" / "bronze_flickr_photo" / "**" / "*.parquet",
        "occurrence_candidates": root / "silver" / "silver_occurrence_candidate" / "**" / "*.parquet",
        "dwc_occurrence": root / "gold" / "dwc_occurrence" / "**" / "*.parquet",
    }
    with duckdb.connect(str(db)) as conn:
        for view_name, pattern in view_patterns.items():
            escaped_pattern = str(pattern).replace("'", "''")
            conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_pattern}')")
    return db
