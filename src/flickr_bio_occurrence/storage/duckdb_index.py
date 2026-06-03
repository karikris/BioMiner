from __future__ import annotations

from glob import glob
from pathlib import Path

import duckdb


def create_qa_views(*, db_path: str | Path, data_root: str | Path) -> Path:
    db = Path(db_path)
    root = Path(data_root)
    db.parent.mkdir(parents=True, exist_ok=True)
    required_view_patterns = {
        "raw_photos": root / "bronze" / "bronze_flickr_photo" / "**" / "*.parquet",
        "occurrence_candidates": root / "silver" / "silver_occurrence_candidate" / "**" / "*.parquet",
        "dwc_occurrence": root / "gold" / "dwc_occurrence" / "**" / "*.parquet",
    }
    optional_view_patterns = {
        "vision_predictions": root / "silver" / "silver_vision_prediction" / "**" / "*.parquet",
    }
    with duckdb.connect(str(db)) as conn:
        for view_name, pattern in required_view_patterns.items():
            escaped_pattern = str(pattern).replace("'", "''")
            conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_pattern}')")
        for view_name, pattern in optional_view_patterns.items():
            if not glob(str(pattern), recursive=True):
                continue
            escaped_pattern = str(pattern).replace("'", "''")
            conn.execute(f"CREATE OR REPLACE VIEW {view_name} AS SELECT * FROM read_parquet('{escaped_pattern}')")
    return db
