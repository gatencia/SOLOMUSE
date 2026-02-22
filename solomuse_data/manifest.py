import csv
from pathlib import Path
from typing import List, Dict, Any

MANIFEST_COLUMNS = [
    "dataset",
    "track_id",
    "x_path",
    "y_path",
    "x_path_abs",
    "y_path_abs",
    "sr",
    "channels",
    "duration_s",
    "license",
    "has_mix",
    "mse",
    "corr",
    "notes"
]

def write_manifest_pairs(rows: List[Dict[str, Any]], path: str):
    """
    Write a list of pair metadata to a CSV manifest.
    
    Args:
        rows: List of dicts matching MANIFEST_COLUMNS.
        path: Output path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    with open(p, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        
        for row in rows:
            # Filter row to only include relevant columns to avoid errors if extra data is passed
            filtered_row = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
            writer.writerow(filtered_row)
