import os
import csv
import logging
import soundfile as sf
from pathlib import Path
from solomuse_data.manifest import MANIFEST_COLUMNS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def recover_manifest(dataset_name: str, output_root: str):
    """
    Rebuilds index/slakh/manifest.csv by scanning the pairs folder.
    """
    output_root = Path(output_root)
    pairs_dir = output_root / "pairs" / dataset_name
    manifest_dir = output_root / "index" / dataset_name
    manifest_path = manifest_dir / "manifest.csv"
    
    if not pairs_dir.exists():
        logger.error(f"Pairs directory not found: {pairs_dir}")
        return

    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_rows = []
    track_dirs = sorted([d for d in pairs_dir.iterdir() if d.is_dir()])
    
    logger.info(f"Scanning {len(track_dirs)} tracks in {pairs_dir}...")
    
    for t_dir in track_dirs:
        track_id = t_dir.name
        x_path = t_dir / "x_backing.wav"
        y_path = t_dir / "y_solo.wav"
        
        if x_path.exists() and y_path.exists():
            try:
                # Get audio metadata
                info = sf.info(str(x_path))
                
                row = {
                    "dataset": dataset_name,
                    "track_id": track_id,
                    "x_path": str(x_path.relative_to(output_root)),
                    "y_path": str(y_path.relative_to(output_root)),
                    "x_path_abs": str(x_path),
                    "y_path_abs": str(y_path),
                    "sr": info.samplerate,
                    "channels": info.channels,
                    "duration_s": info.duration,
                    "has_mix": True,
                    "notes": "Recovered from disk failure"
                }
                manifest_rows.append(row)
            except Exception as e:
                logger.warning(f"Could not read {track_id}: {e}")
    
    logger.info(f"Found {len(manifest_rows)} valid pairs. Writing to {manifest_path}...")
    
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in manifest_rows:
            filtered_row = {k: row.get(k, "") for k in MANIFEST_COLUMNS}
            writer.writerow(filtered_row)
            
    logger.info("Manifest recovered successfully!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    
    recover_manifest(args.dataset, args.output_root)
