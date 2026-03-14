import os
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def aggressive_cleanup(output_root: str, slakh_root: str):
    """
    Deletes redundant directories to solve the 500GB quota issue.
    1. Deletes raw slakh download (~80GB)
    2. Deletes pairs directory (~70GB)
    3. Deletes y.wav in segments (~85GB) IF renderer_target.npy exists.
    """
    output_root = Path(output_root)
    slakh_root = Path(slakh_root)
    
    # 1. Delete Raw Slakh
    if slakh_root.exists():
        logger.info(f"Removing raw dataset: {slakh_root} (~80GB)")
        try:
            shutil.rmtree(slakh_root)
        except Exception as e:
            logger.error(f"Failed to remove raw slakh: {e}")
            
    # 2. Delete Pairs
    pairs_dir = output_root / "pairs" / "slakh"
    if pairs_dir.exists():
        logger.info(f"Removing redundant pairs directory: {pairs_dir} (~70GB)")
        try:
            shutil.rmtree(pairs_dir)
        except Exception as e:
            logger.error(f"Failed to remove pairs: {e}")

    # 3. Delete redundant segment y.wav
    segments_dir = output_root / "segments" / "slakh"
    if segments_dir.exists():
        logger.info("Scanning segments for redundant y.wav files...")
        count = 0
        freed_bytes = 0
        # Walk through all segment dirs
        for track_dir in segments_dir.iterdir():
            if not track_dir.is_dir(): continue
            for seg_dir in track_dir.iterdir():
                if not seg_dir.is_dir(): continue
                y_wav = seg_dir / "y.wav"
                target_npy = seg_dir / "renderer_target.npy"
                
                # Only delete if the target extraction is already done
                if y_wav.exists() and target_npy.exists():
                    try:
                        freed_bytes += y_wav.stat().st_size
                        y_wav.unlink()
                        count += 1
                    except Exception as e:
                        pass
        
        logger.info(f"Removed {count} redundant y.wav files. Freed {freed_bytes / (1024**3):.2f} GB.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--slakh-root", required=True)
    args = parser.parse_args()
    
    aggressive_cleanup(args.output_root, args.slakh_root)
