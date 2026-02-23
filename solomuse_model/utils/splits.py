import pandas as pd
import numpy as np
import logging
from pathlib import Path
from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

def create_track_grouped_splits(manifest_path: str | Path, 
                                split_manifest_path: str | Path, 
                                cfg: PipelineConfig, 
                                force_regenerate: bool = False) -> pd.DataFrame:
    """
    Reads a dataset manifest and deterministically assigns every segment from a 
    given track_id into the exact same split (train/val/test) to prevent data leakage.
    
    Args:
        manifest_path: Source manifest containing at least `track_id` and `segment_id`.
        split_manifest_path: Destination path to save the generated splits.
        cfg: PipelineConfig containing seed and split ratios.
        force_regenerate: Overwrite existing split file if True.
        
    Returns:
        DataFrame containing the split column.
    """
    manifest_path = Path(manifest_path)
    split_manifest_path = Path(split_manifest_path)
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Source manifest not found at {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    if "track_id" not in df.columns or "segment_id" not in df.columns:
        raise ValueError(f"Manifest at {manifest_path} must contain 'track_id' and 'segment_id' columns.")

    if split_manifest_path.exists() and not force_regenerate:
        logger.info(f"Loading existing persistent splits from {split_manifest_path}")
        split_df = pd.read_csv(split_manifest_path)
        
        # Backward compatibility check: Are tracks leaked across splits?
        track_splits = split_df.groupby("track_id")["split"].nunique()
        leaking_tracks = track_splits[track_splits > 1]
        
        if not leaking_tracks.empty:
            logger.warning(f"CRITICAL: Found {len(leaking_tracks)} tracks leaking across train/val/test in existing split file!")
            logger.warning("This means the previous split was randomized by segment_id, not track_id.")
            logger.warning("Pass `--force-regenerate-splits` to rebuild, or delete the file manually.")
            raise RuntimeError(f"Data leakage detected in existing split manifest: {split_manifest_path}")
            
        return split_df

    logger.info("Generating deterministic TRACK-GROUPED splits...")
    
    # Extract unique tracks and shuffle them deterministically
    unique_tracks = df["track_id"].unique()
    np.random.seed(cfg.seed)
    np.random.shuffle(unique_tracks)
    
    n_total_tracks = len(unique_tracks)
    if n_total_tracks == 0:
        raise ValueError("Manifest contains no tracks. Cannot generate splits.")
        
    train_ratio = getattr(cfg, "intent_train_ratio", 0.8)
    val_ratio = getattr(cfg, "intent_val_ratio", 0.1)
    
    # 1. Calculate boundaries
    n_train = max(1, int(n_total_tracks * train_ratio))
    n_val = int(n_total_tracks * val_ratio)
    n_test = n_total_tracks - n_train - n_val
    
    if n_test < 0:
        n_test = 0
        n_val = n_total_tracks - n_train
        n_train = max(1, n_total_tracks - n_val)

    # 2. Slice tracks
    train_tracks = set(unique_tracks[:n_train])
    val_tracks = set(unique_tracks[n_train:n_train+n_val])
    test_tracks = set(unique_tracks[n_train+n_val:])
    
    # 3. Assign splits segment by segment
    def assign_split(tid):
        if tid in train_tracks: return "train"
        if tid in val_tracks: return "val"
        return "test"
        
    df["split"] = df["track_id"].apply(assign_split)
    
    # Save the deterministic output
    df.to_csv(split_manifest_path, index=False)
    logger.info(f"Saved persistent tracking splits to {split_manifest_path}")
    logger.info(f"Track counts -> Train: {n_train} | Val: {n_val} | Test: {n_test}")
    
    return df
