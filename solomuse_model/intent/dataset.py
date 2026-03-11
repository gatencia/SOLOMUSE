import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import logging
from solomuse_data.config import PipelineConfig
import logging

logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    """
    Dataset for training the Intent Planner.
    Reads manifest_intent.csv, loads situation.npy (input) and intent_targets.npy (target).
    """
    def __init__(self, data_df: pd.DataFrame, cfg: PipelineConfig):
        """
        Args:
            data_df: DataFrame containing the manifest rows for this split.
            cfg: Configuration object (e.g., PipelineConfig) to access output_root.
        """
        self.rows = data_df.to_dict('records')
        self.segments_root = Path(cfg.output_root) / "segments"
        self.fail_input = getattr(cfg, "intent_fail_on_nonfinite_input", True)
        self.fail_target = getattr(cfg, "intent_fail_on_nonfinite_target", True)
        logger.info(f"Loaded {len(self.rows)} items for dataset split.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        
        # Paths are typically constructed relative to the dataset root, 
        # but the manifest has x_path_abs or we can reconstruct from segment_id
        # Actually, manifest_intent.csv from build_pairs might not have absolute paths if rebuilt.
        # Let's assume standard structure: output_root/segments/{dataset}/{track_id}/{segment_id}
        
        # Since we ran run_intent_target_build, we know files are in segment_id folder
        # We need the absolute path. If the manifest has x_path or dataset/track/segment, we rebuild it.
        # The segment manifest typically lives in segments/{dataset}/manifest.csv
        track_id = row.get("track_id")
        segment_id = row.get("segment_id")
        dataset_name = row.get("dataset", "slakh") # Default fallback if missing
        
        # Try finding the directory
        seg_dir = self.segments_root / dataset_name / track_id / segment_id
        
        situation_path = seg_dir / "situation.npy"
        intent_path = seg_dir / "intent_targets.npy"
        
        if not situation_path.exists() or not intent_path.exists():
            # Fallback for dynamic reconstruction if x_path_abs was written
            if "x_path_abs" in row and Path(row["x_path_abs"]).parent.exists():
                seg_dir = Path(row["x_path_abs"]).parent
                situation_path = seg_dir / "situation.npy"
                intent_path = seg_dir / "intent_targets.npy"
                
            if not situation_path.exists() or not intent_path.exists():
                print(f"DEBUG IntentDataset Error for segment_id={repr(segment_id)} track_id={repr(track_id)}")
                print(f"  segments_root={repr(str(self.segments_root))}")
                print(f"  dataset_name={repr(dataset_name)}")
                print(f"  Calculated situation_path={repr(str(situation_path))} | Exists? {situation_path.exists()}")
                print(f"  Calculated intent_path={repr(str(intent_path))} | Exists? {intent_path.exists()}")
                if seg_dir.exists():
                     print(f"  seg_dir exists. Contents: {os.listdir(seg_dir)}")
                else:
                     print(f"  seg_dir does NOT exist.")
                     if seg_dir.parent.exists():
                         print(f"  seg_dir.parent exists. Contents: {os.listdir(seg_dir.parent)}")
                # Return zeros as fallback to prevent crash, though it ruins batching if unchecked. 
                # Ideally we map a viable shape or filter these out in init.
                # Since we don't know F here, we raise ValueError
                raise ValueError(f"Missing files context for {segment_id}")

        # Load
        # sit_vec: [32]
        sit_vec = np.load(situation_path).astype(np.float32)
        
        if self.fail_input and not np.isfinite(sit_vec).all():
            raise ValueError(f"CRITICAL: Non-finite values (NaN/Inf) detected in Situation input array for track={track_id}, segment={segment_id} at {situation_path}")
            
        # intent_mat: [F, 7]
        intent_mat = np.load(intent_path).astype(np.float32)
        
        if self.fail_target and not np.isfinite(intent_mat).all():
            raise ValueError(f"CRITICAL: Non-finite values (NaN/Inf) detected in Intent target array for track={track_id}, segment={segment_id} at {intent_path}")
        
        # Broadcast situation vector to [F, 32] so it matches time resolution
        # This acts as a global conditioning vector at every timestep
        F = intent_mat.shape[0]
        sit_mat = np.tile(sit_vec, (F, 1))
        
        return {
            "situation": torch.from_numpy(sit_mat), 
            "intent": torch.from_numpy(intent_mat),
            "track_id": track_id,
            "segment_id": segment_id
        }

def build_intent_dataloaders(cfg: PipelineConfig, dataset_name: str):
    """
    Builds the Train, Validation, and Test dataloaders for the baseline intent planner.
    Reads splits from target/manifest_intent_splits.csv if it exists.
    If not, it deterministically partitions manifest_intent.csv.
    """
    from torch.utils.data import DataLoader
    from solomuse_model.utils.splits import create_track_grouped_splits
    
    targets_dir = Path(cfg.output_root) / "segments" / dataset_name
    manifest_path = targets_dir / "manifest_intent.csv"
    split_manifest_path = targets_dir / "manifest_intent_splits.csv"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Intent manifest not found at {manifest_path}. Run intent target extraction first.")
        
    df = pd.read_csv(manifest_path)
    
    force_regenerate = getattr(cfg, "force_regenerate_splits", False)
    
    # 1. Split Generation / Loading using track-grouped utility
    try:
        df = create_track_grouped_splits(manifest_path, split_manifest_path, cfg, force_regenerate)
    except RuntimeError as repr_err:
        logger.error(str(repr_err))
        raise
        
    if "split" not in df.columns:
        raise ValueError(f"'split' column missing after processing {split_manifest_path}")

    # Explicit leakage assertion as requested by user
    track_splits = df.groupby("track_id")["split"].nunique()
    leaking_tracks = track_splits[track_splits > 1]
    if not leaking_tracks.empty:
        raise AssertionError(f"FATAL: Track leakage detected across splits. The following tracks appear in multiple sets: {leaking_tracks.index.tolist()}")

    # 2. Build Datasets
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    
    train_ds = IntentDataset(train_df, cfg)
    val_ds = IntentDataset(val_df, cfg)
    test_ds = IntentDataset(test_df, cfg)
    
    # 3. Build Dataloaders
    train_loader = DataLoader(
        train_ds, 
        batch_size=cfg.intent_batch_size, 
        shuffle=True, 
        drop_last=False, # Must be False or tiny datasets will fail
        pin_memory=torch.backends.mps.is_available() or torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=cfg.intent_batch_size, 
        shuffle=False, 
        drop_last=False
    )

    test_loader = DataLoader(
        test_ds, 
        batch_size=cfg.intent_batch_size, 
        shuffle=False, 
        drop_last=False
    )
    
    logger.info(f"Intent Datasets Built -> "
                f"Train: {len(train_ds)} segs ({train_df['track_id'].nunique()} tracks) | "
                f"Val: {len(val_ds)} segs ({val_df['track_id'].nunique()} tracks) | "
                f"Test: {len(test_ds)} segs ({test_df['track_id'].nunique()} tracks)")
    
    return train_loader, val_loader, test_loader
