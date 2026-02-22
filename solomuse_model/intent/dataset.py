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
                logger.error(f"Missing artifacts for segment {segment_id} at {seg_dir}")
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
    
    targets_dir = Path(cfg.output_root) / "segments" / dataset_name
    manifest_path = targets_dir / "manifest_intent.csv"
    split_manifest_path = targets_dir / "manifest_intent_splits.csv"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Intent manifest not found at {manifest_path}. Run intent target extraction first.")
        
    df = pd.read_csv(manifest_path)
    
    # 1. Split Generation / Loading
    if not split_manifest_path.exists():
        logger.info("No split manifest found. Generating deterministic splits...")
        
        # Shuffle deterministically
        df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
        
        n_total = len(df)
        if n_total == 0:
            raise ValueError("Manifest is empty. Cannot generate splits.")
            
        # Enforce at least 1 train sample
        n_train = max(1, int(n_total * getattr(cfg, "intent_train_ratio", 0.8)))
        n_val = int(n_total * getattr(cfg, "intent_val_ratio", 0.1))
        
        # The rest goes to test (could be 0)
        n_test = n_total - n_train - n_val
        if n_test < 0:
            n_test = 0
            n_val = n_total - n_train
            n_train = max(1, n_total - n_val)
            
        splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
        
        # Pad differences if rounding errors
        if len(splits) > n_total:
            splits = splits[:n_total]
        elif len(splits) < n_total:
            splits += ["test"] * (n_total - len(splits))
            
        df["split"] = splits
        
        # Save so future runs (eval/infer) don't drift
        df.to_csv(split_manifest_path, index=False)
        logger.info(f"Saved persistent splits to {split_manifest_path}")
    else:
        logger.info(f"Loading persistent splits from {split_manifest_path}")
        df = pd.read_csv(split_manifest_path)
        if "split" not in df.columns:
            raise ValueError(f"'split' column missing in {split_manifest_path}")

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
    
    logger.info(f"Intent Datasets Built -> Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    
    return train_loader, val_loader, test_loader
