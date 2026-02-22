import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import hashlib
import logging

logger = logging.getLogger(__name__)

class IntentDataset(Dataset):
    """
    Dataset for training the Intent Planner.
    Reads manifest_intent.csv, loads situation.npy (input) and intent_targets.npy (target).
    """
    def __init__(self, manifest_path: str, split: str = "train", val_ratio: float = 0.1):
        """
        Args:
            manifest_path: Path to manifest_intent.csv
            split: "train" or "val"
            val_ratio: Fraction of data to use for validation.
        """
        self.manifest_path = Path(manifest_path)
        if self.manifest_path.name != "manifest_intent.csv":
            logger.warning(f"Expected manifest_intent.csv, got {self.manifest_path.name}")
            
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            logger.error(f"Failed to read manifest {manifest_path}: {e}")
            self.rows = []
            return
            
        # Deterministic split via hash(track_id)
        # Using track_id ensures segments from the same track stay in the same split
        def get_split(track_id):
            h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
            return "val" if (h % 100) < (val_ratio * 100) else "train"
            
        df["split"] = df["track_id"].apply(get_split)
        
        self.rows = df[df["split"] == split].to_dict('records')
        logger.info(f"Loaded {len(self.rows)} items for {split} split from {manifest_path}")

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
        segments_dir = self.manifest_path.parent
        track_id = row.get("track_id")
        segment_id = row.get("segment_id")
        
        # Try finding the directory
        seg_dir = segments_dir / track_id / segment_id
        
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
        # intent_mat: [F, 7]
        intent_mat = np.load(intent_path).astype(np.float32)
        
        # Broadcast situation vector to [F, 32] so it matches time resolution
        # This acts as a global conditioning vector at every timestep
        F = intent_mat.shape[0]
        sit_mat = np.tile(sit_vec, (F, 1))
        
        return torch.from_numpy(sit_mat), torch.from_numpy(intent_mat)
