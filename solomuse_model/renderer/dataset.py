import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
import hashlib
from pathlib import Path
from typing import Optional

from solomuse_model.renderer.types import RendererInputV1, RendererTargetV1

logger = logging.getLogger(__name__)

class RendererDataset(Dataset):
    """
    Dataset for training the Renderer.
    Yields (x_audio, intent_targets, situation_vector, target_codes).
    """
    def __init__(self, manifest_path: str, split: str = "train", val_ratio: float = 0.1):
        self.manifest_path = Path(manifest_path)
        
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            logger.error(f"Failed to read manifest {manifest_path}: {e}")
            self.rows = []
            return
            
        if "split" not in df.columns:
            logger.warning(f"Manifest {manifest_path} lacks a 'split' column. Falling back to simple track hashing.")
            def get_split(track_id):
                h = int(hashlib.md5(str(track_id).encode()).hexdigest(), 16)
                return "val" if (h % 100) < (val_ratio * 100) else "train"
            df["split"] = df["track_id"].apply(get_split)
            
        self.rows = df[df["split"] == split].to_dict('records')
        logger.info(f"Loaded {len(self.rows)} items for {split} split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        segments_dir = self.manifest_path.parent
        track_id = row.get("track_id")
        segment_id = row.get("segment_id")
        
        seg_dir = segments_dir / track_id / segment_id
        
        x_path = seg_dir / "x.wav"
        sit_path = seg_dir / "situation.npy"
        intent_path = seg_dir / "intent_targets.npy"
        target_path = seg_dir / "renderer_target.npy"
        
        if not target_path.exists():
            raise FileNotFoundError(f"Missing {target_path}")
            
        # Load arrays
        # Backing is loaded as path since loading audio in Dataset `__getitem__` can be slow,
        # but for simple V1 datasets it's acceptable. We return the path to let collation sort it out
        # or load directly. We load directly here for simplicity of returning tensors.
        import soundfile as sf
        x_audio, sr = sf.read(str(x_path), dtype="float32", always_2d=True)
        if x_audio.shape[1] > 1:
            x_audio = np.mean(x_audio, axis=1)
        else:
            x_audio = x_audio.flatten()
            
        sit_vec = np.load(sit_path).astype(np.float32)
        intent_mat = np.load(intent_path).astype(np.float32)
        target_codes = np.load(target_path).astype(np.float32) # [F, chunk_size] for WaveChunk
        
        inp = RendererInputV1(
            x_audio=x_audio,
            x_audio_path=str(x_path),
            intent_sequence=intent_mat,
            situation_vector=sit_vec,
            sr=sr
        )
        
        targ = RendererTargetV1(
            y_audio=None,
            y_audio_path=str(seg_dir / "y.wav"),
            target_codes=target_codes
        )
        
        return inp, targ
