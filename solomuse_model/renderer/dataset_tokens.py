import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class RendererTokenDataset(Dataset):
    """
    Dataset for training the V2 Transformer Token LM Renderer.
    Yields (x_tokens, intent_aligned, situation, y_tokens).
    """
    def __init__(self, manifest_path: str, split: str = "train"):
        self.manifest_path = Path(manifest_path)
        
        try:
            df = pd.read_csv(manifest_path)
        except Exception as e:
            logger.error(f"Failed to read token manifest {manifest_path}: {e}")
            self.rows = []
            return
            
        if "split" not in df.columns:
            raise ValueError(f"FATAL: Manifest {manifest_path} lacks a 'split' column! Must use `manifest_renderer_tokens.csv`.")
            
        self.rows = df[df["split"] == split].to_dict('records')
        logger.info(f"Loaded {len(self.rows)} items for {split} split")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        
        x_path = Path(row["x_tokens_path"])
        y_path = Path(row["y_tokens_path"])
        
        # situation and intent_aligned are implicitly in the same directory as x/y
        seg_dir = x_path.parent
        sit_path = seg_dir / "situation.npy"
        intent_path = seg_dir / "intent_aligned.npy"
        
        try:
            x_tokens = np.load(str(x_path)).astype(np.int64)
            y_tokens = np.load(str(y_path)).astype(np.int64)
            sit_vec = np.load(str(sit_path)).astype(np.float32)
            intent_aligned = np.load(str(intent_path)).astype(np.float32)
        except (FileNotFoundError, PermissionError) as e:
            raise FileNotFoundError(f"Missing required token components in {seg_dir}: {e}")
        
        # Time dimension enforcement min length
        min_f = min(x_tokens.shape[0], y_tokens.shape[0], intent_aligned.shape[0])
        x_tokens = x_tokens[:min_f]
        y_tokens = y_tokens[:min_f]
        intent_aligned = intent_aligned[:min_f]
        
        return {
            "x_tokens": torch.from_numpy(x_tokens),
            "y_tokens": torch.from_numpy(y_tokens),
            "intent_aligned": torch.from_numpy(intent_aligned),
            "situation": torch.from_numpy(sit_vec)
        }

def token_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Pads variable length sequences to the max length in the batch.
    x_tokens/y_tokens: padded with 0 (assuming vocab size >= 0, typical for EnCodec) OR a specific pad token if supported by model.
                       Here we pad with 0 because x=0 is a valid token, but we will pass a `padding_mask` to the transformer.
    intent_aligned: padded with 0.0
    """
    from torch.nn.utils.rnn import pad_sequence
    
    x_list = [b["x_tokens"] for b in batch]
    y_list = [b["y_tokens"] for b in batch]
    i_list = [b["intent_aligned"] for b in batch]
    s_list = [b["situation"] for b in batch]
    
    # Store explicit lengths for masking
    lengths = torch.tensor([x.size(0) for x in x_list], dtype=torch.long)
    
    # Pad [B, T, Q]
    x_padded = pad_sequence(x_list, batch_first=True, padding_value=0)
    y_padded = pad_sequence(y_list, batch_first=True, padding_value=0)
    
    # Pad [B, T, D_int]
    i_padded = pad_sequence(i_list, batch_first=True, padding_value=0.0)
    
    # Stack [B, 32]
    s_stacked = torch.stack(s_list)
    
    # Create padding boolean mask where True = valid, False = padded
    # Standard PyTorch Transformer uses True for ignored/padded tokens, but let's just emit lengths.
    
    return {
        "x_tokens": x_padded,
        "y_tokens": y_padded,
        "intent_aligned": i_padded,
        "situation": s_stacked,
        "lengths": lengths
    }
