import torch
import numpy as np
from typing import Dict

def compute_intent_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute validation metrics for intent predictions.
    
    Args:
        preds: Tensor of shape [B, F, 7]
        targets: Tensor of shape [B, F, 7]
        
    Returns:
        Dict of metric names to values.
    """
    # Mean Absolute Error (overall)
    mae = torch.nn.functional.l1_loss(preds, targets).item()
    
    # Per-dimension MAE
    mae_per_dim = torch.mean(torch.abs(preds - targets), dim=(0, 1)).cpu().numpy()
    
    metrics = {
        "val_mae_overall": mae,
        "val_mae_play_prob": float(mae_per_dim[0]),
        "val_mae_density": float(mae_per_dim[1]),
        "val_mae_register": float(mae_per_dim[2]),
        "val_mae_tension": float(mae_per_dim[3]),
        "val_mae_dynamics": float(mae_per_dim[4]),
        "val_mae_onset": float(mae_per_dim[5]),
        "val_mae_boundary": float(mae_per_dim[6])
    }
    
    return metrics
