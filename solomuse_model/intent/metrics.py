import torch
import numpy as np
from typing import Dict

def compute_intent_metrics(
    preds: torch.Tensor, 
    targets: torch.Tensor,
    threshold: float | None = None
) -> Dict[str, float]:
    """
    Compute validation metrics for intent predictions.
    
    Args:
        preds: Tensor of shape [B, F, 7]
        targets: Tensor of shape [B, F, 7]
        threshold: Optional threshold for computing frame accuracy.
        
    Returns:
        Dict of metric names to values.
    """
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch in metrics: preds {preds.shape} vs targets {targets.shape}")

    # Aggregated Error
    mse = torch.nn.functional.mse_loss(preds, targets).item()
    mae = torch.nn.functional.l1_loss(preds, targets).item()
    
    # Per-dimension MAE
    mae_per_dim = torch.mean(torch.abs(preds - targets), dim=(0, 1)).cpu().numpy()
    
    metrics = {
        "mse": mse,
        "mae": mae,
        "val_mae_overall": mae
    }
    
    # Try mapping to known semantic names if D=7, else generic dim_N
    dim_names = ["play_prob", "density", "register", "tension", "dynamics", "onset", "boundary"]
    for i, err in enumerate(mae_per_dim):
        name = dim_names[i] if i < len(dim_names) else f"dim_{i}"
        metrics[f"val_mae_{name}"] = float(err)
    
    if threshold is not None:
        with torch.no_grad():
            pred_binary = (preds > threshold).float()
            target_binary = (targets > threshold).float()
            correct = (pred_binary == target_binary).float()
            acc = correct.mean().item()
            metrics[f"frame_accuracy@{threshold}"] = acc
            
    return metrics
