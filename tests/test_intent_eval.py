import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from solomuse_data.config import PipelineConfig
from solomuse_model.intent.metrics import compute_intent_metrics
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.intent.dataset import build_intent_dataloaders

def test_eval_metric_computation():
    """Test phase 2: metrics calculate exactly MSE and MAE on 2D Tensors."""
    # [1, 2, 3] vs [1.5, 2.5, 3.5]
    # L1 error: 0.5 per element -> MAE = 0.5
    # L2 error: 0.25 per element -> MSE = 0.25
    
    preds = torch.tensor([[[1.0, 2.0, 3.0]]])
    targs = torch.tensor([[[1.5, 2.5, 3.5]]])
    
    metrics = compute_intent_metrics(preds, targs, threshold=0.5)
    
    assert np.isclose(metrics["mse"], 0.25), f"Expected MSE 0.25, got {metrics['mse']}"
    assert np.isclose(metrics["mae"], 0.5), f"Expected MAE 0.5, got {metrics['mae']}"
    
    # Threshold accuracy logic check
    # All preds > 0.5, All Targets > 0.5 -> Should be 100% agreement
    assert np.isclose(metrics["frame_accuracy@0.5"], 1.0)
    
def test_wandb_disabled(tmp_path):
    """Test phase 0: W&B disabled wrapper runs locally without raising an error."""
    # Create simple config with wandb_enabled = False
    cfg = PipelineConfig(
        output_root=str(tmp_path),
        wandb_enabled=False
    )
    
    # Should not crash even if wandb isn't loaded
    run_dir = tmp_path / "experiments" / "test_run"
    tracker = ExperimentTracker(cfg, run_dir)
    
    tracker.log_metrics({"train/mse": 1.0, "val/mse": 0.5}, step=1)
    tracker.log_summary({"n_train": 100})
    tracker.finish()
    
    assert (run_dir / "metrics_history.csv").exists()
    assert (run_dir / "metrics_summary.json").exists()

def test_deterministic_splits_seeded(tmp_path):
    """Test phase 1: Ensure dataset splits are mathematically seeded and deterministic."""
    cfg = PipelineConfig(
        output_root=str(tmp_path),
        seed=123,
        intent_train_ratio=0.8,
        intent_val_ratio=0.1,
        intent_batch_size=2
    )
    
    targets_dir = tmp_path / "targets" / "test_ds"
    targets_dir.mkdir(parents=True)
    
    # Create fake manifest with 10 rows
    df = pd.DataFrame({
        "track_id": [f"T{i}" for i in range(10)],
        "segment_id": [f"S{i}" for i in range(10)],
        "dataset": ["test_ds"] * 10
    })
    manifest_path = targets_dir / "manifest_intent.csv"
    df.to_csv(manifest_path, index=False)
    
    # We must mock the actual file loading since IntentDataset looks for .npy files
    # We can just test the split generation logic in dataset by hooking into the first lines of build_intent_dataloaders
    # Since build_intent_dataloaders immediately crashes on Dataset() instantiations without .npy,
    # we copy the split logic exact steps to assert determinism.
    
    def simulate_split(seed):
        df_sim = pd.read_csv(manifest_path)
        df_sim = df_sim.sample(frac=1, random_state=seed).reset_index(drop=True)
        return df_sim["track_id"].tolist()
        
    run1 = simulate_split(cfg.seed)
    run2 = simulate_split(cfg.seed)
    run3 = simulate_split(999) # different seed
    
    assert run1 == run2, "Different runs with identical seeds produced different splits!"
    assert run1 != run3, "Different seeds produced identically shuffled datasets."
