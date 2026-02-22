import torch
import numpy as np
from pathlib import Path
import logging
import datetime
from typing import Dict, Any

from torch.utils.data import DataLoader
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.metrics import compute_intent_metrics
from solomuse_model.intent.dataset import IntentDataset
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.intent.infer import IntentInferencer

logger = logging.getLogger(__name__)

def evaluate_run(
    model: torch.nn.Module, 
    dataloader: DataLoader, 
    cfg: PipelineConfig,
    device: torch.device
) -> Dict[str, float]:
    """
    Core functional block for evaluating an intent model over a Dataloader.
    Yields strictly metrics (MSE, MAE, Frame Accuracy).
    """
    model.eval()
    
    total_metrics = {}
    batch_count = 0
    
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            
            # Catch NaN in inputs (Evaluation should fail loudly too)
            if not torch.isfinite(x).all():
                raise RuntimeError(f"NaN/Inf found in evaluation input batch {batch_idx}.")
            if not torch.isfinite(y).all():
                raise RuntimeError(f"NaN/Inf found in evaluation target batch {batch_idx}.")
                
            preds = model(x)
            
            # Metrics computation
            b_metrics = compute_intent_metrics(preds, y, threshold=cfg.intent_eval_binary_threshold)
            
            # Accumulate safely
            for k, v in b_metrics.items():
                total_metrics[k] = total_metrics.get(k, 0.0) + v
                
            batch_count += 1
            
    if batch_count == 0:
        return {}
        
    for k in total_metrics.keys():
        total_metrics[k] /= batch_count
        
    return total_metrics

def run_eval_intent(cfg: PipelineConfig, dataset_name: str, split: str = "test"):
    """
    CLI endpoint for standalone evaluation of a trained intent checkpoint.
    Saves plots, metrics, and JSON summaries to the experiments folder.
    """
    logger.info(f"Starting standalone Intent evaluation on split: '{split}'")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    import pandas as pd
    
    # Extract split DataFrame (assumes manifest_intent_splits.csv exists)
    targets_dir = Path(cfg.output_root) / "targets" / dataset_name
    split_manifest = targets_dir / "manifest_intent_splits.csv"
    
    if not split_manifest.exists():
        raise FileNotFoundError(f"Split manifest not found at {split_manifest}. Run training or dataset script first.")
        
    df = pd.read_csv(split_manifest)
    
    if split != "all":
        df = df[df["split"] == split]
        
    if len(df) == 0:
        logger.warning(f"No samples found for split '{split}'. Aborting eval.")
        return
        
    eval_ds = IntentDataset(df, cfg)
    eval_loader = DataLoader(eval_ds, batch_size=cfg.intent_batch_size, shuffle=False)
    
    # Load Model using unified inferencer (ensure checkpoint checking logic)
    inferencer = IntentInferencer(cfg)
    model = inferencer.model
    
    # Prepare Tracker
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/processed/experiments/intent") / f"eval_{split}_{ts}"
    tracker = ExperimentTracker(cfg, run_dir, job_type="eval")
    
    logger.info(f"Computing metrics over {len(eval_ds)} segments...")
    metrics = evaluate_run(model, eval_loader, cfg, device)
    
    logger.info("Evaluation metrics computed. Generating Sample Plots...")
    
    # Grab 3 random items to plot
    model.eval()
    plot_count = min(3, len(eval_ds))
    indices = np.random.choice(len(eval_ds), plot_count, replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            x, y = eval_ds[idx]
            x_b = x.unsqueeze(0).to(device)
            p_b = model(x_b)
            
            p = p_b.squeeze(0).cpu().numpy()
            t = y.cpu().numpy()
            
            filename = f"sample_{i}_idx{idx}"
            title = f"Eval Split ({split}) - Segment {idx}"
            tracker.log_prediction_plot(t, p, title, filename)
            
    summary = {
        "dataset": dataset_name,
        "split": split,
        "n_samples": len(eval_ds),
        "metrics": metrics
    }
    
    logger.info("===== EVALUATION SUMMARY =====")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    tracker.log_summary(summary)
    tracker.finish()
    logger.info(f"Evaluation artifacts saved to {run_dir}")
