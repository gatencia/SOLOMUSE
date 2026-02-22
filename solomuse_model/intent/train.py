import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.metrics import compute_intent_metrics
from solomuse_model.intent.dataset import build_intent_dataloaders
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.paths import get_intent_checkpoint_path
import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_train_intent(cfg: PipelineConfig, dataset_name: str):
    """
    Train the baseline Intent Planner on a compiled dataset manifest.
    """
    logger.info(f"Starting intent planner training for {dataset_name}...")
    
    # 1. Datasets & Loaders
    try:
        train_loader, val_loader, test_loader = build_intent_dataloaders(cfg, dataset_name)
    except Exception as e:
        logger.error(f"Failed to initialize datasets: {e}")
        return

    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    train_ds = train_loader.dataset
    val_ds = val_loader.dataset
    test_ds = test_loader.dataset
    
    logger.info(f"Training setup:")
    logger.info(f"  - Train items: {len(train_ds)} ({num_train_batches} batches)")
    logger.info(f"  - Val items: {len(val_ds)} ({num_val_batches} batches)")
    logger.info(f"  - Batch size: {cfg.intent_batch_size}")
    
    if num_train_batches == 0:
        raise RuntimeError(f"Dataset loaded 0 batches. Adjust config `intent_batch_size` or supply more data.")

    # Target Scale Diagnostics
    logger.info("Extracting first batch for sample diagnostics...")
    for diag_X, diag_Y in train_loader:
        logger.info(f"  - Situation [X] shape: {diag_X.shape}, mean: {diag_X.mean().item():.4f}, std: {diag_X.std().item():.4f}, min: {diag_X.min().item():.4f}, max: {diag_X.max().item():.4f}")
        logger.info(f"  - Intent    [Y] shape: {diag_Y.shape}, mean: {diag_Y.mean().item():.4f}, std: {diag_Y.std().item():.4f}, min: {diag_Y.min().item():.4f}, max: {diag_Y.max().item():.4f}")
        
        if not torch.isfinite(diag_X).all():
            logger.warning("! WARNING: NaN/Inf detected in X diagnostics !")
        if not torch.isfinite(diag_Y).all():
            logger.warning("! WARNING: NaN/Inf detected in Y diagnostics !")
        break
        
    if getattr(cfg, "intent_overfit_one_batch", False):
        logger.warning("!!! OVERFIT_ONE_BATCH MODE FLAG ENABLED. Training strictly on the first batch over and over. !!!")
        overfit_batch = next(iter(train_loader))

    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if cfg.intent_model_type.lower() == "gru":
        model = IntentPlannerGRU_V1(
            input_dim=32, # Situation V1
            hidden_dim=cfg.intent_hidden_dim,
            num_layers=cfg.intent_num_layers,
            output_dim=7, # Intent D=7
            dropout=cfg.intent_dropout
        ).to(device)
    else:
        logger.error(f"Unsupported intent_model_type: {cfg.intent_model_type}")
        return
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.intent_lr)
    criterion = nn.MSELoss()
    
    # 4. Tracker Setup
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(cfg.output_root) / "experiments" / "intent" / f"train_{ts}"
    tracker = ExperimentTracker(cfg, run_dir, job_type="train")
    
    # 5. Training Loop
    best_val_loss = float('inf')
    best_ckpt_path = get_intent_checkpoint_path(cfg)
    best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  - Checkpoint save path: {best_ckpt_path}")
    
    training_steps_executed = 0
    
    for epoch in range(cfg.intent_epochs):
        model.train()
        train_loss = 0.0
        
        # tqdm wrapper for epoch
        pbar_loader = [overfit_batch] if getattr(cfg, "intent_overfit_one_batch", False) else train_loader
        pbar = tqdm(pbar_loader, desc=f"Epoch {epoch+1}/{cfg.intent_epochs} [Train]")
        
        for batch_idx, (X, Y) in enumerate(pbar):
            X, Y = X.to(device), Y.to(device)
            
            if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
                raise RuntimeError(f"Input/Target contains NaN/Inf. Epoch {epoch+1}, Batch {batch_idx}")
            
            optimizer.zero_grad()
            preds = model(X)
            
            if not torch.isfinite(preds).all():
                raise RuntimeError(f"Model predictions contain NaN/Inf. Epoch {epoch+1}, Batch {batch_idx}. Preds min/max: {preds.min().item():.4f} / {preds.max().item():.4f}")
            
            loss = criterion(preds, Y)
            
            if not torch.isfinite(loss):
                logger.error(f"Loss is NaN/Inf! Inputs mean/std: {X.mean().item():.4f}/{X.std().item():.4f} | Preds mean/std/min/max: {preds.mean().item():.4f}/{preds.std().item():.4f}/{preds.min().item():.4f}/{preds.max().item():.4f} | Target min/max: {Y.min().item():.4f}/{Y.max().item():.4f}")
                raise RuntimeError(f"Training loss is NaN. Epoch {epoch+1}, Batch {batch_idx}")
            
            loss.backward()
            
            # Gradient clipping
            clip_val = getattr(cfg, "intent_grad_clip", 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"Gradient norm is NaN/Inf after clipping. Epoch {epoch+1}, Batch {batch_idx}")
            
            
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            training_steps_executed += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        if training_steps_executed == 0:
            raise RuntimeError(f"Epoch {epoch+1} completed with 0 optimization steps (zero batches)! Check dataloader setup.")
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targs = []
        
        with torch.no_grad():
            for X, Y in DataLoader(val_ds, batch_size=cfg.intent_batch_size):
                X, Y = X.to(device), Y.to(device)
                preds = model(X)
                loss = criterion(preds, Y)
                val_loss += loss.item() * X.size(0)
                
                val_preds.append(preds)
                val_targs.append(Y)
                
        val_loss /= len(val_ds) if len(val_ds) > 0 else 1.0
        
        eval_metric = float('inf')
        has_val = len(val_preds) > 0
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targs = []
        
        with torch.no_grad():
            for X, Y in DataLoader(val_ds, batch_size=cfg.intent_batch_size):
                X, Y = X.to(device), Y.to(device)
                preds = model(X)
                loss = criterion(preds, Y)
                val_loss += loss.item() * X.size(0)
                
                val_preds.append(preds)
                val_targs.append(Y)
                
        val_loss /= len(val_ds) if len(val_ds) > 0 else 1.0
        
        has_val = len(val_preds) > 0
        
        # Build Epoch Metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/mse": train_loss,
            "learning_rate": optimizer.param_groups[0]['lr']
        }
        
        if has_val:
            all_preds = torch.cat(val_preds, dim=0)
            all_targs = torch.cat(val_targs, dim=0)
            v_metrics = compute_intent_metrics(all_preds, all_targs, threshold=cfg.intent_eval_binary_threshold)
            
            for k, v in v_metrics.items():
                epoch_metrics[f"val/{k}"] = v
                
            eval_metric = epoch_metrics["val/mse"]
            logger.info(f"Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f}, Val MSE: {eval_metric:.4f}, Val MAE: {epoch_metrics['val/mae']:.4f}")
        else:
            eval_metric = train_loss
            logger.info(f"Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f} (No validation split; selecting checkpoint by train loss)")
            
        tracker.log_metrics(epoch_metrics, step=epoch+1)
            
        # Checkpoint
        if eval_metric < best_val_loss:
            best_val_loss = eval_metric
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss if has_val else None,
                'train_loss': train_loss,
                'cfg': cfg.model_dump()
            }, best_ckpt_path)
            metric_str = f"Val MSE {best_val_loss:.4f}" if has_val else f"Train MSE {best_val_loss:.4f}"
            logger.info(f"Saved new best checkpoint with {metric_str}")
            
    # Save final artifacts and close tracker
    summary = {
        "dataset": dataset_name,
        "epochs": cfg.intent_epochs,
        "best_val_loss": best_val_loss,
        "metric_used": "val/mse" if has_val else "train/mse",
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "checkpoint_path": str(best_ckpt_path)
    }
    tracker.log_summary(summary)
    tracker.finish(best_ckpt_path=best_ckpt_path)
    
    logger.info(f"Training completed. Best checkpoint tracked and saved at {best_ckpt_path}")
