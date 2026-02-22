import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
import json
import csv
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.metrics import compute_intent_metrics
from solomuse_model.intent.dataset import build_intent_dataloaders
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.paths import get_intent_checkpoint_path
from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1
import datetime
from tqdm import tqdm

logger = logging.getLogger(__name__)
import math

def sanitize_for_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj): return "NaN"
        if math.isinf(obj): return "Infinity" if obj > 0 else "-Infinity"
        return obj
    elif isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    return obj

def tensor_stats_dict(tensor: torch.Tensor, name: str) -> dict:
    """Helper to dump summary statistics for any tensor payload."""
    if not isinstance(tensor, torch.Tensor):
        return {"name": name, "type": str(type(tensor)), "error": "Not a tensor"}
        
    stats = {
        "name": name,
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "is_finite": torch.isfinite(tensor).all().item(),
        "nan_count": torch.isnan(tensor).sum().item(),
        "inf_count": torch.isinf(tensor).sum().item(),
    }
    
    if stats["is_finite"]:
        stats.update({
            "min": tensor.min().item(),
            "max": tensor.max().item(),
            "mean": tensor.mean().item(),
            "std": tensor.std().item()
        })
    else:
        # If not finite, we can still get min/max using robust functions 
        # or just skip to avoid throwing more errors.
        valid_mask = torch.isfinite(tensor)
        if valid_mask.any():
            valid_tensor = tensor[valid_mask]
            stats.update({
                "min": valid_tensor.min().item(),
                "max": valid_tensor.max().item()
            })
            
    return stats

def collect_grad_stats(model: nn.Module) -> dict:
    """Calculates statistics for gradients across all parameters."""
    all_finite = True
    first_bad_param = None
    bad_param_count = 0
    param_count = 0
    none_grad_count = 0
    per_param_stats = {}
    
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is None:
            none_grad_count += 1
            continue
            
        is_fin = torch.isfinite(param.grad).all().item()
        
        stat = {
            "name": name,
            "shape": list(param.grad.shape),
            "is_finite": is_fin,
            "nan_count": torch.isnan(param.grad).sum().item(),
            "inf_count": torch.isinf(param.grad).sum().item()
        }
        
        if is_fin:
            stat.update({
                "min": param.grad.min().item(),
                "max": param.grad.max().item(),
                "mean": param.grad.mean().item(),
                "std": param.grad.std().item(),
                "abs_max": param.grad.abs().max().item()
            })
        else:
            all_finite = False
            bad_param_count += 1
            if first_bad_param is None:
                first_bad_param = name
            
            valid_mask = torch.isfinite(param.grad)
            if valid_mask.any():
                valid_param = param.grad[valid_mask]
                stat.update({
                    "min": valid_param.min().item(),
                    "max": valid_param.max().item(),
                    "abs_max": valid_param.abs().max().item()
                })
                
        per_param_stats[name] = stat
        
    # Build compact return payload
    # Top 10 by abs_max, plus any bad ones
    finite_stats = [v for v in per_param_stats.values() if v["is_finite"]]
    finite_stats.sort(key=lambda x: x.get("abs_max", 0.0), reverse=True)
    top_params = finite_stats[:10]
    
    bad_stats = [v for v in per_param_stats.values() if not v["is_finite"]]
    
    saved_params = {s["name"]: s for s in (top_params + bad_stats)}
    
    return {
        "all_finite_pre_clip": all_finite,
        "first_bad_param": first_bad_param,
        "bad_param_count": bad_param_count,
        "param_count": param_count,
        "none_grad_count": none_grad_count,
        "params": saved_params
    }


def run_train_intent(cfg: PipelineConfig, dataset_name: str):
    """
    Train the baseline Intent Planner on a compiled dataset manifest.
    """
    logger.info(f"Starting intent planner training for {dataset_name}...")
    
    # Debug skip settings
    skip_bad_batches = getattr(cfg, "intent_skip_bad_batches", False)
    max_bad_batches = getattr(cfg, "intent_max_bad_batches_per_epoch", 0)
    save_crash_batches = getattr(cfg, "intent_debug_save_crash_batches", True)
    
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
    for diag_batch in train_loader:
        diag_X = diag_batch["situation"]
        diag_Y = diag_batch["intent"]
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
    device_str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if getattr(cfg, "intent_force_cpu_debug", False):
        device_str = 'cpu'
        logger.warning("intent_force_cpu_debug is enabled. Overriding device to CPU.")
        
    device = torch.device(device_str)
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
    
    weight_decay = getattr(cfg, "intent_weight_decay", 1e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.intent_lr, weight_decay=weight_decay)
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
        bad_batches_this_epoch = 0
        
        # tqdm wrapper for epoch
        pbar_loader = [overfit_batch] if getattr(cfg, "intent_overfit_one_batch", False) else train_loader
        pbar = tqdm(pbar_loader, desc=f"Epoch {epoch+1}/{cfg.intent_epochs} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            X = batch["situation"].to(device)
            Y = batch["intent"].to(device)
            segment_ids = batch["segment_id"]
            track_ids = batch["track_id"]
            
            crash_reason = None
            stage = "Pre-Forward"
            loss = None
            preds = None
            grad_info = None
            
            try:
                if not torch.isfinite(X).all() or not torch.isfinite(Y).all():
                    crash_reason = "Input/Target contains NaN/Inf."
                    raise ValueError(crash_reason)
                
                optimizer.zero_grad()
                
                stage = "Forward"
                preds = model(X)
                
                if not torch.isfinite(preds).all():
                    crash_reason = "Model predictions contain NaN/Inf."
                    raise ValueError(crash_reason)
                
                stage = "Loss"
                loss = criterion(preds, Y)
                
                if not torch.isfinite(loss):
                    crash_reason = "Training loss is NaN."
                    raise ValueError(crash_reason)
                
                stage = "Backward"
                loss.backward()
                
                # Intercept non-finite gradients BEFORE clipping
                stage = "Backward-Gradients"
                grad_info = collect_grad_stats(model)
                if not grad_info["all_finite_pre_clip"]:
                    crash_reason = f"Parameter gradients became NaN/Inf. First bad param: {grad_info['first_bad_param']}"
                    raise ValueError(crash_reason)
                
                stage = "Gradient-Clip"
                clip_val = getattr(cfg, "intent_grad_clip", 1.0)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
                grad_info["clip_grad_norm_return"] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                
                if not torch.isfinite(grad_norm):
                    crash_reason = "clip_grad_norm_ produced non-finite norm despite finite pre-clip grads."
                    raise ValueError(crash_reason)
                
                stage = "Optimizer-Step"
                skip_step = getattr(cfg, "intent_skip_optimizer_step_on_large_grad_norm", False)
                thresh_val = getattr(cfg, "intent_large_grad_norm_threshold", 100.0)
                
                if skip_step and grad_norm > thresh_val:
                    logger.warning(f"⏩ Skipping optimizer step at Epoch {epoch+1} Batch {batch_idx}: Grad norm ({grad_norm:.2f}) > {thresh_val}")
                    optimizer.zero_grad() # clear explosive gradients
                else:
                    optimizer.step()
                
                # Check model params finite
                for name, param in model.named_parameters():
                    if not torch.isfinite(param).all():
                        crash_reason = f"Parameter {name} became NaN/Inf."
                        raise ValueError(crash_reason)
                        
            except Exception as e:
                # Capture Diagnostic Telemetry
                logger.error(f"Batch Crash at Epoch {epoch+1}, Batch {batch_idx}. Stage: {stage}. Reason: {str(e)}")
                
                if save_crash_batches:
                    debug_dir = Path(cfg.output_root) / "experiments" / "intent" / "debug_crashes"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    
                    payload = {
                        "epoch": epoch + 1,
                        "batch_idx": batch_idx,
                        "stage": stage,
                        "reason": str(e),
                        "backend": str(device),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                        "grad_clip": getattr(cfg, "intent_grad_clip", 1.0),
                        "weight_decay": getattr(cfg, "intent_weight_decay", 1e-5),
                        "segment_ids": segment_ids,
                        "track_ids": track_ids,
                        "gradients": grad_info,
                        "stats": {
                            "situation": tensor_stats_dict(X, "situation"),
                            "intent_target": tensor_stats_dict(Y, "intent_target"),
                            "predictions": tensor_stats_dict(preds, "predictions") if preds is not None else None,
                            "loss": tensor_stats_dict(loss, "loss") if loss is not None else None
                        }
                    }
                    
                    crash_file = debug_dir / f"crash_ep{epoch+1}_b{batch_idx}.json"
                    with open(crash_file, "w") as f:
                        json.dump(sanitize_for_json(payload), f, indent=2)
                        
                    csv_file = debug_dir / "bad_batches.csv"
                    write_header = not csv_file.exists()
                    with open(csv_file, "a", newline="") as f:
                        writer = csv.writer(f)
                        if write_header:
                            writer.writerow(["timestamp", "epoch", "batch_idx", "stage", "reason", "segment_ids"])
                        writer.writerow([
                            datetime.datetime.now().isoformat(),
                            epoch + 1, batch_idx, stage, str(e),
                            "|".join(segment_ids)
                        ])
                        
                    logger.error(f"Saved crash diagnostics to {crash_file}")
                
                # Check skip bounds
                bad_batches_this_epoch += 1
                if skip_bad_batches and (max_bad_batches == 0 or bad_batches_this_epoch <= max_bad_batches):
                    logger.warning(f"⏩ SKIP BAD BATCHES ENABLED. Skipping failing batch and carrying on.")
                    continue
                else:
                    logger.error("❌ Fatal boundary reached. Aborting training.")
                    raise RuntimeError(f"Training crashed at Epoch {epoch+1} Batch {batch_idx}: {str(e)}") from e
            
            train_loss += loss.item() * X.size(0)
            training_steps_executed += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        if bad_batches_this_epoch > 0 and skip_bad_batches:
            logger.warning(f"Epoch {epoch+1} stats: skipped {bad_batches_this_epoch} bad batches.")
            
        if training_steps_executed == 0:
            raise RuntimeError(f"Epoch {epoch+1} completed with 0 optimization steps (zero batches)! Check dataloader setup.")
            
        train_loss /= len(train_ds)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targs = []
        
        with torch.no_grad():
            for batch in DataLoader(val_ds, batch_size=cfg.intent_batch_size):
                X = batch["situation"].to(device)
                Y = batch["intent"].to(device)
                preds = model(X)
                loss = criterion(preds, Y)
                val_loss += loss.item() * X.size(0)
                
                val_preds.append(preds)
                val_targs.append(Y)
                
        val_loss /= len(val_ds) if len(val_ds) > 0 else 1.0
        
        eval_metric = float('inf')
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
