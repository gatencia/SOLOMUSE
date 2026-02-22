import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import os
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.dataset import IntentDataset
from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1
from solomuse_model.intent.metrics import compute_intent_metrics
from tqdm import tqdm

logger = logging.getLogger(__name__)

def run_train_intent(cfg: PipelineConfig, dataset_name: str):
    """
    Train the baseline Intent Planner on a compiled dataset manifest.
    """
    logger.info(f"Starting intent planner training for {dataset_name}...")
    
    # 1. Manifest path
    manifest_candidates = [
        Path(cfg.output_root) / "segments" / dataset_name / "manifest_intent.csv",
        Path(cfg.output_root) / "manifest_intent.csv"
    ]
    
    manifest_path = None
    for p in manifest_candidates:
        if p.exists():
            manifest_path = p
            break
            
    if not manifest_path:
        logger.error(f"Intent manifest not found for {dataset_name}")
        return

    # 2. Datasets & Loaders
    try:
        train_ds = IntentDataset(manifest_path, split="train", val_ratio=cfg.intent_val_split)
        val_ds = IntentDataset(manifest_path, split="val", val_ratio=cfg.intent_val_split)
    except Exception as e:
        logger.error(f"Failed to initialize datasets: {e}")
        return

    if len(train_ds) == 0:
        logger.warning("Empty training dataset. Aborting.")
        return
        
    # We use a custom collate to handle variable sequence lengths if they exist, 
    # but currently segments are fixed window (e.g. 6s), so they should all be F frames.
    # Just in case, PyTorch DataLoader defaults should work if F is constant.
    # Otherwise, pad_sequence is needed. Assuming F is constant for now based on V1 design.
    
    train_loader = DataLoader(train_ds, batch_size=cfg.intent_batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.intent_batch_size, shuffle=False)

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
    
    # 4. Training Loop
    best_val_loss = float('inf')
    ckpt_dir = Path(cfg.output_root) / "models" / "intent_v1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = ckpt_dir / "best.pt"
    
    for epoch in range(cfg.intent_epochs):
        model.train()
        train_loss = 0.0
        
        # tqdm wrapper for epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.intent_epochs} [Train]")
        for X, Y in pbar:
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()
            preds = model(X)
            
            loss = criterion(preds, Y)
            loss.backward()
            
            # Optional gradient clip
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
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
        
        if len(val_preds) > 0:
            all_preds = torch.cat(val_preds, dim=0)
            all_targs = torch.cat(val_targs, dim=0)
            metrics = compute_intent_metrics(all_preds, all_targs)
            
            logger.info(f"Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}, Val MAE: {metrics['val_mae_overall']:.4f}")
        else:
            logger.info(f"Epoch {epoch+1} Summary: Train MSE: {train_loss:.4f} (No validation data)")
            
        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'cfg': cfg.model_dump()
            }, best_ckpt_path)
            logger.info(f"Saved new best checkpoint with Val MSE {best_val_loss:.4f}")
            
    logger.info(f"Training completed. Best checkpoint saved at {best_ckpt_path}")
