import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import json

# Fallback for Apple Silicon compilation
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.dataset_tokens import RendererTokenDataset, token_collate_fn
from solomuse_model.renderer.model_v2.transformer_lm import TokenTransformerRenderer
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.paths import get_renderer_checkpoint_path

logger = logging.getLogger(__name__)

def run_train_renderer_v2(cfg: PipelineConfig, dataset_name: str):
    logger.info(f"Starting Renderer V2 (Token LM) training for {dataset_name}...")
    
    # 1. Dataset Manifest
    segments_dir = Path(cfg.output_root) / "segments" / dataset_name
    manifest_path = segments_dir / "manifest_renderer_tokens.csv"
    
    try:
        with open(manifest_path, 'r'):
            pass
    except (FileNotFoundError, PermissionError):
        logger.error(f"Renderer token manifest not found or blocked: {manifest_path}. Run renderer-token-targets first.")
        return
        
    try:
        train_ds = RendererTokenDataset(str(manifest_path), split="train")
        val_ds = RendererTokenDataset(str(manifest_path), split="val")
    except Exception as e:
        logger.error(f"Failed to load RendererTokenDatasets: {e}")
        return
        
    if len(train_ds) == 0:
        logger.warning("Empty training dataset. Aborting.")
        return
        
    logger.info(f"Datasets Built -> Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=cfg.renderer_batch_size, shuffle=True, collate_fn=token_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.renderer_batch_size, shuffle=False, collate_fn=token_collate_fn)

    # 2. Model
    # Due to severe PyTorch MPS bus errors with causal masking in TransformerDecoder,
    # we explicitly fallback to CPU for Apple Silicon users to preserve stability.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available():
        logger.warning("Apple Silicon MPS selected, but bypassed to CPU natively due to fatal PyTorch Causal Masking bus errors.")
    logger.info(f"Using device: {device}")
    
    # Codec hyperparameters (hardcoded for Encodec 24kHz @ 6kbps for now)
    num_codebooks = 4
    vocab_size = 1024
    
    model = TokenTransformerRenderer(
        d_model=cfg.renderer_hidden_dim, 
        nhead=8, 
        num_layers=3, 
        num_codebooks=num_codebooks, 
        vocab_size=vocab_size
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.renderer_lr)
    
    # Ignore index 0 if it's reserved for padding, but EnCodec output is often 0-indexed.
    # Dataset token_collate_fn uses 0 for sequence padding.
    # To truly learn all valid EnCodec tokens (0-1023), 0 cannot be the pad_token logically.
    # However, since we'll predict using sequence lengths natively we won't strictly enforce ignore_index=0
    # unless we offset all tokens by +1, which is a good standard approach:
    # A true padding token offset is required if sequence lengths vary wildly.
    
    # Standard cross entropy for language modeling
    # expects (N, C, d1) where C = classes. We have [B, T, Q, Vocab] -> requires reshape.
    criterion = nn.CrossEntropyLoss()
    
    # 3. Training Loop
    ckpt_dir = Path(cfg.output_root) / "models" / f"renderer_{cfg.renderer_model_version}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = get_renderer_checkpoint_path(cfg)
    logger.info(f"Renderer V2 checkpoint will be saved to: {best_path}")
    
    tracker = ExperimentTracker(cfg, ckpt_dir, job_type="train_renderer_v2")
    best_loss = float('inf')
    
    if getattr(cfg, "renderer_overfit_one_batch", False):
        logger.warning("OVERFIT MODE: Iterating constantly on a single batch")
        single_batch = next(iter(train_loader))
        train_loader_iter = lambda: (single_batch for _ in range(len(train_loader)))
    else:
        train_loader_iter = lambda: train_loader
        
    for epoch in range(cfg.renderer_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader_iter(), total=len(train_loader), desc=f"Ep {epoch+1}/{cfg.renderer_epochs} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            x_tokens = batch["x_tokens"].to(device)
            y_tokens = batch["y_tokens"].to(device)
            intent = batch["intent_aligned"].to(device)
            sit = batch["situation"].to(device)
            
            # Predict targets (teacher forcing). The model masks context automatically.
            optimizer.zero_grad()
            logits = model(x_tokens=x_tokens, intent_aligned=intent, situation=sit, y_tokens=y_tokens)
            
            # Logits shape: [B, T, Q, VocabSize]
            # Targets shape: [B, T, Q]
            # PyTorch MPS backend causes a fatal Bus Error when evaluating 4D spatial Cross-Entropy.
            # Flattening to 2D [B*T*Q, VocabSize] prevents this crash natively.
            logits_flat = logits.view(-1, logits.size(-1))
            y_tokens_flat = y_tokens.view(-1)
            
            loss = criterion(logits_flat, y_tokens_flat)
            
            if not torch.isfinite(loss):
                # Crash export
                crash_pack = {
                    "epoch": epoch, "batch": batch_idx,
                    "x_shape": list(x_tokens.shape),
                    "loss_val": loss.item()
                }
                with open(ckpt_dir / f"crash_ep{epoch}_b{batch_idx}.json", 'w') as f:
                    json.dump(crash_pack, f, indent=2)
                raise RuntimeError(f"NaN Loss crash logged to {ckpt_dir}/crash_ep{epoch}_b{batch_idx}.json")
            
            loss.backward()
            grad_clip = getattr(cfg, "renderer_grad_clip", 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_loss += loss.item() * x_tokens.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if len(val_ds) > 0:
                for batch in DataLoader(val_ds, batch_size=cfg.renderer_batch_size, collate_fn=token_collate_fn):
                    x_tokens = batch["x_tokens"].to(device)
                    y_tokens = batch["y_tokens"].to(device)
                    intent = batch["intent_aligned"].to(device)
                    sit = batch["situation"].to(device)
                    
                    logits = model(x_tokens=x_tokens, intent_aligned=intent, situation=sit, y_tokens=y_tokens)
                    logits_flat = logits.view(-1, logits.size(-1))
                    y_tokens_flat = y_tokens.view(-1)
                    
                    loss = criterion(logits_flat, y_tokens_flat)
                    val_loss += loss.item() * x_tokens.size(0)
                val_loss /= len(val_ds)
            else:
                val_loss = train_loss
                
        logger.info(f"Epoch {epoch+1} - Train CE: {train_loss:.4f}, Val CE: {val_loss:.4f}")
        tracker.log_metrics({"train/ce": train_loss, "val/ce": val_loss}, step=epoch+1)
        
        if val_loss <= best_loss:
            best_loss = val_loss
            logger.info(f"Epoch {epoch+1}: Saving Best Output Checkpoint ({best_loss:.4f})")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch,
                'model_structure': 'TokenTransformerRenderer'
            }, best_path)
            
    logger.info("Renderer V2 Training Complete.")
    tracker.log_summary({"best_val_ce": best_loss})
    tracker.finish(best_ckpt_path=best_path)
