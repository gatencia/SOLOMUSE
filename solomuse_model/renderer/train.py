import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.dataset import RendererDataset
from solomuse_model.renderer.model_v1 import RendererConv1D_V1
from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.utils.experiment_tracking import ExperimentTracker
from solomuse_model.paths import get_renderer_checkpoint_path

logger = logging.getLogger(__name__)

def run_train_renderer(cfg: PipelineConfig, dataset_name: str):
    """
    Train the baseline Renderer network on the targeted dataset encodings.
    """
    logger.info(f"Starting renderer training for {dataset_name}...")
    
    # 1. Manifest
    from solomuse_model.utils.splits import create_track_grouped_splits
    
    segments_dir = Path(cfg.output_root) / "segments" / dataset_name
    split_manifest_path = segments_dir / "manifest_intent_splits.csv"
    
    # Check if we need to generate splits first
    if not split_manifest_path.exists() or getattr(cfg, "force_regenerate_splits", False):
        logger.info("Shared track-grouped split manifest not found (or forced). Generating...")
        
        # Try finding a source manifest
        source_candidates = [
            segments_dir / "manifest_intent.csv",
            segments_dir / "manifest_renderer.csv"
        ]
        source_manifest = next((p for p in source_candidates if p.exists()), None)
        
        if not source_manifest:
            logger.error(f"Cannot find source intent or renderer manifest to build splits for {dataset_name}")
            return
            
        try:
            create_track_grouped_splits(source_manifest, split_manifest_path, cfg, force_regenerate=True)
        except RuntimeError as e:
            logger.error(str(e))
            return
            
    if not split_manifest_path.exists():
        logger.error(f"Split manifest not found for {dataset_name} at {split_manifest_path}")
        return

    # 2. Datasets
    try:
        train_ds = RendererDataset(str(split_manifest_path), split="train", val_ratio=getattr(cfg, "intent_val_ratio", 0.1))
        val_ds = RendererDataset(str(split_manifest_path), split="val", val_ratio=getattr(cfg, "intent_val_ratio", 0.1))
    except Exception as e:
        logger.error(f"Failed to load Renderer datasets: {e}")
        return

    if len(train_ds) == 0:
        logger.warning("Empty training dataset. Aborting.")
        return

    # Custom collation to encode X on the fly
    codec = WaveChunkCodec(frame_ms=cfg.renderer_frame_ms, hop_ms=cfg.renderer_hop_ms, target_sr=cfg.canonical_sample_rate)
    
    if codec.code_type == "discrete":
        raise NotImplementedError(
            "Discrete token training not yet implemented. "
            "The baseline RendererConv1D_V1 model expects continuous latents/wavechunks and an MSELoss optimization path. "
            "Please follow the docs/renderer_upgrade_plan.md to integrate the autoregressive transformer logic."
        )
    
    def collate_fn(batch):
        xs, ints, sits, ys = [], [], [], []
        # Batch items are tuples: (RendererInputV1, RendererTargetV1)
        for inp, targ in batch:
            # Encode x_audio on the fly
            x_code = codec.encode(inp['x_audio'], inp['sr'])
            
            # intent is [F, 7]
            # situation is [32] -> broadcast to [F, 32]
            f_len = x_code.shape[0]
            
            sit_b = torch.tensor(inp['situation_vector']).unsqueeze(0).repeat(f_len, 1)
            int_t = torch.tensor(inp['intent_sequence'])
            # Mismatches in F can happen due to audio length diffs, truncate to shortest
            min_f = min(f_len, int_t.shape[0], targ['target_codes'].shape[0])
            
            xs.append(torch.tensor(x_code[:min_f]))
            ints.append(int_t[:min_f])
            sits.append(sit_b[:min_f])
            ys.append(torch.tensor(targ['target_codes'][:min_f]))
            
        # Due to windowing variability, stack or pad (stack assuming rigid lengths)
        # We assume dataset enforces rigid segment lengths right now (e.g. 6s), so stack works.
        try:
            bx = torch.stack(xs)
            bi = torch.stack(ints)
            bs = torch.stack(sits)
            by = torch.stack(ys)
            return bx, bi, bs, by
        except RuntimeError:
            # Fallback to pad if segments aren't perfectly aligned
            from torch.nn.utils.rnn import pad_sequence
            bx = pad_sequence(xs, batch_first=True)
            bi = pad_sequence(ints, batch_first=True)
            bs = pad_sequence(sits, batch_first=True)
            by = pad_sequence(ys, batch_first=True)
            return bx, bi, bs, by

    train_loader = DataLoader(train_ds, batch_size=cfg.renderer_batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=cfg.renderer_batch_size, shuffle=False, collate_fn=collate_fn)

    logger.info(f"Loaded {len(train_ds)} items for train split")
    logger.info(f"Loaded {len(val_ds)} items for val split")
    
    if len(train_loader) > 0:
        _bx, _bi, _bs, _by = next(iter(train_loader))
        logger.info(f"Diagnostics - X: shape={_bx.shape}, mean={_bx.mean().item():.4f}, std={_bx.std().item():.4f}")
        logger.info(f"Diagnostics - Target Y: shape={_by.shape}, min={_by.min().item():.4f}, max={_by.max().item():.4f}")

    # 3. Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # dimensions
    c_x = int((cfg.renderer_frame_ms / 1000) * cfg.canonical_sample_rate)
    d_int = 7
    d_sit = 32
    c_y = c_x # Assuming y has same chunk dim
    
    if cfg.renderer_model_type.lower() == "conv1d":
        model = RendererConv1D_V1(
            c_x=c_x, d_int=d_int, d_sit=d_sit, c_y=c_y,
            hidden_dim=cfg.renderer_hidden_dim, 
            num_blocks=3
        ).to(device)
    else:
        logger.error(f"Unsupported renderer: {cfg.renderer_model_type}")
        return
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.renderer_lr)
    criterion = nn.MSELoss()
    
    # 4. Training
    ckpt_dir = Path(cfg.output_root) / "models" / f"renderer_{cfg.renderer_model_version}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = get_renderer_checkpoint_path(cfg)
    logger.info(f"Renderer checkpoint will be saved to: {best_path}")
    
    tracker = ExperimentTracker(cfg, ckpt_dir, job_type="train_renderer")
    best_loss = float('inf')
    
    if getattr(cfg, "renderer_overfit_one_batch", False):
        logger.warning("OVERFIT MODE: Forcing pipeline to run on a single batch iteratively.")
        single_batch = next(iter(train_loader))
        train_loader_iter = lambda: (single_batch for _ in range(len(train_loader)))
    else:
        train_loader_iter = lambda: train_loader
    
    for epoch in range(cfg.renderer_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader_iter(), total=len(train_loader), desc=f"Ep {epoch+1}/{cfg.renderer_epochs} [Train]")
        for batch_idx, (bx, bi, bs, by) in enumerate(pbar):
            if not (torch.isfinite(bx).all() and torch.isfinite(bi).all() and torch.isfinite(bs).all() and torch.isfinite(by).all()):
                raise RuntimeError(f"NaN/Inf found in Renderer inputs/targets. Ep {epoch+1}, Batch {batch_idx}")
            
            bx, bi, bs, by = bx.to(device), bi.to(device), bs.to(device), by.to(device)
            
            optimizer.zero_grad()
            preds = model(bx, bi, bs)
            
            if not torch.isfinite(preds).all():
                raise RuntimeError(f"NaN/Inf found in Renderer predictions. Ep {epoch+1}, Batch {batch_idx}")
                
            loss = criterion(preds, by)
            
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf found in Renderer loss. Ep {epoch+1}, Batch {batch_idx}")
                
            loss.backward()
            
            grad_clip = getattr(cfg, "renderer_grad_clip", 1.0)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(f"NaN/Inf in Iterator gradients before clipping. Ep {epoch+1}, Batch {batch_idx}")
                
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            if len(val_ds) > 0:
                for bx, bi, bs, by in DataLoader(val_ds, batch_size=cfg.renderer_batch_size, collate_fn=collate_fn):
                    bx, bi, bs, by = bx.to(device), bi.to(device), bs.to(device), by.to(device)
                    preds = model(bx, bi, bs)
                    val_loss += criterion(preds, by).item() * bx.size(0)
                val_loss /= len(val_ds)
            else:
                val_loss = train_loss
        
        logger.info(f"Epoch {epoch+1} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
        
        metrics = {"train/mse": train_loss, "val/mse": val_loss, "train/lr": optimizer.param_groups[0]['lr']}
        tracker.log_metrics(metrics, step=epoch+1)
        
        if val_loss <= best_loss: # <= ensures at least 1 save if val_loss is flat
            best_loss = val_loss
            logger.info(f"Epoch {epoch+1}: New best validation MSE ({best_loss:.4f}). Saving checkpoint.")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, best_path)
            
    logger.info("Renderer training completed!")
    tracker.log_summary({"best_val_mse": best_loss})
    tracker.finish(best_ckpt_path=best_path)
