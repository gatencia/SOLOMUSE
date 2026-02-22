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

logger = logging.getLogger(__name__)

def run_train_renderer(cfg: PipelineConfig, dataset_name: str):
    """
    Train the baseline Renderer network on the targeted dataset encodings.
    """
    logger.info(f"Starting renderer training for {dataset_name}...")
    
    # 1. Manifest
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
        logger.error(f"Intent/Renderer manifest not found for {dataset_name}")
        return

    # 2. Datasets
    try:
        train_ds = RendererDataset(manifest_path, split="train", val_ratio=0.1)
        val_ds = RendererDataset(manifest_path, split="val", val_ratio=0.1)
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
    best_loss = float('inf')
    ckpt_dir = Path(cfg.output_root) / "models" / "renderer_v1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"
    
    for epoch in range(cfg.renderer_epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{cfg.renderer_epochs} [Train]")
        for bx, bi, bs, by in pbar:
            bx, bi, bs, by = bx.to(device), bi.to(device), bs.to(device), by.to(device)
            
            optimizer.zero_grad()
            preds = model(bx, bi, bs)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * bx.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for bx, bi, bs, by in DataLoader(val_ds, batch_size=cfg.renderer_batch_size, collate_fn=collate_fn):
                bx, bi, bs, by = bx.to(device), bi.to(device), bs.to(device), by.to(device)
                preds = model(bx, bi, bs)
                val_loss += criterion(preds, by).item() * bx.size(0)
                
        val_loss /= max(len(val_ds), 1.0)
        
        logger.info(f"Epoch {epoch+1} - Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss
            }, best_path)
            
    logger.info(f"Renderer training done! Best saved to {best_path}")
