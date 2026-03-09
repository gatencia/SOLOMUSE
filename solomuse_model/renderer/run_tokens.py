import logging
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio
from solomuse_model.renderer.alignment import upsample_intent_to_tokens

logger = logging.getLogger(__name__)

def run_renderer_token_build(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Extract and align all features for Renderer V2 training (targets + conditioning).
    Saves x_tokens, y_tokens, and upsampled intent to segment folder.
    Writes manifest_renderer_tokens.csv.
    """
    logger.info(f"Running renderer V2 token builder for: {dataset}")
    
    # 1. Locate dataset manifest and ensure splits
    from solomuse_model.utils.splits import create_track_grouped_splits
    manifest_path = Path(cfg.output_root) / "segments" / dataset / "manifest.csv"
    split_manifest_path = Path(cfg.output_root) / "segments" / dataset / "manifest_renderer_splits.csv"
    
    try:
        df = create_track_grouped_splits(manifest_path, split_manifest_path, cfg, force_regenerate=False)
    except Exception as e:
        logger.error(f"Failed to load or generate splits for token targets: {e}")
        return

    if limit:
        df = df.head(limit)

    output_manifest_rows = []
    
    # 2. Instantiate Codec
    if cfg.renderer_representation == "encodec":
        try:
            from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
            codec = EnCodecAdapter()
        except Exception as e:
            logger.warning(f"Failed to initialize EnCodecAdapter: {e}")
            codec = None
    else:
        logger.error(f"Renderer tokens require encodec representation, got: {cfg.renderer_representation}")
        return

    # 3. Process
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"RendererTokens {dataset}"):
        segment_id = row.get("segment_id", "unknown")
        track_id = row.get("track_id", "unknown")
        
        # reconstruct path
        seg_dir = manifest_path.parent / track_id / segment_id
        x_path = seg_dir / "x.wav"
        y_path = seg_dir / "y.wav"
        sit_path = seg_dir / "situation.npy"
        
        # Check for intent predictions vs targets
        intent_path = seg_dir / "intent_pred.npy"
        if not intent_path.exists():
            intent_path = seg_dir / "intent_targets.npy"
        
        x_tokens_path = seg_dir / "x_tokens.npy"
        y_tokens_path = seg_dir / "y_tokens.npy"
        
        missing = []
        if not y_path.exists():
            missing.append("y.wav")
        if not x_path.exists():
            missing.append("x.wav")
        if not sit_path.exists():
            missing.append("situation.npy")
        if not intent_path.exists():
            missing.append("intent")
            
        if missing:
            logger.warning(f"Skipping {segment_id} due to missing files: {missing}")
            # Could record skipped segments, but for now we continue
            continue
            
        if not overwrite and x_tokens_path.exists() and y_tokens_path.exists():
            try:
                # Assuming valid if they exist, grab shape from y_tokens
                y_toks = np.load(y_tokens_path)
                row_res = _make_token_manifest_row(row, x_tokens_path, y_tokens_path, y_toks.shape, codec.frame_rate_hz() if codec else 75.0, cfg.intent_hz)
                output_manifest_rows.append(row_res)
                continue
            except Exception as e:
                logger.warning(f"Corrupted cache for {segment_id}, reprocessing.")

        try:
            # Load audio
            x_audio, sr_x = read_audio(str(x_path))
            y_audio, sr_y = read_audio(str(y_path))
            
            # Load conditioning
            intent_mat = np.load(str(intent_path))
            
            if not codec:
                raise RuntimeError("Codec not instantiated.")
                
            # Encode
            x_tokens = codec.encode(x_audio, sr_x)
            y_tokens = codec.encode(y_audio, sr_y)
            
            # Ensure shape match
            if x_tokens.shape[0] != y_tokens.shape[0]:
                min_f = min(x_tokens.shape[0], y_tokens.shape[0])
                x_tokens = x_tokens[:min_f]
                y_tokens = y_tokens[:min_f]

            # Upsample Intent
            token_hz = codec.frame_rate_hz()
            target_frames = y_tokens.shape[0]
            aligned_intent = upsample_intent_to_tokens(intent_mat, cfg.intent_hz, token_hz, target_frames)
            
            # Save aligned intent
            intent_aligned_path = seg_dir / "intent_aligned.npy"
            np.save(str(intent_aligned_path), aligned_intent)
            
            # Save tokens
            np.save(str(x_tokens_path), x_tokens)
            np.save(str(y_tokens_path), y_tokens)
            
            row_res = _make_token_manifest_row(row, x_tokens_path, y_tokens_path, y_tokens.shape, token_hz, cfg.intent_hz)
            output_manifest_rows.append(row_res)
            
        except Exception as e:
            logger.error(f"Failed to process segment {segment_id}: {e}")
            continue

    # 4. Write manifest
    if output_manifest_rows:
        out_manifest_path = manifest_path.parent / "manifest_renderer_tokens.csv"
        with open(out_manifest_path, "w", newline="") as f:
            keys = output_manifest_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_manifest_rows)
        logger.info(f"Renderer Token manifest written to {out_manifest_path}")

def _make_token_manifest_row(seg_row, x_tokens_path, y_tokens_path, token_shape, token_hz, intent_hz):
    split = seg_row.get("split", "unknown")
    
    return {
        "dataset": seg_row.get("dataset"),
        "track_id": seg_row.get("track_id"),
        "segment_id": seg_row.get("segment_id"),
        "split": split,
        "x_tokens_path": str(x_tokens_path),
        "y_tokens_path": str(y_tokens_path),
        "token_frames": token_shape[0] if len(token_shape) > 0 else 0,
        "token_dim": token_shape[1] if len(token_shape) > 1 else 0,
        "token_hz": token_hz,
        "intent_hz": intent_hz
    }
