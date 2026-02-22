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
from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.prepare_targets import extract_renderer_targets_v1

logger = logging.getLogger(__name__)

def run_renderer_target_build(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Extract and save renderer target codes for a dataset of segments.
    Needs manifest_intent.csv as a guarantee that segment structure exists.
    """
    logger.info(f"Running renderer target builder for: {dataset}")
    
    # 1. Locate dataset manifest (we rely on manifest_intent to ensure earlier steps passed)
    manifest_candidates = [
        Path(cfg.output_root) / "segments" / dataset / "manifest_intent.csv",
        Path(cfg.output_root) / "manifest_intent.csv"
    ]
    
    manifest_path = None
    for p in manifest_candidates:
        if p.exists():
            manifest_path = p
            break
            
    if not manifest_path:
        logger.error(f"Intent manifest not found for {dataset}")
        return

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest {manifest_path}: {e}")
        return

    if limit:
        df = df.head(limit)

    output_manifest_rows = []
    
    # 2. Instantiate Codec
    if cfg.renderer_representation == "wavechunk":
        # Temporary baseline
        codec = WaveChunkCodec(
            frame_ms=cfg.renderer_frame_ms,
            hop_ms=cfg.renderer_hop_ms,
            target_sr=cfg.canonical_sample_rate
        )
    else:
        logger.error(f"Unsupported codec representation: {cfg.renderer_representation}")
        return

    # 3. Process
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Renderer {dataset}"):
        segment_id = row.get("segment_id", "unknown")
        # reconstruct path
        seg_dir = manifest_path.parent / row.get("track_id", "") / segment_id
        y_path = seg_dir / "y.wav"
        
        target_file = seg_dir / "renderer_target.npy"
        meta_path = seg_dir / "meta.json"
        
        if not y_path.exists():
            logger.warning(f"Missing y.wav for {segment_id} at {seg_dir}")
            continue
            
        if target_file.exists() and not overwrite:
            try:
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        if "renderer" in meta:
                            output_manifest_rows.append(_make_manifest_row(row, meta["renderer"]))
                            continue
            except:
                pass

        try:
            # Load audio for encoding
            y_audio, sr = read_audio(y_path)
            
            # Encode
            codes = extract_renderer_targets_v1(y_audio, sr, cfg, codec)
            
            # Save artifacts
            np.save(target_file, codes)
                
            # Update meta.json
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            
            meta["renderer"] = {
                "version": cfg.renderer_target_version,
                "representation": cfg.renderer_representation,
                "codec_hz": codec.frame_rate_hz(),
                "frames": codes.shape[0],
                "dim": codes.shape[1]
            }
            
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                
            output_manifest_rows.append(_make_manifest_row(row, meta["renderer"]))
            
        except Exception as e:
            logger.error(f"Failed to process segment {segment_id}: {e}")
            continue

    # 4. Write manifest
    if output_manifest_rows:
        out_manifest_path = manifest_path.parent / "manifest_renderer.csv"
        with open(out_manifest_path, "w", newline="") as f:
            keys = output_manifest_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_manifest_rows)
        logger.info(f"Renderer manifest written to {out_manifest_path}")

def _make_manifest_row(seg_row, ren_meta):
    res = {
        "dataset": seg_row.get("dataset"),
        "track_id": seg_row.get("track_id"),
        "segment_id": seg_row.get("segment_id"),
        "renderer_version": ren_meta.get("version"),
        "renderer_repr": ren_meta.get("representation"),
        "renderer_frames": ren_meta.get("frames"),
        "renderer_dim": ren_meta.get("dim")
    }
    return res
