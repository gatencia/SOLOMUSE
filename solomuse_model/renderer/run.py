import logging
import json
import csv
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
import concurrent.futures

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio
from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.prepare_targets import extract_renderer_targets_v1

logger = logging.getLogger(__name__)

def process_renderer_target(args: Tuple[pd.Series, Path, PipelineConfig, bool, object]) -> Optional[Dict]:
    """
    Process a single segment for renderer target extraction.
    Returns a manifest row.
    """
    row, manifest_dir, cfg, overwrite, codec = args
    segment_id = row.get("segment_id", "unknown")
    track_id = str(row.get("track_id", ""))
    seg_dir = manifest_dir / track_id / segment_id
    y_path = seg_dir / "y.wav"
    
    target_file = seg_dir / "renderer_target.npy"
    meta_path = seg_dir / "meta.json"
    
    if not y_path.exists():
        return None
        
    if target_file.exists() and not overwrite:
        try:
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "renderer" in meta:
                        return _make_manifest_row(row, meta["renderer"])
        except:
            pass

    try:
        y_audio, sr = read_audio(y_path)
        codes = extract_renderer_targets_v1(y_audio, sr, cfg, codec)
        np.save(target_file, codes)
            
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
            
        return _make_manifest_row(row, meta["renderer"])
        
    except Exception as e:
        logger.error(f"Failed to process segment {segment_id}: {e}")
        return None

def run_renderer_target_build(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Extract and save renderer target codes for a dataset of segments using multiprocessing.
    """
    logger.info(f"Running renderer target builder for: {dataset} with {cfg.num_workers} workers...")
    
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

    # 2. Instantiate Codec (Codec Must be picklable or instantiated in worker)
    # WaveChunkCodec is simple. EnCodecAdapter might be heavy but picklable.
    if cfg.renderer_representation == "wavechunk":
        codec = WaveChunkCodec(
            frame_ms=cfg.renderer_frame_ms,
            hop_ms=cfg.renderer_hop_ms,
            target_sr=cfg.canonical_sample_rate
        )
    elif cfg.renderer_representation == "encodec":
        from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
        codec = EnCodecAdapter()
    else:
        logger.error(f"Unsupported codec representation: {cfg.renderer_representation}")
        return

    output_manifest_rows = []
    tasks = [(row, manifest_path.parent, cfg, overwrite, codec) for _, row in df.iterrows()]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {executor.submit(process_renderer_target, task): task[0].get("segment_id", "unknown") for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Renderer {dataset}"):
            try:
                result = future.result()
                if result:
                    output_manifest_rows.append(result)
            except Exception as e:
                segment_id = futures[future]
                logger.error(f"Renderer task for {segment_id} failed: {e}")

    if output_manifest_rows:
        output_manifest_rows.sort(key=lambda x: str(x.get("segment_id")))
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
