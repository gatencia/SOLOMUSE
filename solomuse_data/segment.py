import csv
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from typing import List, Dict, Tuple

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio, write_audio
from solomuse_data.audio_ops import compute_peak_dbfs, compute_stats

logger = logging.getLogger(__name__)

def process_track_segmentation(args: Tuple[pd.Series, PipelineConfig, str, Path]) -> List[Dict]:
    """
    Process segmentation for a single track (row in manifest).
    Returns a list of segment manifest rows.
    """
    row, cfg, dataset_name, output_dir = args
    track_id = row.get("track_id", "unknown")
    x_path = row.get("x_path")
    y_path = row.get("y_path")
    license_str = row.get("license", "unknown")
    
    if not x_path or not y_path:
        return []

    def resolve_path(p_str):
        p = Path(p_str)
        if p.is_absolute():
            return p
        return Path(cfg.output_root) / p

    x_path_resolved = resolve_path(x_path)
    y_path_resolved = resolve_path(y_path)
    
    track_segments = []
    
    try:
        x, sr_x = read_audio(str(x_path_resolved))
        y, sr_y = read_audio(str(y_path_resolved))
        
        if sr_x != cfg.canonical_sample_rate or sr_y != cfg.canonical_sample_rate:
            logger.warning(f"Skipping {track_id}: bad sample rate")
            return []
            
        length = min(x.shape[0], y.shape[0])
        window_samples = int(cfg.segment_seconds * cfg.canonical_sample_rate)
        hop_samples = int(cfg.segment_hop_seconds * cfg.canonical_sample_rate)
        
        for start_idx in range(0, length - window_samples + 1, hop_samples):
            end_idx = start_idx + window_samples
            
            seg_x = x[start_idx:end_idx]
            seg_y = y[start_idx:end_idx]
            
            if seg_x.shape[0] != window_samples:
                continue
                
            energy_x = float(np.mean(seg_x**2))
            energy_y = float(np.mean(seg_y**2))
            total_energy = energy_x + energy_y
            
            if total_energy < cfg.min_segment_energy:
                continue
                
            segment_id = f"{track_id}_{start_idx}"
            seg_dir = output_dir / track_id / segment_id
            seg_dir.mkdir(parents=True, exist_ok=True)
            
            out_x = seg_dir / "x.wav"
            out_y = seg_dir / "y.wav"
            
            write_audio(str(out_x), seg_x, cfg.canonical_sample_rate)
            write_audio(str(out_y), seg_y, cfg.canonical_sample_rate)
            
            peak_x = float(compute_peak_dbfs(seg_x))
            peak_y = float(compute_peak_dbfs(seg_y))
            
            meta = {
                "dataset": dataset_name,
                "track_id": track_id,
                "segment_id": segment_id,
                "start_s": float(start_idx / sr_x),
                "end_s": float(end_idx / sr_x),
                "sr": sr_x,
                "channels": seg_x.shape[1],
                "license": license_str,
                "energy": total_energy,
                "rms_x": float(np.sqrt(energy_x)),
                "rms_y": float(np.sqrt(energy_y)),
                "peak_x_dbfs": peak_x,
                "peak_y_dbfs": peak_y
            }
            
            with open(seg_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
                
            root_path = Path(cfg.output_root).absolute()
            try:
                x_rel = out_x.absolute().relative_to(root_path)
                y_rel = out_y.absolute().relative_to(root_path)
            except ValueError:
                x_rel = out_x.name
                y_rel = out_y.name

            track_segments.append({
                "dataset": dataset_name,
                "track_id": track_id,
                "segment_id": segment_id,
                "x_path": str(x_rel),
                "y_path": str(y_rel),
                "x_path_abs": str(out_x.absolute()),
                "y_path_abs": str(out_y.absolute()),
                "start_s": meta["start_s"],
                "end_s": meta["end_s"],
                "sr": meta["sr"],
                "channels": meta["channels"],
                "duration_s": cfg.segment_seconds,
                "energy": meta["energy"],
                "license": meta["license"]
            })
            
    except Exception as e:
        logger.error(f"Error segmenting {track_id}: {e}")
        
    return track_segments

def segment_dataset(dataset_name: str, cfg: PipelineConfig):
    """
    Segment pairs manifest into fixed windows using multiprocessing.
    """
    logger.info(f"Segmenting dataset: {dataset_name} with {cfg.num_workers} workers...")
    
    input_manifest_path = Path(cfg.output_root) / "pairs" / dataset_name / "manifest.csv"
    if not input_manifest_path.exists():
        logger.error(f"Pairs manifest not found at {input_manifest_path}")
        return

    output_dir = Path(cfg.output_root) / "segments" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pd.read_csv(input_manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        return

    manifest_rows = []
    
    # Prepare arguments for multiprocessing
    tasks = [(row, cfg, dataset_name, output_dir) for _, row in df.iterrows()]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {executor.submit(process_track_segmentation, task): task[0].get("track_id", "unknown") for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Segmenting {dataset_name}"):
            try:
                result = future.result()
                manifest_rows.extend(result)
            except Exception as e:
                track_id = futures[future]
                logger.error(f"Segmentation task for {track_id} failed: {e}")

    # Write Manifest
    out_manifest_path = output_dir / "manifest.csv"
    keys = ["dataset", "track_id", "segment_id", "x_path", "y_path", "x_path_abs", "y_path_abs", "start_s", "end_s", "sr", "channels", "duration_s", "energy", "license"]
    
    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    logger.info(f"Segmentation complete. Created {len(manifest_rows)} segments. Manifest: {out_manifest_path}")
