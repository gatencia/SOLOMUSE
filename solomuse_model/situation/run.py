import os
import json
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional, List, Dict, Tuple
import concurrent.futures

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio
from solomuse_model.situation.extract import extract_situation_v1
from solomuse_model.situation.vectorize import vectorize_situation_v1

logger = logging.getLogger(__name__)

def process_segment_situation(args: Tuple[pd.Series, Path, PipelineConfig, bool]) -> Optional[Dict]:
    """
    Process a single segment for situation extraction.
    Returns a manifest row.
    """
    row, manifest_dir, cfg, overwrite = args
    segment_id = row.get("segment_id", "unknown")
    track_id = str(row.get("track_id", ""))
    seg_dir = manifest_dir / track_id / segment_id
    x_path = seg_dir / "x.wav"
    
    if not x_path.exists():
        return None
        
    situation_file = seg_dir / "situation.npy"
    
    if situation_file.exists() and not overwrite:
        try:
            meta_path = seg_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "situation" in meta:
                        return _make_manifest_row(row, meta["situation"])
        except:
            pass

    try:
        audio, sr = read_audio(x_path)
        features = extract_situation_v1(audio, sr, cfg)
        vector = vectorize_situation_v1(features)
        
        if cfg.situation_save_npy:
            np.save(situation_file, vector)
            
        meta_path = seg_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        
        meta["situation"] = {
            "version": features["version"],
            "tempo_bpm": features["tempo_bpm"],
            "loudness_lufs": features["loudness_lufs"],
            "rms_mean": features["rms_mean"],
            "vector_len": len(vector)
        }
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return _make_manifest_row(row, meta["situation"])
        
    except Exception as e:
        logger.error(f"Failed to process segment {segment_id}: {e}")
        return None

def run_situation_extraction(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Run situation extraction for a dataset of segments using multiprocessing.
    """
    logger.info(f"Running situation extraction for dataset: {dataset} with {cfg.num_workers} workers...")
    
    manifest_candidates = [
        Path(cfg.output_root) / "segments" / dataset / "manifest.csv",
        Path(cfg.output_root) / "manifest_segments.csv"
    ]
    
    manifest_path = None
    for p in manifest_candidates:
        if p.exists():
            manifest_path = p
            break
            
    if not manifest_path:
        logger.error(f"Segments manifest not found for {dataset}")
        return

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest {manifest_path}: {e}")
        return

    if limit:
        df = df.head(limit)

    output_manifest_rows = []
    
    tasks = [(row, manifest_path.parent, cfg, overwrite) for _, row in df.iterrows()]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {executor.submit(process_segment_situation, task): task[0].get("segment_id", "unknown") for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Situation {dataset}"):
            try:
                result = future.result()
                if result:
                    output_manifest_rows.append(result)
            except Exception as e:
                segment_id = futures[future]
                logger.error(f"Situation task for {segment_id} failed: {e}")

    if output_manifest_rows:
        # Sort by segment_id or maintain original order? 
        # as_completed loses order. Let's sort to be deterministic.
        output_manifest_rows.sort(key=lambda x: str(x.get("segment_id")))
        
        out_manifest_path = manifest_path.parent / "manifest_situation.csv"
        with open(out_manifest_path, "w", newline="") as f:
            keys = output_manifest_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_manifest_rows)
        logger.info(f"Situation manifest written to {out_manifest_path}")

def _make_manifest_row(seg_row, situation_meta):
    """Combine segment metadata with situation summary."""
    res = {
        "dataset": seg_row.get("dataset"),
        "track_id": seg_row.get("track_id"),
        "segment_id": seg_row.get("segment_id"),
        "situation_version": situation_meta.get("version"),
        "tempo_bpm": situation_meta.get("tempo_bpm"),
        "loudness_lufs": situation_meta.get("loudness_lufs"),
        "rms_mean": situation_meta.get("rms_mean")
    }
    return res
