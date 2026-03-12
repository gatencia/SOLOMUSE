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
from solomuse_model.intent.extract_targets import extract_intent_targets_v1
from solomuse_model.intent.vectorize import vectorize_intent_v1

logger = logging.getLogger(__name__)

def process_segment_intent(args: Tuple[pd.Series, Path, PipelineConfig, bool]) -> Optional[Dict]:
    """
    Process a single segment for intent target extraction.
    Returns a manifest row.
    """
    row, manifest_dir, cfg, overwrite = args
    segment_id = row.get("segment_id", "unknown")
    track_id = str(row.get("track_id", ""))
    seg_dir = manifest_dir / track_id / segment_id
    x_path = seg_dir / "x.wav"
    y_path = seg_dir / "y.wav"
    
    if not x_path.exists() or not y_path.exists():
        return None
        
    intent_file = seg_dir / "intent_targets.npy"
    
    if intent_file.exists() and not overwrite:
        try:
            meta_path = seg_dir / "meta.json"
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                    if "intent" in meta:
                        return _make_manifest_row(row, meta["intent"])
        except:
            pass

    try:
        x_audio, sr_x = read_audio(x_path)
        y_audio, sr_y = read_audio(y_path)
        
        # Extract
        seq = extract_intent_targets_v1(x_audio, y_audio, sr_y, {}, cfg)
        vector_matrix = vectorize_intent_v1(seq)
        
        # Save artifacts
        np.save(intent_file, vector_matrix)
            
        # Update meta.json
        meta_path = seg_dir / "meta.json"
        meta = {}
        if meta_path.exists():
            with open(meta_path, "r") as f:
                meta = json.load(f)
        
        meta["intent"] = {
            "version": seq["version"],
            "hz": seq["hz"],
            "frames": vector_matrix.shape[0]
        }
        
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
            
        return _make_manifest_row(row, meta["intent"])
        
    except Exception as e:
        logger.error(f"Failed to process segment {segment_id}: {e}")
        return None

def run_intent_target_build(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Extract and save intent targets for a dataset of segments using multiprocessing.
    """
    logger.info(f"Running intent dataset builder for: {dataset} with {cfg.num_workers} workers...")
    
    # 1. Locate segment manifest
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

    # 2. Load manifest
    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest {manifest_path}: {e}")
        return

    if limit:
        df = df.head(limit)

    output_manifest_rows = []
    
    # 3. Prepare tasks
    tasks = [(row, manifest_path.parent, cfg, overwrite) for _, row in df.iterrows()]
    
    # 4. Use ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        futures = {executor.submit(process_segment_intent, task): task[0].get("segment_id", "unknown") for task in tasks}
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Intent {dataset}"):
            try:
                result = future.result()
                if result:
                    output_manifest_rows.append(result)
            except Exception as e:
                segment_id = futures[future]
                logger.error(f"Intent task for {segment_id} failed: {e}")

    # 5. Write manifest_intent.csv
    if output_manifest_rows:
        # Sort for determinism
        output_manifest_rows.sort(key=lambda x: str(x.get("segment_id")))
        
        out_manifest_path = manifest_path.parent / "manifest_intent.csv"
        with open(out_manifest_path, "w", newline="") as f:
            keys = output_manifest_rows[0].keys()
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(output_manifest_rows)
        logger.info(f"Intent manifest written to {out_manifest_path}")

def _make_manifest_row(seg_row, intent_meta):
    """Combine segment metadata with intent summary."""
    res = {
        "dataset": seg_row.get("dataset"),
        "track_id": seg_row.get("track_id"),
        "segment_id": seg_row.get("segment_id"),
        "intent_version": intent_meta.get("version"),
        "intent_hz": intent_meta.get("hz"),
        "intent_frames": intent_meta.get("frames")
    }
    return res
