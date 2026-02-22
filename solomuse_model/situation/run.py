import os
import json
import csv
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio
from solomuse_model.situation.extract import extract_situation_v1
from solomuse_model.situation.vectorize import vectorize_situation_v1

logger = logging.getLogger(__name__)

def run_situation_extraction(cfg: PipelineConfig, dataset: str, limit: Optional[int] = None, overwrite: bool = False):
    """
    Run situation extraction for a dataset of segments.
    """
    logger.info(f"Running situation extraction for dataset: {dataset}")
    
    # 1. Locate segment manifest
    # Support both global and per-dataset manifest locations
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
    
    # 3. Iterate segments
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Situation {dataset}"):
        segment_id = row.get("segment_id", "unknown")
        # Reconstruct path dynamically from manifest location
        track_id = str(row.get("track_id", ""))
        seg_dir = manifest_path.parent / track_id / segment_id
        x_path = seg_dir / "x.wav"
        
        if not x_path.exists():
            logger.warning(f"Missing x.wav for segment {segment_id} at {seg_dir}")
            continue
            
        situation_file = seg_dir / "situation.npy"
        
        if situation_file.exists() and not overwrite:
            # Load existing if available and not overwriting
            try:
                # We still want to build the manifest row
                # Load meta to get features
                meta_path = seg_dir / "meta.json"
                if meta_path.exists():
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                        if "situation" in meta:
                            output_manifest_rows.append(_make_manifest_row(row, meta["situation"]))
                            continue
            except:
                pass

        try:
            # Load audio
            audio, sr = read_audio(x_path)
            
            # Extract
            features = extract_situation_v1(audio, sr, cfg)
            vector = vectorize_situation_v1(features)
            
            # Save artifacts
            if cfg.situation_save_npy:
                np.save(situation_file, vector)
                
            # Update meta.json
            meta_path = seg_dir / "meta.json"
            meta = {}
            if meta_path.exists():
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            
            # Add compact summary to meta
            meta["situation"] = {
                "version": features["version"],
                "tempo_bpm": features["tempo_bpm"],
                "loudness_lufs": features["loudness_lufs"],
                "rms_mean": features["rms_mean"],
                "vector_len": len(vector)
            }
            
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                
            # Prepare manifest row
            output_manifest_rows.append(_make_manifest_row(row, meta["situation"]))
            
        except Exception as e:
            logger.error(f"Failed to process segment {segment_id}: {e}")
            continue

    # 4. Write manifest_situation.csv
    if output_manifest_rows:
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
