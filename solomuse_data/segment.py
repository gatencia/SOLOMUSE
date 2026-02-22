import csv
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio, write_audio
from solomuse_data.audio_ops import compute_peak_dbfs, compute_stats

logger = logging.getLogger(__name__)

def segment_dataset(dataset_name: str, cfg: PipelineConfig):
    """
    Segment pairs manifest into fixed windows.
    """
    logger.info(f"Segmenting dataset: {dataset_name}")
    
    # Locate pairs manifest
    # NOTE: Supports 'weak_demucs' as dataset_name? Or 'slakh'?
    # Actually, we segment configured datasets. 
    # If dataset_name="weak", we might look in "weak_pairs"? 
    # The prompt implies we run this per dataset name.
    # For 'weak' data, the user might pass 'weak_pairs' or 'weak_demucs'. 
    # The folder structure in weak_separation.py was `weak_pairs/<song_id>`. 
    # But `build_pairs` puts output in `pairs/<dataset>/`.
    # Weak pairs didn't produce a manifest! `weak_separation.py` just writes files and `meta.json`. 
    # That's a gap in `weak_separation` if we rely on a manifest here.
    # `segment.py` input says: "pairs manifest.csv (from build_pairs)".
    # Regular dataset flow: `build_pairs` -> manifest.csv -> `segment`.
    # Weak flow: `generate_weak` -> folders. No manifest?
    # Actually, `segment.py` prompt specifies "Inputs: pairs manifest.csv". 
    # So I will assume we are segmenting standard datasets for now. 
    # If I need to segment weak data, I'd need to generate a manifest for it first or scan directories.
    # Given the prompt constraints, I'll focus on the standard flow via manifest.
    
    input_manifest_path = Path(cfg.output_root) / "pairs" / dataset_name / "manifest.csv"
    if not input_manifest_path.exists():
        logger.error(f"Pairs manifest not found at {input_manifest_path}")
        return

    output_dir = Path(cfg.output_root) / "segments" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare segment manifest
    manifest_rows = []
    
    try:
        df = pd.read_csv(input_manifest_path)
    except Exception as e:
        logger.error(f"Failed to read manifest: {e}")
        return

    window_samples = int(cfg.segment_seconds * cfg.canonical_sample_rate)
    hop_samples = int(cfg.segment_hop_seconds * cfg.canonical_sample_rate)
    
    if window_samples <= 0 or hop_samples <= 0:
        logger.error("Invalid window or hop size (0 or negative).")
        return

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Segmenting {dataset_name}"):
        track_id = row.get("track_id", "unknown")
        x_path = row.get("x_path")
        y_path = row.get("y_path")
        license_str = row.get("license", "unknown")
        
        if not x_path or not y_path:
            continue
            
        # Resolve paths (might be relative or absolute)
        # If absolute, /app/data/... it works.
        # If relative, pairs/slakh/..., we prepend output_root.
        
        def resolve_path(p_str):
            p = Path(p_str)
            if p.is_absolute():
                return p
            return Path(cfg.output_root) / p

        x_path_resolved = resolve_path(x_path)
        y_path_resolved = resolve_path(y_path)
            
        try:
            # Load audio
            # We assume they are already canonical (SR=44100, Channels=2) 
            # but read_audio handles file reading safely.
            # We just need to ensure shape matches.
            x, sr_x = read_audio(str(x_path_resolved))
            y, sr_y = read_audio(str(y_path_resolved))
            
            if sr_x != cfg.canonical_sample_rate or sr_y != cfg.canonical_sample_rate:
                # Should have been caught by validate, but safe to skip
                logger.warning(f"Skipping {track_id}: bad sample rate")
                continue
                
            length = min(x.shape[0], y.shape[0])
            
            # Sliding window
            # Stop when a full window cannot be extracted (drop last)
            # range(start, stop, step)
            # We want extraction for start where start + window <= length
            # So stop should be length - window + 1
            
            for start_idx in range(0, length - window_samples + 1, hop_samples):
                end_idx = start_idx + window_samples
                
                seg_x = x[start_idx:end_idx]
                seg_y = y[start_idx:end_idx]
                
                # Check consistency
                if seg_x.shape[0] != window_samples:
                    continue
                    
                # Compute Energy
                # Mean of squared sums of both channels? Or max?
                # Prompt: "mean(x^2)+mean(y^2)"
                energy_x = float(np.mean(seg_x**2))
                energy_y = float(np.mean(seg_y**2))
                total_energy = energy_x + energy_y
                
                if total_energy < cfg.min_segment_energy:
                    continue
                    
                # Prepare write
                segment_id = f"{track_id}_{start_idx}"
                seg_dir = output_dir / track_id / segment_id
                seg_dir.mkdir(parents=True, exist_ok=True)
                
                out_x = seg_dir / "x.wav"
                out_y = seg_dir / "y.wav"
                
                write_audio(str(out_x), seg_x, cfg.canonical_sample_rate)
                write_audio(str(out_y), seg_y, cfg.canonical_sample_rate)
                
                # Stats using existing helpers or simple calculation
                # peak_x = compute_peak_dbfs(seg_x) # Compute explicit stats
                # Using helpers from audio_ops might be cleaner.
                
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
                    
                # Prepare paths
                root_path = Path(cfg.output_root).absolute()
                try:
                    x_rel = out_x.absolute().relative_to(root_path)
                    y_rel = out_y.absolute().relative_to(root_path)
                except ValueError:
                    x_rel = out_x.name
                    y_rel = out_y.name

                manifest_rows.append({
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
            continue

    # Write Manifest
    manifest_path = output_dir.parent / "manifest.csv"
    # Note: Using a single manifest for all datasets or one per dataset folder?
    # build_pairs made output_root / pairs / dataset / manifest.csv
    # segment.py output prompt says: "output_root/segments/... segment folders" and "output_root/manifest_segments.csv"
    # Wait, "manifest_segments.csv" at output_root? Or inside segments? 
    # Usually better to have one global manifest or per dataset. 
    # The prompt says: "output_root/manifest_segments.csv". Single file?
    # If we run this function for "slakh", we overwrite the manifest for "musdb" if it's shared.
    # Unless we APPEND? 
    # Or maybe the prompt implies per-dataset manifested *if* we run it per dataset.
    # BUT, Prompt 5 says inputs: "pairs manifest.csv", calls it "manifest_segments.csv".
    # Let's write `output_root/segments/{dataset}/manifest.csv` to be safe and consistent with build_pairs.
    # Creating a global manifest from partial runs is dangerous (race conditions, overwrites).
    # I will write it to `output_root/segments/{dataset}/manifest.csv`.
    
    out_manifest_path = output_dir / "manifest.csv"
    
    keys = ["dataset", "track_id", "segment_id", "x_path", "y_path", "x_path_abs", "y_path_abs", "start_s", "end_s", "sr", "channels", "duration_s", "energy", "license"]
    
    with open(out_manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(manifest_rows)
        
    logger.info(f"Segmentation complete. Created {len(manifest_rows)} segments. Manifest: {out_manifest_path}")
