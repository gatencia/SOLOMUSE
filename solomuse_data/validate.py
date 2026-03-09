import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List
from tqdm import tqdm

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio
from solomuse_data.audio_ops import compute_peak_dbfs

logger = logging.getLogger(__name__)

def validate_pairs(cfg: PipelineConfig, dataset: str) -> Dict:
    """
    Validate the pairs manifest.
    Checks:
    - Files exist
    - SR == canonical
    - Channels == canonical
    - X and Y length match
    - Peak levels within limit (+0.2dB tolerance)
    - Duration >= segment_seconds
    """
    logger.info("Validating Pairs...")
    report = {
        "passed": 0,
        "failed": 0,
        "failures": [],
        "stats": {
            "durations": [],
            "peaks_x": [],
            "peaks_y": []
        }
    }

    # Locate manifest(s)
    manifest_paths = []
    p = Path(cfg.output_root) / "pairs" / dataset / "manifest.csv"
    if p.exists():
        manifest_paths.append(p)
    
    if not manifest_paths:
        logger.warning(f"No pair manifests found for {dataset}.")
        return report

    # Load all manifests
    df_list = []
    for p in manifest_paths:
        try:
            df_list.append(pd.read_csv(p))
        except Exception as e:
            logger.warning(f"Failed to read manifest {p} (Sandbox lock): {e}")
            
    if not df_list:
        return report
        
    df = pd.concat(df_list, ignore_index=True)
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating Pairs"):
        track_id = row.get("track_id", "unknown")
        dataset = row.get("dataset", "unknown")
        x_path = row.get("x_path")
        y_path = row.get("y_path")
        
        failure_reasons = []
        
        # 1. Check existence
        if not x_path or not Path(x_path).exists():
            failure_reasons.append(f"X file missing: {x_path}")
        if not y_path or not Path(y_path).exists():
            failure_reasons.append(f"Y file missing: {y_path}")
            
        if failure_reasons:
            report["failed"] += 1
            report["failures"].append({"dataset": dataset, "track_id": track_id, "reasons": failure_reasons})
            continue
            
        try:
            # 2. Read Audio (Rigorous check)
            # io.read_audio ensures float32 and [T, C]
            x_audio, x_sr = read_audio(x_path)
            y_audio, y_sr = read_audio(y_path)
            
            # 3. Check SR
            if x_sr != cfg.canonical_sample_rate:
                failure_reasons.append(f"X sample rate {x_sr} != {cfg.canonical_sample_rate}")
            if y_sr != cfg.canonical_sample_rate:
                failure_reasons.append(f"Y sample rate {y_sr} != {cfg.canonical_sample_rate}")
                
            # 4. Check Channels
            if x_audio.shape[1] != cfg.canonical_channels:
                failure_reasons.append(f"X channels {x_audio.shape[1]} != {cfg.canonical_channels}")
            if y_audio.shape[1] != cfg.canonical_channels:
                failure_reasons.append(f"Y channels {y_audio.shape[1]} != {cfg.canonical_channels}")
                
            # 5. Check Length Alignment
            if x_audio.shape[0] != y_audio.shape[0]:
                failure_reasons.append(f"Length mismatch: X={x_audio.shape[0]}, Y={y_audio.shape[0]}")
                
            # 6. Check Duration
            duration = x_audio.shape[0] / x_sr
            if duration < cfg.segment_seconds:
                failure_reasons.append(f"Duration {duration:.2f}s < {cfg.segment_seconds}s")
            
            report["stats"]["durations"].append(float(duration))
                
            # 7. Check Peak Limits
            # Tolerance 0.2 dB
            limit = cfg.peak_limit_dbfs + 0.2
            peak_x = compute_peak_dbfs(x_audio)
            peak_y = compute_peak_dbfs(y_audio)
            
            report["stats"]["peaks_x"].append(float(peak_x))
            report["stats"]["peaks_y"].append(float(peak_y))
            
            if peak_x > limit:
                failure_reasons.append(f"X peak {peak_x:.2f} > limit {limit}")
            if peak_y > limit:
                failure_reasons.append(f"Y peak {peak_y:.2f} > limit {limit}")
                
        except Exception as e:
            failure_reasons.append(f"Exception during validation: {e}")
            
        if failure_reasons:
            report["failed"] += 1
            report["failures"].append({"dataset": dataset, "track_id": track_id, "reasons": failure_reasons})
        else:
            report["passed"] += 1

    # Summarize stats
    if report["stats"]["durations"]:
        report["summary"] = {
            "duration_min": float(np.min(report["stats"]["durations"])),
            "duration_mean": float(np.mean(report["stats"]["durations"])),
            "duration_max": float(np.max(report["stats"]["durations"])),
            "peak_x_max": float(np.max(report["stats"]["peaks_x"])),
            "peak_y_max": float(np.max(report["stats"]["peaks_y"]))
        }
        
    # Write report
    report_path = Path(cfg.output_root) / "reports" / "validate_pairs.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
        
    # Print Summary
    print("\n" + "="*40)
    print(f"PAIRS VALIDATION REPORT: {report['passed']} PASSED, {report['failed']} FAILED")
    if report['failed'] > 0:
        print(f"Top failure reason: {report['failures'][0]['reasons'][0]}")
    if 'summary' in report:
        print("Stats:")
        for k, v in report['summary'].items():
            print(f"  {k}: {v:.2f}")
    print("="*40 + "\n")
        
    return report

def validate_segments(cfg: PipelineConfig, dataset: str) -> Dict:
    """
    Validate the segments manifest.
    """
    logger.info(f"Validating Segments for {dataset}...")
    report = {
        "passed": 0,
        "failed": 0,
        "failures": []
    }
    
    manifest_path = Path(cfg.output_root) / "segments" / dataset / "manifest.csv"
    if not manifest_path.exists():
        logger.warning(f"Segments manifest not found at {manifest_path}")
        return report

    try:
        df = pd.read_csv(manifest_path)
    except Exception as e:
        logger.error(f"Failed to read segments manifest: {e}")
        return report
        
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating Segments"):
        seg_id = row.get("segment_id", "unknown")
        x_path = row.get("x_path")
        y_path = row.get("y_path")
        
        failure_reasons = []
        
        if not x_path or not Path(x_path).exists():
             failure_reasons.append("X file missing")
             # Skip remaining checks
             report["failed"] += 1
             report["failures"].append({"segment_id": seg_id, "reasons": failure_reasons})
             continue

        if not y_path or not Path(y_path).exists():
             failure_reasons.append("Y file missing")
             report["failed"] += 1
             report["failures"].append({"segment_id": seg_id, "reasons": failure_reasons})
             continue
             
        try:
            x_audio, x_sr = read_audio(x_path)
            y_audio, y_sr = read_audio(y_path)
            
            # Check SR/Channels
            if x_sr != cfg.canonical_sample_rate or y_sr != cfg.canonical_sample_rate:
                failure_reasons.append("Sample rate mismatch")
            if x_audio.shape[1] != cfg.canonical_channels or y_audio.shape[1] != cfg.canonical_channels:
                failure_reasons.append("Channel mismatch")
                
            # Check Alignment
            if x_audio.shape[0] != y_audio.shape[0]:
                failure_reasons.append("Length mismatch")
                
            # Check Duration Logic
            # target samples = sr * segment_seconds
            target_samples = int(cfg.canonical_sample_rate * cfg.segment_seconds)
            # Tolerance ±1 sample
            if abs(x_audio.shape[0] - target_samples) > 1:
                failure_reasons.append(f"Duration mismatch: {x_audio.shape[0]} vs {target_samples}")
                
            # Check Energy (Silence)
            # RMS
            rms_x = np.sqrt(np.mean(x_audio**2))
            rms_y = np.sqrt(np.mean(y_audio**2))
            
            # Logic: If BOTH are silent? Or if just one?
            # Usually we filter segments where TARGET (Y) is silent?
            # Or if context (X) is silent?
            # Prompt: "energy >= cfg.min_segment_energy". 
            # Implies we reject if *total* energy is low? Or max of x/y?
            # Training on silence is useless.
            # Let's check max(rms_x, rms_y) or just reject if Y is silent?
            # Usually we want Y to be present.
            if rms_y < cfg.min_segment_energy and rms_x < cfg.min_segment_energy:
                 failure_reasons.append("Silent segment (both X and Y below threshold)")
            
        except Exception as e:
            failure_reasons.append(f"Exception: {e}")
            
        if failure_reasons:
            report["failed"] += 1
            report["failures"].append({"segment_id": seg_id, "reasons": failure_reasons})
        else:
            report["passed"] += 1
            
    # Write report
    report_path = Path(cfg.output_root) / "reports" / "validate_segments.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print Summary
    print("\n" + "="*40)
    print(f"SEGMENTS VALIDATION REPORT: {report['passed']} PASSED, {report['failed']} FAILED")
    if report['failed'] > 0:
        print(f"Top failure reason: {report['failures'][0]['reasons'][0]}")
    print("="*40 + "\n")
        
    return report

if __name__ == "__main__":
    # If run directly? (Not implementing CLI integration in this file, CLI calls validation)
    pass
