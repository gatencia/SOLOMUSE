import json
import logging
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import concurrent.futures

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio, write_audio
from solomuse_data.audio_ops import canonicalize_audio, compute_stats, ensure_channels, resample_audio, loudness_normalize
from solomuse_data.dataset_adapters.base import DatasetAdapter, Track
from solomuse_data.dataset_adapters.slakh import SlakhAdapter
from solomuse_data.dataset_adapters.musdb import MusDBAdapter
from solomuse_data.manifest import write_manifest_pairs

logger = logging.getLogger(__name__)

# Mix consistency thresholds
MSE_THRESHOLD = 1e-4
CORR_THRESHOLD = 0.98

def get_adapter(dataset_name: str, root: str, cfg: PipelineConfig) -> DatasetAdapter:
    root_path = Path(root)
    if dataset_name == "slakh":
        return SlakhAdapter(root_path, cfg)
    elif dataset_name == "musdb":
        return MusDBAdapter(root_path, cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

def align_stems(stems: List[np.ndarray]) -> List[np.ndarray]:
    """Trim all stems to the minimum length."""
    if not stems:
        return []
    min_len = min(s.shape[0] for s in stems)
    return [s[:min_len, :] for s in stems]

def check_mix_consistency(mix_est: np.ndarray, mix_true: np.ndarray) -> Dict:
    """
    Compare estimated mix (sum of stems) with true mixture.
    Both must be aligned and same shape.
    """
    # Ensure same length
    min_len = min(mix_est.shape[0], mix_true.shape[0])
    m_est = mix_est[:min_len]
    m_true = mix_true[:min_len]

    # MSE
    diff = m_est - m_true
    mse = float(np.mean(diff ** 2))

    # Correlation (flatten to 1D)
    # Handle silence?
    if np.sum(m_est**2) < 1e-9 or np.sum(m_true**2) < 1e-9:
        corr = 0.0
    else:
        # np.corrcoef returns matrix
        corr = float(np.corrcoef(m_est.flatten(), m_true.flatten())[0, 1])

    return {
        "mse": mse,
        "corr": corr,
        "max_abs_err": float(np.max(np.abs(diff)))
    }

def process_track(track_info: Tuple[Track, DatasetAdapter, PipelineConfig, Path]) -> Dict:
    """
    Process a single track. Returns result dict (manifest row or error info).
    Must be top-level for pickling.
    """
    track, adapter, cfg, output_dir = track_info
    track_id = track.track_id
    
    try:
        # 1. Resolve Stems
        stems = adapter.list_stems(track)
        solo_paths = adapter.resolve_solo_stems(stems)
        
        if not solo_paths:
            return {"status": "skipped", "track_id": track_id, "reason": "no_solo"}
            
        backing_paths = adapter.resolve_backing_stems(stems, solo_paths)
        if not backing_paths:
             return {"status": "skipped", "track_id": track_id, "reason": "no_backing"}

        # Helper to load and process format (SR/Channels) without loudness norm
        def process_stem(p):
            a, sr = read_audio(str(p))
            a = ensure_channels(a, cfg.canonical_channels)
            if sr != cfg.canonical_sample_rate:
                a = resample_audio(a, sr, cfg.canonical_sample_rate)
            return a
        
        # logger.info(f"  [{track_id}] Loading {len(solo_paths)} solo and {len(backing_paths)} backing stems...")
        
        # Load stems
        solo_audios = [process_stem(p) for p in solo_paths]
        backing_audios = [process_stem(p) for p in backing_paths]
        
        # logger.info(f"  [{track_id}] Aligning and Summing...")
        
        # Align (trim to min length of ALL stems + mix if exists)
        # We align solo and backing first
        all_stems = solo_audios + backing_audios
        
        # Get mix path for alignment and consistency
        mix_path = adapter.get_mix_path(track)
        mix_audio = None
        if mix_path:
            mix_audio = process_stem(mix_path)
            all_stems.append(mix_audio)
            
        all_stems = align_stems(all_stems)
        
        # Unpack
        offset = len(solo_audios)
        solo_aligned = all_stems[:offset]
        backing_aligned = all_stems[offset:offset+len(backing_audios)]
        
        if mix_path:
            mix_aligned = all_stems[-1]
        else:
            mix_aligned = None
            
        # Sum
        y_solo = sum(solo_aligned) if solo_aligned else np.zeros_like(backing_aligned[0]) # Should not accept empty solo
        x_backing = sum(backing_aligned)
        if isinstance(y_solo, int): y_solo = np.zeros_like(x_backing) # Safety if list empty but didn't trigger skip
        
        mix_est = x_backing + y_solo
        
        # Now Normalize
        sr = cfg.canonical_sample_rate
        
        # Calculate Loudness of estimated mix
        # We handle silence case
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(mix_est)
        except ValueError:
            loudness = -np.inf
            
        if loudness > -70: # Not silent
            target = cfg.lufs_target
            gain_db = target - loudness
            gain_lin = 10**(gain_db / 20.0)
        else:
            gain_lin = 1.0 # Verify silence handling
            
        # Apply Gain
        y_solo *= gain_lin
        x_backing *= gain_lin
        mix_est *= gain_lin

        # Enforce Peak Limiting on components
        # If components peak above limit, attenuate them individually?
        # OR attenuate globally to preserve mix balance?
        # Validation checks strict peak limit on X and Y.
        # If we attenuate globally we are safe.
        # Calculating peak of both -> max peak -> required attenuation.
        
        peak_x = float(np.max(np.abs(x_backing)))
        peak_y = float(np.max(np.abs(y_solo)))
        max_peak_linear = max(peak_x, peak_y)
        
        # Convert config limit (dBFS) to linear
        if cfg.peak_limit_dbfs < 0:
            limit_linear = 10**(cfg.peak_limit_dbfs / 20.0)
            
            if max_peak_linear > limit_linear:
                # Need additional attenuation
                # gain = limit / peak
                # Avoid div by zero
                if max_peak_linear > 1e-9:
                    limiter_gain = limit_linear / max_peak_linear
                    y_solo *= limiter_gain
                    x_backing *= limiter_gain
                    mix_est *= limiter_gain # Keep consistency
        
        # Consistency Check (on normalized signals)
        metrics = {"mse": 0.0, "corr": 1.0}
        has_mix = False
        
        if mix_aligned is not None:
            # Normalize true mix to SAME target
            # If true mix is -18 LUFS, and our stems are good, they should match.
            mix_true_norm = loudness_normalize(mix_aligned, sr, cfg.lufs_target, cfg.peak_limit_dbfs)
            metrics = check_mix_consistency(mix_est, mix_true_norm)
            has_mix = True
            
            # if metrics["mse"] > MSE_THRESHOLD:
            #     logger.warning(f"Track {track_id} MSE high: {metrics['mse']:.6f}")
                
        # Compute stats
        stats_x = compute_stats(x_backing, sr)
        stats_y = compute_stats(y_solo, sr)
        
        # Write files
        track_dir = output_dir / track_id
        track_dir.mkdir(parents=True, exist_ok=True)
        
        x_path = track_dir / "x_backing.wav"
        y_path = track_dir / "y_solo.wav"
        
        write_audio(str(x_path), x_backing, sr)
        write_audio(str(y_path), y_solo, sr)
        
        metadata = {
            "dataset": track.dataset, # track object provides dataset name
            "track_id": track_id,
            "license": adapter.get_metadata(track).get("license", "unknown"),
            "solo_stems": [p.name for p in solo_paths],
            "backing_stems": [p.name for p in backing_paths],
            "sr": sr,
            "channels": cfg.canonical_channels,
            "duration_s": x_backing.shape[0] / sr,
            "stats_x": stats_x,
            "stats_y": stats_y,
            "mix_metrics": metrics if has_mix else None
        }
        
        with open(track_dir / "meta.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Prepare paths for manifest
        # We calculate relative path from output_root for portability
        # cfg.output_root is absolute?
        root_path = Path(cfg.output_root).absolute()
        
        try:
            x_rel = x_path.absolute().relative_to(root_path)
            y_rel = y_path.absolute().relative_to(root_path)
        except ValueError:
            # Fallback if not relative (e.g. diff drive)
            x_rel = x_path.name
            y_rel = y_path.name
            
        return {
            "status": "success",
            "dataset": track.dataset,
            "track_id": track_id,
            "x_path": str(x_rel),
            "y_path": str(y_rel),
            "x_path_abs": str(x_path.absolute()),
            "y_path_abs": str(y_path.absolute()),
            "sr": sr,
            "channels": cfg.canonical_channels,
            "duration_s": metadata["duration_s"],
            "license": metadata["license"],
            "has_mix": has_mix,
            "mse": metrics.get("mse", ""),
            "corr": metrics.get("corr", ""),
            "notes": f"mse={metrics.get('mse',0):.1e}" if has_mix and metrics.get('mse',0) > MSE_THRESHOLD else ""
        }
        
    except Exception as e:
        # Check disk space on failure
        total, used, free = shutil.disk_usage(output_dir)
        logger.error(f"Failed to process {track_id}: {e}. Disk Space: {free // (1024**3)}GB free of {total // (1024**3)}GB", exc_info=True)
        return {"status": "error", "track_id": track_id, "reason": f"{str(e)} (Free: {free // (1024**3)}GB)"}

def build_pairs_for_dataset(dataset_name: str, cfg: PipelineConfig) -> Path:
    logger.info(f"Building pairs for {dataset_name} with {cfg.num_workers} workers...")
    
    if dataset_name not in cfg.dataset_roots:
        logger.error(f"Dataset root for {dataset_name} not found in config.")
        return None

    adapter = get_adapter(dataset_name, cfg.dataset_roots[dataset_name], cfg)
    tracks = adapter.list_tracks()
    
    output_dir = Path(cfg.output_root) / "pairs" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Proactive System Check
    logger.info("--- System Diagnostics ---")
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info(f"File Descriptor Limits: Soft={soft}, Hard={hard}")
    except ImportError:
        logger.info("File Descriptor Limits: resource module not available")

    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"Available Memory: {mem.available // (1024**2)} MB / {mem.total // (1024**2)} MB")
    except ImportError:
        logger.info("Available Memory: psutil module not available")

    total, used, free = shutil.disk_usage(output_dir)
    free_gb = free // (1024**3)
    logger.info(f"Free Disk Space: {free_gb} GB")
    logger.info("--------------------------")
    
    # Heuristic: ~200MB per track for Slakh (backing + solo) if full duration
    if dataset_name == "slakh" and len(tracks) > 2000 and free_gb < 400:
        logger.warning(f"CAUTION: Slakh full dataset requires ~400GB of disk space. You only have {free_gb}GB free.")
    
    manifest_rows = []
    skipped_rows = []
    
    # Prepare arguments for multiprocessing
    # Note: Adapter and Config must be picklable. They are simple objects/dataclasses, so should be fine.
    # We pass adapter and cfg to worker.
    # 'track' is a dataclass.
    
    tasks = [(track, adapter, cfg, output_dir) for track in tracks]
    
    # Use ProcessPoolExecutor
    # max_workers = cfg.num_workers
    # If multiprocessing context issues (e.g. CUDA or advanced libs), check context. 
    # But here simple numpy/scipy.
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=cfg.num_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_track, task): task[0].track_id for task in tasks}
        
        # Iterate as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {dataset_name}"):
            track_id = futures[future]
            try:
                result = future.result()
                status = result.get("status")
                
                if status == "success":
                    # Remove status field before appending manifest row
                    row = result.copy()
                    del row["status"]
                    manifest_rows.append(row)
                elif status == "skipped":
                    skipped_rows.append(result)
                    logger.debug(f"Skipped {track_id}: {result.get('reason')}")
                elif status == "error":
                    skipped_rows.append(result)
                    logger.error(f"Error processing {track_id}: {result.get('reason')}")
                    
            except Exception as e:
                logger.error(f"Future for {track_id} raised exception: {e}")
                skipped_rows.append({"track_id": track_id, "status": "error", "reason": str(e)})

    # Write Manifest
    manifest_path = output_dir / "manifest.csv"
    write_manifest_pairs(manifest_rows, str(manifest_path))
    
    logger.info(f"Processed {len(manifest_rows)} tracks. Skipped {len(skipped_rows)}.")
    return manifest_path
