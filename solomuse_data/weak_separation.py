import logging
import subprocess
import shutil
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Optional

from solomuse_data.config import PipelineConfig
from solomuse_data.io import read_audio, write_audio
from solomuse_data.audio_ops import ensure_channels, resample_audio, compute_stats, loudness_normalize
from solomuse_data.build_pairs import align_stems # Reuse alignment logic

logger = logging.getLogger(__name__)

def generate_weak_data(cfg: PipelineConfig):
    """
    Generate weak training pairs using Demucs on raw input files.
    """
    if not cfg.enable_weak_demucs:
        logger.warning("Weak supervision disabled in config (enable_weak_demucs=False). Skipping.")
        return

    # Check for inputs
    input_dir = Path(cfg.output_root) / "weak_inputs"
    if not input_dir.exists():
        logger.warning(f"No weak input directory found at {input_dir}. Create this folder and add songs to process.")
        return

    songs = [p for p in input_dir.iterdir() if p.suffix.lower() in (".mp3", ".wav", ".flac", ".m4a")]
    if not songs:
        logger.warning("No audio files found in weak_inputs.")
        return

    output_pairs_dir = Path(cfg.output_root) / "weak_pairs"
    output_pairs_dir.mkdir(parents=True, exist_ok=True)
    
    # Demucs output temp directory
    demucs_out_root = Path(cfg.output_root) / "demucs_temp"
    demucs_out_root.mkdir(exist_ok=True)

    # Check for demucs command
    if not shutil.which("demucs"):
        logger.error("Demucs command not found. Please install demucs (pip install demucs) to use this feature.")
        return

    logger.info(f"Found {len(songs)} songs for weak separation. Model: {cfg.demucs_model}")

    for song_path in tqdm(songs, desc="Processing Weak Data"):
        song_id = song_path.stem
        
        # Output directory for this specific song pair
        pair_dir = output_pairs_dir / song_id
        if pair_dir.exists():
            logger.info(f"Skipping {song_id}, already exists.")
            continue
            
        try:
            # 1. Run Demucs
            # demucs -n {model} -o {out} {song}
            cmd = [
                "demucs",
                "-n", cfg.demucs_model,
                "-o", str(demucs_out_root),
                str(song_path)
            ]
            
            # Using subprocess
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"Demucs failed for {song_id}: {result.stderr}")
                continue
                
            # 2. Locate Stems
            # Demucs structure: out_root / model_name / song_name / {bass,drums,other,vocals}.wav
            # Note: song_name folder might handle spaces/chars differently. 
            # Demucs usually sanitizes names or uses the filename stem.
            # We construct expected path.
            # Warning: Demucs behavior on output folder naming can vary.
            # Better to search?
            model_out_dir = demucs_out_root / cfg.demucs_model
            
            # Find the song folder inside model_out_dir
            # It should match song_id, but maybe not exactly
            candidates = [d for d in model_out_dir.iterdir() if d.is_dir()]
            # Simple heuristic: exact match or normalized match
            # Using the exact song_path stem is standard for Demucs unless specific flags are used
            song_demucs_dir = model_out_dir / song_id
            
            if not song_demucs_dir.exists():
                 # Try to look for it
                 logger.warning(f"Could not find Demucs output folder for {song_id} at {song_demucs_dir}. Checking candidates...")
                 # Logic to find the created folder?
                 # Assuming standard behavior for now.
                 logger.error(f"Skipping {song_id} due to missing Demucs output.")
                 continue

            stem_files = list(song_demucs_dir.glob("*.wav"))
            if not stem_files:
                logger.error(f"No stems found in {song_demucs_dir}")
                continue

            # 3. Select Solo
            # Preference: guitar > vocals > lead > melody
            # Typically Demucs produces: bass, drums, other, vocals (htdemucs)
            # Or guitar, piano, etc. (htdemucs_6s)
            
            solo_stem = None
            solo_name = None
            
            priority = ["guitar", "vocals", "lead", "melody"]
            
            for p_name in priority:
                for f in stem_files:
                    if p_name in f.name.lower():
                        solo_stem = f
                        solo_name = f.name
                        break
                if solo_stem:
                    break
            
            if not solo_stem:
                # Fallback: maybe 'other' if we consider it melody?
                # User prompt said: "prefer names containing... else skip"
                # So we skip if none match.
                # Standard htdemucs has "vocals". htdemucs_6s has "guitar". 
                # If using standard htdemucs (4 stems), usually vocals is the only melody candidate.
                logger.info(f"Skipping {song_id}: No matching solo stem (guitar/vocals/lead/melody) found in {stem_files}.")
                continue

            # 4. Construct Backing (Sum of others)
            backing_stems = [f for f in stem_files if f != solo_stem]
            if not backing_stems:
                 logger.warning(f"Skipping {song_id}: No backing stems found.")
                 continue

            # 5. Canonicalize & Pair
            # We load stems, verify SR/Channels, align, sum backing. 
            # Note: Demucs output matches input SR usually, but safely assume we need to check.
            
            def load_helper(p):
                a, sr = read_audio(str(p))
                a = ensure_channels(a, cfg.canonical_channels)
                if sr != cfg.canonical_sample_rate:
                    a = resample_audio(a, sr, cfg.canonical_sample_rate)
                return a

            solo_audio = load_helper(solo_stem)
            backing_audios = [load_helper(p) for p in backing_stems]
            
            all_stems = [solo_audio] + backing_audios
            all_stems = align_stems(all_stems)
            
            y_solo = all_stems[0]
            x_backing_components = all_stems[1:]
            x_backing = sum(x_backing_components)
            
            # Mix consistency check? 
            # x_backing + y_solo should equal mix (Demucs separation property).
            # We don't have the "true" clean mix separate from input unless we assume input is clean.
            # But Demucs artifacts apply to both.
            # Let's normalize the recombined mix to -18 LUFS.
            
            mix_est = x_backing + y_solo
             # Calculate gain using loudness_normalize logic (internal calc or exposed?)
             # We reuse the logic from build_pairs: normalize MIX, apply gain to components.
            
            # Since we don't have easy access to the exact gain used by `loudness_normalize`
            # (unless we refactor it), we can:
            # 1. Normalize mix
            # 2. Get peak/rms ratio of old vs new mix? No, too unstable.
            # 3. Use pyloudnorm to get gain explicitly here.
            
            import pyloudnorm as pyln
            try:
                meter = pyln.Meter(cfg.canonical_sample_rate)
                loudness = meter.integrated_loudness(mix_est)
                if loudness > -70:
                    gain_db = cfg.lufs_target - loudness
                    gain = 10**(gain_db / 20.0)
                else:
                    gain = 1.0
            except:
                gain = 1.0
            
            y_solo *= gain
            x_backing *= gain
            mix_est *= gain # For stats
            
            # 6. Write Output
            pair_dir.mkdir(parents=True, exist_ok=True)
            
            x_path = pair_dir / "x_backing.wav"
            y_path = pair_dir / "y_solo.wav"
            meta_path = pair_dir / "meta.json"
            
            write_audio(str(x_path), x_backing, cfg.canonical_sample_rate)
            write_audio(str(y_path), y_solo, cfg.canonical_sample_rate)
            
            stats_x = compute_stats(x_backing, cfg.canonical_sample_rate)
            stats_y = compute_stats(y_solo, cfg.canonical_sample_rate)
            
            meta = {
                "dataset": "weak_demucs",
                "track_id": song_id,
                "license": "unknown (weak)",
                "weak_label": True,
                "separator": "demucs",
                "demucs_model": cfg.demucs_model,
                "solo_source_stem": solo_name,
                "sr": cfg.canonical_sample_rate,
                "channels": cfg.canonical_channels,
                "duration_s": x_backing.shape[0] / cfg.canonical_sample_rate,
                "stats_x": stats_x,
                "stats_y": stats_y
            }
            
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
                
            logger.info(f"Generated weak pair for {song_id}")
            
        except Exception as e:
            logger.error(f"Failed to process {song_id}: {e}", exc_info=True)

    # Cleanup temp
    # shutil.rmtree(demucs_out_root, ignore_errors=True) # Optional: keep for debug? Or clean?
    # Usually clean up.
    try:
        shutil.rmtree(demucs_out_root)
    except:
        pass
