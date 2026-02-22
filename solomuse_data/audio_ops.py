import numpy as np
import pyloudnorm as pyln
import resampy
from typing import Dict, Tuple, Optional
from solomuse_data.config import PipelineConfig

def ensure_channels(audio: np.ndarray, target_channels: int) -> np.ndarray:
    """
    Ensure audio has target_channels.
    Only supports:
    - Mono (1) -> Stereo (2): Duplicate channel
    - Multichannel (>2) -> Stereo (2): Downmix (mean) -> Duplicate if result is mono? No, mean reduces to 1D? No, mean(axis=1) is [T].
    - Stereo (2) -> Stereo (2): No-op
    """
    if audio.ndim == 1:
        # Should have been handled by io.py but safeguard
        audio = audio[:, np.newaxis]

    current_channels = audio.shape[1]

    if current_channels == target_channels:
        return audio

    if target_channels == 2:
        if current_channels == 1:
            # Mono to Stereo: Duplicate
            return np.repeat(audio, 2, axis=1)
        elif current_channels > 2:
            # Downmix to Mono first then to Stereo? Or just take first 2?
            # Prompt says: "downmix to target by mean across channels then upmix if needed"
            # Mean across channels results in mono [T, 1]
            mono = np.mean(audio, axis=1, keepdims=True)
            return np.repeat(mono, 2, axis=1)
    
    # If target is 1 (not required by spec but good to have)
    if target_channels == 1:
        return np.mean(audio, axis=1, keepdims=True)

    raise ValueError(f"Unsupported channel conversion: {current_channels} -> {target_channels}")

def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio using resampy.
    Input must be [T, C] float32.
    """
    if orig_sr == target_sr:
        return audio

    # Resampy handles [T, C] correctly if axis=0 (default)
    # audio is [T, C].
    resampled = resampy.resample(audio, orig_sr, target_sr, axis=0, filter='kaiser_best')
    return resampled.astype(np.float32)

def compute_peak_dbfs(audio: np.ndarray) -> float:
    """Compute peak dBFS."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return -np.inf
    return 20 * np.log10(max(peak, 1e-12))

def compute_rms(audio: np.ndarray) -> float:
    """Compute RMS amplitude."""
    return np.sqrt(np.mean(audio**2))

def loudness_normalize(audio: np.ndarray, sr: int, target_lufs: float, peak_limit_dbfs: float) -> np.ndarray:
    """
    Normalize audio to target LUFS and apply peak limiting if necessary.
    """
    meter = pyln.Meter(sr) 
    # measure_integrated_loudness usually expects [T, C].Mono handles [T, 1] ok?
    # pyloudnorm docs: input data: (samples, channels)
    try:
        loudness = meter.integrated_loudness(audio)
    except ValueError:
        # Silence or too short
        loudness = -np.inf

    # Normalize to target LUFS
    if loudness == -np.inf:
        # Silence remains silence
        return audio
    
    # Calculate gain
    gain_db = target_lufs - loudness
    gain_lin = 10**(gain_db / 20.0)
    
    # Apply gain
    normalized = audio * gain_lin
    
    # Check peak
    peak_db = compute_peak_dbfs(normalized)
    
    if peak_db > peak_limit_dbfs:
        # Exceeds limit. Apply attenuation to meet limit exactly.
        # Required attenuation = peak_db - peak_limit_dbfs
        attenuation_db = peak_db - peak_limit_dbfs
        attenuation_lin = 10**(-attenuation_db / 20.0)
        normalized = normalized * attenuation_lin
        
    return normalized.astype(np.float32)

def compute_stats(audio: np.ndarray, sr: int, target_lufs: Optional[float] = None) -> Dict:
    """Compute audio statistics."""
    meter = pyln.Meter(sr)
    try:
        lufs = meter.integrated_loudness(audio)
    except ValueError:
        lufs = -np.inf

    return {
        "duration_s": audio.shape[0] / sr,
        "rms": float(compute_rms(audio)),
        "peak_dbfs": float(compute_peak_dbfs(audio)),
        "lufs": float(lufs),
        "channels": audio.shape[1]
    }

def canonicalize_audio(audio: np.ndarray, sr: int, cfg: PipelineConfig) -> Tuple[np.ndarray, int, Dict]:
    """
    Full canonicalization pipeline:
    1. Ensure channels
    2. Resample
    3. Normalize
    """
    # 1. Channels
    audio = ensure_channels(audio, cfg.canonical_channels)
    
    # 2. Resample
    if sr != cfg.canonical_sample_rate:
        audio = resample_audio(audio, sr, cfg.canonical_sample_rate)
        sr = cfg.canonical_sample_rate
        
    # 3. Normalize
    audio = loudness_normalize(audio, sr, cfg.lufs_target, cfg.peak_limit_dbfs)
    
    stats = compute_stats(audio, sr)
    
    return audio, sr, stats
