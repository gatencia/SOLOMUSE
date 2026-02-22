import numpy as np
import librosa
import pyloudnorm as pylan
from typing import Any
from solomuse_model.situation.types import SituationFeaturesV1
from solomuse_data.config import PipelineConfig

def extract_situation_v1(audio: np.ndarray, sr: int, cfg: PipelineConfig) -> SituationFeaturesV1:
    """
    Extract musical situation features from audio.
    
    Args:
        audio: Audio array, shape [T, C].
        sr: Sample rate.
        cfg: Pipeline configuration.
        
    Returns:
        Structured SituationFeaturesV1.
    """
    # 1. Mono mix for analysis
    if audio.ndim > 1 and audio.shape[1] > 1:
        y = np.mean(audio, axis=1)
    else:
        y = audio.flatten()

    # 2. Duration
    duration_s = float(len(y) / sr)
    
    # 3. Handle silence
    if np.max(np.abs(y)) < 1e-6:
        return _make_silent_features(sr, duration_s)

    # 4. Hop/Window logic
    # We use cfg.situation_frame_hz for time-series resolution if enabled
    hop_length = int(sr / cfg.situation_frame_hz)
    
    # 5. Rhythmic Features (Onset Strength / Tempo)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_mean = float(np.mean(onset_env))
    onset_std = float(np.std(onset_env))
    
    # Tempo/Beats
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, onset_envelope=onset_env, hop_length=hop_length)
    # tempo is a scalar or array? librosa 0.10+ returns scalar
    tempo_bpm = float(tempo[0] if isinstance(tempo, (np.ndarray, list)) else tempo)
    
    # Beat Phase / Confidence (Simple estimate)
    beat_conf = 0.0
    beat_phase = 0.0
    if len(beats) > 1:
        # Distance between beats compared to expected interval
        intervals = np.diff(beats)
        expected_interval = (60.0 / tempo_bpm) * (sr / hop_length) if tempo_bpm > 0 else 0
        if expected_interval > 0:
            beat_conf = float(1.0 - np.clip(np.mean(np.abs(intervals - expected_interval)) / expected_interval, 0, 1))
        
        # Phase of last sample relative to beat sequence
        beat_phase = float((len(onset_env) % expected_interval) / expected_interval) if expected_interval > 0 else 0.0

    # 6. Energetic Features (RMS / Loudness)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_mean = float(np.mean(rms))
    rms_std = float(np.std(rms))
    
    # Loudness (LUFS) - Requires at least 400ms by spec, but we handle shorter
    try:
        meter = pylan.Meter(sr)
        loudness = meter.integrated_loudness(audio)
    except:
        loudness = -70.0 # Silence floor
        
    # 7. Harmonic Features (Chroma)
    # Use higher hop for chroma to match situation_chroma_hz
    chroma_hop = int(sr / cfg.situation_chroma_hz)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=chroma_hop)
    chroma_mean = np.mean(chroma, axis=1).tolist()
    chroma_std = np.std(chroma, axis=1).tolist()
    
    # 8. Spectral Features
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    centroid_mean = float(np.mean(centroid))
    centroid_std = float(np.std(centroid))
    
    # Spectral Flux (approximated by onset envelope)
    flux_mean = onset_mean
    flux_std = onset_std
    
    features: SituationFeaturesV1 = {
        "version": "v1",
        "sr": sr,
        "duration_s": duration_s,
        "tempo_bpm": float(np.nan_to_num(tempo_bpm)),
        "beat_confidence": float(np.nan_to_num(beat_conf)),
        "beat_phase": float(np.nan_to_num(beat_phase)),
        "onset_strength_mean": float(np.nan_to_num(onset_mean)),
        "onset_strength_std": float(np.nan_to_num(onset_std)),
        "rms_mean": float(np.nan_to_num(rms_mean)),
        "rms_std": float(np.nan_to_num(rms_std)),
        "loudness_lufs": float(np.nan_to_num(loudness)),
        "chroma_mean": [float(v) for v in np.nan_to_num(chroma_mean)],
        "chroma_std": [float(v) for v in np.nan_to_num(chroma_std)],
        "spectral_centroid_mean": float(np.nan_to_num(centroid_mean)),
        "spectral_centroid_std": float(np.nan_to_num(centroid_std)),
        "spectral_flux_mean": float(np.nan_to_num(flux_mean)),
        "spectral_flux_std": float(np.nan_to_num(flux_std)),
        "rms_curve": rms.tolist() if cfg.situation_include_curves else None,
        "onset_curve": onset_env.tolist() if cfg.situation_include_curves else None,
        "chroma_curve": chroma.T.tolist() if cfg.situation_include_curves else None,
    }
    
    return features

def _make_silent_features(sr: int, duration_s: float) -> SituationFeaturesV1:
    """Return a zeroed feature set for silence."""
    return {
        "version": "v1",
        "sr": sr,
        "duration_s": duration_s,
        "tempo_bpm": 0.0,
        "beat_confidence": 0.0,
        "beat_phase": 0.0,
        "onset_strength_mean": 0.0,
        "onset_strength_std": 0.0,
        "rms_mean": 0.0,
        "rms_std": 0.0,
        "loudness_lufs": -70.0,
        "chroma_mean": [0.0] * 12,
        "chroma_std": [0.0] * 12,
        "spectral_centroid_mean": 0.0,
        "spectral_centroid_std": 0.0,
        "spectral_flux_mean": 0.0,
        "spectral_flux_std": 0.0,
        "rms_curve": None,
        "onset_curve": None,
        "chroma_curve": None,
    }
