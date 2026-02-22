import numpy as np
import librosa
from typing import Any
from solomuse_model.intent.types import IntentFrameV1, IntentSequenceV1
from solomuse_data.config import PipelineConfig

def extract_intent_targets_v1(x_audio: np.ndarray, y_audio: np.ndarray, sr: int, situation_features: Any, cfg: PipelineConfig) -> IntentSequenceV1:
    """
    Extract intent control targets from solo audio (y) aligned with backing (x).
    
    Args:
        x_audio: Backing audio array, shape [T, C].
        y_audio: Solo audio array, shape [T, C].
        sr: Sample rate.
        situation_features: Situation features from Layer 1.
        cfg: Pipeline configuration.
        
    Returns:
        Structured IntentSequenceV1 at cfg.intent_hz.
    """
    # 1. Mono mix for analysis
    if y_audio.ndim > 1 and y_audio.shape[1] > 1:
        y = np.mean(y_audio, axis=1)
    else:
        y = y_audio.flatten()
        
    if x_audio.ndim > 1 and x_audio.shape[1] > 1:
        x = np.mean(x_audio, axis=1)
    else:
        x = x_audio.flatten()

    duration_s = float(len(y) / sr)
    hz = cfg.intent_hz
    hop_length = int(sr / hz)
    
    # Target frame count
    n_frames = int(np.ceil(len(y) / hop_length))
    
    # 2. Extract Base Features (y)
    # RMS (Dynamics/Play Prob)
    rms_y = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_y = _pad_or_trim(rms_y, n_frames)
    
    # Onsets
    onset_env_y = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_env_y = _pad_or_trim(onset_env_y, n_frames)
    
    # Spectral Centroid (Register Proxy)
    if cfg.intent_use_centroid_for_register:
        centroid_y = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        centroid_y = _pad_or_trim(centroid_y, n_frames)
    else:
        centroid_y = np.zeros(n_frames)
        
    # Chroma (Tension Proxy)
    if cfg.intent_use_chroma_tension_proxy:
        # Use chroma_cqt or stft
        chroma_y = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length)
        chroma_x = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop_length)
        
        # [12, F]
        chroma_y = _pad_or_trim_2d(chroma_y, n_frames)
        chroma_x = _pad_or_trim_2d(chroma_x, n_frames)
    else:
        chroma_y = np.zeros((12, n_frames))
        chroma_x = np.zeros((12, n_frames))

    # 3. Process into Intent Proxies
    
    # A. Dynamics & Play Prob
    # Normalize RMS to [0, 1] locally or via fixed max. We use a sensible proxy max (e.g. 0.5 for RMS amplitude).
    dynamics_norm = np.clip(rms_y / 0.5, 0, 1)
    
    # Play prob is high when RMS is above a small noise floor
    noise_floor = 0.01
    play_prob = np.clip((rms_y - noise_floor) / (0.1 - noise_floor), 0, 1)
    
    # B. Onset Prob
    # Normalize onset envelope
    onset_max = np.max(onset_env_y) if np.max(onset_env_y) > 0 else 1.0
    onset_prob = np.clip(onset_env_y / onset_max, 0, 1)
    
    # C. Density Proxy (Smoothed Onsets or Spectral Flux)
    # Simple moving average of onset_prob
    kernel = np.ones(int(hz * 0.5)) / int(hz * 0.5) # 500ms window
    density_proxy = np.convolve(onset_prob, kernel, mode='same')
    density_proxy = np.clip(density_proxy, 0, 1)
    
    # D. Register Norm
    # Centroid typically 0-8000 Hz. Normalize to [0, 1]
    register_norm = np.clip(centroid_y / 8000.0, 0, 1)
    
    # E. Tension Proxy (Chroma Mismatch)
    # Cosine distance between solo and backing chroma
    # If play_prob is low, tension doesn't matter much, but we compute it anyway.
    norm_x = np.linalg.norm(chroma_x, axis=0)
    norm_y = np.linalg.norm(chroma_y, axis=0)
    
    # Avoid div by zero
    valid_mask = (norm_x > 1e-6) & (norm_y > 1e-6)
    cosine_sim = np.zeros(n_frames)
    
    # Dot product along pitch class axis
    dot_prod = np.sum(chroma_x * chroma_y, axis=0)
    cosine_sim[valid_mask] = dot_prod[valid_mask] / (norm_x[valid_mask] * norm_y[valid_mask])
    
    # Tension = 1 - similarity
    tension_proxy = np.clip(1.0 - cosine_sim, 0, 1)
    # Force tension to 0 when not playing
    tension_proxy = tension_proxy * (play_prob > 0.1)
    
    # F. Phrase Boundary Hint
    # High when transitioning from playing to resting, or prolonged rest
    # 1 - play_prob, but smoothed
    phrase_boundary_hint = 1.0 - play_prob
    # Emphasize true boundaries: drop in play prob + no onsets recently
    
    # 4. Assemble Sequence
    frames = []
    for i in range(n_frames):
        frame: IntentFrameV1 = {
            "play_prob": float(np.nan_to_num(play_prob[i])),
            "density_proxy": float(np.nan_to_num(density_proxy[i])),
            "register_norm": float(np.nan_to_num(register_norm[i])),
            "tension_proxy": float(np.nan_to_num(tension_proxy[i])),
            "dynamics_norm": float(np.nan_to_num(dynamics_norm[i])),
            "onset_prob": float(np.nan_to_num(onset_prob[i])),
            "phrase_boundary_hint": float(np.nan_to_num(phrase_boundary_hint[i]))
        }
        frames.append(frame)
        
    seq: IntentSequenceV1 = {
        "version": "v1",
        "sr": sr,
        "duration_s": duration_s,
        "hz": hz,
        "frames": frames
    }
    
    return seq

def _pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    if len(arr) < target_len:
        return np.pad(arr, (0, target_len - len(arr)))
    return arr[:target_len]

def _pad_or_trim_2d(arr: np.ndarray, target_len: int) -> np.ndarray:
    # arr is [12, F]
    features = arr.shape[0]
    F = arr.shape[1]
    if F < target_len:
        return np.pad(arr, ((0, 0), (0, target_len - F)))
    return arr[:, :target_len]
