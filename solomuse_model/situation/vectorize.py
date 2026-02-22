import numpy as np
from typing import List
from solomuse_model.situation.types import SituationFeaturesV1

def vectorize_situation_v1(features: SituationFeaturesV1) -> np.ndarray:
    """
    Vectorize SituationFeaturesV1 into a fixed-length numeric vector.
    
    Fixed Layout (32 dimensions):
    0:  tempo_norm (normalized by 200 BPM)
    1:  beat_confidence [0, 1]
    2:  beat_phase_sin (sine of phase)
    3:  beat_phase_cos (cosine of phase)
    4:  rms_mean
    5:  rms_std
    6:  loudness_norm (normalized from LUFS: (loudness + 70) / 70)
    7:  onset_mean
    8:  onset_std
    9-20: chroma_mean [12]
    21: spectral_centroid_mean_norm (normalized by 22050 Hz)
    22: spectral_centroid_std_norm (normalized by 22050 Hz)
    23: spectral_flux_mean (onset strength mean)
    24: spectral_flux_std (onset strength std)
    25-31: Reserved for future expansion (set to 0)
    
    Args:
        features: SituationFeaturesV1 object.
        
    Returns:
        np.ndarray of shape [32] and dtype float32.
    """
    vec = np.zeros(32, dtype=np.float32)
    
    # Rhythmic
    vec[0] = np.clip(features["tempo_bpm"] / 200.0, 0, 1.5)
    vec[1] = features["beat_confidence"]
    
    # Phase to circular encoding
    phase_rad = features["beat_phase"] * 2.0 * np.pi
    vec[2] = np.sin(phase_rad)
    vec[3] = np.cos(phase_rad)
    
    # Energy
    vec[4] = features["rms_mean"]
    vec[5] = features["rms_std"]
    vec[6] = np.clip((features["loudness_lufs"] + 70.0) / 70.0, 0, 1)
    
    # Onset
    vec[7] = features["onset_strength_mean"]
    vec[8] = features["onset_strength_std"]
    
    # Harmonic
    chroma = np.array(features["chroma_mean"], dtype=np.float32)
    vec[9:21] = chroma
    
    # Spectral
    vec[21] = np.clip(features["spectral_centroid_mean"] / 22050.0, 0, 1)
    vec[22] = np.clip(features["spectral_centroid_std"] / 22050.0, 0, 1)
    vec[23] = features["spectral_flux_mean"]
    vec[24] = features["spectral_flux_std"]
    
    return vec
