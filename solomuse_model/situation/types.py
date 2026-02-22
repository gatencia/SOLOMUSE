from typing import TypedDict, List, Optional, Union
import numpy as np

class SituationFeaturesV1(TypedDict):
    """
    Structured musical features extracted from an audio segment.
    """
    version: str
    sr: int
    duration_s: float
    
    # Rhythmic features
    tempo_bpm: float
    beat_confidence: float
    beat_phase: float  # [0, 1]
    onset_strength_mean: float
    onset_strength_std: float
    
    # Energetic features
    rms_mean: float
    rms_std: float
    loudness_lufs: float
    
    # Harmonic/Spectral features
    chroma_mean: List[float]  # 12-dim
    chroma_std: List[float]   # 12-dim
    spectral_centroid_mean: float
    spectral_centroid_std: float
    spectral_flux_mean: float
    spectral_flux_std: float
    
    # Optional curves (time-series)
    rms_curve: Optional[List[float]]
    onset_curve: Optional[List[float]]
    chroma_curve: Optional[List[List[float]]]
