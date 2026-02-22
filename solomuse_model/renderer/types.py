from typing import TypedDict, Optional, Any
import numpy as np

class RendererInputV1(TypedDict):
    """
    Contract for data flowing into the renderer generator.
    """
    x_audio: Optional[np.ndarray]     # Backing context audio
    x_audio_path: Optional[str]       # Or path to backing audio
    intent_sequence: np.ndarray       # [F, D] predicted or target intent
    situation_vector: Optional[np.ndarray] # [D_sit] segment situation state
    sr: int

class RendererTargetV1(TypedDict):
    """
    Contract for rendering targets during training.
    """
    y_audio: Optional[np.ndarray]     # Solo audio waveform
    y_audio_path: Optional[str]       # Or path
    target_codes: Optional[np.ndarray]# Precomputed codec sequence [F, CodecDim]
