import numpy as np
from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.codec_interface import AudioCodec

def extract_renderer_targets_v1(y_audio: np.ndarray, sr: int, cfg: PipelineConfig, codec: AudioCodec) -> np.ndarray:
    """
    Encode the solo audio into the target representation required by the renderer pipeline.
    
    Args:
        y_audio: Solo audio array [T] or [T, C]
        sr: Sample rate
        cfg: Pipeline configuration
        codec: Instantiated AudioCodec
        
    Returns:
        Encoded frames [F, D]
    """
    codes = codec.encode(y_audio, sr)
    return codes
