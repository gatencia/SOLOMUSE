import logging
import numpy as np

from solomuse_model.renderer.codec_interface import AudioCodec
from solomuse_model.renderer.token_contracts import DiscreteTokenStream

logger = logging.getLogger(__name__)

class EnCodecAdapter(AudioCodec):
    """
    Placeholder adapter for EnCodec. 
    Implements the standard AudioCodec interface but raises NotImplementedError 
    on execution until the encodec package is fully integrated and tested.
    """
    def __init__(self, target_bandwidth: float = 6.0, target_sr: int = 24000):
        self.target_bandwidth = target_bandwidth
        self.target_sr = target_sr
        
        try:
            import encodec
            self._has_encodec = True
            logger.info("EnCodec imported successfully. (Integration logic is still pending)")
        except ImportError:
            self._has_encodec = False
            
    def encode(self, audio: np.ndarray, sr: int) -> DiscreteTokenStream:
        if not self._has_encodec:
            raise NotImplementedError(
                "EnCodec is not installed. Please install it to use the discrete neural codec. "
                "(Hint: pip install encodec)"
            )
        raise NotImplementedError("EnCodec runtime encode integration is pending the autoregressive token model upgrade.")

    def decode(self, codes: DiscreteTokenStream, sr: int) -> np.ndarray:
        if not self._has_encodec:
            raise NotImplementedError("EnCodec is not installed.")
        raise NotImplementedError("EnCodec runtime decode integration is pending the autoregressive token model upgrade.")

    def frame_rate_hz(self) -> float:
        # EnCodec at 24kHz with hop 320 gives exactly 75 Hz
        return 75.0
        
    @property
    def code_type(self) -> str:
        return "discrete"
        
    @property
    def code_dim(self) -> int:
        return None
        
    @property
    def num_codebooks(self) -> int:
        # Assuming bandwidth=6.0 -> 4 codebooks for EnCodec 24kHz
        return 4
        
    @property
    def vocab_size(self) -> int:
        return 1024
