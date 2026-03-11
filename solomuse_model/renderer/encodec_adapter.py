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
        self._first_encode = True
        self._first_decode = True
        
        # Apple Silicon MPS currently lacks aten::_weight_norm_interface required by EnCodec.
        # This forces PyTorch to safely fallback to CPU for this specific layer.
        import os
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        
        try:
            import encodec
            self._has_encodec = True
            logger.info("EnCodec imported successfully. Initializing model...")
            
            import torch
            from encodec import EncodecModel
            # Instantiate model
            force_cpu = os.environ.get("SOLOMUSE_FORCE_CPU", "0") == "1"
            if force_cpu:
                self.device = torch.device('cpu')
                logger.info("Forcing CPU for EnCodec (SOLOMUSE_FORCE_CPU=1)")
            else:
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
            self.model = EncodecModel.encodec_model_24khz()
            self.model.set_target_bandwidth(self.target_bandwidth)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"EnCodec model loaded on {self.device} at {self.target_bandwidth}k bandwidth.")
            
            import torchaudio
            self.resampler = None # We will lazily instantiate to match input SR
            self.torchaudio = torchaudio
            
        except ImportError:
            self._has_encodec = False
            self.model = None
            
    def encode(self, audio: np.ndarray, sr: int, cache_path: str | None = None) -> DiscreteTokenStream:
        """
        Encode raw audio to discrete EnCodec tokens.
        Output shape: [F, Q] where F=frames, Q=codebooks
        """
        if cache_path is not None:
            import os
            if os.path.exists(cache_path):
                tokens = np.load(cache_path)
                # Quick shape sanity check
                if tokens.ndim == 2 and tokens.shape[1] == self.num_codebooks:
                    if self._first_encode:
                        logger.info(f"[EnCodec] Cache hit for first call at {cache_path}. Fast-loading shape {tokens.shape}")
                        self._first_encode = False
                    return tokens
        
        if not self._has_encodec:
            raise NotImplementedError(
                "EnCodec is not installed. Please install it to use the discrete neural codec. "
                "(Run: `pip install encodec torchaudio`)"
            )
            
        import torch
        
        with torch.no_grad():
            # Ensure float32 and shape [1, 1, T] 
            wav = torch.from_numpy(audio).float().to(self.device)
            if wav.ndim == 1:
                wav = wav.unsqueeze(0).unsqueeze(0)
            elif wav.ndim == 2:
                # Assumed [T, C], take first channel or mean
                wav = wav.mean(dim=1).unsqueeze(0).unsqueeze(0)
                
            if sr != self.target_sr:
                if self.resampler is None or self.resampler.orig_freq != sr:
                    self.resampler = self.torchaudio.transforms.Resample(sr, self.target_sr).to(self.device)
                wav = self.resampler(wav)
                
            # Perform encoding
            # Encodec returns a list of tuples: [(codes, scale_factor), ...]
            # We usually just run it all at once so list length is 1
            encoded_frames = self.model.encode(wav)
            codes = encoded_frames[0][0] # shape: [B=1, Q, F]
            
            # Extract and format to [F, Q]
            tokens = codes.squeeze(0).transpose(0, 1).cpu().numpy().astype(np.int32)
            
            # Truncate to the exact target bandwidth num_codebooks
            if tokens.shape[1] > self.num_codebooks:
                tokens = tokens[:, :self.num_codebooks]
            
            # Validation
            if not np.isfinite(tokens).all():
                raise ValueError("EnCodec produced NaN/Inf tokens!")
                
            if self._first_encode:
                logger.info(f"[EnCodec] First Encode Diagnostics:")
                logger.info(f"  Input Audio SR: {sr} -> Target SR: {self.target_sr}")
                logger.info(f"  Output Tokens Shape: {tokens.shape} (Frames x Codebooks)")
                logger.info(f"  Tokens Min/Max: {tokens.min()} / {tokens.max()}")
                self._first_encode = False
                
            if cache_path is not None:
                np.save(cache_path, tokens)
                
            return tokens

    def decode(self, codes: DiscreteTokenStream, sr: int) -> np.ndarray:
        """
        Decode [F, Q] tokens back to raw audio array [T].
        Note: The returned sequence is at EnCodec's native 24kHz.
        """
        if not self._has_encodec:
            raise NotImplementedError("EnCodec is not installed. Run: `pip install encodec`")
            
        import torch
        
        # Validation
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"EnCodecAdapter decode expects [F, Q] with Q={self.num_codebooks}. Got {codes.shape}")
            
        with torch.no_grad():
            # Format to [B=1, Q, F]
            codes_tensor = torch.from_numpy(codes).long().transpose(0, 1).unsqueeze(0).to(self.device)
            
            # Decode expects a list of tuples [(codes, None)] if we don't have scaling factors
            frames = [(codes_tensor, None)]
            wav = self.model.decode(frames)
            
            # Output is [1, 1, T]
            out_audio = wav.squeeze().cpu().numpy().astype(np.float32)
            
            if not np.isfinite(out_audio).all():
                raise ValueError("EnCodec output contains NaN/Inf values!")
                
            # If user wants a specific SR back, we should resample here, 
            # though natively we return 24k and they config expect 44.1k often.
            if sr != self.target_sr:
                if self.resampler is None or self.resampler.orig_freq != self.target_sr or self.resampler.new_freq != sr:
                    self.resampler = self.torchaudio.transforms.Resample(self.target_sr, sr).to(self.device)
                # Resample expects [channels, time] so unsqueeze
                out_t = torch.from_numpy(out_audio).unsqueeze(0).to(self.device)
                out_audio = self.resampler(out_t).squeeze(0).cpu().numpy().astype(np.float32)
                
            if self._first_decode:
                logger.info(f"[EnCodec] First Decode Diagnostics:")
                logger.info(f"  Input Tokens: {codes.shape} -> Output Wav: {out_audio.shape}")
                logger.info(f"  Output Min/Max: {out_audio.min():.4f} / {out_audio.max():.4f}")
                self._first_decode = False
                
            return out_audio

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
