from abc import ABC, abstractmethod
import numpy as np

class AudioCodec(ABC):
    """
    Abstract interface for compression/representation codecs used by the renderer.
    This allows swapping between raw waveforms, spectral representations, or neural codecs (e.g. EnCodec).
    """
    
    @abstractmethod
    def encode(self, audio: np.ndarray, sr: int, cache_path: str | None = None) -> np.ndarray:
        """
        Encode an audio waveform into the codec representation.
        
        Args:
            audio: [T] or [T, C] float32 array
            sr: sample rate
            cache_path: Optional path to save/load cached representation
            
        Returns:
            np.ndarray consisting of the encoded representation, typically [F, dims]
        """
        pass
        
    @abstractmethod
    def decode(self, codes: np.ndarray, sr: int) -> np.ndarray:
        """
        Decode the representation back to audio.
        
        Args:
            codes: np.ndarray, typically [F, dims]
            sr: expected target sample rate
            
        Returns:
            [T] float32 audio array
        """
        pass
        
    @abstractmethod
    def frame_rate_hz(self) -> float:
        """Return the effective frame rate / frequency of the encoded sequence."""
        pass
        
    @property
    @abstractmethod
    def code_type(self) -> str:
        """Return 'continuous' or 'discrete'"""
        pass
        
    @property
    @abstractmethod
    def code_dim(self) -> int:
        """Return dimensionality of continuous representations (or None if discrete)"""
        pass
        
    @property
    @abstractmethod
    def num_codebooks(self) -> int:
        """Return number of parallel quantizers/codebooks for discrete representation (or None if continuous)"""
        pass
        
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Return the dictionary size per codebook for discrete representation (or None if continuous)"""
        pass

class WaveChunkCodec(AudioCodec):
    """
    A baseline placeholder codec that just chops the waveform into sequential frames.
    Format: [F, chunk_samples]
    """
    def __init__(self, frame_ms: float = 20.0, hop_ms: float = 10.0, target_sr: int = 44100):
        self.frame_ms = frame_ms
        self.hop_ms = hop_ms
        self.target_sr = target_sr
        
        self.frame_size = int((frame_ms / 1000.0) * target_sr)
        self.hop_size = int((hop_ms / 1000.0) * target_sr)
        
    def encode(self, audio: np.ndarray, sr: int, cache_path: str | None = None) -> np.ndarray:
        if cache_path is not None:
            import os
            if os.path.exists(cache_path):
                # Naive cache load for baseline proxy
                return np.load(cache_path)
                
        if sr != self.target_sr:
            # Bypass librosa resample for tests to avoid the caching bug if sr matches, 
            # or use a simple numpy abstraction if not.
            pass
            
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1) # Mono mix
            
        # Due to a pytest/librosa interaction bug (`cannot cache function '_localmax'`),
        # we do a manual stride/frame instead of librosa.util.frame just for the baseline placeholder.
        # This keeps the environment robust.
        
        num_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        if num_frames <= 0:
            return np.zeros((1, self.frame_size), dtype=audio.dtype)
            
        
        frames = np.lib.stride_tricks.as_strided(
            audio, 
            shape=(num_frames, self.frame_size),
            strides=(audio.strides[0] * self.hop_size, audio.strides[0])
        ).copy()
        
        if cache_path is not None:
            np.save(cache_path, frames)
            
        return frames
        
    def decode(self, codes: np.ndarray, sr: int) -> np.ndarray:
        # Note: True overlap-add is needed for proper reconstruction with hop < frame
        # For this naive placeholder, if it's strictly framed without hop overlap, we could just reshape.
        # But librosa framing often uses hop. 
        # A simple fallback for now is to just take the non-overlapping centers or ignore overlap smoothing
        # since this is just a baseline interface proxy before we bring in EnCodec.
        
        F, chunk_size = codes.shape
        # Flatten naive assuming it's mostly disjoint or we just accept the artifacts for the baseline proxy
        # A more robust reverse would use librosa.feature.inverse.overlap_add but it requires scipy.
        # Simple disjoint reconstruction for testing codec pipeline viability:
        
        if self.frame_size == self.hop_size:
            audio = codes.flatten()
        else:
            # Very naive overlap-add
            t_len = (F - 1) * self.hop_size + self.frame_size
            audio = np.zeros(t_len, dtype=np.float32)
            counts = np.zeros(t_len, dtype=np.float32)
            
            for i in range(F):
                start = i * self.hop_size
                end = start + self.frame_size
                audio[start:end] += codes[i]
                counts[start:end] += 1.0
                
            safe_counts = np.where(counts > 0, counts, 1.0)
            audio /= safe_counts
            
        import librosa
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=self.target_sr, target_sr=sr)
            
        return audio

    def frame_rate_hz(self) -> float:
        return 1000.0 / self.hop_ms

    @property
    def code_type(self) -> str:
        return "continuous"
        
    @property
    def code_dim(self) -> int:
        return self.frame_size
        
    @property
    def num_codebooks(self) -> int:
        return None
        
    @property
    def vocab_size(self) -> int:
        return None
