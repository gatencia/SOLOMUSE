import shutil
import subprocess
import tempfile
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

def detect_ffmpeg(ffmpeg_path: Optional[str] = None) -> Optional[str]:
    """Detect ffmpeg executable."""
    if ffmpeg_path and Path(ffmpeg_path).exists():
        return ffmpeg_path
    return shutil.which("ffmpeg")

def read_audio(path: str, ffmpeg_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Read audio file into float32 numpy array.
    
    Args:
        path: Path to audio file.
        ffmpeg_path: Path to ffmpeg executable (optional).

    Returns:
        Tuple of (audio, samplerate).
        Audio is shaped [T, C] (time, channels) and is float32.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        # Try reading with soundfile first (fastest, supports WAV/FLAC/OGG)
        audio, sr = sf.read(str(p), always_2d=True, dtype="float32")
        return audio, sr
    except Exception as e_sf:
        # Fallback to ffmpeg
        ffmpeg_bin = detect_ffmpeg(ffmpeg_path)
        if not ffmpeg_bin:
            raise RuntimeError(f"Could not read {path} with soundfile and ffmpeg is not available. Soundfile error: {e_sf}")

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", mode="wb") as tmp_file:
                # Decode to temporary PCM WAV (float32)
                cmd = [
                    ffmpeg_bin,
                    "-y",           # Overwrite output
                    "-i", str(p),   # Input
                    "-f", "wav",    # Format
                    "-c:a", "pcm_f32le", # Codec: float32 little endian
                    tmp_file.name
                ]
                
                # Run ffmpeg silently
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Now read the wav
                audio, sr = sf.read(tmp_file.name, always_2d=True, dtype="float32")
                return audio, sr
                
        except subprocess.CalledProcessError as e_subprocess:
             raise RuntimeError(f"FFmpeg failed to decode {path}") from e_subprocess
        except Exception as e_gen:
             raise RuntimeError(f"Failed to read {path} via ffmpeg fallback") from e_gen

def write_audio(path: str, audio: np.ndarray, sr: int):
    """
    Write audio array to file.
    
    Args:
        path: Output path.
        audio: Audio data (must be [T, C] float32).
        sr: Sample rate.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    
    if audio.ndim != 2:
        raise ValueError(f"Audio must be 2D [T, C]. Got shape {audio.shape}")
        
    if audio.dtype != np.float32:
        # Warn or cast? Let's cast to safety but for rigorous pipeline, maybe warn?
        # The prompt says "Assert audio is float32", so let's raise if strict, or just convert.
        # "Ensure float32 output" implies we should convert if not.
        audio = audio.astype(np.float32)

    if not np.isfinite(audio).all():
        # Check for NaNs/Infs which cause libsndfile to crash or error
        num_nan = np.isnan(audio).sum()
        num_inf = np.isinf(audio).sum()
        raise ValueError(f"Metadata check failed for {path}: Contains {num_nan} NaNs and {num_inf} Infs.")

    try:
        sf.write(str(p), audio, sr, subtype="FLOAT")
    except Exception as e:
        # If soundfile fails, try to get more system context
        try:
            import os
            # List open files for this process if possible
            fds = os.listdir('/proc/self/fd') if os.path.exists('/proc/self/fd') else []
            fd_count = len(fds)
        except:
            fd_count = -1
        
        raise RuntimeError(f"soundfile.write failed for {p}. Error: {e}. Open FDs: {fd_count}")
