import pytest
import numpy as np
import torch
import os
from pathlib import Path
from solomuse_model.renderer.encodec_adapter import EnCodecAdapter

# Need encodec installed to test
try:
    import encodec
    import torchaudio
    HAS_ENCODEC = True
except ImportError:
    HAS_ENCODEC = False

@pytest.fixture
def adapter():
    return EnCodecAdapter(target_bandwidth=6.0, target_sr=24000)

@pytest.mark.skipif(not HAS_ENCODEC, reason="encodec not installed")
def test_encodec_roundtrip(adapter):
    """Test full cycle: raw float32 -> tokens [F, Q] -> raw float32."""
    sr = 44100
    duration_s = 2.0
    samples = int(sr * duration_s)
    
    # Generate reproducible white noise sine wave combo
    np.random.seed(42)
    fake_audio = np.sin(2 * np.pi * 440.0 * np.arange(samples) / sr).astype(np.float32)
    fake_audio += np.random.normal(0, 0.1, size=samples).astype(np.float32)
    
    # Encode
    tokens = adapter.encode(fake_audio, sr)
    assert tokens.ndim == 2
    assert tokens.dtype == np.int32
    assert tokens.shape[1] == 4 # Q=4 for 6kbps
    
    # Check deterministic frame length: 75hz * 2s = 150 frames. 
    assert 148 <= tokens.shape[0] <= 152, f"Expected ~150 frames, got {tokens.shape[0]}"
    
    # Decode
    out_audio = adapter.decode(tokens, sr)
    
    assert out_audio.ndim == 1
    assert out_audio.dtype == np.float32
    # Ensure it's finite
    assert np.isfinite(out_audio).all()
    # Length should match roughly 2 seconds at 44.1k
    diff = abs(len(out_audio) - samples)
    assert diff < sr * 0.1, "Decoded audio length differs by >100ms from input"
    
@pytest.mark.skipif(not HAS_ENCODEC, reason="encodec not installed")
def test_token_caching(adapter, tmp_path):
    """Test that cache_path bypasses encoding."""
    sr = 44100
    samples = int(sr * 1.0)
    fake_audio = np.random.normal(0, 0.1, size=samples).astype(np.float32)
    cache_path = tmp_path / "renderer_tokens.npy"
    
    # First ENCODE: Computes and saves
    tokens1 = adapter.encode(fake_audio, sr, cache_path=str(cache_path))
    assert cache_path.exists()
    
    # Modify cache file explicitly to prove it's reading the cache, not recomputing
    poisoned_tokens = np.zeros_like(tokens1)
    np.save(str(cache_path), poisoned_tokens)
    
    # Second ENCODE: Should ONLY read cache
    tokens2 = adapter.encode(fake_audio, sr, cache_path=str(cache_path))
    assert (tokens2 == 0).all(), "Adapter recomputed instead of reading cache!"
    assert tokens1.shape == tokens2.shape

@pytest.mark.skipif(not HAS_ENCODEC, reason="encodec not installed")
def test_decode_writes_audio(adapter, tmp_path):
    """Test decoding and saving to wav."""
    import scipy.io.wavfile as wavfile
    # Make a tiny valid token array
    F = 75 # 1 second at 75hz
    tokens = np.random.randint(0, 500, size=(F, 4)).astype(np.int32)
    
    out_audio = adapter.decode(tokens, sr=44100)
    wav_path = tmp_path / "test_out.wav"
    wavfile.write(wav_path, 44100, (out_audio * 32767).astype(np.int16))
    
    assert wav_path.exists()
    assert wav_path.stat().st_size > 44100 * 2 # At least 1 sec of 16-bit
