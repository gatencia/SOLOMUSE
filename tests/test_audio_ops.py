import numpy as np
import pytest
import pyloudnorm as pyln
from solomuse_data.audio_ops import (
    ensure_channels,
    resample_audio,
    loudness_normalize,
    canonicalize_audio,
    compute_peak_dbfs
)
from solomuse_data.config import PipelineConfig

def test_ensure_channels_mono_to_stereo():
    """Verify mono (1ch) is duplicated to stereo (2ch)."""
    # Create mono signal [100, 1]
    mono = np.random.rand(100, 1).astype(np.float32)
    stereo = ensure_channels(mono, target_channels=2)
    
    assert stereo.shape == (100, 2)
    assert np.array_equal(stereo[:, 0], mono[:, 0])
    assert np.array_equal(stereo[:, 1], mono[:, 0])

def test_ensure_channels_stereo_noop():
    """Verify stereo (2ch) remains unchanged."""
    stereo = np.random.rand(100, 2).astype(np.float32)
    out = ensure_channels(stereo, target_channels=2)
    assert np.array_equal(stereo, out)

def test_resample_duration_preserved():
    """Verify duration follows sample rate ratio."""
    sr_in = 48000
    sr_out = 44100
    duration = 1.0
    
    # 1 second sine wave
    t = np.linspace(0, duration, int(sr_in * duration), endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sine = sine[:, np.newaxis] # [T, 1]
    
    resampled = resample_audio(sine, sr_in, sr_out)
    
    expected_samples = int(sr_out * duration)
    # Allow small drift due to resampling filter
    assert abs(resampled.shape[0] - expected_samples) < 100 

def test_lufs_normalize_hits_target():
    """Verify signal is normalized to -18 LUFS."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave is easy to measure
    # Full scale sine has integrated loudness around -3.01 dBFS (peak) -> LUFS?
    # Actually, LUFS for sine is approx -3.01 relative to full scale.
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sine = np.column_stack([sine, sine]) # Stereo
    
    target_lufs = -18.0
    normalized = loudness_normalize(sine, sr, target_lufs, peak_limit_dbfs=0.0)
    
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(normalized)
    
    assert np.isclose(lufs, target_lufs, atol=0.1)

def test_peak_is_capped():
    """Verify peak limiting works when target LUFS would cause clipping."""
    sr = 44100
    duration = 0.5
    # Create a signal with very high dynamic range where boosting to -18 LUFS might clip peaks
    # Actually, easiest is to request -5 LUFS from a quiet signal that has a spike?
    # Or just use a sine wave and set peak limit lower than the target would imply?
    
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    sine = np.column_stack([sine, sine])
    
    # Target -10 LUFS, but limit peak to -6 dBFS
    # Sine at -10 LUFS would peak around -7? 
    # Let's set limit to -20 dBFS, target -18 LUFS. 
    # If we hit -18 LUFS, peak will be around -15 dBFS (sine crest factor is 3dB).
    # So peak limiter should kick in at -20.
    
    target_lufs = -18.0
    peak_limit_dbfs = -20.0
    
    normalized = loudness_normalize(sine, sr, target_lufs, peak_limit_dbfs)
    
    peak = compute_peak_dbfs(normalized)
    assert peak <= peak_limit_dbfs + 0.01 # allow float error
    
    # LUFS will be lower than target due to limiting
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(normalized)
    assert lufs < target_lufs

def test_canonicalize_audio_outputs_sr_channels():
    """Verify full pipeline produces correct format."""
    cfg = PipelineConfig(
        output_root="/tmp",
        canonical_sample_rate=44100,
        canonical_channels=2,
        lufs_target=-18.0
    )
    
    # Input: 48kHz Mono
    sr_in = 48000
    audio_in = np.random.rand(sr_in, 1).astype(np.float32) * 0.1
    
    audio_out, sr_out, stats = canonicalize_audio(audio_in, sr_in, cfg)
    
    assert sr_out == 44100
    assert audio_out.shape[1] == 2
    assert audio_out.dtype == np.float32
    assert stats["channels"] == 2
    assert np.isclose(stats["lufs"], -18.0, atol=1.0) # Noise LUT vs Sine LUT can vary but should be close
