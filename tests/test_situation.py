import pytest
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_model.situation.extract import extract_situation_v1
from solomuse_model.situation.vectorize import vectorize_situation_v1
from solomuse_model.situation.run import run_situation_extraction

@pytest.fixture
def cfg():
    return PipelineConfig(
        output_root="./data/processed_test",
        dataset_roots={"mock": "./data/raw/mock"},
        situation_frame_hz=100,
        situation_chroma_hz=10
    )

def test_rms_envelope_detected(cfg):
    """Verify RMS changes with signal amplitude."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sinusoid with increasing amplitude
    audio = (t * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = audio[:, np.newaxis] # [T, 1]
    
    features = extract_situation_v1(audio, sr, cfg)
    assert features["rms_mean"] > 0
    assert features["rms_std"] > 0

def test_onset_strength_detects_bursts(cfg):
    """Verify onsets are found in pulse trains."""
    sr = 44100
    duration = 1.0
    audio = np.zeros((int(sr * duration), 1), dtype=np.float32)
    # Add bursts
    for i in range(5):
        start = int(i * 0.2 * sr)
        audio[start:start+100] = 1.0
        
    features = extract_situation_v1(audio, sr, cfg)
    assert features["onset_strength_mean"] > 0
    # Higher complexity check: tempo should be roughly 300 BPM (5 pulses/sec) or half-time (150)
    assert 140 <= features["tempo_bpm"] <= 310

def test_chroma_has_dominant_pitch_class_for_tone(cfg):
    """Verify 440Hz -> A chroma."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz is A (index 9 in chroma: C=0, C#=1, ..., A=9)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    audio = audio[:, np.newaxis]
    
    features = extract_situation_v1(audio, sr, cfg)
    chroma = np.array(features["chroma_mean"])
    # Index 9 should be the maximum
    assert np.argmax(chroma) == 9

def test_silence_no_nan(cfg):
    """Ensure zero input doesn't crash or return NaN."""
    sr = 44100
    audio = np.zeros((sr, 1), dtype=np.float32)
    features = extract_situation_v1(audio, sr, cfg)
    assert features["rms_mean"] == 0
    assert features["loudness_lufs"] <= -70

def test_vector_shape_constant(cfg):
    """Verify fixed length across different inputs."""
    sr = 44100
    audio = np.random.randn(sr, 1).astype(np.float32)
    features = extract_situation_v1(audio, sr, cfg)
    vector = vectorize_situation_v1(features)
    assert vector.shape == (32,)
    assert vector.dtype == np.float32

def test_run_situation_writes_artifacts_and_manifest(cfg, tmp_path):
    """Integration test for the situation runner."""
    # Setup mock segment environment
    output_root = tmp_path / "processed"
    cfg.output_root = str(output_root)
    
    seg_dir = output_root / "segments" / "mock" / "track1" / "seg1"
    seg_dir.mkdir(parents=True)
    
    # Create x.wav
    sr = 44100
    audio = np.random.randn(sr, 2).astype(np.float32)
    import soundfile as sf
    x_path = seg_dir / "x.wav"
    sf.write(x_path, audio, sr)
    
    # Create manifest
    manifest_dir = output_root / "segments" / "mock"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    
    df = pd.DataFrame([{
        "segment_id": "seg1",
        "dataset": "mock",
        "track_id": "track1",
        "x_path_abs": str(x_path)
    }])
    df.to_csv(manifest_path, index=False)
    
    # Run
    run_situation_extraction(cfg, "mock")
    
    # Verify artifacts
    assert (seg_dir / "situation.npy").exists()
    assert (seg_dir / "meta.json").exists()
    assert (manifest_dir / "manifest_situation.csv").exists()
    
    with open(seg_dir / "meta.json", "r") as f:
        meta = json.load(f)
        assert "situation" in meta
        assert meta["situation"]["version"] == "v1"

