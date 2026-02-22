import pytest
import numpy as np
import os
import json
import pandas as pd
from pathlib import Path

from solomuse_data.config import PipelineConfig
from solomuse_model.intent.extract_targets import extract_intent_targets_v1
from solomuse_model.intent.vectorize import vectorize_intent_v1
from solomuse_model.intent.run import run_intent_target_build

@pytest.fixture
def cfg():
    return PipelineConfig(
        output_root="./data/processed_test_intent",
        dataset_roots={"mock": "./data/raw/mock"},
        intent_hz=10
    )

def test_intent_shape_matches_duration(cfg):
    """(6s at 10Hz ≈ 60 frames)"""
    sr = 44100
    duration = 6.0
    x_audio = np.zeros((int(sr * duration), 1), dtype=np.float32)
    y_audio = np.zeros((int(sr * duration), 1), dtype=np.float32)
    
    seq = extract_intent_targets_v1(x_audio, y_audio, sr, {}, cfg)
    vec = vectorize_intent_v1(seq)
    
    assert vec.shape[0] == 60
    assert vec.shape[1] == 7

def test_activity_low_on_silence(cfg):
    """Ensure silence yields near-0 activity."""
    sr = 44100
    duration = 1.0
    x_audio = np.zeros((sr, 1), dtype=np.float32)
    y_audio = np.zeros((sr, 1), dtype=np.float32)
    
    seq = extract_intent_targets_v1(x_audio, y_audio, sr, {}, cfg)
    vec = vectorize_intent_v1(seq)
    
    # play_prob is idx 0
    assert np.all(vec[:, 0] < 0.1)

def test_onset_prob_high_on_burst_signal(cfg):
    """Impulses yield high probabilities."""
    sr = 44100
    duration = 1.0
    x_audio = np.zeros((sr, 1), dtype=np.float32)
    y_audio = np.zeros((sr, 1), dtype=np.float32)
    
    # Add a burst at 0.5s
    burst_start = int(0.5 * sr)
    y_audio[burst_start:burst_start+100] = 1.0
    
    seq = extract_intent_targets_v1(x_audio, y_audio, sr, {}, cfg)
    vec = vectorize_intent_v1(seq)
    
    # onset_prob is idx 5
    # There should be at least one high onset prob around frame 5 (0.5s * 10hz)
    assert np.max(vec[:, 5]) > 0.5

def test_tension_proxy_higher_for_mismatched_tones(cfg):
    """Compare matching vs conflicting tones."""
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, sr, endpoint=False)
    
    # 440 Hz (A)
    tone_A = np.sin(2 * np.pi * 440 * t).astype(np.float32)[:, np.newaxis]
    # 466.16 Hz (A#, dissonant minor second)
    tone_Asharp = np.sin(2 * np.pi * 466.16 * t).astype(np.float32)[:, np.newaxis]
    
    # Matching: backing = A, solo = A
    seq_match = extract_intent_targets_v1(tone_A, tone_A, sr, {}, cfg)
    vec_match = vectorize_intent_v1(seq_match)
    
    # Mismatched: backing = A, solo = A#
    seq_mismatch = extract_intent_targets_v1(tone_A, tone_Asharp, sr, {}, cfg)
    vec_mismatch = vectorize_intent_v1(seq_mismatch)
    
    # tension_proxy is idx 3
    mean_tension_match = np.mean(vec_match[:, 3])
    mean_tension_mismatch = np.mean(vec_mismatch[:, 3])
    
    assert mean_tension_mismatch > mean_tension_match

def test_run_intent_writes_artifacts_and_manifest(cfg, tmp_path):
    """End-to-end dataset builder test."""
    output_root = tmp_path / "processed_intent"
    cfg.output_root = str(output_root)
    
    seg_dir = output_root / "segments" / "mock" / "track1" / "seg1"
    seg_dir.mkdir(parents=True)
    
    sr = 44100
    x_audio = np.zeros((sr, 1), dtype=np.float32)
    y_audio = np.ones((sr, 1), dtype=np.float32) * 0.5
    
    import soundfile as sf
    x_path = seg_dir / "x.wav"
    y_path = seg_dir / "y.wav"
    sf.write(x_path, x_audio, sr)
    sf.write(y_path, y_audio, sr)
    
    manifest_dir = output_root / "segments" / "mock"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest.csv"
    
    df = pd.DataFrame([{
        "segment_id": "seg1",
        "dataset": "mock",
        "track_id": "track1",
        "x_path_abs": str(x_path),
        "y_path_abs": str(y_path)
    }])
    df.to_csv(manifest_path, index=False)
    
    run_intent_target_build(cfg, "mock")
    
    assert (seg_dir / "intent_targets.npy").exists()
    assert (seg_dir / "meta.json").exists()
    assert (manifest_dir / "manifest_intent.csv").exists()
    
    with open(seg_dir / "meta.json", "r") as f:
        meta = json.load(f)
        assert "intent" in meta

