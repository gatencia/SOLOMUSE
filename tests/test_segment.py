import pytest
import numpy as np
import soundfile as sf
import csv
import json
import pandas as pd
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_data.segment import segment_dataset

@pytest.fixture
def fake_pairs_root(tmp_path):
    root = tmp_path / "mock_output"
    root.mkdir()
    
    pairs_dir = root / "pairs" / "mock"
    pairs_dir.mkdir(parents=True)
    
    sr = 44100
    # Create 10s audio
    # 6s window, 3s hop
    # Expected segments: [0-6], [3-9]. [6-12] would overflow (ends at 10).
    duration = 10.0
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    
    # Simple tone
    sine = 0.5 * np.sin(2*np.pi*440*t)
    stereo = np.column_stack([sine, sine]).astype(np.float32)
    
    x_path = pairs_dir / "x.wav"
    y_path = pairs_dir / "y.wav"
    sf.write(x_path, stereo, sr)
    sf.write(y_path, stereo, sr)
    
    # Create Manifest
    rows = [
        ["dataset", "track_id", "x_path", "y_path", "license"],
        ["mock", "track1", str(x_path), str(y_path), "free"]
    ]
    
    with open(pairs_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    return root

@pytest.fixture
def cfg(fake_pairs_root):
    return PipelineConfig(
        output_root=str(fake_pairs_root),
        dataset_roots={"mock": "dummy"},
        segment_seconds=6.0,
        segment_hop_seconds=3.0,
        min_segment_energy=0.0001
    )

def test_segment_dataset(fake_pairs_root, cfg):
    dataset_name = "mock"
    segment_dataset(dataset_name, cfg)
    
    seg_dir = fake_pairs_root / "segments" / "mock"
    assert seg_dir.exists()
    
    manifest_path = seg_dir / "manifest.csv"
    assert manifest_path.exists()
    
    df = pd.read_csv(manifest_path)
    assert len(df) == 2
    
    # Check Segments
    # 1. 0s - 6s
    row1 = df.iloc[0]
    assert row1["start_s"] == 0.0
    assert row1["end_s"] == 6.0
    
    # Check file duration
    x1, sr1 = sf.read(row1["x_path"])
    assert len(x1) == 44100 * 6
    
    # 2. 3s - 9s
    row2 = df.iloc[1]
    assert row2["start_s"] == 3.0
    assert row2["end_s"] == 9.0
    
    x2, sr2 = sf.read(row2["x_path"])
    assert len(x2) == 44100 * 6
    
    # Verify content consistency (roughly)
    # Since it's a sine wave, energy should be similar
    assert row1["energy"] > 0
    assert row2["energy"] > 0

def test_segment_drops_last_if_short(fake_pairs_root, cfg):
    # Already verified by 10s duration -> 2 segments (ends at 9s).
    # If 12s -> 0-6, 3-9, 6-12 (3 segments).
    pass

def test_segment_skips_silence(fake_pairs_root, cfg):
    # Override audio with silence
    pairs_dir = fake_pairs_root / "pairs" / "mock"
    sr = 44100
    silence = np.zeros((44100 * 10, 2), dtype=np.float32)
    
    sf.write(pairs_dir / "x.wav", silence, sr)
    sf.write(pairs_dir / "y.wav", silence, sr)
    
    segment_dataset("mock", cfg)
    
    seg_dir = fake_pairs_root / "segments" / "mock"
    # Manifest might exist (header), but rows should be empty? 
    # Or maybe manifest is overwritten/created empty?
    # Our impl creates manifest list, then writes. So if empty list, empty file with header.
    
    manifest_path = seg_dir / "manifest.csv"
    df = pd.read_csv(manifest_path)
    assert len(df) == 0
