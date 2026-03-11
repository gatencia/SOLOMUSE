import numpy as np
import pytest
import soundfile as sf
import csv
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_data.validate import validate_pairs

@pytest.fixture
def fake_output_root(tmp_path):
    root = tmp_path / "mock_output"
    root.mkdir()
    
    # Create fake dataset structure
    ds_pairs_dir = root / "pairs" / "mock"
    ds_pairs_dir.mkdir(parents=True)
    
    # Create audio files
    sr = 44100
    duration = 6.0
    t = np.linspace(0, duration, int(sr*duration), endpoint=False)
    sine = 0.5 * np.sin(2*np.pi*440*t)
    stereo = np.column_stack([sine, sine])
    
    # 1. Valid Pair
    sf.write(ds_pairs_dir / "valid_x.wav", stereo, sr)
    sf.write(ds_pairs_dir / "valid_y.wav", stereo, sr)
    
    # 2. Wrong SR
    sf.write(ds_pairs_dir / "wrong_sr_x.wav", stereo, 48000)
    sf.write(ds_pairs_dir / "wrong_sr_y.wav", stereo, 44100)
    
    # 3. Clipped
    clipped = 2.0 * sine # +6dB roughly
    clipped_stereo = np.column_stack([clipped, clipped])
    sf.write(ds_pairs_dir / "clipped_x.wav", clipped_stereo, sr)
    sf.write(ds_pairs_dir / "clipped_y.wav", stereo, sr)
    
    # 4. Length Mismatch
    sf.write(ds_pairs_dir / "short_x.wav", stereo[:1000], sr)
    sf.write(ds_pairs_dir / "short_y.wav", stereo, sr)
    
    # Create Manifest
    manifest_rows = [
        ["dataset", "track_id", "x_path", "y_path"],
        ["mock", "valid", str(ds_pairs_dir / "valid_x.wav"), str(ds_pairs_dir / "valid_y.wav")],
        ["mock", "wrong_sr", str(ds_pairs_dir / "wrong_sr_x.wav"), str(ds_pairs_dir / "wrong_sr_y.wav")],
        ["mock", "clipped", str(ds_pairs_dir / "clipped_x.wav"), str(ds_pairs_dir / "clipped_y.wav")],
        ["mock", "mismatch", str(ds_pairs_dir / "short_x.wav"), str(ds_pairs_dir / "short_y.wav")],
        ["mock", "missing", str(ds_pairs_dir / "nonexistent.wav"), str(ds_pairs_dir / "valid_y.wav")]
    ]
    
    with open(ds_pairs_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(manifest_rows)
        
    return root

@pytest.fixture
def fake_segments_root(fake_output_root):
    # Create segments directory and manifest
    root = fake_output_root
    seg_dir = root / "segments" / "mock"
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 44100
    # Create segment files
    # 6.0s exact
    exact_samples = int(sr * 6.0)
    # Use sine wave to pass energy check
    t = np.linspace(0, 6.0, exact_samples, endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    exact_audio = np.column_stack([sine, sine])
    
    # 1. Valid
    sf.write(seg_dir / "valid_x.wav", exact_audio, sr)
    sf.write(seg_dir / "valid_y.wav", exact_audio, sr)
    
    # 2. Bad Duration (too short)
    short_audio = exact_audio[:-100] # -100 samples
    sf.write(seg_dir / "bad_dur_x.wav", short_audio, sr)
    sf.write(seg_dir / "bad_dur_y.wav", short_audio, sr)
    
    # Manifest
    rows = [
        ["segment_id", "x_path", "y_path"],
        ["seg_valid", str(seg_dir / "valid_x.wav"), str(seg_dir / "valid_y.wav")],
        ["seg_bad_dur", str(seg_dir / "bad_dur_x.wav"), str(seg_dir / "bad_dur_y.wav")]
    ]
    
    with open(seg_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)
        
    return root

@pytest.fixture
def cfg(fake_output_root):

    return PipelineConfig(
        output_root=str(fake_output_root),
        dataset_roots={"mock": "dummy"}
    )

def test_validate_pairs_catches_errors(fake_output_root, cfg):
    # We need to ensure 'mock' is a valid known dataset key OR modify PipelineConfig in fixture
    # But wait, config.py validation checks keys in dataset_roots.
    # We hacked config previously to allow "mock".
    
    report = validate_pairs(cfg, "mock")
    
    assert report["passed"] == 1 # Only 'valid' row passes
    assert report["failed"] == 4 # wrong_sr, clipped, mismatch, missing
    
    failures = {f["track_id"]: f["reasons"] for f in report["failures"]}
    
    assert "wrong_sr" in failures
    assert any("48000" in r for r in failures["wrong_sr"])
    
    assert "clipped" in failures
    assert any("peak" in r for r in failures["clipped"])
    
    assert "mismatch" in failures
    assert any("Length mismatch" in r for r in failures["mismatch"])
    
    assert "missing" in failures
    assert any("missing" in r for r in failures["missing"])

from solomuse_data.validate import validate_segments

def test_validate_segments_catches_errors(fake_segments_root, cfg):
    report = validate_segments(cfg, "mock")
    
    assert report["passed"] == 1 # valid
    assert report["failed"] == 1 # bad_dur
    
    failures = {f["segment_id"]: f["reasons"] for f in report["failures"]}
    
    assert "seg_bad_dur" in failures
    assert any("Duration mismatch" in r for r in failures["seg_bad_dur"])
