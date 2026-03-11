import os
import json
import csv
import numpy as np
import pytest
import tempfile
import soundfile as sf
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_data.inspection.unified import UnifiedArtifactExporter

@pytest.fixture
def mock_report_env():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = PipelineConfig(
            output_root=str(tmp_path),
            intent_hz=10.0
        )
        
        dataset = "mock_data"
        seg_root = tmp_path / "segments" / dataset
        track_id = "TrackA"
        seg_id = "TrackA_0"
        seg_dir = seg_root / track_id / seg_id
        seg_dir.mkdir(parents=True)
        
        # 1. Create Manifest
        manifest_path = seg_root / "manifest.csv"
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "track_id", "segment_id", "start_s", "end_s", "duration_s", "x_path", "y_path", "split"])
            writer.writeheader()
            writer.writerow({
                "dataset": dataset,
                "track_id": track_id,
                "segment_id": seg_id,
                "start_s": 0.0,
                "end_s": 6.0,
                "x_path": "x.wav",
                "y_path": "y.wav",
                "split": "train"
            })
            
        # 2. Create Splits
        split_path = seg_root.parent / "manifest_intent_splits.csv"
        with open(split_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["segment_id", "track_id", "split"])
            writer.writeheader()
            writer.writerow({"segment_id": seg_id, "track_id": track_id, "split": "train"})
            
        # 3. Create Artifacts
        # Situation: [32]
        np.save(seg_dir / "situation.npy", np.random.randn(32).astype(np.float32))
        # Intent Targets: [60, 7] - all ones
        np.save(seg_dir / "intent_targets.npy", np.ones((60, 7)).astype(np.float32))
        # Intent Pred: [60, 7] - all zeros (so MSE should be 1.0)
        np.save(seg_dir / "intent_pred.npy", np.zeros((60, 7)).astype(np.float32))
        # Renderer Target: [600, 256] -> all zero
        np.save(seg_dir / "renderer_target.npy", np.zeros((600, 256)).astype(np.float32))
        
        # 4. Create Audio (x, y, y_hat)
        # x: 1s of 441hz sine wave
        t = np.linspace(0, 1, cfg.canonical_sample_rate)
        x_audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        sf.write(tmp_path / "x.wav", x_audio, cfg.canonical_sample_rate)
        # y: ground truth solo (0.1 gain)
        sf.write(tmp_path / "y.wav", 0.1 * x_audio, cfg.canonical_sample_rate)
        # y_hat: predicted solo (0.05 gain)
        sf.write(seg_dir / "y_hat.wav", 0.05 * x_audio, cfg.canonical_sample_rate)
        # Update manifest to point to these
        with open(manifest_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["dataset", "track_id", "segment_id", "start_s", "end_s", "duration_s", "x_path", "y_path", "split"])
            writer.writeheader()
            writer.writerow({
                "dataset": dataset,
                "track_id": track_id,
                "segment_id": seg_id,
                "start_s": 0.0,
                "end_s": 1.0,
                "duration_s": 1.0,
                "x_path": str(tmp_path / "x.wav"),
                "y_path": str(tmp_path / "y.wav"),
                "split": "train"
            })

        yield cfg, dataset, seg_id, seg_dir

def test_unified_export_comprehensive(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    exporter = UnifiedArtifactExporter(cfg, dataset)
    
    report = exporter.run_export(limit=1)
    row = report[0]
    
    # 1. Existence and shapes
    assert row["has_situation"] == 1
    assert row["has_intent_targets"] == 1
    assert row["has_y_hat"] == 1
    
    # 2. Audio Metrics
    # X Peak was 0.5
    assert abs(row["x_peak"] - 0.5) < 1e-4
    # Y RMS should be lower than X
    assert row["y_rms"] < row["x_rms"]
    assert row["y_hat_rms"] > 0
    
    # 3. Intent Error Metrics
    # target=1s, pred=0s -> MSE=1.0
    assert abs(row["intent_mse"] - 1.0) < 1e-4
    
    # 4. Renderer Quality
    # GT=0.1, Pred=0.05. Noise = 0.05. Signal=0.1.
    # Signal Power = 0.01, Noise Power = 0.0025.
    # SNR = 10 * log10(4) ~= 6.02
    assert row["y_hat_snr"] > 0
    assert abs(row["y_hat_snr"] - 6.02) < 0.1
    
    print(f"Verified enhanced metrics: SNR={row['y_hat_snr']}, MSE={row['intent_mse']}")

def test_unified_export_basic(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    exporter = UnifiedArtifactExporter(cfg, dataset)
    
    report = exporter.run_export(limit=1)
    
    assert len(report) == 1
    row = report[0]
    
    assert row["segment_id"] == seg_id
    assert row["split"] == "train"
    assert row["has_situation"] == 1
    assert row["has_intent_targets"] == 1
    assert row["intent_targets_shape"] == "(60, 7)"
    assert row["intent_targets_min"] == 1.0
    assert row["intent_targets_max"] == 1.0
    
    # Alignment checks
    assert row["expected_intent_frames"] == 60
    assert row["actual_intent_frames"] == 60
    assert row["alignment_ok"] == 1
    
    # Zero check
    assert row["renderer_target_all_zero"] == 1
    assert "All-zero renderer target" in row["notes"]

def test_unified_export_missing_files(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    # Delete some artifacts
    (seg_dir / "situation.npy").unlink()
    
    exporter = UnifiedArtifactExporter(cfg, dataset)
    report = exporter.run_export()
    
    row = report[0]
    assert row["has_situation"] == 0
    assert row["situation_path"] == ""
    assert "Missing situation" in row["notes"]
    assert row["alignment_ok"] == 1 # Intent still matches

def test_unified_export_alignment_error(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    # Overwrite intent with wrong shape
    np.save(seg_dir / "intent_targets.npy", np.ones((30, 7)).astype(np.float32))
    
    exporter = UnifiedArtifactExporter(cfg, dataset)
    report = exporter.run_export()
    
    row = report[0]
    assert row["actual_intent_frames"] == 30
    assert row["alignment_ok"] == 0
    assert "Intent frame mismatch" in row["notes"]

def test_split_load_correct_path(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    exporter = UnifiedArtifactExporter(cfg, dataset)
    
    # Verify split is loaded (should be 'train' from mock_report_env)
    report = exporter.run_export(limit=1)
    assert report[0]["split"] == "train"

def test_alignment_ok_missing_artifacts(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    exporter = UnifiedArtifactExporter(cfg, dataset)
    
    # 1. Missing situation
    (seg_dir / "situation.npy").unlink()
    report = exporter.run_export(limit=1)
    assert report[0]["alignment_ok"] == 0
    assert "Missing situation.npy" in report[0]["notes"]
    
    # 2. Reset and missing intent
    np.save(seg_dir / "situation.npy", np.zeros(32))
    (seg_dir / "intent_targets.npy").unlink()
    report = exporter.run_export(limit=1)
    assert report[0]["alignment_ok"] == 0
    assert "Missing intent_targets.npy" in report[0]["notes"]

def test_silence_detection_synthetic(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    exporter = UnifiedArtifactExporter(cfg, dataset)
    
    # Create absolute silence
    silent_wav = Path(cfg.output_root) / "absolute_silence.wav"
    sf.write(silent_wav, np.zeros(1000), cfg.canonical_sample_rate)
    
    # Update manifest to point to it
    import pandas as pd
    manifest_path = Path(exporter.segment_root) / "manifest.csv"
    df = pd.read_csv(manifest_path)
    if len(df) > 0:
        df.loc[0, "y_path"] = str(silent_wav)
        df.to_csv(manifest_path, index=False)
        
    report = exporter.run_export(limit=1, action="silence-audit", rms_threshold=0.01)
    assert report[0]["y_is_silent"] == 1
    assert report[0]["y_rms"] == 0.0

def test_sanity_triples_balanced(mock_report_env):
    cfg, dataset, seg_id, seg_dir = mock_report_env
    # Add another segment in a different split
    seg2_id = "TrackA_1"
    seg2_dir = seg_dir.parent / seg2_id
    seg2_dir.mkdir()
    np.save(seg2_dir / "situation.npy", np.zeros(32))
    
    manifest_path = seg_dir.parent / "manifest.csv"
    with open(manifest_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([dataset, "TrackA", seg2_id, 1.0, 2.0, 1.0, "x.wav", "y.wav"])
        
    split_path = seg_dir.parent / "manifest_intent_splits.csv"
    with open(split_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([seg2_id, "TrackA", "test"])
        
    exporter = UnifiedArtifactExporter(cfg, dataset)
    out_dir = Path(cfg.output_root) / "balanced_triples"
    # Sample 2 segments, balanced (should pick 1 train, 1 test)
    exporter.run_export(action="sanity-triples", output_path=str(out_dir), num_samples=2, balanced_by_split=True)
    
    assert (out_dir / seg_id).exists()
    assert (out_dir / seg2_id).exists()
