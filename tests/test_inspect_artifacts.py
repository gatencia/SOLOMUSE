import pytest
import numpy as np
from pathlib import Path
import csv
from unittest.mock import patch, MagicMock

from solomuse_data.config import PipelineConfig
from solomuse_data.inspect_artifacts import ArtifactInspector

@pytest.fixture
def mock_cfg():
    return PipelineConfig(
        output_root="/tmp/solomuse_test_output",
        canonical_sample_rate=44100,
        renderer_frame_ms=20.0,
        renderer_hop_ms=10.0
    )

@pytest.fixture
def mock_segment_dir(tmp_path):
    seg_dir = tmp_path / "segments" / "test_dataset" / "Track00007" / "Track00007_264600"
    seg_dir.mkdir(parents=True)
    return tmp_path, seg_dir

def test_missing_file_graceful_handling(mock_cfg, mock_segment_dir, capsys):
    tmp_path, seg_dir = mock_segment_dir
    mock_cfg.output_root = str(tmp_path)
    
    # Intentionally missing all .npy files
    inspector = ArtifactInspector(mock_cfg, "test_dataset")
    inspector.run_sample(sample_size=1)
    
    captured = capsys.readouterr().out
    assert "situation      : MISSING" in captured
    assert "intent_targets : MISSING" in captured
    assert "renderer_target: MISSING" in captured

def test_split_leakage_detection(mock_cfg, tmp_path, caplog):
    mock_cfg.output_root = str(tmp_path)
    seg_root = tmp_path / "segments" / "test_dataset"
    seg_root.mkdir(parents=True)
    
    # Create fake splits with an intentional leak (seg_B is in train and val)
    split_file = seg_root / "manifest_intent_splits.csv"
    with open(split_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["track_id", "segment_id", "split"])
        writer.writeheader()
        writer.writerow({"track_id": "T1", "segment_id": "seg_A", "split": "train"})
        writer.writerow({"track_id": "T1", "segment_id": "seg_B", "split": "train"})
        writer.writerow({"track_id": "T1", "segment_id": "seg_B", "split": "val"})  # LEAK
        writer.writerow({"track_id": "T2", "segment_id": "seg_C", "split": "test"})
        
    with pytest.raises(AssertionError, match="FATAL: ALGORITHMIC TRACK LEAKAGE in manifest_intent_splits.csv!"):
        inspector = ArtifactInspector(mock_cfg, "test_dataset")
        inspector.run_splits()

@patch("solomuse_model.renderer.codec_interface.WaveChunkCodec.decode")
@patch("scipy.io.wavfile.write")
def test_decode_output_written(mock_wav_write, mock_decode, mock_cfg, mock_segment_dir, capsys):
    tmp_path, seg_dir = mock_segment_dir
    mock_cfg.output_root = str(tmp_path)
    
    # Create a fake renderer target
    arr = np.random.randn(10, 882).astype(np.float32)
    np.save(seg_dir / "renderer_target.npy", arr)
    
    # Mock codec decode to return dummy audio
    mock_decode.return_value = np.random.randn(44100)
    
    # We must mock output path resolution
    with patch("solomuse_model.paths.get_renderer_checkpoint_path", return_value="fake.pt"):
        inspector = ArtifactInspector(mock_cfg, "test_dataset")
        assert (seg_dir / "renderer_target.npy").exists()
        inspector.run_decode(sample_size=10)
        
        out_wav = tmp_path / "experiments" / "inspection_decodes" / "decoded_Track00007_264600.wav"
        assert out_wav.exists()
