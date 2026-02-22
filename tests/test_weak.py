import pytest
import shutil
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch
import soundfile as sf
from solomuse_data.config import PipelineConfig
from solomuse_data.weak_separation import generate_weak_data

@pytest.fixture
def mock_output_root(tmp_path):
    root = tmp_path / "mock_root"
    root.mkdir()
    
    # Check inputs exists
    input_dir = root / "weak_inputs"
    input_dir.mkdir()
    
    # Create a dummy song
    sr = 44100
    audio = np.zeros((44100, 2), dtype=np.float32)
    sf.write(input_dir / "mysong.mp3", audio, sr)
    
    return root

@pytest.fixture
def cfg(mock_output_root):
    return PipelineConfig(
        output_root=str(mock_output_root),
        enable_weak_demucs=True,
        dataset_roots={"mock": "dummy"}
    )

def test_generate_weak_skip_disabled():
    cfg = PipelineConfig(output_root="/tmp", enable_weak_demucs=False)
    with patch("solomuse_data.weak_separation.shutil.which") as mock_which:
        generate_weak_data(cfg)
        mock_which.assert_not_called()

def test_generate_weak_logic(mock_output_root, cfg):
    # Mock shutil.which to say demucs exists
    # Mock subprocess.run to simulate success
    # Mock finding output files? 
    # The code looks for files in output_root/demucs_temp/...
    # We can create those files "simulated" by the mock side effect? 
    # Or just create them in the fixture before running, assuming we know where it *looks*.
    # But subprocess runs FIRST. Logic sequence:
    # 1. subprocess.run -> creates files (we mock this action)
    # 2. glob checks for files
    
    demucs_temp = mock_output_root / "demucs_temp"
    model_dir = demucs_temp / "htdemucs" / "mysong"
    
    def mock_subprocess(*args, **kwargs):
        # Simulate demucs creating files
        model_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        audio = np.zeros((44100, 2), dtype=np.float32)
        
        sf.write(model_dir / "vocals.wav", audio, sr)
        sf.write(model_dir / "drums.wav", audio, sr)
        sf.write(model_dir / "bass.wav", audio, sr)
        sf.write(model_dir / "other.wav", audio, sr)
        
        return MagicMock(returncode=0)
        
    with patch("solomuse_data.weak_separation.shutil.which", return_value="/usr/bin/demucs"), \
         patch("subprocess.run", side_effect=mock_subprocess) as mock_run:
         
         generate_weak_data(cfg)
         
         # Assert output exists
         out_pair = mock_output_root / "weak_pairs" / "mysong"
         assert out_pair.exists()
         assert (out_pair / "x_backing.wav").exists()
         assert (out_pair / "y_solo.wav").exists()
         assert (out_pair / "meta.json").exists()
         
         # Check meta
         import json
         with open(out_pair / "meta.json") as f:
             meta = json.load(f)
             assert meta["weak_label"] is True
             assert meta["solo_source_stem"] == "vocals.wav"
             assert meta["separator"] == "demucs"

def test_generate_weak_skip_no_solo(mock_output_root, cfg):
        
    demucs_temp = mock_output_root / "demucs_temp"
    model_dir = demucs_temp / "htdemucs" / "mysong"
    
    def mock_subprocess(*args, **kwargs):
        model_dir.mkdir(parents=True, exist_ok=True)
        sr = 44100
        audio = np.zeros((100, 2), dtype=np.float32)
        # Create files WITHOUT vocals/guitar/lead/melody
        sf.write(model_dir / "drums.wav", audio, sr)
        sf.write(model_dir / "bass.wav", audio, sr)
        sf.write(model_dir / "other.wav", audio, sr)
        return MagicMock(returncode=0)
        
    with patch("solomuse_data.weak_separation.shutil.which", return_value="/usr/bin/demucs"), \
         patch("subprocess.run", side_effect=mock_subprocess):
         
         generate_weak_data(cfg)
         
         # Should skip
         out_pair = mock_output_root / "weak_pairs" / "mysong"
         assert not out_pair.exists()
