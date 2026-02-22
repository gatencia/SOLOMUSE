import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import soundfile as sf

from solomuse_data.config import PipelineConfig
from solomuse_data.build_pairs import build_pairs_for_dataset
from solomuse_data.dataset_adapters.base import DatasetAdapter, Track

# Mock Adapter
class MockAdapter(DatasetAdapter):
    def __init__(self, root, cfg):
        super().__init__(root, cfg)
        self.tracks_data = {} # id -> stems dict

    def list_tracks(self):
        return [Track("mock", tid, self.root / tid) for tid in self.tracks_data]

    def list_stems(self, track: Track):
        return self.tracks_data[track.track_id]["stems"]

    def get_mix_path(self, track: Track):
        return self.tracks_data[track.track_id].get("mix")

    def resolve_solo_stems(self, stems):
        # Simplistic: "solo" in name
        return [p for n, p in stems.items() if "solo" in n]

    def resolve_backing_stems(self, stems, solo):
        solo_set = set(solo)
        return [p for p in stems.values() if p not in solo_set]

@pytest.fixture
def mock_dataset_root(tmp_path):
    root = tmp_path / "mock_ds"
    root.mkdir()
    
    # Track 1: Perfect Mix
    t1 = root / "track1"
    t1.mkdir()
    
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Create stems
    # Backing: 220Hz
    backing = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    backing = np.column_stack([backing, backing])
    sf.write(t1 / "backing.wav", backing, sr)
    
    # Solo: 440Hz
    solo = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    solo = np.column_stack([solo, solo])
    sf.write(t1 / "solo.wav", solo, sr)
    
    # Mix
    mix = backing + solo
    sf.write(t1 / "mix.wav", mix, sr)
    
    return root

@pytest.fixture
def cfg(tmp_path, mock_dataset_root):
    return PipelineConfig(
        output_root=str(tmp_path / "output"),
        dataset_roots={"mock": str(mock_dataset_root)},
        solo_stem_policy="lead_any"
    )

def test_build_pairs_perfect_mix(mock_dataset_root, cfg):
    # Setup mock adapter logic
    t1 = mock_dataset_root / "track1"
    tracks_data = {
        "track1": {
            "stems": {
                "backing": t1 / "backing.wav",
                "solo": t1 / "solo.wav"
            },
            "mix": t1 / "mix.wav"
        }
    }
    
    with patch("solomuse_data.build_pairs.get_adapter") as mock_get:
        adapter = MockAdapter(mock_dataset_root, cfg)
        adapter.tracks_data = tracks_data
        mock_get.return_value = adapter
        
        manifest_path = build_pairs_for_dataset("mock", cfg)
        
        assert manifest_path.exists()
        
        # Check output files
        out_dir = Path(cfg.output_root) / "pairs" / "mock" / "track1"
        assert (out_dir / "x_backing.wav").exists()
        assert (out_dir / "y_solo.wav").exists()
        assert (out_dir / "meta.json").exists()
        
        # Verify mix consistency in meta
        import json
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
            metrics = meta["mix_metrics"]
            assert metrics is not None
            assert metrics["mse"] < 1e-5 # Should be very close to 0
            assert metrics["corr"] > 0.99

def test_build_pairs_skip_no_slolo(mock_dataset_root, cfg):
    # Track 2: No solo
    t2 = mock_dataset_root / "track2"
    t2.mkdir()
    sf.write(t2 / "backing.wav", np.zeros((44100, 2)), 44100)
    
    tracks_data = {
        "track2": {
            "stems": {"backing": t2 / "backing.wav"},
            "mix": None
        }
    }
    
    with patch("solomuse_data.build_pairs.get_adapter") as mock_get:
        adapter = MockAdapter(mock_dataset_root, cfg)
        adapter.tracks_data = tracks_data
        mock_get.return_value = adapter
        
        build_pairs_for_dataset("mock", cfg)
        
        out_dir = Path(cfg.output_root) / "pairs" / "mock" / "track2"
        assert not out_dir.exists() # Should be skipped

def test_broken_mix_detection(mock_dataset_root, cfg):
    # Track 3: Mix != Sum
    t3 = mock_dataset_root / "track3"
    t3.mkdir()
    sr = 44100
    
    backing = np.zeros((sr, 2), dtype=np.float32)
    solo = np.ones((sr, 2), dtype=np.float32) * 0.5
    mix = np.zeros((sr, 2), dtype=np.float32) # Wrong mix (silence)
    
    sf.write(t3 / "backing.wav", backing, sr)
    sf.write(t3 / "solo.wav", solo, sr)
    sf.write(t3 / "mix.wav", mix, sr)
    
    tracks_data = {
        "track3": {
            "stems": {"backing": t3 / "backing.wav", "solo": t3 / "solo.wav"},
            "mix": t3 / "mix.wav"
        }
    }
    
    with patch("solomuse_data.build_pairs.get_adapter") as mock_get:
        adapter = MockAdapter(mock_dataset_root, cfg)
        adapter.tracks_data = tracks_data
        mock_get.return_value = adapter
        
        build_pairs_for_dataset("mock", cfg)
        
        out_dir = Path(cfg.output_root) / "pairs" / "mock" / "track3"
        import json
        with open(out_dir / "meta.json") as f:
            meta = json.load(f)
            metrics = meta["mix_metrics"]
            # MSE should be high (0.5^2 = 0.25 roughly)
            assert metrics["mse"] > 0.01 
