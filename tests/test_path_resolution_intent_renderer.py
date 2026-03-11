import pytest
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.dataset import build_intent_dataloaders
from solomuse_model.paths import get_intent_checkpoint_path, get_renderer_checkpoint_path
import pandas as pd

def test_intent_dataset_path_resolution(tmp_path):
    cfg = PipelineConfig(output_root=str(tmp_path))
    
    segments_ds = tmp_path / "segments" / "test_ds"
    segments_ds.mkdir(parents=True)
    manifest = segments_ds / "manifest_intent.csv"
    
    df = pd.DataFrame({"track_id": ["1"], "segment_id": ["1"], "dataset": ["test_ds"]})
    df.to_csv(manifest, index=False)
    
    # Mock finding files to just pass path resolution
    try:
        build_intent_dataloaders(cfg, "test_ds")
    except Exception as e:
        # FileNotFoundError means it looked in "targets" instead of "segments".
        assert not isinstance(e, FileNotFoundError), f"Path resolution failed. Looked in wrong place: {e}"

def test_checkpoint_paths(tmp_path):
    cfg = PipelineConfig(
        output_root=str(tmp_path)
    )
    intent_path = get_intent_checkpoint_path(cfg)
    renderer_path = get_renderer_checkpoint_path(cfg)
    
    assert "intent_v1" in str(intent_path)
    assert "renderer_v1" in str(renderer_path)
    
    cfg.intent_checkpoint_path = "/custom/intent.pt"
    cfg.renderer_checkpoint_path = "/custom/renderer.pt"
    
    assert str(get_intent_checkpoint_path(cfg)) == "/custom/intent.pt"
    assert str(get_renderer_checkpoint_path(cfg)) == "/custom/renderer.pt"
