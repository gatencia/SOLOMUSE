import pytest
import numpy as np
from pathlib import Path
import csv
from unittest.mock import patch, MagicMock
import torch
import pandas as pd

from solomuse_data.config import PipelineConfig
from solomuse_model.intent.train import run_train_intent
from solomuse_model.intent.infer import IntentInferencer

def test_paths_helper_resolves_consistently():
    cfg = PipelineConfig(output_root="/tmp/s", intent_model_version="v1")
    from solomuse_model.paths import get_intent_checkpoint_path
    p = get_intent_checkpoint_path(cfg)
    assert "/tmp/s/checkpoints/intent/model_v1.pt" in str(p)

def test_intent_inferencer_initialization_with_none_path(tmp_path):
    cfg = PipelineConfig(output_root=str(tmp_path))
    # Should not crash if path is None
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.load", return_value={"model_state_dict": {}}):
            with patch("solomuse_model.intent.model_v1.IntentPlannerGRU_V1.load_state_dict"):
                IntentInferencer(cfg)

def test_intent_train_nan_dataset_raises(tmp_path):
    cfg = PipelineConfig(output_root=str(tmp_path), intent_epochs=1, intent_batch_size=2)
    
    # Ensure manifest exists for initialization
    seg_dir = tmp_path / "segments" / "test_nan_set"
    seg_dir.mkdir(parents=True)
    manifest_path = seg_dir / "manifest_intent.csv"
    manifest_path.write_text("dataset,track_id,segment_id,intent_version,intent_hz,intent_frames,split\ntest_nan_set,T1,S1,v1,10.0,60,train\n")
    
    # Fake dataset that yields NaNs
    fake_train_ds = MagicMock()
    fake_train_ds.__len__.return_value = 5
    
    # Create tensors with NaNs
    nan_x = torch.full((32,), float('nan'))
    nan_y = torch.full((60, 7), float('nan'))
    
    fake_batch = {
        "situation": nan_x.unsqueeze(0), 
        "intent": nan_y.unsqueeze(0),
        "segment_id": ["S1"],
        "track_id": ["T1"]
    }
    
    # Use lists directly for loaders, they are iterables
    mock_train_loader = [fake_batch]
    mock_train_loader.dataset = fake_train_ds
    mock_val_loader = []
    mock_val_loader.dataset = MagicMock()

    # Mock the dataloaders
    with patch("torch.utils.data.DataLoader", side_effect=[mock_train_loader, mock_val_loader, []]):
        # Mock dataset instantiation so it doesn't fail parsing an empty manifest
        with patch("solomuse_model.intent.dataset.IntentDataset", return_value=fake_train_ds):
            # Also mock the manifest check
            with patch("pathlib.Path.exists", return_value=True):
                # We need to mock create_track_grouped_splits since it validates the CSV physically
                with patch("solomuse_model.utils.splits.create_track_grouped_splits", side_effect=lambda manifest, *args, **kwargs: pd.read_csv(manifest)):
                    with pytest.raises(RuntimeError, match="contains NaN/Inf"):
                        run_train_intent(cfg, "test_nan_set")
