import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock

from solomuse_model.intent.dataset import IntentDataset
from solomuse_model.renderer.dataset import RendererDataset
from solomuse_data.config import PipelineConfig

@pytest.fixture
def mock_split_manifest(tmp_path):
    # Create a dummy split manifest
    data = [
        {"track_id": "Track01", "segment_id": "Seg1", "split": "train"},
        {"track_id": "Track01", "segment_id": "Seg2", "split": "train"},
        {"track_id": "Track02", "segment_id": "Seg3", "split": "train"},
        {"track_id": "Track03", "segment_id": "Seg4", "split": "val"},
        {"track_id": "Track04", "segment_id": "Seg5", "split": "test"},
    ]
    df = pd.DataFrame(data)
    
    # We need to simulate the directories for datasets because they might check for existing files
    base_dir = tmp_path / "segments" / "dataset_mock"
    base_dir.mkdir(parents=True)
    manifest_path = base_dir / "manifest_mock_splits.csv"
    df.to_csv(manifest_path, index=False)
    
    # Create mock files so Dataset __getitem__ doesn't fail if strictly required 
    # (they only actually load in __getitem__, but just in case)
    return manifest_path

@pytest.fixture
def mock_missing_split_manifest(tmp_path):
    # Missing split column
    data = [
        {"track_id": "Track01", "segment_id": "Seg1"},
        {"track_id": "Track02", "segment_id": "Seg3"},
    ]
    df = pd.DataFrame(data)
    manifest_path = tmp_path / "manifest_missing.csv"
    df.to_csv(manifest_path, index=False)
    return manifest_path

def test_intent_dataset_respects_splits(mock_split_manifest):
    # IntentDataset inherently relies on being fed rows from build_intent_dataloaders,
    # but let's test how instances reflect counts from the split explicitly if they were loading independently.
    # Actually, IntentDataset takes a list of dicts directly.
    df = pd.read_csv(mock_split_manifest)
    train_rows = df[df["split"] == "train"].to_dict("records")
    val_rows = df[df["split"] == "val"].to_dict("records")
    
    # Passing rows directly
    # Since dataset takes pre-parsed rows we just verify counts and ensure no tracking leaks.
    ds_train = IntentDataset(df[df["split"] == "train"], cfg=PipelineConfig(output_root=""))
    ds_val = IntentDataset(df[df["split"] == "val"], cfg=PipelineConfig(output_root=""))
    
    
    assert len(ds_train) == 3
    assert len(ds_val) == 1
    
def test_renderer_dataset_respects_splits(mock_split_manifest):
    # RendererDataset reads the file itself.
    ds_train = RendererDataset(str(mock_split_manifest), split="train")
    ds_val = RendererDataset(str(mock_split_manifest), split="val")
    ds_test = RendererDataset(str(mock_split_manifest), split="test")
    
    assert len(ds_train) == 3
    assert len(ds_val) == 1
    assert len(ds_test) == 1
    
def test_renderer_dataset_rejects_missing_splits(mock_missing_split_manifest):
    # RendererDataset MUST reject manifests that don't explicitly have a splits column anymore. 
    # To fix leakage, we removed the fallback track hashing.
    with pytest.raises(ValueError, match="lacks a 'split' column"):
        RendererDataset(str(mock_missing_split_manifest), split="train")
