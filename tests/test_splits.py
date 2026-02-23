import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from solomuse_data.config import PipelineConfig
from solomuse_model.utils.splits import create_track_grouped_splits

@pytest.fixture
def mock_cfg():
    cfg = PipelineConfig(output_root="dummy", seed=42)
    cfg.intent_train_ratio = 0.6
    cfg.intent_val_ratio = 0.2
    cfg.intent_test_ratio = 0.2
    return cfg

@pytest.fixture
def synthetic_manifest_path(tmp_path):
    # 10 tracks, each with 5 segments = 50 segments
    rows = []
    for t in range(10):
        track_id = f"Track{t:03d}"
        for s in range(5):
            seg_id = f"{track_id}_seg{s}"
            rows.append({"track_id": track_id, "segment_id": seg_id})
            
    df = pd.DataFrame(rows)
    p = tmp_path / "manifest_test.csv"
    df.to_csv(p, index=False)
    return p

def test_track_grouped_split_distribution(synthetic_manifest_path, mock_cfg, tmp_path):
    out_path = tmp_path / "splits_out.csv"
    
    df = create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg)
    
    assert "split" in df.columns
    # 10 tracks * 0.6 = 6 train tracks * 5 segs = 30 segs
    # 10 tracks * 0.2 = 2 val tracks * 5 segs = 10 segs 
    # 10 tracks * 0.2 = 2 test tracks * 5 segs = 10 segs
    
    counts = df["split"].value_counts()
    assert counts["train"] == 30
    assert counts["val"] == 10
    assert counts["test"] == 10

def test_track_grouped_split_leakage_assertion(synthetic_manifest_path, mock_cfg, tmp_path):
    out_path = tmp_path / "splits_out.csv"
    df = create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg)
    
    # Assert NO leakage naturally generated
    track_splits = df.groupby("track_id")["split"].nunique()
    assert (track_splits == 1).all()

    # Manually poison the file to test detection mechanism on reload
    poisoned_df = df.copy()
    # Force Track000's first segment into test, while the rest are in its original split
    poisoned_df.loc[0, "split"] = "test" 
    poisoned_df.to_csv(out_path, index=False)
    
    # Run again without force_regenerate, it should load and crash
    with pytest.raises(RuntimeError, match="Data leakage detected"):
        create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg, force_regenerate=False)

def test_deterministic_reproducibility(synthetic_manifest_path, mock_cfg, tmp_path):
    out1 = tmp_path / "out1.csv"
    out2 = tmp_path / "out2.csv"
    
    df1 = create_track_grouped_splits(synthetic_manifest_path, out1, mock_cfg)
    df2 = create_track_grouped_splits(synthetic_manifest_path, out2, mock_cfg)
    
    assert list(df1["split"]) == list(df2["split"])
    
def test_old_split_detection_force_override(synthetic_manifest_path, mock_cfg, tmp_path):
    out_path = tmp_path / "splits_out.csv"
    df = create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg)
    
    # Poison the file
    df.loc[0, "split"] = "test" 
    df.to_csv(out_path, index=False)
    
    # Should throw without flag
    with pytest.raises(RuntimeError):
        create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg, force_regenerate=False)
        
    # Should successfully regenerate With flag
    df_fixed = create_track_grouped_splits(synthetic_manifest_path, out_path, mock_cfg, force_regenerate=True)
    track_splits = df_fixed.groupby("track_id")["split"].nunique()
    assert (track_splits == 1).all()
