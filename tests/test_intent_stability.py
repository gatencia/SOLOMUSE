import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import patch, MagicMock
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.dataset import IntentDataset, build_intent_dataloaders
from solomuse_model.intent.train import run_train_intent, tensor_stats_dict
from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1

@pytest.fixture
def mock_cfg(tmp_path):
    cfg = PipelineConfig(output_root=str(tmp_path))
    cfg.intent_batch_size = 2
    cfg.intent_epochs = 1
    cfg.intent_hidden_dim = 16
    cfg.intent_num_layers = 1
    # Stability specific defaults
    cfg.intent_skip_bad_batches = False
    cfg.intent_fail_on_nonfinite_input = True
    cfg.intent_fail_on_nonfinite_target = True
    cfg.intent_force_cpu_debug = True
    return cfg

@pytest.fixture
def mock_manifest_df():
    data = {
        "track_id": ["track_01", "track_02", "track_03"],
        "segment_id": ["seg_01", "seg_02", "seg_03"],
        "dataset": ["mock", "mock", "mock"],
        "split": ["train", "train", "train"]
    }
    return pd.DataFrame(data)

def test_config_backward_compatibility():
    """Test old defaults safely load Without breaking the new pipeline"""
    cfg = PipelineConfig(output_root="dummy")
    assert hasattr(cfg, 'intent_skip_bad_batches')
    assert hasattr(cfg, 'intent_max_bad_batches_per_epoch')
    assert cfg.intent_weight_decay == 1e-5
    assert not cfg.intent_force_cpu_debug

def test_finite_data_guard_raises_on_input(mock_cfg, mock_manifest_df, tmp_path):
    # Setup mock data dirs
    seg_dir = tmp_path / "segments" / "mock" / "track_01" / "seg_01"
    seg_dir.mkdir(parents=True)
    
    # Write NaN sit_vec
    sit_vec = np.ones(32, dtype=np.float32)
    sit_vec[0] = np.nan
    np.save(seg_dir / "situation.npy", sit_vec)
    
    # Write valid intent_targets
    intent_mat = np.ones((60, 7), dtype=np.float32)
    np.save(seg_dir / "intent_targets.npy", intent_mat)
    
    dataset = IntentDataset(mock_manifest_df.iloc[[0]], mock_cfg)
    
    # Assert ValueError
    with pytest.raises(ValueError, match="Non-finite values \\(NaN/Inf\\) detected in Situation input array"):
        _ = dataset[0]

def test_tensor_stats_dict():
    # Helper should correctly count nans
    t = torch.tensor([1.0, 2.0, float('nan'), float('inf')])
    stats = tensor_stats_dict(t, "test_tensor")
    assert stats["name"] == "test_tensor"
    assert stats["is_finite"] is False
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1

def test_metadata_propagation(mock_cfg, mock_manifest_df, tmp_path):
    seg_dir = tmp_path / "segments" / "mock" / "track_01" / "seg_01"
    seg_dir.mkdir(parents=True)
    np.save(seg_dir / "situation.npy", np.zeros(32, dtype=np.float32))
    np.save(seg_dir / "intent_targets.npy", np.zeros((60, 7), dtype=np.float32))
    
    dataset = IntentDataset(mock_manifest_df.iloc[[0]], mock_cfg)
    item = dataset[0]
    
    assert "situation" in item
    assert "intent" in item
    assert item["track_id"] == "track_01"
    assert item["segment_id"] == "seg_01"
    
@patch("solomuse_model.intent.train.build_intent_dataloaders")
def test_batch_diagnostics_artifact_generation(mock_build, mock_cfg, tmp_path):
    # Dataloader mocks
    mock_batch = {
        "situation": torch.zeros(2, 60, 32),
        "intent": torch.full((2, 60, 7), float("nan")), # Inject NaN here
        "segment_id": ["seg_01", "seg_02"],
        "track_id": ["track_01", "track_02"]
    }
    mock_loader = [mock_batch]
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__getitem__.return_value = {
        "situation": torch.zeros(60, 32),
        "intent": torch.zeros(60, 7),
        "segment_id": "seg_01",
        "track_id": "track_01"
    }
    mock_loader_obj = MagicMock()
    mock_loader_obj.__iter__.side_effect = lambda: iter(mock_loader)
    mock_loader_obj.dataset = mock_dataset
    mock_loader_obj.__len__.return_value = 1
    
    # Mock return
    mock_build.return_value = (mock_loader_obj, mock_loader_obj, mock_loader_obj)
    
    # Run train expecting fatal RuntimeError
    with pytest.raises(RuntimeError):
        run_train_intent(mock_cfg, "mock_dataset")
        
    # Verify artifacts were generated
    debug_dir = tmp_path / "experiments" / "intent" / "debug_crashes"
    assert debug_dir.exists()
    
    json_files = list(debug_dir.glob("*.json"))
    assert len(json_files) == 1
    
    csv_file = debug_dir / "bad_batches.csv"
    assert csv_file.exists()
    
    df = pd.read_csv(csv_file)
    assert len(df) == 1
    assert df.iloc[0]["stage"] == "Pre-Forward"
    assert "seg_01|seg_02" in df.iloc[0]["segment_ids"]

@patch("solomuse_model.intent.train.build_intent_dataloaders")
def test_skip_bad_batch_debug_mode(mock_build, mock_cfg):
    mock_cfg.intent_skip_bad_batches = True
    mock_cfg.intent_epochs = 1
    
    mock_batch_good = {
        "situation": torch.zeros(2, 60, 32),
        "intent": torch.zeros(2, 60, 7),
        "segment_id": ["seg_01", "seg_02"],
        "track_id": ["track_01", "track_02"]
    }
    
    mock_batch_bad = {
        "situation": torch.zeros(2, 60, 32),
        "intent": torch.full((2, 60, 7), float("nan")),
        "segment_id": ["seg_03", "seg_04"],
        "track_id": ["track_03", "track_04"]
    }
    
    # 2 good, 1 bad batch so training_steps > 0 completes
    mock_loader = [mock_batch_good, mock_batch_bad, mock_batch_good]
    
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 6
    mock_dataset.__getitem__.return_value = {
        "situation": torch.zeros(60, 32),
        "intent": torch.zeros(60, 7),
        "segment_id": "seg_01",
        "track_id": "track_01"
    }
    mock_loader_obj = MagicMock()
    mock_loader_obj.__iter__.side_effect = lambda: iter(mock_loader)
    mock_loader_obj.dataset = mock_dataset
    mock_loader_obj.__len__.return_value = 3
    
    mock_build.return_value = (mock_loader_obj, mock_loader_obj, mock_loader_obj)
    
    # Should not raise exception
    try:
        run_train_intent(mock_cfg, "mock_dataset")
    except RuntimeError as e:
        pytest.fail(f"Training crashed despite skip_bad_batches being enabled! {e}")

from solomuse_model.intent.train import collect_grad_stats
import torch.nn as nn

def test_collect_grad_stats_nan():
    model = nn.Linear(10, 2)
    x = torch.randn(1, 10)
    y = model(x)
    y.sum().backward()
    
    # Poison the weight gradient
    model.weight.grad[0, 0] = float("nan")
    
    stats = collect_grad_stats(model)
    assert not stats["all_finite_pre_clip"]
    assert stats["first_bad_param"] == "weight"
    assert stats["bad_param_count"] == 1

@patch("solomuse_model.intent.train.build_intent_dataloaders")
@patch("solomuse_model.intent.train.collect_grad_stats")
def test_backward_gradients_diagnostic_hook(mock_collect, mock_build, mock_cfg):
    mock_batch = {
        "situation": torch.zeros(2, 60, 32),
        "intent": torch.zeros(2, 60, 7),
        "segment_id": ["seg_01", "seg_02"],
        "track_id": ["track_01", "track_02"]
    }
    mock_loader = [mock_batch]
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__getitem__.return_value = {
        "situation": torch.zeros(60, 32),
        "intent": torch.zeros(60, 7),
        "segment_id": "seg_01",
        "track_id": "track_01"
    }
    mock_loader_obj = MagicMock()
    mock_loader_obj.__iter__.side_effect = lambda: iter(mock_loader)
    mock_loader_obj.dataset = mock_dataset
    mock_loader_obj.__len__.return_value = 1
    mock_build.return_value = (mock_loader_obj, mock_loader_obj, mock_loader_obj)
    
    mock_collect.return_value = {
        "all_finite_pre_clip": False,
        "first_bad_param": "gru.weight_ih_l0",
        "bad_param_count": 1,
        "param_count": 10,
        "none_grad_count": 0,
        "params": {}
    }
    
    with pytest.raises(RuntimeError, match="Training crashed at Epoch 1 Batch 0: Parameter gradients became NaN/Inf. First bad param: gru.weight_ih_l0"):
        run_train_intent(mock_cfg, "mock_dataset")

@patch("solomuse_model.intent.train.build_intent_dataloaders")
@patch("torch.nn.utils.clip_grad_norm_")
def test_optimizer_step_skip_on_large_grad_norm(mock_clip, mock_build, mock_cfg):
    mock_cfg.intent_skip_optimizer_step_on_large_grad_norm = True
    mock_cfg.intent_large_grad_norm_threshold = 50.0
    
    mock_batch = {
        "situation": torch.zeros(2, 60, 32),
        "intent": torch.zeros(2, 60, 7),
        "segment_id": ["seg_01", "seg_02"],
        "track_id": ["track_01", "track_02"]
    }
    mock_loader = [mock_batch]
    mock_dataset = MagicMock()
    mock_dataset.__len__.return_value = 2
    mock_dataset.__getitem__.return_value = {
        "situation": torch.zeros(60, 32),
        "intent": torch.zeros(60, 7),
        "segment_id": "seg_01",
        "track_id": "track_01"
    }
    mock_loader_obj = MagicMock()
    mock_loader_obj.__iter__.side_effect = lambda: iter(mock_loader)
    mock_loader_obj.dataset = mock_dataset
    mock_loader_obj.__len__.return_value = 1
    mock_build.return_value = (mock_loader_obj, mock_loader_obj, mock_loader_obj)
    
    # Return a massive, but finite, norm
    mock_clip.return_value = torch.tensor(150.0)
    
    with patch("torch.optim.AdamW.step") as mock_step:
        run_train_intent(mock_cfg, "mock_dataset")
        # Step should be skipped due to norm > 50.0
        mock_step.assert_not_called()
