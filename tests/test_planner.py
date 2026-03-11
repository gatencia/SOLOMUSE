import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.infer import IntentInferencer
from solomuse_model.intent.dataset import IntentDataset
from solomuse_model.pipeline import SoloMusePipeline

@pytest.fixture
def mock_dataset(tmp_path):
    output_root = tmp_path / "processed_planner"
    seg_dir = output_root / "segments" / "mock" / "track1" / "seg1"
    seg_dir.mkdir(parents=True)
    
    # 32 dims situation
    sit_vec = np.random.rand(32).astype(np.float32)
    np.save(seg_dir / "situation.npy", sit_vec)
    
    # 60 frames (6s at 10Hz), 7 dims intent
    intent_mat = np.random.rand(60, 7).astype(np.float32)
    np.save(seg_dir / "intent_targets.npy", intent_mat)
    
    manifest_dir = output_root / "segments" / "mock"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest_intent.csv"
    
    df = pd.DataFrame([{
        "segment_id": "seg1",
        "dataset": "mock",
        "track_id": "track1"
    }])
    df.to_csv(manifest_path, index=False)
    
    return manifest_path

def test_dataset_loads_shapes(mock_dataset):
    cfg = PipelineConfig(output_root=str(Path(mock_dataset).parent.parent.parent))
    df = pd.read_csv(mock_dataset)
    ds = IntentDataset(df, cfg)
    assert len(ds) == 1
    
    batch = ds[0]
    X, Y = batch["situation"], batch["intent"]
    
    # X should be Broadcasted to F: [60, 32]
    assert X.shape == (60, 32)
    # Y should be [60, 7]
    assert Y.shape == (60, 7)

def test_model_forward_shape():
    model = IntentPlannerGRU_V1(input_dim=32, hidden_dim=64, num_layers=1, output_dim=7)
    
    B, F, D_in = 2, 60, 32
    X = torch.randn(B, F, D_in)
    
    preds = model(X)
    
    assert preds.shape == (B, F, 7)
    # Sigmoid check
    assert torch.all(preds >= 0.0) and torch.all(preds <= 1.0)

def test_train_one_step_runs(mock_dataset):
    cfg = PipelineConfig(output_root=str(Path(mock_dataset).parent.parent.parent))
    df = pd.read_csv(mock_dataset)
    ds = IntentDataset(df, cfg)
    loader = torch.utils.data.DataLoader(ds, batch_size=1)
    
    model = IntentPlannerGRU_V1(input_dim=32, hidden_dim=64, num_layers=1, output_dim=7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    batch = next(iter(loader))
    X, Y = batch["situation"], batch["intent"]
    
    # Initial weights
    initial_weight = model.linear.weight.clone()
    
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, Y)
    loss.backward()
    optimizer.step()
    
    # Check weight update
    assert not torch.allclose(initial_weight, model.linear.weight)

def test_infer_returns_expected_shape():
    cfg = PipelineConfig(
        output_root="/tmp/fake_root",
        intent_hidden_dim=64,
        intent_num_layers=1
    )
    # passing no checkpoint sets ready=True for uninitialized inference.
    from unittest.mock import patch
    with patch("pathlib.Path.exists", return_value=True):
        with patch("torch.load", return_value={"model_state_dict": {}}):
            with patch("solomuse_model.intent.model_v1.IntentPlannerGRU_V1.load_state_dict"):
                inferencer = IntentInferencer(cfg, checkpoint_path=None)
    
    # single segment simulation
    sit_vec = np.random.rand(32).astype(np.float32)
    frames = 45 # 4.5s
    
    preds = inferencer.predict_sequence(sit_vec, num_frames=frames)
    
    assert type(preds) == np.ndarray
    assert preds.shape == (frames, 7)
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)

def test_pipeline_plan_intent_stub_or_model_path():
    cfg = PipelineConfig(
        output_root="/tmp/fake_root",
        intent_hz=10,
        intent_checkpoint_path=None
    )
    pipeline = SoloMusePipeline(cfg)
    
    # Should yield zero array
    sit_vec = np.zeros(32, dtype=np.float32)
    preds = pipeline.plan_intent(sit_vec, duration_s=1.0)
    
    assert preds.shape == (10, 7)
    assert np.all(preds == 0.0)
