import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Force lock CPU for tests to prevent random hardware Segmentation Faults on Apple Silicon Native
if hasattr(torch.backends, 'mps'):
    torch.backends.mps.is_available = lambda: False
    torch.backends.mps.is_built = lambda: False
from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1
from solomuse_model.intent.model_v2 import IntentPlannerTransformer_V2
from solomuse_data.config import PipelineConfig
from solomuse_model.intent.infer import IntentInferencer
from torch import optim, nn

def test_intent_gru_v1_regression():
    model = IntentPlannerGRU_V1()
    B, T, D = 2, 60, 32
    x = torch.randn(B, T, D)
    out = model(x)
    
    assert out.shape == (B, T, 7)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert torch.isfinite(out).all()

def test_intent_transformer_v2_shape():
    model = IntentPlannerTransformer_V2(
        input_dim=32,
        hidden_dim=128,
        num_layers=2,
        nhead=8,
        output_dim=7,
        dropout=0.1
    )
    B, T, D = 2, 60, 32
    x = torch.randn(B, T, D)
    out = model(x)
    
    assert out.shape == (B, T, 7)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert torch.isfinite(out).all()

def test_intent_transformer_v2_overfit_smoke():
    model = IntentPlannerTransformer_V2(dropout=0.0) # Disable dropout for faster pure overfit
    B, T, D = 1, 60, 32
    # Mock inputs
    x = torch.randn(B, T, D)
    y_target = torch.rand(B, T, 7)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    initial_loss = None
    for i in range(10):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y_target)
        loss.backward()
        optimizer.step()
        
        if i == 0:
            initial_loss = loss.item()
            
    final_loss = loss.item()
    # Loss should decrease significantly after 10 steps identically trained
    assert final_loss < initial_loss
    assert torch.isfinite(out).all()

def test_infer_intent_transformer_v2():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = PipelineConfig(
            output_root=str(tmp_path),
            intent_model_type="transformer"
        )
        
        # Instantiate model directly to bypass train checkpoint requirement for this test,
        # but we use the inferencer wrapper normally.
        # We will manually inject the model state dict into the expected filepath
        model_dir = tmp_path / "models" / "intent_v1"
        model_dir.mkdir(parents=True)
        ckpt_path = model_dir / "best.pt"
        
        test_model = IntentPlannerTransformer_V2()
        torch.save({
            'model_state_dict': test_model.state_dict(),
            'cfg': cfg.model_dump()
        }, ckpt_path)
        
        # Test Inference Wrapper directly
        inferencer = IntentInferencer(cfg)
        assert inferencer.ready
        
        fake_situation = np.random.randn(32).astype(np.float32)
        out_preds = inferencer.predict_sequence(fake_situation, num_frames=60)
        
        assert out_preds.shape == (60, 7)
        assert np.isfinite(out_preds).all()
        assert out_preds.min() >= 0.0
        assert out_preds.max() <= 1.0
