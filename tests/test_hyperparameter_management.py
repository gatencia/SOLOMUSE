import pytest
import yaml
import torch
from pathlib import Path
from solomuse_data.config import PipelineConfig, load_config
from solomuse_model.intent.model_v2 import IntentPlannerTransformer_V2
from solomuse_model.renderer.model_v2.transformer_lm import TokenTransformerRenderer

def test_config_yaml_overrides(tmp_path):
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        "output_root": str(tmp_path),
        "intent_d_model": 512,
        "intent_num_layers": 4,
        "renderer_d_model": 1024,
        "renderer_epochs": 150,
        "inference_temperature": 0.5
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    cfg = load_config(str(config_file))
    
    assert cfg.intent_d_model == 512
    assert cfg.intent_num_layers == 4
    assert cfg.renderer_d_model == 1024
    assert cfg.renderer_epochs == 150
    assert cfg.inference_temperature == 0.5
    # Verify legacy sync
    assert cfg.intent_hidden_dim == 512
    assert cfg.renderer_hidden_dim == 1024

def test_config_legacy_alias_sync(tmp_path):
    config_file = tmp_path / "legacy_config.yaml"
    config_data = {
        "output_root": str(tmp_path),
        "intent_hidden_dim": 137,
        "renderer_hidden_dim": 99
    }
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    
    cfg = load_config(str(config_file))
    
    # Legacy hidden_dim should sync to d_model
    assert cfg.intent_d_model == 137
    assert cfg.renderer_d_model == 99

def test_intent_model_construction():
    cfg = PipelineConfig(
        output_root="/tmp",
        intent_d_model=256,
        intent_num_layers=6,
        intent_num_heads=8,
        intent_ffn_dim=1024,
        intent_dropout=0.1
    )
    
    model = IntentPlannerTransformer_V2(
        input_dim=32,
        hidden_dim=cfg.intent_d_model,
        num_layers=cfg.intent_num_layers,
        nhead=cfg.intent_num_heads,
        ffn_dim=cfg.intent_ffn_dim,
        output_dim=7,
        dropout=cfg.intent_dropout
    )
    
    # Check dimensions
    assert model.input_proj.out_features == 256
    # Access internal transformer layers if possible, but checking param count or structure is easier
    # encoder_layer = model.transformer_encoder.layers[0]
    # assert encoder_layer.self_attn.embed_dim == 256
    # assert encoder_layer.linear1.out_features == 1024
    
    # Structural check
    repr_str = repr(model)
    assert "d_model=256" in repr_str or "embed_dim=256" in repr_str
    assert "nhead=8" in repr_str
    assert "dim_feedforward=1024" in repr_str

def test_renderer_model_construction():
    cfg = PipelineConfig(
        output_root="/tmp",
        renderer_d_model=768,
        renderer_num_layers=12,
        renderer_num_heads=12,
        renderer_ffn_dim=3072,
        renderer_dropout=0.1
    )
    
    model = TokenTransformerRenderer(
        d_model=cfg.renderer_d_model,
        nhead=cfg.renderer_num_heads,
        num_layers=cfg.renderer_num_layers,
        ffn_dim=cfg.renderer_ffn_dim,
        num_codebooks=4,
        vocab_size=1024,
        dropout=cfg.renderer_dropout
    )
    
    assert model.d_model == 768
    repr_str = repr(model)
    assert "d_model=768" in repr_str or "embed_dim=768" in repr_str
    assert "nhead=12" in repr_str
    assert "dim_feedforward=3072" in repr_str

def test_cli_overrides_integration(tmp_path, caplog):
    """
    Simulate CLI overrides and verify they propagate to the config.
    """
    from solomuse_data.cli import main
    import sys
    
    config_file = tmp_path / "base_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump({"output_root": str(tmp_path)}, f)
        
    # Mock sys.argv
    test_args = [
        "solomuse", 
        "train-intent", 
        "--config", str(config_file), 
        "--dataset", "mock",
        "--intent-lr", "0.005",
        "--intent-d-model", "999",
        "--intent-epochs", "1"
    ]
    
    # We want to intercept the call to run_train_intent to see the cfg it gets
    import solomuse_model.intent.train
    original_run = solomuse_model.intent.train.run_train_intent
    
    captured_cfg = None
    def mock_run(cfg, dataset):
        nonlocal captured_cfg
        captured_cfg = cfg
        # Don't actually run training
        pass
        
    solomuse_model.intent.train.run_train_intent = mock_run
    
    # We need to monkeypatch sys.argv
    import unittest.mock
    with unittest.mock.patch.object(sys, 'argv', test_args):
        try:
            main()
        except SystemExit:
            pass
            
    # Restore
    solomuse_model.intent.train.run_train_intent = original_run
    
    assert captured_cfg is not None
    assert captured_cfg.intent_lr == 0.005
    assert captured_cfg.intent_d_model == 999
    assert captured_cfg.intent_epochs == 1

def test_training_setup_logging(tmp_path, caplog):
    """
    Verify the 'Training setup' block is printed with resolved hyperparams.
    """
    from solomuse_model.intent.train import run_train_intent
    import logging
    
    caplog.set_level(logging.INFO)
    
    cfg = PipelineConfig(
        output_root=str(tmp_path),
        intent_d_model=256,
        intent_num_layers=3,
        intent_epochs=1,
        intent_lr=0.001
    )
    
    # Mocking build_intent_dataloaders to avoid data loading
    import solomuse_model.intent.train
    import unittest.mock
    
    with unittest.mock.patch('solomuse_model.intent.train.build_intent_dataloaders') as mock_loaders:
        # Return empty loaders or similar to trigger error after logging setup
        # but before crashing the whole test
        mock_loaders.return_value = ([], [], [])
        
        try:
            run_train_intent(cfg, "mock")
        except Exception:
            pass
            
    log_text = caplog.text
    assert "INTENT TRAINING HYPERPARAMETERS" in log_text
    assert "D_Model:       256" in log_text
    assert "Num Layers:    3" in log_text
    assert "Learning Rate: 0.001" in log_text
