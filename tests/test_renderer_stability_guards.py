import pytest
import torch
from unittest.mock import patch
from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.train import run_train_renderer

def test_renderer_nan_input_guard(tmp_path):
    cfg = PipelineConfig(
        output_root=str(tmp_path), 
        renderer_epochs=1, 
        renderer_batch_size=2,
        renderer_model_type="conv1d"
    )
    
    seg_dir = tmp_path / "segments" / "test_ds"
    seg_dir.mkdir(parents=True)
    (seg_dir / "manifest_renderer.csv").write_text("dataset,track_id,segment_id,split\ntest_ds,T1,S1,train\n")
    
    with patch("solomuse_model.renderer.train.DataLoader") as mock_dl:
        mock_dl.return_value = [
            (
                torch.full((2, 10, 882), float('nan')), # NaN input!
                torch.zeros((2, 10, 7)), 
                torch.zeros((2, 10, 32)), 
                torch.zeros((2, 10, 882)) 
            )
        ]
        
        with patch("solomuse_model.renderer.train.RendererDataset") as mock_ds:
            mock_ds.return_value.__len__.return_value = 1
            
            with patch("solomuse_model.utils.splits.create_track_grouped_splits", return_value=None):
                with pytest.raises(RuntimeError, match="NaN/Inf found in Renderer inputs/targets"):
                    run_train_renderer(cfg, "test_ds")


def test_renderer_nan_prediction_guard(tmp_path):
    cfg = PipelineConfig(
        output_root=str(tmp_path), 
        renderer_epochs=1,
        renderer_model_type="conv1d"
    )
    
    seg_dir = tmp_path / "segments" / "test_ds"
    seg_dir.mkdir(parents=True)
    (seg_dir / "manifest_renderer.csv").write_text("dataset,track_id,segment_id,split\ntest_ds,T1,S1,train\n")
    
    with patch("solomuse_model.renderer.train.DataLoader") as mock_dl:
        mock_dl.return_value = [
            (
                torch.zeros((2, 10, 882)), 
                torch.zeros((2, 10, 7)), 
                torch.zeros((2, 10, 32)), 
                torch.zeros((2, 10, 882))
            )
        ]
        
        with patch("solomuse_model.renderer.train.RendererDataset") as mock_ds:
            mock_ds.return_value.__len__.return_value = 1
            
            with patch("solomuse_model.renderer.train.RendererConv1D_V1") as mock_model:
                instance = mock_model.return_value
                instance.to.return_value = instance
                # Force model to output NaNs
                instance.return_value = torch.full((2, 10, 882), float('nan'))
                instance.parameters.return_value = [torch.nn.Parameter(torch.zeros(1))]
                
                with patch("solomuse_model.utils.splits.create_track_grouped_splits", return_value=None):
                    with pytest.raises(RuntimeError, match="NaN/Inf found in Renderer predictions"):
                        run_train_renderer(cfg, "test_ds")
