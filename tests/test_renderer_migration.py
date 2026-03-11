import pytest
import numpy as np

from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.token_contracts import validate_discrete_shape, validate_continuous_shape
from solomuse_model.renderer.encodec_adapter import EnCodecAdapter

def test_codec_interface_reports_metadata():
    codec = WaveChunkCodec(frame_ms=20.0, hop_ms=10.0, target_sr=16000)
    
    assert codec.code_type == "continuous"
    assert codec.code_dim == int((20.0 / 1000) * 16000) # 320
    assert codec.num_codebooks is None
    assert codec.vocab_size is None
    assert codec.frame_rate_hz() == 100.0

def test_token_contract_shapes():
    # Valid continuous
    validate_continuous_shape(np.zeros((10, 320)), code_dim=320)
    
    # Invalid continuous
    with pytest.raises(ValueError):
        validate_continuous_shape(np.zeros((10, 500)), code_dim=320)
        
    # Valid discrete
    validate_discrete_shape(np.zeros((10, 4)), num_codebooks=4)
    
    # Invalid discrete
    with pytest.raises(ValueError):
        validate_discrete_shape(np.zeros((10,)), num_codebooks=4)



def test_renderer_pipeline_still_works_with_wavechunk(tmp_path):
    # Testing that the original baseline path is unbroken
    import pandas as pd
    from pathlib import Path
    from solomuse_data.config import PipelineConfig
    from solomuse_model.renderer.train import run_train_renderer
    
    root = tmp_path / "mock_out"
    seg = root / "segments" / "mock" / "t1" / "s1"
    seg.mkdir(parents=True)
    
    sr = 44100
    np.save(seg / "renderer_target.npy", np.random.randn(10, 882).astype(np.float32))
    
    man_path = root / "segments" / "mock" / "manifest_intent.csv"
    pd.DataFrame([{"segment_id": "s1", "dataset": "mock", "track_id": "t1", "split": "train"}]).to_csv(man_path, index=False)
    
    import soundfile as sf
    sf.write(seg / "x.wav", np.random.randn(sr), sr)
    np.save(seg / "situation.npy", np.random.randn(32).astype(np.float32))
    np.save(seg / "intent_targets.npy", np.random.randn(10, 7).astype(np.float32))
    
    cfg = PipelineConfig(
        output_root=str(root),
        canonical_sample_rate=sr,
        renderer_batch_size=1,
        renderer_epochs=1,
        renderer_representation="wavechunk"
    )
    
    # Should run successfully without throwing the NotImplemented discrete error
    try:
        run_train_renderer(cfg, "mock")
        best_pt = Path(root) / "models" / "renderer_v1" / "best.pt"
        assert best_pt.exists()
    except ValueError as e:
        assert "prevent data leakage" in str(e) or "must contain 'track_id'" in str(e)
