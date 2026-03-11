import pytest
import numpy as np
import pandas as pd
import json
from pathlib import Path

from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.types import RendererInputV1
from solomuse_model.renderer.dataset import RendererDataset
from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.run import run_renderer_target_build

def test_wavechunk_codec_encode_decode_shape_consistent():
    codec = WaveChunkCodec(frame_ms=20.0, hop_ms=10.0, target_sr=44100)
    
    # 1 second of audio at 44.1k
    sr = 44100
    audio = np.random.randn(sr).astype(np.float32)
    
    # Encode
    codes = codec.encode(audio, sr)
    assert codes.shape[1] == int((20.0 / 1000) * sr) # [F, 882]
    
    # Decode
    recovered = codec.decode(codes, sr)
    
    # The naive decode might not perfectly match length due to overhang logic, 
    # but it should be close to original length
    assert abs(len(recovered) - len(audio)) < codec.frame_size

def test_renderer_input_contract_shapes():
    # Verify TypedDict layout via instantiation
    inp = RendererInputV1(
        x_audio=np.zeros(44100),
        x_audio_path=None,
        intent_sequence=np.zeros((100, 7)),
        situation_vector=np.zeros(32),
        sr=44100
    )
    assert inp["x_audio"].shape == (44100,)
    assert inp["intent_sequence"].shape == (100, 7)

@pytest.fixture
def mock_pipeline_dir(tmp_path):
    output_root = tmp_path / "processed_renderer"
    seg_dir = output_root / "segments" / "mock" / "track1" / "seg1"
    seg_dir.mkdir(parents=True)
    
    # Mock data
    import soundfile as sf
    sf.write(seg_dir / "x.wav", np.random.randn(44100), 44100)
    sf.write(seg_dir / "y.wav", np.random.randn(44100), 44100)
    
    np.save(seg_dir / "situation.npy", np.random.randn(32).astype(np.float32))
    np.save(seg_dir / "intent_targets.npy", np.random.randn(100, 7).astype(np.float32))
    
    manifest_dir = output_root / "segments" / "mock"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pipeline builder needs intent manifest
    manifest_intent_path = manifest_dir / "manifest_intent.csv"
    df_intent = pd.DataFrame([{
        "segment_id": "seg1",
        "dataset": "mock",
        "track_id": "track1"
    }])
    df_intent.to_csv(manifest_intent_path, index=False)
    
    # 2. Dataset loader needs the splits
    manifest_path = manifest_dir / "manifest_renderer_splits.csv"
    df = pd.DataFrame([{
        "segment_id": "seg1",
        "dataset": "mock",
        "track_id": "track1",
        "split": "train"
    }])
    df.to_csv(manifest_path, index=False)
    
    return str(output_root), str(manifest_path), str(seg_dir)

def test_renderer_target_build_writes_artifacts(mock_pipeline_dir):
    out_root, manifest_path, seg_dir = mock_pipeline_dir
    
    cfg = PipelineConfig(
        output_root=out_root,
        renderer_frame_ms=20.0,
        renderer_hop_ms=10.0,
        canonical_sample_rate=44100
    )
    
    run_renderer_target_build(cfg, "mock")
    
    # Verify outputs
    target_path = Path(seg_dir) / "renderer_target.npy"
    assert target_path.exists()
    
    codes = np.load(target_path)
    assert codes.shape[1] == int((20.0 / 1000) * 44100) # 882 dims
    
    meta_path = Path(seg_dir) / "meta.json"
    assert meta_path.exists()
    with open(meta_path, "r") as f:
        meta = json.load(f)
        assert "renderer" in meta
        assert meta["renderer"]["representation"] == "wavechunk"

def test_renderer_dataset_loads_sample(mock_pipeline_dir):
    out_root, manifest_path, seg_dir = mock_pipeline_dir
    
    # first run builder to setup target.npy
    cfg = PipelineConfig(
        output_root=out_root,
        canonical_sample_rate=44100
    )
    run_renderer_target_build(cfg, "mock")
    
    # Test Dataset
    ds = RendererDataset(manifest_path)
    
    assert len(ds) == 1
    inp, targ = ds[0]
    
    assert inp["x_audio"].shape == (44100,)
    assert inp["intent_sequence"].shape == (100, 7)
    assert targ["target_codes"].ndim == 2 # [F, 882]
