import numpy as np
import pytest
from pathlib import Path
import os
import pandas as pd
import sys
from unittest.mock import MagicMock

from solomuse_model.renderer.alignment import upsample_intent_to_tokens

def test_align_intent_upsample():
    # Say intent is 2 seconds at 10hz = 20 frames, with 7 features
    intent_hz = 10.0
    intent_mat = np.arange(20)[:, None].repeat(7, axis=1) # [20, 7]
    assert intent_mat.shape == (20, 7)
    
    # Say token is 75Hz for 2 seconds = 150 frames
    token_hz = 75.0
    target_frames = 150
    
    upsampled = upsample_intent_to_tokens(intent_mat, intent_hz, token_hz, target_frames)
    
    # Verify shape
    assert upsampled.shape == (150, 7)
    
    # target_idx = target_frames-1 = 149
    # source_idx = floor(149 * (10 / 75)) = floor(149 * 0.1333) = floor(19.86) = 19
    assert upsampled[-1, 0] == 19
    assert upsampled[0, 0] == 0
    # middle frame (1s = 75 frames for tokens, 10 frames for intent)
    assert upsampled[75, 0] == 10

def test_token_manifest_splits(tmp_path):
    from solomuse_model.renderer.run_tokens import run_renderer_token_build
    from solomuse_data.config import PipelineConfig
    
    # Mock EnCodecAdapter dependencies
    if "encodec" not in sys.modules:
        sys.modules['encodec'] = MagicMock()
    if "torchaudio" not in sys.modules:
        sys.modules['torchaudio'] = MagicMock()
    
    output_root = tmp_path / "data" / "processed"
    segments_dir = output_root / "segments" / "mock"
    
    track_id = "TrackA"
    seg_id = "seg1"
    seg_dir = segments_dir / track_id / seg_id
    seg_dir.mkdir(parents=True, exist_ok=True)
    
    # Write fake files
    (seg_dir / "x.wav").write_bytes(b"")
    (seg_dir / "y.wav").write_bytes(b"")
    np.save(seg_dir / "situation.npy", np.zeros(32))
    np.save(seg_dir / "intent_targets.npy", np.zeros((60, 7)))
    
    # Write manifest_renderer_splits.csv and base manifest.csv
    manifest_base_path = segments_dir / "manifest.csv"
    manifest_splits_path = segments_dir / "manifest_renderer_splits.csv"
    
    df = pd.DataFrame([{
        "dataset": "mock",
        "track_id": track_id,
        "segment_id": seg_id,
        "split": "val"
    }])
    df.to_csv(manifest_splits_path, index=False)
    df.drop(columns=["split"]).to_csv(manifest_base_path, index=False)
    
    cfg = PipelineConfig(
        output_root=str(output_root),
        renderer_representation="encodec",
    )
    if not hasattr(cfg, 'intent_hz'):
        cfg.intent_hz = 10.0
    
    # FAKE read_audio and adapter
    import solomuse_model.renderer.run_tokens
    solomuse_model.renderer.run_tokens.read_audio = lambda x: (np.zeros(44100), 44100)
    
    class FakeAdapter:
        def encode(self, x, sr):
            return np.zeros((150, 4), dtype=np.int32)
        def frame_rate_hz(self):
            return 75.0
            
    import solomuse_model.renderer.encodec_adapter
    solomuse_model.renderer.encodec_adapter.EnCodecAdapter = FakeAdapter
    
    # Run builder
    run_renderer_token_build(cfg, "mock", limit=1, overwrite=True)
    
    # Check manifest
    out_manifest = segments_dir / "manifest_renderer_tokens.csv"
    assert out_manifest.exists()
    
    out_df = pd.read_csv(out_manifest)
    assert len(out_df) == 1
    assert out_df.iloc[0]["split"] == "val"
    assert out_df.iloc[0]["token_frames"] == 150
    assert out_df.iloc[0]["token_dim"] == 4
    assert out_df.iloc[0]["token_hz"] == 75.0
    assert out_df.iloc[0]["intent_hz"] == cfg.intent_hz
    
    # Check saved files
    assert (seg_dir / "x_tokens.npy").exists()
    assert (seg_dir / "y_tokens.npy").exists()
    assert (seg_dir / "intent_aligned.npy").exists()
    
    aligned = np.load(seg_dir / "intent_aligned.npy")
    assert aligned.shape == (150, 7)
