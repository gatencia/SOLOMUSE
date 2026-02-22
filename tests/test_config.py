import pytest
from solomuse_data.config import PipelineConfig, load_config, save_config

def test_defaults_locked():
    """Verify that locked defaults are correct."""
    cfg = PipelineConfig(output_root="/tmp")
    assert cfg.canonical_sample_rate == 44100
    assert cfg.lufs_target == -18.0
    assert cfg.solo_stem_policy == "lead_any"
    assert cfg.canonical_channels == 2

def test_strict_validation_failures():
    """Verify that invalid configurations raise errors."""
    # Test invalid sample rate (even though it's frozen, if we try to force it or bypass)
    with pytest.raises(ValueError, match="canonical_sample_rate"):
        # Pydantic 2 frozen fields can only be set at init. 
        # But even if we pass a different value, validation should catch it if frozen wasn't enough
        # or if we are loading from dict
        PipelineConfig(output_root="/tmp", canonical_sample_rate=48000)

    with pytest.raises(ValueError, match="lufs_target"):
        PipelineConfig(output_root="/tmp", lufs_target=-14.0)

    with pytest.raises(ValueError, match="dataset_roots"):
        PipelineConfig(output_root="/tmp", dataset_roots={"unknown_db": "/path"})

    with pytest.raises(ValueError, match="segment_hop_seconds"):
        PipelineConfig(output_root="/tmp", segment_seconds=5.0, segment_hop_seconds=6.0)

def test_load_save_roundtrip(tmp_path):
    """Verify that we can save and load the config."""
    cfg = PipelineConfig(
        output_root=str(tmp_path / "output"),
        dataset_roots={"slakh": str(tmp_path / "slakh")},
        num_workers=8
    )
    
    yaml_path = tmp_path / "config.yaml"
    save_config(cfg, str(yaml_path))
    
    loaded_cfg = load_config(str(yaml_path))
    assert loaded_cfg.model_dump() == cfg.model_dump()
    assert loaded_cfg.num_workers == 8
    assert loaded_cfg.dataset_roots["slakh"] == str(tmp_path / "slakh")

def test_peak_limit_validation():
    """Verify peak limit validation."""
    with pytest.raises(ValueError, match="peak_limit_dbfs"):
        PipelineConfig(output_root="/tmp", peak_limit_dbfs=1.0)
    
    # Boundary condition
    cfg = PipelineConfig(output_root="/tmp", peak_limit_dbfs=0.0)
    assert cfg.peak_limit_dbfs == 0.0
