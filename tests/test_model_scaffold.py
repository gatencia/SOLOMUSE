import pytest
from solomuse_data.config import PipelineConfig
from solomuse_model.pipeline import SoloMusePipeline
from solomuse_model.situation.contracts import SituationModel
from solomuse_model.intent.contracts import IntentModel
from solomuse_model.renderer.contracts import AudioRenderer

@pytest.fixture
def mock_config():
    return PipelineConfig(
        output_root="./data/processed",
        dataset_roots={"mock": "./data/raw/mock"},
        model_enable=True
    )

def test_pipeline_class_instantiates(mock_config):
    """Verify SoloMusePipeline can be created with a config."""
    pipeline = SoloMusePipeline(mock_config)
    assert pipeline.cfg == mock_config

def test_config_has_new_model_fields():
    """Check default values for new fields in PipelineConfig."""
    # We need a minimal valid config
    cfg = PipelineConfig(output_root="./test_out")
    
    assert cfg.model_enable is True
    assert cfg.intent_model_version == "v1"
    assert cfg.live_chunk_ms == 250
    assert cfg.live_hop_ms == 125
    assert cfg.state_hz == 20
    assert cfg.intent_hz == 10
    assert cfg.codec_frame_hz == 50

def test_contracts_are_runtime_checkable():
    """Verify that the Protocols are working as expected."""
    class ValidSituation:
        def summarize(self, backing_audio): return "ok"
    
    assert isinstance(ValidSituation(), SituationModel)
    
    class InvalidSituation:
        pass
    
    assert not isinstance(InvalidSituation(), SituationModel)

def test_cli_model_status_import():
    """Ensure the model-status function can be imported and doesn't crash on import."""
    from solomuse_data.cli import model_status
    assert callable(model_status)
