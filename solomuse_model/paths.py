import os
from pathlib import Path
from solomuse_data.config import PipelineConfig

def get_intent_checkpoint_path(cfg: PipelineConfig) -> Path:
    """
    Returns the unified exact path for the Intent Planner checkpoint.
    If the user has explicitly overridden `intent_checkpoint_path`, use that.
    Otherwise, fall back to the default generated path based on output_root and version.
    """
    if cfg.intent_checkpoint_path:
        return Path(cfg.intent_checkpoint_path)
    return Path(cfg.output_root) / "models" / f"intent_{cfg.intent_model_version}" / "best.pt"

def get_renderer_checkpoint_path(cfg: PipelineConfig) -> Path:
    """
    Returns the unified exact path for the Renderer checkpoint.
    """
    if cfg.renderer_checkpoint_path:
        return Path(cfg.renderer_checkpoint_path)
    return Path(cfg.output_root) / "models" / f"renderer_{cfg.renderer_model_version}" / "best.pt"
