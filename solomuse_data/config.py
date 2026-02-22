import json
import yaml
from pathlib import Path
from typing import Dict, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# Allowed datasets for validation
KNOWN_DATASETS = {"slakh", "musdb", "medleydb", "moisesdb", "weak_songs", "mock"}

class PipelineConfig(BaseModel):
    # LOCKED CONFIGURATION - DO NOT CHANGE
    canonical_sample_rate: int = Field(default=44100, frozen=True)
    canonical_channels: int = Field(default=2, frozen=True)
    lufs_target: float = Field(default=-18.0, frozen=True)
    solo_stem_policy: Literal["lead_any"] = Field(default="lead_any", frozen=True)
    
    # Configurable parameters with strict defaults
    peak_limit_dbfs: float = -1.0
    segment_seconds: float = 6.0
    segment_hop_seconds: float = 3.0
    min_segment_energy: float = 1e-4
    
    dataset_roots: Dict[str, str] = Field(default_factory=dict)
    output_root: str
    
    seed: int = 1337
    enable_weak_demucs: bool = False
    demucs_model: str = "htdemucs"
    num_workers: int = 4
    strict: bool = True

    @field_validator("canonical_sample_rate")
    @classmethod
    def validate_sample_rate(cls, v):
        if v != 44100:
            raise ValueError("canonical_sample_rate must be 44100")
        return v

    @field_validator("lufs_target")
    @classmethod
    def validate_lufs(cls, v):
        if v != -18.0:
            raise ValueError("lufs_target must be -18.0")
        return v

    @field_validator("canonical_channels")
    @classmethod # Typo correction: logic is correct but decorator target needs to exist. Wait, field is canonical_channels
    def validate_channels(cls, v):
        if v != 2:
            raise ValueError("canonical_channels must be 2")
        return v
         
    @field_validator("solo_stem_policy")
    @classmethod
    def validate_policy(cls, v):
        if v != "lead_any":
             raise ValueError("solo_stem_policy must be 'lead_any'")
        return v

    @field_validator("peak_limit_dbfs")
    @classmethod
    def validate_peak(cls, v):
        if v > 0:
            raise ValueError("peak_limit_dbfs must be <= 0")
        return v

    @field_validator("dataset_roots")
    @classmethod
    def validate_datasets(cls, v):
        for key in v.keys():
            if key not in KNOWN_DATASETS:
                raise ValueError(f"Unknown dataset '{key}'. Must be one of {KNOWN_DATASETS}")
        return v

    @model_validator(mode='after')
    def validate_segments(self):
        if self.segment_hop_seconds > self.segment_seconds:
            raise ValueError(f"segment_hop_seconds ({self.segment_hop_seconds}) cannot be greater than segment_seconds ({self.segment_seconds})")
        return self

def load_config(path: str) -> PipelineConfig:
    """Load configuration from a YAML or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    
    with open(p, "r") as f:
        if p.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        elif p.suffix == ".json":
            data = json.load(f)
        else:
            raise ValueError("Config file must be .yaml, .yml, or .json")
    
    return PipelineConfig(**data)

def save_config(cfg: PipelineConfig, path: str):
    """Save configuration to a YAML or JSON file."""
    p = Path(path)
    data = cfg.model_dump()
    
    with open(p, "w") as f:
        if p.suffix in (".yaml", ".yml"):
            yaml.dump(data, f, default_flow_style=False)
        elif p.suffix == ".json":
            json.dump(data, f, indent=2)
        else:
             raise ValueError("Config file output must be .yaml, .yml, or .json")
