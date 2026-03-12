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

    # --- 3-Layer Model Configuration ---
    model_enable: bool = True
    situation_model_version: str = "v1"
    intent_model_version: str = "v1"
    renderer_model_version: str = "v1"
    
    # Real-time / Live performance settings
    live_chunk_ms: int = 250
    live_hop_ms: int = 125
    
    # Refresh rates (Hz)
    state_hz: int = 20
    intent_hz: int = 10
    codec_frame_hz: int = 50

    # --- Situation Extraction (Layer 1) ---
    situation_enable: bool = True
    situation_use_x_only: bool = True
    situation_frame_hz: int = 100
    situation_chroma_hz: int = 10
    situation_include_curves: bool = False
    situation_save_npy: bool = True
    situation_feature_version: str = "v1"

    # --- Intent Targets (Layer 2) ---
    intent_enable: bool = True
    intent_hz: int = 10
    intent_feature_version: str = "v1"
    intent_use_centroid_for_register: bool = True
    intent_use_chroma_tension_proxy: bool = True

    # --- Dataset Split Settings ---
    intent_train_ratio: float = 0.8
    intent_val_ratio: float = 0.1
    intent_test_ratio: float = 0.1
    force_regenerate_splits: bool = False

    # --- Baseline Intent Planner (Training/Inference) ---
    intent_model_type: str = "gru"
    intent_hidden_dim: int = 128 # Legacy alias for intent_d_model
    intent_d_model: int = 256
    intent_num_layers: int = 2
    intent_num_heads: int = 8
    intent_ffn_dim: int = 1024
    intent_dropout: float = 0.1
    intent_lr: float = 3e-4 # Updated default
    intent_weight_decay: float = 1e-2 # Updated default
    intent_grad_clip: float = 1.0 # Updated default
    intent_epochs: int = 50 # Updated default
    intent_batch_size: int = 16
    intent_warmup_steps: int = 2000
    intent_lr_schedule: str = "cosine" # "constant", "cosine"
    intent_checkpoint_path: str | None = None
    intent_overfit_one_batch: bool = False
    
    # --- Intent Training Stability & Debugging ---
    intent_skip_bad_batches: bool = False
    intent_max_bad_batches_per_epoch: int = 0
    intent_fail_on_nonfinite_input: bool = True
    intent_fail_on_nonfinite_target: bool = True
    intent_skip_optimizer_step_on_large_grad_norm: bool = False
    intent_large_grad_norm_threshold: float = 100.0
    intent_debug_save_crash_batches: bool = True
    intent_debug_log_batch_stats_every_n: int = 0
    intent_force_cpu_debug: bool = False
    
    # Eval config
    intent_eval_binary_threshold: float | None = 0.5

    # --- Renderer Data (Layer 3) ---
    renderer_enable: bool = True
    renderer_representation: str = "wavechunk" 
    renderer_frame_ms: float = 20.0
    renderer_hop_ms: float = 10.0
    renderer_target_version: str = "v1"
    renderer_token_cache: bool = True
    
    # --- Renderer Network (Training/Inference) ---
    renderer_model_type: str = "conv1d"
    renderer_hidden_dim: int = 128 # Legacy alias for renderer_d_model
    renderer_d_model: int = 768
    renderer_num_layers: int = 6 # Bumped default
    renderer_num_heads: int = 12
    renderer_ffn_dim: int = 3072
    renderer_lr: float = 2e-4 # Updated default
    renderer_weight_decay: float = 1e-2 # Updated default
    renderer_grad_clip: float = 1.0
    renderer_epochs: int = 100 # Updated default
    renderer_batch_size: int = 8
    renderer_warmup_steps: int = 5000
    renderer_lr_schedule: str = "cosine"
    renderer_dropout: float = 0.1
    renderer_checkpoint_path: str | None = None
    renderer_overfit_one_batch: bool = False
    overlap_add_enable: bool = True

    # --- Inference Sampling (Layer 3) ---
    inference_temperature: float = 0.9
    inference_top_k: int = 50
    inference_top_p: float = 0.95
    
    # --- W&B Experiment Tracking ---
    wandb_enabled: bool = False
    wandb_project: str = "solomuse"
    wandb_entity: str | None = None
    wandb_mode: str = "online"
    wandb_tags: list[str] = Field(default_factory=lambda: ["intent"])
    wandb_group: str | None = None
    wandb_run_name: str | None = None
    wandb_log_every_n_steps: int = 1
    wandb_watch_model: bool = False
    wandb_save_checkpoints_as_artifacts: bool = True

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

    @field_validator("intent_model_type")
    @classmethod
    def validate_intent_model(cls, v):
        valid = {"gru", "transformer"}
        if v.lower() not in valid:
            raise ValueError(f"intent_model_type must be one of {valid}")
        return v.lower()

    @field_validator("renderer_model_type")
    @classmethod
    def validate_renderer_model(cls, v):
        valid = {"conv1d", "token_transformer"}
        if v.lower() not in valid:
            raise ValueError(f"renderer_model_type must be one of {valid}")
        return v.lower()

    @field_validator("situation_model_version")
    @classmethod
    def validate_situation_version(cls, v):
        valid = {"v1"}
        if v.lower() not in valid:
            raise ValueError(f"situation_model_version must be one of {valid}")
        return v.lower()

    @field_validator("intent_model_version")
    @classmethod
    def validate_intent_version(cls, v):
        valid = {"v1", "v2"}
        if v.lower() not in valid:
            raise ValueError(f"intent_model_version must be one of {valid}")
        return v.lower()

    @field_validator("renderer_model_version")
    @classmethod
    def validate_renderer_version(cls, v):
        valid = {"v1", "v2"}
        if v.lower() not in valid:
            raise ValueError(f"renderer_model_version must be one of {valid}")
        return v.lower()

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
    def sync_hyperparam_aliases(self):
        # Sync Intent defaults/aliases
        if self.intent_hidden_dim != 128 and self.intent_d_model == 256:
            self.intent_d_model = self.intent_hidden_dim
        elif self.intent_d_model != 256:
            self.intent_hidden_dim = self.intent_d_model
            
        # Sync Renderer defaults/aliases
        if self.renderer_hidden_dim != 128 and self.renderer_d_model == 768:
            self.renderer_d_model = self.renderer_hidden_dim
        elif self.renderer_d_model != 768:
            self.renderer_hidden_dim = self.renderer_d_model
            
        return self

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
