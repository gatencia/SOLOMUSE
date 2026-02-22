import torch
import numpy as np
import logging
from pathlib import Path
from typing import Optional

from solomuse_model.intent.model_v1 import IntentPlannerGRU_V1
from solomuse_data.config import PipelineConfig
from solomuse_model.paths import get_intent_checkpoint_path

logger = logging.getLogger(__name__)

class IntentInferencer:
    """Wrapper for handling sequence prediction using trained models."""
    def __init__(self, cfg: PipelineConfig, checkpoint_path: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.model = IntentPlannerGRU_V1(
            input_dim=32,
            hidden_dim=cfg.intent_hidden_dim,
            num_layers=cfg.intent_num_layers,
            output_dim=7,
            dropout=0.0 # No dropout for eval
        ).to(self.device)
        self.model.eval()
        self.ready = False
        
        path_to_load = get_intent_checkpoint_path(cfg)
        logger.info(f"Inferencer attempting to load checkpoint from: {path_to_load}")
        
        if path_to_load and Path(path_to_load).exists():
            try:
                checkpoint = torch.load(path_to_load, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.ready = True
                logger.info(f"Successfully loaded intent planner weights.")
            except Exception as e:
                logger.error(f"Failed to load checkpoint {path_to_load}: {e}")
                raise
        else:
            raise FileNotFoundError(
                f"No checkpoint found at provided path: {path_to_load}. "
                f"Model cannot run inference without trained weights. "
                f"Please run `python -m solomuse_data.cli train-intent` first."
            )

    def predict_sequence(self, situation_vector: np.ndarray, num_frames: int) -> np.ndarray:
        """
        Predict an F-length intent sequence from a singular situation vector.
        
        Args:
            situation_vector: [32] situation summary
            num_frames: desired F
            
        Returns:
            np.ndarray of shape [F, 7].
        """
        if not self.ready:
            raise RuntimeError("IntentInferencer model is not initialized/loaded.")
        
        # Broadcast situation: [32] -> [1, F, 32]
        sit_mat = np.tile(situation_vector, (num_frames, 1))
        # Add batch dim
        x_tensor = torch.from_numpy(sit_mat).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            preds = self.model(x_tensor)
            
        # preds: [1, F, 7]
        return preds.squeeze(0).cpu().numpy()
