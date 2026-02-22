import logging
from typing import Any
from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

class SoloMusePipeline:
    """
    Central orchestration module for the 3-layer live improvisation system.
    Wires together Situation, Intent, and Renderer layers.
    """
    def __init__(self, cfg: PipelineConfig):
        """
        Initialize the pipeline with configuration.
        
        Args:
            cfg: The pipeline configuration object.
        """
        self.cfg = cfg
        logger.info(f"Initialized SoloMusePipeline with versions: "
                    f"Situation={cfg.situation_model_version}, "
                    f"Intent={cfg.intent_model_version}, "
                    f"Renderer={cfg.renderer_model_version}")

    def summarize_situation(self, backing_audio: Any) -> Any:
        """
        Summarize the musical situation from backing audio.
        """
        return self.compute_situation(backing_audio)

    def compute_situation(self, audio: Any) -> Any:
        """
        Compute situation features and vector for raw audio.
        """
        from solomuse_model.situation.extract import extract_situation_v1
        from solomuse_model.situation.vectorize import vectorize_situation_v1
        
        # Ensure we have a sample rate from config
        sr = self.cfg.canonical_sample_rate
        features = extract_situation_v1(audio, sr, self.cfg)
        vector = vectorize_situation_v1(features)
        return vector

    def load_situation(self, situation_path: str) -> Any:
        """
        Load a precomputed situation vector.
        """
        import numpy as np
        return np.load(situation_path)

    def plan_intent(self, situation_summary: Any, duration_s: float) -> Any:
        """
        Plan the musical intent based on the situation summary.
        If a checkpoint is configured, runs inference using the baseline planner.
        
        Args:
            situation_summary: Vector from compute_situation [32].
            duration_s: Expected duration in seconds to derive frame count.
        """
        if self.cfg.intent_checkpoint_path:
            logger.info(f"Planning intent using model from {self.cfg.intent_checkpoint_path}")
            from solomuse_model.intent.infer import IntentInferencer
            inferencer = IntentInferencer(self.cfg, self.cfg.intent_checkpoint_path)
            num_frames = int(duration_s * self.cfg.intent_hz)
            return inferencer.predict_sequence(situation_summary, num_frames)
        else:
            logger.info("No intent model checkpoint found. Returning zero intent array.")
            import numpy as np
            num_frames = int(duration_s * self.cfg.intent_hz)
            return np.zeros((num_frames, 7), dtype=np.float32)
        
    def compute_intent_targets(self, x_audio: Any, y_audio: Any, situation_features: Any = None) -> Any:
        """
        Compute intent target vectors from backing and solo audio.
        """
        from solomuse_model.intent.extract_targets import extract_intent_targets_v1
        from solomuse_model.intent.vectorize import vectorize_intent_v1
        
        sr = self.cfg.canonical_sample_rate
        seq = extract_intent_targets_v1(x_audio, y_audio, sr, situation_features, self.cfg)
        return vectorize_intent_v1(seq)
        
    def load_intent_targets(self, intent_path: str) -> Any:
        """
        Load precomputed intent target matrix.
        """
        import numpy as np
        return np.load(intent_path)

    def render_audio(self, intent_plan: Any) -> Any:
        """
        Stub implementation for audio rendering.
        
        Args:
            intent_plan: Plan from plan_intent.
            
        Returns:
            A stub rendered audio segment.
        """
        logger.debug("Rendering audio...")
        # TODO: Implement Renderer layer
        raise NotImplementedError("Audio rendering not implemented yet.")

    def process_step(self, backing_audio: Any) -> Any:
        """
        Orchestrates a single step of the pipeline.
        
        Args:
            backing_audio: Input backing context.
            
        Returns:
            Rendered solo audio.
        """
        situation = self.summarize_situation(backing_audio)
        intent = self.plan_intent(situation)
        audio = self.render_audio(intent)
        return audio
