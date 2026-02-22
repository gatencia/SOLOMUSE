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

    def render_audio(self, x_audio: Any, intent_plan: Any, situation_summary: Any = None) -> Any:
        """
        Produce solo audio given backing context and an intent plan.
        
        Args:
            x_audio: Input backing context array.
            intent_plan: Intent [F, 7].
            situation_summary: Optional situation array [32].
            
        Returns:
            y_hat_audio: [T] array
        """
        logger.debug("Rendering audio...")
        if not self.cfg.renderer_enable:
            logger.warning("Renderer disabled. Returning silence.")
            import numpy as np
            return np.zeros_like(x_audio)
            
        from solomuse_model.renderer.infer import render_segment
        if situation_summary is None:
            import numpy as np
            situation_summary = np.zeros(32, dtype=np.float32)
            
        y_hat = render_segment(x_audio, intent_plan, situation_summary, self.cfg, self.cfg.renderer_checkpoint_path)
        return y_hat

    def run_live_simulation(self, x_full: Any, chunk_size_s: float = 1.0) -> Any:
        """
        Runs the full streaming overlap-add pipeline on a complete backing track.
        """
        logger.info(f"Simulating live stream (chunk_size={chunk_size_s}s)")
        from solomuse_model.renderer.streaming import LiveSimulationRunner
        runner = LiveSimulationRunner(self)
        y_out = runner.run_stream(x_full, chunk_size_s)
        return y_out

    def process_step(self, backing_audio: Any) -> Any:
        """
        Orchestrates a single discrete step of the pipeline.
        
        Args:
            backing_audio: Input backing context.
            
        Returns:
            Rendered solo audio chunk.
        """
        situation = self.summarize_situation(backing_audio)
        # Assumes step is short enough that intent dur equals audio len
        duration_s = float(len(backing_audio)) / self.cfg.canonical_sample_rate
        intent = self.plan_intent(situation, duration_s)
        audio = self.render_audio(backing_audio, intent, situation)
        return audio
