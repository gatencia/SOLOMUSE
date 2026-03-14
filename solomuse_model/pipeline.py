import logging
import time
import torch
import numpy as np
from typing import Any, Dict, Optional
from pathlib import Path
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

    def summarize_situation(self, backing_audio: np.ndarray) -> np.ndarray:
        """
        Summarize the musical situation from backing audio.
        """
        start_time = time.perf_counter()
        vector = self.compute_situation(backing_audio)
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f" [Latency] Layer 1 (Situation): {latency:.2f}ms")
        
        assert vector.shape == (32,), f"Situation vector shape mismatch: expected (32,), got {vector.shape}"
        return vector

    def compute_situation(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute situation features and vector for raw audio.
        """
        from solomuse_model.situation.extract import extract_situation_v1
        from solomuse_model.situation.vectorize import vectorize_situation_v1
        
        sr = self.cfg.canonical_sample_rate
        # Ensure audio is float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
            
        features = extract_situation_v1(audio, sr, self.cfg)
        vector = vectorize_situation_v1(features)
        return vector

    def plan_intent(self, situation_summary: np.ndarray, duration_s: float) -> np.ndarray:
        """
        Plan the musical intent based on the situation summary.
        
        Args:
            situation_summary: Vector from compute_situation [32].
            duration_s: Expected duration in seconds to derive frame count.
        """
        start_time = time.perf_counter()
        num_frames = int(duration_s * self.cfg.intent_hz)
        
        from solomuse_model.paths import get_intent_checkpoint_path
        ckpt_path = get_intent_checkpoint_path(self.cfg)
        
        if ckpt_path.exists():
            logger.info(f"Planning intent using {self.cfg.intent_model_type} model from {ckpt_path}")
            from solomuse_model.intent.infer import IntentInferencer
            inferencer = IntentInferencer(self.cfg, str(ckpt_path))
            intent = inferencer.predict_sequence(situation_summary, num_frames)
        else:
            logger.info(f"No intent model checkpoint found at {ckpt_path}. Returning zero intent array.")
            intent = np.zeros((num_frames, 7), dtype=np.float32)
            
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f" [Latency] Layer 2 (Intent): {latency:.2f}ms")
        
        assert intent.shape == (num_frames, 7), f"Intent shape mismatch: expected ({num_frames}, 7), got {intent.shape}"
        return intent

    def render_audio(self, x_audio: np.ndarray, intent_plan: np.ndarray, situation_summary: np.ndarray = None) -> Dict[str, Any]:
        """
        Produce solo audio given backing context and an intent plan.
        
        Returns:
            Dict containing 'y_hat' (audio) and optional 'y_tokens' (if V2).
        """
        start_time = time.perf_counter()
        
        if not self.cfg.renderer_enable:
            logger.warning("Renderer disabled. Returning silence.")
            return {"y_hat": np.zeros_like(x_audio)}
            
        if situation_summary is None:
            situation_summary = np.zeros(32, dtype=np.float32)
            
        outputs = {}
        
        if self.cfg.renderer_model_type == "token_transformer":
            from solomuse_model.renderer.infer_v2 import TokenRendererSimulator
            from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
            from solomuse_model.renderer.alignment import upsample_intent_to_tokens
            from solomuse_model.paths import get_renderer_checkpoint_path
            
            ckpt_path = get_renderer_checkpoint_path(self.cfg)
            if not Path(ckpt_path).exists():
                logger.warning(f"Renderer checkpoint not found at {ckpt_path}. Returning silence.")
                outputs["y_hat"] = np.zeros_like(x_audio)
            else:
                sim = TokenRendererSimulator(self.cfg, ckpt_path)
                adapter = EnCodecAdapter()
                
                # 1. Get X Tokens
                x_tokens = adapter.encode(x_audio, self.cfg.canonical_sample_rate)
                
                # 2. Align Intent
                # EnCodec tokens are strictly 75Hz
                intent_aligned = upsample_intent_to_tokens(intent_plan, self.cfg.intent_hz, 75.0, x_tokens.shape[0])
                
                # 3. Generate
                y_tokens, y_hat = sim.generate(x_tokens, intent_aligned, situation_summary)
                outputs["y_hat"] = y_hat
                outputs["y_tokens"] = y_tokens
            
        else: # conv1d / v1
            from solomuse_model.renderer.infer import render_segment
            from solomuse_model.paths import get_renderer_checkpoint_path
            ckpt_path = get_renderer_checkpoint_path(self.cfg)
            y_hat = render_segment(x_audio, intent_plan, situation_summary, self.cfg, str(ckpt_path) if ckpt_path.exists() else None)
            outputs["y_hat"] = y_hat
            
        latency = (time.perf_counter() - start_time) * 1000
        logger.info(f" [Latency] Layer 3 (Renderer): {latency:.2f}ms")
        
        return outputs

    def run_pipeline_infer(self, x_audio: np.ndarray, segment_id: str, output_dir: Optional[Path] = None):
        """
        Unified end-to-end inference for a single segment.
        """
        logger.info(f"=== Starting Unified Pipeline Inference for {segment_id} ===")
        total_start = time.perf_counter()
        
        # 1. Situation
        situation = self.summarize_situation(x_audio)
        
        # 2. Intent
        duration_s = float(len(x_audio)) / self.cfg.canonical_sample_rate
        intent = self.plan_intent(situation, duration_s)
        
        # 3. Renderer
        render_results = self.render_audio(x_audio, intent, situation)
        y_hat = render_results["y_hat"]
        
        total_latency = (time.perf_counter() - total_start) * 1000
        logger.info(f"=== Pipeline Inference Complete (Total Latency: {total_latency:.2f}ms) ===")
        
        # 4. Save Artifacts
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            import soundfile as sf
            sf.write(output_dir / "y_hat.wav", y_hat, self.cfg.canonical_sample_rate)
            np.save(output_dir / "intent_pred.npy", intent)
            np.save(output_dir / "situation.npy", situation)
            
            if "y_tokens" in render_results:
                np.save(output_dir / "y_hat_tokens.npy", render_results["y_tokens"])
                
            logger.info(f"Saved inference results to {output_dir}")
            
        return render_results

    def run_live_simulation(self, x_full: np.ndarray, chunk_size_s: float = 1.0) -> np.ndarray:
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
