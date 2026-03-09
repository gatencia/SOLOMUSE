import torch
import numpy as np
import logging
import time
from pathlib import Path
from solomuse_data.config import PipelineConfig

from solomuse_model.pipeline import SoloMusePipeline
from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.model_v1 import RendererConv1D_V1

logger = logging.getLogger(__name__)

class LiveSimulationRunner:
    """
    Simulates real-time chunked stream behavior over a static audio file.
    Maintains overlap-add buffers yielding a continuous output.
    """
    def __init__(self, pipeline: SoloMusePipeline):
        self.pl = pipeline
        self.cfg = pipeline.cfg
        self.sr = self.cfg.canonical_sample_rate
        
        # Segment length for the model (6s)
        self.window_samples = int(self.cfg.segment_seconds * self.sr)
        
    def _apply_crossfade(self, global_audio: np.ndarray, chunk_audio: np.ndarray, start_idx: int, hop_samples: int):
        """
        Smoothly overlap-add a new chunk into the global buffer using linear cross-fading.
        """
        chunk_len = len(chunk_audio)
        fade_len = min(hop_samples, chunk_len)
        
        # 1. First part of the chunk: cross-fade with what was already there
        # (This is the 'overlap' part from the previous step's tail)
        # However, in a rolling window simulator where we render 6s every hop,
        # we usually just want to take the portion starting at 'hop' and fade it in.
        
        # Simpler OLA strategy for Live-Sim:
        # Every step we render a 6s segment. We take the portion [ -hop : ] 
        # but that's only valid if we are streaming FUTURE.
        
        # Correct streaming logic: 
        # At Time T, we have audio up to T. We render [T-6, T] and get solo [T-6, T].
        # We append the slice [T-hop, T] to our output.
        # To avoid clicks, we cross-fade the boundary between [T-hop-fade, T-hop] 
        # and the new chunk.
        
        # But wait, if the model is causal, then solo[T_n] only depends on backing[<T_n].
        # So we can just concatenate if the model is perfectly stateful.
        # Since it's not (we restart context or it's a CNN), we cross-fade.
        
        # Let's use a standard 50ms fade
        fade_samples = int(0.050 * self.sr)
        
        end_idx = start_idx + chunk_len
        if end_idx > len(global_audio):
            return # Should not happen in sim
            
        # Linear fade in
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = 1.0 - fade_in
        
        # Apply
        global_audio[start_idx : start_idx + fade_samples] = (
            global_audio[start_idx : start_idx + fade_samples] * fade_out + 
            chunk_audio[:fade_samples] * fade_in
        )
        # Direct copy for the rest
        global_audio[start_idx + fade_samples : end_idx] = chunk_audio[fade_samples:]

    def run_stream(self, x_full: np.ndarray, hop_size_s: float = 0.5) -> np.ndarray:
        """
        Simulate picking up audio chunks every hop_size_s using a rolling 6s context window.
        """
        hop_samples = int(hop_size_s * self.sr)
        total_samples = len(x_full)
        y_out = np.zeros(total_samples, dtype=np.float32)
        
        logger.info(f"Starting Live-Sim: Total Length={total_samples/self.sr:.2f}s, Hop={hop_size_s}s, Context={self.cfg.segment_seconds}s")
        
        # We start at position 0, but we need 6s of context.
        # For the beginning of the file, we zero-pad.
        
        metrics = []
        
        for pos in range(0, total_samples, hop_samples):
            step_start = time.perf_counter()
            
            # 1. Extract rolling 6s window ending at 'pos + hop_samples'
            context_end = pos + hop_samples
            context_start = context_end - self.window_samples
            
            if context_start < 0:
                # Pad beginning with zeros
                x_window = np.zeros(self.window_samples, dtype=np.float32)
                available = x_full[0 : context_end]
                x_window[self.window_samples - len(available):] = available
            else:
                x_window = x_full[context_start : context_end]
                
            if len(x_window) < self.window_samples:
                # Near end of file, pad with zeros
                tmp = np.zeros(self.window_samples, dtype=np.float32)
                tmp[:len(x_window)] = x_window
                x_window = tmp

            # 2. Run Pipeline Inference on this 6s window
            # Use segment_id for logging/debug
            seg_id = f"sim_pos_{pos}"
            render_results = self.pl.run_pipeline_infer(x_window, segment_id=seg_id)
            y_hat_window = render_results["y_hat"]
            
            # 3. Overlap-Add the rendered chunk
            # We take the *last* hop_samples of the rendered 6s window
            # and place it at pos in y_out.
            new_chunk = y_hat_window[-hop_samples:]
            
            if pos == 0:
                y_out[0 : hop_samples] = new_chunk
            else:
                self._apply_crossfade(y_out, new_chunk, pos, hop_samples)
                
            step_latency = (time.perf_counter() - step_start) * 1000
            metrics.append({
                "pos_s": pos / self.sr,
                "latency_ms": step_latency
            })
            
            # Progress log
            if pos % (hop_samples * 10) == 0:
                logger.info(f" Live-Sim Progress: {pos/total_samples*100:.1f}% | Last Latency: {step_latency:.1f}ms")

        avg_latency = np.mean([m['latency_ms'] for m in metrics])
        logger.info(f"Live-Sim Complete. Average Step Latency: {avg_latency:.2f}ms")
        
        return y_out
