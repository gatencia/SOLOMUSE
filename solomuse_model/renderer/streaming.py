import torch
import numpy as np
import logging
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
        
        # Determine internal model steps
        # E.g. renderer processes 20ms chunks, hop 10ms
        self.frame_samples = int((self.cfg.renderer_frame_ms / 1000.0) * self.sr)
        self.hop_samples = int((self.cfg.renderer_hop_ms / 1000.0) * self.sr)
        
        # Audio Buffer state
        self.out_buffer = [] # list of generated audio
        self.ola_buffer = np.zeros(self.frame_samples * 10, dtype=np.float32) # Lookahead space for Overlap-Add
        self.ola_idx = 0
        
        # Model loading (bypass purely offline pipeline)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = RendererConv1D_V1(
            c_x=self.frame_samples, d_int=7, d_sit=32, c_y=self.frame_samples,
            hidden_dim=self.cfg.renderer_hidden_dim, num_blocks=3
        ).to(self.device)
        self.model.eval()
        
        if self.cfg.renderer_checkpoint_path:
            try:
                ckpt = torch.load(self.cfg.renderer_checkpoint_path, map_location=self.device)
                self.model.load_state_dict(ckpt['model_state_dict'])
            except Exception as e:
                logger.warning(f"Could not load renderer checkpoint: {e}")

    def run_stream(self, x_full: np.ndarray, chunk_size_s: float = 1.0) -> np.ndarray:
        """
        Simulate picking up audio chunks every chunk_size_s and streaming predictions out.
        """
        chunk_samples = int(chunk_size_s * self.sr)
        total_samples = len(x_full)
        
        current_sit = np.zeros(32, dtype=np.float32)
        current_intent = np.zeros((0, 7), dtype=np.float32)
        
        y_out = np.zeros(total_samples, dtype=np.float32)
        
        for pos in range(0, total_samples, chunk_samples):
            logger.debug(f"Streaming chunk pos {pos}/{total_samples}")
            end_pos = min(pos + chunk_samples, total_samples)
            x_chunk = x_full[pos:end_pos]
            
            # 1. Update Situation given current chunk
            current_sit = self.pl.summarize_situation(x_chunk)
            
            # 2. Plan Intent for the duration of this chunk
            duration = (end_pos - pos) / self.sr
            intent_block = self.pl.plan_intent(current_sit, duration_s=duration) # [F_i, 7]
            
            # 3. Fast-forward Renderer block over this chunk
            # Calculate how many renderer frames fit in this chunk
            num_frames = 1 + (len(x_chunk) - self.frame_samples) // self.hop_samples
            if num_frames > 0:
                # 1D conv needs multiple frames to operate, use lib.stride or WaveChunk logic
                x_strided = np.lib.stride_tricks.as_strided(
                    x_chunk,
                    shape=(num_frames, self.frame_samples),
                    strides=(x_chunk.strides[0] * self.hop_samples, x_chunk.strides[0])
                )
                bx = torch.tensor(x_strided).unsqueeze(0).to(self.device) # [1, F, C]
                
                # Align Intent (Renderer rate ms might differ from intent rate Hz)
                # Resampling intent to match renderer Frames is standard.
                # Here we shortcut it since baseline tests force them to match visually:
                f_safe = min(num_frames, intent_block.shape[0])
                if f_safe == 0: continue
                
                bi = torch.tensor(intent_block[:f_safe]).unsqueeze(0).to(self.device)
                bs = torch.tensor(current_sit).unsqueeze(0).repeat(f_safe, 1).unsqueeze(0).to(self.device)
                bx = bx[:, :f_safe, :]
                
                with torch.no_grad():
                    preds = self.model(bx, bi, bs) # [1, F, C]
                    
                codes = preds.squeeze(0).cpu().numpy()
                
                # Overlap-add into y_out
                # Real live system would push to an asynchronous queue
                for i in range(f_safe):
                    p_start = pos + i * self.hop_samples
                    p_end = p_start + self.frame_samples
                    if p_end <= total_samples:
                        y_out[p_start:p_end] += codes[i]
                        
        return y_out
