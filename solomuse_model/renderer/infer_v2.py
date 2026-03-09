import logging
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.model_v2.transformer_lm import TokenTransformerRenderer
from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
from solomuse_model.paths import get_renderer_checkpoint_path

logger = logging.getLogger(__name__)

class TokenRendererSimulator:
    """
    Inference generator driving the Transformer LM to emit autoregressive tokens.
    """
    def __init__(self, cfg: PipelineConfig, ckpt_path: str):
        self.cfg = cfg
        # Due to severe PyTorch MPS bus errors with causal masking in TransformerDecoder,
        # we explicitly fallback to CPU for Apple Silicon users to preserve stability.
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.num_codebooks = 4
        self.vocab_size = 1024
        
        self.model = TokenTransformerRenderer(
            d_model=cfg.renderer_hidden_dim, 
            nhead=8, 
            num_layers=3, 
            num_codebooks=self.num_codebooks, 
            vocab_size=self.vocab_size
        )
        
        logger.info(f"Loading TokenTransformerRenderer weights from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Load Decoder Audio Adapter
        self.adapter = EnCodecAdapter()
        
    def _sample(self, logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> torch.Tensor:
        """
        Samples a single token from a probability distribution.
        Logits shape: [Vocab_Size]
        Returns: [1] (Index)
        """
        if temperature == 0.0:
            return torch.argmax(logits, dim=-1, keepdim=True)
            
        logits = logits / temperature
        
        if top_k > 0:
            top_v, top_i = torch.topk(logits, top_k)
            logits[logits < top_v[-1]] = float('-inf')
            
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return token
        
    @torch.no_grad()
    def generate(self, 
                 x_tokens: np.ndarray, 
                 intent_aligned: np.ndarray, 
                 situation: np.ndarray, 
                 temperature: float = 0.0,
                 top_k: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates solo waveform from conditioning buffers.
        
        Args:
            x_tokens: [T, Q] Support frames
            intent_aligned: [T, 7]
            situation: [32]
            
        Returns:
            y_tokens: [T, Q] discrete codes
            y_audio: [Length] raw 24kHz audio waveform
        """
        T, Q = x_tokens.shape
        
        x_t = torch.from_numpy(x_tokens).long().unsqueeze(0).to(self.device) # [1, T, Q]
        int_t = torch.from_numpy(intent_aligned).float().unsqueeze(0).to(self.device) # [1, T, 7]
        sit_t = torch.from_numpy(situation).float().unsqueeze(0).to(self.device) # [1, 32]
        
        # Initialize y_tokens with an arbitrary start frame (typically zeros, meaning silence padding)
        # We will build this autoregressively: predicting step t+1 based on inputs up to t
        y_gen = torch.zeros((1, 1, Q), dtype=torch.long, device=self.device)
        
        logger.info(f"Starting AR generation for {T} frames [Temp={temperature}, Top-K={top_k}]...")
        
        for t in range(T - 1):
            # Model sees sequence up to current generated length
            # Note: We enforce the causal mask automatically inside the forward block
            # Evaluate current context
            logits = self.model(
                x_tokens=x_t[:, :t+1, :], 
                intent_aligned=int_t[:, :t+1, :], 
                situation=sit_t, 
                y_tokens=y_gen
            )
            
            # Logits: [1, seq_len, Q, VocabSize]
            # Take the logits for the *last* sequence step we just passed in 
            next_step_logits = logits[:, -1, :, :] # shape: [1, Q, VocabSize]
            
            next_tokens_q = []
            for q in range(Q):
                token_q = self._sample(next_step_logits[0, q, :], temperature=temperature, top_k=top_k)
                next_tokens_q.append(token_q)
                
            next_frame = torch.cat(next_tokens_q).unsqueeze(0).unsqueeze(1) # [1, 1, Q]
            y_gen = torch.cat([y_gen, next_frame], dim=1) # Append along time dim
            
        y_final_tokens = y_gen.squeeze(0).cpu().numpy().astype(np.int32)
        
        # Decode back to raw waveform! Config states 44.1k is standard export
        # Encodec natively outputs 24kHz, we pass explicitly via adapter API
        logger.info("Generation complete. Decoding EnCodec tokens back to raw audio...")
        y_audio = self.adapter.decode(y_final_tokens, sr=self.cfg.canonical_sample_rate)
        
        return y_final_tokens, y_audio

def run_inference_v2(cfg: PipelineConfig, dataset_name: str, segment_id: str,
                     temperature: float = 0.0, top_k: int = 0):
    ckpt_path = get_renderer_checkpoint_path(cfg)
    
    # Locate inputs
    track_id = segment_id.split('_')[0]
    seg_dir = Path(cfg.output_root) / "segments" / dataset_name / track_id / segment_id
    
    # Avoid sandbox stat permission error by using try-except instead of .exists()
    try:
        with open(seg_dir / "x_tokens.npy", 'rb'):
            pass
    except (FileNotFoundError, PermissionError):
        logger.error(f"Cannot find segment directory at {seg_dir} or permission denied.")
        return
        
    x_path = seg_dir / "x_tokens.npy"
    i_path = seg_dir / "intent_aligned.npy"
    s_path = seg_dir / "situation.npy"
    
    try:
        x_tok = np.load(str(x_path))
        i_al = np.load(str(i_path))
        s_vec = np.load(str(s_path))
    except (FileNotFoundError, PermissionError) as e:
        logger.error(f"Missing conditioning variables in {seg_dir}. Ensure `renderer-token-targets` was run. ({e})")
        return
    except Exception as e:
        logger.error(f"Failed to load arrays: {e}")
        return

    sim = TokenRendererSimulator(cfg, ckpt_path)
    
    try:
        y_tok, y_wav = sim.generate(x_tok, i_al, s_vec, temperature, top_k)
    except Exception as e:
        logger.error(f"Inference failure: {e}")
        return
        
    out_tok_obj = seg_dir / f"y_tokens_pred_v2.npy"
    out_wav_obj = seg_dir / f"y_hat_v2.wav"
    
    np.save(str(out_tok_obj), y_tok)
    
    import soundfile as sf
    sf.write(str(out_wav_obj), y_wav, cfg.canonical_sample_rate)
    
    logger.info(f"Successfully generated outputs!")
    logger.info(f" Saved Tokens: {out_tok_obj}")
    logger.info(f" Saved Audio: {out_wav_obj}")
