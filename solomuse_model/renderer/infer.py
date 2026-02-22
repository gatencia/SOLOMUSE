import torch
import numpy as np
import logging
from typing import Optional

from solomuse_data.config import PipelineConfig
from solomuse_model.renderer.codec_interface import WaveChunkCodec
from solomuse_model.renderer.model_v1 import RendererConv1D_V1

logger = logging.getLogger(__name__)

def render_segment(x_audio: np.ndarray, intent_seq: np.ndarray, situation_vec: np.ndarray, cfg: PipelineConfig, checkpoint_path: Optional[str] = None) -> np.ndarray:
    """
    Offline render of a full segment given the prepared inputs.
    
    Args:
        x_audio: [T] backing track
        intent_seq: [F, 7] planned intent
        situation_vec: [32] overall situation
        cfg: Pipeline Config
        checkpoint_path: Path to best.pt
        
    Returns:
        y_hat_audio: [T] synthesized solo track
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # 1. Init Codec
    if cfg.renderer_representation == "wavechunk":
        codec = WaveChunkCodec(frame_ms=cfg.renderer_frame_ms, hop_ms=cfg.renderer_hop_ms, target_sr=cfg.canonical_sample_rate)
    else:
        raise ValueError("Unsupported codec")
        
    c_x = codec.frame_size
    d_int = 7
    d_sit = 32
    c_y = c_x
    
    # 2. Encode inputs
    x_encoded = codec.encode(x_audio, cfg.canonical_sample_rate) # [Fx, c_x]
    
    # Align frame count min
    f_len = min(x_encoded.shape[0], intent_seq.shape[0])
    
    bx = torch.tensor(x_encoded[:f_len]).unsqueeze(0).to(device) # [1, F, c_x]
    bi = torch.tensor(intent_seq[:f_len]).unsqueeze(0).to(device) # [1, F, 7]
    bs = torch.tensor(situation_vec).unsqueeze(0).repeat(f_len, 1).unsqueeze(0).to(device) # [1, F, 32]
    
    # 3. Model
    model = RendererConv1D_V1(c_x, d_int, d_sit, c_y, hidden_dim=cfg.renderer_hidden_dim, num_blocks=3).to(device)
    model.eval()
    
    if checkpoint_path:
        try:
            ckpt = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {checkpoint_path}, using intialized weights. ({e})")
            
    with torch.no_grad():
        preds = model(bx, bi, bs) # [1, F, c_y]
        
    pred_codes = preds.squeeze(0).cpu().numpy() # [F, c_y]
    
    # 4. Decode
    y_hat_audio = codec.decode(pred_codes, cfg.canonical_sample_rate)
    
    return y_hat_audio
