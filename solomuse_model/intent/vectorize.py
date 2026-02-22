import numpy as np
from solomuse_model.intent.types import IntentSequenceV1

def vectorize_intent_v1(intent_seq: IntentSequenceV1) -> np.ndarray:
    """
    Vectorize an IntentSequenceV1 into a matrix of shape [F, 7].
    
    Fixed Dimension Layout:
    0: play_prob
    1: density_proxy
    2: register_norm
    3: tension_proxy
    4: dynamics_norm
    5: onset_prob
    6: phrase_boundary_hint
    
    Args:
        intent_seq: IntentSequenceV1 object.
        
    Returns:
        np.ndarray of shape [F, 7] and dtype float32.
    """
    frames = intent_seq["frames"]
    F = len(frames)
    D = 7
    vec = np.zeros((F, D), dtype=np.float32)
    
    for i, frame in enumerate(frames):
        vec[i, 0] = frame["play_prob"]
        vec[i, 1] = frame["density_proxy"]
        vec[i, 2] = frame["register_norm"]
        vec[i, 3] = frame["tension_proxy"]
        vec[i, 4] = frame["dynamics_norm"]
        vec[i, 5] = frame["onset_prob"]
        vec[i, 6] = frame["phrase_boundary_hint"]
        
    return vec
