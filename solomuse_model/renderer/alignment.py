import numpy as np

def upsample_intent_to_tokens(intent_matrix: np.ndarray, intent_hz: float, token_hz: float, target_frames: int) -> np.ndarray:
    """
    Upsample an intent sequence [N, D] to exactly target_frames [T, D]
    based on the ratio between token_hz and intent_hz.
    Uses nearest neighbor interpolation for categorical/discrete states, allowing
    event boundaries to stay sharp.
    """
    if len(intent_matrix.shape) != 2:
        raise ValueError(f"Expected intent_matrix to be 2D, got {intent_matrix.shape}")
        
    N, D = intent_matrix.shape
    
    if target_frames == 0:
        return np.empty((0, D), dtype=intent_matrix.dtype)
        
    if N == 0:
        raise ValueError("Cannot upsample an empty intent matrix.")
        
    # Nearest neighbor mapping
    # For each target frame t, what is the corresponding source frame n?
    # Time t (in sec) = target_idx / token_hz
    # Source idx = time * intent_hz = target_idx * (intent_hz / token_hz)
    
    target_indices = np.arange(target_frames)
    source_indices = np.floor(target_indices * (intent_hz / token_hz)).astype(np.int32)
    
    # Clip to valid range just in case due to rounding or segment length mismatches
    source_indices = np.clip(source_indices, 0, N - 1)
    
    upsampled = intent_matrix[source_indices]
    
    return upsampled
