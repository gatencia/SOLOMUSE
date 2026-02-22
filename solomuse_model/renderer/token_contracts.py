from typing import TypeAlias
import numpy as np

# Type alias for descriptive hinting. 
# Expected shape: [F, Q] where F is frames/timesteps, Q is number of codebooks
DiscreteTokenStream: TypeAlias = np.ndarray

# Type alias for continuous representation hinting.
# Expected shape: [F, D] where F is frames/timesteps, D is dimension features
ContinuousLatentStream: TypeAlias = np.ndarray

def validate_discrete_shape(tokens: np.ndarray, num_codebooks: int):
    """
    Validates that the discrete token array strictly conforms to [F, Q].
    """
    if tokens.ndim != 2:
        raise ValueError(f"Expected 2D discrete token array [F, Q], got shape {tokens.shape}")
    if tokens.shape[1] != num_codebooks:
        raise ValueError(f"Expected {num_codebooks} codebooks, got {tokens.shape[1]}")

def validate_continuous_shape(latents: np.ndarray, code_dim: int):
    """
    Validates that the continuous latent array strictly conforms to [F, D].
    """
    if latents.ndim != 2:
        raise ValueError(f"Expected 2D continuous latent array [F, D], got shape {latents.shape}")
    if latents.shape[1] != code_dim:
        raise ValueError(f"Expected dimension {code_dim}, got {latents.shape[1]}")
