from typing import TypedDict, List
import numpy as np

class IntentFrameV1(TypedDict):
    """
    Structured control features extracted from a solo audio frame.
    """
    play_prob: float            # Phrase vs rest (0..1)
    density_proxy: float        # Note/event activity estimate (0..1)
    register_norm: float        # Estimated pitch height proxy (0..1)
    tension_proxy: float        # Harmonic distance proxy [solo chroma vs backing] (0..1)
    dynamics_norm: float        # Local energy/loudness (0..1)
    onset_prob: float           # Onset emphasis (0..1)
    phrase_boundary_hint: float # Likely boundary/rest marker (0..1)

class IntentSequenceV1(TypedDict):
    """
    A sequence of intent frames extracted from an audio segment.
    """
    version: str
    sr: int
    duration_s: float
    hz: int                     # Flame rate (e.g., 10 Hz)
    frames: List[IntentFrameV1]
