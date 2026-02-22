from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class SituationModel(Protocol):
    """
    Interface for the Situation layer that summarizes the musical context.
    The Situation layer is responsible for taking raw backing audio (or features)
    and producing a high-level summary of the 'musical situation'.
    """
    def summarize(self, backing_audio: Any) -> Any:
        """
        Summarize the current musical context from backing audio.
        
        Args:
            backing_audio: The input audio segment (usually context).
            
        Returns:
            A representation of the musical situation (e.g., chord, energy, style).
        """
        ...
