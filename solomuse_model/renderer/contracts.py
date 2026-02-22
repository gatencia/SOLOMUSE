from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class AudioRenderer(Protocol):
    """
    Interface for the Renderer layer that synthesizes audio.
    The Renderer layer takes the intent plan and synthesizes the actual
    soloist audio.
    """
    def render(self, intent_plan: Any) -> Any:
        """
        Render audio based on the planned intent.
        
        Args:
            intent_plan: The output from the Intent layer.
            
        Returns:
            The synthesized audio segment.
        """
        ...
