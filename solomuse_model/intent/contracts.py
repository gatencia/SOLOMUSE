from typing import Protocol, Any, runtime_checkable

@runtime_checkable
class IntentModel(Protocol):
    """
    Interface for the Intent layer that plans musical gestures.
    The Intent layer takes the situation summary and decides what the
    soloist should 'intend' to play next.
    """
    def plan(self, situation_summary: Any) -> Any:
        """
        Plan the soloist's musical intent based on the current situation.
        
        Args:
            situation_summary: The output from the Situation layer.
            
        Returns:
            A representation of the musical intent (e.g., target notes, phrasing).
        """
        ...
