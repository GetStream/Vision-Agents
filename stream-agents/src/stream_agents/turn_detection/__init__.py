from .turn_detection import (
    TurnEvent,
    TurnEventData,
    BaseTurnDetector,
    TurnDetection,
)
from .fal_turn_detection import FalTurnDetection


# Lazy import for KrispTurnDetection to avoid krisp_audio dependency issues
def _get_krisp_turn_detection():
    try:
        from .krisp.krisp_turn_detection import KrispTurnDetection

        return KrispTurnDetection
    except ImportError:
        return None


# Make KrispTurnDetection available but only import when needed
KrispTurnDetection = _get_krisp_turn_detection()

__all__ = [
    # Base classes and types
    "TurnEvent",
    "TurnEventData",
    "BaseTurnDetector",
    "TurnDetection",
    # Implementations
    "FalTurnDetection",
    "KrispTurnDetection",
]
