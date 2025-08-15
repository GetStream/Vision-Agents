"""
Stream Agents Package

This package provides AI agent functionality for Stream video calls.
It uses the stream-py package for STT and TTS services.
"""

from .agents import Agent, Tool, PreProcessor, LLM

# Import STT, TTS, STS from stream-py package (they are imported in agents.py)
try:
    from getstream.plugins.common.stt import STT
    from getstream.plugins.common.tts import TTS
    from getstream.plugins.common.sts import STS
except ImportError:
    # Fallback if stream-py is not installed
    STT = None
    TTS = None
    STS = None

# TurnDetectionAdapter removed - use TurnDetection protocol directly

__all__ = [
    "Agent",
    "Tool",
    "PreProcessor",
    "LLM",
    "STT",
    "TTS",
    "STS",
]

__version__ = "0.1.0"

#   To resume this session: cursor-agent --resume=a524bb8f-60f1-4520-9abf-0ee07d3d1f12
