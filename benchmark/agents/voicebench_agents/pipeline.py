"""LLM backends the reference agents can run against."""

import os

from vision_agents.plugins import openai, stream


def build_llm(kind: str = ""):
    """Return a realtime OpenAI LLM or an acceleration bundle.

    Empty kind reads VOICEBENCH_PIPELINE, defaulting to realtime. Accelerated
    modality names come from VOICEBENCH_MODEL / _STT / _TTS / _VOICE; empty
    strings leave the router's defaults.
    """
    if not kind:
        kind = os.environ.get("VOICEBENCH_PIPELINE", "realtime")
    if kind == "accelerated":
        return stream.Accelerated(
            model=os.environ.get("VOICEBENCH_MODEL", ""),
            stt=os.environ.get("VOICEBENCH_STT", ""),
            tts=os.environ.get("VOICEBENCH_TTS", ""),
            voice=os.environ.get("VOICEBENCH_VOICE", ""),
            greeting="Hello, how can I help?",
        )
    if kind != "realtime":
        raise ValueError(f"unknown pipeline {kind!r}")
    return openai.Realtime()
