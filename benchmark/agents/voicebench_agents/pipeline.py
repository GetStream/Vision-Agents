"""LLM backends the reference agents can run against."""

import os
from pathlib import Path

from vision_agents.plugins import openai, stream

# As-shipped acceleration pipeline, matching examples/agents/customer_support.
# The skills its subagent may run live under agents/accelerated/{pack}/skills/.
DEFAULT_ACCELERATED_STT = "gemini/gemini-3.5-transcribe-live"
DEFAULT_ACCELERATED_TTS = "inworld/inworld-tts-2-flash"
DEFAULT_ACCELERATED_MODEL = "gemini/gemini-3.5-flash-lite"
DEFAULT_ACCELERATED_SUBAGENT = "openai/gpt-5.6-sol"
DEFAULT_CUSTOMER_ID = "voicebench"

ACCELERATED_AGENTS = Path(__file__).resolve().parent.parent / "accelerated"


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return default


async def sync_accelerated_pack(pack: str) -> None:
    """Store the skills this pack's subagent may run as an agent config.

    The directory holds skills and nothing else, so the config carries no
    instructions of its own and the contract prompt every other target is given
    stays the one the router runs.
    """
    path = ACCELERATED_AGENTS / pack
    if not path.is_dir():
        raise FileNotFoundError(f"accelerated agent directory missing: {path}")
    await stream.sync_agent(
        pack,
        path=str(path),
        customer_id=_env("STREAM_ACCELERATION_CUSTOMER_ID", DEFAULT_CUSTOMER_ID),
    )


def build_llm(kind: str, pack: str):
    """Return a realtime OpenAI LLM or an acceleration bundle.

    Accelerated modality names come from VOICEBENCH_MODEL / _STT / _TTS /
    _SUBAGENT / _VOICE. Unset names use the as-shipped customer_support triple.
    Accelerated also names the stored pack config, so the skills
    sync_accelerated_pack wrote are the ones its subagent may run.
    """
    if kind == "accelerated":
        return stream.Accelerated(
            config=pack,
            model=_env("VOICEBENCH_MODEL", DEFAULT_ACCELERATED_MODEL),
            stt=_env("VOICEBENCH_STT", DEFAULT_ACCELERATED_STT),
            tts=_env("VOICEBENCH_TTS", DEFAULT_ACCELERATED_TTS),
            subagent=_env("VOICEBENCH_SUBAGENT", DEFAULT_ACCELERATED_SUBAGENT),
            voice=os.environ.get("VOICEBENCH_VOICE", "").strip(),
            customer_id=_env("STREAM_ACCELERATION_CUSTOMER_ID", DEFAULT_CUSTOMER_ID),
        )
    if kind != "realtime":
        raise ValueError(f"unknown pipeline {kind!r}")
    return openai.Realtime()
