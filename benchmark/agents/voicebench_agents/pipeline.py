"""LLM backends the reference agents can run against."""

import os
from pathlib import Path

from vision_agents.plugins import openai, stream

# The acceleration pipeline the bench runs by default. The subagent and the skills it may
# run live under agents/accelerated/{pack}/.
DEFAULT_ACCELERATED_STT = "deepgram/flux-general-en"
DEFAULT_ACCELERATED_TTS = "elevenlabs/eleven_v4_turbo"
DEFAULT_ACCELERATED_MODEL = "gemma/gemma-4-26B-A4B-it"
DEFAULT_CUSTOMER_ID = "voicebench"

ACCELERATED_AGENTS = Path(__file__).resolve().parent.parent / "accelerated"

# Names, IDs and addresses the frozen scenarios use. Without them a transcriber hears
# Alvarez as Arborists and last-four 9821 as 9822.
PACK_KEYTERMS: dict[str, list[str]] = {
    "restaurant": ["Alvarez", "Patel", "512-555-0142"],
    "healthcare": [
        "Maya Chen",
        "Leo Chen",
        "Priyanka Radhakrishnan",
        "Radhakrishnan",
        "ABC123456",
        "ABC000111",
        "QW4T9-8821",
        "Oak Street Pharmacy",
        "Westlake Compounding",
    ],
    "telecom": ["4471", "9821", "Cedar", "14 Cedar Lane", "840"],
}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return default


def _customer_id() -> str | None:
    """The customer id for a local router, or None for one behind Stream's proxy.

    A customer id makes the SDK drop STREAM_API_KEY, which the proxy needs.
    """
    if os.environ.get("STREAM_ACCELERATION_AUTHENTICATE", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        return None
    return _env("STREAM_ACCELERATION_CUSTOMER_ID", DEFAULT_CUSTOMER_ID)


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
        customer_id=_customer_id(),
    )


def build_llm(kind: str, pack: str):
    """Return a realtime OpenAI LLM or an acceleration bundle.

    Accelerated modality names come from VOICEBENCH_MODEL / _STT / _TTS /
    _VOICE. Unset names use the bench's default pipeline.
    Accelerated also names the stored pack config, so the subagent and skills
    sync_accelerated_pack wrote are the ones the harness runs.
    """
    if kind == "accelerated":
        return stream.Accelerated(
            config=pack,
            model=_env("VOICEBENCH_MODEL", DEFAULT_ACCELERATED_MODEL),
            stt=_env("VOICEBENCH_STT", DEFAULT_ACCELERATED_STT),
            tts=_env("VOICEBENCH_TTS", DEFAULT_ACCELERATED_TTS),
            voice=os.environ.get("VOICEBENCH_VOICE", "").strip(),
            customer_id=_customer_id(),
            keyterms=PACK_KEYTERMS.get(pack, []),
        )
    if kind != "realtime":
        raise ValueError(f"unknown pipeline {kind!r}")
    return openai.Realtime()
