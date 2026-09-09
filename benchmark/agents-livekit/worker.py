"""LiveKit reference worker for Voicebench.

Reads the contract Voicebench puts in the dispatch metadata, exposes those tools
against the world server, and answers on a STT/LLM/TTS pipeline through LiveKit
Inference. Set VOICEBENCH_LIVEKIT_PIPELINE=realtime for OpenAI Realtime.
"""

import asyncio
import json
import os
import urllib.error
import urllib.request
from typing import Any

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
    inference,
)
from livekit.plugins import openai

load_dotenv()

PIPELINE = os.environ.get("VOICEBENCH_LIVEKIT_PIPELINE", "inference").strip() or "inference"

server = AgentServer(
    host="127.0.0.1",
    port=int(os.environ.get("VOICEBENCH_WORKER_PORT", "8081")),
)


def call_world(world_url: str, tool: str, args: dict[str, Any]) -> dict[str, Any]:
    """POST one tool call to the Voicebench world server."""
    request = urllib.request.Request(
        f"{world_url.rstrip('/')}/v1/session/tools/{tool}",
        data=json.dumps(args).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        return {"error": f"HTTP {exc.code}: {detail}"}
    if not body:
        return {}
    parsed = json.loads(body)
    if isinstance(parsed, dict):
        return parsed
    return {"result": parsed}


def build_tool(world_url: str, schema: dict[str, Any]):
    """Turn one contract tool schema into a LiveKit raw function tool."""
    name = schema["name"]

    @function_tool(
        raw_schema={
            "name": name,
            "description": schema.get("description", ""),
            "parameters": schema.get("parameters")
            or {"type": "object", "properties": {}},
        }
    )
    async def tool(raw_arguments: dict[str, object]) -> dict[str, Any]:
        return await asyncio.to_thread(call_world, world_url, name, dict(raw_arguments))

    return tool


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    if value:
        return value
    return default


def build_session() -> AgentSession:
    """Build the LiveKit session: Inference STT/LLM/TTS, or OpenAI Realtime."""
    if PIPELINE == "realtime":
        return AgentSession(
            llm=openai.realtime.RealtimeModel(
                model=_env("VOICEBENCH_LIVEKIT_MODEL", "gpt-realtime-2"),
                voice=_env("VOICEBENCH_LIVEKIT_VOICE", "marin"),
            )
        )
    if PIPELINE != "inference":
        raise RuntimeError(f"voicebench: unknown LIVEKIT pipeline {PIPELINE!r}")
    tts_model = _env("VOICEBENCH_LIVEKIT_TTS", "inworld/inworld-tts-2-flash")
    voice = _env("VOICEBENCH_LIVEKIT_VOICE", "Ashley")
    tts = inference.TTS(model=tts_model, voice=voice) if voice else inference.TTS(model=tts_model)
    return AgentSession(
        stt=inference.STT(
            model=_env("VOICEBENCH_LIVEKIT_STT", "google/gemini-3.5-transcribe-live"),
            language="en",
        ),
        llm=inference.LLM(model=_env("VOICEBENCH_LIVEKIT_MODEL", "google/gemini-3.5-flash-lite")),
        tts=tts,
    )


@server.rtc_session(agent_name=os.environ.get("LIVEKIT_AGENT_NAME", "voicebench"))
async def entrypoint(ctx: JobContext) -> None:
    if not ctx.job.metadata:
        raise RuntimeError(
            "voicebench: dispatch metadata with world_url, instructions, and tools is required"
        )
    metadata = json.loads(ctx.job.metadata)
    world_url = metadata["world_url"]
    agent = Agent(
        instructions=metadata["instructions"],
        tools=[build_tool(world_url, schema) for schema in metadata["tools"]],
    )
    session = build_session()
    await session.start(agent, room=ctx.room)
    await session.generate_reply(instructions="Greet the caller briefly and wait.")


if __name__ == "__main__":
    cli.run_app(server)
