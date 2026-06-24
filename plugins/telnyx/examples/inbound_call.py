"""Minimal inbound Telnyx phone example.

Run after starting ngrok and routing your Telnyx number to a Call Control App:

    NGROK_URL=example.ngrok-free.app uv run plugins/telnyx/examples/inbound_call.py
"""

from __future__ import annotations

import asyncio
import argparse
import logging
import os
import uuid

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream, telnyx
from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

from telnyx_helpers import (
    cleanup_telnyx_example_setup,
    prepare_telnyx_example_setup,
    TelnyxClient,
    TelnyxConfig,
    media_stream_url,
    preflight_inbound,
    require_env,
    TelnyxSetupError,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])
call_registry = telnyx.TelnyxCallRegistry()
telnyx_client: TelnyxClient | None = None
telnyx_config: TelnyxConfig | None = None


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(status_code=500, content={"detail": str(exc)})


async def create_agent() -> tuple[Agent, StreamEdge]:
    edge = getstream.Edge()
    agent = Agent(
        edge=edge,
        agent_user=User(id="ai-agent", name="AI Assistant"),
        instructions=(
            "Speak English. Keep replies short and natural. You are answering "
            "an inbound Telnyx test call through Vision Agents. Start by saying "
            "this is a quick inbound Telnyx bridge test and ask whether the "
            "audio is clear."
        ),
        llm=gemini.Realtime(),
    )
    return agent, edge


async def prepare_call(call_id: str, from_number: str):
    agent, edge = await create_agent()
    sanitized_number = "".join(ch for ch in from_number if ch.isalnum()) or call_id
    phone_user = User(
        name=f"Inbound Telnyx call from {from_number}",
        id=f"phone-{sanitized_number}",
    )
    await edge.create_users([agent.agent_user, phone_user])
    stream_call = await agent.create_call("default", call_id)
    logger.info("Prepared Stream call %s for %s", call_id, from_number)
    return agent, phone_user, stream_call


async def wait_for_start(
    stream: telnyx.TelnyxMediaStream, timeout: float = 10.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not stream.has_started:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("Telnyx media stream did not start in time")
        await asyncio.sleep(0.05)


@app.post("/telnyx/events")
async def telnyx_events(request: Request):
    if telnyx_client is None or telnyx_config is None:
        raise RuntimeError("Telnyx example was not initialized")

    data = await request.json()
    event_type = data.get("data", {}).get("event_type")
    payload = data.get("data", {}).get("payload", {})
    logger.info("Telnyx webhook event: %s", event_type or data)

    if event_type == "call.initiated" and payload.get("direction") == "incoming":
        call_control_id = payload["call_control_id"]
        from_number = payload.get("from", "unknown")
        call_id = str(uuid.uuid4())
        telnyx_call = call_registry.create(
            call_id,
            webhook_data=data,
            prepare=lambda: prepare_call(call_id, from_number),
        )
        stream_url = media_stream_url(
            telnyx_config.ngrok_url,
            call_id,
            telnyx_call.token,
        )
        response = await asyncio.to_thread(
            telnyx_client.answer_call,
            call_control_id,
            stream_url=stream_url,
        )
        logger.info("Answered inbound Telnyx call: %s", response.get("data", response))

    return {"ok": True}


@app.websocket("/telnyx/media/{call_id}/{token}")
async def media_stream(websocket: WebSocket, call_id: str, token: str):
    telnyx_call = call_registry.validate(call_id, token)
    logger.info("Media stream connected for inbound call %s", call_id)

    telnyx_stream = telnyx.TelnyxMediaStream(websocket)
    await telnyx_stream.accept()
    telnyx_call.telnyx_stream = telnyx_stream
    stream_task = asyncio.create_task(telnyx_stream.run())

    try:
        agent, phone_user, stream_call = await telnyx_call.await_prepare()
        telnyx_call.stream_call = stream_call

        await telnyx.attach_phone_to_call(stream_call, telnyx_stream, phone_user.id)
        await wait_for_start(telnyx_stream)

        async with agent.join(stream_call, participant_wait_timeout=0):
            await agent.simple_response(
                text="Greet the caller and ask whether the inbound Telnyx audio is clear."
            )
            await stream_task
    finally:
        if not stream_task.done():
            stream_task.cancel()
        call_registry.remove(call_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal inbound Telnyx call.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--phone-number-id",
        default=None,
        help="Telnyx phone number resource ID. Defaults to TELNYX_PHONE_NUMBER_ID.",
    )
    parser.add_argument(
        "--phone-number",
        default=None,
        help="Telnyx inbound number. Defaults to TELNYX_PHONE_NUMBER.",
    )
    parser.add_argument(
        "--call-control-app-id",
        default=None,
        help="Existing Telnyx Call Control App ID. Defaults to TELNYX_CALL_CONTROL_APP_ID.",
    )
    parser.add_argument(
        "--ngrok-url",
        default=None,
        help="Public ngrok hostname. Defaults to NGROK_URL or local ngrok autodetection.",
    )
    parser.add_argument(
        "--setup-telnyx",
        action="store_true",
        help="Create a temporary Call Control App, route the phone number, and restore it on exit.",
    )
    return parser.parse_args()


def main() -> None:
    global telnyx_client, telnyx_config

    args = parse_args()
    values = require_env(
        ["STREAM_API_KEY", "STREAM_API_SECRET", "GOOGLE_API_KEY", "TELNYX_API_KEY"]
    )
    telnyx_client = TelnyxClient(values["TELNYX_API_KEY"])
    setup = prepare_telnyx_example_setup(
        telnyx_client,
        api_key=values["TELNYX_API_KEY"],
        phone_number=args.phone_number or os.environ.get("TELNYX_PHONE_NUMBER"),
        ngrok_url=args.ngrok_url or os.environ.get("NGROK_URL"),
        call_control_app_id=(
            args.call_control_app_id or os.environ.get("TELNYX_CALL_CONTROL_APP_ID")
        ),
        phone_number_id=(
            args.phone_number_id or os.environ.get("TELNYX_PHONE_NUMBER_ID")
        ),
        setup_telnyx=args.setup_telnyx,
        route_phone_number=True,
    )
    telnyx_config = setup.config
    resolved_phone_number_id = setup.phone_number_id or (
        args.phone_number_id or os.environ.get("TELNYX_PHONE_NUMBER_ID")
    )
    if not resolved_phone_number_id:
        raise TelnyxSetupError(
            "Missing TELNYX_PHONE_NUMBER_ID. Pass `--setup-telnyx` to discover "
            "and route the Telnyx number automatically."
        )

    try:
        preflight_inbound(
            telnyx_client,
            config=telnyx_config,
            telnyx_phone_number_id=resolved_phone_number_id,
        )

        logger.info("Inbound Telnyx runner ready. Call %s.", telnyx_config.phone_number)
        uvicorn.run(app, host=args.host, port=args.port)
    finally:
        cleanup_telnyx_example_setup(telnyx_client, setup)


if __name__ == "__main__":
    main()
