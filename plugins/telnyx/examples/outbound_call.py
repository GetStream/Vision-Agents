"""Minimal outbound Telnyx phone example.

Run from the repository root or this directory after starting ngrok:

    NGROK_URL=example.ngrok-free.app uv run plugins/telnyx/examples/outbound_call.py --to +15551234567
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
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream, telnyx
from vision_agents.plugins.getstream.stream_edge_transport import StreamEdge

from telnyx_helpers import (
    cleanup_telnyx_example_setup,
    prepare_telnyx_example_setup,
    TelnyxClient,
    TelnyxConfig,
    media_stream_url,
    preflight_outbound,
    require_env,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()
call_registry = telnyx.TelnyxCallRegistry()
call_done: asyncio.Event | None = None


@app.post("/telnyx/events")
async def telnyx_events(request: Request):
    data = await request.json()
    event_type = data.get("data", {}).get("event_type")
    logger.info("Telnyx webhook event: %s", event_type or data)
    return {"ok": True}


async def create_agent() -> tuple[Agent, StreamEdge]:
    edge = getstream.Edge()
    agent = Agent(
        edge=edge,
        agent_user=User(id="ai-agent", name="AI Assistant"),
        instructions=(
            "Speak English. Keep replies short and natural. You are making a "
            "Telnyx test call through Vision Agents. Start by saying this is a "
            "quick Telnyx media streaming test and ask whether the audio is clear."
        ),
        llm=gemini.Realtime(),
    )
    return agent, edge


async def prepare_call(call_id: str):
    agent, edge = await create_agent()
    phone_user = User(name=f"Telnyx outbound {call_id[:8]}", id=f"phone-{call_id}")
    await edge.create_users([agent.agent_user, phone_user])
    stream_call = await agent.create_call("default", call_id)
    logger.info("Prepared Stream call %s", call_id)
    return agent, phone_user, stream_call


async def wait_for_start(
    stream: telnyx.TelnyxMediaStream, timeout: float = 10.0
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while not stream.has_started:
        if asyncio.get_running_loop().time() >= deadline:
            raise TimeoutError("Telnyx media stream did not start in time")
        await asyncio.sleep(0.05)


@app.websocket("/telnyx/media/{call_id}/{token}")
async def media_stream(websocket: WebSocket, call_id: str, token: str):
    telnyx_call = call_registry.validate(call_id, token)
    logger.info("Media stream connected for outbound call %s", call_id)

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
                text="Start the call and ask whether the Telnyx audio is clear."
            )
            await stream_task
    finally:
        if not stream_task.done():
            stream_task.cancel()
        call_registry.remove(call_id)
        if call_done is not None:
            call_done.set()


async def run_with_server(
    *,
    client: TelnyxClient,
    config: TelnyxConfig,
    to_number: str,
    from_number: str,
    host: str,
    port: int,
) -> None:
    global call_done

    call_done = asyncio.Event()
    call_id = str(uuid.uuid4())
    telnyx_call = call_registry.create(call_id, prepare=lambda: prepare_call(call_id))
    stream_url = media_stream_url(config.ngrok_url, call_id, telnyx_call.token)

    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level="info"))
    server_task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.1)

    logger.info("Dialing %s from %s", to_number, from_number)
    response = await asyncio.to_thread(
        client.dial_call,
        connection_id=config.call_control_app_id,
        from_number=from_number,
        to_number=to_number,
        stream_url=stream_url,
    )
    logger.info("Telnyx dial accepted: %s", response.get("data", response))
    await call_done.wait()
    server.should_exit = True
    await server_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal outbound Telnyx call.")
    parser.add_argument("--to", dest="to_number", required=True)
    parser.add_argument(
        "--from",
        dest="from_number",
        default=None,
        help="Telnyx caller ID. Defaults to TELNYX_PHONE_NUMBER.",
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
        help="Create a temporary Call Control App for this run and delete it on exit.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--skip-verified-destination-check",
        action="store_true",
        help="Skip Telnyx verified-number preflight for unrestricted accounts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values = require_env(
        ["STREAM_API_KEY", "STREAM_API_SECRET", "GOOGLE_API_KEY", "TELNYX_API_KEY"]
    )
    client = TelnyxClient(values["TELNYX_API_KEY"])
    setup = prepare_telnyx_example_setup(
        client,
        api_key=values["TELNYX_API_KEY"],
        phone_number=args.from_number or os.environ.get("TELNYX_PHONE_NUMBER"),
        ngrok_url=args.ngrok_url or os.environ.get("NGROK_URL"),
        call_control_app_id=(
            args.call_control_app_id or os.environ.get("TELNYX_CALL_CONTROL_APP_ID")
        ),
        setup_telnyx=args.setup_telnyx,
    )
    try:
        preflight_outbound(
            client,
            config=setup.config,
            to_number=args.to_number,
            check_verified_destination=not args.skip_verified_destination_check,
        )
        asyncio.run(
            run_with_server(
                client=client,
                config=setup.config,
                to_number=args.to_number,
                from_number=setup.config.phone_number,
                host=args.host,
                port=args.port,
            )
        )
    finally:
        cleanup_telnyx_example_setup(client, setup)


if __name__ == "__main__":
    main()
