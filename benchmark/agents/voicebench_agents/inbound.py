"""Inbound Telnyx runner shared by the three vertical agents."""

import asyncio
import contextlib
import logging
import os
import uuid
from collections.abc import Awaitable, Callable

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import JSONResponse
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from vision_agents.core import Agent, User
from vision_agents.plugins import telnyx
from vision_agents.plugins.telnyx.example_helpers import (
    TelnyxClient,
    TelnyxConfig,
    TelnyxSetupError,
    cleanup_telnyx_example_setup,
    media_stream_url,
    parse_verified_telnyx_webhook,
    preflight_inbound,
    prepare_telnyx_example_setup,
    require_env,
    require_telnyx_public_key,
)

logger = logging.getLogger(__name__)

CreateAgent = Callable[..., Awaitable[Agent]]


class _Runtime:
    client: TelnyxClient | None = None
    config: TelnyxConfig | None = None
    public_key: str | None = None


def build_app(create_agent: CreateAgent, runtime: _Runtime) -> FastAPI:
    load_dotenv()
    app = FastAPI()
    app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=["*"])
    call_registry = telnyx.TelnyxCallRegistry()

    @app.exception_handler(Exception)
    async def global_exception_handler(_request: Request, exc: Exception):
        logger.exception("Unhandled exception")
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    async def prepare_call(call_id: str):
        agent = await create_agent()
        phone_user = User(name=f"Benchmark call {call_id[:8]}", id=f"phone-{call_id}")
        await agent.edge.create_users([agent.agent_user, phone_user])
        stream_call = await agent.create_call("default", call_id)
        return agent, phone_user, stream_call

    @app.post("/telnyx/events")
    async def telnyx_events(request: Request):
        if runtime.client is None or runtime.config is None or runtime.public_key is None:
            raise RuntimeError("Telnyx agent was not initialized")

        data = await parse_verified_telnyx_webhook(request, runtime.public_key)
        event_type = data.get("data", {}).get("event_type")
        payload = data.get("data", {}).get("payload", {})
        logger.info("Telnyx webhook event: %s", event_type)

        if event_type == "call.initiated" and payload.get("direction") == "incoming":
            call_control_id = payload["call_control_id"]
            call_id = str(uuid.uuid4())
            telnyx_call = call_registry.create(
                call_id,
                webhook_data=data,
                prepare=lambda: prepare_call(call_id),
            )
            stream_url = media_stream_url(
                runtime.config.ngrok_url, call_id, telnyx_call.token
            )
            await asyncio.to_thread(
                runtime.client.answer_call,
                call_control_id,
                stream_url=stream_url,
            )
        return {"ok": True}

    @app.websocket("/telnyx/media/{call_id}/{token}")
    async def media_stream(websocket: WebSocket, call_id: str, token: str):
        telnyx_call = call_registry.validate(call_id, token)
        telnyx_stream = telnyx.TelnyxMediaStream(websocket)
        await telnyx_stream.accept()
        telnyx_call.telnyx_stream = telnyx_stream
        stream_task = asyncio.create_task(telnyx_stream.run())
        try:
            agent, phone_user, stream_call = await telnyx_call.await_prepare()
            telnyx_call.stream_call = stream_call
            await telnyx.attach_phone_to_call(stream_call, telnyx_stream, phone_user.id)
            deadline = asyncio.get_running_loop().time() + 10.0
            while not telnyx_stream.has_started:
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError("Telnyx media stream did not start in time")
                await asyncio.sleep(0.05)
            async with agent.join(stream_call, participant_wait_timeout=0):
                await agent.simple_response(text="Greet the caller briefly and wait.")
                await stream_task
        finally:
            if not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task
            call_registry.remove(call_id)

    return app


def serve(
    create_agent: CreateAgent,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    phone_number_id: str | None = None,
    phone_number: str | None = None,
    call_control_app_id: str | None = None,
    ngrok_url: str | None = None,
    setup_telnyx: bool = False,
) -> None:
    logging.basicConfig(level=logging.INFO)
    runtime = _Runtime()
    app = build_app(create_agent, runtime)
    values = require_env(
        ["STREAM_API_KEY", "STREAM_API_SECRET", "GOOGLE_API_KEY", "TELNYX_API_KEY"]
    )
    runtime.public_key = require_telnyx_public_key()
    runtime.client = TelnyxClient(values["TELNYX_API_KEY"])
    setup = prepare_telnyx_example_setup(
        runtime.client,
        api_key=values["TELNYX_API_KEY"],
        phone_number=phone_number or os.environ.get("TELNYX_PHONE_NUMBER"),
        ngrok_url=ngrok_url or os.environ.get("NGROK_URL"),
        call_control_app_id=call_control_app_id or os.environ.get("TELNYX_CALL_CONTROL_APP_ID"),
        phone_number_id=phone_number_id or os.environ.get("TELNYX_PHONE_NUMBER_ID"),
        setup_telnyx=setup_telnyx,
        route_phone_number=True,
    )
    runtime.config = setup.config
    resolved_phone_number_id = setup.phone_number_id or (
        phone_number_id or os.environ.get("TELNYX_PHONE_NUMBER_ID")
    )
    if not resolved_phone_number_id:
        raise TelnyxSetupError(
            "Missing TELNYX_PHONE_NUMBER_ID. Pass --setup-telnyx to route the number."
        )
    try:
        preflight_inbound(
            runtime.client,
            config=setup.config,
            telnyx_phone_number_id=resolved_phone_number_id,
        )
        logger.info("Vertical agent ready")
        uvicorn.run(app, host=host, port=port)
    finally:
        cleanup_telnyx_example_setup(runtime.client, setup)
