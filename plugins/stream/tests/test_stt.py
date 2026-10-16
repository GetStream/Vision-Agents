import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import numpy as np
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import Transcript
from vision_agents.core.stt.events import (
    STTConnectedEvent,
    STTDisconnectedEvent,
    STTErrorEvent,
)
from vision_agents.plugins import stream
from vision_agents.plugins.stream import RouterError

SETTLE = 2.0

ALICE = Participant({}, user_id="alice", id="alice-1")
BOB = Participant({}, user_id="bob", id="bob-1")


def audio(level: int) -> PcmData:
    """Twenty milliseconds at 16 kHz, every sample `level`, so a frame says who sent it."""
    return PcmData(
        samples=np.full(320, level, dtype=np.int16),
        sample_rate=16_000,
        channels=1,
        format=AudioFormat.S16,
    )


@dataclass
class Opened:
    """One transcription socket the router accepted, and what came over it."""

    socket: web.WebSocketResponse
    start: dict[str, Any] = field(default_factory=dict)
    audio: list[bytes] = field(default_factory=list, repr=False)
    closed: asyncio.Event = field(default_factory=asyncio.Event)


class Router:
    """A stand-in for the router's transcription socket.

    It is a real server rather than a stub object, so what the plugin sends is what a
    router would receive. Like the router, it knows nothing of participants: a socket is
    all it has to tell one speaker's audio from another's.
    """

    def __init__(self):
        self.url = ""
        self.opened: list[Opened] = []
        self.heard: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        # refusing makes opening a socket answer 429, as a customer over its limit is.
        self.refusing = False

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/stt/stream", self._stream)
        return app

    async def say(self, index: int, text: str) -> None:
        """Send a final transcript down the socket opened `index`th."""
        await self.opened[index].socket.send_json(
            {"type": "transcript", "text": text, "final": True}
        )

    async def next_heard(self) -> tuple[int, bytes]:
        """Which socket the next audio frame came over, and the frame."""
        return await asyncio.wait_for(self.heard.get(), SETTLE)

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        if self.refusing:
            return web.json_response(
                status=429,
                data={
                    "error": {
                        "message": "no more sessions today",
                        "type": "rate_limited",
                        "code": "rate_limited",
                        "doc_url": "https://getstream.io/agents/docs/api/errors/#rate_limited",
                    }
                },
            )
        socket = web.WebSocketResponse()
        await socket.prepare(request)
        opened = Opened(socket=socket)
        self.opened.append(opened)
        index = len(self.opened) - 1

        async for message in socket:
            if message.type == WSMsgType.TEXT:
                opened.start = json.loads(message.data)
            elif message.type == WSMsgType.BINARY:
                opened.audio.append(message.data)
                await self.heard.put((index, message.data))
        opened.closed.set()
        return socket


class TestSTT:
    @pytest.fixture
    async def router(self) -> AsyncIterator[Router]:
        fake = Router()
        server = TestServer(fake.app())
        await server.start_server()
        fake.url = str(server.make_url("")).rstrip("/")
        yield fake
        await server.close()

    @pytest.fixture
    async def stt(self, router: Router) -> AsyncIterator[stream.STT]:
        transcriber = stream.STT(
            "en-low-latency", url=router.url, customer_id="acme", tags={"team": "a"}
        )
        await transcriber.start()
        yield transcriber
        await transcriber.close()

    async def test_a_late_transcript_is_credited_to_whoever_said_it(
        self, router: Router, stt: stream.STT
    ):
        await stt.process_audio(audio(100), ALICE)
        alices, _ = await router.next_heard()
        await stt.process_audio(audio(200), BOB)
        await router.next_heard()

        await router.say(alices, "What would it take for you to stay with Stream?")
        transcript = await asyncio.wait_for(stt.output.get(), SETTLE)

        assert isinstance(transcript, Transcript)
        assert transcript.participant == ALICE

    async def test_one_participant_is_heard_on_one_socket_as_before(
        self, router: Router, stt: stream.STT
    ):
        frames = [audio(level) for level in (100, 200, 300)]
        for frame in frames:
            await stt.process_audio(frame, ALICE)
        for _ in frames:
            await router.next_heard()

        assert len(router.opened) == 1
        assert router.opened[0].start == {
            "type": "start",
            "config_id": "",
            "target": "en-low-latency",
            "languages": [],
            "tags": {"team": "a"},
            "sample_rate": 16000,
            "stt": {},
        }
        assert router.opened[0].audio == [f.samples.tobytes() for f in frames]

    async def test_a_new_participant_sending_at_once_loses_and_reorders_nothing(
        self, router: Router, stt: stream.STT
    ):
        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()

        frames = [audio(level) for level in range(201, 211)]
        await asyncio.gather(*(stt.process_audio(f, BOB) for f in frames))
        for _ in frames:
            await router.next_heard()

        assert len(router.opened) == 2
        assert router.opened[1].audio == [f.samples.tobytes() for f in frames]

    async def test_a_refused_start_is_raised_by_start(self, router: Router):
        router.refusing = True
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")

        with pytest.raises(RouterError):
            await transcriber.start()

    async def test_a_socket_refused_for_a_new_participant_is_reported_once(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)

        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.refusing = True
        await stt.process_audio(audio(200), BOB)
        await stt.process_audio(audio(201), BOB)
        await stt.process_audio(audio(101), ALICE)
        index, _ = await router.next_heard()
        await stt.events.wait()

        assert index == 0, "the participant already heard is heard on"
        assert len(errors) == 1
        assert isinstance(errors[0].error, RouterError)
        assert len(router.opened) == 1

    async def test_closing_closes_every_socket_and_reports_one_disconnection(
        self, router: Router
    ):
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")
        seen: list[type] = []

        @transcriber.events.subscribe
        async def on_connection(event: STTConnectedEvent | STTDisconnectedEvent):
            seen.append(type(event))

        await transcriber.start()
        await transcriber.process_audio(audio(100), ALICE)
        await transcriber.process_audio(audio(200), BOB)
        await router.next_heard()
        await router.next_heard()

        await transcriber.close()
        for opened in router.opened:
            await asyncio.wait_for(opened.closed.wait(), SETTLE)
        await transcriber.events.wait()

        assert len(router.opened) == 2
        assert seen == [STTConnectedEvent, STTDisconnectedEvent]
