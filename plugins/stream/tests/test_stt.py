import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import aiohttp
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
from vision_agents.plugins.stream import stt as stt_module

SETTLE = 2.0

ALICE = Participant({}, user_id="alice", id="alice-1")
BOB = Participant({}, user_id="bob", id="bob-1")
CAROL = Participant({}, user_id="carol", id="carol-1")


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
        # dropping makes opening a socket hang up before answering, as a router going away does.
        self.dropping = False
        # refusing_start is the error the router answers a start frame with, as it does a
        # model that cannot take what was asked of it, before closing the socket.
        self.refusing_start = ""
        # started_after is how long the router takes to answer a start frame, as a provider
        # that is slow to open does.
        self.started_after = 0.0
        # silent makes the router read a start frame and answer nothing, as a stuck provider does.
        self.silent = False
        # start_seen is set when a start frame reaches the router.
        self.start_seen = asyncio.Event()
        # hanging_up makes the router close the socket on a start frame without a word.
        self.hanging_up = False
        # answering holds every handshake while it is clear, as a slow router does.
        self.answering = asyncio.Event()
        self.answering.set()
        # asked is set when a handshake reaches the router.
        self.asked = asyncio.Event()
        # handshakes counts the handshakes that reached the router, ended those it is done with.
        self.handshakes = 0
        self.ended = 0
        self._ending = asyncio.Condition()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_get("/v1/stt/stream", self._stream)
        return app

    async def say(self, index: int, text: str) -> None:
        """Send a final transcript down the socket opened `index`th."""
        await self.opened[index].socket.send_json(
            {"type": "transcript", "text": text, "final": True}
        )

    async def fail(self, index: int, error: str) -> None:
        """Send an error frame down the socket opened `index`th."""
        await self.opened[index].socket.send_json({"type": "error", "error": error})

    async def next_heard(self) -> tuple[int, bytes]:
        """Which socket the next audio frame came over, and the frame."""
        return await asyncio.wait_for(self.heard.get(), SETTLE)

    async def settled(self) -> None:
        """Wait until every handshake that reached the router is done with, its socket closed."""
        async with self._ending:
            await asyncio.wait_for(
                self._ending.wait_for(lambda: self.ended == self.handshakes), SETTLE
            )

    async def _stream(self, request: web.Request) -> web.StreamResponse:
        self.handshakes += 1
        try:
            return await self._answer(request)
        finally:
            async with self._ending:
                self.ended += 1
                self._ending.notify_all()

    async def _answer(self, request: web.Request) -> web.StreamResponse:
        self.asked.set()
        await self.answering.wait()
        if self.dropping:
            assert request.transport is not None
            request.transport.close()
            return web.Response()
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
                self.start_seen.set()
                if self.silent:
                    continue
                if self.refusing_start:
                    await socket.send_json(
                        {"type": "error", "error": self.refusing_start}
                    )
                    await socket.send_json({"type": "closed"})
                    await socket.close()
                elif self.hanging_up:
                    await socket.close()
                else:
                    await asyncio.sleep(self.started_after)
                    await socket.send_json(
                        {"type": "started", "provider": "deepgram", "model": "nova-3"}
                    )
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
        later = audio(211)
        await stt.process_audio(later, BOB)
        heard_later = await router.next_heard()

        assert heard_later == (1, later.samples.tobytes()), (
            "Bob is heard once his socket is open"
        )
        assert len(router.opened) == 2
        assert router.opened[1].audio == [f.samples.tobytes() for f in frames + [later]]

    async def test_a_refused_start_is_raised_by_start(self, router: Router):
        router.refusing = True
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")

        with pytest.raises(RouterError):
            await transcriber.start()

    async def test_a_start_cancelled_while_waiting_closes_its_socket(
        self, router: Router
    ):
        router.silent = True
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")
        starting = asyncio.create_task(transcriber.start())
        await asyncio.wait_for(router.start_seen.wait(), SETTLE)

        starting.cancel()
        with pytest.raises(asyncio.CancelledError):
            await starting

        await asyncio.wait_for(router.opened[0].closed.wait(), SETTLE)

    async def test_a_socket_refused_for_a_new_participant_is_reported_once(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []
        reported = asyncio.Event()

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)
            reported.set()

        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.refusing = True
        await stt.process_audio(audio(200), BOB)
        await asyncio.wait_for(reported.wait(), SETTLE)
        router.refusing = False
        await stt.process_audio(audio(201), BOB)
        await stt.process_audio(audio(101), ALICE)
        alices = await router.next_heard()
        await stt.process_audio(audio(300), CAROL)
        carols = await router.next_heard()
        await stt.events.wait()

        assert alices == (0, audio(101).samples.tobytes()), "Alice is still heard"
        assert carols == (1, audio(300).samples.tobytes()), (
            "Bob's socket is not asked again"
        )
        assert router.handshakes == 3, "start, Bob's refused one and Carol's"
        assert len(errors) == 1
        assert isinstance(errors[0].error, RouterError)
        assert len(router.opened) == 2

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

    async def test_a_dropped_handshake_is_reported_once_and_not_raised(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []
        reported = asyncio.Event()

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)
            reported.set()

        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.dropping = True
        await stt.process_audio(audio(200), BOB)
        await asyncio.wait_for(reported.wait(), SETTLE)
        router.dropping = False
        await stt.process_audio(audio(201), BOB)
        await stt.process_audio(audio(101), ALICE)
        alices = await router.next_heard()
        await stt.process_audio(audio(300), CAROL)
        carols = await router.next_heard()
        await stt.events.wait()

        assert alices == (0, audio(101).samples.tobytes()), "Alice is still heard"
        assert carols == (1, audio(300).samples.tobytes()), (
            "Bob's socket is not asked again"
        )
        # aiohttp asks a dropped GET once more before giving up, so Bob's is asked twice.
        assert router.handshakes == 4, "start, Bob's dropped one and Carol's"
        assert len(errors) == 1
        assert isinstance(errors[0].error, aiohttp.ClientError)

    async def test_a_slow_handshake_holds_up_nobody_else_and_loses_nothing(
        self, router: Router, stt: stream.STT
    ):
        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.answering.clear()

        bobs = [audio(level) for level in range(201, 206)]
        for frame in bobs[:3]:
            await asyncio.wait_for(stt.process_audio(frame, BOB), 0.2)
        await asyncio.wait_for(stt.process_audio(audio(101), ALICE), 0.2)
        alices_next = await router.next_heard()
        router.answering.set()
        for frame in bobs[3:]:
            await stt.process_audio(frame, BOB)
        for _ in bobs:
            await router.next_heard()

        assert alices_next == (0, audio(101).samples.tobytes())
        assert len(router.opened) == 2
        assert router.opened[1].audio == [f.samples.tobytes() for f in bobs]

    async def test_closing_during_a_handshake_stops_it_quietly(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)

        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.answering.clear()
        router.asked.clear()
        await stt.process_audio(audio(200), BOB)
        await asyncio.wait_for(router.asked.wait(), SETTLE)

        await stt.close()
        router.answering.set()
        await router.settled()
        await stt.events.wait()

        assert router.heard.empty(), "nothing Bob said reached the router"
        assert errors == [], "a handshake cut short by close is no failure"

    async def test_audio_while_closing_opens_no_socket(
        self, router: Router, stt: stream.STT
    ):
        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.answering.clear()
        router.asked.clear()
        await stt.process_audio(audio(200), BOB)
        await asyncio.wait_for(router.asked.wait(), SETTLE)

        await asyncio.gather(stt.close(), stt.process_audio(audio(300), CAROL))
        router.answering.set()
        await router.settled()

        assert router.handshakes == 2, "start and Bob's, which close cut short"
        assert router.heard.empty(), "nothing Bob or Carol said reached the router"

    async def test_audio_before_start_opens_no_socket(self, router: Router):
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")
        await transcriber.process_audio(audio(100), ALICE)
        await transcriber.start()
        await transcriber.process_audio(audio(101), ALICE)
        heard = await router.next_heard()
        await transcriber.close()

        assert heard == (0, audio(101).samples.tobytes())
        assert len(router.opened) == 1

    async def test_keyterms_are_sent_in_the_start_frame_beside_the_options(
        self, router: Router
    ):
        transcriber = stream.STT(
            "en-low-latency",
            url=router.url,
            customer_id="acme",
            options={"interim": True, "keyterms": ["old"]},
            keyterms=["Stream", "Vision Agents"],
        )
        await transcriber.start()
        await transcriber.close()

        assert router.opened[0].start["stt"] == {
            "interim": True,
            "keyterms": ["Stream", "Vision Agents"],
        }

    async def test_a_start_the_router_refuses_is_raised_by_start_with_its_message(
        self, router: Router
    ):
        router.refusing_start = (
            "microsoft/MAI-Transcribe-2-Streaming does not take keyterms"
        )
        transcriber = stream.STT(
            "microsoft/MAI-Transcribe-2-Streaming",
            url=router.url,
            customer_id="acme",
            keyterms=["Stream"],
        )
        seen: list[object] = []

        @transcriber.events.subscribe
        async def on_event(event: STTErrorEvent | STTConnectedEvent):
            seen.append(event)

        with pytest.raises(RouterError) as raised:
            await transcriber.start()
        await transcriber.events.wait()
        await transcriber.close()

        assert str(raised.value) == router.refusing_start
        assert seen == [], (
            "the refusal is raised, not also reported, and nothing connected"
        )

    async def test_a_router_that_starts_late_still_starts_and_transcribes(
        self, router: Router
    ):
        router.started_after = 0.5
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")

        await transcriber.start()
        await transcriber.process_audio(audio(100), ALICE)
        index, _ = await router.next_heard()
        await router.say(index, "hello")
        transcript = await asyncio.wait_for(transcriber.output.get(), SETTLE)
        await transcriber.close()

        assert isinstance(transcript, Transcript)
        assert transcript.text == "hello"

    async def test_a_socket_closed_before_it_started_is_raised_by_start(
        self, router: Router
    ):
        router.hanging_up = True
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")

        with pytest.raises(RouterError, match="closed the socket before it started"):
            await asyncio.wait_for(transcriber.start(), SETTLE)
        await transcriber.close()

    async def test_a_router_that_never_starts_is_given_up_on_and_its_socket_closed(
        self, router: Router, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(stt_module, "STARTED", 0.2)
        router.silent = True
        transcriber = stream.STT("en-low-latency", url=router.url, customer_id="acme")

        with pytest.raises(TimeoutError, match="within 0.2 s"):
            await asyncio.wait_for(transcriber.start(), SETTLE)
        await asyncio.wait_for(router.opened[0].closed.wait(), SETTLE)
        await transcriber.close()

    async def test_an_error_after_the_start_is_reported_as_an_event(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []
        reported = asyncio.Event()

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)
            reported.set()

        await router.fail(0, "deepgram hung up")
        await asyncio.wait_for(reported.wait(), SETTLE)

        assert [str(e.error) for e in errors] == ["deepgram hung up"]

    async def test_a_later_participants_refusal_is_reported_as_an_event(
        self, router: Router, stt: stream.STT
    ):
        errors: list[STTErrorEvent] = []
        reported = asyncio.Event()

        @stt.events.subscribe
        async def on_error(event: STTErrorEvent):
            errors.append(event)
            reported.set()

        await stt.process_audio(audio(100), ALICE)
        await router.next_heard()
        router.refusing_start = "bob refused"
        await stt.process_audio(audio(200), BOB)
        await asyncio.wait_for(reported.wait(), SETTLE)
        await stt.events.wait()

        assert [str(e.error) for e in errors] == ["bob refused"]

    async def test_empty_keyterms_leave_what_the_options_named(self, router: Router):
        transcriber = stream.STT(
            "en-low-latency",
            url=router.url,
            customer_id="acme",
            options={"keyterms": ["old"]},
            keyterms=[],
        )
        await transcriber.start()
        await transcriber.close()

        assert router.opened[0].start["stt"] == {"keyterms": ["old"]}
