import asyncio
import logging
from typing import Any, Optional

import aiohttp
from getstream.video.rtc.track_util import PcmData
from vision_agents.core import stt
from vision_agents.core.edge.types import Participant
from vision_agents.core.stt import TranscriptResponse
from vision_agents.core.utils.utils import cancel_and_wait

from ._backend import Backend
from ._errors import RouterError
from ._routerconfig import ensure_router
from ._socket import Socket

logger = logging.getLogger(__name__)

# SAMPLE_RATE is what the router transcribes at, and what every provider behind it wants.
SAMPLE_RATE = 16_000

# STARTED is how long start waits for the router to say a provider is transcribing. The
# router tries the candidates one after another, each under its own connect bound
# (acceleration/internal/stt/<provider>/<provider>.go): 15 s for cartesia (cartesia.go:191),
# inworld and muse (muse.go:246), 30 s for most, 60 s for parakeet (parakeet.go:125),
# whose Baseten cold starts are slow. 90 s lets the slowest of them fail over
# to a 30 s one before a router that never answers is given up on.
STARTED = 90.0


class STT(stt.STT):
    """Transcription routed through the acceleration backend.

    For a pipeline that stays in Python: the turns, the model and the conversation are all
    here, and only the transcribing is somewhere else. Failover and cost tracking work
    exactly as they do inside a session, because it is the same router doing the routing.

    Each participant is transcribed on a socket of their own, and what comes back on it is
    credited to them. The router's socket carries no participant, and a transcript lags
    the audio it is of, so on a shared socket a line still arriving when someone else
    starts talking would go to the wrong speaker. The socket opened by `start` goes to
    whoever speaks first; everyone after gets a new one on their first frame, opened in
    the background so nobody else's audio waits for it, with their frames held in order
    until it is open. Sockets are kept until `close`.
    """

    def __init__(
        self,
        target: str = "",
        language: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
        url: Optional[str] = None,
        customer_id: Optional[str] = None,
        config_id: str = "",
        options: Optional[dict[str, Any]] = None,
        backend: Optional[Backend] = None,
        keyterms: Optional[list[str]] = None,
        region: Optional[str] = None,
    ):
        """Route transcription to `target`.

        Args:
            target: A `provider/model` name or a capability shortcut such as
                `en-low-latency`. It may instead be held in the named config.
            language: A language hint, which narrows the candidates.
            tags: Cost labels carried onto every request.
            url: The router's base URL. Defaults to `STREAM_ACCELERATION_URL`.
            region: The hosted router in one region, such as `us-east4`, instead of
                a URL. Defaults to `STREAM_ACCELERATION_REGION`.
            customer_id: Who the work is billed to. Defaults to
                `STREAM_ACCELERATION_CUSTOMER_ID`.
            config_id: A stored router config to take the options from, by name or by id.
            options: Per-call overrides of that config's stt block. Usually built by
                `Router.stt.realtime`.
            backend: A backend to use as it is, instead of building one from `url` and
                `customer_id`.
            keyterms: Words the transcriber would otherwise get wrong, such as names and
                member IDs, at most 100. A model that takes none refuses them, and `start`
                raises that refusal. Empty leaves whatever `options` or the config named.
        """
        super().__init__(provider_name="stream")
        self.model = target
        self.language = language
        self.tags = tags or {}
        self.config_id = config_id
        self.options = options or {}
        if keyterms:
            self.options = {**self.options, "keyterms": keyterms}
        self.backend = backend or Backend(
            url=url, customer_id=customer_id, region=region
        )

        # _unclaimed is the socket opened by start, until somebody speaks into it.
        self._unclaimed: Optional[Socket] = None
        self._sockets: dict[str, Socket] = {}
        self._speakers: dict[Socket, Participant] = {}
        # _held is the audio of each participant whose socket is still opening.
        self._held: dict[str, list[bytes]] = {}
        self._opening: list[asyncio.Task] = []
        self._readers: list[asyncio.Task] = []

    async def start(self):
        """Open the first socket and wait until the router is transcribing on it.

        The router bounds each provider's connect but not the walk down the candidates,
        and a router that is stuck answers nothing, so the wait is bounded by `STARTED`.

        Raises:
            RouterError: if the router refused the socket or what to transcribe on it,
                such as a model that takes no keyterms, or closed it before it started.
            TimeoutError: if the router said nothing within `STARTED` seconds.
        """
        await super().start()
        await ensure_router(self.config_id, self.backend)
        self._unclaimed = self._socket()
        started: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        await self._open(self._unclaimed, started)
        try:
            # Not shielded: _read checks started.done(), so the future wait_for cancels on
            # a timeout is left alone, and nothing is set on it that nobody would read.
            await asyncio.wait_for(started, STARTED)
        except asyncio.TimeoutError as timeout:
            await self._unclaimed.close()
            raise TimeoutError(
                f"the router did not start transcribing within {STARTED:g} s"
            ) from timeout
        except BaseException:
            await self._unclaimed.close()
            raise
        self._on_connected()

    async def __aenter__(self) -> "STT":
        """Start transcribing, for a caller holding the session itself."""
        await self.start()
        return self

    async def __aexit__(self, *exception) -> None:
        await self.close()

    async def process_audio(
        self,
        pcm_data: PcmData,
        participant: Participant,
    ):
        """Send audio to be transcribed on the participant's own socket, resampled to what
        the router expects."""
        if self.closed or not self.started:
            return

        payload = pcm_data.resample(SAMPLE_RATE, 1).samples.tobytes()
        socket = self._sockets.get(participant.id) or self._claim(participant)
        held = self._held.get(participant.id)
        if held is not None:
            held.append(payload)
        elif socket.open:
            await socket.send_audio(payload)

    async def close(self):
        """Close every socket, including one still opening, and stop reading them."""
        # closed is set first, so audio arriving while this runs opens no socket it would miss.
        self.closed = True
        await cancel_and_wait(*self._opening)
        self._opening.clear()
        self._held.clear()
        for reader in self._readers:
            await cancel_and_wait(reader)
        self._readers.clear()
        sockets = list(self._sockets.values())
        if self._unclaimed is not None:
            sockets.append(self._unclaimed)
            self._unclaimed = None
        self._sockets.clear()
        self._speakers.clear()
        await asyncio.gather(*(socket.close() for socket in sockets))
        if sockets:
            self._on_disconnected()
        await super().close()

    def _socket(self) -> Socket:
        return Socket(self.backend.socket("/v1/stt/stream"), self.backend.headers)

    async def _open(
        self, socket: Socket, started: Optional[asyncio.Future[None]] = None
    ) -> None:
        """Connect `socket`, say what to transcribe on it and start reading it.

        `started` is resolved by the router's first answer: `started`, or its `error`.
        """
        await socket.connect()
        await socket.send(
            {
                "type": "start",
                "config_id": self.config_id,
                "target": self.model,
                "languages": [self.language] if self.language else [],
                "tags": self.tags,
                "sample_rate": SAMPLE_RATE,
                "stt": self.options,
            }
        )
        self._readers.append(asyncio.create_task(self._read(socket, started)))

    def _claim(self, participant: Participant) -> Socket:
        """The socket `participant` is heard on: the one start opened if nobody has it yet,
        or a new one, opened in the background while their frames are held."""
        unclaimed, self._unclaimed = self._unclaimed, None
        socket = unclaimed or self._socket()
        self._sockets[participant.id] = socket
        self._speakers[socket] = participant
        if unclaimed is None:
            self._held[participant.id] = []
            self._opening.append(
                asyncio.create_task(self._open_for(participant, socket))
            )
        return socket

    async def _open_for(self, participant: Participant, socket: Socket) -> None:
        """Open a participant's socket, then send what they said meanwhile, in order.

        A socket the router refuses is reported once and kept, closed, so the rest of that
        participant's audio is dropped as a closed socket's always was, rather than each
        frame asking again.
        """
        held = self._held[participant.id]
        context = f"opening a socket for {participant.id}"
        try:
            await self._open(socket)
            context = f"sending what {participant.id} said while it opened"
            while held:
                await socket.send_audio(held.pop(0))
        except (RouterError, aiohttp.ClientError, OSError) as e:
            await socket.close()
            self._emit_error_event(e, context=context)
        finally:
            self._held.pop(participant.id, None)

    async def _read(
        self, socket: Socket, started: Optional[asyncio.Future[None]] = None
    ) -> None:
        """Turn what the router says on `socket` into transcripts on the output stream.

        Until `started` is resolved, an `error` frame is the refusal `start` raises, so it
        is not reported a second time as an event.
        """
        async for frame in socket.frames():
            if isinstance(frame, bytes):
                continue
            if started is not None and not started.done():
                if frame.get("type") == "started":
                    started.set_result(None)
                    continue
                if frame.get("type") == "error":
                    started.set_exception(RouterError(frame.get("error", "")))
                    continue
            self._received(frame, self._speakers.get(socket))
        if started is not None and not started.done():
            started.set_exception(
                RouterError("the router closed the socket before it started")
            )

    def _received(self, frame: dict[str, Any], speaker: Optional[Participant]) -> None:
        kind = frame.get("type", "")

        if kind == "transcript":
            text = frame.get("text", "")
            if not text or speaker is None:
                return
            self._emit_transcript_event(
                text=text,
                participant=speaker,
                response=TranscriptResponse(
                    confidence=frame.get("confidence"),
                    language=frame.get("language"),
                    processing_time_ms=frame.get("processing_time_ms"),
                    audio_duration_ms=frame.get("audio_duration_ms"),
                    model_name=frame.get("model"),
                ),
                mode="final" if frame.get("final") else "delta",
            )
        elif kind == "error":
            self._emit_error_event(
                RuntimeError(frame.get("error", "")), context=frame.get("context", "")
            )
