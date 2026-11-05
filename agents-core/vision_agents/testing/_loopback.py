"""In-process ``EdgeTransport`` with no network in it.

Microphones write 16 kHz mono PCM into the call in 20 ms packets at real
time, and whatever the agent publishes is paced out at the rate it would be
heard rather than the rate it was synthesised. The pacing is what makes turn
taking mean anything: a TTS provider streams an utterance far faster than it
is spoken.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import aiortc
import numpy as np
from getstream.video.rtc import AudioStreamTrack
from getstream.video.rtc.track_util import AudioFormat, PcmData
from vision_agents.core.agents.conversation import InMemoryConversation
from vision_agents.core.edge.call import Call
from vision_agents.core.edge.edge_transport import EdgeTransport
from vision_agents.core.edge.events import (
    AudioReceivedEvent,
    CallEndedEvent,
    ParticipantJoinedEvent,
    ParticipantLeftEvent,
)
from vision_agents.core.edge.types import Connection, Participant, User
from vision_agents.core.utils.utils import cancel_and_wait

if TYPE_CHECKING:
    from vision_agents.core.agents.agents import Agent

logger = logging.getLogger(__name__)

PLUGIN_NAME = "loopback"
SAMPLE_RATE = 16000
CHUNK_MS = 20
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000

AudioHandler = Callable[[PcmData], Awaitable[None]]


@dataclass
class LoopbackCall:
    """Minimal ``Call`` for the loopback transport."""

    id: str


class LoopbackMicrophone:
    """One participant's end of the call.

    Always sends something, because a real call carries the room even when
    nobody is talking: silence between utterances, and utterances spliced
    into that stream.
    """

    def __init__(self, edge: "LoopbackEdge", participant: Participant) -> None:
        self._edge = edge
        self._participant = participant
        self._playing: np.ndarray | None = None
        self._offset = 0
        self._finished: asyncio.Future[float] | None = None
        self._task = asyncio.create_task(self._run())

    @property
    def participant(self) -> Participant:
        return self._participant

    async def play(self, pcm: PcmData) -> float:
        """Splice an utterance into the stream and wait until all of it was sent.

        Returns:
            ``time.monotonic()`` when the last sample left the microphone,
            which is the moment the speaker stopped talking.

        Raises:
            RuntimeError: If an utterance is already playing or the microphone is stopped.
        """
        if self._playing is not None:
            raise RuntimeError("Microphone is already playing an utterance")
        if self._task.done():
            raise RuntimeError("Microphone is stopped")
        samples = pcm.resample(SAMPLE_RATE, 1).to_int16().samples.reshape(-1)
        if samples.size == 0:
            return time.monotonic()
        self._finished = asyncio.get_running_loop().create_future()
        self._offset = 0
        self._playing = samples
        return await self._finished

    async def stop(self) -> None:
        """Close this end of the call. A pending ``play`` returns right away."""
        await cancel_and_wait(self._task)
        if self._finished is not None and not self._finished.done():
            self._finished.set_result(time.monotonic())
        self._playing = None

    async def _run(self) -> None:
        interval = CHUNK_MS / 1000
        next_at = time.monotonic()
        while True:
            next_at += interval
            await asyncio.sleep(max(0.0, next_at - time.monotonic()))
            self._edge._receive_audio(self._participant, self._next_chunk())

    def _next_chunk(self) -> np.ndarray:
        chunk = np.zeros(CHUNK_SAMPLES, dtype=np.int16)
        if self._playing is None:
            return chunk
        end = min(self._offset + CHUNK_SAMPLES, len(self._playing))
        chunk[: end - self._offset] = self._playing[self._offset : end]
        self._offset = end
        if end >= len(self._playing):
            self._playing = None
            if self._finished is not None and not self._finished.done():
                self._finished.set_result(time.monotonic())
        return chunk


class LoopbackEdge(EdgeTransport[Call]):
    """EdgeTransport that joins an ``Agent`` to a call with no network.

    Open a :meth:`microphone` to put a participant on the call and speak
    into it; register a handler with :meth:`on_agent_audio` to hear what
    the agent publishes, as 20 ms frames of 16 kHz mono PCM delivered at
    real time. Custom events the agent sends are kept in ``custom_events``.
    """

    def __init__(self) -> None:
        super().__init__()
        self._call: Call | None = None
        self._mics: list[LoopbackMicrophone] = []
        self._audio_handlers: list[AudioHandler] = []
        self._pump_task: asyncio.Task[None] | None = None
        self._participant_present = asyncio.Event()
        self._idle_since = 0.0
        self._connection: LoopbackConnection | None = None
        self.custom_events: list[dict[str, Any]] = []

    async def authenticate(self, user: User) -> None:
        return None

    async def create_call(self, call_id: str, **kwargs: Any) -> LoopbackCall:
        return LoopbackCall(id=call_id)

    def create_audio_track(self) -> AudioStreamTrack:
        """The call carries 16 kHz mono; the track resamples whatever TTS emits."""
        return AudioStreamTrack(
            sample_rate=SAMPLE_RATE, channels=1, format=AudioFormat.S16
        )

    def open_demo(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def join(self, agent: "Agent", call: Call, **kwargs: Any) -> Connection:
        self._call = call
        if not self._mics:
            self._idle_since = time.time()
        for mic in self._mics:
            self.events.send(
                ParticipantJoinedEvent(
                    plugin_name=PLUGIN_NAME, participant=mic.participant, call=call
                )
            )
        self._connection = LoopbackConnection(self)
        return self._connection

    async def publish_tracks(
        self,
        audio_track: Optional[aiortc.MediaStreamTrack],
        video_track: Optional[aiortc.MediaStreamTrack],
    ) -> None:
        """Start hearing the agent: read its track at real time and fan frames out."""
        if audio_track is not None and self._pump_task is None:
            self._pump_task = asyncio.create_task(self._pump(audio_track))

    async def create_conversation(
        self, call: Call, user: User, instructions: str
    ) -> InMemoryConversation:
        return InMemoryConversation(instructions=instructions, messages=[])

    def add_track_subscriber(self, track_id: str) -> None:
        return None

    async def send_custom_event(self, data: dict[str, Any]) -> None:
        self.custom_events.append(data)

    async def close(self) -> None:
        """Stop every microphone and stop hearing the agent. Safe to call twice."""
        mics, self._mics = self._mics, []
        for mic in mics:
            await mic.stop()
        self._participant_present.clear()
        if self._pump_task is not None:
            await cancel_and_wait(self._pump_task)
            self._pump_task = None
        self._connection = None

    def microphone(self, participant: Participant) -> LoopbackMicrophone:
        """Put ``participant`` on the call and return their microphone.

        Needs a running event loop: the microphone starts sending right away.
        """
        mic = LoopbackMicrophone(self, participant)
        self._mics.append(mic)
        self._idle_since = 0.0
        self._participant_present.set()
        if self._call is not None:
            self.events.send(
                ParticipantJoinedEvent(
                    plugin_name=PLUGIN_NAME, participant=participant, call=self._call
                )
            )
        return mic

    def on_agent_audio(self, handler: AudioHandler) -> None:
        """Register a coroutine that is handed every frame the agent publishes."""
        self._audio_handlers.append(handler)

    async def hang_up(self) -> None:
        """End the call from the far side: every participant leaves, then the call ends."""
        call = self._call
        mics, self._mics = self._mics, []
        for mic in mics:
            await mic.stop()
            if call is not None:
                self.events.send(
                    ParticipantLeftEvent(
                        plugin_name=PLUGIN_NAME, participant=mic.participant, call=call
                    )
                )
        self._participant_present.clear()
        self._idle_since = time.time()
        self._call = None
        if call is not None:
            self.events.send(CallEndedEvent(plugin_name=PLUGIN_NAME, call=call))

    def _receive_audio(self, participant: Participant, samples: np.ndarray) -> None:
        pcm = PcmData(
            samples=samples,
            sample_rate=SAMPLE_RATE,
            format=AudioFormat.S16,
            channels=1,
            participant=participant,
        )
        self.events.send(
            AudioReceivedEvent(
                plugin_name=PLUGIN_NAME, pcm_data=pcm, participant=participant
            )
        )

    async def _pump(self, track: aiortc.MediaStreamTrack) -> None:
        """Read the agent's track the way a peer would: one paced 20 ms frame at a time."""
        try:
            while True:
                frame = await track.recv()
                pcm = PcmData.from_av_frame(frame).resample(SAMPLE_RATE, 1)
                for handler in self._audio_handlers:
                    try:
                        await handler(pcm)
                    except Exception:
                        logger.exception("Error while hearing the agent's audio")
        except aiortc.MediaStreamError:
            logger.debug("Agent audio track ended")


class LoopbackConnection(Connection):
    """Connection for the loopback transport."""

    def __init__(self, transport: LoopbackEdge) -> None:
        super().__init__()
        self._transport = transport

    def idle_since(self) -> float:
        return self._transport._idle_since

    async def wait_for_participant(self, timeout: Optional[float] = None) -> None:
        await asyncio.wait_for(
            self._transport._participant_present.wait(), timeout=timeout
        )

    async def close(self, timeout: float = 2.0) -> None:
        await self._transport.close()
