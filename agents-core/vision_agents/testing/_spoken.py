"""A simulated conversation held out loud over a loopback edge.

The caller gets a TTS voice and an STT ear. Each line is synthesised and
paced into the agent's audio input at real time; the agent's TTS output is
transcribed back, and that transcript is what the judge reads. The text the
agent meant to say is kept next to it so a failure caused by the voice can
be told from one caused by the answer.
"""

import asyncio
import logging
import math
import time
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import numpy as np
from getstream.video.rtc.track_util import PcmData
from vision_agents.core.agents.agents import Agent
from vision_agents.core.edge.types import Participant
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.core.llm.realtime import Realtime
from vision_agents.core.stt.stt import STT, Transcript
from vision_agents.core.tts.tts import TTS
from vision_agents.core.utils.utils import cancel_and_wait

from ._events import RunEvent
from ._loopback import LoopbackEdge, LoopbackMicrophone
from ._session import observe_tool_calls

logger = logging.getLogger(__name__)

# RMS of int16 samples above which a frame counts as sound rather than room noise.
ENERGY_THRESHOLD = 300.0
_POLL = 0.05

CALLER = Participant(original=None, user_id="caller", id="caller")


class CallerError(Exception):
    """The caller's own voice or ears failed, so the trial says nothing about the agent."""


class AgentSilentError(Exception):
    """The agent did not answer within the turn timeout."""


@dataclass
class SpokenLine:
    """What came back after the caller said one line.

    Attributes:
        heard: What the caller's STT made of the agent's reply.
        intended: What the agent's LLM meant to say.
        voice_to_voice_ms: Time from the caller falling silent to the first
            frame of agent audio with energy in it, or ``None`` if the agent
            was never heard.
        duration_ms: Time from the caller falling silent to the end of the reply.
        events: Tool calls the agent made while replying.
    """

    heard: str
    intended: str
    voice_to_voice_ms: float | None
    duration_ms: float
    events: list[RunEvent]


class SpokenConversation:
    """Joins an agent to a loopback call and talks to it through TTS and STT.

    Args:
        agent: Agent under test. Its edge must be a ``LoopbackEdge``.
        voice: TTS that speaks the caller's lines.
        ears: STT that transcribes the agent's replies.
        turn_timeout: Seconds to wait for the agent to answer a line.
        settle: Seconds of quiet after the agent's last sound or transcript
            before the reply counts as finished. The reply is only considered
            once the agent's LLM has produced its final text (unless the LLM
            is a ``Realtime`` one, which gives no such signal), so the quiet
            window is a heuristic for the tail of the speech, not for the
            answer itself. Speech with no LLM text behind it, such as a bare
            ``agent.say()``, is heard but never counts as an answer.
    """

    def __init__(
        self,
        agent: Agent,
        voice: TTS,
        ears: STT,
        turn_timeout: float = 60.0,
        settle: float = 1.5,
    ) -> None:
        if not isinstance(agent.edge, LoopbackEdge):
            raise ValueError(
                "Audio mode needs an Agent built with a LoopbackEdge "
                f"(vision_agents.testing.LoopbackEdge), got {type(agent.edge).__name__}"
            )
        self._agent = agent
        self._edge: LoopbackEdge = agent.edge
        self._voice = voice
        self._ears = ears
        self._turn_timeout = turn_timeout
        self._settle = settle
        self._agent_participant = Participant(
            original=None, user_id=agent.agent_user.id or "agent", id="agent"
        )
        self._realtime = isinstance(agent.llm, Realtime)
        self._stack = AsyncExitStack()
        self._mic: LoopbackMicrophone | None = None
        self._listen_task: asyncio.Task[None] | None = None
        self._tool_events: list[RunEvent] = []
        self._heard: list[str] = []
        self._intended: list[str] = []
        self._spoke_until = 0.0
        self._first_sound_at: float | None = None
        self._last_sound_at = 0.0
        self._last_transcript_at = 0.0
        self._failure: Exception | None = None

    async def __aenter__(self) -> "SpokenConversation":
        await self.start()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def start(self) -> None:
        """Open the caller's ears and voice, then join the agent to the call."""
        try:
            await self._ears.start()
            await self._voice.start()
        except Exception as exc:
            await self.close()
            raise CallerError(
                f"Could not start the caller's voice or ears: {exc}"
            ) from exc
        try:
            self._listen_task = asyncio.create_task(self._listen())
            self._edge.on_agent_audio(self._hear)
            self._agent.events.subscribe(self._on_agent_reply)
            self._stack.enter_context(
                observe_tool_calls(self._agent.llm, self._tool_events)
            )
            self._mic = self._edge.microphone(CALLER)
            call = await self._agent.create_call("default", f"simulation-{uuid4()}")
            await self._stack.enter_async_context(self._agent.join(call))
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Leave the call and release the caller's voice and ears."""
        try:
            await self._stack.aclose()
        finally:
            if self._listen_task is not None:
                await cancel_and_wait(self._listen_task)
                self._listen_task = None
            await self._voice.close()
            await self._ears.close()

    async def say(self, text: str) -> SpokenLine:
        """Speak one line into the call and wait for the whole of what comes back.

        Raises:
            CallerError: If the caller's voice or ears failed.
            AgentSilentError: If the agent did not answer within ``turn_timeout``.
        """
        if self._mic is None:
            raise RuntimeError(
                "SpokenConversation not started. Use 'async with' or call start()."
            )
        if self._failure is not None:
            raise CallerError(f"The caller's ears failed: {self._failure}")
        if self._agent.closed:
            raise AgentSilentError("The call ended before the caller could speak")
        self._heard.clear()
        self._intended.clear()
        self._tool_events.clear()
        self._first_sound_at = None
        # Nothing the agent says while the caller is still talking is an answer.
        self._spoke_until = math.inf

        speech = await self._speak(text)
        # play() returns when the caller stopped talking, which is the moment
        # every reply is timed from and the earliest the agent could answer.
        try:
            self._spoke_until = await self._mic.play(speech)
        except RuntimeError as exc:
            raise AgentSilentError(
                "The call ended while the caller was speaking"
            ) from exc
        deadline = self._spoke_until + self._turn_timeout

        while True:
            await asyncio.sleep(_POLL)
            if self._failure is not None:
                raise CallerError(f"The caller's ears failed: {self._failure}")
            if self._agent.closed:
                raise AgentSilentError("The call ended before the agent answered")
            now = time.monotonic()
            last_activity = max(
                self._spoke_until, self._last_sound_at, self._last_transcript_at
            )
            answered = bool(self._intended) or self._realtime
            if self._heard and answered and now - last_activity >= self._settle:
                break
            if now >= deadline:
                raise AgentSilentError(
                    f"Agent did not reply within {self._turn_timeout}s"
                )

        return SpokenLine(
            heard=" ".join(self._heard),
            intended=" ".join(self._intended),
            voice_to_voice_ms=(
                None
                if self._first_sound_at is None
                else (self._first_sound_at - self._spoke_until) * 1000
            ),
            duration_ms=(last_activity - self._spoke_until) * 1000,
            events=list(self._tool_events),
        )

    async def _speak(self, text: str) -> PcmData:
        """Synthesise one line of the caller's speech."""
        speech: PcmData | None = None
        try:
            async for chunk in self._voice.send_iter(text, CALLER):
                if chunk.data is None:
                    continue
                speech = chunk.data if speech is None else speech.append(chunk.data)
        except Exception as exc:
            raise CallerError(f"The caller's voice failed: {exc}") from exc
        if speech is None or speech.samples.size == 0:
            raise CallerError("The caller's voice said nothing")
        return speech

    async def _hear(self, pcm: PcmData) -> None:
        """The caller listening: agent audio goes into the caller's own STT."""
        now = time.monotonic()
        if _rms(pcm) >= ENERGY_THRESHOLD:
            self._last_sound_at = now
            if self._first_sound_at is None and now > self._spoke_until:
                self._first_sound_at = now
        try:
            await self._ears.process_audio(pcm, self._agent_participant)
        except Exception as exc:
            self._failure = exc

    async def _listen(self) -> None:
        """Collect what the caller's STT made of the agent's speech."""
        async for item in self._ears.output:
            if isinstance(item, Transcript) and item.final:
                self._last_transcript_at = time.monotonic()
                if item.text.strip():
                    self._heard.append(item.text.strip())

    async def _on_agent_reply(self, event: LLMResponseFinalEvent) -> None:
        if event.text.strip():
            self._intended.append(event.text.strip())


def _rms(pcm: PcmData) -> float:
    samples = pcm.samples.astype(np.float64)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))
