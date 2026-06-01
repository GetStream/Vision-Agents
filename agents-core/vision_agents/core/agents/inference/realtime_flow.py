import asyncio
import logging
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Tracer

from ...edge.types import Participant
from ...events import EventManager
from ...llm import Realtime
from ...llm.realtime import (
    RealtimeAgentSpeechEnded,
    RealtimeAgentSpeechStarted,
    RealtimeAgentTranscript,
    RealtimeAudioOutput,
    RealtimeAudioOutputDone,
    RealtimeUserSpeechEnded,
    RealtimeUserSpeechStarted,
    RealtimeUserTranscript,
)
from ...utils.exceptions import log_exceptions
from ...utils.stream import Stream
from ...utils.utils import cancel_and_wait
from ..events import (
    AgentTurnEndedEvent,
    AgentTurnStartedEvent,
    UserTurnEndedEvent,
    UserTurnStartedEvent,
)
from ..transcript import TranscriptStore
from .audio import (
    AudioInputStream,
    AudioOutputChunk,
    AudioOutputStream,
)
from .base import InferenceFlow

if TYPE_CHECKING:
    from vision_agents.core.agents import Conversation

logger = logging.getLogger(__name__)
tracer: Tracer = trace.get_tracer("agents")


class RealtimeInferenceFlow(InferenceFlow):
    def __init__(
        self,
        audio_input: AudioInputStream,
        audio_output: AudioOutputStream,
        llm: Realtime,
        transcripts: TranscriptStore,
        agent_user_id: str,
        conversation: "Conversation",
        events: EventManager,
        otlp_context: Context | None = None,
    ):
        self._transcripts = transcripts
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._llm = llm
        self.events = events
        self.events.register(
            UserTurnStartedEvent,
            UserTurnEndedEvent,
            AgentTurnStartedEvent,
            AgentTurnEndedEvent,
        )
        self._llm_output_processing_task: asyncio.Task | None = None
        self._audio_input_task: asyncio.Task | None = None
        self._audio_output_task: asyncio.Task | None = None

        self._agent_user_id = agent_user_id
        self._conversation = conversation
        self._otlp_context = otlp_context

        self._simple_response_lock = asyncio.Lock()
        self._running = False

    async def start(self):
        # Start tasks to:
        #  1. Read audio frames and pass them to LLM directly
        #  2. Run LLM and write the results (transcripts and output audio) to the queues
        if self._running:
            raise RuntimeError("The flow is already running")
        self._running = True

        llm_output = self._llm.output
        self._audio_input_task = asyncio.create_task(
            self.process_audio_input(self._audio_input)
        )
        self._llm_output_processing_task = asyncio.create_task(
            self.process_llm_output(llm_output, self._audio_output)
        )

    async def stop(self):
        self._running = False

        tasks = [
            self._audio_input_task,
            self._audio_output_task,
            self._llm_output_processing_task,
        ]
        to_cancel = [t for t in tasks if t is not None]
        if to_cancel:
            await cancel_and_wait(*to_cancel)

        self._audio_output.close()

    async def interrupt(self):
        # propagate interrupt() calls everywhere and empty the buffers.
        await self._llm.interrupt()
        self._transcripts.flush_agent_transcript()
        self._transcripts.flush_users_transcripts()

        self._audio_output.clear()
        # flush() signals downstream consumers (e.g. rtc audio track)
        # to drop their buffers and reset the state too.
        await self._audio_output.flush()

    async def process_audio_input(self, audio_input: AudioInputStream) -> None:
        llm = self._llm
        async for chunk in audio_input:
            with log_exceptions(logger, "Error while processing audio input"):
                await llm.process_audio(chunk.data, chunk.participant)

    async def process_llm_output(
        self,
        llm_output: Stream[
            RealtimeAudioOutput
            | RealtimeAudioOutputDone
            | RealtimeUserTranscript
            | RealtimeAgentTranscript
            | RealtimeUserSpeechStarted
            | RealtimeUserSpeechEnded
            | RealtimeAgentSpeechStarted
            | RealtimeAgentSpeechEnded
        ],
        audio_output: AudioOutputStream,
    ) -> None:
        async for item in llm_output:
            with log_exceptions(logger, "Error while processing Realtime output"):
                if isinstance(item, RealtimeAudioOutput):
                    # Received audio from Realtime llm.
                    # Pass it to the audio output.
                    await audio_output.send(AudioOutputChunk(data=item.data))
                elif isinstance(item, RealtimeAudioOutputDone):
                    # The audio output is complete.
                    if item.interrupted:
                        # Audio is complete because it was interrupted.
                        logger.info("👉 Participant barged-in, interrupting the agent")
                        await self.interrupt()
                    else:
                        # The model finished talking, emit a final empty chunk.
                        await audio_output.send(AudioOutputChunk(final=True))
                elif isinstance(item, RealtimeUserSpeechStarted):
                    self.events.send(UserTurnStartedEvent(participant=item.participant))
                elif isinstance(item, RealtimeUserSpeechEnded):
                    self.events.send(UserTurnEndedEvent(participant=item.participant))
                elif isinstance(item, RealtimeAgentSpeechStarted):
                    self.events.send(AgentTurnStartedEvent())
                elif isinstance(item, RealtimeAgentSpeechEnded):
                    self.events.send(AgentTurnEndedEvent(interrupted=item.interrupted))
                elif isinstance(item, RealtimeUserTranscript):
                    # Received a user transcript.
                    # Sync it to the conversation.
                    logger.info(f"🎤 [User transcript]: {item.text}")

                    with tracer.start_as_current_span(
                        "agent.on_realtime_user_transcript",
                        context=self._otlp_context,
                    ):
                        # Finalize any pending agent transcript before starting user's
                        participant = item.participant
                        agent_update = self._transcripts.flush_agent_transcript()
                        if agent_update:
                            await self._conversation.upsert_message(
                                message_id=agent_update.message_id,
                                role="assistant",
                                user_id=agent_update.user_id,
                                content=agent_update.text,
                                completed=True,
                                replace=True,
                            )
                        update = self._transcripts.update_user_transcript(
                            participant_id=participant.id,
                            user_id=participant.user_id,
                            text=item.text,
                            mode=item.mode,
                            drop=item.mode == "final",
                        )
                        if update:
                            await self._conversation.upsert_message(
                                message_id=update.message_id,
                                role="user",
                                user_id=update.user_id,
                                content=update.text,
                                completed=update.mode == "final",
                                replace=update.mode != "delta",
                            )
                elif isinstance(item, RealtimeAgentTranscript):
                    # Received an agent transcript.
                    # Sync it to the conversation.
                    logger.info(f"🎤 [Agent transcript]: {item.text}")

                    with tracer.start_as_current_span(
                        "agent.on_realtime_agent_transcript",
                        context=self._otlp_context,
                    ):
                        # Finalize any pending user transcripts before starting agent's
                        for user_update in self._transcripts.flush_users_transcripts():
                            await self._conversation.upsert_message(
                                message_id=user_update.message_id,
                                role="user",
                                user_id=user_update.user_id,
                                content=user_update.text,
                                completed=True,
                                replace=True,
                            )
                        update = self._transcripts.update_agent_transcript(
                            text=item.text,
                            mode=item.mode,
                            drop=item.mode == "final",
                        )
                        if update:
                            await self._conversation.upsert_message(
                                message_id=update.message_id,
                                role="assistant",
                                user_id=update.user_id,
                                content=update.text,
                                completed=update.mode == "final",
                                replace=update.mode != "delta",
                            )

    async def simple_response(
        self,
        text: str,
        participant: Participant,
        interrupt: bool = True,
    ) -> None:
        """Ask the LLM to reply to an injected instruction.

        The injected ``text`` is forwarded to ``LLM.simple_response`` (which
        inserts it into the LLM's own conversation history as a user message).
        It is intentionally NOT written to the external conversation store —
        only the assistant reply that flows back through ``process_llm_output``
        is recorded there, keeping the transcript limited to "visible" dialogue.

        Args:
            text: Instruction or message to inject.
            participant: Participant the injected turn is attributed to.
            interrupt: If True (default), interrupt the current LLM response
                before forwarding the request to LLM.
                If False, sends the request to LLM straight away as a part of the
                current conversation.
        """
        async with self._simple_response_lock:
            if interrupt:
                await self.interrupt()

            # Realtime models normally don't return anything on simple_response,
            # but we iterate anyway to honor the contract.
            async for _ in self._llm.simple_response(
                text=text, participant=participant
            ):
                ...

    async def say(self, text: str, interrupt: bool = False) -> None:
        """Speak ``text`` directly through TTS, bypassing the LLM.

        Args:
            text: The utterance to speak.
            interrupt: If True, preempt any in-flight LLM turn and clear the
                TTS/audio pipeline before speaking. If False (default), the
                utterance queues behind whatever is already flowing through
                the TTS pipeline — useful for "one sec, checking…" fillers.
        """
        logger.warning('"say" is not supported by Realtime LLMs')

    def set_conversation(self, conversation: "Conversation") -> None:
        self._conversation = conversation
