import asyncio
import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.trace import Tracer
from vision_agents.core.agents.events import (
    AgentTurnEndedEvent,
    AgentTurnStartedEvent,
    UserTranscriptEvent,
    UserTurnEndedEvent,
    UserTurnStartedEvent,
)
from vision_agents.core.agents.transcript import TranscriptStore
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM, Realtime
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.stt import STT
from vision_agents.core.stt.stt import Transcript
from vision_agents.core.tts import TTS
from vision_agents.core.tts.tts import (
    TTSInput,
    TTSInputEnd,
    TTSOutputChunk,
    TTSOutputEnd,
)
from vision_agents.core.turn_detection import TurnDetector, TurnEnded, TurnStarted
from vision_agents.core.utils.exceptions import log_exceptions
from vision_agents.core.utils.stream import Stream
from vision_agents.core.utils.text import sanitize_text
from vision_agents.core.utils.tokenizer import TTSSentenceTokenizer
from vision_agents.core.utils.utils import cancel_and_wait

from .audio import AudioInputStream, AudioOutputChunk, AudioOutputStream
from .base import InferenceFlow
from .llm_turn import LLMTurn

if TYPE_CHECKING:
    from vision_agents.core.agents import Conversation

logger = logging.getLogger(__name__)
tracer: Tracer = trace.get_tracer("agents")


class TranscribingInferenceFlow(InferenceFlow):
    def __init__(
        self,
        audio_input: AudioInputStream,
        audio_output: AudioOutputStream,
        llm: LLM,
        transcripts: TranscriptStore,
        agent_user_id: str,
        conversation: "Conversation",
        events: EventManager,
        stt: STT | None = None,
        turn_detector: TurnDetector | None = None,
        tts: TTS | None = None,
        otlp_context: Context | None = None,
    ):
        if isinstance(llm, Realtime):
            raise ValueError(
                f"Realtime LLMs are not supported by {self.__class__.__name__}"
            )
        self._llm = llm
        self.events = events
        self.events.register(
            UserTurnStartedEvent,
            UserTurnEndedEvent,
            UserTranscriptEvent,
            AgentTurnStartedEvent,
            AgentTurnEndedEvent,
            LLMResponseFinalEvent,
        )

        if turn_detector and stt and stt.turn_detection:
            logger.warning(
                "STT already provides turn detection; ignoring the TurnDetector plugin."
            )
            turn_detector = None
        self._turn_detector = turn_detector
        # Neither the STT nor an external TurnDetector drives turn boundaries,
        # so a final Transcript is treated as the commit signal on its own.
        self._no_turn_detection = turn_detector is None and (
            not stt or not stt.turn_detection
        )

        self._transcripts = transcripts
        self._audio_input = audio_input
        self._audio_output = audio_output
        self._stt_output_task: asyncio.Task | None = None
        self._stt = stt

        self._turn_detection_task: asyncio.Task | None = None
        self._llm_turn: LLMTurn | None = None
        self._llm_output = Stream[LLMResponseDelta | LLMResponseFinal]()
        self._llm_output_processing_task: asyncio.Task | None = None

        self._audio_input_task: asyncio.Task | None = None
        self._tts = tts
        self._tts_task: asyncio.Task | None = None
        self._tts_input = Stream[TTSInput | TTSInputEnd]()
        self._tts_output = Stream[TTSOutputChunk | TTSOutputEnd]()
        self._audio_output_task: asyncio.Task | None = None
        self._tts_tokenizer = TTSSentenceTokenizer()

        self._agent_user_id = agent_user_id
        self._conversation = conversation
        self._otlp_context = otlp_context

        self._say_lock = asyncio.Lock()
        self._simple_response_lock = asyncio.Lock()

        self._running = False

    async def start(self):
        if self._running:
            raise RuntimeError("The flow is already running")
        self._running = True

        # Start input audio processing if STT is not None
        # (it can be None for backwards compatibility with the old event-based Agent)
        if self._stt:
            stt_output = self._stt.output
            if self._turn_detector:
                turndetector_output = self._turn_detector.output
                self._turn_detection_task = asyncio.create_task(
                    self.process_turn_detection(turndetector_output, stt_output)
                )

            self._audio_input_task = asyncio.create_task(
                self.process_audio_input(self._audio_input)
            )
            self._stt_output_task = asyncio.create_task(
                self.process_stt_output(stt_output, self._llm_output)
            )
        self._llm_output_processing_task = asyncio.create_task(
            self.process_llm_output(self._llm_output, self._tts_input)
        )

        # TTS can be None for backwards compatibility with the old event-based Agent
        if self._tts:
            self._tts_task = asyncio.create_task(
                self.process_tts(self._tts_input, self._tts_output)
            )
            self._audio_output_task = asyncio.create_task(
                self.write_audio_output(self._tts_output, self._audio_output)
            )

    async def stop(self):
        self._running = False
        if self._llm_turn:
            await self._llm_turn.cancel()
            self._llm_turn = None

        tasks = [
            self._audio_input_task,
            self._audio_output_task,
            self._llm_output_processing_task,
            self._tts_task,
            self._stt_output_task,
            self._turn_detection_task,
        ]
        to_cancel = [t for t in tasks if t is not None]
        if to_cancel:
            await cancel_and_wait(*to_cancel)

        self._llm_output.close()
        self._tts_input.close()
        self._tts_output.close()
        self._audio_output.close()

    async def interrupt(self):
        if self._llm_turn is not None:
            await self._llm_turn.cancel()
            self._llm_turn = None
        else:
            await self._llm.interrupt()
        self._transcripts.flush_users_transcripts()
        self._llm_output.clear()
        self._tts_input.clear()
        self._tts_tokenizer.flush()
        if self._tts:
            await self._tts.interrupt()
        self._tts_output.clear()
        self._tts_output.send_nowait(TTSOutputEnd(interrupted=True))

        self._audio_output.clear()
        # flush() signals downstream consumers (e.g. rtc audio track)
        # to drop their buffers and reset the state too.
        await self._audio_output.flush()

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
            interrupt: If True (default), preempt any in-flight LLM turn before
                starting the new one. If False, drop silently when a turn is
                already in flight (best-effort; avoids cascade cancellations
                from periodic callers).
        """
        async with self._simple_response_lock:
            llm_turn = self._llm_turn
            busy = (
                llm_turn is not None
                and llm_turn.started
                and not llm_turn.finalized
                and not llm_turn.cancelled
            )
            if busy and not interrupt:
                logger.debug(
                    "simple_response dropped: LLM turn already in flight (text=%r)",
                    text,
                )
                return
            if busy:
                await self.interrupt()

            llm_turn = LLMTurn(
                transcript=text, participant=participant, events=self.events
            )
            llm_turn.start(llm=self._llm)
            llm_turn.confirm()
            llm_turn.finalize(self._llm_output)
            self._llm_turn = llm_turn

    async def say(self, text: str, interrupt: bool = False) -> None:
        """Speak ``text`` directly through TTS, bypassing the LLM.

        Args:
            text: The utterance to speak.
            interrupt: If True, preempt any in-flight LLM turn and clear the
                TTS/audio pipeline before speaking. If False (default), the
                utterance queues behind whatever is already flowing through
                the TTS pipeline — useful for "one sec, checking…" fillers.
        """
        async with self._say_lock:
            if interrupt:
                await self.interrupt()

            # Send both shapes so either TTS mode can pick the one it understands:
            #   - streaming TTS consumes delta=True through the tokenizer, then
            #     TTSInputEnd flushes the remainder;
            #   - non-streaming TTS consumes delta=False directly.
            await self._tts_input.send(TTSInput(text=text, delta=True))
            await self._tts_input.send(TTSInput(text=text, delta=False))
            await self._tts_input.send(TTSInputEnd())

    def set_conversation(self, conversation: "Conversation") -> None:
        self._conversation = conversation

    async def process_audio_input(self, audio_input: AudioInputStream) -> None:
        turn_detector = self._turn_detector
        stt = self._stt
        if stt is None:
            return

        async for chunk in audio_input:
            with log_exceptions(logger, "Error while processing audio input"):
                if turn_detector:
                    # TurnDetector.process_audio is expected to be non-blocking,
                    # so it's safe to simply call it in the loop
                    await turn_detector.process_audio(chunk.data, chunk.participant)
                await stt.process_audio(chunk.data, chunk.participant)

    async def process_turn_detection(
        self,
        turn_detector_output: Stream[TurnStarted | TurnEnded],
        stt_output: Stream[TurnStarted | TurnEnded | Transcript],
    ) -> None:
        """
        Process turn detection outputs.

        TurnDetector plugins run asynchronously, and they write their output to the Stream.
        This task reads

        Args:
            turn_detector_output:
            stt_output:

        Returns:

        """
        async for item in turn_detector_output:
            with log_exceptions(logger, "Error while processing turn detection output"):
                await stt_output.send(item)

    async def process_stt_output(
        self,
        stt_output: Stream[TurnStarted | TurnEnded | Transcript],
        llm_output: Stream[LLMResponseDelta | LLMResponseFinal],
    ) -> None:
        """
        Process the Transcripts and turn detection outputs.

        This task:
        - accumulates STT transcripts
        - interrupts the pipeline when TurnStarted is received
        - keeps track of the user's speech using TurnEnded events
        - finalizes the turn and emits LLM responses downstream when user is done
          talking and when the final transcript is ready

        Notes:
            - The LLM turns are confirmed immediately after receivng TurnEnded
              regardless of their transcript.
              They may be received long before the final transcript is avaialble.


        """
        async for event in stt_output:
            with log_exceptions(logger, "Error while processing STT output"):
                participant = event.participant
                if self._llm_turn and self._llm_turn.finalized:
                    self._llm_turn = None

                if isinstance(event, TurnStarted):
                    logger.info(
                        "👉 Participant %s barged-in, interrupting the agent",
                        event.participant.user_id,
                    )
                    self.events.send(UserTurnStartedEvent(participant=participant))
                    await self.interrupt()

                elif isinstance(event, Transcript):
                    if event.final:
                        # Final transcript is ready
                        logger.info(f"🎤 [Transcript Complete]: {event.text}")
                        self.events.send(
                            UserTranscriptEvent(
                                text=event.text or "",
                                participant=participant,
                            )
                        )

                        # Update the bufferred transcript and mark it as "final".
                        self._transcripts.update_user_transcript(
                            participant_id=participant.id,
                            user_id=participant.user_id,
                            text=event.text,
                            mode="final",
                            drop=False,
                        )

                        with tracer.start_as_current_span(
                            "agent.on_stt_transcript_event_sync_conversation",
                            context=self._otlp_context,
                        ):
                            await self._conversation.upsert_message(
                                message_id=str(uuid4()),
                                role="user",
                                user_id=participant.user_id,
                                content=event.text or "",
                                completed=True,
                                replace=True,  # Replace any partial transcripts
                                original=event,
                            )

                    else:
                        # Partial transcript is ready
                        logger.info(f"🎤 [Transcript Partial]: {event.text}")
                        self._transcripts.update_user_transcript(
                            participant_id=participant.id,
                            user_id=participant.user_id,
                            text=event.text,
                            mode=event.mode,
                        )

                elif isinstance(event, TurnEnded):
                    # The user finished speaking.
                    # Get the transcript
                    logger.info(
                        "👉 Participant %s finished speaking",
                        event.participant.user_id,
                    )
                    buffer = self._transcripts.get_buffer(
                        participant_id=participant.id,
                        user_id=participant.user_id,
                    )
                    transcript = buffer.text.strip() if buffer else ""

                    if not event.eager and self._llm_turn:
                        # Confirm the current LLM turn if it's not an eager event
                        self._llm_turn.confirm()
                    elif not event.eager and not self._llm_turn:
                        # Create and confrim a new turn if it's not started yet.
                        # This may happen when the utterance is short (e.g. "Hello")
                        # and the user stopped speaking, but the STT is slow, and we
                        # haven't received any transcripts yet.
                        self._llm_turn = LLMTurn(
                            transcript=transcript,
                            participant=event.participant,
                            events=self.events,
                        )
                        self._llm_turn.confirm()
                    elif event.eager and transcript:
                        # It's an eager turn, which me means we can send the transcript
                        # to the LLM and wait for it to be confirmed.
                        # Cancel the running turn if the transcript is different.
                        if self._llm_turn and self._llm_turn.transcript != transcript:
                            await self._llm_turn.cancel()

                        # Start a new LLM turn
                        self._llm_turn = LLMTurn(
                            transcript=transcript,
                            participant=event.participant,
                            events=self.events,
                        )
                        logger.info(
                            '🤖 Starting eager LLM turn for transcript "%s"', transcript
                        )
                        self._llm_turn.start(llm=self._llm)

                # This section is common for all events processed by this task.
                buffer = self._transcripts.get_buffer(
                    participant_id=participant.id,
                    user_id=participant.user_id,
                )
                transcript = buffer.text.strip() if buffer else ""

                # Transcript is empty, skip processing
                if not buffer or not transcript:
                    continue

                # There's an in-flight LLM response and
                # the transcript has changed, cancel it
                llm_turn = self._llm_turn
                if llm_turn is not None and llm_turn.transcript != transcript:
                    await llm_turn.cancel()

                # Start a new turn after getting a final transcript
                # if it's not running already.
                if buffer.final and (
                    not llm_turn or llm_turn.cancelled or not llm_turn.started
                ):
                    # Store the confirmed state because the turn will be reset here.
                    confirmed = llm_turn and llm_turn.confirmed
                    # Start a new turn for the final transcript.
                    llm_turn = LLMTurn(
                        transcript=transcript,
                        participant=event.participant,
                        events=self.events,
                    )
                    logger.info(
                        '🤖 Starting non-eager LLM turn for transcript "%s"', transcript
                    )
                    llm_turn.start(llm=self._llm)
                    # Confirm the turn if the previous one was confirmed, or
                    # when no turn detection is wired up — in that mode a
                    # final Transcript is itself the end-of-turn signal.
                    if confirmed or self._no_turn_detection:
                        llm_turn.confirm()
                    self._llm_turn = llm_turn

                # Finalize the LLM turn and stream the LLM results if:
                # - the transcript is confirmed and final
                # - the turn is confirmed
                if (
                    buffer.final
                    and llm_turn
                    and llm_turn.transcript == buffer.text
                    and llm_turn.confirmed
                    and not llm_turn.finalized
                ):
                    llm_turn.finalize(llm_output)
                    self.events.send(UserTurnEndedEvent(participant=event.participant))
                    buffer.reset()

    async def process_llm_output(
        self,
        llm_output: Stream[LLMResponseDelta | LLMResponseFinal],
        tts_input: Stream[TTSInput | TTSInputEnd],
    ):
        async for item in llm_output:
            with log_exceptions(logger, "Error while processing audio input"):
                if isinstance(item, LLMResponseDelta):
                    # Process the delta response
                    text = sanitize_text(item.delta or "")
                    logger.info(f"🤖 [LLM response delta]: {text}")
                    # Update the conversation with a new delta
                    await self._conversation.upsert_message(
                        message_id=item.item_id,
                        role="assistant",
                        user_id=self._agent_user_id,
                        content=text,
                        content_index=item.content_index,
                        completed=False,  # Still streaming
                    )
                    # Send the delta chunk to TTS.
                    await tts_input.send(TTSInput(text=text, delta=True))
                elif isinstance(item, LLMResponseFinal):
                    # Process the final response
                    text = sanitize_text(item.text) or ""
                    logger.info(f"🤖 [LLM response final]: {text}")
                    await self._conversation.upsert_message(
                        message_id=item.item_id,
                        role="assistant",
                        user_id=self._agent_user_id,
                        content=text,
                        completed=True,
                        replace=True,  # Replace any partial content from deltas
                    )
                    # Send the full response so non-streaming TTS can speak it,
                    # then signal end so streaming TTS can flush tokenizer state.
                    await tts_input.send(TTSInput(text=item.text, delta=False))
                    await tts_input.send(TTSInputEnd())

    async def process_tts(
        self,
        tts_input: Stream[TTSInput | TTSInputEnd],
        tts_output: Stream[TTSOutputChunk | TTSOutputEnd],
    ) -> None:
        # No TTS provided, exit early
        if not self._tts:
            return

        async for item in tts_input:
            with log_exceptions(logger, "Error while processing TTS input"):
                if self._tts.streaming:
                    if isinstance(item, TTSInputEnd):
                        # Flush any partial sentence still in the tokenizer.
                        remainder = self._tts_tokenizer.flush()
                        if remainder:
                            async for chunk in self._tts.send_iter(remainder):
                                await tts_output.send(chunk)
                    elif item.delta:
                        # Chunk: accumulate and emit on sentence boundaries.
                        text = self._tts_tokenizer.update(item.text)
                        if text:
                            async for chunk in self._tts.send_iter(text):
                                await tts_output.send(chunk)
                    # delta=False is redundant for streaming TTS (the deltas
                    # already carry the content); ignored.
                else:
                    # Non-streaming TTS only acts on complete utterances.
                    if isinstance(item, TTSInput) and not item.delta:
                        async for chunk in self._tts.send_iter(item.text):
                            await tts_output.send(chunk)

    async def write_audio_output(
        self,
        tts_output: Stream[TTSOutputChunk | TTSOutputEnd],
        audio_output: AudioOutputStream,
    ):
        speaking = False
        async for item in tts_output:
            with log_exceptions(logger, "Error while processing TTS output"):
                if isinstance(item, TTSOutputEnd):
                    if speaking:
                        self.events.send(
                            AgentTurnEndedEvent(interrupted=item.interrupted)
                        )
                        speaking = False
                    continue
                if not speaking:
                    self.events.send(AgentTurnStartedEvent())
                    speaking = True
                await audio_output.send(
                    AudioOutputChunk(data=item.data, final=item.final)
                )
                if item.final:
                    self.events.send(AgentTurnEndedEvent())
                    speaking = False
