import asyncio
import logging
from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import Tracer
from vision_agents.core.edge.types import Participant
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM
from vision_agents.core.llm.events import LLMResponseFinalEvent
from vision_agents.core.llm.llm import LLMResponseDelta, LLMResponseFinal
from vision_agents.core.utils.exceptions import log_task_exception
from vision_agents.core.utils.stream import Stream
from vision_agents.core.utils.utils import cancel_and_wait

logger = logging.getLogger(__name__)
tracer: Tracer = trace.get_tracer("agents")


__all__ = ["LLMTurn"]


@dataclass
class LLMTurn:
    transcript: str
    participant: Participant
    events: EventManager
    stream: Stream[LLMResponseDelta | LLMResponseFinal] = field(
        init=False, default_factory=Stream
    )
    _llm_response_task: asyncio.Task | None = field(default=None, init=False)
    _finalize_task: asyncio.Task | None = field(default=None, init=False)
    _cancelled: bool = field(default=False, init=False)
    _confirmed: bool = field(default=False, init=False)

    @property
    def finalized(self) -> bool:
        """True once the LLM response task and the finalize task have both reached a terminal state.

        "Terminal" covers normal completion, an exception, or cancellation —
        anything that causes ``asyncio.Task.done()`` to return True. A turn
        whose ``finalize()`` was never called is not completed.
        """
        response = self._llm_response_task
        finalize = self._finalize_task
        if response is None or finalize is None:
            return False
        return response.done() and finalize.done()

    @property
    def confirmed(self) -> bool:
        return self._confirmed

    @property
    def cancelled(self) -> bool:
        """True if ``cancel()`` has been invoked on this turn."""
        return self._cancelled

    @property
    def started(self) -> bool:
        return bool(self._llm_response_task)

    def start(self, llm: LLM) -> None:
        if self._llm_response_task is not None:
            raise RuntimeError("LLM response task is already running")
        task = asyncio.create_task(self._do_llm_response(llm))
        task.add_done_callback(
            log_task_exception("LLMTurn: failed to get a response from LLM")
        )
        self._llm_response_task = task

    async def cancel(self):
        if self.finalized or self.cancelled or not self.started:
            # The turn is done, nothing to cancel
            return

        logger.info('🤖 Cancelling LLM turn for transcript "%s"', self.transcript)

        self._cancelled = True
        if self._llm_response_task is not None:
            await cancel_and_wait(self._llm_response_task)
            self._llm_response_task = None
        if self._finalize_task is not None:
            await cancel_and_wait(self._finalize_task)
            self._finalize_task = None

    def finalize(self, output: Stream[LLMResponseDelta | LLMResponseFinal]) -> None:
        if self._finalize_task is not None:
            # Exit early if the turn has already been finalized or in progress
            return
        if not self.started:
            raise RuntimeError("LLM turn must be started first")
        if not self.confirmed:
            raise RuntimeError("LLM turn must be confirmed first")

        logger.info('🤖 Finalizing LLM turn for transcript "%s"', self.transcript)
        task = asyncio.create_task(self._do_finalize(output=output))
        task.add_done_callback(
            log_task_exception("LLMTurn: failed to finalize the turn")
        )
        self._finalize_task = task

    def confirm(self):
        """
        Mark the LLMTurn as confirmed
        """
        if self._confirmed:
            raise RuntimeError("LLM turn is already confirmed")

        self._confirmed = True

    async def _do_llm_response(self, llm: LLM):
        with tracer.start_as_current_span("simple_response"):
            try:
                async for item in llm.simple_response(
                    self.transcript, self.participant
                ):
                    await self.stream.send(item)
                    if isinstance(item, LLMResponseFinal):
                        self.events.send(
                            LLMResponseFinalEvent(
                                plugin_name=llm.provider_name,
                                text=item.text,
                                model=item.model or llm.model,
                            )
                        )
                        llm.metrics.on_llm_response(
                            provider=llm.provider_name,
                            model=item.model,
                            latency_ms=item.latency_ms,
                            time_to_first_token_ms=item.time_to_first_token_ms,
                            input_tokens=item.input_tokens,
                            output_tokens=item.output_tokens,
                        )

            except asyncio.CancelledError:
                # Interrupt llm when the turn is cancelled to give it a chance
                # to stop some heavy async work (e.g. inference in a different thread)
                await llm.interrupt()
                raise
            finally:
                # Close stream no matter what to free consumer iterators
                self.stream.close()

    async def _do_finalize(self, output: Stream[LLMResponseDelta | LLMResponseFinal]):
        """
        The LLM turn is confirmed and final, we can now read the LLM response chunks
          and send them to the output.

        Args:
            output: output stream for LLM response chunks

        """
        async for item in self.stream:
            await output.send(item)
