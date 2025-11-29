"""Agent launcher with warmup support."""

import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable, Optional, Union, cast

if TYPE_CHECKING:
    from .agents import Agent

logger = logging.getLogger(__name__)


class AgentProcess:
    """
    Add info here about the thread/process. Enabling warm up to work well in a multiprocess env
    """

    pass


class AgentLauncher:
    """
    Agent launcher that handles warmup and lifecycle management.

    The launcher ensures all components (LLM, TTS, STT, turn detection)
    are warmed up before the agent is launched.
    """

    def __init__(
        self,
        create_agent: Callable[..., Union["Agent", Awaitable["Agent"]]],
        join_call: Optional[Callable[..., Union[None, Awaitable[None]]]] = None,
    ):
        """
        Initialize the agent launcher.

        Args:
            create_agent: A function that creates and returns an Agent instance
            join_call: Optional function that handles joining a call with the agent
        """
        self.create_agent = create_agent
        self.join_call = join_call
        self._agent: Optional["Agent"] = None
        self._warmed_up = False
        self._warmup_lock = asyncio.Lock()

    async def warmup(self, **kwargs) -> None:
        """
        Warm up all agent components.

        This method creates the agent and calls warmup on LLM, TTS, STT,
        and turn detection components if they exist. It ensures warmup is
        only called once.

        Args:
            **kwargs: Additional keyword arguments to pass to create_agent
        """
        async with self._warmup_lock:
            if self._warmed_up:
                logger.debug("Agent already warmed up, skipping")
                return

            logger.info("Creating agent...")

            # Create the agent
            result = self.create_agent(**kwargs)
            if asyncio.iscoroutine(result):
                agent: "Agent" = await result
            else:
                agent = cast("Agent", result)

            self._agent = agent

            logger.info("Warming up agent components...")

            # Warmup tasks to run in parallel
            warmup_tasks = []

            # Warmup LLM (including Realtime)
            if agent.llm and hasattr(agent.llm, "warmup"):
                logger.debug("Warming up LLM: %s", agent.llm.__class__.__name__)
                warmup_tasks.append(agent.llm.warmup())

            # Warmup TTS
            if agent.tts and hasattr(agent.tts, "warmup"):
                logger.debug("Warming up TTS: %s", agent.tts.__class__.__name__)
                warmup_tasks.append(agent.tts.warmup())

            # Warmup STT
            if agent.stt and hasattr(agent.stt, "warmup"):
                logger.debug("Warming up STT: %s", agent.stt.__class__.__name__)
                warmup_tasks.append(agent.stt.warmup())

            # Warmup turn detection
            if agent.turn_detection and hasattr(agent.turn_detection, "warmup"):
                logger.debug(
                    "Warming up turn detection: %s",
                    agent.turn_detection.__class__.__name__,
                )
                warmup_tasks.append(agent.turn_detection.warmup())

            # Warmup processors
            if agent.processors:
                logger.debug("Warming up processors")
                for processor in agent.processors:
                    if hasattr(processor, "warmup"):
                        logger.debug("Warming up processor: %s", processor.name)
                        warmup_tasks.append(processor.warmup())

            # Run all warmups in parallel
            if warmup_tasks:
                await asyncio.gather(*warmup_tasks)

            self._warmed_up = True
            logger.info("Agent warmup completed")

    async def launch(self, **kwargs) -> "Agent":
        """
        Launch the agent with warmup.

        This ensures warmup is called before returning the agent.

        Args:
            **kwargs: Additional keyword arguments to pass to create_agent

        Returns:
            The warmed-up agent instance
        """
        await self.warmup(**kwargs)
        assert self._agent is not None, "Agent should be created during warmup"
        return self._agent
