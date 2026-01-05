import asyncio
import logging
from typing import TYPE_CHECKING, Awaitable, Callable

from vision_agents.core.utils.utils import await_or_run
from vision_agents.core.warmup import WarmupCache, Warmable

if TYPE_CHECKING:
    from .agents import Agent

logger = logging.getLogger(__name__)


class AgentLauncher:
    """
    Agent launcher that handles warmup and lifecycle management.

    The launcher ensures all components (LLM, TTS, STT, turn detection)
    are warmed up before the agent is launched.
    """

    def __init__(
        self,
        create_agent: Callable[..., "Agent" | Awaitable["Agent"]],
        join_call: Callable[..., None | Awaitable[None]] | None = None,
    ):
        """
        Initialize the agent launcher.

        Args:
            create_agent: A function that creates and returns an Agent instance
            join_call: Optional function that handles joining a call with the agent
        """
        self.create_agent = create_agent
        self.join_call = join_call
        self._warmup_lock = asyncio.Lock()
        self._warmup_cache = WarmupCache()

    async def warmup(self) -> None:
        """
        Warm up all agent components.

        This method creates the agent and calls warmup() on LLM, TTS, STT,
        and turn detection components if they exist.
        """
        async with self._warmup_lock:
            logger.info("Creating agent...")

            # Create a dry-run Agent instance and warmup its components for the first time.
            agent: "Agent" = await await_or_run(self.create_agent)
            logger.info("Warming up agent components...")
            await self._warmup_agent(agent)

            logger.info("Agent warmup completed")

    async def launch(self, **kwargs) -> "Agent":
        """
        Launch the agent.

        Args:
            **kwargs: Additional keyword arguments to pass to create_agent

        Returns:
            The Agent instance
        """
        agent: "Agent" = await await_or_run(self.create_agent, **kwargs)
        await self._warmup_agent(agent)
        return agent

    async def _warmup_agent(self, agent: "Agent") -> None:
        """
        Go over the Agent's dependencies and trigger `.warmup()` on them.

        It is safe to call `._warmup_agent()` multiple times.

        Args:
            agent: Agent to be warmed up

        Returns:

        """
        # Warmup tasks to run in parallel
        warmup_tasks = []

        # Warmup LLM (including Realtime)
        if agent.llm and isinstance(agent.llm, Warmable):
            warmup_tasks.append(agent.llm.warmup(self._warmup_cache))

        # Warmup TTS
        if agent.tts and isinstance(agent.tts, Warmable):
            warmup_tasks.append(agent.tts.warmup(self._warmup_cache))

        # Warmup STT
        if agent.stt and isinstance(agent.stt, Warmable):
            warmup_tasks.append(agent.stt.warmup(self._warmup_cache))

        # Warmup turn detection
        if agent.turn_detection and isinstance(agent.turn_detection, Warmable):
            warmup_tasks.append(agent.turn_detection.warmup(self._warmup_cache))

        # Warmup processors
        if agent.processors:
            for processor in agent.processors:
                if isinstance(processor, Warmable):
                    warmup_tasks.append(processor.warmup(self._warmup_cache))

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks)
