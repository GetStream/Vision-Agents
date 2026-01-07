import asyncio
import logging
import weakref
from typing import TYPE_CHECKING, Awaitable, Callable, Optional

from vision_agents.core.utils.utils import await_or_run, cancel_and_wait
from vision_agents.core.warmup import Warmable, WarmupCache

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
        agent_idle_timeout: float = 60.0,
        agent_idle_cleanup_interval: float = 5.0,
    ):
        """
        Initialize the agent launcher.

        Args:
            create_agent: A function that creates and returns an Agent instance
            join_call: Optional function that handles joining a call with the agent
            agent_idle_timeout: Optional timeout in seconds for agent to stay alone on the call. Default - `60.0`.
                `0` means idle agents won't leave the call until it's ended.

        """
        self.create_agent = create_agent
        self.join_call = join_call
        self._warmup_lock = asyncio.Lock()
        self._warmup_cache = WarmupCache()

        if agent_idle_timeout < 0:
            raise ValueError("agent_idle_timeout must be >= 0")
        self._agent_idle_timeout = agent_idle_timeout

        if agent_idle_cleanup_interval <= 0:
            raise ValueError("agent_idle_cleanup_interval must be > 0")
        self._agent_idle_cleanup_interval = agent_idle_cleanup_interval

        self._active_agents: weakref.WeakSet[Agent] = weakref.WeakSet()

        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._warmed_up: bool = False

    async def start(self):
        if self._running:
            raise RuntimeError("AgentLauncher is already running")
        logger.debug("Starting AgentLauncher")
        self._running = True
        await self.warmup()
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_agents())
        logger.debug("AgentLauncher started")

    async def stop(self):
        logger.debug("Stopping AgentLauncher")
        self._running = False
        if self._cleanup_task:
            await cancel_and_wait(self._cleanup_task)
        logger.debug("AgentLauncher stopped")

    async def warmup(self) -> None:
        """
        Warm up all agent components.

        This method creates the agent and calls warmup() on LLM, TTS, STT,
        and turn detection components if they exist.
        """
        if self._warmed_up or self._warmup_lock.locked():
            return

        async with self._warmup_lock:
            logger.info("Creating agent...")

            # Create a dry-run Agent instance and warmup its components for the first time.
            agent: "Agent" = await await_or_run(self.create_agent)
            logger.info("Warming up agent components...")
            await self._warmup_agent(agent)
            self._warmed_up = True

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
        self._active_agents.add(agent)
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
        for processor in agent.processors:
            if isinstance(processor, Warmable):
                warmup_tasks.append(processor.warmup(self._warmup_cache))

        if warmup_tasks:
            await asyncio.gather(*warmup_tasks)

    async def _cleanup_idle_agents(self) -> None:
        if not self._agent_idle_timeout:
            return

        while self._running:
            # Collect idle agents first to close them all at once
            idle_agents = []
            for agent in self._active_agents:
                agent_idle_for = agent.idle_for()
                if agent_idle_for >= self._agent_idle_timeout:
                    logger.info(
                        f'Agent with user_id "{agent.agent_user.id}" is idle for {round(agent_idle_for, 2)}s, '
                        f"closing it after {self._agent_idle_timeout}s timeout"
                    )
                    idle_agents.append(agent)

            if idle_agents:
                coros = [asyncio.shield(a.close()) for a in idle_agents]
                result = await asyncio.shield(
                    asyncio.gather(*coros, return_exceptions=True)
                )
                for agent, r in zip(idle_agents, result):
                    if isinstance(r, Exception):
                        logger.error(
                            f"Failed to close idle agent with user_id {agent.agent_user.id}",
                            exc_info=r,
                        )

            await asyncio.sleep(self._agent_idle_cleanup_interval)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
