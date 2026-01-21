import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Coroutine,
    Optional,
)

from vision_agents.core.utils.utils import await_or_run, cancel_and_wait
from vision_agents.core.warmup import Warmable, WarmupCache

from .exceptions import MaxConcurrentSessionsExceeded, MaxSessionsPerCallExceeded

if TYPE_CHECKING:
    from .agents import Agent

logger = logging.getLogger(__name__)


@dataclass
class AgentSession:
    agent: "Agent"
    call_id: str
    started_at: datetime
    task: asyncio.Task
    created_by: Optional[Any] = None

    @property
    def finished(self) -> bool:
        return self.task.done()

    @property
    def id(self) -> str:
        return self.agent.id

    async def wait(self):
        """
        Wait for the session task to finish running.
        """
        return await self.task


# TODO: Rename to `AgentManager`.
class AgentLauncher:
    """
    Agent launcher that handles warmup and lifecycle management.

    The launcher ensures all components (LLM, TTS, STT, turn detection)
    are warmed up before the agent is launched.
    """

    def __init__(
        self,
        create_agent: Callable[..., "Agent" | Coroutine[Any, Any, "Agent"]],
        join_call: Callable[["Agent", str, str], Coroutine],
        agent_idle_timeout: float = 60.0,
        max_concurrent_sessions: Optional[int] = 50,
        max_sessions_per_call: Optional[int] = None,
        max_session_duration_seconds: Optional[float] = None,
        cleanup_interval: float = 5.0,
    ):
        """
        Initialize the agent launcher.

        Args:
            create_agent: A function that creates and returns an Agent instance
            join_call: Optional function that handles joining a call with the agent
            agent_idle_timeout: Optional timeout in seconds for agent to stay alone on the call. Default - `60.0`.
                `0` means idle agents won't leave the call until it's ended.

        """
        self._create_agent = create_agent
        self._join_call = join_call
        self._warmup_lock = asyncio.Lock()
        self._warmup_cache = WarmupCache()

        if max_concurrent_sessions is not None and max_concurrent_sessions <= 0:
            raise ValueError("max_concurrent_sessions must be > 0 or None")
        self._max_concurrent_sessions = max_concurrent_sessions
        if max_sessions_per_call is not None and max_sessions_per_call <= 0:
            raise ValueError("max_sessions_per_call must be > 0 or None")
        self._max_sessions_per_call = max_sessions_per_call
        if (
            max_session_duration_seconds is not None
            and max_session_duration_seconds <= 0
        ):
            raise ValueError("max_session_duration_seconds must be > 0 or None")
        self._max_session_duration_seconds = max_session_duration_seconds

        if agent_idle_timeout < 0:
            raise ValueError("agent_idle_timeout must be >= 0")
        self._agent_idle_timeout = agent_idle_timeout

        if cleanup_interval <= 0:
            raise ValueError("cleanup_interval must be > 0")
        self._cleanup_interval: float = cleanup_interval

        self._running = False
        self._cleanup_task: Optional[asyncio.Task] = None
        self._warmed_up: bool = False
        self._sessions: dict[str, AgentSession] = {}
        self._calls: dict[str, set[str]] = {}

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

        coros = [cancel_and_wait(s.task) for s in self._sessions.values()]
        for result in asyncio.as_completed(coros):
            try:
                await result
            except Exception as exc:
                logger.error(f"Failed to cancel the agent task: {exc}")

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
            agent: "Agent" = await await_or_run(self._create_agent)
            logger.info("Warming up agent components...")
            await self._warmup_agent(agent)
            self._warmed_up = True

            logger.info("Agent warmup completed")

    @property
    def warmed_up(self) -> bool:
        return self._warmed_up

    @property
    def running(self) -> bool:
        return self._running

    @property
    def ready(self) -> bool:
        return self.warmed_up and self.running

    async def launch(self, **kwargs) -> "Agent":
        """
        Launch the agent.

        Args:
            **kwargs: Additional keyword arguments to pass to create_agent

        Returns:
            The Agent instance
        """
        agent: "Agent" = await await_or_run(self._create_agent, **kwargs)
        await self._warmup_agent(agent)
        return agent

    async def start_session(
        self,
        call_id: str,
        call_type: str = "default",
        created_by: Optional[Any] = None,
        video_track_override_path: Optional[str] = None,
    ) -> AgentSession:
        sessions_total = len(self._sessions)
        if len(self._sessions) == self._max_concurrent_sessions:
            raise MaxConcurrentSessionsExceeded(
                f"Maximum concurrent sessions exceeded:"
                f" {sessions_total}/{self._max_concurrent_sessions} sessions active"
            )

        call_sessions_total = len(self._calls.get(call_id, set()))
        if call_sessions_total == self._max_sessions_per_call:
            raise MaxSessionsPerCallExceeded(
                f"Maximum sessions exceeded for call "
                f"'{call_id}': {call_sessions_total}/{self._max_sessions_per_call}"
            )

        agent: "Agent" = await self.launch()
        if video_track_override_path:
            agent.set_video_track_override_path(video_track_override_path)

        task = asyncio.create_task(
            self._join_call(agent, call_type, call_id), name=f"agent-{agent.id}"
        )

        # Remove the session when the task is done
        # or when the AgentSession is garbage-collected
        # in case the done callback wasn't fired
        def _finalizer(session_id_: str, call_id_: str, *_):
            session_ = self._sessions.pop(session_id_, None)
            if session_ is not None:
                call_sessions = self._calls.get(call_id_, set())
                if call_sessions:
                    call_sessions.discard(session_id_)

        task.add_done_callback(partial(_finalizer, agent.id, call_id))
        session = AgentSession(
            agent=agent,
            task=task,
            started_at=datetime.now(timezone.utc),
            call_id=call_id,
            created_by=created_by,
        )
        self._sessions[agent.id] = session
        self._calls.setdefault(call_id, set()).add(agent.id)
        logger.info(f"Start agent session with id {session.id}")
        return session

    async def close_session(self, session_id: str, wait: bool = False) -> bool:
        """
        Close session with id `session_id`.
        Returns `True` if session was found and closed, `False` otherwise.

        Args:
            session_id: session id
            wait: when True, wait for the underlying agent to finish.
                Otherwise, just cancel the task and return.

        Returns:
            `True` if session was found and closed, `False` otherwise.
        """
        session = self._sessions.pop(session_id, None)
        if session is None:
            # The session is either closed or doesn't exist, exit early
            return False
        call_sessions = self._calls.get(session.call_id)
        if call_sessions:
            call_sessions.discard(session.id)

        logger.info(f"Closing agent session with id {session.id}")
        if wait:
            await cancel_and_wait(session.task)
        else:
            session.task.cancel()
        return True

    def get_session(self, session_id: str) -> Optional[AgentSession]:
        return self._sessions.get(session_id)

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
            for session in self._sessions.values():
                agent = session.agent
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

            await asyncio.sleep(self._cleanup_interval)

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.stop()
