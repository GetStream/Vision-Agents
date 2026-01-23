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
    """
    Represents an active agent session within a call.

    An AgentSession wraps an Agent instance along with metadata about the session,
    including when it started, which call it belongs to, and the async task running
    the agent's call handler.
    """

    agent: "Agent"
    call_id: str
    started_at: datetime
    task: asyncio.Task
    created_by: Optional[Any] = None

    @property
    def finished(self) -> bool:
        """Return True if the session task has completed."""
        return self.task.done()

    @property
    def id(self) -> str:
        """Return the session ID (same as the agent ID)."""
        return self.agent.id

    async def wait(self):
        """
        Wait for the session task to finish running.
        """
        return await self.task

    def on_call_for(self) -> float:
        """
        Return the number of seconds for how long the agent has been on the call.
        Returns 0.0 if the agent has not joined a call yet.

        Returns:
            Duration in seconds since the agent joined the call, or 0.0 if not on a call.
        """
        return self.agent.on_call_for()

    def idle_for(self) -> float:
        """
        Return the idle time for this session if there are no other participants except the agent.

        Returns:
            Idle time in seconds, or 0.0 if the session is active.
        """
        return self.agent.idle_for()


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
        max_concurrent_sessions: Optional[int] = None,
        max_sessions_per_call: Optional[int] = None,
        max_session_duration_seconds: Optional[float] = None,
        cleanup_interval: float = 5.0,
    ):
        """
        Initialize the agent launcher.

        Args:
            create_agent: A function that creates and returns an Agent instance.
            join_call: A coroutine function that handles joining a call with the agent.
            agent_idle_timeout: Timeout in seconds for an agent to stay alone on a call
                before being automatically closed. Default is 60.0 seconds.
                Set to 0 to disable idle timeout (agents won't leave until the call ends).
            max_concurrent_sessions: Maximum number of concurrent sessions allowed across
                all calls. Default is None (unlimited).
            max_sessions_per_call: Maximum number of sessions allowed per call_id.
                Default is None (unlimited).
            max_session_duration_seconds: Maximum duration in seconds for a session
                before it is automatically closed. Default is None (unlimited).
            cleanup_interval: Interval in seconds between cleanup checks for idle
                or expired sessions. Default is 5.0 seconds.
        """
        self._create_agent = create_agent
        self._join_call = join_call
        self._warmup_lock = asyncio.Lock()
        self._warmup_cache = WarmupCache()
        self._start_lock = asyncio.Lock()

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

    async def start(self) -> None:
        """
        Start the agent launcher.

        This method warms up the agent components and starts the background
        cleanup task for managing idle and expired sessions.

        Raises:
            RuntimeError: If the launcher is already running.
        """
        if self._running:
            raise RuntimeError("AgentLauncher is already running")
        logger.debug("Starting AgentLauncher")
        self._running = True
        await self.warmup()
        self._cleanup_task = asyncio.create_task(self._cleanup_idle_sessions())
        logger.debug("AgentLauncher started")

    async def stop(self) -> None:
        """
        Stop the agent launcher and close all active sessions.

        This method cancels the cleanup task, then cancels and waits for
        all active session tasks to complete.
        """
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
            try:
                logger.info("Warming up agent components...")
                await self._warmup_agent(agent)
                self._warmed_up = True
                logger.info("Agent warmup completed")
            finally:
                await agent.close()

    @property
    def warmed_up(self) -> bool:
        """Return True if the agent components have been warmed up."""
        return self._warmed_up

    @property
    def running(self) -> bool:
        """Return True if the launcher is currently running."""
        return self._running

    @property
    def ready(self) -> bool:
        """Return True if the launcher is warmed up and running."""
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
        """
        Start a new agent session for a call.

        Creates a new agent, joins the specified call, and returns an AgentSession
        object to track the session.

        Args:
            call_id: Unique identifier for the call to join.
            call_type: Type of call. Default is "default".
            created_by: Optional metadata about who/what created this session.
            video_track_override_path: Optional path to a video file to use
                instead of a live video track.

        Returns:
            An AgentSession object representing the new session.

        Raises:
            MaxConcurrentSessionsExceeded: If the maximum number of concurrent
                sessions has been reached.
            MaxSessionsPerCallExceeded: If the maximum number of sessions for
                this call_id has been reached.
        """
        async with self._start_lock:
            if (
                self._max_concurrent_sessions
                and len(self._sessions) >= self._max_concurrent_sessions
            ):
                raise MaxConcurrentSessionsExceeded(
                    f"Reached maximum concurrent sessions of {self._max_concurrent_sessions}"
                )

            call_sessions_total = len(self._calls.get(call_id, set()))
            if (
                self._max_sessions_per_call
                and call_sessions_total >= self._max_sessions_per_call
            ):
                raise MaxSessionsPerCallExceeded(
                    f"Reached maximum sessions per call of {self._max_sessions_per_call}"
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
            logger.info(f"Started agent session with id {session.id}")
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
        """
        Get a session by its ID.

        Args:
            session_id: The session ID to look up.

        Returns:
            The AgentSession if found, None otherwise.
        """
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

    async def _cleanup_idle_sessions(self) -> None:
        if not self._agent_idle_timeout and not self._max_session_duration_seconds:
            return
        max_session_duration_seconds = self._max_session_duration_seconds or float(
            "inf"
        )

        while self._running:
            # Collect idle agents first to close them all at once
            to_close = []
            for session in self._sessions.values():
                agent = session.agent
                on_call_for = agent.on_call_for()
                idle_for = agent.idle_for()
                if 0 < self._agent_idle_timeout <= idle_for:
                    logger.info(
                        f'Closing session "{session.id}" with '
                        f'user_id "{agent.agent_user.id}" after being '
                        f"idle for {round(idle_for, 2)}s "
                        f"(idle timeout is {self._agent_idle_timeout}s)"
                    )
                    to_close.append(agent)
                elif on_call_for >= max_session_duration_seconds:
                    logger.info(
                        f'Closing session "{session.id}" with user_id "{agent.agent_user.id}" '
                        f"after reaching the maximum session "
                        f"duration of {max_session_duration_seconds}s"
                    )
                    to_close.append(agent)

            if to_close:
                coros = [
                    asyncio.shield(self.close_session(s.id, wait=False))
                    for s in to_close
                ]
                result = await asyncio.shield(
                    asyncio.gather(*coros, return_exceptions=True)
                )
                for agent, r in zip(to_close, result):
                    if isinstance(r, Exception):
                        logger.error(
                            f"Failed to close agent with user_id {agent.agent_user.id}",
                            exc_info=r,
                        )

            await asyncio.sleep(self._cleanup_interval)

    async def __aenter__(self) -> "AgentLauncher":
        """Enter the async context manager, starting the launcher."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit the async context manager, stopping the launcher."""
        await self.stop()
