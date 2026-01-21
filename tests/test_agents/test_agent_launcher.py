import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vision_agents.core import Agent, AgentLauncher, User
from vision_agents.core.agents.exceptions import (
    MaxConcurrentSessionsExceeded,
    MaxSessionsPerCallExceeded,
)
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.tts import TTS
from vision_agents.core.utils.utils import cancel_and_wait
from vision_agents.core.warmup import Warmable


class DummyTTS(TTS):
    async def stream_audio(self, *_, **__):
        return b""

    async def stop_audio(self) -> None: ...


class DummyLLM(LLM, Warmable[bool]):
    def __init__(self):
        super(DummyLLM, self).__init__()
        self.warmed_up = False

    async def simple_response(self, *_, **__) -> LLMResponseEvent[Any]:
        return LLMResponseEvent(text="Simple response", original=None)

    async def on_warmup(self) -> bool:
        return True

    def on_warmed_up(self, *_) -> None:
        self.warmed_up = True


@pytest.fixture()
async def stream_edge_mock() -> MagicMock:
    mock = MagicMock()
    mock.events = EventManager()
    return mock


async def join_call_noop(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    await asyncio.sleep(10)


class TestAgentLauncher:
    async def test_warmup(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call_noop)
        await launcher.warmup()
        assert llm.warmed_up

    async def test_launch(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call_noop)
        agent = await launcher.launch()
        assert agent

    async def test_idle_sessions_stopped(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            agent_idle_timeout=1.0,
            cleanup_interval=0.5,
        )
        with patch.object(Agent, "idle_for", return_value=10):
            # Start the launcher internals
            async with launcher:
                # Launch a couple of idle agents
                session1 = await launcher.start_session(call_id="1")
                session2 = await launcher.start_session(call_id="2")
                # Sleep 2s to let the launcher clean up the agents
                await asyncio.sleep(2)

        # The agents must be closed
        assert session1.agent.closed
        assert session2.agent.closed

    async def test_idle_sessions_alive_with_idle_timeout_zero(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            agent_idle_timeout=0,
        )
        with patch.object(Agent, "idle_for", return_value=10):
            # Start the launcher internals
            async with launcher:
                # Launch a couple of idle agents
                agent1 = await launcher.launch()
                agent2 = await launcher.launch()
                # Sleep 2s to let the launcher clean up the agents
                await asyncio.sleep(2)

        # The agents must not be closed because agent_idle_timeout=0
        assert not agent1.closed
        assert not agent2.closed

    async def test_active_agents_alive(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            agent_idle_timeout=1.0,
            cleanup_interval=0.5,
        )
        with patch.object(Agent, "idle_for", return_value=0):
            # Start the launcher internals
            async with launcher:
                # Launch a couple of active agents (idle_for=0)
                agent1 = await launcher.launch()
                agent2 = await launcher.launch()
                # Sleep 2s to let the launcher clean up the agents
                await asyncio.sleep(2)

        # The agents must not be closed
        assert not agent1.closed
        assert not agent2.closed

    async def test_start_session(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(2)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        session = await launcher.start_session(call_id="test", call_type="default")
        assert session
        assert session.id
        assert session.call_id
        assert session.agent
        assert session.started_at
        assert session.created_by is None
        assert not session.finished

        assert launcher.get_session(session_id=session.id)

        # Wait for session to stop (it just sleeps)
        await session.wait()
        assert session.finished
        assert not launcher.get_session(session_id=session.id)

    async def test_close_session_exists(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(10)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        session = await launcher.start_session(call_id="test", call_type="default")
        assert session

        await launcher.close_session(session_id=session.id, wait=True)
        assert session.finished
        assert session.task.done()
        assert not launcher.get_session(session_id=session.id)

    async def test_close_session_doesnt_exist(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(10)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        # Closing a non-existing session doesn't fail
        await launcher.close_session(session_id="session-id", wait=True)

    async def test_get_session_doesnt_exist(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(10)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        session = launcher.get_session(session_id="session-id")
        assert session is None

    async def test_stop_multiple_sessions(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(10)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        async with launcher:
            session1 = await launcher.start_session(call_id="test", call_type="default")
            session2 = await launcher.start_session(call_id="test", call_type="default")
            session3 = await launcher.start_session(call_id="test", call_type="default")

        assert session1.finished
        assert session2.finished
        assert session3.finished

    async def test_session_cleaned_up_after_finish(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(1)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        session = await launcher.start_session(call_id="test", call_type="default")
        assert session

        await session.wait()
        assert session.finished
        # The session becomes unavailable after it's done
        assert launcher.get_session(session_id=session.id) is None

    async def test_session_cleaned_up_after_cancel(self, stream_edge_mock):
        llm = DummyLLM()
        tts = DummyTTS()

        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=llm,
                tts=tts,
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(
            agent: Agent, call_type: str, call_id: str, **kwargs
        ) -> None:
            await asyncio.sleep(1)

        launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
        async with launcher:
            session = await launcher.start_session(call_id="test", call_type="default")
            assert session

            await cancel_and_wait(session.task)
            assert session.finished
            # The session becomes unavailable if it was cancelled
            assert launcher.get_session(session_id=session.id) is None

    async def test_max_concurrent_agents_invalid(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        with pytest.raises(ValueError, match="max_concurrent_sessions must be > 0"):
            AgentLauncher(
                create_agent=create_agent,
                join_call=join_call_noop,
                max_concurrent_sessions=0,
            )

        with pytest.raises(ValueError, match="max_concurrent_sessions must be > 0"):
            AgentLauncher(
                create_agent=create_agent,
                join_call=join_call_noop,
                max_concurrent_sessions=-1,
            )

    async def test_max_sessions_per_call_invalid(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        with pytest.raises(ValueError, match="max_sessions_per_call must be > 0"):
            AgentLauncher(
                create_agent=create_agent,
                join_call=join_call_noop,
                max_sessions_per_call=0,
            )
        with pytest.raises(ValueError, match="max_sessions_per_call must be > 0"):
            AgentLauncher(
                create_agent=create_agent,
                join_call=join_call_noop,
                max_sessions_per_call=-1,
            )

    async def test_max_concurrent_agents_exceeded(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            max_concurrent_sessions=2,
        )
        async with launcher:
            session1 = await launcher.start_session(call_id="call1")
            session2 = await launcher.start_session(call_id="call2")

            with pytest.raises(MaxConcurrentSessionsExceeded):
                await launcher.start_session(call_id="call3")

            # Close one session and try to create a new one again
            await launcher.close_session(session_id=session1.id)
            session3 = await launcher.start_session(call_id="call3")
            assert session3 is not None

    async def test_max_concurrent_agents_can_create_after_session_ends(
        self, stream_edge_mock
    ):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(*args, **kwargs):
            await asyncio.sleep(1)

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
            max_concurrent_sessions=2,
        )
        async with launcher:
            session1 = await launcher.start_session(call_id="call1")
            session2 = await launcher.start_session(call_id="call2")
            with pytest.raises(MaxConcurrentSessionsExceeded):
                await launcher.start_session(call_id="call3")

            await session1.wait()

            # Can create a new session when the previous one ends
            session3 = await launcher.start_session(call_id="call3")
            assert session3 is not None

    async def test_max_sessions_per_call_exceeded(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            max_sessions_per_call=2,
        )
        async with launcher:
            session1 = await launcher.start_session(call_id="same_call")
            session2 = await launcher.start_session(call_id="same_call")

            with pytest.raises(MaxSessionsPerCallExceeded):
                await launcher.start_session(call_id="same_call")

            # Different call should still work
            session3 = await launcher.start_session(call_id="call2")
            assert session3 is not None

            # Close one session
            await launcher.close_session(session_id=session1.id, wait=True)

            # Now we should be able to start a new session for the same call
            session4 = await launcher.start_session(call_id="same_call")
            assert session4 is not None

    async def test_max_sessions_per_call_can_create_after_session_ends(
        self, stream_edge_mock
    ):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        async def join_call(*args, **kwargs):
            await asyncio.sleep(1)

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call,
            max_sessions_per_call=2,
        )
        async with launcher:
            session1 = await launcher.start_session(call_id="same_call")
            session2 = await launcher.start_session(call_id="same_call")

            with pytest.raises(MaxSessionsPerCallExceeded):
                await launcher.start_session(call_id="same_call")

            await session1.wait()
            # Can create a new session when the previous one ends
            session3 = await launcher.start_session(call_id="same_call")
            assert session3 is not None

    async def test_max_concurrent_agents_none_allows_unlimited(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            max_concurrent_sessions=None,
        )
        async with launcher:
            # Start many sessions - should not raise
            sessions = []
            for i in range(10):
                session = await launcher.start_session(call_id=f"call{i}")
                sessions.append(session)

            assert len(sessions) == 10

    async def test_max_sessions_per_call_none_allows_unlimited(self, stream_edge_mock):
        async def create_agent(**kwargs) -> Agent:
            return Agent(
                llm=DummyLLM(),
                tts=DummyTTS(),
                edge=stream_edge_mock,
                agent_user=User(name="test"),
            )

        launcher = AgentLauncher(
            create_agent=create_agent,
            join_call=join_call_noop,
            max_concurrent_sessions=None,
            max_sessions_per_call=None,
        )
        async with launcher:
            # Start many sessions for the same call - should not raise
            sessions = []
            for i in range(10):
                session = await launcher.start_session(call_id="same_call")
                sessions.append(session)

            assert len(sessions) == 10
