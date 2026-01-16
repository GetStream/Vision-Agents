import asyncio
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from vision_agents.core import Agent, AgentLauncher, User
from vision_agents.core.events import EventManager
from vision_agents.core.llm import LLM
from vision_agents.core.llm.llm import LLMResponseEvent
from vision_agents.core.tts import TTS
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

    async def on_warmed_up(self, *_) -> None:
        self.warmed_up = True


@pytest.fixture()
async def stream_edge_mock() -> MagicMock:
    mock = MagicMock()
    mock.events = EventManager()
    return mock


async def join_call_noop(
    agent: Agent, call_type: str, call_id: str, **kwargs
) -> None: ...


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

    async def test_idle_agents_stopped(self, stream_edge_mock):
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
            agent_idle_cleanup_interval=0.5,
        )
        with patch.object(Agent, "idle_for", return_value=10):
            # Start the launcher internals
            async with launcher:
                # Launch a couple of idle agents
                agent1 = await launcher.launch()
                agent2 = await launcher.launch()
                # Sleep 2s to let the launcher clean up the agents
                await asyncio.sleep(2)

        # The agents must be closed
        assert agent1.closed
        assert agent2.closed

    async def test_idle_agents_alive_with_idle_timeout_zero(self, stream_edge_mock):
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
            agent_idle_cleanup_interval=0.5,
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
