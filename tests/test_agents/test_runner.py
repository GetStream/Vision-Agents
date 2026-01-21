import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Header, HTTPException, Response
from httpx import ASGITransport, AsyncClient
from vision_agents.core import Agent, AgentLauncher, Runner, ServeOptions, User
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
async def agent_launcher():
    async def create_agent(**kwargs) -> Agent:
        stream_edge_mock = MagicMock()
        stream_edge_mock.events = EventManager()

        return Agent(
            llm=DummyLLM(),
            tts=DummyTTS(),
            edge=stream_edge_mock,
            agent_user=User(name="test"),
        )

    async def join_call(*args, **kwargs):
        await asyncio.sleep(10)

    launcher = AgentLauncher(create_agent=create_agent, join_call=join_call)
    return launcher


@pytest.fixture()
async def runner(agent_launcher) -> Runner:
    runner = Runner(launcher=agent_launcher)
    return runner


@pytest.fixture()
async def test_client_factory():
    @asynccontextmanager
    async def factory(runner: Runner):
        async with LifespanManager(runner.fast_api):
            async with AsyncClient(
                transport=ASGITransport(app=runner.fast_api),
                base_url="http://test",
            ) as client:
                yield client

    return factory


class TestRunnerServe:
    async def test_health(self, agent_launcher, test_client_factory) -> None:
        runner = Runner(launcher=agent_launcher)
        async with test_client_factory(runner) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200

    async def test_ready(self, agent_launcher, test_client_factory) -> None:
        runner = Runner(launcher=agent_launcher)
        async with test_client_factory(runner) as client:
            resp = await client.get("/ready")
            assert resp.status_code == 200

    async def test_start_session_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            assert resp_json["call_id"] == "test"
            session_id = resp_json["session_id"]
            assert session_id
            assert resp_json["session_started_at"]
            assert agent_launcher.get_session(session_id)

    async def test_start_session_current_user_stored(
        self, agent_launcher, test_client_factory
    ) -> None:
        def get_current_user(user_id=Header()) -> User:
            return User(id=user_id)

        opts = ServeOptions(
            get_current_user=get_current_user,
        )
        runner = Runner(launcher=agent_launcher, serve_options=opts)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions",
                json={"call_id": "test", "call_type": "default"},
                headers={"User-Id": "123"},
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            assert resp_json["call_id"] == "test"
            session_id = resp_json["session_id"]
            assert session_id
            assert resp_json["session_started_at"]
            session = agent_launcher.get_session(session_id)
            assert session.created_by == User(id="123")

    async def test_start_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_start():
            raise HTTPException(status_code=403)

        opts = ServeOptions(can_start_session=can_start)
        runner = Runner(launcher=agent_launcher, serve_options=opts)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions",
                json={"call_id": "test", "call_type": "default"},
            )
            assert resp.status_code == 403

    async def test_close_session_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.delete(f"/sessions/{session_id}")
            assert resp.status_code == 204
            assert agent_launcher.get_session(session_id) is None

    async def test_close_session_doesnt_exist_fails(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.delete("/sessions/some-id")
            assert resp.status_code == 404

    async def test_close_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_close():
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_close_session=can_close),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.delete(f"/sessions/{session_id}")
            assert resp.status_code == 403
            assert agent_launcher.get_session(session_id)

    async def test_close_session_beacon_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.post(f"/sessions/{session_id}/close")
            assert resp.status_code == 200
            assert agent_launcher.get_session(session_id) is None

    async def test_close_session_beacon_doesnt_exist_fails(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post("/sessions/some-id/close")
            assert resp.status_code == 404

    async def test_close_session_beacon_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_close():
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_close_session=can_close),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.post(f"/sessions/{session_id}/close")
            assert resp.status_code == 403
            assert agent_launcher.get_session(session_id)

    async def test_get_session_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.get(f"/sessions/{session_id}")
            assert resp.status_code == 200
            resp_json = resp.json()
            assert resp_json["session_id"] == session_id
            assert resp_json["call_id"] == "test"
            assert resp_json["session_started_at"]

    async def test_get_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_view():
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_session=can_view),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.get(f"/sessions/{session_id}")
            assert resp.status_code == 403

    async def test_get_session_doesnt_exist_404(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.get("/sessions/123123")
            assert resp.status_code == 404

    async def test_get_session_metrics_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            session = agent_launcher.get_session(session_id)
            assert session
            session.agent.metrics.llm_latency_ms__avg.update(250)
            session.agent.metrics.llm_time_to_first_token_ms__avg.update(250)
            session.agent.metrics.stt_latency_ms__avg.update(250)
            session.agent.metrics.tts_latency_ms__avg.update(250)
            session.agent.metrics.llm_input_tokens__total.inc(250)
            session.agent.metrics.llm_output_tokens__total.inc(250)

            resp = await client.get(f"/sessions/{session_id}/metrics")
            assert resp.status_code == 200
            resp_json = resp.json()
            assert resp_json["session_id"] == session_id
            assert resp_json["call_id"] == "test"
            assert resp_json["session_started_at"]
            assert resp_json["metrics_generated_at"]
            metrics = resp_json["metrics"]
            assert metrics["llm_latency_ms__avg"] == 250
            assert metrics["llm_time_to_first_token_ms__avg"] == 250
            assert metrics["stt_latency_ms__avg"] == 250
            assert metrics["tts_latency_ms__avg"] == 250
            assert metrics["llm_input_tokens__total"] == 250
            assert metrics["llm_output_tokens__total"] == 250

    async def test_get_session_metrics_doesnt_exist_404(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.get("/sessions/123123/metrics")
            assert resp.status_code == 404

    async def test_get_session_metrics_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_view_metrics():
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_metrics=can_view_metrics),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/sessions", json={"call_id": "test", "call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            session = agent_launcher.get_session(session_id)
            assert session
            session.agent.metrics.llm_latency_ms__avg.update(250)
            session.agent.metrics.llm_time_to_first_token_ms__avg.update(250)
            session.agent.metrics.stt_latency_ms__avg.update(250)
            session.agent.metrics.tts_latency_ms__avg.update(250)
            session.agent.metrics.llm_input_tokens__total.inc(250)
            session.agent.metrics.llm_output_tokens__total.inc(250)

            resp = await client.get(f"/sessions/{session_id}/metrics")
            assert resp.status_code == 403

    async def test_fastapi_bypass(self, agent_launcher, test_client_factory) -> None:
        custom_app = FastAPI()

        @custom_app.get("/hello-world")
        def hello_world():
            return Response(status_code=200, content="Hello world")

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(fast_api=custom_app),
        )

        async with test_client_factory(runner) as client:
            resp = await client.get("/hello-world")
            assert resp.status_code == 200
            assert resp.content.decode() == "Hello world"
