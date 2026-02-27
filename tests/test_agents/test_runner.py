import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import MagicMock

import pytest
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, HTTPException, Response
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

    def on_warmed_up(self, *_) -> None:
        self.warmed_up = True


@pytest.fixture()
def agent_launcher_factory():
    def factory(**launcher_kwargs) -> AgentLauncher:
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

        return AgentLauncher(
            create_agent=create_agent, join_call=join_call, **launcher_kwargs
        )

    return factory


@pytest.fixture()
def agent_launcher(agent_launcher_factory):
    return agent_launcher_factory()


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
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            assert resp_json["call_id"] == "test"
            session_id = resp_json["session_id"]
            assert session_id
            assert resp_json["session_started_at"]
            assert agent_launcher.get_session(session_id)

    async def test_start_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_start(call_id: str):
            raise HTTPException(status_code=403)

        opts = ServeOptions(can_start_session=can_start)
        runner = Runner(launcher=agent_launcher, serve_options=opts)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions",
                json={"call_type": "default"},
            )
            assert resp.status_code == 403

    async def test_start_session_permission_receives_call_id(
        self, agent_launcher, test_client_factory
    ) -> None:
        received_call_ids: list[str] = []

        def can_start(call_id: str):
            received_call_ids.append(call_id)

        opts = ServeOptions(can_start_session=can_start)
        runner = Runner(launcher=agent_launcher, serve_options=opts)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/my-call-123/sessions",
                json={"call_type": "default"},
            )
            assert resp.status_code == 201
            assert received_call_ids == ["my-call-123"]

    async def test_close_session_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            resp = await client.delete(f"/calls/test/sessions/{session_id}")
            assert resp.status_code == 202

    async def test_close_session_not_found(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.delete("/calls/test/sessions/some-id")
            assert resp.status_code == 404

    async def test_close_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_close(call_id: str):
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_close_session=can_close),
        )

        async with test_client_factory(runner) as client:
            resp = await client.delete("/calls/test/sessions/some-id")
            assert resp.status_code == 403

    async def test_close_session_beacon_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            resp = await client.post(f"/calls/test/sessions/{session_id}/close")
            assert resp.status_code == 202

    async def test_close_session_beacon_not_found(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post("/calls/test/sessions/some-id/close")
            assert resp.status_code == 404

    async def test_close_session_beacon_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_close(call_id: str):
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_close_session=can_close),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post("/calls/test/sessions/some-id/close")
            assert resp.status_code == 403

    async def test_get_session_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.get(f"/calls/test/sessions/{session_id}")
            assert resp.status_code == 200
            resp_json = resp.json()
            assert resp_json["session_id"] == session_id
            assert resp_json["call_id"] == "test"
            assert resp_json["session_started_at"]

    async def test_get_session_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def can_view(call_id: str):
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_session=can_view),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            resp_json = resp.json()
            session_id = resp_json["session_id"]

            assert agent_launcher.get_session(session_id)

            resp = await client.get(f"/calls/test/sessions/{session_id}")
            assert resp.status_code == 403

    async def test_get_session_doesnt_exist_404(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.get("/calls/test/sessions/123123")
            assert resp.status_code == 404

    async def test_get_session_metrics_success(
        self, agent_launcher, test_client_factory
    ) -> None:
        runner = Runner(launcher=agent_launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
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

            await agent_launcher.registry.update_metrics(
                "test", session_id, session.agent.metrics.to_dict()
            )

            resp = await client.get(f"/calls/test/sessions/{session_id}/metrics")
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
            resp = await client.get("/calls/test/sessions/123123/metrics")
            assert resp.status_code == 404

    async def test_get_session_metrics_no_permissions_fail(
        self, agent_launcher, test_client_factory
    ) -> None:
        def deny_view_metrics(call_id: str):
            raise HTTPException(status_code=403)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_metrics=deny_view_metrics),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
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

            resp = await client.get(f"/calls/test/sessions/{session_id}/metrics")
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

    async def test_close_session_permission_receives_call_id(
        self, agent_launcher, test_client_factory
    ) -> None:
        received_call_ids: list[str] = []

        def can_close(call_id: str):
            received_call_ids.append(call_id)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_close_session=can_close),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/my-call-456/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            await client.delete(f"/calls/my-call-456/sessions/{session_id}")
            assert received_call_ids == ["my-call-456"]

    async def test_view_session_permission_receives_call_id(
        self, agent_launcher, test_client_factory
    ) -> None:
        received_call_ids: list[str] = []

        def can_view(call_id: str):
            received_call_ids.append(call_id)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_session=can_view),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/my-call-789/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            await client.get(f"/calls/my-call-789/sessions/{session_id}")
            assert received_call_ids == ["my-call-789"]

    async def test_view_metrics_permission_receives_call_id(
        self, agent_launcher, test_client_factory
    ) -> None:
        received_call_ids: list[str] = []

        def can_view_m(call_id: str):
            received_call_ids.append(call_id)

        runner = Runner(
            launcher=agent_launcher,
            serve_options=ServeOptions(can_view_metrics=can_view_m),
        )

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/my-call-abc/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201
            session_id = resp.json()["session_id"]

            await client.get(f"/calls/my-call-abc/sessions/{session_id}/metrics")
            assert received_call_ids == ["my-call-abc"]

    async def test_start_session_max_concurrent_sessions_exceeded(
        self, agent_launcher_factory, test_client_factory
    ) -> None:
        launcher = agent_launcher_factory(max_concurrent_sessions=1)
        runner = Runner(launcher=launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test-1/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201

            resp = await client.post(
                "/calls/test-2/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 429
            assert (
                resp.json()["detail"] == "Reached maximum number of concurrent sessions"
            )

    async def test_start_session_max_sessions_per_call_exceeded(
        self, agent_launcher_factory, test_client_factory
    ) -> None:
        launcher = agent_launcher_factory(max_sessions_per_call=1)
        runner = Runner(launcher=launcher)

        async with test_client_factory(runner) as client:
            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 201

            resp = await client.post(
                "/calls/test/sessions", json={"call_type": "default"}
            )
            assert resp.status_code == 429
            assert (
                resp.json()["detail"]
                == "Reached maximum number of sessions for this call"
            )
