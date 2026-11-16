from typing import Any

import httpx
import pytest
from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import (
    API_KEY_ENV,
    API_SECRET_ENV,
    AUTHENTICATE_ENV,
    CUSTOMER_ENV,
    DEFAULT_URL,
    REGION_ENV,
    URL_ENV,
    Backend,
)

REGIONAL_HOST = "accelerate-us-east4.gcp.stream-io-api.com"
WHEN = "2026-01-01T00:00:00Z"


class RegionalRouter:
    """The hosted router in us-east4, answering one authenticated read.

    Requests keep the regional URL the client built and only their transport is swapped, so
    what is under test is where the client sends them and what it sends.
    """

    def __init__(self):
        self.asked: list[httpx.Request] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.asked.append(request)
        if request.method != "GET" or request.url.path != "/v1/agents/configs":
            return httpx.Response(404, json={"message": "not found"})
        if (
            request.url.host != REGIONAL_HOST
            or request.headers.get("api_key") != "test-key"
            or request.headers.get("stream-auth-type") != "jwt"
            or not request.headers.get("Authorization")
        ):
            return httpx.Response(
                401, json={"message": "expected an authenticated regional request"}
            )
        return httpx.Response(
            200,
            json=[
                {
                    "id": "config-1",
                    "name": "docs",
                    "mode": "text",
                    "region": "us",
                    "created_at": WHEN,
                    "updated_at": WHEN,
                }
            ],
        )


@pytest.fixture
def router(monkeypatch: pytest.MonkeyPatch) -> RegionalRouter:
    for name in (URL_ENV, REGION_ENV, CUSTOMER_ENV, AUTHENTICATE_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(API_KEY_ENV, "test-key")
    monkeypatch.setenv(API_SECRET_ENV, "test-secret")

    fake = RegionalRouter()
    transport = httpx.MockTransport(fake.handle)
    real = httpx.AsyncClient

    def routed(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        return real(*args, transport=transport, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", routed)
    return fake


async def read_config(api: stream.Client, router: RegionalRouter) -> None:
    config = await api.agent("docs").config()

    assert config is not None and config.id == "config-1"
    request = router.asked[-1]
    assert request.url.host == REGIONAL_HOST
    assert request.headers["api_key"] == "test-key"
    assert request.headers["stream-auth-type"] == "jwt"
    assert request.headers["Authorization"]


class TestRegionalClients:
    async def test_the_default_client_reads_a_config(self, router: RegionalRouter):
        await read_config(stream.Client(), router)

    async def test_an_explicit_region_reads_a_config(self, router: RegionalRouter):
        await read_config(stream.Client(region="us-east4"), router)

    async def test_a_region_from_the_environment_reads_a_config(
        self, router: RegionalRouter, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(REGION_ENV, "us-east4")

        await read_config(stream.Client(), router)

    async def test_a_regions_own_url_with_a_trailing_slash_reads_a_config(
        self, router: RegionalRouter
    ):
        api = stream.Client(region="us-east4", url=DEFAULT_URL + "/")

        await read_config(api, router)

    @pytest.mark.usefixtures("router")
    def test_a_conflicting_url_and_region_are_refused(self):
        with pytest.raises(ValueError, match="not both"):
            stream.Client(region="europe-west4", url=DEFAULT_URL)

    @pytest.mark.usefixtures("router")
    def test_an_invalid_region_with_a_url_is_refused(self):
        with pytest.raises(ValueError):
            stream.Client(region="us-east4.evil.com/x", url=DEFAULT_URL)

    async def test_a_regional_client_can_act_for_a_user(self, router: RegionalRouter):
        api = stream.Client(region="us-east4")

        user = api.as_user("jean", "user-token")

        assert isinstance(user.backend, Backend)
        assert user.backend.url == DEFAULT_URL
        assert user.backend.user_id == "jean"
        assert user.backend.token == "user-token"
        await read_config(user, router)
