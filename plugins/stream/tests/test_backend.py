import pytest

from vision_agents.plugins.stream._backend import (
    AUTHENTICATE_ENV,
    CUSTOMER_ENV,
    DEFAULT_URL,
    URL_ENV,
    Backend,
)


@pytest.fixture
def no_router_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (URL_ENV, CUSTOMER_ENV, AUTHENTICATE_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_router_env")
class TestBackend:
    def test_goes_to_the_hosted_router_through_the_proxy_when_nothing_names_another(
        self,
    ):
        backend = Backend(api_key="vak_live_x", token="token-for-jim")

        assert backend.url == DEFAULT_URL
        assert backend.authenticate is True

    def test_leaves_the_proxy_off_for_a_router_it_was_pointed_at(self):
        backend = Backend(url="http://localhost:8080", customer_id="examples")

        assert backend.url == "http://localhost:8080"
        assert backend.authenticate is False
