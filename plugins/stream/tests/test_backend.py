from dataclasses import replace

import pytest

from vision_agents.plugins.stream._backend import (
    AUTHENTICATE_ENV,
    CUSTOMER_ENV,
    DEFAULT_URL,
    REGION_ENV,
    URL_ENV,
    Backend,
    region_url,
)


@pytest.fixture
def no_router_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (URL_ENV, REGION_ENV, CUSTOMER_ENV, AUTHENTICATE_ENV):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.usefixtures("no_router_env")
class TestBackend:
    def test_goes_to_the_hosted_router_through_the_proxy_when_nothing_names_another(
        self,
    ):
        backend = Backend(api_key="vak_live_x", token="token-for-jim")

        assert backend.url == DEFAULT_URL
        assert backend.authenticate is True

    def test_repr_shows_neither_the_secret_nor_the_token(self):
        backend = Backend(
            url="http://localhost:8080",
            api_key="vak_live_x",
            api_secret="vas_live_shh",
            token="token-shh",
        )

        shown = repr(backend)

        assert "vas_live_shh" not in shown
        assert "token-shh" not in shown
        assert "vak_live_x" in shown

    def test_leaves_the_proxy_off_for_a_router_it_was_pointed_at(self):
        backend = Backend(url="http://localhost:8080", customer_id="examples")

        assert backend.url == "http://localhost:8080"
        assert backend.authenticate is False

    def test_a_region_is_the_hosted_router_there(self):
        backend = Backend(
            region="us-east4", api_key="vak_live_x", token="token-for-jim"
        )

        assert backend.url == "https://accelerate-us-east4.gcp.stream-io-api.com"
        assert backend.authenticate is True

    def test_a_region_in_the_environment_gives_way_to_a_url_there(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(REGION_ENV, "us-east4")
        monkeypatch.setenv(URL_ENV, "http://localhost:8080")

        assert Backend(customer_id="acme").url == "http://localhost:8080"

        monkeypatch.delenv(URL_ENV)
        backend = Backend(api_key="vak_live_x", token="token-for-jim")

        assert backend.url == "https://accelerate-us-east4.gcp.stream-io-api.com"
        assert backend.authenticate is True

    def test_a_region_passed_in_beats_a_url_in_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setenv(URL_ENV, "http://localhost:8080")

        backend = Backend(
            region="europe-west4", api_key="vak_live_x", token="token-for-jim"
        )

        assert backend.url == "https://accelerate-europe-west4.gcp.stream-io-api.com"

    def test_a_url_and_a_region_together_are_refused(self):
        with pytest.raises(ValueError):
            Backend(
                url="https://acceleration.example.com",
                region="us-east4",
                customer_id="acme",
            )

    def test_a_region_survives_being_copied(self):
        backend = Backend(
            region="us-east4", api_key="vak_live_x", token="token-for-jim"
        )

        copied = replace(backend, acting_for="jlahey")

        assert copied.url == backend.url
        assert copied.region == "us-east4"

    @pytest.mark.parametrize(
        "name",
        [
            "US-EAST4",
            "us-east4.evil.com/x",
            "us_east4",
            "useast4",
            "us-east",
            "us-east4\n",
        ],
    )
    def test_anything_but_a_region_name_is_refused(self, name: str):
        with pytest.raises(ValueError, match="is not a region"):
            Backend(region=name, customer_id="acme")
        with pytest.raises(ValueError, match="is not a region"):
            region_url(name)

    def test_a_lookalike_of_a_regional_url_is_not_taken_for_the_hosted_router(self):
        backend = Backend(
            url="https://accelerate-us-east4.evil.com.gcp.stream-io-api.com",
            customer_id="acme",
        )

        assert backend.authenticate is False
