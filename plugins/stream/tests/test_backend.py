import base64
import hashlib
import hmac
import json
from dataclasses import replace
from typing import Any

import pytest

from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import (
    AUTHENTICATE_ENV,
    CUSTOMER_ENV,
    DEFAULT_URL,
    REGION_ENV,
    URL_ENV,
    USAGE_CATEGORIES_HEADER,
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


def _claims(token: str) -> dict[str, Any]:
    payload = token.split(".")[1]
    return json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))


def _verified(token: str, secret: str) -> dict[str, Any]:
    signing, _, signature = token.rpartition(".")
    mac = hmac.new(secret.encode(), signing.encode(), hashlib.sha256).digest()
    assert base64.urlsafe_b64encode(mac).rstrip(b"=").decode() == signature
    return _claims(token)


@pytest.mark.usefixtures("no_router_env")
class TestTeamsAndCategories:
    def test_a_token_carries_its_team_and_categories(self):
        api = stream.Client(
            url="http://router",
            api_key="key",
            api_secret="secret",
            authenticate=True,
            team="finance",
            usage_categories=["project:onboarding", "cohort:b", "project:onboarding"],
        )
        backend = api.backend

        server = _verified(backend._server_token(), "secret")
        assert server["server"] is True
        assert server["team"] == "finance"
        assert server["usage_categories"] == ["cohort:b", "project:onboarding"]
        assert _verified(backend._proxy_token(), "secret") == server

        token = api.create_user_token(
            "user-42", team="sales", usage_categories=["project:onboarding"]
        )
        claims = _verified(token, "secret")
        assert claims["user_id"] == "user-42"
        assert claims["team"] == "sales"
        assert claims["usage_categories"] == ["project:onboarding"]
        assert claims["exp"] - claims["iat"] == 3600

        # Behind the proxy the claims are the whole story: it deletes the header.
        assert USAGE_CATEGORIES_HEADER not in backend.headers

    def test_a_token_without_a_team_names_none(self):
        backend = Backend(url="http://router", api_key="key", api_secret="secret")

        claims = _claims(backend._server_token())
        assert "team" not in claims
        assert "usage_categories" not in claims
        assert USAGE_CATEGORIES_HEADER not in backend.headers

    @pytest.mark.parametrize(
        "categories",
        [
            [f"c{i}" for i in range(11)],
            [""],
            ["x" * 65],
            ["project onboarding"],
            "project:onboarding",
        ],
    )
    def test_categories_are_checked_before_anything_is_signed(self, categories):
        with pytest.raises(ValueError):
            Backend(
                url="http://router",
                api_key="key",
                api_secret="secret",
                usage_categories=categories,
            )

    @pytest.mark.parametrize(
        ("team", "categories", "match"),
        [
            ("fin ance", [], "not 1 to 59"),
            ("a:b", [], "not 1 to 59"),
            ("x" * 65, [], "not 1 to 59"),
            ("finance", ["team:sales"], "use team="),
            ("finance", ["Team:sales"], "use team="),
            ("finance", [f"c{i}" for i in range(10)], "at most 10"),
        ],
    )
    def test_a_team_is_checked_before_anything_is_signed(self, team, categories, match):
        with pytest.raises(ValueError, match=match):
            Backend(
                url="http://router",
                api_key="key",
                api_secret="secret",
                team=team,
                usage_categories=categories,
            )

    def test_without_the_proxy_the_team_and_categories_go_in_the_header(self):
        keyed = Backend(
            url="http://router",
            api_key="key",
            api_secret="secret",
            team="finance",
            usage_categories=["project:onboarding"],
        )
        assert (
            keyed.headers[USAGE_CATEGORIES_HEADER] == "project:onboarding,team:finance"
        )

        trusted = Backend(url="http://router", customer_id="acme", team="finance")
        assert trusted.headers[USAGE_CATEGORIES_HEADER] == "team:finance"
        sent = trusted.client().get_async_httpx_client().headers
        assert sent[USAGE_CATEGORIES_HEADER] == "team:finance"

    def test_a_user_token_cannot_be_minted_without_a_secret(self):
        device = stream.Client(url="http://router", api_key="key", token="jean-token")
        trusted = stream.Client(url="http://router", customer_id="acme")

        for client in (device, trusted):
            with pytest.raises(ValueError, match="secret"):
                client.create_user_token("user-42", team="finance")

    def test_a_token_handed_in_takes_no_team(self):
        with pytest.raises(ValueError, match="signed with"):
            Backend(url="http://router", api_key="key", token="t", team="finance")

    def test_for_user_keeps_the_apps_credential_and_names_the_user(self):
        api = stream.Client(
            url="http://router", api_key="key", api_secret="secret", team="ops"
        )

        alice = api.for_user("user-42", team="finance")

        assert alice.server_side
        headers = alice.backend.headers
        assert headers["X-Stream-User-Id"] == "user-42"
        assert headers["Stream-Auth-Type"] == "server"
        assert headers[USAGE_CATEGORIES_HEADER] == "team:finance"
        claims = _claims(headers["Authorization"].removeprefix("Bearer "))
        assert claims["server"] is True
        assert claims["team"] == "finance"
        # The parent's own team stays the parent's.
        assert api.backend.team == "ops"

        bob = api.for_user("bob")
        assert bob.backend.team is None
        assert USAGE_CATEGORIES_HEADER not in bob.backend.headers

    def test_for_user_behind_the_proxy_is_a_server_token_naming_the_user(self):
        api = stream.Client(
            url="http://router", api_key="key", api_secret="secret", authenticate=True
        )

        headers = api.for_user("user-42", team="finance").backend.headers

        assert headers["X-Stream-User-Id"] == "user-42"
        claims = _claims(headers["Authorization"].removeprefix("Bearer "))
        assert claims["server"] is True
        assert "user_id" not in claims
        assert claims["team"] == "finance"

    def test_for_user_is_server_side_only(self):
        device = stream.Client(url="http://router", api_key="key", token="jean-token")

        with pytest.raises(ValueError, match="only a backend"):
            device.for_user("user-42")
