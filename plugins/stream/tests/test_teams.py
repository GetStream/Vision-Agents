import asyncio
import datetime
import os
import uuid
from decimal import Decimal
from typing import AsyncIterator, Awaitable, Callable

import pytest
from vision_agents.plugins import stream
from vision_agents.plugins.stream._backend import API_KEY_ENV, API_SECRET_ENV, URL_ENV

from .spend_router import WHEN, WINDOW_END, SpendRouter, served

LIMITS = "/v1/spend/limits"
LLM_TARGET = os.getenv("STREAM_ACCELERATION_LLM", "llm-fast")
# Usage reaches the spend counters through the usage service, a few seconds after the work.
COUNTED_WITHIN = 90.0


@pytest.fixture
async def spend_router() -> AsyncIterator[SpendRouter]:
    async with served() as fake:
        yield fake


@pytest.fixture
def spend_api(spend_router: SpendRouter) -> stream.Client:
    return stream.Client(url=spend_router.url, customer_id="acme")


class TestSetLimit:
    async def test_set_limit_creates_then_updates(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        created = await spend_api.teams.set_limit("finance", interval="weekly", usd=250)
        assert spend_router.body("POST", LIMITS) == {
            "subject": "team",
            "team": "finance",
            "interval": "weekly",
            "limit_micros": 250_000_000,
        }

        updated = await spend_api.teams.set_limit(
            "finance", interval="weekly", usd="300"
        )

        assert updated.id == created.id
        assert spend_router.requests("POST", LIMITS) == 1
        assert spend_router.body("PATCH", f"{LIMITS}/{created.id}") == {
            "limit_micros": 300_000_000
        }
        assert [row["limit_micros"] for row in spend_router.limits] == [300_000_000]

    async def test_a_team_within_an_app_is_its_own_limit(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        across = await spend_api.teams.set_limit("finance", interval="daily", usd=100)
        within = await spend_api.teams.set_limit(
            "finance", app_id="1234", interval="daily", usd=40
        )

        assert within.id != across.id
        assert spend_router.body("POST", LIMITS)["subject"] == "app_team"
        assert spend_router.requests("PATCH", f"{LIMITS}/{across.id}") == 0

    async def test_a_create_that_loses_a_race_patches_the_winner(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        winner = spend_router.add_limit(
            subject="team", team="finance", interval="daily", limit_micros=1
        )
        spend_router.lists_before_create = 1

        updated = await spend_api.teams.set_limit("finance", interval="daily", usd=40)

        assert spend_router.requests("POST", LIMITS) == 1
        assert updated.id == winner["id"]
        assert updated.limit_micros == 40_000_000

    async def test_remove_limit_without_one_does_nothing(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        await spend_api.teams.remove_limit("finance", interval="daily")

        assert spend_router.requests("GET", LIMITS) == 1
        assert not any(line.startswith("DELETE") for line in spend_router.asked)

    async def test_remove_limit_deletes_the_teams_own(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.add_limit(subject="team", team="finance", interval="daily")
        kept = spend_router.add_limit(
            subject="app_team", team="finance", app_id="1234", interval="daily"
        )

        await spend_api.teams.remove_limit("finance", interval="daily")

        assert spend_router.limits == [kept]

    async def test_a_team_is_checked_before_any_call(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        for bad in ("fin ance", "a:b", "x" * 60, ""):
            with pytest.raises(ValueError):
                await spend_api.teams.set_limit(bad, interval="daily", usd=1)
        assert spend_router.asked == []


class TestList:
    async def test_teams_lists_what_the_router_answers(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spent = {
            "interval": "daily",
            "window_start": WHEN,
            "window_end": WINDOW_END,
            "spent_micros": 12_500_000,
        }
        spend_router.teams = [
            {"name": "finance", "apps": ["1234"], "spent": [spent], "limits": []},
            {"name": "sales", "apps": ["5678"], "spent": [spent], "limits": []},
        ]

        teams = await spend_api.teams.list()
        assert [team.name for team in teams] == ["finance", "sales"]
        assert stream.usd(teams[0].spent[0].spent_micros) == Decimal("12.5")
        assert "app_id" not in spend_router.query("GET", "/v1/teams")

        within = await spend_api.teams.list(app_id="1234")
        assert [team.name for team in within] == ["finance"]
        assert spend_router.query("GET", "/v1/teams")["app_id"] == "1234"


class TestAccount:
    async def test_the_account_shows_the_spend_hard_limit(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.account["spend_hard_limit"] = {
            "daily": {
                "limit_micros": 100_000_000,
                "spent_micros": 40_000_000,
                "room_micros": 60_000_000,
                "window_end": WINDOW_END,
            }
        }

        account = await spend_api.account.get()

        assert account.spend_hard_limit and account.spend_hard_limit.daily
        assert stream.usd(account.spend_hard_limit.daily.room_micros) == 60
        assert stream.usd(account.spend_hard_limit.daily.limit_micros) == 100
        assert not account.spend_hard_limit.monthly

        del spend_router.account["spend_hard_limit"]
        assert not (await spend_api.account.get()).spend_hard_limit

    async def test_credits_iterates_every_page(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.page_size = 2
        spend_router.credits = [
            {"id": i, "kind": "top_up", "amount_micros": i, "created_at": WHEN}
            for i in range(1, 6)
        ]

        entries = [entry async for entry in spend_api.account.credits()]

        assert [entry.id for entry in entries] == [1, 2, 3, 4, 5]
        assert spend_router.requests("GET", "/v1/account/credits") == 3


class TestSpend:
    async def test_spend_by_team(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.buckets = [
            {
                "bucket": WHEN,
                "value": "finance",
                "cost_micros_total": 3_000_000,
                "request_count": 4,
            }
        ]
        start = datetime.datetime(2026, 11, 1, tzinfo=datetime.timezone.utc)

        buckets = await spend_api.usage.spend(
            start=start,
            end=start + datetime.timedelta(days=7),
            group_by="who:team",
            categories=["project:onboarding"],
        )

        assert [(b.value, b.cost_micros_total) for b in buckets] == [
            ("finance", 3_000_000)
        ]
        query = spend_router.query("GET", "/v1/stats/spend")
        assert query["group_by"] == "who:team"
        assert query["granularity"] == "daily"
        assert query.getall("category") == ["project:onboarding"]
        with pytest.raises(ValueError, match="needs a timezone"):
            await spend_api.usage.spend(
                start=datetime.datetime(2026, 11, 1), end=start, group_by="who:team"
            )


@pytest.mark.integration
class TestOnARouterThatBills:
    """Against a hosted router that bills, with an app's key and secret."""

    @pytest.fixture
    def api(self) -> stream.Client:
        needed = (URL_ENV, API_KEY_ENV, API_SECRET_ENV)
        missing = [name for name in needed if not os.getenv(name)]
        if missing:
            pytest.fail(
                f"These tests need a router that bills and an app's credential: set "
                f"{', '.join(missing)}, and STREAM_ACCELERATION_AUTHENTICATE=1 for a router "
                "behind the proxy.",
                pytrace=False,
            )
        return stream.Client()

    async def test_work_for_a_user_counts_against_their_team(self, api: stream.Client):
        team = f"t10-{uuid.uuid4().hex[:12]}"

        await _answer(api.for_user("t10-user", team=team))

        await _counted(lambda: api.teams.usage(team))

    async def test_work_for_a_user_counts_against_their_category(
        self, api: stream.Client
    ):
        category = f"project:t10-{uuid.uuid4().hex[:12]}"

        await _answer(api.for_user("t10-user", usage_categories=[category]))

        await _counted(lambda: api.spend_limits.usage("category", category=category))


async def _answer(client: stream.Client) -> None:
    """One billable LLM answer, run as the client."""
    llm = stream.LLM(LLM_TARGET, backend=client.backend)
    await llm.start()
    try:
        answers = [answer async for answer in llm.simple_response("Say hello.")]
    finally:
        await llm.close()
    assert answers and answers[-1].text


async def _counted(read: Callable[[], Awaitable[stream.SubjectUsage]]) -> None:
    """Wait for the subject's daily spend to show the answer's cost."""
    deadline = asyncio.get_running_loop().time() + COUNTED_WITHIN
    while True:
        used = await read()
        daily = next(w for w in used.windows or [] if w.interval == "daily")
        if daily.spent_micros > 0:
            return
        if asyncio.get_running_loop().time() > deadline:
            pytest.fail(f"no spend counted within {COUNTED_WITHIN:.0f} s: {used}")
        await asyncio.sleep(2)
