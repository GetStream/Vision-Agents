import datetime
from decimal import Decimal
from typing import AsyncIterator

import pytest
from vision_agents.plugins import stream
from vision_agents.plugins.stream.spend_limits import micros

from .spend_router import SpendRouter, served

LIMITS = "/v1/spend/limits"
GRANTS = "/v1/spend/limits/grants"
USAGE = "/v1/spend/limits/usage"


@pytest.fixture
async def spend_router() -> AsyncIterator[SpendRouter]:
    async with served() as fake:
        yield fake


@pytest.fixture
def spend_api(spend_router: SpendRouter) -> stream.Client:
    return stream.Client(url=spend_router.url, customer_id="acme")


class TestMoney:
    def test_dollars_become_micros_exactly(self):
        assert micros("0.000001") == 1
        assert micros(Decimal("19.99")) == 19_990_000
        assert micros(5) == 5_000_000
        assert micros(" 250 ") == 250_000_000
        with pytest.raises(TypeError, match="float"):
            micros(0.1)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            micros(True)

    def test_an_amount_finer_than_a_micro_or_below_zero_is_refused(self):
        with pytest.raises(ValueError, match="finer than a millionth"):
            micros("0.0000001")
        with pytest.raises(ValueError, match="below zero"):
            micros(-1)
        with pytest.raises(ValueError, match="not an amount"):
            micros("ten")
        with pytest.raises(ValueError, match="not an amount"):
            micros("NaN")

    def test_micros_read_back_as_dollars(self):
        assert stream.usd(19_990_000) == Decimal("19.99")
        assert stream.usd(1) == Decimal("0.000001")


class TestCreate:
    async def test_the_level_follows_the_ids(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        cases = [
            ({"user_id": "user-42", "app_id": "1234"}, "user"),
            ({"app_id": "1234"}, "app"),
            ({}, "organization"),
        ]
        for subject, id_name in (("user", "user_id"), ("api_key", "api_key")):
            for ids, level in cases:
                named = dict(ids)
                if "user_id" in named:
                    named[id_name] = named.pop("user_id")
                created = await spend_api.spend_limits.create(
                    subject, interval="weekly", usd=5, **named
                )

                sent = spend_router.body("POST", LIMITS)
                assert "level" not in sent
                assert {k: sent[k] for k in named} == named
                assert created.level == (subject if level == "user" else level)
                spend_router.limits.clear()

    async def test_a_level_given_is_sent(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        await spend_api.spend_limits.create(
            "user", app_id="1234", level="app", interval="daily", usd="2.50"
        )

        sent = spend_router.body("POST", LIMITS)
        assert sent["level"] == "app"
        assert sent["limit_micros"] == 2_500_000

    async def test_a_team_with_an_app_is_an_app_team(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        await spend_api.spend_limits.create(
            "team", team="finance", app_id="1234", interval="daily", usd=40
        )
        assert spend_router.body("POST", LIMITS)["subject"] == "app_team"

        await spend_api.spend_limits.create(
            "team", team="finance", interval="weekly", usd=250
        )
        sent = spend_router.body("POST", LIMITS)
        assert sent["subject"] == "team"
        assert "app_id" not in sent

        with pytest.raises(ValueError, match="use subject='team'"):
            await spend_api.spend_limits.create(
                "category", category="team:finance", interval="daily", usd=1
            )
        assert spend_router.requests("POST", LIMITS) == 2

    async def test_a_category_with_an_app_is_an_app_category(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        await spend_api.spend_limits.create(
            "category",
            category="project:onboarding",
            app_id="1234",
            interval="monthly",
            usd=100,
        )
        sent = spend_router.body("POST", LIMITS)
        assert sent["subject"] == "app_category"
        assert sent["category"] == "project:onboarding"

    async def test_a_subject_without_its_id_is_refused_before_any_call(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        with pytest.raises(ValueError, match="names its team"):
            await spend_api.spend_limits.create("team", interval="daily", usd=1)
        with pytest.raises(ValueError, match="team= is for subject='team'"):
            await spend_api.spend_limits.create(
                "user", team="finance", interval="daily", usd=1
            )
        with pytest.raises(ValueError, match="names its app"):
            await spend_api.spend_limits.create(
                "app_team", team="finance", interval="daily", usd=1
            )
        with pytest.raises(ValueError, match="not 1 to 59"):
            await spend_api.spend_limits.create(
                "team", team="fin ance", interval="daily", usd=1
            )
        assert spend_router.asked == []

    async def test_a_limit_above_its_parent_is_refused(self, spend_api: stream.Client):
        with pytest.raises(stream.RouterError) as refused:
            await spend_api.spend_limits.create(
                "organization", interval="monthly", usd=50_000
            )
        assert refused.value.code == "limit_above_parent"
        assert refused.value.status == 400


class TestChange:
    async def test_update_sends_only_what_changes(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        row = spend_router.add_limit(subject="organization")

        updated = await spend_api.spend_limits.update(row["id"], usd="19.99")

        assert spend_router.body("PATCH", f"{LIMITS}/{row['id']}") == {
            "limit_micros": 19_990_000
        }
        assert updated.limit_micros == 19_990_000
        with pytest.raises(ValueError, match="changes the amount"):
            await spend_api.spend_limits.update(row["id"])

    async def test_delete_removes_it(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        row = spend_router.add_limit(subject="organization")

        await spend_api.spend_limits.delete(row["id"])

        assert spend_router.limits == []
        with pytest.raises(stream.RouterError) as refused:
            await spend_api.spend_limits.delete(row["id"])
        assert refused.value.code == "limit_not_found"

    async def test_list_reads_every_page_with_its_filters(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.page_size = 2
        for user in ("a", "b", "c", "d", "e"):
            spend_router.add_limit(subject="user", app_id="1234", user_id=user)
        spend_router.add_limit(subject="organization")

        limits = await spend_api.spend_limits.list(subject="user", app_id="1234")

        assert [limit.user_id for limit in limits] == ["a", "b", "c", "d", "e"]
        assert spend_router.requests("GET", LIMITS) == 3
        query = spend_router.query("GET", LIMITS)
        assert query["subject"] == "user"
        assert query["app_id"] == "1234"
        assert query["cursor"] == "4"


class TestGrants:
    async def test_a_grant_sends_its_expiry_in_utc(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.add_limit(subject="team", team="finance", interval="daily")
        paris = datetime.timezone(datetime.timedelta(hours=2))

        granted = await spend_api.teams.grant(
            "finance",
            usd=5000,
            reason="Q4 close",
            expires_at=datetime.datetime(2026, 11, 15, 2, 0, tzinfo=paris),
        )

        sent = spend_router.body("POST", GRANTS)
        assert sent["subject"] == "team"
        assert sent["team"] == "finance"
        assert sent["amount_micros"] == 5_000_000_000
        assert sent["expires_at"] == "2026-11-15T00:00:00+00:00"
        assert granted.remaining_micros == 5_000_000_000

        await spend_api.teams.grant("finance", usd=1, reason="no expiry")
        assert "expires_at" not in spend_router.body("POST", GRANTS)

        with pytest.raises(ValueError, match="needs a timezone"):
            await spend_api.teams.grant(
                "finance",
                usd=1,
                reason="naive",
                expires_at=datetime.datetime(2026, 11, 15),
            )
        assert spend_router.requests("POST", GRANTS) == 2

    async def test_a_grant_without_a_limit_is_no_limit(self, spend_api: stream.Client):
        with pytest.raises(stream.RouterError) as refused:
            await spend_api.spend_limits.grant(
                "user", app_id="1234", user_id="user-42", usd=10, reason="ticket 8812"
            )
        assert refused.value.code == "no_limit"
        assert refused.value.status == 404

    async def test_grants_returns_every_page(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.page_size = 2
        for _ in range(5):
            spend_router.add_grant(subject="organization")

        grants = await spend_api.spend_limits.grants(state="live")

        assert len(grants) == 5
        assert spend_router.requests("GET", GRANTS) == 3
        assert spend_router.query("GET", GRANTS)["state"] == "live"

    async def test_revoke_grant_posts_its_reason(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        row = spend_router.add_grant(subject="organization")

        revoked = await spend_api.spend_limits.revoke_grant(
            row["id"], reason="close finished early"
        )

        assert spend_router.body("POST", f"{GRANTS}/{row['id']}/revoke") == {
            "reason": "close finished early"
        }
        assert revoked.state == "revoked"


class TestUsage:
    async def test_usage_returns_the_one_subject_of_the_list(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        used = await spend_api.spend_limits.usage(
            "user", app_id="1234", user_id="user-42"
        )

        assert used.user_id == "user-42"
        assert stream.usd(used.windows[0].spent_micros) == Decimal("1.25")
        query = spend_router.query("GET", USAGE)
        assert query.getall("user_id") == ["user-42"]
        assert query["subject"] == "user"

        team = await spend_api.teams.usage("finance", app_id="1234")
        assert team.team == "finance"
        assert spend_router.query("GET", USAGE)["subject"] == "app_team"

    async def test_usage_many_answers_one_per_subject_in_order(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        used = await spend_api.spend_limits.usage_many(
            "user", app_id="1234", user_id=["a", "b", "c"]
        )

        assert [u.user_id for u in used] == ["a", "b", "c"]
        assert spend_router.query("GET", USAGE).getall("user_id") == ["a", "b", "c"]

        with pytest.raises(ValueError, match="at most 100"):
            await spend_api.spend_limits.usage_many(
                "user", app_id="1234", user_id=[f"u{i}" for i in range(101)]
            )
        with pytest.raises(ValueError, match="one per subject"):
            await spend_api.spend_limits.usage_many(
                "user", app_id=["1", "2"], user_id=["a", "b", "c"]
            )
        assert spend_router.requests("GET", USAGE) == 1

    async def test_reset_names_its_subject_and_reason(
        self, spend_api: stream.Client, spend_router: SpendRouter
    ):
        spend_router.add_limit(subject="user", app_id="1234", user_id="user-42")

        await spend_api.spend_limits.reset(
            "user",
            app_id="1234",
            user_id="user-42",
            reason="refunded a broken session",
            intervals=["daily"],
        )

        assert spend_router.body("POST", f"{USAGE}/reset") == {
            "subject": "user",
            "app_id": "1234",
            "user_id": "user-42",
            "reason": "refunded a broken session",
            "intervals": ["daily"],
        }
        with pytest.raises(stream.RouterError) as refused:
            await spend_api.spend_limits.reset(
                "team", team="sales", reason="nothing to reset"
            )
        assert refused.value.code == "no_limit"


class TestNames:
    def test_there_is_no_limits_attribute(self):
        api = stream.Client(url="http://router", customer_id="acme")

        with pytest.raises(AttributeError):
            api.limits  # type: ignore[attr-defined]
