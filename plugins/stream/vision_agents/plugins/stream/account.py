"""The organization's account with Stream, and what its work cost."""

from __future__ import annotations

import datetime
from typing import Any, AsyncIterator, List, Literal, Optional, Sequence

from ._backend import Backend, check_team
from ._generated.api.default import get_account, get_spend, list_credits
from ._generated.models import (
    Account,
    CreditEntry,
    Granularity,
    SpendBucket,
    SpendScope,
)
from ._generated.types import UNSET
from .responses import _set, _unwrapped
from .spend_limits import _utc

# The most credit entries the router answers in one page.
_CREDIT_PAGE = 100


class OrganizationAccount:
    """The organization's balance, credit and what Stream allows it. Server side only."""

    def __init__(self, backend: Backend):
        self._backend = backend

    async def get(self) -> Account:
        """The account: balance, status, tier and Stream's spend limit for the organization.

        ``spend_hard_limit`` is Stream's limit on what the organization may spend in a day
        and in a month, whatever its credit, with each window's limit, spend and room. Unset
        when Stream set none. Only Stream's staff change it, and no spend limit of the
        customer's or grant lifts it.

        Raises ``RouterError`` with ``code="billing_not_enabled"`` (404) where the router
        bills nobody.
        """
        return await _unwrapped(
            get_account.asyncio(client=self._backend.client()),
            "reading the account",
        )

    async def credits(self) -> AsyncIterator[CreditEntry]:
        """Every credit entry, newest first, read page by page as it is iterated."""
        cursor: Any = UNSET
        while True:
            page = await _unwrapped(
                list_credits.asyncio(
                    client=self._backend.client(), limit=_CREDIT_PAGE, cursor=cursor
                ),
                "listing the account's credit",
            )
            for entry in page.items:
                yield entry
            if not page.has_more or not page.next_cursor:
                return
            cursor = page.next_cursor


class Usage:
    """What the app's or the organization's work cost, over time."""

    def __init__(self, backend: Backend):
        self._backend = backend

    async def spend(
        self,
        *,
        start: datetime.datetime,
        end: datetime.datetime,
        group_by: str = "modality",
        scope: Literal["app", "organization"] = "app",
        granularity: Literal["daily", "hourly"] = "daily",
        limit: int = 6,
        tags: Sequence[str] = (),
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        categories: Sequence[str] = (),
    ) -> List[SpendBucket]:
        """Spend from ``start`` up to ``end``, in buckets of the granularity, one series per
        value of ``group_by``.

        Args:
            group_by: ``"modality"``, who asked (``"who:app"``, ``"who:api_key"``,
                ``"who:user"``, ``"who:team"`` or ``"who:category"``), or a cost label.
            scope: The app's spend, or every app's of the organization (server side only).
            limit: How many values keep a series of their own; the rest are summed.
            tags: Only requests carrying every one of these cost labels, ``"key:value"``.
            team: Only the work of this team's users.
            categories: Only work carrying every one of these usage categories.
        """
        if team is not None:
            check_team(team)
        query: dict[str, Any] = _set(api_key=api_key, user_id=user_id, team=team)
        if tags:
            query["tag"] = list(tags)
        if categories:
            query["category"] = list(categories)
        return await _unwrapped(
            get_spend.asyncio(
                client=self._backend.client(),
                from_=_utc(start, "start"),
                to=_utc(end, "end"),
                group_by=group_by,
                scope=SpendScope(scope),
                granularity=Granularity(granularity),
                limit=limit,
                **query,
            ),
            "reading spend",
        )
