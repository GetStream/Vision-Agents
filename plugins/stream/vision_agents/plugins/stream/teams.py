"""The customer's teams and their spend limits."""

from __future__ import annotations

import datetime
from typing import List, Optional

from ._backend import Backend, check_team
from ._generated.api.default import list_teams
from ._generated.models import Limit, LimitGrant, SubjectUsage, Team
from ._generated.types import Unset
from .responses import RouterError, _set, _unwrapped
from .spend_limits import USD, Interval, SpendLimits


class Teams:
    """The organization's teams. Server side only.

    A team has no members list: a user is in the team its token names, from
    ``create_user_token(..., team=...)`` or ``for_user(..., team=...)``. A team's spend is
    the spend of the work those tokens ran, across the organization or within one app.
    """

    def __init__(self, backend: Backend, spend_limits: SpendLimits):
        self._backend = backend
        self._limits = spend_limits

    async def list(self, *, app_id: Optional[str] = None) -> List[Team]:
        """Every team with a limit or with spend this month, each with its apps, its spend
        in the current windows and its limits. With ``app_id``, only within that app."""
        return await _unwrapped(
            list_teams.asyncio(client=self._backend.client(), **_set(app_id=app_id)),
            "listing teams",
        )

    async def set_limit(
        self,
        team: str,
        *,
        interval: Interval,
        usd: USD,
        app_id: Optional[str] = None,
        note: str = "",
    ) -> Limit:
        """Make the team's limit for ``interval`` the amount given: created when it has
        none, changed when it has one. Across the organization, or within ``app_id``."""
        existing = await self._limit(team, interval, app_id)
        if existing is None:
            try:
                return await self._limits.create(
                    "team",
                    team=team,
                    app_id=app_id,
                    interval=interval,
                    usd=usd,
                    note=note,
                )
            except RouterError as refused:
                # Somebody else created it since it was looked up.
                if refused.code != "limit_exists":
                    raise
            existing = await self._limit(team, interval, app_id)
            if existing is None:
                raise RouterError(
                    f"the {interval} limit on team {team} exists and could not be found"
                )
        return await self._limits.update(existing.id, usd=usd, note=note or None)

    async def remove_limit(
        self, team: str, *, interval: Interval, app_id: Optional[str] = None
    ) -> None:
        """Delete the team's limit for ``interval``. Nothing happens when it has none."""
        existing = await self._limit(team, interval, app_id)
        if existing is None:
            return
        try:
            await self._limits.delete(existing.id)
        except RouterError as refused:
            # Somebody else deleted it since it was looked up.
            if refused.status != 404:
                raise

    async def usage(self, team: str, *, app_id: Optional[str] = None) -> SubjectUsage:
        """What the team spent in its current windows, against its limits and grants."""
        return await self._limits.usage("team", team=team, app_id=app_id)

    async def grant(
        self,
        team: str,
        *,
        usd: USD,
        reason: str,
        expires_at: Optional[datetime.datetime] = None,
        app_id: Optional[str] = None,
    ) -> LimitGrant:
        """A one-time amount the team may spend past its spent limits, as
        ``spend_limits.grant`` gives one."""
        return await self._limits.grant(
            "team",
            team=team,
            app_id=app_id,
            usd=usd,
            reason=reason,
            expires_at=expires_at,
        )

    async def _limit(
        self, team: str, interval: Interval, app_id: Optional[str]
    ) -> Optional[Limit]:
        """The team's own limit in the interval, across the organization or within the app."""
        check_team(team)
        for limit in await self._limits.list(
            subject="team", team=team, app_id=app_id, interval=interval
        ):
            in_app = "" if isinstance(limit.app_id, Unset) else limit.app_id
            if (
                limit.team == team
                and limit.interval == interval
                and in_app == (app_id or "")
            ):
                return limit
        return None
