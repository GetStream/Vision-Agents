"""The customer's spend limits: on the organization, its apps, keys, end users, teams and usage
categories."""

from __future__ import annotations

import datetime
from decimal import Decimal, InvalidOperation
from typing import Any, List, Literal, Optional, Sequence, TypeVar, Union

from ._backend import Backend, check_category, check_team
from ._generated.api.default import (
    create_limit,
    create_limit_grant,
    delete_limit,
    get_limit_usage,
    list_limit_grants,
    list_limits,
    reset_limit_usage,
    revoke_limit_grant,
    update_limit,
)
from ._generated.models import (
    Limit,
    LimitChange,
    LimitGrant,
    LimitGrantInput,
    LimitGrantRevocation,
    LimitGrantState,
    LimitInput,
    LimitInterval,
    LimitLevel,
    LimitResetInput,
    LimitSubject,
    SubjectUsage,
)
from ._generated.types import UNSET
from .responses import RouterError, _deleted, _unwrapped

Subject = Literal[
    "organization",
    "app",
    "api_key",
    "user",
    "team",
    "app_team",
    "category",
    "app_category",
]
Level = Literal["organization", "app", "api_key", "user"]
Interval = Literal["daily", "weekly", "biweekly", "monthly"]
GrantState = Literal["live", "used", "expired", "revoked"]
# Dollars, written exactly. A float is refused: 0.1 + 0.2 is not a price.
USD = Union[Decimal, int, str]
Ids = Union[str, Sequence[str], None]

MICROS_PER_USD = 1_000_000
# The most subjects one usage read names, as the router counts them.
MAX_USAGE_SUBJECTS = 100
# The page size lists are read in, the most the router answers.
_PAGE = 200


def usd(micros: int) -> Decimal:
    """Dollars from the millionths of a dollar the router answers in, exactly."""
    return Decimal(micros).scaleb(-6)


def micros(amount: USD) -> int:
    """Millionths of a dollar from dollars, exactly, or why they are not an amount."""
    if isinstance(amount, (bool, float)) or not isinstance(amount, (int, str, Decimal)):
        raise TypeError(
            f"an amount is a Decimal, an int or a str of dollars, not {type(amount).__name__}"
            ": a float is not exact"
        )
    try:
        value = Decimal(amount.strip() if isinstance(amount, str) else amount)
    except InvalidOperation:
        raise ValueError(f"{amount!r} is not an amount of dollars") from None
    if not value.is_finite():
        raise ValueError(f"{amount!r} is not an amount of dollars")
    exact = value.scaleb(6)
    if exact != exact.to_integral_value():
        raise ValueError(f"{amount!r} is finer than a millionth of a dollar")
    if exact < 0:
        raise ValueError(f"{amount!r} is below zero")
    return int(exact)


class SpendLimits:
    """Spend limits on the organization, its apps, keys, users, teams and usage categories.

    Every limit here is the customer's own. Stream's spend limit for the organization is
    never among them: it is ``spend_hard_limit`` on ``api.account.get()``, and nothing here
    lifts it. Server side only: the router refuses a user's token, 403 ``server_side_only``.

    Work a limit refuses raises ``RouterError`` with ``status=402,
    code="spend_limit_reached"``, or ``429 user_limit_reached`` for one end user's limit.
    Stream's limit refuses with ``402 system_limit_reached``.
    """

    def __init__(self, backend: Backend):
        self._backend = backend

    async def list(
        self,
        *,
        subject: Optional[Subject] = None,
        level: Optional[Level] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
        interval: Optional[Interval] = None,
    ) -> List[Limit]:
        """Every limit the filters match, read page by page.

        ``subject="team"`` lists the teams' limits, ``"category"`` every other category's.
        With ``app_id`` they are the ones within that app.
        """
        filters = _filters(subject, app_id, api_key, user_id, team, category)
        if level is not None:
            filters["level"] = LimitLevel(level)
        if interval is not None:
            filters["interval"] = LimitInterval(interval)
        limits: List[Limit] = []
        cursor: Any = UNSET
        while True:
            page = await _unwrapped(
                list_limits.asyncio(
                    client=self._backend.client(), limit=_PAGE, cursor=cursor, **filters
                ),
                "listing spend limits",
            )
            limits.extend(page.items or [])
            if not page.has_more or not page.next_cursor:
                return limits
            cursor = page.next_cursor

    async def create(
        self,
        subject: Subject,
        *,
        interval: Interval,
        usd: USD,
        level: Optional[Level] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
        note: str = "",
    ) -> Limit:
        """Create a limit: what one subject may spend in each window of ``interval``.

        The level is the most specific one the ids name, which the router works out:
        ``subject="user", user_id=...`` is that user's own limit, ``subject="user",
        app_id=...`` the default for every user of the app, and ``subject="user"`` alone the
        default for every user of the organization. Keys go the same way. ``level=``
        overrides it. ``subject="team"`` with ``app_id=`` is the team within that app.

        Raises ``RouterError`` with ``code`` ``limit_above_parent`` (the limit is above one
        of a subject that contains it), ``limit_exists`` (the subject has one at that level
        and interval; change it with ``update``) or ``use_team``.
        """
        kind, ids = _target(subject, app_id, api_key, user_id, team, category)
        body = _with_ids(
            LimitInput(
                subject=kind, interval=LimitInterval(interval), limit_micros=micros(usd)
            ),
            ids,
        )
        if level is not None:
            body.level = LimitLevel(level)
        if note:
            body.note = note
        return await _unwrapped(
            create_limit.asyncio(client=self._backend.client(), body=body),
            f"creating a {interval} spend limit on {_named(kind, ids)}",
        )

    async def update(
        self, limit_id: int, *, usd: Optional[USD] = None, note: Optional[str] = None
    ) -> Limit:
        """Change a limit's amount, its note or both. What it is on never changes."""
        if usd is None and note is None:
            raise ValueError("an update changes the amount, the note or both")
        change = LimitChange()
        if usd is not None:
            change.limit_micros = micros(usd)
        if note is not None:
            change.note = note
        return await _unwrapped(
            update_limit.asyncio(limit_id, client=self._backend.client(), body=change),
            f"updating the spend limit {limit_id}",
        )

    async def delete(self, limit_id: int) -> None:
        """Delete a limit. Its subject's spend from then on is held only to its parents'."""
        await _deleted(
            delete_limit.asyncio_detailed(limit_id, client=self._backend.client()),
            f"deleting the spend limit {limit_id}",
        )

    async def grant(
        self,
        subject: Subject,
        *,
        usd: USD,
        reason: str,
        expires_at: Optional[datetime.datetime] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
    ) -> LimitGrant:
        """A one-time amount one subject may spend past its own spent limits.

        It is drawn while any of the subject's limits is spent, across windows, until it is
        used up, ``expires_at`` passes or it is revoked. It never lifts a parent's limit or
        Stream's. "$10 more today" is a grant that expires at the next midnight UTC.

        Raises ``RouterError`` with ``code`` ``no_limit`` (404: the subject has no limit of
        the customer's to grant past), ``invalid_expiry`` or ``too_many_grants``.
        """
        if not reason:
            raise ValueError("a grant needs a reason")
        kind, ids = _target(subject, app_id, api_key, user_id, team, category)
        body = _with_ids(
            LimitGrantInput(subject=kind, amount_micros=micros(usd), reason=reason), ids
        )
        if expires_at is not None:
            body.expires_at = _utc(expires_at, "expires_at")
        return await _unwrapped(
            create_limit_grant.asyncio(client=self._backend.client(), body=body),
            f"granting {_named(kind, ids)} past its spend limits",
        )

    async def grants(
        self,
        *,
        subject: Optional[Subject] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
        state: Optional[GrantState] = None,
    ) -> List[LimitGrant]:
        """Every grant the filters match, read page by page."""
        filters = _filters(subject, app_id, api_key, user_id, team, category)
        if state is not None:
            filters["state"] = LimitGrantState(state)
        grants: List[LimitGrant] = []
        cursor: Any = UNSET
        while True:
            page = await _unwrapped(
                list_limit_grants.asyncio(
                    client=self._backend.client(), limit=_PAGE, cursor=cursor, **filters
                ),
                "listing spend limit grants",
            )
            grants.extend(page.items or [])
            if not page.has_more or not page.next_cursor:
                return grants
            cursor = page.next_cursor

    async def revoke_grant(self, grant_id: int, *, reason: str) -> LimitGrant:
        """End a grant before it is used up or expires."""
        if not reason:
            raise ValueError("revoking a grant needs a reason")
        return await _unwrapped(
            revoke_limit_grant.asyncio(
                grant_id,
                client=self._backend.client(),
                body=LimitGrantRevocation(reason=reason),
            ),
            f"revoking the grant {grant_id}",
        )

    async def usage(
        self,
        subject: Subject,
        *,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
    ) -> SubjectUsage:
        """What one subject spent in its current windows, against its limits and grants."""
        answered = await self._usage(subject, app_id, api_key, user_id, team, category)
        if len(answered) != 1:
            raise RouterError(
                f"reading the spend of one {subject}: the router answered {len(answered)}"
            )
        return answered[0]

    async def usage_many(
        self,
        subject: Subject,
        *,
        app_id: Ids = None,
        api_key: Ids = None,
        user_id: Ids = None,
        team: Ids = None,
        category: Ids = None,
    ) -> List[SubjectUsage]:
        """What several subjects spent, one ``SubjectUsage`` each, in the order asked.

        Each id is one value for every subject, or a sequence of one per subject: at most
        100, and every sequence of more than one the same length.
        """
        return await self._usage(subject, app_id, api_key, user_id, team, category)

    async def reset(
        self,
        subject: Subject,
        *,
        reason: str,
        intervals: Optional[Sequence[Interval]] = None,
        app_id: Optional[str] = None,
        api_key: Optional[str] = None,
        user_id: Optional[str] = None,
        team: Optional[str] = None,
        category: Optional[str] = None,
    ) -> SubjectUsage:
        """Count a subject's limits from now until each window ends, every interval's or
        those named. Nothing is refunded: what it spent stays spent.

        Raises ``RouterError`` with ``code="no_limit"`` for a subject with no limit of the
        customer's.
        """
        if not reason:
            raise ValueError("a reset needs a reason")
        kind, ids = _target(subject, app_id, api_key, user_id, team, category)
        body = _with_ids(LimitResetInput(subject=kind, reason=reason), ids)
        if intervals is not None:
            body.intervals = [LimitInterval(interval) for interval in intervals]
        return await _unwrapped(
            reset_limit_usage.asyncio(client=self._backend.client(), body=body),
            f"resetting the spend of {_named(kind, ids)}",
        )

    async def _usage(
        self,
        subject: Subject,
        app_id: Ids,
        api_key: Ids,
        user_id: Ids,
        team: Ids,
        category: Ids,
    ) -> List[SubjectUsage]:
        lists = {
            name: _listed(value)
            for name, value in {
                "app_id": app_id,
                "api_key": api_key,
                "user_id": user_id,
                "team": team,
                "category": category,
            }.items()
            if value is not None
        }
        count = max((len(values) for values in lists.values()), default=1)
        if count > MAX_USAGE_SUBJECTS:
            raise ValueError(
                f"a usage read names at most {MAX_USAGE_SUBJECTS} subjects, this one {count}"
            )
        for name, values in lists.items():
            if not values:
                raise ValueError(f"{name} names no subject")
            if 1 < len(values) != count:
                raise ValueError(
                    f"{name} names {len(values)} subjects and another id {count}: give one "
                    "value for all of them, or one per subject"
                )
        kind = _kind(
            subject,
            bool(lists.get("app_id")),
            lists.get("team", []),
            lists.get("category", []),
        )
        return await _unwrapped(
            get_limit_usage.asyncio(
                client=self._backend.client(), subject=kind, **lists
            ),
            f"reading the spend of {kind}",
        )


def _target(
    subject: Subject,
    app_id: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
    team: Optional[str],
    category: Optional[str],
) -> tuple[LimitSubject, dict[str, str]]:
    """The subject a write is about, as the router names it, and the ids it names."""
    kind = _kind(
        subject,
        bool(app_id),
        [team] if team is not None else [],
        [category] if category is not None else [],
    )
    ids = {
        name: value
        for name, value in {
            "app_id": app_id,
            "api_key": api_key,
            "user_id": user_id,
            "team": team,
            "category": category,
        }.items()
        if value
    }
    return kind, ids


def _kind(
    subject: Subject, in_app: bool, teams: Sequence[str], categories: Sequence[str]
) -> LimitSubject:
    """The subject as the router names it: a team or category within an app is app_team or
    app_category. Refuses what the router would, before anything is sent."""
    for category in categories:
        if category[: len("team:")].lower() == "team:":
            raise ValueError(
                f"category {category!r} names a team: use subject='team' with team="
            )
        check_category(category)
    for team in teams:
        check_team(team)
    if subject in ("team", "app_team"):
        if not teams:
            raise ValueError(f"a {subject} subject names its team in team=")
        if subject == "team" and in_app:
            subject = "app_team"
    elif teams:
        raise ValueError(f"team= is for subject='team', not {subject!r}")
    if subject in ("category", "app_category"):
        if not categories:
            raise ValueError(f"a {subject} subject names its category in category=")
        if subject == "category" and in_app:
            subject = "app_category"
    elif categories:
        raise ValueError(f"category= is for subject='category', not {subject!r}")
    if subject in ("app_team", "app_category") and not in_app:
        raise ValueError(f"an {subject} subject names its app in app_id=")
    return LimitSubject(subject)


def _filters(
    subject: Optional[Subject],
    app_id: Optional[str],
    api_key: Optional[str],
    user_id: Optional[str],
    team: Optional[str],
    category: Optional[str],
) -> dict[str, Any]:
    """A list's filters, with the subject as the router names it."""
    if subject is not None:
        kind, filters = _target(subject, app_id, api_key, user_id, team, category)
        return {"subject": kind, **filters}
    if team is not None:
        check_team(team)
    if category is not None:
        _kind("category", False, [], [category])
    return {
        name: value
        for name, value in {
            "app_id": app_id,
            "api_key": api_key,
            "user_id": user_id,
            "team": team,
            "category": category,
        }.items()
        if value
    }


_Body = TypeVar("_Body", LimitInput, LimitGrantInput, LimitResetInput)


def _with_ids(body: _Body, ids: dict[str, str]) -> _Body:
    """The body naming the ids, each of which all three bodies have."""
    for name, value in ids.items():
        setattr(body, name, value)
    return body


def _listed(value: Union[str, Sequence[str]]) -> List[str]:
    return [value] if isinstance(value, str) else list(value)


def _named(kind: LimitSubject, ids: dict[str, str]) -> str:
    """A subject as a message names it, such as "team finance" or "user user-42"."""
    named = ", ".join(f"{name} {value}" for name, value in ids.items())
    return f"{kind} ({named})" if named else str(kind)


def _utc(when: datetime.datetime, name: str) -> datetime.datetime:
    """An aware time in UTC. A naive one is refused: which day it means depends on where the
    process runs."""
    if when.tzinfo is None or when.utcoffset() is None:
        raise ValueError(f"{name} needs a timezone, such as datetime.timezone.utc")
    return when.astimezone(datetime.timezone.utc)
