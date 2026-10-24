from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_interval import LimitInterval
from ...models.limit_level import LimitLevel
from ...models.limit_page import LimitPage
from ...models.limit_subject import LimitSubject
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    subject: LimitSubject | Unset = UNSET,
    level: LimitLevel | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    interval: LimitInterval | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_subject: str | Unset = UNSET
    if not isinstance(subject, Unset):
        json_subject = subject.value

    params["subject"] = json_subject

    json_level: str | Unset = UNSET
    if not isinstance(level, Unset):
        json_level = level.value

    params["level"] = json_level

    params["app_id"] = app_id

    params["api_key"] = api_key

    params["user_id"] = user_id

    params["team"] = team

    params["category"] = category

    json_interval: str | Unset = UNSET
    if not isinstance(interval, Unset):
        json_interval = interval.value

    params["interval"] = json_interval

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/spend/limits",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | LimitPage | None:
    if response.status_code == 200:
        response_200 = LimitPage.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 402:
        response_402 = ErrorResponse.from_dict(response.json())

        return response_402

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | LimitPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    level: LimitLevel | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    interval: LimitInterval | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | LimitPage]:
    """List the organization's spend limits

     Every limit of the caller's organization, by id, with what each can spend once its parents are
    counted.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        level (LimitLevel | Unset): Whose ids a limit names. A key's or a user's limit at
            organization or app level is a default for every key or user there, each counted on its
            own; at api_key or user level it names one. The most specific level wins. Every other
            subject has one level: the organization, a category and a team are organization level; an
            app, and a category or team within one, app level.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        interval (LimitInterval | Unset): The window a limit counts spend over, in UTC: the day,
            the week from Monday, 14 days counted from Monday 2024-01-01, or the calendar month.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitPage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        level=level,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        interval=interval,
        limit=limit,
        cursor=cursor,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    level: LimitLevel | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    interval: LimitInterval | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | LimitPage | None:
    """List the organization's spend limits

     Every limit of the caller's organization, by id, with what each can spend once its parents are
    counted.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        level (LimitLevel | Unset): Whose ids a limit names. A key's or a user's limit at
            organization or app level is a default for every key or user there, each counted on its
            own; at api_key or user level it names one. The most specific level wins. Every other
            subject has one level: the organization, a category and a team are organization level; an
            app, and a category or team within one, app level.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        interval (LimitInterval | Unset): The window a limit counts spend over, in UTC: the day,
            the week from Monday, 14 days counted from Monday 2024-01-01, or the calendar month.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitPage
    """

    return sync_detailed(
        client=client,
        subject=subject,
        level=level,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        interval=interval,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    level: LimitLevel | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    interval: LimitInterval | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | LimitPage]:
    """List the organization's spend limits

     Every limit of the caller's organization, by id, with what each can spend once its parents are
    counted.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        level (LimitLevel | Unset): Whose ids a limit names. A key's or a user's limit at
            organization or app level is a default for every key or user there, each counted on its
            own; at api_key or user level it names one. The most specific level wins. Every other
            subject has one level: the organization, a category and a team are organization level; an
            app, and a category or team within one, app level.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        interval (LimitInterval | Unset): The window a limit counts spend over, in UTC: the day,
            the week from Monday, 14 days counted from Monday 2024-01-01, or the calendar month.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitPage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        level=level,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        interval=interval,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    level: LimitLevel | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    interval: LimitInterval | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | LimitPage | None:
    """List the organization's spend limits

     Every limit of the caller's organization, by id, with what each can spend once its parents are
    counted.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        level (LimitLevel | Unset): Whose ids a limit names. A key's or a user's limit at
            organization or app level is a default for every key or user there, each counted on its
            own; at api_key or user level it names one. The most specific level wins. Every other
            subject has one level: the organization, a category and a team are organization level; an
            app, and a category or team within one, app level.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        interval (LimitInterval | Unset): The window a limit counts spend over, in UTC: the day,
            the week from Monday, 14 days counted from Monday 2024-01-01, or the calendar month.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitPage
    """

    return (
        await asyncio_detailed(
            client=client,
            subject=subject,
            level=level,
            app_id=app_id,
            api_key=api_key,
            user_id=user_id,
            team=team,
            category=category,
            interval=interval,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
