from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_grant_page import LimitGrantPage
from ...models.limit_grant_state import LimitGrantState
from ...models.limit_subject import LimitSubject
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    subject: LimitSubject | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    state: LimitGrantState | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_subject: str | Unset = UNSET
    if not isinstance(subject, Unset):
        json_subject = subject.value

    params["subject"] = json_subject

    params["app_id"] = app_id

    params["api_key"] = api_key

    params["user_id"] = user_id

    params["team"] = team

    params["category"] = category

    json_state: str | Unset = UNSET
    if not isinstance(state, Unset):
        json_state = state.value

    params["state"] = json_state

    params["limit"] = limit

    params["cursor"] = cursor

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/spend/limits/grants",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | LimitGrantPage | None:
    if response.status_code == 200:
        response_200 = LimitGrantPage.from_dict(response.json())

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

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | LimitGrantPage]:
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
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    state: LimitGrantState | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | LimitGrantPage]:
    """List the organization's grants

     Newest first, revoked, expired and used ones included, each with what it paid for and what it has
    left.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        state (LimitGrantState | Unset): Where a grant stands: live (it is drawn once a limit of
            its subject is spent), used (it paid for its amount), expired (past expires_at) or
            revoked.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrantPage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        state=state,
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
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    state: LimitGrantState | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | LimitGrantPage | None:
    """List the organization's grants

     Newest first, revoked, expired and used ones included, each with what it paid for and what it has
    left.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        state (LimitGrantState | Unset): Where a grant stands: live (it is drawn once a limit of
            its subject is spent), used (it paid for its amount), expired (past expires_at) or
            revoked.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrantPage
    """

    return sync_detailed(
        client=client,
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        state=state,
        limit=limit,
        cursor=cursor,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    state: LimitGrantState | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> Response[ErrorResponse | LimitGrantPage]:
    """List the organization's grants

     Newest first, revoked, expired and used ones included, each with what it paid for and what it has
    left.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        state (LimitGrantState | Unset): Where a grant stands: live (it is drawn once a limit of
            its subject is spent), used (it paid for its amount), expired (past expires_at) or
            revoked.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | LimitGrantPage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
        state=state,
        limit=limit,
        cursor=cursor,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject | Unset = UNSET,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
    state: LimitGrantState | Unset = UNSET,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
) -> ErrorResponse | LimitGrantPage | None:
    """List the organization's grants

     Newest first, revoked, expired and used ones included, each with what it paid for and what it has
    left.

    Server-side only: one server-side credential of any app grants for any subject of its organization.

    Args:
        subject (LimitSubject | Unset): Whose spend a limit counts: the organization, one app, a
            key, an end user of an app, a usage category across the organization or within one app, or
            a team across the organization or within one app. A team is the user's team, as its token
            names it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):
        state (LimitGrantState | Unset): Where a grant stands: live (it is drawn once a limit of
            its subject is spent), used (it paid for its amount), expired (past expires_at) or
            revoked.
        limit (int | Unset): Up to 200. Omitted is 50.
        cursor (str | Unset): The next_cursor of the previous page, sent with the same filters.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | LimitGrantPage
    """

    return (
        await asyncio_detailed(
            client=client,
            subject=subject,
            app_id=app_id,
            api_key=api_key,
            user_id=user_id,
            team=team,
            category=category,
            state=state,
            limit=limit,
            cursor=cursor,
        )
    ).parsed
