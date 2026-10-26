from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.limit_subject import LimitSubject
from ...models.subject_usage import SubjectUsage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    subject: LimitSubject,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_subject = subject.value
    params["subject"] = json_subject

    params["app_id"] = app_id

    params["api_key"] = api_key

    params["user_id"] = user_id

    params["team"] = team

    params["category"] = category

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/spend/limits/usage",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | SubjectUsage | None:
    if response.status_code == 200:
        response_200 = SubjectUsage.from_dict(response.json())

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
) -> Response[ErrorResponse | SubjectUsage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
) -> Response[ErrorResponse | SubjectUsage]:
    """Read what a subject spent against its limits

     The subject's current window in every interval: what it spent, the limit in force with its effective
    amount, its baseline since a reset and its room, and once for the organization the operator's spend
    hard limit, which is what admission refuses first. Spend is read from the usage records, so it is
    what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SubjectUsage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
) -> ErrorResponse | SubjectUsage | None:
    """Read what a subject spent against its limits

     The subject's current window in every interval: what it spent, the limit in force with its effective
    amount, its baseline since a reset and its room, and once for the organization the operator's spend
    hard limit, which is what admission refuses first. Spend is read from the usage records, so it is
    what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SubjectUsage
    """

    return sync_detailed(
        client=client,
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
) -> Response[ErrorResponse | SubjectUsage]:
    """Read what a subject spent against its limits

     The subject's current window in every interval: what it spent, the limit in force with its effective
    amount, its baseline since a reset and its room, and once for the organization the operator's spend
    hard limit, which is what admission refuses first. Spend is read from the usage records, so it is
    what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SubjectUsage]
    """

    kwargs = _get_kwargs(
        subject=subject,
        app_id=app_id,
        api_key=api_key,
        user_id=user_id,
        team=team,
        category=category,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    subject: LimitSubject,
    app_id: str | Unset = UNSET,
    api_key: str | Unset = UNSET,
    user_id: str | Unset = UNSET,
    team: str | Unset = UNSET,
    category: str | Unset = UNSET,
) -> ErrorResponse | SubjectUsage | None:
    """Read what a subject spent against its limits

     The subject's current window in every interval: what it spent, the limit in force with its effective
    amount, its baseline since a reset and its room, and once for the organization the operator's spend
    hard limit, which is what admission refuses first. Spend is read from the usage records, so it is
    what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (str | Unset):
        api_key (str | Unset):
        user_id (str | Unset):
        team (str | Unset):
        category (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SubjectUsage
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
        )
    ).parsed
