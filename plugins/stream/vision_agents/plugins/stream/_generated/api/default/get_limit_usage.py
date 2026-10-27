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
    app_id: list[str] | None | Unset = UNSET,
    api_key: list[str] | None | Unset = UNSET,
    user_id: list[str] | None | Unset = UNSET,
    team: list[str] | None | Unset = UNSET,
    category: list[str] | None | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_subject = subject.value
    params["subject"] = json_subject

    json_app_id: list[str] | None | Unset
    if isinstance(app_id, Unset):
        json_app_id = UNSET
    elif isinstance(app_id, list):
        json_app_id = app_id

    else:
        json_app_id = app_id
    params["app_id"] = json_app_id

    json_api_key: list[str] | None | Unset
    if isinstance(api_key, Unset):
        json_api_key = UNSET
    elif isinstance(api_key, list):
        json_api_key = api_key

    else:
        json_api_key = api_key
    params["api_key"] = json_api_key

    json_user_id: list[str] | None | Unset
    if isinstance(user_id, Unset):
        json_user_id = UNSET
    elif isinstance(user_id, list):
        json_user_id = user_id

    else:
        json_user_id = user_id
    params["user_id"] = json_user_id

    json_team: list[str] | None | Unset
    if isinstance(team, Unset):
        json_team = UNSET
    elif isinstance(team, list):
        json_team = team

    else:
        json_team = team
    params["team"] = json_team

    json_category: list[str] | None | Unset
    if isinstance(category, Unset):
        json_category = UNSET
    elif isinstance(category, list):
        json_category = category

    else:
        json_category = category
    params["category"] = json_category

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/spend/limits/usage",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | list[SubjectUsage] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = SubjectUsage.from_dict(response_200_item_data)

            response_200.append(response_200_item)

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
) -> Response[ErrorResponse | list[SubjectUsage]]:
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
    app_id: list[str] | None | Unset = UNSET,
    api_key: list[str] | None | Unset = UNSET,
    user_id: list[str] | None | Unset = UNSET,
    team: list[str] | None | Unset = UNSET,
    category: list[str] | None | Unset = UNSET,
) -> Response[ErrorResponse | list[SubjectUsage]]:
    """Read what subjects spent against their limits

     Each subject's current window in every interval: what it spent, the limit in force with its
    effective amount, its baseline since a reset and its room, and for the organization the operator's
    spend hard limit, which is what admission refuses first, the same on every subject. Up to 100
    subjects of one kind, answered in the order asked: repeat the subject's own id (app_id for an app,
    api_key, user_id, team or category), and send app_id once for subjects within one app or once per
    subject, so every list has one value or one per subject. One subject answers a list of one. Spend is
    read from the usage records, so it is what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (list[str] | None | Unset): The app of each subject, for every subject but
            organization, category and team. One goes with every subject.
        api_key (list[str] | None | Unset): The key of each subject, for subject api_key.
        user_id (list[str] | None | Unset): The end user of each subject, for subject user.
        team (list[str] | None | Unset): The team of each subject, for subject team or app_team.
        category (list[str] | None | Unset): The usage category of each subject, for subject
            category or app_category.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[SubjectUsage]]
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
    app_id: list[str] | None | Unset = UNSET,
    api_key: list[str] | None | Unset = UNSET,
    user_id: list[str] | None | Unset = UNSET,
    team: list[str] | None | Unset = UNSET,
    category: list[str] | None | Unset = UNSET,
) -> ErrorResponse | list[SubjectUsage] | None:
    """Read what subjects spent against their limits

     Each subject's current window in every interval: what it spent, the limit in force with its
    effective amount, its baseline since a reset and its room, and for the organization the operator's
    spend hard limit, which is what admission refuses first, the same on every subject. Up to 100
    subjects of one kind, answered in the order asked: repeat the subject's own id (app_id for an app,
    api_key, user_id, team or category), and send app_id once for subjects within one app or once per
    subject, so every list has one value or one per subject. One subject answers a list of one. Spend is
    read from the usage records, so it is what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (list[str] | None | Unset): The app of each subject, for every subject but
            organization, category and team. One goes with every subject.
        api_key (list[str] | None | Unset): The key of each subject, for subject api_key.
        user_id (list[str] | None | Unset): The end user of each subject, for subject user.
        team (list[str] | None | Unset): The team of each subject, for subject team or app_team.
        category (list[str] | None | Unset): The usage category of each subject, for subject
            category or app_category.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[SubjectUsage]
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
    app_id: list[str] | None | Unset = UNSET,
    api_key: list[str] | None | Unset = UNSET,
    user_id: list[str] | None | Unset = UNSET,
    team: list[str] | None | Unset = UNSET,
    category: list[str] | None | Unset = UNSET,
) -> Response[ErrorResponse | list[SubjectUsage]]:
    """Read what subjects spent against their limits

     Each subject's current window in every interval: what it spent, the limit in force with its
    effective amount, its baseline since a reset and its room, and for the organization the operator's
    spend hard limit, which is what admission refuses first, the same on every subject. Up to 100
    subjects of one kind, answered in the order asked: repeat the subject's own id (app_id for an app,
    api_key, user_id, team or category), and send app_id once for subjects within one app or once per
    subject, so every list has one value or one per subject. One subject answers a list of one. Spend is
    read from the usage records, so it is what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (list[str] | None | Unset): The app of each subject, for every subject but
            organization, category and team. One goes with every subject.
        api_key (list[str] | None | Unset): The key of each subject, for subject api_key.
        user_id (list[str] | None | Unset): The end user of each subject, for subject user.
        team (list[str] | None | Unset): The team of each subject, for subject team or app_team.
        category (list[str] | None | Unset): The usage category of each subject, for subject
            category or app_category.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | list[SubjectUsage]]
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
    app_id: list[str] | None | Unset = UNSET,
    api_key: list[str] | None | Unset = UNSET,
    user_id: list[str] | None | Unset = UNSET,
    team: list[str] | None | Unset = UNSET,
    category: list[str] | None | Unset = UNSET,
) -> ErrorResponse | list[SubjectUsage] | None:
    """Read what subjects spent against their limits

     Each subject's current window in every interval: what it spent, the limit in force with its
    effective amount, its baseline since a reset and its room, and for the organization the operator's
    spend hard limit, which is what admission refuses first, the same on every subject. Up to 100
    subjects of one kind, answered in the order asked: repeat the subject's own id (app_id for an app,
    api_key, user_id, team or category), and send app_id once for subjects within one app or once per
    subject, so every list has one value or one per subject. One subject answers a list of one. Spend is
    read from the usage records, so it is what admission counts within a few seconds.

    Server-side only: one server-side credential of any app changes every limit of its organization, and
    a request that is not server-side, reads included, answers 403 server_side_only.

    Args:
        subject (LimitSubject): Whose spend a limit counts: the organization, one app, a key, an
            end user of an app, a usage category across the organization or within one app, or a team
            across the organization or within one app. A team is the user's team, as its token names
            it.
        app_id (list[str] | None | Unset): The app of each subject, for every subject but
            organization, category and team. One goes with every subject.
        api_key (list[str] | None | Unset): The key of each subject, for subject api_key.
        user_id (list[str] | None | Unset): The end user of each subject, for subject user.
        team (list[str] | None | Unset): The team of each subject, for subject team or app_team.
        category (list[str] | None | Unset): The usage category of each subject, for subject
            category or app_category.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | list[SubjectUsage]
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
