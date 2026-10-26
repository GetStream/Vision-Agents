from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.scheduled_task import ScheduledTask
from ...types import Response


def _get_kwargs(
    id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/scheduled-tasks/{id}/resume".format(
            id=quote(str(id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ScheduledTask | None:
    if response.status_code == 200:
        response_200 = ScheduledTask.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

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
) -> Response[ErrorResponse | ScheduledTask]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse | ScheduledTask]:
    """Resume a scheduled task

     The next occurrence is counted from now, so missed occurrences are never replayed, and a run of
    failures is forgiven. A one-shot whose time came while it was paused is completed, its occurrence
    listed in its runs as missed (error task_paused). Any other task with no occurrence left, such as a
    failed one-shot or a repeat past its until, is refused (400 schedule_in_past) and left as it was:
    give it a new schedule, then resume it. A completed task cannot be resumed (409).

    Args:
        id (str): The task, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | ScheduledTask | None:
    """Resume a scheduled task

     The next occurrence is counted from now, so missed occurrences are never replayed, and a run of
    failures is forgiven. A one-shot whose time came while it was paused is completed, its occurrence
    listed in its runs as missed (error task_paused). Any other task with no occurrence left, such as a
    failed one-shot or a repeat past its until, is refused (400 schedule_in_past) and left as it was:
    give it a new schedule, then resume it. A completed task cannot be resumed (409).

    Args:
        id (str): The task, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTask
    """

    return sync_detailed(
        id=id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ErrorResponse | ScheduledTask]:
    """Resume a scheduled task

     The next occurrence is counted from now, so missed occurrences are never replayed, and a run of
    failures is forgiven. A one-shot whose time came while it was paused is completed, its occurrence
    listed in its runs as missed (error task_paused). Any other task with no occurrence left, such as a
    failed one-shot or a repeat past its until, is refused (400 schedule_in_past) and left as it was:
    give it a new schedule, then resume it. A completed task cannot be resumed (409).

    Args:
        id (str): The task, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        id=id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ErrorResponse | ScheduledTask | None:
    """Resume a scheduled task

     The next occurrence is counted from now, so missed occurrences are never replayed, and a run of
    failures is forgiven. A one-shot whose time came while it was paused is completed, its occurrence
    listed in its runs as missed (error task_paused). Any other task with no occurrence left, such as a
    failed one-shot or a repeat past its until, is refused (400 schedule_in_past) and left as it was:
    give it a new schedule, then resume it. A completed task cannot be resumed (409).

    Args:
        id (str): The task, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTask
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
        )
    ).parsed
