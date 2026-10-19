from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.scheduled_task import ScheduledTask
from ...models.update_scheduled_task_request import UpdateScheduledTaskRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: UpdateScheduledTaskRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "patch",
        "url": "/v1/agents/scheduled-tasks/{id}".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
    body: UpdateScheduledTaskRequest,
) -> Response[ErrorResponse | ScheduledTask]:
    """Change a scheduled task

     A field left out is kept. A schedule given replaces the whole schedule, except a timezone left out
    of it keeps the task's. next moves the next occurrence only — a snooze, or a skip when it lands past
    the following one — and on a one-shot it is the reschedule. An end user's device may change title,
    schedule and next; instructions are server-side, and a device sending them is refused with a 403.

    Args:
        id (str): The task, as returned when it was created.
        body (UpdateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: UpdateScheduledTaskRequest,
) -> ErrorResponse | ScheduledTask | None:
    """Change a scheduled task

     A field left out is kept. A schedule given replaces the whole schedule, except a timezone left out
    of it keeps the task's. next moves the next occurrence only — a snooze, or a skip when it lands past
    the following one — and on a one-shot it is the reschedule. An end user's device may change title,
    schedule and next; instructions are server-side, and a device sending them is refused with a 403.

    Args:
        id (str): The task, as returned when it was created.
        body (UpdateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTask
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: UpdateScheduledTaskRequest,
) -> Response[ErrorResponse | ScheduledTask]:
    """Change a scheduled task

     A field left out is kept. A schedule given replaces the whole schedule, except a timezone left out
    of it keeps the task's. next moves the next occurrence only — a snooze, or a skip when it lands past
    the following one — and on a one-shot it is the reschedule. An end user's device may change title,
    schedule and next; instructions are server-side, and a device sending them is refused with a 403.

    Args:
        id (str): The task, as returned when it was created.
        body (UpdateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: UpdateScheduledTaskRequest,
) -> ErrorResponse | ScheduledTask | None:
    """Change a scheduled task

     A field left out is kept. A schedule given replaces the whole schedule, except a timezone left out
    of it keeps the task's. next moves the next occurrence only — a snooze, or a skip when it lands past
    the following one — and on a one-shot it is the reschedule. An end user's device may change title,
    schedule and next; instructions are server-side, and a device sending them is refused with a 403.

    Args:
        id (str): The task, as returned when it was created.
        body (UpdateScheduledTaskRequest):

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
            body=body,
        )
    ).parsed
