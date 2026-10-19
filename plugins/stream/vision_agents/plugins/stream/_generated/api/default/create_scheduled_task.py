from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_scheduled_task_request import CreateScheduledTaskRequest
from ...models.error_response import ErrorResponse
from ...models.scheduled_task import ScheduledTask
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: CreateScheduledTaskRequest,
    x_stream_user_id: str | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}
    if not isinstance(x_stream_user_id, Unset):
        headers["X-Stream-User-Id"] = x_stream_user_id

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/scheduled-tasks",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ScheduledTask | None:
    if response.status_code == 201:
        response_201 = ScheduledTask.from_dict(response.json())

        return response_201

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 501:
        response_501 = ErrorResponse.from_dict(response.json())

        return response_501

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
    *,
    client: AuthenticatedClient | Client,
    body: CreateScheduledTaskRequest,
    x_stream_user_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ScheduledTask]:
    """Schedule a task for a user

     Server-side only. The task belongs to the user named with X-Stream-User-Id, and without one there is
    nobody for it to belong to, so the create is refused. The agent is named by agent or config_id, or
    taken from session_id's own config, and the follow-up lands in that session's conversation; without
    a session, the first occurrence opens one. An id the caller chooses may be sent again only to be
    told it is taken (409), the same answer session create gives. scheduling.enabled on the config is
    not required: that flag only offers the tools to the model. deliver_to says where an occurrence
    goes, in order, and quiet_hours holds one that comes due in them until they end.

    Args:
        x_stream_user_id (str | Unset): The user the task is for, named by the backend acting for
            them. Required: without a named user the create is refused.
        body (CreateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        body=body,
        x_stream_user_id=x_stream_user_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: CreateScheduledTaskRequest,
    x_stream_user_id: str | Unset = UNSET,
) -> ErrorResponse | ScheduledTask | None:
    """Schedule a task for a user

     Server-side only. The task belongs to the user named with X-Stream-User-Id, and without one there is
    nobody for it to belong to, so the create is refused. The agent is named by agent or config_id, or
    taken from session_id's own config, and the follow-up lands in that session's conversation; without
    a session, the first occurrence opens one. An id the caller chooses may be sent again only to be
    told it is taken (409), the same answer session create gives. scheduling.enabled on the config is
    not required: that flag only offers the tools to the model. deliver_to says where an occurrence
    goes, in order, and quiet_hours holds one that comes due in them until they end.

    Args:
        x_stream_user_id (str | Unset): The user the task is for, named by the backend acting for
            them. Required: without a named user the create is refused.
        body (CreateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTask
    """

    return sync_detailed(
        client=client,
        body=body,
        x_stream_user_id=x_stream_user_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateScheduledTaskRequest,
    x_stream_user_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ScheduledTask]:
    """Schedule a task for a user

     Server-side only. The task belongs to the user named with X-Stream-User-Id, and without one there is
    nobody for it to belong to, so the create is refused. The agent is named by agent or config_id, or
    taken from session_id's own config, and the follow-up lands in that session's conversation; without
    a session, the first occurrence opens one. An id the caller chooses may be sent again only to be
    told it is taken (409), the same answer session create gives. scheduling.enabled on the config is
    not required: that flag only offers the tools to the model. deliver_to says where an occurrence
    goes, in order, and quiet_hours holds one that comes due in them until they end.

    Args:
        x_stream_user_id (str | Unset): The user the task is for, named by the backend acting for
            them. Required: without a named user the create is refused.
        body (CreateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTask]
    """

    kwargs = _get_kwargs(
        body=body,
        x_stream_user_id=x_stream_user_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CreateScheduledTaskRequest,
    x_stream_user_id: str | Unset = UNSET,
) -> ErrorResponse | ScheduledTask | None:
    """Schedule a task for a user

     Server-side only. The task belongs to the user named with X-Stream-User-Id, and without one there is
    nobody for it to belong to, so the create is refused. The agent is named by agent or config_id, or
    taken from session_id's own config, and the follow-up lands in that session's conversation; without
    a session, the first occurrence opens one. An id the caller chooses may be sent again only to be
    told it is taken (409), the same answer session create gives. scheduling.enabled on the config is
    not required: that flag only offers the tools to the model. deliver_to says where an occurrence
    goes, in order, and quiet_hours holds one that comes due in them until they end.

    Args:
        x_stream_user_id (str | Unset): The user the task is for, named by the backend acting for
            them. Required: without a named user the create is refused.
        body (CreateScheduledTaskRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTask
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
            x_stream_user_id=x_stream_user_id,
        )
    ).parsed
