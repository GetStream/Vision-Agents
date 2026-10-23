from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.list_scheduled_tasks_state import ListScheduledTasksState
from ...models.scheduled_task_page import ScheduledTaskPage
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    state: ListScheduledTasksState | Unset = UNSET,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["limit"] = limit

    params["cursor"] = cursor

    json_state: str | Unset = UNSET
    if not isinstance(state, Unset):
        json_state = state.value

    params["state"] = json_state

    params["config_id"] = config_id

    params["session_id"] = session_id

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/scheduled-tasks",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ScheduledTaskPage | None:
    if response.status_code == 200:
        response_200 = ScheduledTaskPage.from_dict(response.json())

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
) -> Response[ErrorResponse | ScheduledTaskPage]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    state: ListScheduledTasksState | Unset = UNSET,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ScheduledTaskPage]:
    """The scheduled tasks of the user, or of the customer

     Newest first, cursor-paged. A device lists its own user's tasks, including the ones its backend made
    in its name. A backend naming a user with X-Stream-User-Id lists that user's; a backend naming
    nobody lists every task of the customer, instructions included: a customer's whole task list is
    support-visible, so put nothing in instructions a user must not have read back to them. Devices see
    instructions too, for the same reason.

    Args:
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.
        state (ListScheduledTasksState | Unset): Only tasks in this state.
        config_id (str | Unset): Only tasks on this agent config.
        session_id (str | Unset): Only tasks following up in this session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTaskPage]
    """

    kwargs = _get_kwargs(
        limit=limit,
        cursor=cursor,
        state=state,
        config_id=config_id,
        session_id=session_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    state: ListScheduledTasksState | Unset = UNSET,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
) -> ErrorResponse | ScheduledTaskPage | None:
    """The scheduled tasks of the user, or of the customer

     Newest first, cursor-paged. A device lists its own user's tasks, including the ones its backend made
    in its name. A backend naming a user with X-Stream-User-Id lists that user's; a backend naming
    nobody lists every task of the customer, instructions included: a customer's whole task list is
    support-visible, so put nothing in instructions a user must not have read back to them. Devices see
    instructions too, for the same reason.

    Args:
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.
        state (ListScheduledTasksState | Unset): Only tasks in this state.
        config_id (str | Unset): Only tasks on this agent config.
        session_id (str | Unset): Only tasks following up in this session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTaskPage
    """

    return sync_detailed(
        client=client,
        limit=limit,
        cursor=cursor,
        state=state,
        config_id=config_id,
        session_id=session_id,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    state: ListScheduledTasksState | Unset = UNSET,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
) -> Response[ErrorResponse | ScheduledTaskPage]:
    """The scheduled tasks of the user, or of the customer

     Newest first, cursor-paged. A device lists its own user's tasks, including the ones its backend made
    in its name. A backend naming a user with X-Stream-User-Id lists that user's; a backend naming
    nobody lists every task of the customer, instructions included: a customer's whole task list is
    support-visible, so put nothing in instructions a user must not have read back to them. Devices see
    instructions too, for the same reason.

    Args:
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.
        state (ListScheduledTasksState | Unset): Only tasks in this state.
        config_id (str | Unset): Only tasks on this agent config.
        session_id (str | Unset): Only tasks following up in this session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ScheduledTaskPage]
    """

    kwargs = _get_kwargs(
        limit=limit,
        cursor=cursor,
        state=state,
        config_id=config_id,
        session_id=session_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    limit: int | Unset = UNSET,
    cursor: str | Unset = UNSET,
    state: ListScheduledTasksState | Unset = UNSET,
    config_id: str | Unset = UNSET,
    session_id: str | Unset = UNSET,
) -> ErrorResponse | ScheduledTaskPage | None:
    """The scheduled tasks of the user, or of the customer

     Newest first, cursor-paged. A device lists its own user's tasks, including the ones its backend made
    in its name. A backend naming a user with X-Stream-User-Id lists that user's; a backend naming
    nobody lists every task of the customer, instructions included: a customer's whole task list is
    support-visible, so put nothing in instructions a user must not have read back to them. Devices see
    instructions too, for the same reason.

    Args:
        limit (int | Unset): Up to 200. Omitted is 25.
        cursor (str | Unset): The next_cursor of the previous page. Omitted is the first page.
        state (ListScheduledTasksState | Unset): Only tasks in this state.
        config_id (str | Unset): Only tasks on this agent config.
        session_id (str | Unset): Only tasks following up in this session.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ScheduledTaskPage
    """

    return (
        await asyncio_detailed(
            client=client,
            limit=limit,
            cursor=cursor,
            state=state,
            config_id=config_id,
            session_id=session_id,
        )
    ).parsed
