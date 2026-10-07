from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.audit_page import AuditPage
from ...models.audit_query import AuditQuery
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    body: AuditQuery | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/audit/query",
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AuditPage | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AuditPage.from_dict(response.json())

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

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AuditPage | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AuditQuery | Unset = UNSET,
) -> Response[AuditPage | ErrorResponse]:
    """List the changes made to the app's configuration

     Every change somebody made to the app's configuration, newest first: the agents, their skills, the
    knowledge they read, the routers, the plugin logins and the policies. Each entry names what changed,
    who changed it, which client they used, and the before and after of every field that moved.

    What an agent does while it runs is not here: a session, a call and a simulation run are traffic
    rather than configuration, and are read from their own endpoints. `resource_type`, `resource_id`,
    `agent_id`, `source` and `action` narrow the list; a deleted resource's entries stay, and its
    deletion is one of them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AuditQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AuditPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: AuditQuery | Unset = UNSET,
) -> AuditPage | ErrorResponse | None:
    """List the changes made to the app's configuration

     Every change somebody made to the app's configuration, newest first: the agents, their skills, the
    knowledge they read, the routers, the plugin logins and the policies. Each entry names what changed,
    who changed it, which client they used, and the before and after of every field that moved.

    What an agent does while it runs is not here: a session, a call and a simulation run are traffic
    rather than configuration, and are read from their own endpoints. `resource_type`, `resource_id`,
    `agent_id`, `source` and `action` narrow the list; a deleted resource's entries stay, and its
    deletion is one of them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AuditQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AuditPage | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: AuditQuery | Unset = UNSET,
) -> Response[AuditPage | ErrorResponse]:
    """List the changes made to the app's configuration

     Every change somebody made to the app's configuration, newest first: the agents, their skills, the
    knowledge they read, the routers, the plugin logins and the policies. Each entry names what changed,
    who changed it, which client they used, and the before and after of every field that moved.

    What an agent does while it runs is not here: a session, a call and a simulation run are traffic
    rather than configuration, and are read from their own endpoints. `resource_type`, `resource_id`,
    `agent_id`, `source` and `action` narrow the list; a deleted resource's entries stay, and its
    deletion is one of them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AuditQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AuditPage | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: AuditQuery | Unset = UNSET,
) -> AuditPage | ErrorResponse | None:
    """List the changes made to the app's configuration

     Every change somebody made to the app's configuration, newest first: the agents, their skills, the
    knowledge they read, the routers, the plugin logins and the policies. Each entry names what changed,
    who changed it, which client they used, and the before and after of every field that moved.

    What an agent does while it runs is not here: a session, a call and a simulation run are traffic
    rather than configuration, and are read from their own endpoints. `resource_type`, `resource_id`,
    `agent_id`, `source` and `action` narrow the list; a deleted resource's entries stay, and its
    deletion is one of them.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (AuditQuery | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AuditPage | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
