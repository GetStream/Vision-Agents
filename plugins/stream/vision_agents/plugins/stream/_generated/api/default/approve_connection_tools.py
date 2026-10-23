from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.approved_connection_tools import ApprovedConnectionTools
from ...models.connection_tools_approval import ConnectionToolsApproval
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: ConnectionToolsApproval,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connections/{id}/tools/approve".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ApprovedConnectionTools | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ApprovedConnectionTools.from_dict(response.json())

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

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ApprovedConnectionTools | ErrorResponse]:
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
    body: ConnectionToolsApproval,
) -> Response[ApprovedConnectionTools | ErrorResponse]:
    """Approve a connection's tools again

     Pins tools of the connection at the schema the provider lists now, for the connection's current
    grant. A session pins a tool granted by name or by pattern the first time it offers it, and leaves
    it out while the provider lists it at another schema, which validate reports as needs_reapproval
    with the changed_tools. This is how a bearer or api_key connection gets them back; an OAuth one may
    reconnect instead, which begins a new grant.

    Each tool named must be one GET .../tools lists, which a validate refreshes, at the schema_digest
    sent: a tool it does not list is a 400, and one it lists at another digest a 409. The pins move
    together, and only while each is still what it was when read and the grant has not changed since;
    otherwise a 409. A refused approve writes nothing. The next session offers the tools, and the next
    validate answers connected when no other tool changed; last_validation keeps what the last validate
    found until then. One audit row, tool_reapproved, names the tools and their digests. A schema_digest
    an agent config grants is not a pin: change the config instead. Who may approve them is who may read
    the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionToolsApproval): The tools of a connection to approve at the schema the
            provider lists now. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ApprovedConnectionTools | ErrorResponse]
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
    body: ConnectionToolsApproval,
) -> ApprovedConnectionTools | ErrorResponse | None:
    """Approve a connection's tools again

     Pins tools of the connection at the schema the provider lists now, for the connection's current
    grant. A session pins a tool granted by name or by pattern the first time it offers it, and leaves
    it out while the provider lists it at another schema, which validate reports as needs_reapproval
    with the changed_tools. This is how a bearer or api_key connection gets them back; an OAuth one may
    reconnect instead, which begins a new grant.

    Each tool named must be one GET .../tools lists, which a validate refreshes, at the schema_digest
    sent: a tool it does not list is a 400, and one it lists at another digest a 409. The pins move
    together, and only while each is still what it was when read and the grant has not changed since;
    otherwise a 409. A refused approve writes nothing. The next session offers the tools, and the next
    validate answers connected when no other tool changed; last_validation keeps what the last validate
    found until then. One audit row, tool_reapproved, names the tools and their digests. A schema_digest
    an agent config grants is not a pin: change the config instead. Who may approve them is who may read
    the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionToolsApproval): The tools of a connection to approve at the schema the
            provider lists now. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ApprovedConnectionTools | ErrorResponse
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
    body: ConnectionToolsApproval,
) -> Response[ApprovedConnectionTools | ErrorResponse]:
    """Approve a connection's tools again

     Pins tools of the connection at the schema the provider lists now, for the connection's current
    grant. A session pins a tool granted by name or by pattern the first time it offers it, and leaves
    it out while the provider lists it at another schema, which validate reports as needs_reapproval
    with the changed_tools. This is how a bearer or api_key connection gets them back; an OAuth one may
    reconnect instead, which begins a new grant.

    Each tool named must be one GET .../tools lists, which a validate refreshes, at the schema_digest
    sent: a tool it does not list is a 400, and one it lists at another digest a 409. The pins move
    together, and only while each is still what it was when read and the grant has not changed since;
    otherwise a 409. A refused approve writes nothing. The next session offers the tools, and the next
    validate answers connected when no other tool changed; last_validation keeps what the last validate
    found until then. One audit row, tool_reapproved, names the tools and their digests. A schema_digest
    an agent config grants is not a pin: change the config instead. Who may approve them is who may read
    the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionToolsApproval): The tools of a connection to approve at the schema the
            provider lists now. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ApprovedConnectionTools | ErrorResponse]
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
    body: ConnectionToolsApproval,
) -> ApprovedConnectionTools | ErrorResponse | None:
    """Approve a connection's tools again

     Pins tools of the connection at the schema the provider lists now, for the connection's current
    grant. A session pins a tool granted by name or by pattern the first time it offers it, and leaves
    it out while the provider lists it at another schema, which validate reports as needs_reapproval
    with the changed_tools. This is how a bearer or api_key connection gets them back; an OAuth one may
    reconnect instead, which begins a new grant.

    Each tool named must be one GET .../tools lists, which a validate refreshes, at the schema_digest
    sent: a tool it does not list is a 400, and one it lists at another digest a 409. The pins move
    together, and only while each is still what it was when read and the grant has not changed since;
    otherwise a 409. A refused approve writes nothing. The next session offers the tools, and the next
    validate answers connected when no other tool changed; last_validation keeps what the last validate
    found until then. One audit row, tool_reapproved, names the tools and their digests. A schema_digest
    an agent config grants is not a pin: change the config instead. Who may approve them is who may read
    the connection.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connection, as returned when it was created.
        body (ConnectionToolsApproval): The tools of a connection to approve at the schema the
            provider lists now. An unknown field is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ApprovedConnectionTools | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
