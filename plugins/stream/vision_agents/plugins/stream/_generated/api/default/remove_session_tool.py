from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    name: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/sessions/{id}/tools/{name}".format(
            id=quote(str(id), safe=""),
            name=quote(str(name), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 204:
        response_204 = cast(Any, None)
        return response_204

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
) -> Response[Any | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Withdraw one of the caller's tools from a running session

     The model is no longer offered the tool from the next turn. The change waits for the turn in
    progress and every tool call in flight to finish, as adding does, so once it is answered no call of
    the tool is in flight and none can be made: a handler for it can go. Do not wait for this from
    inside a call of one of the session's tools, which it waits for. Only the caller's own tools are
    withdrawn, whether declared when the session was created or added since, and by the kind of caller
    that opened the session, as for adding: anyone else is refused with a 403. A name the caller has no
    tool by, and a session that is not running, are a 404.

    Args:
        id (str): The session, as returned when it was created.
        name (str): The tool's name, as the caller declared it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        name=name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Withdraw one of the caller's tools from a running session

     The model is no longer offered the tool from the next turn. The change waits for the turn in
    progress and every tool call in flight to finish, as adding does, so once it is answered no call of
    the tool is in flight and none can be made: a handler for it can go. Do not wait for this from
    inside a call of one of the session's tools, which it waits for. Only the caller's own tools are
    withdrawn, whether declared when the session was created or added since, and by the kind of caller
    that opened the session, as for adding: anyone else is refused with a 403. A name the caller has no
    tool by, and a session that is not running, are a 404.

    Args:
        id (str): The session, as returned when it was created.
        name (str): The tool's name, as the caller declared it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        id=id,
        name=name,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Withdraw one of the caller's tools from a running session

     The model is no longer offered the tool from the next turn. The change waits for the turn in
    progress and every tool call in flight to finish, as adding does, so once it is answered no call of
    the tool is in flight and none can be made: a handler for it can go. Do not wait for this from
    inside a call of one of the session's tools, which it waits for. Only the caller's own tools are
    withdrawn, whether declared when the session was created or added since, and by the kind of caller
    that opened the session, as for adding: anyone else is refused with a 403. A name the caller has no
    tool by, and a session that is not running, are a 404.

    Args:
        id (str): The session, as returned when it was created.
        name (str): The tool's name, as the caller declared it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        name=name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    name: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Withdraw one of the caller's tools from a running session

     The model is no longer offered the tool from the next turn. The change waits for the turn in
    progress and every tool call in flight to finish, as adding does, so once it is answered no call of
    the tool is in flight and none can be made: a handler for it can go. Do not wait for this from
    inside a call of one of the session's tools, which it waits for. Only the caller's own tools are
    withdrawn, whether declared when the session was created or added since, and by the kind of caller
    that opened the session, as for adding: anyone else is refused with a 403. A name the caller has no
    tool by, and a session that is not running, are a 404.

    Args:
        id (str): The session, as returned when it was created.
        name (str): The tool's name, as the caller declared it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            name=name,
            client=client,
        )
    ).parsed
