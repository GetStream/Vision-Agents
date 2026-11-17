from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.add_session_tools_request import AddSessionToolsRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: AddSessionToolsRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/tools".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
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
) -> Response[Any | ErrorResponse]:
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
    body: AddSessionToolsRequest,
) -> Response[Any | ErrorResponse]:
    """Give a running session more of the caller's tools

     Declares tools on a running session as though they had been declared when it was created, for tools
    that exist only partway through a conversation, such as the ones a card shown in it brings. Whoever
    runs the caller's tools is asked to run them over the events socket, and the model is offered them
    from its next request: a reply being written finishes with the tools it started with. A tool named
    as one the caller already has replaces it.

    The kind of caller that opened the session declares its tools: a device those of the session it
    opened, and a backend those of a session a backend opened. Anyone else is refused with a 403. A name
    the session offers that is not the caller's (the router's own, a connector's, a worker's or another
    agent's) is a 409. A session that is not running is a 404. A speech-to-speech model that takes its
    tools only when it opens, such as Gemini Live, cannot be given more during the call: 400.

    Args:
        id (str): The session, as returned when it was created.
        body (AddSessionToolsRequest): Tools to give a running session's caller.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
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
    body: AddSessionToolsRequest,
) -> Any | ErrorResponse | None:
    """Give a running session more of the caller's tools

     Declares tools on a running session as though they had been declared when it was created, for tools
    that exist only partway through a conversation, such as the ones a card shown in it brings. Whoever
    runs the caller's tools is asked to run them over the events socket, and the model is offered them
    from its next request: a reply being written finishes with the tools it started with. A tool named
    as one the caller already has replaces it.

    The kind of caller that opened the session declares its tools: a device those of the session it
    opened, and a backend those of a session a backend opened. Anyone else is refused with a 403. A name
    the session offers that is not the caller's (the router's own, a connector's, a worker's or another
    agent's) is a 409. A session that is not running is a 404. A speech-to-speech model that takes its
    tools only when it opens, such as Gemini Live, cannot be given more during the call: 400.

    Args:
        id (str): The session, as returned when it was created.
        body (AddSessionToolsRequest): Tools to give a running session's caller.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
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
    body: AddSessionToolsRequest,
) -> Response[Any | ErrorResponse]:
    """Give a running session more of the caller's tools

     Declares tools on a running session as though they had been declared when it was created, for tools
    that exist only partway through a conversation, such as the ones a card shown in it brings. Whoever
    runs the caller's tools is asked to run them over the events socket, and the model is offered them
    from its next request: a reply being written finishes with the tools it started with. A tool named
    as one the caller already has replaces it.

    The kind of caller that opened the session declares its tools: a device those of the session it
    opened, and a backend those of a session a backend opened. Anyone else is refused with a 403. A name
    the session offers that is not the caller's (the router's own, a connector's, a worker's or another
    agent's) is a 409. A session that is not running is a 404. A speech-to-speech model that takes its
    tools only when it opens, such as Gemini Live, cannot be given more during the call: 400.

    Args:
        id (str): The session, as returned when it was created.
        body (AddSessionToolsRequest): Tools to give a running session's caller.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
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
    body: AddSessionToolsRequest,
) -> Any | ErrorResponse | None:
    """Give a running session more of the caller's tools

     Declares tools on a running session as though they had been declared when it was created, for tools
    that exist only partway through a conversation, such as the ones a card shown in it brings. Whoever
    runs the caller's tools is asked to run them over the events socket, and the model is offered them
    from its next request: a reply being written finishes with the tools it started with. A tool named
    as one the caller already has replaces it.

    The kind of caller that opened the session declares its tools: a device those of the session it
    opened, and a backend those of a session a backend opened. Anyone else is refused with a 403. A name
    the session offers that is not the caller's (the router's own, a connector's, a worker's or another
    agent's) is a 409. A session that is not running is a 404. A speech-to-speech model that takes its
    tools only when it opens, such as Gemini Live, cannot be given more during the call: 400.

    Args:
        id (str): The session, as returned when it was created.
        body (AddSessionToolsRequest): Tools to give a running session's caller.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
