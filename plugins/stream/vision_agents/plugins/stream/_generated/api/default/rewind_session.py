from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.rewind_session_request import RewindSessionRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: RewindSessionRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/rewind".format(
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
    body: RewindSessionRequest,
) -> Response[Any | ErrorResponse]:
    """Go back to a response and carry on from there

     The conversation continues as though nothing after the named response had been said: the reply being
    spoken is abandoned, the agent's history is cut back to the end of that response, and every later
    response is marked rewound, so neither the responses nor their items list them again. The named
    response itself is kept.
    The history is rebuilt from what the session recorded, the question and the answer of each turn, so
    a session that recorded nothing cannot be rewound: an incognito one, one on a deployment with no
    store, and a native speech-to-speech one, whose model keeps its own context. A persistent
    conversation is refused as well, because its transcript lives in Chat and would bring the rewound
    turns back the next time it opened; fork it at the response instead.

    Args:
        id (str): The session, as returned when it was created.
        body (RewindSessionRequest):

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
    body: RewindSessionRequest,
) -> Any | ErrorResponse | None:
    """Go back to a response and carry on from there

     The conversation continues as though nothing after the named response had been said: the reply being
    spoken is abandoned, the agent's history is cut back to the end of that response, and every later
    response is marked rewound, so neither the responses nor their items list them again. The named
    response itself is kept.
    The history is rebuilt from what the session recorded, the question and the answer of each turn, so
    a session that recorded nothing cannot be rewound: an incognito one, one on a deployment with no
    store, and a native speech-to-speech one, whose model keeps its own context. A persistent
    conversation is refused as well, because its transcript lives in Chat and would bring the rewound
    turns back the next time it opened; fork it at the response instead.

    Args:
        id (str): The session, as returned when it was created.
        body (RewindSessionRequest):

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
    body: RewindSessionRequest,
) -> Response[Any | ErrorResponse]:
    """Go back to a response and carry on from there

     The conversation continues as though nothing after the named response had been said: the reply being
    spoken is abandoned, the agent's history is cut back to the end of that response, and every later
    response is marked rewound, so neither the responses nor their items list them again. The named
    response itself is kept.
    The history is rebuilt from what the session recorded, the question and the answer of each turn, so
    a session that recorded nothing cannot be rewound: an incognito one, one on a deployment with no
    store, and a native speech-to-speech one, whose model keeps its own context. A persistent
    conversation is refused as well, because its transcript lives in Chat and would bring the rewound
    turns back the next time it opened; fork it at the response instead.

    Args:
        id (str): The session, as returned when it was created.
        body (RewindSessionRequest):

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
    body: RewindSessionRequest,
) -> Any | ErrorResponse | None:
    """Go back to a response and carry on from there

     The conversation continues as though nothing after the named response had been said: the reply being
    spoken is abandoned, the agent's history is cut back to the end of that response, and every later
    response is marked rewound, so neither the responses nor their items list them again. The named
    response itself is kept.
    The history is rebuilt from what the session recorded, the question and the answer of each turn, so
    a session that recorded nothing cannot be rewound: an incognito one, one on a deployment with no
    store, and a native speech-to-speech one, whose model keeps its own context. A persistent
    conversation is refused as well, because its transcript lives in Chat and would bring the rewound
    turns back the next time it opened; fork it at the response instead.

    Args:
        id (str): The session, as returned when it was created.
        body (RewindSessionRequest):

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
