from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_response import AgentResponse
from ...models.create_response_request import CreateResponseRequest
from ...models.error import Error
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: CreateResponseRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sessions/{id}/responses".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AgentResponse | Error | None:
    if response.status_code == 202:
        response_202 = AgentResponse.from_dict(response.json())

        return response_202

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[AgentResponse | Error]:
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
    body: CreateResponseRequest,
) -> Response[AgentResponse | Error]:
    """Ask the agent something and get a handle on the answer

     The same thing respond does, with an id back. That is the whole difference and the reason this
    exists: respond returns nothing, so a caller that wants to follow one particular turn has to watch
    the socket and guess which events belong to it. With an id it can ask for that turn's items instead.
    It returns as soon as the turn has started, not when it has finished. A model takes seconds and a
    request that waited them out would time out on anything long enough to be worth asking.

    Args:
        id (str):
        body (CreateResponseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponse | Error]
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
    body: CreateResponseRequest,
) -> AgentResponse | Error | None:
    """Ask the agent something and get a handle on the answer

     The same thing respond does, with an id back. That is the whole difference and the reason this
    exists: respond returns nothing, so a caller that wants to follow one particular turn has to watch
    the socket and guess which events belong to it. With an id it can ask for that turn's items instead.
    It returns as soon as the turn has started, not when it has finished. A model takes seconds and a
    request that waited them out would time out on anything long enough to be worth asking.

    Args:
        id (str):
        body (CreateResponseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponse | Error
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
    body: CreateResponseRequest,
) -> Response[AgentResponse | Error]:
    """Ask the agent something and get a handle on the answer

     The same thing respond does, with an id back. That is the whole difference and the reason this
    exists: respond returns nothing, so a caller that wants to follow one particular turn has to watch
    the socket and guess which events belong to it. With an id it can ask for that turn's items instead.
    It returns as soon as the turn has started, not when it has finished. A model takes seconds and a
    request that waited them out would time out on anything long enough to be worth asking.

    Args:
        id (str):
        body (CreateResponseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AgentResponse | Error]
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
    body: CreateResponseRequest,
) -> AgentResponse | Error | None:
    """Ask the agent something and get a handle on the answer

     The same thing respond does, with an id back. That is the whole difference and the reason this
    exists: respond returns nothing, so a caller that wants to follow one particular turn has to watch
    the socket and guess which events belong to it. With an id it can ask for that turn's items instead.
    It returns as soon as the turn has started, not when it has finished. A model takes seconds and a
    request that waited them out would time out on anything long enough to be worth asking.

    Args:
        id (str):
        body (CreateResponseRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AgentResponse | Error
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
