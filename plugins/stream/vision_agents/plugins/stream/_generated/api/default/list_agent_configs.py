from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.agent_config import AgentConfig
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    name: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["name"] = name

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/configs",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[AgentConfig] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = AgentConfig.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = Error.from_dict(response.json())

        return response_403

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[AgentConfig]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    name: str | Unset = UNSET,
) -> Response[Error | list[AgentConfig]]:
    """The agent configs the calling customer holds

     With a name this is how a name becomes a config, which is what lets a backend say "docs" instead of
    an id it never chose.
    Server-side only, as it always was: a config carries the instructions the agent runs under, and
    those are not a page's business. A page addressing an agent by name does not need this -- it sends
    the name on the create-session request and the router resolves it, which is the same lookup without
    handing the instructions over.

    Args:
        name (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[AgentConfig]]
    """

    kwargs = _get_kwargs(
        name=name,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    name: str | Unset = UNSET,
) -> Error | list[AgentConfig] | None:
    """The agent configs the calling customer holds

     With a name this is how a name becomes a config, which is what lets a backend say "docs" instead of
    an id it never chose.
    Server-side only, as it always was: a config carries the instructions the agent runs under, and
    those are not a page's business. A page addressing an agent by name does not need this -- it sends
    the name on the create-session request and the router resolves it, which is the same lookup without
    handing the instructions over.

    Args:
        name (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[AgentConfig]
    """

    return sync_detailed(
        client=client,
        name=name,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    name: str | Unset = UNSET,
) -> Response[Error | list[AgentConfig]]:
    """The agent configs the calling customer holds

     With a name this is how a name becomes a config, which is what lets a backend say "docs" instead of
    an id it never chose.
    Server-side only, as it always was: a config carries the instructions the agent runs under, and
    those are not a page's business. A page addressing an agent by name does not need this -- it sends
    the name on the create-session request and the router resolves it, which is the same lookup without
    handing the instructions over.

    Args:
        name (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[AgentConfig]]
    """

    kwargs = _get_kwargs(
        name=name,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    name: str | Unset = UNSET,
) -> Error | list[AgentConfig] | None:
    """The agent configs the calling customer holds

     With a name this is how a name becomes a config, which is what lets a backend say "docs" instead of
    an id it never chose.
    Server-side only, as it always was: a config carries the instructions the agent runs under, and
    those are not a page's business. A page addressing an agent by name does not need this -- it sends
    the name on the create-session request and the router resolves it, which is the same lookup without
    handing the instructions over.

    Args:
        name (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[AgentConfig]
    """

    return (
        await asyncio_detailed(
            client=client,
            name=name,
        )
    ).parsed
