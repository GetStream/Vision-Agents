from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.plugin import Plugin
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    q: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["q"] = q

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/plugins",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[Plugin] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Plugin.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[Plugin]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
) -> Response[Error | list[Plugin]]:
    """The hosted MCP servers an agent may attach

     A built-in catalog, not the customer's own rows. q filters by name, category or description.
    Connecting one is a login on a config, not a change to this list.

    Args:
        q (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Plugin]]
    """

    kwargs = _get_kwargs(
        q=q,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
) -> Error | list[Plugin] | None:
    """The hosted MCP servers an agent may attach

     A built-in catalog, not the customer's own rows. q filters by name, category or description.
    Connecting one is a login on a config, not a change to this list.

    Args:
        q (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Plugin]
    """

    return sync_detailed(
        client=client,
        q=q,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
) -> Response[Error | list[Plugin]]:
    """The hosted MCP servers an agent may attach

     A built-in catalog, not the customer's own rows. q filters by name, category or description.
    Connecting one is a login on a config, not a change to this list.

    Args:
        q (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Plugin]]
    """

    kwargs = _get_kwargs(
        q=q,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    q: str | Unset = UNSET,
) -> Error | list[Plugin] | None:
    """The hosted MCP servers an agent may attach

     A built-in catalog, not the customer's own rows. q filters by name, category or description.
    Connecting one is a login on a config, not a change to this list.

    Args:
        q (str | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Plugin]
    """

    return (
        await asyncio_detailed(
            client=client,
            q=q,
        )
    ).parsed
