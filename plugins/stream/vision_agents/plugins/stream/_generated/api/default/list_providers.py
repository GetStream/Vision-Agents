from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.modality import Modality
from ...models.provider import Provider
from ...types import Response


def _get_kwargs(
    modality: Modality,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/{modality}/providers".format(
            modality=quote(str(modality), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[Provider] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Provider.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if response.status_code == 404:
        response_404 = Error.from_dict(response.json())

        return response_404

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[Provider]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | list[Provider]]:
    """List the providers configured for a modality and their live health

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Provider]]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Error | list[Provider] | None:
    """List the providers configured for a modality and their live health

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Provider]
    """

    return sync_detailed(
        modality=modality,
        client=client,
    ).parsed


async def asyncio_detailed(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | list[Provider]]:
    """List the providers configured for a modality and their live health

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Provider]]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    modality: Modality,
    *,
    client: AuthenticatedClient | Client,
) -> Error | list[Provider] | None:
    """List the providers configured for a modality and their live health

    Args:
        modality (Modality): What kind of work was done. The first four are routed across
            providers. Memory, knowledge and phone are recorded but not routed, since there is one
            memory store, one knowledge base and one vendor per number, so the provider paths do not
            serve them while the statistics paths do.
             Example: tts.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Provider]
    """

    return (
        await asyncio_detailed(
            modality=modality,
            client=client,
        )
    ).parsed
