from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.router_config import RouterConfig
from ...types import Response


def _get_kwargs() -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/router/configs",
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[RouterConfig] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = RouterConfig.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = Error.from_dict(response.json())

        return response_401

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Error | list[RouterConfig]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | list[RouterConfig]]:
    """The router configs the calling customer holds

     A router config is what an agent config is for a session, for a caller that routes one modality at a
    time: the target, the language and every per-modality option, decided once and named. It is separate
    from an agent config because it configures transcribing, speaking, answering and searching on their
    own, with no conversation behind them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[RouterConfig]]
    """

    kwargs = _get_kwargs()

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
) -> Error | list[RouterConfig] | None:
    """The router configs the calling customer holds

     A router config is what an agent config is for a session, for a caller that routes one modality at a
    time: the target, the language and every per-modality option, decided once and named. It is separate
    from an agent config because it configures transcribing, speaking, answering and searching on their
    own, with no conversation behind them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[RouterConfig]
    """

    return sync_detailed(
        client=client,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
) -> Response[Error | list[RouterConfig]]:
    """The router configs the calling customer holds

     A router config is what an agent config is for a session, for a caller that routes one modality at a
    time: the target, the language and every per-modality option, decided once and named. It is separate
    from an agent config because it configures transcribing, speaking, answering and searching on their
    own, with no conversation behind them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[RouterConfig]]
    """

    kwargs = _get_kwargs()

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
) -> Error | list[RouterConfig] | None:
    """The router configs the calling customer holds

     A router config is what an agent config is for a session, for a caller that routes one modality at a
    time: the target, the language and every per-modality option, decided once and named. It is separate
    from an agent config because it configures transcribing, speaking, answering and searching on their
    own, with no conversation behind them.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[RouterConfig]
    """

    return (
        await asyncio_detailed(
            client=client,
        )
    ).parsed
