from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    provider: str,
    *,
    region: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["region"] = region

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/provider-keys/{provider}".format(
            provider=quote(str(provider), safe=""),
        ),
        "params": params,
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
    provider: str,
    *,
    client: AuthenticatedClient | Client,
    region: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Forget a key the customer keeps for a vendor

     Work is routed on the deployment's own keys from the next request. A knowledge base kept in the
    customer's own turbopuffer account is refused until a key for its region is put back: it is never
    read from the deployment's account.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        provider (str): The provider the key is for, such as anthropic.
        region (str | Unset): The region the key is kept for. Omitted is us.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        provider=provider,
        region=region,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    provider: str,
    *,
    client: AuthenticatedClient | Client,
    region: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Forget a key the customer keeps for a vendor

     Work is routed on the deployment's own keys from the next request. A knowledge base kept in the
    customer's own turbopuffer account is refused until a key for its region is put back: it is never
    read from the deployment's account.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        provider (str): The provider the key is for, such as anthropic.
        region (str | Unset): The region the key is kept for. Omitted is us.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        provider=provider,
        client=client,
        region=region,
    ).parsed


async def asyncio_detailed(
    provider: str,
    *,
    client: AuthenticatedClient | Client,
    region: str | Unset = UNSET,
) -> Response[Any | ErrorResponse]:
    """Forget a key the customer keeps for a vendor

     Work is routed on the deployment's own keys from the next request. A knowledge base kept in the
    customer's own turbopuffer account is refused until a key for its region is put back: it is never
    read from the deployment's account.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        provider (str): The provider the key is for, such as anthropic.
        region (str | Unset): The region the key is kept for. Omitted is us.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        provider=provider,
        region=region,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    provider: str,
    *,
    client: AuthenticatedClient | Client,
    region: str | Unset = UNSET,
) -> Any | ErrorResponse | None:
    """Forget a key the customer keeps for a vendor

     Work is routed on the deployment's own keys from the next request. A knowledge base kept in the
    customer's own turbopuffer account is refused until a key for its region is put back: it is never
    read from the deployment's account.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        provider (str): The provider the key is for, such as anthropic.
        region (str | Unset): The region the key is kept for. Omitted is us.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            provider=provider,
            client=client,
            region=region,
        )
    ).parsed
