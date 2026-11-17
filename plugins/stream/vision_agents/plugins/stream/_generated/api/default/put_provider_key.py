from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.provider_key import ProviderKey
from ...models.provider_key_request import ProviderKeyRequest
from ...types import Response


def _get_kwargs(
    *,
    body: ProviderKeyRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/agents/provider-keys",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | ProviderKey | None:
    if response.status_code == 200:
        response_200 = ProviderKey.from_dict(response.json())

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
) -> Response[ErrorResponse | ProviderKey]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ProviderKeyRequest,
) -> Response[ErrorResponse | ProviderKey]:
    """Keep a key for the customer's own account at a vendor

     Bring your own key (BYOK). The key replaces the one kept for the provider in the region, and is
    stored sealed and never returned. From the next request, each model of that provider in that region
    is tried on the customer's key first, immediately before the deployment's own instance of it, which
    is asked next when the customer's fails, as any next candidate is. Work on the customer's key is
    billed to them by the vendor, and by the router at its BYOK price, a fee of 20% of what the work
    costs at the vendor's rates. A turbopuffer key keeps the knowledge bases the customer first writes
    from then on in their own turbopuffer account, and a daytona key makes their sandboxes in their own
    Daytona organization. A provider whose work is never routed on a customer's key is refused with
    provider_takes_no_key, naming the ones that are. Only a key is taken, never an endpoint: the
    deployment's own is used.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ProviderKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ProviderKey]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    body: ProviderKeyRequest,
) -> ErrorResponse | ProviderKey | None:
    """Keep a key for the customer's own account at a vendor

     Bring your own key (BYOK). The key replaces the one kept for the provider in the region, and is
    stored sealed and never returned. From the next request, each model of that provider in that region
    is tried on the customer's key first, immediately before the deployment's own instance of it, which
    is asked next when the customer's fails, as any next candidate is. Work on the customer's key is
    billed to them by the vendor, and by the router at its BYOK price, a fee of 20% of what the work
    costs at the vendor's rates. A turbopuffer key keeps the knowledge bases the customer first writes
    from then on in their own turbopuffer account, and a daytona key makes their sandboxes in their own
    Daytona organization. A provider whose work is never routed on a customer's key is refused with
    provider_takes_no_key, naming the ones that are. Only a key is taken, never an endpoint: the
    deployment's own is used.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ProviderKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ProviderKey
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: ProviderKeyRequest,
) -> Response[ErrorResponse | ProviderKey]:
    """Keep a key for the customer's own account at a vendor

     Bring your own key (BYOK). The key replaces the one kept for the provider in the region, and is
    stored sealed and never returned. From the next request, each model of that provider in that region
    is tried on the customer's key first, immediately before the deployment's own instance of it, which
    is asked next when the customer's fails, as any next candidate is. Work on the customer's key is
    billed to them by the vendor, and by the router at its BYOK price, a fee of 20% of what the work
    costs at the vendor's rates. A turbopuffer key keeps the knowledge bases the customer first writes
    from then on in their own turbopuffer account, and a daytona key makes their sandboxes in their own
    Daytona organization. A provider whose work is never routed on a customer's key is refused with
    provider_takes_no_key, naming the ones that are. Only a key is taken, never an endpoint: the
    deployment's own is used.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ProviderKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | ProviderKey]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: ProviderKeyRequest,
) -> ErrorResponse | ProviderKey | None:
    """Keep a key for the customer's own account at a vendor

     Bring your own key (BYOK). The key replaces the one kept for the provider in the region, and is
    stored sealed and never returned. From the next request, each model of that provider in that region
    is tried on the customer's key first, immediately before the deployment's own instance of it, which
    is asked next when the customer's fails, as any next candidate is. Work on the customer's key is
    billed to them by the vendor, and by the router at its BYOK price, a fee of 20% of what the work
    costs at the vendor's rates. A turbopuffer key keeps the knowledge bases the customer first writes
    from then on in their own turbopuffer account, and a daytona key makes their sandboxes in their own
    Daytona organization. A provider whose work is never routed on a customer's key is refused with
    provider_takes_no_key, naming the ones that are. Only a key is taken, never an endpoint: the
    deployment's own is used.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (ProviderKeyRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | ProviderKey
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
