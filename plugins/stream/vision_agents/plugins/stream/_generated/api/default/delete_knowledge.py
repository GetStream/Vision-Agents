from http import HTTPStatus
from typing import Any, cast

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response


def _get_kwargs(
    *,
    namespace: str,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["namespace"] = namespace

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "delete",
        "url": "/v1/agents/knowledge",
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
    *,
    client: AuthenticatedClient | Client,
    namespace: str,
) -> Response[Any | ErrorResponse]:
    """Delete a knowledge base

     Everything in it goes, for good: its passages, the documents it was filled with and the pages it was
    kept filled from. A knowledge base an agent config reads is refused with a 409 naming those agents,
    so no agent is left looking things up in nothing.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        namespace (str): The knowledge base, as it was named when it was filled.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        namespace=namespace,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    namespace: str,
) -> Any | ErrorResponse | None:
    """Delete a knowledge base

     Everything in it goes, for good: its passages, the documents it was filled with and the pages it was
    kept filled from. A knowledge base an agent config reads is refused with a 409 naming those agents,
    so no agent is left looking things up in nothing.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        namespace (str): The knowledge base, as it was named when it was filled.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        client=client,
        namespace=namespace,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    namespace: str,
) -> Response[Any | ErrorResponse]:
    """Delete a knowledge base

     Everything in it goes, for good: its passages, the documents it was filled with and the pages it was
    kept filled from. A knowledge base an agent config reads is refused with a 409 naming those agents,
    so no agent is left looking things up in nothing.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        namespace (str): The knowledge base, as it was named when it was filled.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        namespace=namespace,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    namespace: str,
) -> Any | ErrorResponse | None:
    """Delete a knowledge base

     Everything in it goes, for good: its passages, the documents it was filled with and the pages it was
    kept filled from. A knowledge base an agent config reads is refused with a 409 naming those agents,
    so no agent is left looking things up in nothing.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        namespace (str): The knowledge base, as it was named when it was filled.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            namespace=namespace,
        )
    ).parsed
