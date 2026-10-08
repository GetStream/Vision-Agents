from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    path: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/connections/{id}/proxy/{path}".format(
            id=quote(str(id), safe=""),
            path=quote(str(path), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = cast(Any, None)
        return response_200

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 413:
        response_413 = ErrorResponse.from_dict(response.json())

        return response_413

    if response.status_code == 429:
        response_429 = ErrorResponse.from_dict(response.json())

        return response_429

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
    path: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Call a connection's provider directly (GET)

     Forwards the request to the connector's api_base with path appended, and answers with the provider's
    answer as it came: status, headers and body. The request goes as it came, but for the router's own
    credentials and caller headers (Authorization, X-Api-Key, Stream-Auth-Type, X-Stream-*, X-Customer-
    Id) and query parameters (api_key, token, customer_id, user_id), which never reach the provider; the
    connection's own credential is added instead. On a 401 the credential is renewed and the request
    sent once more when the scheme can renew it. A provider's 429 and Retry-After come back as they are,
    and the connection's calls are then refused with a 429 here until that Retry-After passes. A path
    with a dot segment, which would leave api_base, is refused. The body is at most 1 MiB. Point a
    provider's own SDK at this URL as its base URL, with a server-side token as its token and X-Api-Key
    and Stream-Auth-Type as extra headers. An app-owned connection is the app's backend's; a user-owned
    one is reached only by a backend acting for that user (X-Stream-User-Id). Each call that is sent
    leaves one proxy_call audit row.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        path=path,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    path: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Call a connection's provider directly (GET)

     Forwards the request to the connector's api_base with path appended, and answers with the provider's
    answer as it came: status, headers and body. The request goes as it came, but for the router's own
    credentials and caller headers (Authorization, X-Api-Key, Stream-Auth-Type, X-Stream-*, X-Customer-
    Id) and query parameters (api_key, token, customer_id, user_id), which never reach the provider; the
    connection's own credential is added instead. On a 401 the credential is renewed and the request
    sent once more when the scheme can renew it. A provider's 429 and Retry-After come back as they are,
    and the connection's calls are then refused with a 429 here until that Retry-After passes. A path
    with a dot segment, which would leave api_base, is refused. The body is at most 1 MiB. Point a
    provider's own SDK at this URL as its base URL, with a server-side token as its token and X-Api-Key
    and Stream-Auth-Type as extra headers. An app-owned connection is the app's backend's; a user-owned
    one is reached only by a backend acting for that user (X-Stream-User-Id). Each call that is sent
    leaves one proxy_call audit row.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        id=id,
        path=path,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    path: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse]:
    """Call a connection's provider directly (GET)

     Forwards the request to the connector's api_base with path appended, and answers with the provider's
    answer as it came: status, headers and body. The request goes as it came, but for the router's own
    credentials and caller headers (Authorization, X-Api-Key, Stream-Auth-Type, X-Stream-*, X-Customer-
    Id) and query parameters (api_key, token, customer_id, user_id), which never reach the provider; the
    connection's own credential is added instead. On a 401 the credential is renewed and the request
    sent once more when the scheme can renew it. A provider's 429 and Retry-After come back as they are,
    and the connection's calls are then refused with a 429 here until that Retry-After passes. A path
    with a dot segment, which would leave api_base, is refused. The body is at most 1 MiB. Point a
    provider's own SDK at this URL as its base URL, with a server-side token as its token and X-Api-Key
    and Stream-Auth-Type as extra headers. An app-owned connection is the app's backend's; a user-owned
    one is reached only by a backend acting for that user (X-Stream-User-Id). Each call that is sent
    leaves one proxy_call audit row.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        path=path,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    path: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | None:
    """Call a connection's provider directly (GET)

     Forwards the request to the connector's api_base with path appended, and answers with the provider's
    answer as it came: status, headers and body. The request goes as it came, but for the router's own
    credentials and caller headers (Authorization, X-Api-Key, Stream-Auth-Type, X-Stream-*, X-Customer-
    Id) and query parameters (api_key, token, customer_id, user_id), which never reach the provider; the
    connection's own credential is added instead. On a 401 the credential is renewed and the request
    sent once more when the scheme can renew it. A provider's 429 and Retry-After come back as they are,
    and the connection's calls are then refused with a 429 here until that Retry-After passes. A path
    with a dot segment, which would leave api_base, is refused. The body is at most 1 MiB. Point a
    provider's own SDK at this URL as its base URL, with a server-side token as its token and X-Api-Key
    and Stream-Auth-Type as extra headers. An app-owned connection is the app's backend's; a user-owned
    one is reached only by a backend acting for that user (X-Stream-User-Id). Each call that is sent
    leaves one proxy_call audit row.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str):
        path (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            path=path,
            client=client,
        )
    ).parsed
