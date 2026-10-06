from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.app_settings import AppSettings
from ...models.error_response import ErrorResponse
from ...models.stream_credentials import StreamCredentials
from ...types import Response


def _get_kwargs(
    *,
    body: StreamCredentials,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/settings/app/stream/credentials",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AppSettings | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = AppSettings.from_dict(response.json())

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
) -> Response[AppSettings | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: StreamCredentials,
) -> Response[AppSettings | ErrorResponse]:
    """Register the calling app's own Stream app

     The keys the router acts in the calling app's own Stream app with, from now on, for every
    conversation, transcript, call and phone line. Each key is checked with Stream: it has to belong to
    the calling app, and the app may be neither suspended nor taking requests without checking their
    tokens. Keys left out are dropped, and the sessions acting in the app end. Needs stream.tenancy=app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    Behind a proxy, the proxy has to declare the caller a server.

    Args:
        body (StreamCredentials): The keys the router acts in the calling app's own Stream app
            with. Secrets are written and never read back: no answer carries one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AppSettings | ErrorResponse]
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
    body: StreamCredentials,
) -> AppSettings | ErrorResponse | None:
    """Register the calling app's own Stream app

     The keys the router acts in the calling app's own Stream app with, from now on, for every
    conversation, transcript, call and phone line. Each key is checked with Stream: it has to belong to
    the calling app, and the app may be neither suspended nor taking requests without checking their
    tokens. Keys left out are dropped, and the sessions acting in the app end. Needs stream.tenancy=app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    Behind a proxy, the proxy has to declare the caller a server.

    Args:
        body (StreamCredentials): The keys the router acts in the calling app's own Stream app
            with. Secrets are written and never read back: no answer carries one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AppSettings | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: StreamCredentials,
) -> Response[AppSettings | ErrorResponse]:
    """Register the calling app's own Stream app

     The keys the router acts in the calling app's own Stream app with, from now on, for every
    conversation, transcript, call and phone line. Each key is checked with Stream: it has to belong to
    the calling app, and the app may be neither suspended nor taking requests without checking their
    tokens. Keys left out are dropped, and the sessions acting in the app end. Needs stream.tenancy=app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    Behind a proxy, the proxy has to declare the caller a server.

    Args:
        body (StreamCredentials): The keys the router acts in the calling app's own Stream app
            with. Secrets are written and never read back: no answer carries one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AppSettings | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: StreamCredentials,
) -> AppSettings | ErrorResponse | None:
    """Register the calling app's own Stream app

     The keys the router acts in the calling app's own Stream app with, from now on, for every
    conversation, transcript, call and phone line. Each key is checked with Stream: it has to belong to
    the calling app, and the app may be neither suspended nor taking requests without checking their
    tokens. Keys left out are dropped, and the sessions acting in the app end. Needs stream.tenancy=app.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.
    Behind a proxy, the proxy has to declare the caller a server.

    Args:
        body (StreamCredentials): The keys the router acts in the calling app's own Stream app
            with. Secrets are written and never read back: no answer carries one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AppSettings | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
