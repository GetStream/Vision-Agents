from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    connector_id: str,
    provider_app_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/connectors/events/{connector_id}/{provider_app_id}".format(
            connector_id=quote(str(connector_id), safe=""),
            provider_app_id=quote(str(provider_app_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | str | None:
    if response.status_code == 200:
        response_200 = response.text
        return response_200

    if response.status_code == 401:
        response_401 = cast(Any, None)
        return response_401

    if response.status_code == 404:
        response_404 = cast(Any, None)
        return response_404

    if response.status_code == 413:
        response_413 = cast(Any, None)
        return response_413

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse | str]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse | str]:
    """Receive a provider app's event

     Where a provider delivers the events of one customer's provider app: the Request URL of a customer's
    Slack app, for one. Unauthenticated because the provider is not a customer: each request is checked
    by the verifier the connector's manifest names (channel.verifier) against that app's own signing
    secret, so an event signed for another app is refused and changes nothing. A URL verification is
    answered with its challenge as text/plain. A signal that a grant ended moves the app's customer's
    connections of that account to needs_reauthorization, unless they connected after the event. A
    message goes to the channel bridge, which writes it into the thread channel of its external thread
    in Stream Chat; a retried delivery is dropped. The body is at most 256 KiB. No SDK wraps it: only a
    provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        connector_id=connector_id,
        provider_app_id=provider_app_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | str | None:
    """Receive a provider app's event

     Where a provider delivers the events of one customer's provider app: the Request URL of a customer's
    Slack app, for one. Unauthenticated because the provider is not a customer: each request is checked
    by the verifier the connector's manifest names (channel.verifier) against that app's own signing
    secret, so an event signed for another app is refused and changes nothing. A URL verification is
    answered with its challenge as text/plain. A signal that a grant ended moves the app's customer's
    connections of that account to needs_reauthorization, unless they connected after the event. A
    message goes to the channel bridge, which writes it into the thread channel of its external thread
    in Stream Chat; a retried delivery is dropped. The body is at most 256 KiB. No SDK wraps it: only a
    provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse | str
    """

    return sync_detailed(
        connector_id=connector_id,
        provider_app_id=provider_app_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Any | ErrorResponse | str]:
    """Receive a provider app's event

     Where a provider delivers the events of one customer's provider app: the Request URL of a customer's
    Slack app, for one. Unauthenticated because the provider is not a customer: each request is checked
    by the verifier the connector's manifest names (channel.verifier) against that app's own signing
    secret, so an event signed for another app is refused and changes nothing. A URL verification is
    answered with its challenge as text/plain. A signal that a grant ended moves the app's customer's
    connections of that account to needs_reauthorization, unless they connected after the event. A
    message goes to the channel bridge, which writes it into the thread channel of its external thread
    in Stream Chat; a retried delivery is dropped. The body is at most 256 KiB. No SDK wraps it: only a
    provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse | str]
    """

    kwargs = _get_kwargs(
        connector_id=connector_id,
        provider_app_id=provider_app_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    connector_id: str,
    provider_app_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Any | ErrorResponse | str | None:
    """Receive a provider app's event

     Where a provider delivers the events of one customer's provider app: the Request URL of a customer's
    Slack app, for one. Unauthenticated because the provider is not a customer: each request is checked
    by the verifier the connector's manifest names (channel.verifier) against that app's own signing
    secret, so an event signed for another app is refused and changes nothing. A URL verification is
    answered with its challenge as text/plain. A signal that a grant ended moves the app's customer's
    connections of that account to needs_reauthorization, unless they connected after the event. A
    message goes to the channel bridge, which writes it into the thread channel of its external thread
    in Stream Chat; a retried delivery is dropped. The body is at most 256 KiB. No SDK wraps it: only a
    provider calls it.

    Args:
        connector_id (str):
        provider_app_id (str):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse | str
    """

    return (
        await asyncio_detailed(
            connector_id=connector_id,
            provider_app_id=provider_app_id,
            client=client,
        )
    ).parsed
