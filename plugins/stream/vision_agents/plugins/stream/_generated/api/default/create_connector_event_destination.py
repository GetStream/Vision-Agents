from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_event_destination_request import (
    ConnectorEventDestinationRequest,
)
from ...models.connector_event_destination_secret import ConnectorEventDestinationSecret
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: ConnectorEventDestinationRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connectors/{id}/event-destinations".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    if response.status_code == 201:
        response_201 = ConnectorEventDestinationSecret.from_dict(response.json())

        return response_201

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

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ConnectorEventDestinationSecret | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorEventDestinationRequest,
) -> Response[ConnectorEventDestinationSecret | ErrorResponse]:
    """Forward a connector's provider events to a URL

     Adds a URL the connector's raw provider events are forwarded to, for the deliveries of the app's own
    provider app, such as its Slack app. A connector takes 3 destinations at most. The response carries
    the destination's signing secret, once.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        body (ConnectorEventDestinationRequest): An event destination to create. An unknown field
            is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorEventDestinationSecret | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorEventDestinationRequest,
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    """Forward a connector's provider events to a URL

     Adds a URL the connector's raw provider events are forwarded to, for the deliveries of the app's own
    provider app, such as its Slack app. A connector takes 3 destinations at most. The response carries
    the destination's signing secret, once.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        body (ConnectorEventDestinationRequest): An event destination to create. An unknown field
            is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorEventDestinationSecret | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorEventDestinationRequest,
) -> Response[ConnectorEventDestinationSecret | ErrorResponse]:
    """Forward a connector's provider events to a URL

     Adds a URL the connector's raw provider events are forwarded to, for the deliveries of the app's own
    provider app, such as its Slack app. A connector takes 3 destinations at most. The response carries
    the destination's signing secret, once.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        body (ConnectorEventDestinationRequest): An event destination to create. An unknown field
            is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorEventDestinationSecret | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    body: ConnectorEventDestinationRequest,
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    """Forward a connector's provider events to a URL

     Adds a URL the connector's raw provider events are forwarded to, for the deliveries of the app's own
    provider app, such as its Slack app. A connector takes 3 destinations at most. The response carries
    the destination's signing secret, once.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        body (ConnectorEventDestinationRequest): An event destination to create. An unknown field
            is refused rather than ignored.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorEventDestinationSecret | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
