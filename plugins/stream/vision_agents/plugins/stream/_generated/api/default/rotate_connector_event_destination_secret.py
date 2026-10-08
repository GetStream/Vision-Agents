from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.connector_event_destination_secret import ConnectorEventDestinationSecret
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    destination_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/connectors/{id}/event-destinations/{destination_id}/rotate-secret".format(
            id=quote(str(id), safe=""),
            destination_id=quote(str(destination_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = ConnectorEventDestinationSecret.from_dict(response.json())

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
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ConnectorEventDestinationSecret | ErrorResponse]:
    """Rotate an event destination's signing secret

     Makes a new signing secret for the destination and returns it, once. For the next 24 hours every
    forward is signed with both the new and the old secret, space-separated in webhook-signature, so the
    receiver can move to the new one without a forward failing its check. A rotation during another
    drops the oldest secret.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorEventDestinationSecret | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        destination_id=destination_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    """Rotate an event destination's signing secret

     Makes a new signing secret for the destination and returns it, once. For the next 24 hours every
    forward is signed with both the new and the old secret, space-separated in webhook-signature, so the
    receiver can move to the new one without a forward failing its check. A rotation during another
    drops the oldest secret.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorEventDestinationSecret | ErrorResponse
    """

    return sync_detailed(
        id=id,
        destination_id=destination_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[ConnectorEventDestinationSecret | ErrorResponse]:
    """Rotate an event destination's signing secret

     Makes a new signing secret for the destination and returns it, once. For the next 24 hours every
    forward is signed with both the new and the old secret, space-separated in webhook-signature, so the
    receiver can move to the new one without a forward failing its check. A rotation during another
    drops the oldest secret.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ConnectorEventDestinationSecret | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        destination_id=destination_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    destination_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> ConnectorEventDestinationSecret | ErrorResponse | None:
    """Rotate an event destination's signing secret

     Makes a new signing secret for the destination and returns it, once. For the next 24 hours every
    forward is signed with both the new and the old secret, space-separated in webhook-signature, so the
    receiver can move to the new one without a forward failing its check. A rotation during another
    drops the oldest secret.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The connector, such as slack_bot.
        destination_id (str): The destination, as returned when it was created.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ConnectorEventDestinationSecret | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            destination_id=destination_id,
            client=client,
        )
    ).parsed
