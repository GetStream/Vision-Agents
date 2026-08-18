from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.placed_call import PlacedCall
from ...models.transfer_call_request import TransferCallRequest
from ...types import Response


def _get_kwargs(
    *,
    body: TransferCallRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/calls/transfer",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | PlacedCall | None:
    if response.status_code == 202:
        response_202 = PlacedCall.from_dict(response.json())

        return response_202

    if response.status_code == 400:
        response_400 = Error.from_dict(response.json())

        return response_400

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
) -> Response[Error | PlacedCall]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TransferCallRequest,
) -> Response[Error | PlacedCall]:
    """Bring a human onto a call that is already happening

     Stream's SIP is inbound only, so a transfer is a second leg rather than a handover: the vendor dials
    the human and the answered leg is routed into the same Stream call, after which the agent can leave.
    The caller is never moved, so nothing is lost if nobody answers.

    Args:
        body (TransferCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | PlacedCall]
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
    body: TransferCallRequest,
) -> Error | PlacedCall | None:
    """Bring a human onto a call that is already happening

     Stream's SIP is inbound only, so a transfer is a second leg rather than a handover: the vendor dials
    the human and the answered leg is routed into the same Stream call, after which the agent can leave.
    The caller is never moved, so nothing is lost if nobody answers.

    Args:
        body (TransferCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | PlacedCall
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: TransferCallRequest,
) -> Response[Error | PlacedCall]:
    """Bring a human onto a call that is already happening

     Stream's SIP is inbound only, so a transfer is a second leg rather than a handover: the vendor dials
    the human and the answered leg is routed into the same Stream call, after which the agent can leave.
    The caller is never moved, so nothing is lost if nobody answers.

    Args:
        body (TransferCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | PlacedCall]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: TransferCallRequest,
) -> Error | PlacedCall | None:
    """Bring a human onto a call that is already happening

     Stream's SIP is inbound only, so a transfer is a second leg rather than a handover: the vendor dials
    the human and the answered leg is routed into the same Stream call, after which the agent can leave.
    The caller is never moved, so nothing is lost if nobody answers.

    Args:
        body (TransferCallRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | PlacedCall
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
