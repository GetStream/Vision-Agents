from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.suspend_telephony_request import SuspendTelephonyRequest
from ...models.telephony_suspension import TelephonySuspension
from ...types import UNSET, Response, Unset


def _get_kwargs(
    customer_id: str,
    *,
    body: SuspendTelephonyRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/ops/customers/{customer_id}/telephony-suspension".format(
            customer_id=quote(str(customer_id), safe=""),
        ),
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | TelephonySuspension | None:
    if response.status_code == 200:
        response_200 = TelephonySuspension.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

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
) -> Response[ErrorResponse | TelephonySuspension]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    customer_id: str,
    *,
    client: AuthenticatedClient,
    body: SuspendTelephonyRequest | Unset = UNSET,
) -> Response[ErrorResponse | TelephonySuspension]:
    """Suspend an app's telephony

     Stops an app using the phone, for abuse or fraud. While it is suspended the app buys no numbers,
    places no calls or transfers, answers no incoming call and starts no campaign, and its running
    campaigns pause at their next call. Calls already in progress are left alone. Suspending an app
    already suspended replaces the reason.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The app.
        body (SuspendTelephonyRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | TelephonySuspension]
    """

    kwargs = _get_kwargs(
        customer_id=customer_id,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    customer_id: str,
    *,
    client: AuthenticatedClient,
    body: SuspendTelephonyRequest | Unset = UNSET,
) -> ErrorResponse | TelephonySuspension | None:
    """Suspend an app's telephony

     Stops an app using the phone, for abuse or fraud. While it is suspended the app buys no numbers,
    places no calls or transfers, answers no incoming call and starts no campaign, and its running
    campaigns pause at their next call. Calls already in progress are left alone. Suspending an app
    already suspended replaces the reason.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The app.
        body (SuspendTelephonyRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | TelephonySuspension
    """

    return sync_detailed(
        customer_id=customer_id,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    customer_id: str,
    *,
    client: AuthenticatedClient,
    body: SuspendTelephonyRequest | Unset = UNSET,
) -> Response[ErrorResponse | TelephonySuspension]:
    """Suspend an app's telephony

     Stops an app using the phone, for abuse or fraud. While it is suspended the app buys no numbers,
    places no calls or transfers, answers no incoming call and starts no campaign, and its running
    campaigns pause at their next call. Calls already in progress are left alone. Suspending an app
    already suspended replaces the reason.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The app.
        body (SuspendTelephonyRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | TelephonySuspension]
    """

    kwargs = _get_kwargs(
        customer_id=customer_id,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    customer_id: str,
    *,
    client: AuthenticatedClient,
    body: SuspendTelephonyRequest | Unset = UNSET,
) -> ErrorResponse | TelephonySuspension | None:
    """Suspend an app's telephony

     Stops an app using the phone, for abuse or fraud. While it is suspended the app buys no numbers,
    places no calls or transfers, answers no incoming call and starts no campaign, and its running
    campaigns pause at their next call. Calls already in progress are left alone. Suspending an app
    already suspended replaces the reason.

    Stream staff only: it needs the ops key.

    Args:
        customer_id (str): The app.
        body (SuspendTelephonyRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | TelephonySuspension
    """

    return (
        await asyncio_detailed(
            customer_id=customer_id,
            client=client,
            body=body,
        )
    ).parsed
