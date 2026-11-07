from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.contact import Contact
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    id: str,
    contact_id: str,
) -> dict[str, Any]:

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/campaigns/{id}/contacts/{contact_id}/stop".format(
            id=quote(str(id), safe=""),
            contact_id=quote(str(contact_id), safe=""),
        ),
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Contact | ErrorResponse | None:
    if response.status_code == 200:
        response_200 = Contact.from_dict(response.json())

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
) -> Response[Contact | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    id: str,
    contact_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Contact | ErrorResponse]:
    """Ring a contact no more

     One waiting for their next attempt is stopped now. One on a call is not cut off: they are stopped
    when the call ends. One the sequence is done with is left as it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The campaign.
        contact_id (str): The contact, as the campaign's contacts list it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Contact | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        contact_id=contact_id,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    contact_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Contact | ErrorResponse | None:
    """Ring a contact no more

     One waiting for their next attempt is stopped now. One on a call is not cut off: they are stopped
    when the call ends. One the sequence is done with is left as it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The campaign.
        contact_id (str): The contact, as the campaign's contacts list it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Contact | ErrorResponse
    """

    return sync_detailed(
        id=id,
        contact_id=contact_id,
        client=client,
    ).parsed


async def asyncio_detailed(
    id: str,
    contact_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Response[Contact | ErrorResponse]:
    """Ring a contact no more

     One waiting for their next attempt is stopped now. One on a call is not cut off: they are stopped
    when the call ends. One the sequence is done with is left as it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The campaign.
        contact_id (str): The contact, as the campaign's contacts list it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Contact | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        contact_id=contact_id,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    contact_id: str,
    *,
    client: AuthenticatedClient | Client,
) -> Contact | ErrorResponse | None:
    """Ring a contact no more

     One waiting for their next attempt is stopped now. One on a call is not cut off: they are stopped
    when the call ends. One the sequence is done with is left as it is.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        id (str): The campaign.
        contact_id (str): The contact, as the campaign's contacts list it.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Contact | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            contact_id=contact_id,
            client=client,
        )
    ).parsed
