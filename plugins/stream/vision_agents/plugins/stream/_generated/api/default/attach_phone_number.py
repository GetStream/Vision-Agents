from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.attach_number_request import AttachNumberRequest
from ...models.attached_number import AttachedNumber
from ...models.error import Error
from ...types import UNSET, Response, Unset


def _get_kwargs(
    e164: str,
    *,
    body: AttachNumberRequest | Unset = UNSET,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/phone/numbers/{e164}/attach".format(
            e164=quote(str(e164), safe=""),
        ),
    }

    if not isinstance(body, Unset):
        _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> AttachedNumber | Error | None:
    if response.status_code == 200:
        response_200 = AttachedNumber.from_dict(response.json())

        return response_200

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
) -> Response[AttachedNumber | Error]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    e164: str,
    *,
    client: AuthenticatedClient | Client,
    body: AttachNumberRequest | Unset = UNSET,
) -> Response[AttachedNumber | Error]:
    """Point a number at a Stream call

     Creates the SIP inbound trunk and routing rule and tells the vendor to send calls there. This is
    what turns a bought number into one that reaches an agent.

    Args:
        e164 (str):
        body (AttachNumberRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AttachedNumber | Error]
    """

    kwargs = _get_kwargs(
        e164=e164,
        body=body,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    e164: str,
    *,
    client: AuthenticatedClient | Client,
    body: AttachNumberRequest | Unset = UNSET,
) -> AttachedNumber | Error | None:
    """Point a number at a Stream call

     Creates the SIP inbound trunk and routing rule and tells the vendor to send calls there. This is
    what turns a bought number into one that reaches an agent.

    Args:
        e164 (str):
        body (AttachNumberRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AttachedNumber | Error
    """

    return sync_detailed(
        e164=e164,
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    e164: str,
    *,
    client: AuthenticatedClient | Client,
    body: AttachNumberRequest | Unset = UNSET,
) -> Response[AttachedNumber | Error]:
    """Point a number at a Stream call

     Creates the SIP inbound trunk and routing rule and tells the vendor to send calls there. This is
    what turns a bought number into one that reaches an agent.

    Args:
        e164 (str):
        body (AttachNumberRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[AttachedNumber | Error]
    """

    kwargs = _get_kwargs(
        e164=e164,
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    e164: str,
    *,
    client: AuthenticatedClient | Client,
    body: AttachNumberRequest | Unset = UNSET,
) -> AttachedNumber | Error | None:
    """Point a number at a Stream call

     Creates the SIP inbound trunk and routing rule and tells the vendor to send calls there. This is
    what turns a bought number into one that reaches an agent.

    Args:
        e164 (str):
        body (AttachNumberRequest | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        AttachedNumber | Error
    """

    return (
        await asyncio_detailed(
            e164=e164,
            client=client,
            body=body,
        )
    ).parsed
