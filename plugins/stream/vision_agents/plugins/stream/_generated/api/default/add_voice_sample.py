from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.voice import Voice
from ...models.voice_sample_request import VoiceSampleRequest
from ...types import Response


def _get_kwargs(
    id: str,
    *,
    body: VoiceSampleRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/voices/{id}/samples".format(
            id=quote(str(id), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | Voice | None:
    if response.status_code == 201:
        response_201 = Voice.from_dict(response.json())

        return response_201

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
) -> Response[Error | Voice]:
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
    body: VoiceSampleRequest,
) -> Response[Error | Voice]:
    """Add a recording to a voice

     The audio is stored in the deployment's object bucket and the voice keeps a reference to it, so
    recordings can be re-sent to a provider that is added later without asking the customer for them
    again. Adding a recording does not re-prepare the voice.

    Args:
        id (str):
        body (VoiceSampleRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Voice]
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
    body: VoiceSampleRequest,
) -> Error | Voice | None:
    """Add a recording to a voice

     The audio is stored in the deployment's object bucket and the voice keeps a reference to it, so
    recordings can be re-sent to a provider that is added later without asking the customer for them
    again. Adding a recording does not re-prepare the voice.

    Args:
        id (str):
        body (VoiceSampleRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Voice
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
    body: VoiceSampleRequest,
) -> Response[Error | Voice]:
    """Add a recording to a voice

     The audio is stored in the deployment's object bucket and the voice keeps a reference to it, so
    recordings can be re-sent to a provider that is added later without asking the customer for them
    again. Adding a recording does not re-prepare the voice.

    Args:
        id (str):
        body (VoiceSampleRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | Voice]
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
    body: VoiceSampleRequest,
) -> Error | Voice | None:
    """Add a recording to a voice

     The audio is stored in the deployment's object bucket and the voice keeps a reference to it, so
    recordings can be re-sent to a provider that is added later without asking the customer for them
    again. Adding a recording does not re-prepare the voice.

    Args:
        id (str):
        body (VoiceSampleRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | Voice
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            body=body,
        )
    ).parsed
