from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.phone_number import PhoneNumber
from ...models.set_number_agent_request import SetNumberAgentRequest
from ...types import Response


def _get_kwargs(
    e164: str,
    *,
    body: SetNumberAgentRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "put",
        "url": "/v1/phone/numbers/{e164}/agent".format(
            e164=quote(str(e164), safe=""),
        ),
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | PhoneNumber | None:
    if response.status_code == 200:
        response_200 = PhoneNumber.from_dict(response.json())

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
) -> Response[ErrorResponse | PhoneNumber]:
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
    body: SetNumberAgentRequest,
) -> Response[ErrorResponse | PhoneNumber]:
    """Have an agent answer a number's calls

     Each caller gets a session of their own with the agent config, and no call is dispatched to a
    worker. Replaces the agent the number had. The number must be attached; detaching or releasing it
    removes its agent.

    Args:
        e164 (str): The number in +15551234567 form.
        body (SetNumberAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | PhoneNumber]
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
    body: SetNumberAgentRequest,
) -> ErrorResponse | PhoneNumber | None:
    """Have an agent answer a number's calls

     Each caller gets a session of their own with the agent config, and no call is dispatched to a
    worker. Replaces the agent the number had. The number must be attached; detaching or releasing it
    removes its agent.

    Args:
        e164 (str): The number in +15551234567 form.
        body (SetNumberAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | PhoneNumber
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
    body: SetNumberAgentRequest,
) -> Response[ErrorResponse | PhoneNumber]:
    """Have an agent answer a number's calls

     Each caller gets a session of their own with the agent config, and no call is dispatched to a
    worker. Replaces the agent the number had. The number must be attached; detaching or releasing it
    removes its agent.

    Args:
        e164 (str): The number in +15551234567 form.
        body (SetNumberAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | PhoneNumber]
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
    body: SetNumberAgentRequest,
) -> ErrorResponse | PhoneNumber | None:
    """Have an agent answer a number's calls

     Each caller gets a session of their own with the agent config, and no call is dispatched to a
    worker. Replaces the agent the number had. The number must be attached; detaching or releasing it
    removes its agent.

    Args:
        e164 (str): The number in +15551234567 form.
        body (SetNumberAgentRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | PhoneNumber
    """

    return (
        await asyncio_detailed(
            e164=e164,
            client=client,
            body=body,
        )
    ).parsed
