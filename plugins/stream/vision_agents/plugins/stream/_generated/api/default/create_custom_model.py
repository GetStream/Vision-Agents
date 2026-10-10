from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.custom_model import CustomModel
from ...models.custom_model_request import CustomModelRequest
from ...models.error_response import ErrorResponse
from ...types import Response


def _get_kwargs(
    *,
    body: CustomModelRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/models",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> CustomModel | ErrorResponse | None:
    if response.status_code == 201:
        response_201 = CustomModel.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 501:
        response_501 = ErrorResponse.from_dict(response.json())

        return response_501

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[CustomModel | ErrorResponse]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CustomModelRequest,
) -> Response[CustomModel | ErrorResponse]:
    """Add a model of the customer's own

     A model the customer serves behind an OpenAI-compatible chat completions endpoint: a fine-tune on
    Baseten, a vLLM or SGLang deployment, a provider the router does not route. A router config or
    session names it as `custom/<name>`, and the router calls it as it calls the models it routes. A
    shared router only dials public https endpoints.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomModelRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CustomModel | ErrorResponse]
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
    body: CustomModelRequest,
) -> CustomModel | ErrorResponse | None:
    """Add a model of the customer's own

     A model the customer serves behind an OpenAI-compatible chat completions endpoint: a fine-tune on
    Baseten, a vLLM or SGLang deployment, a provider the router does not route. A router config or
    session names it as `custom/<name>`, and the router calls it as it calls the models it routes. A
    shared router only dials public https endpoints.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomModelRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CustomModel | ErrorResponse
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CustomModelRequest,
) -> Response[CustomModel | ErrorResponse]:
    """Add a model of the customer's own

     A model the customer serves behind an OpenAI-compatible chat completions endpoint: a fine-tune on
    Baseten, a vLLM or SGLang deployment, a provider the router does not route. A router config or
    session names it as `custom/<name>`, and the router calls it as it calls the models it routes. A
    shared router only dials public https endpoints.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomModelRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[CustomModel | ErrorResponse]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CustomModelRequest,
) -> CustomModel | ErrorResponse | None:
    """Add a model of the customer's own

     A model the customer serves behind an OpenAI-compatible chat completions endpoint: a fine-tune on
    Baseten, a vLLM or SGLang deployment, a provider the router does not route. A router config or
    session names it as `custom/<name>`, and the router calls it as it calls the models it routes. A
    shared router only dials public https endpoints.
    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (CustomModelRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        CustomModel | ErrorResponse
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
