from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.prices import Prices
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    modality: str | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["modality"] = modality

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/prices",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | Prices | None:
    if response.status_code == 200:
        response_200 = Prices.from_dict(response.json())

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
) -> Response[ErrorResponse | Prices]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    modality: str | Unset = UNSET,
) -> Response[ErrorResponse | Prices]:
    """Read the price list

     Every item this router prices, at the rates a customer is charged, which are what every answer's
    X-Stream-Cost-Micros is worked out from: a rate is USD for every per units of its meter, and a long
    prompt is priced whole at its tier's rates. Then what the changes scheduled later will set, soonest
    first, with their reason, an item they withdraw included. The list is the one in memory, so reading
    it costs no query; a change made on another router shows within 30 seconds. Prices are the same for
    everyone, so any caller may read them.

    Args:
        modality (str | Unset): Only this modality's items, such as llm, stt, tts, sms or phone.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Prices]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    modality: str | Unset = UNSET,
) -> ErrorResponse | Prices | None:
    """Read the price list

     Every item this router prices, at the rates a customer is charged, which are what every answer's
    X-Stream-Cost-Micros is worked out from: a rate is USD for every per units of its meter, and a long
    prompt is priced whole at its tier's rates. Then what the changes scheduled later will set, soonest
    first, with their reason, an item they withdraw included. The list is the one in memory, so reading
    it costs no query; a change made on another router shows within 30 seconds. Prices are the same for
    everyone, so any caller may read them.

    Args:
        modality (str | Unset): Only this modality's items, such as llm, stt, tts, sms or phone.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Prices
    """

    return sync_detailed(
        client=client,
        modality=modality,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    modality: str | Unset = UNSET,
) -> Response[ErrorResponse | Prices]:
    """Read the price list

     Every item this router prices, at the rates a customer is charged, which are what every answer's
    X-Stream-Cost-Micros is worked out from: a rate is USD for every per units of its meter, and a long
    prompt is priced whole at its tier's rates. Then what the changes scheduled later will set, soonest
    first, with their reason, an item they withdraw included. The list is the one in memory, so reading
    it costs no query; a change made on another router shows within 30 seconds. Prices are the same for
    everyone, so any caller may read them.

    Args:
        modality (str | Unset): Only this modality's items, such as llm, stt, tts, sms or phone.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Prices]
    """

    kwargs = _get_kwargs(
        modality=modality,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    modality: str | Unset = UNSET,
) -> ErrorResponse | Prices | None:
    """Read the price list

     Every item this router prices, at the rates a customer is charged, which are what every answer's
    X-Stream-Cost-Micros is worked out from: a rate is USD for every per units of its meter, and a long
    prompt is priced whole at its tier's rates. Then what the changes scheduled later will set, soonest
    first, with their reason, an item they withdraw included. The list is the one in memory, so reading
    it costs no query; a change made on another router shows within 30 seconds. Prices are the same for
    everyone, so any caller may read them.

    Args:
        modality (str | Unset): Only this modality's items, such as llm, stt, tts, sms or phone.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Prices
    """

    return (
        await asyncio_detailed(
            client=client,
            modality=modality,
        )
    ).parsed
