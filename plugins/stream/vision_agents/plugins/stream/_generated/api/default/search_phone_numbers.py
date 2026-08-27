from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error import Error
from ...models.number_search_result import NumberSearchResult
from ...models.phone_capability import PhoneCapability
from ...models.phone_number_type import PhoneNumberType
from ...types import UNSET, Response, Unset


def _get_kwargs(
    *,
    vendor: str | Unset = UNSET,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    prefix: str | Unset = UNSET,
    locality: str | Unset = UNSET,
    administrative_area: str | Unset = UNSET,
    number_type: PhoneNumberType | Unset = UNSET,
    features: list[PhoneCapability] | Unset = UNSET,
    limit: int | Unset = 10,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["vendor"] = vendor

    params["country"] = country

    params["area_code"] = area_code

    params["contains"] = contains

    params["prefix"] = prefix

    params["locality"] = locality

    params["administrative_area"] = administrative_area

    json_number_type: str | Unset = UNSET
    if not isinstance(number_type, Unset):
        json_number_type = number_type.value

    params["number_type"] = json_number_type

    json_features: list[str] | Unset = UNSET
    if not isinstance(features, Unset):
        json_features = []
        for features_item_data in features:
            features_item = features_item_data.value
            json_features.append(features_item)

    params["features"] = json_features

    params["limit"] = limit

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/phone/numbers/available",
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | NumberSearchResult | None:
    if response.status_code == 200:
        response_200 = NumberSearchResult.from_dict(response.json())

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
) -> Response[Error | NumberSearchResult]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    vendor: str | Unset = UNSET,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    prefix: str | Unset = UNSET,
    locality: str | Unset = UNSET,
    administrative_area: str | Unset = UNSET,
    number_type: PhoneNumberType | Unset = UNSET,
    features: list[PhoneCapability] | Unset = UNSET,
    limit: int | Unset = 10,
) -> Response[Error | NumberSearchResult]:
    """Search for numbers to buy, at one vendor or all of them

     Naming a vendor searches only that one. Leaving it out asks every vendor that has its credentials,
    at once, and merges what they offer cheapest first. Vendors do not agree on how a search can be
    narrowed, so one whose API cannot express a filter is reported in `skipped` rather than asked
    without it, which would answer a search for one place with numbers from another.

    Args:
        vendor (str | Unset):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        prefix (str | Unset):
        locality (str | Unset):
        administrative_area (str | Unset):
        number_type (PhoneNumberType | Unset): What kind of number it is, which decides who pays
            for the call.
        features (list[PhoneCapability] | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | NumberSearchResult]
    """

    kwargs = _get_kwargs(
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        prefix=prefix,
        locality=locality,
        administrative_area=administrative_area,
        number_type=number_type,
        features=features,
        limit=limit,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    *,
    client: AuthenticatedClient | Client,
    vendor: str | Unset = UNSET,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    prefix: str | Unset = UNSET,
    locality: str | Unset = UNSET,
    administrative_area: str | Unset = UNSET,
    number_type: PhoneNumberType | Unset = UNSET,
    features: list[PhoneCapability] | Unset = UNSET,
    limit: int | Unset = 10,
) -> Error | NumberSearchResult | None:
    """Search for numbers to buy, at one vendor or all of them

     Naming a vendor searches only that one. Leaving it out asks every vendor that has its credentials,
    at once, and merges what they offer cheapest first. Vendors do not agree on how a search can be
    narrowed, so one whose API cannot express a filter is reported in `skipped` rather than asked
    without it, which would answer a search for one place with numbers from another.

    Args:
        vendor (str | Unset):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        prefix (str | Unset):
        locality (str | Unset):
        administrative_area (str | Unset):
        number_type (PhoneNumberType | Unset): What kind of number it is, which decides who pays
            for the call.
        features (list[PhoneCapability] | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | NumberSearchResult
    """

    return sync_detailed(
        client=client,
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        prefix=prefix,
        locality=locality,
        administrative_area=administrative_area,
        number_type=number_type,
        features=features,
        limit=limit,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    vendor: str | Unset = UNSET,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    prefix: str | Unset = UNSET,
    locality: str | Unset = UNSET,
    administrative_area: str | Unset = UNSET,
    number_type: PhoneNumberType | Unset = UNSET,
    features: list[PhoneCapability] | Unset = UNSET,
    limit: int | Unset = 10,
) -> Response[Error | NumberSearchResult]:
    """Search for numbers to buy, at one vendor or all of them

     Naming a vendor searches only that one. Leaving it out asks every vendor that has its credentials,
    at once, and merges what they offer cheapest first. Vendors do not agree on how a search can be
    narrowed, so one whose API cannot express a filter is reported in `skipped` rather than asked
    without it, which would answer a search for one place with numbers from another.

    Args:
        vendor (str | Unset):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        prefix (str | Unset):
        locality (str | Unset):
        administrative_area (str | Unset):
        number_type (PhoneNumberType | Unset): What kind of number it is, which decides who pays
            for the call.
        features (list[PhoneCapability] | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | NumberSearchResult]
    """

    kwargs = _get_kwargs(
        vendor=vendor,
        country=country,
        area_code=area_code,
        contains=contains,
        prefix=prefix,
        locality=locality,
        administrative_area=administrative_area,
        number_type=number_type,
        features=features,
        limit=limit,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    vendor: str | Unset = UNSET,
    country: str,
    area_code: str | Unset = UNSET,
    contains: str | Unset = UNSET,
    prefix: str | Unset = UNSET,
    locality: str | Unset = UNSET,
    administrative_area: str | Unset = UNSET,
    number_type: PhoneNumberType | Unset = UNSET,
    features: list[PhoneCapability] | Unset = UNSET,
    limit: int | Unset = 10,
) -> Error | NumberSearchResult | None:
    """Search for numbers to buy, at one vendor or all of them

     Naming a vendor searches only that one. Leaving it out asks every vendor that has its credentials,
    at once, and merges what they offer cheapest first. Vendors do not agree on how a search can be
    narrowed, so one whose API cannot express a filter is reported in `skipped` rather than asked
    without it, which would answer a search for one place with numbers from another.

    Args:
        vendor (str | Unset):
        country (str):
        area_code (str | Unset):
        contains (str | Unset):
        prefix (str | Unset):
        locality (str | Unset):
        administrative_area (str | Unset):
        number_type (PhoneNumberType | Unset): What kind of number it is, which decides who pays
            for the call.
        features (list[PhoneCapability] | Unset):
        limit (int | Unset):  Default: 10.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | NumberSearchResult
    """

    return (
        await asyncio_detailed(
            client=client,
            vendor=vendor,
            country=country,
            area_code=area_code,
            contains=contains,
            prefix=prefix,
            locality=locality,
            administrative_area=administrative_area,
            number_type=number_type,
            features=features,
            limit=limit,
        )
    ).parsed
