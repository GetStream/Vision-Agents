from http import HTTPStatus
from typing import Any
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.candidate import Candidate
from ...models.error import Error
from ...models.modality import Modality
from ...types import UNSET, Response, Unset


def _get_kwargs(
    modality: Modality,
    target: str,
    *,
    language: list[str] | Unset = UNSET,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    json_language: list[str] | Unset = UNSET
    if not isinstance(language, Unset):
        json_language = language

    params["language"] = json_language

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/{modality}/routes/{target}".format(
            modality=quote(str(modality), safe=""),
            target=quote(str(target), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Error | list[Candidate] | None:
    if response.status_code == 200:
        response_200 = []
        _response_200 = response.json()
        for response_200_item_data in _response_200:
            response_200_item = Candidate.from_dict(response_200_item_data)

            response_200.append(response_200_item)

        return response_200

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
) -> Response[Error | list[Candidate]]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    modality: Modality,
    target: str,
    *,
    client: AuthenticatedClient | Client,
    language: list[str] | Unset = UNSET,
) -> Response[Error | list[Candidate]]:
    """Resolve a provider name or capability shortcut to a ranked candidate list

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory and phone are recorded but not routed, since there is one memory store
            and one vendor per number, so the provider paths do not serve them while the statistics
            paths do.
             Example: tts.
        target (str):
        language (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Candidate]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        target=target,
        language=language,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    modality: Modality,
    target: str,
    *,
    client: AuthenticatedClient | Client,
    language: list[str] | Unset = UNSET,
) -> Error | list[Candidate] | None:
    """Resolve a provider name or capability shortcut to a ranked candidate list

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory and phone are recorded but not routed, since there is one memory store
            and one vendor per number, so the provider paths do not serve them while the statistics
            paths do.
             Example: tts.
        target (str):
        language (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Candidate]
    """

    return sync_detailed(
        modality=modality,
        target=target,
        client=client,
        language=language,
    ).parsed


async def asyncio_detailed(
    modality: Modality,
    target: str,
    *,
    client: AuthenticatedClient | Client,
    language: list[str] | Unset = UNSET,
) -> Response[Error | list[Candidate]]:
    """Resolve a provider name or capability shortcut to a ranked candidate list

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory and phone are recorded but not routed, since there is one memory store
            and one vendor per number, so the provider paths do not serve them while the statistics
            paths do.
             Example: tts.
        target (str):
        language (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Error | list[Candidate]]
    """

    kwargs = _get_kwargs(
        modality=modality,
        target=target,
        language=language,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    modality: Modality,
    target: str,
    *,
    client: AuthenticatedClient | Client,
    language: list[str] | Unset = UNSET,
) -> Error | list[Candidate] | None:
    """Resolve a provider name or capability shortcut to a ranked candidate list

    Args:
        modality (Modality): What kind of work was done. The first three are routed across
            providers. Memory and phone are recorded but not routed, since there is one memory store
            and one vendor per number, so the provider paths do not serve them while the statistics
            paths do.
             Example: tts.
        target (str):
        language (list[str] | Unset):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Error | list[Candidate]
    """

    return (
        await asyncio_detailed(
            modality=modality,
            target=target,
            client=client,
            language=language,
        )
    ).parsed
