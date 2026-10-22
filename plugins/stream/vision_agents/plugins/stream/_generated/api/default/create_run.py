from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.create_run_request import CreateRunRequest
from ...models.error_response import ErrorResponse
from ...models.run import Run
from ...types import Response


def _get_kwargs(
    *,
    body: CreateRunRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/runs",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | Run | None:
    if response.status_code == 200:
        response_200 = Run.from_dict(response.json())

        return response_200

    if response.status_code == 400:
        response_400 = ErrorResponse.from_dict(response.json())

        return response_400

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 403:
        response_403 = ErrorResponse.from_dict(response.json())

        return response_403

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 429:
        response_429 = ErrorResponse.from_dict(response.json())

        return response_429

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if response.status_code == 503:
        response_503 = ErrorResponse.from_dict(response.json())

        return response_503

    if response.status_code == 504:
        response_504 = ErrorResponse.from_dict(response.json())

        return response_504

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[ErrorResponse | Run]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateRunRequest,
) -> Response[ErrorResponse | Run]:
    """Run an agent on one input and wait for its answer

     Runs an agent config as a headless text session on one input, waits until the agent has nothing left
    to do, and answers with the last thing it said. The session is incognito and ends with the request,
    so nothing is kept and no run sees another's input.
    The instructions are the caller's, trusted, and added to the config's for this run alone. A config
    whose role is agent_moderator is then told the answer format the chat review worker parses, after
    both, so no edit to the agent can change it.
    The input is handed to the agent as JSON data, never as instructions.
    Images, at most 8, go with the input in the order sent, so the input can refer to them by their
    place in the list. The agent's vision skill looks at them, or its conversation model when that can
    see; an image that is no HTTP(S) URL or image data URI, or more than 8, is a 400 invalid_request.
    A config that does not start a session, or that cannot see a run's images, is a 400 invalid_config,
    which retrying does not fix, a spent budget a 403 budget_exceeded, which waiting does not fix, an
    app with as many runs in flight as it may have (ROUTER_RUNS_CONCURRENT_PER_APP, 20 by default) a 429
    concurrency_limited with Retry-After, a model that failed or said nothing a 503 model_failed, and a
    run that outlasts its timeout a 504.

    Args:
        body (CreateRunRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Run]
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
    body: CreateRunRequest,
) -> ErrorResponse | Run | None:
    """Run an agent on one input and wait for its answer

     Runs an agent config as a headless text session on one input, waits until the agent has nothing left
    to do, and answers with the last thing it said. The session is incognito and ends with the request,
    so nothing is kept and no run sees another's input.
    The instructions are the caller's, trusted, and added to the config's for this run alone. A config
    whose role is agent_moderator is then told the answer format the chat review worker parses, after
    both, so no edit to the agent can change it.
    The input is handed to the agent as JSON data, never as instructions.
    Images, at most 8, go with the input in the order sent, so the input can refer to them by their
    place in the list. The agent's vision skill looks at them, or its conversation model when that can
    see; an image that is no HTTP(S) URL or image data URI, or more than 8, is a 400 invalid_request.
    A config that does not start a session, or that cannot see a run's images, is a 400 invalid_config,
    which retrying does not fix, a spent budget a 403 budget_exceeded, which waiting does not fix, an
    app with as many runs in flight as it may have (ROUTER_RUNS_CONCURRENT_PER_APP, 20 by default) a 429
    concurrency_limited with Retry-After, a model that failed or said nothing a 503 model_failed, and a
    run that outlasts its timeout a 504.

    Args:
        body (CreateRunRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Run
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: CreateRunRequest,
) -> Response[ErrorResponse | Run]:
    """Run an agent on one input and wait for its answer

     Runs an agent config as a headless text session on one input, waits until the agent has nothing left
    to do, and answers with the last thing it said. The session is incognito and ends with the request,
    so nothing is kept and no run sees another's input.
    The instructions are the caller's, trusted, and added to the config's for this run alone. A config
    whose role is agent_moderator is then told the answer format the chat review worker parses, after
    both, so no edit to the agent can change it.
    The input is handed to the agent as JSON data, never as instructions.
    Images, at most 8, go with the input in the order sent, so the input can refer to them by their
    place in the list. The agent's vision skill looks at them, or its conversation model when that can
    see; an image that is no HTTP(S) URL or image data URI, or more than 8, is a 400 invalid_request.
    A config that does not start a session, or that cannot see a run's images, is a 400 invalid_config,
    which retrying does not fix, a spent budget a 403 budget_exceeded, which waiting does not fix, an
    app with as many runs in flight as it may have (ROUTER_RUNS_CONCURRENT_PER_APP, 20 by default) a 429
    concurrency_limited with Retry-After, a model that failed or said nothing a 503 model_failed, and a
    run that outlasts its timeout a 504.

    Args:
        body (CreateRunRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | Run]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: CreateRunRequest,
) -> ErrorResponse | Run | None:
    """Run an agent on one input and wait for its answer

     Runs an agent config as a headless text session on one input, waits until the agent has nothing left
    to do, and answers with the last thing it said. The session is incognito and ends with the request,
    so nothing is kept and no run sees another's input.
    The instructions are the caller's, trusted, and added to the config's for this run alone. A config
    whose role is agent_moderator is then told the answer format the chat review worker parses, after
    both, so no edit to the agent can change it.
    The input is handed to the agent as JSON data, never as instructions.
    Images, at most 8, go with the input in the order sent, so the input can refer to them by their
    place in the list. The agent's vision skill looks at them, or its conversation model when that can
    see; an image that is no HTTP(S) URL or image data URI, or more than 8, is a 400 invalid_request.
    A config that does not start a session, or that cannot see a run's images, is a 400 invalid_config,
    which retrying does not fix, a spent budget a 403 budget_exceeded, which waiting does not fix, an
    app with as many runs in flight as it may have (ROUTER_RUNS_CONCURRENT_PER_APP, 20 by default) a 429
    concurrency_limited with Retry-After, a model that failed or said nothing a 503 model_failed, and a
    run that outlasts its timeout a 504.

    Args:
        body (CreateRunRequest):

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | Run
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
