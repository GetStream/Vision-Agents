from http import HTTPStatus
from typing import Any

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...models.sync_agent_files_request import SyncAgentFilesRequest
from ...models.sync_agent_result import SyncAgentResult
from ...types import Response


def _get_kwargs(
    *,
    body: SyncAgentFilesRequest,
) -> dict[str, Any]:
    headers: dict[str, Any] = {}

    _kwargs: dict[str, Any] = {
        "method": "post",
        "url": "/v1/agents/sync/files",
    }

    _kwargs["json"] = body.to_dict()

    headers["Content-Type"] = "application/json"

    _kwargs["headers"] = headers
    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> ErrorResponse | SyncAgentResult | None:
    if response.status_code == 200:
        response_200 = SyncAgentResult.from_dict(response.json())

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

    if response.status_code == 409:
        response_409 = ErrorResponse.from_dict(response.json())

        return response_409

    if response.status_code == 413:
        response_413 = ErrorResponse.from_dict(response.json())

        return response_413

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
) -> Response[ErrorResponse | SyncAgentResult]:
    return Response(
        status_code=HTTPStatus(response.status_code),
        content=response.content,
        headers=response.headers,
        parsed=_parse_response(client=client, response=response),
    )


def sync_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentFilesRequest,
) -> Response[ErrorResponse | SyncAgentResult]:
    """Store an agent directory sent as its files

     POST /v1/agents/sync for a client that sends the directory's files as they are and leaves reading
    them to the router, so a CLI, an SDK and the dashboard all mean the same thing by one directory. It
    stores what that sync stores, with the same answer.

    A key agent.yaml does not have, a retired one or a value of the wrong type is refused (400
    `invalid_agent_directory`) with every problem at its file, line and column in the error's
    `problems`. A connector's connection is found among the app's own by id or by label; one the app
    does not have is left out of the agent, stored in `left_out_connectors` and named in a warning,
    unless the connector is required, which is refused. agent.yaml's `ui` is compiled into
    `ui_instructions` here.

    `dry_run` answers with what the sync would store and writes nothing, which is how to show what an
    upload would change before applying it. `GET /v1/agents/configs/{id}/directory` is the reverse: the
    stored agent as files.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentFilesRequest): An agent directory as its files, which the router reads, so
            every client's directory means the same thing. The fields from instructions down are what
            a program sets in code beside the directory: each wins over what the files say, and is
            part of the hash.

            A request sends at most 1024 files, and the directory among them holds at most 1048576
            bytes a file and 8388608 bytes in all.

            The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256,
            in hex, of a run of parts, each written as its length in bytes in decimal, a colon and the
            part. The parts are, for every file of the directory in path order by byte, its path and
            its content; then, for each field from instructions down that is sent, its name and its
            value, in this order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts,
            sts and voice as sent; greeting.text and greeting.mode; video.source and video.max_frames
            in decimal; tags.<key> for each tag in key order; skills with the number of skills in
            decimal, then for each skill in name order skills.<name>.description, .instructions,
            .deadline_ms in decimal, .capture_video as true or false, and .files.<path> for each file
            in path order; then, for a directory whose agent.yaml has a ui key, ui and the version of
            the router's UI compiler, so a router that compiles ui differently syncs the directory
            again. A file sent that is no part of the directory is no part of the hash, so every
            client's sync of one folder has one hash. While a connector is left out sync_hash is
            another hash, so the sync that finds its connection is not taken for the one that left it
            out.

            What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the
            voice, the mode, the region, a greeting, the sandbox, connectors, tools, channels,
            chat_channels, scheduling, floor_control, agents, video and dispatch are left as stored.
            calls is always written whole: a call tool the file does not name goes back to its
            default. keyterms and tags are left as stored when the file has none, an empty list or
            mapping included, so a file cannot clear them. instructions, guardrail, the skills the
            config names, its UI instructions and its knowledge base are the directory's: a file taken
            out is taken out of the agent. simulations are left alone without a simulations file, and
            are exactly what the files declare with one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SyncAgentResult]
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
    body: SyncAgentFilesRequest,
) -> ErrorResponse | SyncAgentResult | None:
    """Store an agent directory sent as its files

     POST /v1/agents/sync for a client that sends the directory's files as they are and leaves reading
    them to the router, so a CLI, an SDK and the dashboard all mean the same thing by one directory. It
    stores what that sync stores, with the same answer.

    A key agent.yaml does not have, a retired one or a value of the wrong type is refused (400
    `invalid_agent_directory`) with every problem at its file, line and column in the error's
    `problems`. A connector's connection is found among the app's own by id or by label; one the app
    does not have is left out of the agent, stored in `left_out_connectors` and named in a warning,
    unless the connector is required, which is refused. agent.yaml's `ui` is compiled into
    `ui_instructions` here.

    `dry_run` answers with what the sync would store and writes nothing, which is how to show what an
    upload would change before applying it. `GET /v1/agents/configs/{id}/directory` is the reverse: the
    stored agent as files.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentFilesRequest): An agent directory as its files, which the router reads, so
            every client's directory means the same thing. The fields from instructions down are what
            a program sets in code beside the directory: each wins over what the files say, and is
            part of the hash.

            A request sends at most 1024 files, and the directory among them holds at most 1048576
            bytes a file and 8388608 bytes in all.

            The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256,
            in hex, of a run of parts, each written as its length in bytes in decimal, a colon and the
            part. The parts are, for every file of the directory in path order by byte, its path and
            its content; then, for each field from instructions down that is sent, its name and its
            value, in this order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts,
            sts and voice as sent; greeting.text and greeting.mode; video.source and video.max_frames
            in decimal; tags.<key> for each tag in key order; skills with the number of skills in
            decimal, then for each skill in name order skills.<name>.description, .instructions,
            .deadline_ms in decimal, .capture_video as true or false, and .files.<path> for each file
            in path order; then, for a directory whose agent.yaml has a ui key, ui and the version of
            the router's UI compiler, so a router that compiles ui differently syncs the directory
            again. A file sent that is no part of the directory is no part of the hash, so every
            client's sync of one folder has one hash. While a connector is left out sync_hash is
            another hash, so the sync that finds its connection is not taken for the one that left it
            out.

            What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the
            voice, the mode, the region, a greeting, the sandbox, connectors, tools, channels,
            chat_channels, scheduling, floor_control, agents, video and dispatch are left as stored.
            calls is always written whole: a call tool the file does not name goes back to its
            default. keyterms and tags are left as stored when the file has none, an empty list or
            mapping included, so a file cannot clear them. instructions, guardrail, the skills the
            config names, its UI instructions and its knowledge base are the directory's: a file taken
            out is taken out of the agent. simulations are left alone without a simulations file, and
            are exactly what the files declare with one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SyncAgentResult
    """

    return sync_detailed(
        client=client,
        body=body,
    ).parsed


async def asyncio_detailed(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentFilesRequest,
) -> Response[ErrorResponse | SyncAgentResult]:
    """Store an agent directory sent as its files

     POST /v1/agents/sync for a client that sends the directory's files as they are and leaves reading
    them to the router, so a CLI, an SDK and the dashboard all mean the same thing by one directory. It
    stores what that sync stores, with the same answer.

    A key agent.yaml does not have, a retired one or a value of the wrong type is refused (400
    `invalid_agent_directory`) with every problem at its file, line and column in the error's
    `problems`. A connector's connection is found among the app's own by id or by label; one the app
    does not have is left out of the agent, stored in `left_out_connectors` and named in a warning,
    unless the connector is required, which is refused. agent.yaml's `ui` is compiled into
    `ui_instructions` here.

    `dry_run` answers with what the sync would store and writes nothing, which is how to show what an
    upload would change before applying it. `GET /v1/agents/configs/{id}/directory` is the reverse: the
    stored agent as files.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentFilesRequest): An agent directory as its files, which the router reads, so
            every client's directory means the same thing. The fields from instructions down are what
            a program sets in code beside the directory: each wins over what the files say, and is
            part of the hash.

            A request sends at most 1024 files, and the directory among them holds at most 1048576
            bytes a file and 8388608 bytes in all.

            The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256,
            in hex, of a run of parts, each written as its length in bytes in decimal, a colon and the
            part. The parts are, for every file of the directory in path order by byte, its path and
            its content; then, for each field from instructions down that is sent, its name and its
            value, in this order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts,
            sts and voice as sent; greeting.text and greeting.mode; video.source and video.max_frames
            in decimal; tags.<key> for each tag in key order; skills with the number of skills in
            decimal, then for each skill in name order skills.<name>.description, .instructions,
            .deadline_ms in decimal, .capture_video as true or false, and .files.<path> for each file
            in path order; then, for a directory whose agent.yaml has a ui key, ui and the version of
            the router's UI compiler, so a router that compiles ui differently syncs the directory
            again. A file sent that is no part of the directory is no part of the hash, so every
            client's sync of one folder has one hash. While a connector is left out sync_hash is
            another hash, so the sync that finds its connection is not taken for the one that left it
            out.

            What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the
            voice, the mode, the region, a greeting, the sandbox, connectors, tools, channels,
            chat_channels, scheduling, floor_control, agents, video and dispatch are left as stored.
            calls is always written whole: a call tool the file does not name goes back to its
            default. keyterms and tags are left as stored when the file has none, an empty list or
            mapping included, so a file cannot clear them. instructions, guardrail, the skills the
            config names, its UI instructions and its knowledge base are the directory's: a file taken
            out is taken out of the agent. simulations are left alone without a simulations file, and
            are exactly what the files declare with one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[ErrorResponse | SyncAgentResult]
    """

    kwargs = _get_kwargs(
        body=body,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    *,
    client: AuthenticatedClient | Client,
    body: SyncAgentFilesRequest,
) -> ErrorResponse | SyncAgentResult | None:
    """Store an agent directory sent as its files

     POST /v1/agents/sync for a client that sends the directory's files as they are and leaves reading
    them to the router, so a CLI, an SDK and the dashboard all mean the same thing by one directory. It
    stores what that sync stores, with the same answer.

    A key agent.yaml does not have, a retired one or a value of the wrong type is refused (400
    `invalid_agent_directory`) with every problem at its file, line and column in the error's
    `problems`. A connector's connection is found among the app's own by id or by label; one the app
    does not have is left out of the agent, stored in `left_out_connectors` and named in a warning,
    unless the connector is required, which is refused. agent.yaml's `ui` is compiled into
    `ui_instructions` here.

    `dry_run` answers with what the sync would store and writes nothing, which is how to show what an
    upload would change before applying it. `GET /v1/agents/configs/{id}/directory` is the reverse: the
    stored agent as files.

    Server-side only: it needs a server-side token, so it cannot be reached from an end user's device.

    Args:
        body (SyncAgentFilesRequest): An agent directory as its files, which the router reads, so
            every client's directory means the same thing. The fields from instructions down are what
            a program sets in code beside the directory: each wins over what the files say, and is
            part of the hash.

            A request sends at most 1024 files, and the directory among them holds at most 1048576
            bytes a file and 8388608 bytes in all.

            The hash of a sync is the router's own, answered as the config's sync_hash: the SHA-256,
            in hex, of a run of parts, each written as its length in bytes in decimal, a colon and the
            part. The parts are, for every file of the directory in path order by byte, its path and
            its content; then, for each field from instructions down that is sent, its name and its
            value, in this order: instructions, guardrail, harness, subagent, sandbox, llm, stt, tts,
            sts and voice as sent; greeting.text and greeting.mode; video.source and video.max_frames
            in decimal; tags.<key> for each tag in key order; skills with the number of skills in
            decimal, then for each skill in name order skills.<name>.description, .instructions,
            .deadline_ms in decimal, .capture_video as true or false, and .files.<path> for each file
            in path order; then, for a directory whose agent.yaml has a ui key, ui and the version of
            the router's UI compiler, so a router that compiles ui differently syncs the directory
            again. A file sent that is no part of the directory is no part of the hash, so every
            client's sync of one folder has one hash. While a connector is left out sync_hash is
            another hash, so the sync that finds its connection is not taken for the one that left it
            out.

            What a sync does with a setting agent.yaml leaves out depends on the setting. A model, the
            voice, the mode, the region, a greeting, the sandbox, connectors, tools, channels,
            chat_channels, scheduling, floor_control, agents, video and dispatch are left as stored.
            calls is always written whole: a call tool the file does not name goes back to its
            default. keyterms and tags are left as stored when the file has none, an empty list or
            mapping included, so a file cannot clear them. instructions, guardrail, the skills the
            config names, its UI instructions and its knowledge base are the directory's: a file taken
            out is taken out of the agent. simulations are left alone without a simulations file, and
            are exactly what the files declare with one.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        ErrorResponse | SyncAgentResult
    """

    return (
        await asyncio_detailed(
            client=client,
            body=body,
        )
    ).parsed
