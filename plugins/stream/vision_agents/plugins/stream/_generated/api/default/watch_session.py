from http import HTTPStatus
from typing import Any, cast
from urllib.parse import quote

import httpx

from ... import errors
from ...client import AuthenticatedClient, Client
from ...models.error_response import ErrorResponse
from ...types import UNSET, Response, Unset


def _get_kwargs(
    id: str,
    *,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
    replay_pending_tools: bool | Unset = False,
) -> dict[str, Any]:

    params: dict[str, Any] = {}

    params["interim"] = interim

    params["decisions"] = decisions

    params["replay_pending_tools"] = replay_pending_tools

    params = {k: v for k, v in params.items() if v is not UNSET and v is not None}

    _kwargs: dict[str, Any] = {
        "method": "get",
        "url": "/v1/agents/sessions/{id}/events".format(
            id=quote(str(id), safe=""),
        ),
        "params": params,
    }

    return _kwargs


def _parse_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Any | ErrorResponse | None:
    if response.status_code == 101:
        response_101 = cast(Any, None)
        return response_101

    if response.status_code == 401:
        response_401 = ErrorResponse.from_dict(response.json())

        return response_401

    if response.status_code == 404:
        response_404 = ErrorResponse.from_dict(response.json())

        return response_404

    if response.status_code == 500:
        response_500 = ErrorResponse.from_dict(response.json())

        return response_500

    if client.raise_on_unexpected_status:
        raise errors.UnexpectedStatus(response.status_code, response.content)
    else:
        return None


def _build_response(
    *, client: AuthenticatedClient | Client, response: httpx.Response
) -> Response[Any | ErrorResponse]:
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
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
    replay_pending_tools: bool | Unset = False,
) -> Response[Any | ErrorResponse]:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded` (pending_work remains true while tools or delegated work are outstanding), `spoke`,
    `turn`, `decision`, `delegated`, `task_settled` (files lists what the work's code handed back, each
    a name, mime_type, url and size, uploaded to a persistent conversation's channel and attached to the
    reply), `task_cancelled`, `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`,
    `backchannel`, `interrupted`, `overlap_decided`, `conversation_compacted`, `models_changed`, `error`
    and `left`.
    `connector_unavailable` names an optional connector binding the session opened without: name (its
    alias), connector_id and reason, one of no_selection, shared_session, caller_unverified,
    connection_unavailable, provider_mismatch, needs_reauthorization, credential_rejected (the provider
    rejected the token or key a bearer or api_key connection holds; only new credentials fix it, so no
    login is offered), not_connected, open_failed, tool_unavailable and selection_dropped (a fork's or a
    reopened chat's selection for an alias its config no longer declares). Every watcher is sent each
    one when it attaches.
    `connector_scope_required` says a connector tool call was refused because the caller's own
    connection lacks access the provider asked for (insufficient_scope or a claims challenge), and a
    step-up consent was begun for it: name (the binding's alias), connector_id, connection_id, scopes
    (what the provider asked for, empty for a claims challenge), authorization_id, launch_url,
    handoff_token and expires_at. A client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. The old grant keeps working until the step-up succeeds, and the same call
    works afterwards in the same session. While that step-up is open, calls refused for the same access
    send no second event.
    Persistent text sessions also emit `conversation_updated` with conversation_id and a complete
    message snapshot: id, command_id, question_id, role, text, state, response_started_at,
    state_started_at, finished_at, duration_ms, saved, persistence_error and attachments. Each
    tool_calling attachment has tool_call_id, name, title, status, phase, summary, immutable started_at,
    execution_started_at, finished_at and duration_ms. A plugin_authorization attachment asks the end
    user to connect a plugin the reply needed, with plugin_id, title, authorize_url, text, thumb_url and
    title_link: a client shows it as a button opening authorize_url. Once the user finishes that login
    the message is sent again with the attachment's status set to connected. A connector_authorization
    attachment asks the end user to connect a connector binding the reply needed with their own account,
    with name (the binding's alias), connector_id, connection_id, authorization_id, title, launch_url,
    handoff_token and expires_at: a client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. Once the user finishes that login the message is sent again with status
    connected and no handoff_token, and the agent carries on by itself. Activity states are thinking,
    queued, tools, writing, completed, failed and cancelled. tool_started includes tool_call_id, tool,
    turn_id and started_at, and pre_speech when the tool's connector binding sets one in its policy;
    tool_ran also includes tool_call_id.
    A respond command carrying command_id emits command_accepted with a nested command receipt
    (command_id, user_message_id, assistant_message_id, state, duplicate). Personal persistent text
    sessions require this ID. A retry with the same text returns the existing IDs without invoking the
    model again; reuse with different text emits an error. Commands with IDs currently accept text only.
    After restart an interrupted command is reported, not rerun.
    An `interrupt` command carrying `command_id` stops that command and emits `command_stopped` with its
    terminal receipt. A stop arriving after its command finished replays that command's receipt and
    leaves the command running now alone; an unknown command is reported as an error. Without
    `command_id` the frame stops whichever reply is current, which is what a caller with no command to
    name means by it.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    `replay_pending_tools=true` opts a durable tool host into replay of external tool calls still
    awaiting results in a live voice session. Completed, cancelled and timed-out requests are excluded
    at snapshot time. Replays retain their tool and turn IDs and may duplicate live delivery; the host
    must persist execution receipts and refuse to repeat uncertain writes. Ordinary status watchers
    should leave this disabled. Persistent text command recovery is unchanged.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`
    (optionally naming a `command_id`), `instructions` or `close` to act on the session. `instructions`
    is server-side only: from an end user's device it changes nothing and is answered with an `error`
    frame, `context` `command`, as `updateSession` refuses it. A `tool_call` is the only frame that must
    be answered: everything else is a report. Tool calls made by durable personal commands carry
    `command_id` and `turn_id`; their result must repeat both values so a result cannot be adopted by
    another command or turn.
    A call to a tool declared with an `approval` waits for a person. The client reports their answer
    with `tool_approval` (`tool_call_id`, `command_id`, `turn_id`, `allowed`, and optionally a `summary`
    shown when they declined), before it answers the call with `tool_result`.
    `tool_result.output` is a string, or an array of parts `[{type: text|image_url, ...}]`. An image has
    an `image_url` object containing `url` (HTTP(S) or data URI), optionally with `detail` of `auto`,
    `low` or `high`. One socket message is at most 5 MB.
    `respond` may carry `images: [{url, detail}]`. These schedule the vision skill; the conversation
    receives the question and later the findings, without raw images. Video capture uses task-correlated
    `get_video_frames` tool requests and `tool_result` replies. Frames are never attached automatically
    to conversational turns.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.
        replay_pending_tools (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        interim=interim,
        decisions=decisions,
        replay_pending_tools=replay_pending_tools,
    )

    response = client.get_httpx_client().request(
        **kwargs,
    )

    return _build_response(client=client, response=response)


def sync(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
    replay_pending_tools: bool | Unset = False,
) -> Any | ErrorResponse | None:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded` (pending_work remains true while tools or delegated work are outstanding), `spoke`,
    `turn`, `decision`, `delegated`, `task_settled` (files lists what the work's code handed back, each
    a name, mime_type, url and size, uploaded to a persistent conversation's channel and attached to the
    reply), `task_cancelled`, `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`,
    `backchannel`, `interrupted`, `overlap_decided`, `conversation_compacted`, `models_changed`, `error`
    and `left`.
    `connector_unavailable` names an optional connector binding the session opened without: name (its
    alias), connector_id and reason, one of no_selection, shared_session, caller_unverified,
    connection_unavailable, provider_mismatch, needs_reauthorization, credential_rejected (the provider
    rejected the token or key a bearer or api_key connection holds; only new credentials fix it, so no
    login is offered), not_connected, open_failed, tool_unavailable and selection_dropped (a fork's or a
    reopened chat's selection for an alias its config no longer declares). Every watcher is sent each
    one when it attaches.
    `connector_scope_required` says a connector tool call was refused because the caller's own
    connection lacks access the provider asked for (insufficient_scope or a claims challenge), and a
    step-up consent was begun for it: name (the binding's alias), connector_id, connection_id, scopes
    (what the provider asked for, empty for a claims challenge), authorization_id, launch_url,
    handoff_token and expires_at. A client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. The old grant keeps working until the step-up succeeds, and the same call
    works afterwards in the same session. While that step-up is open, calls refused for the same access
    send no second event.
    Persistent text sessions also emit `conversation_updated` with conversation_id and a complete
    message snapshot: id, command_id, question_id, role, text, state, response_started_at,
    state_started_at, finished_at, duration_ms, saved, persistence_error and attachments. Each
    tool_calling attachment has tool_call_id, name, title, status, phase, summary, immutable started_at,
    execution_started_at, finished_at and duration_ms. A plugin_authorization attachment asks the end
    user to connect a plugin the reply needed, with plugin_id, title, authorize_url, text, thumb_url and
    title_link: a client shows it as a button opening authorize_url. Once the user finishes that login
    the message is sent again with the attachment's status set to connected. A connector_authorization
    attachment asks the end user to connect a connector binding the reply needed with their own account,
    with name (the binding's alias), connector_id, connection_id, authorization_id, title, launch_url,
    handoff_token and expires_at: a client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. Once the user finishes that login the message is sent again with status
    connected and no handoff_token, and the agent carries on by itself. Activity states are thinking,
    queued, tools, writing, completed, failed and cancelled. tool_started includes tool_call_id, tool,
    turn_id and started_at, and pre_speech when the tool's connector binding sets one in its policy;
    tool_ran also includes tool_call_id.
    A respond command carrying command_id emits command_accepted with a nested command receipt
    (command_id, user_message_id, assistant_message_id, state, duplicate). Personal persistent text
    sessions require this ID. A retry with the same text returns the existing IDs without invoking the
    model again; reuse with different text emits an error. Commands with IDs currently accept text only.
    After restart an interrupted command is reported, not rerun.
    An `interrupt` command carrying `command_id` stops that command and emits `command_stopped` with its
    terminal receipt. A stop arriving after its command finished replays that command's receipt and
    leaves the command running now alone; an unknown command is reported as an error. Without
    `command_id` the frame stops whichever reply is current, which is what a caller with no command to
    name means by it.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    `replay_pending_tools=true` opts a durable tool host into replay of external tool calls still
    awaiting results in a live voice session. Completed, cancelled and timed-out requests are excluded
    at snapshot time. Replays retain their tool and turn IDs and may duplicate live delivery; the host
    must persist execution receipts and refuse to repeat uncertain writes. Ordinary status watchers
    should leave this disabled. Persistent text command recovery is unchanged.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`
    (optionally naming a `command_id`), `instructions` or `close` to act on the session. `instructions`
    is server-side only: from an end user's device it changes nothing and is answered with an `error`
    frame, `context` `command`, as `updateSession` refuses it. A `tool_call` is the only frame that must
    be answered: everything else is a report. Tool calls made by durable personal commands carry
    `command_id` and `turn_id`; their result must repeat both values so a result cannot be adopted by
    another command or turn.
    A call to a tool declared with an `approval` waits for a person. The client reports their answer
    with `tool_approval` (`tool_call_id`, `command_id`, `turn_id`, `allowed`, and optionally a `summary`
    shown when they declined), before it answers the call with `tool_result`.
    `tool_result.output` is a string, or an array of parts `[{type: text|image_url, ...}]`. An image has
    an `image_url` object containing `url` (HTTP(S) or data URI), optionally with `detail` of `auto`,
    `low` or `high`. One socket message is at most 5 MB.
    `respond` may carry `images: [{url, detail}]`. These schedule the vision skill; the conversation
    receives the question and later the findings, without raw images. Video capture uses task-correlated
    `get_video_frames` tool requests and `tool_result` replies. Frames are never attached automatically
    to conversational turns.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.
        replay_pending_tools (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return sync_detailed(
        id=id,
        client=client,
        interim=interim,
        decisions=decisions,
        replay_pending_tools=replay_pending_tools,
    ).parsed


async def asyncio_detailed(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
    replay_pending_tools: bool | Unset = False,
) -> Response[Any | ErrorResponse]:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded` (pending_work remains true while tools or delegated work are outstanding), `spoke`,
    `turn`, `decision`, `delegated`, `task_settled` (files lists what the work's code handed back, each
    a name, mime_type, url and size, uploaded to a persistent conversation's channel and attached to the
    reply), `task_cancelled`, `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`,
    `backchannel`, `interrupted`, `overlap_decided`, `conversation_compacted`, `models_changed`, `error`
    and `left`.
    `connector_unavailable` names an optional connector binding the session opened without: name (its
    alias), connector_id and reason, one of no_selection, shared_session, caller_unverified,
    connection_unavailable, provider_mismatch, needs_reauthorization, credential_rejected (the provider
    rejected the token or key a bearer or api_key connection holds; only new credentials fix it, so no
    login is offered), not_connected, open_failed, tool_unavailable and selection_dropped (a fork's or a
    reopened chat's selection for an alias its config no longer declares). Every watcher is sent each
    one when it attaches.
    `connector_scope_required` says a connector tool call was refused because the caller's own
    connection lacks access the provider asked for (insufficient_scope or a claims challenge), and a
    step-up consent was begun for it: name (the binding's alias), connector_id, connection_id, scopes
    (what the provider asked for, empty for a claims challenge), authorization_id, launch_url,
    handoff_token and expires_at. A client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. The old grant keeps working until the step-up succeeds, and the same call
    works afterwards in the same session. While that step-up is open, calls refused for the same access
    send no second event.
    Persistent text sessions also emit `conversation_updated` with conversation_id and a complete
    message snapshot: id, command_id, question_id, role, text, state, response_started_at,
    state_started_at, finished_at, duration_ms, saved, persistence_error and attachments. Each
    tool_calling attachment has tool_call_id, name, title, status, phase, summary, immutable started_at,
    execution_started_at, finished_at and duration_ms. A plugin_authorization attachment asks the end
    user to connect a plugin the reply needed, with plugin_id, title, authorize_url, text, thumb_url and
    title_link: a client shows it as a button opening authorize_url. Once the user finishes that login
    the message is sent again with the attachment's status set to connected. A connector_authorization
    attachment asks the end user to connect a connector binding the reply needed with their own account,
    with name (the binding's alias), connector_id, connection_id, authorization_id, title, launch_url,
    handoff_token and expires_at: a client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. Once the user finishes that login the message is sent again with status
    connected and no handoff_token, and the agent carries on by itself. Activity states are thinking,
    queued, tools, writing, completed, failed and cancelled. tool_started includes tool_call_id, tool,
    turn_id and started_at, and pre_speech when the tool's connector binding sets one in its policy;
    tool_ran also includes tool_call_id.
    A respond command carrying command_id emits command_accepted with a nested command receipt
    (command_id, user_message_id, assistant_message_id, state, duplicate). Personal persistent text
    sessions require this ID. A retry with the same text returns the existing IDs without invoking the
    model again; reuse with different text emits an error. Commands with IDs currently accept text only.
    After restart an interrupted command is reported, not rerun.
    An `interrupt` command carrying `command_id` stops that command and emits `command_stopped` with its
    terminal receipt. A stop arriving after its command finished replays that command's receipt and
    leaves the command running now alone; an unknown command is reported as an error. Without
    `command_id` the frame stops whichever reply is current, which is what a caller with no command to
    name means by it.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    `replay_pending_tools=true` opts a durable tool host into replay of external tool calls still
    awaiting results in a live voice session. Completed, cancelled and timed-out requests are excluded
    at snapshot time. Replays retain their tool and turn IDs and may duplicate live delivery; the host
    must persist execution receipts and refuse to repeat uncertain writes. Ordinary status watchers
    should leave this disabled. Persistent text command recovery is unchanged.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`
    (optionally naming a `command_id`), `instructions` or `close` to act on the session. `instructions`
    is server-side only: from an end user's device it changes nothing and is answered with an `error`
    frame, `context` `command`, as `updateSession` refuses it. A `tool_call` is the only frame that must
    be answered: everything else is a report. Tool calls made by durable personal commands carry
    `command_id` and `turn_id`; their result must repeat both values so a result cannot be adopted by
    another command or turn.
    A call to a tool declared with an `approval` waits for a person. The client reports their answer
    with `tool_approval` (`tool_call_id`, `command_id`, `turn_id`, `allowed`, and optionally a `summary`
    shown when they declined), before it answers the call with `tool_result`.
    `tool_result.output` is a string, or an array of parts `[{type: text|image_url, ...}]`. An image has
    an `image_url` object containing `url` (HTTP(S) or data URI), optionally with `detail` of `auto`,
    `low` or `high`. One socket message is at most 5 MB.
    `respond` may carry `images: [{url, detail}]`. These schedule the vision skill; the conversation
    receives the question and later the findings, without raw images. Video capture uses task-correlated
    `get_video_frames` tool requests and `tool_result` replies. Frames are never attached automatically
    to conversational turns.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.
        replay_pending_tools (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Response[Any | ErrorResponse]
    """

    kwargs = _get_kwargs(
        id=id,
        interim=interim,
        decisions=decisions,
        replay_pending_tools=replay_pending_tools,
    )

    response = await client.get_async_httpx_client().request(**kwargs)

    return _build_response(client=client, response=response)


async def asyncio(
    id: str,
    *,
    client: AuthenticatedClient | Client,
    interim: bool | Unset = False,
    decisions: bool | Unset = True,
    replay_pending_tools: bool | Unset = False,
) -> Any | ErrorResponse | None:
    """Watch the conversation and answer the model's tool calls

     A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects carrying a
    `type` and the fields of that event.
    The server sends what the conversation did: `joined`, `heard`, `responding`, `response_delta`,
    `responded` (pending_work remains true while tools or delegated work are outstanding), `spoke`,
    `turn`, `decision`, `delegated`, `task_settled` (files lists what the work's code handed back, each
    a name, mime_type, url and size, uploaded to a persistent conversation's channel and attached to the
    reply), `task_cancelled`, `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`,
    `backchannel`, `interrupted`, `overlap_decided`, `conversation_compacted`, `models_changed`, `error`
    and `left`.
    `connector_unavailable` names an optional connector binding the session opened without: name (its
    alias), connector_id and reason, one of no_selection, shared_session, caller_unverified,
    connection_unavailable, provider_mismatch, needs_reauthorization, credential_rejected (the provider
    rejected the token or key a bearer or api_key connection holds; only new credentials fix it, so no
    login is offered), not_connected, open_failed, tool_unavailable and selection_dropped (a fork's or a
    reopened chat's selection for an alias its config no longer declares). Every watcher is sent each
    one when it attaches.
    `connector_scope_required` says a connector tool call was refused because the caller's own
    connection lacks access the provider asked for (insufficient_scope or a claims challenge), and a
    step-up consent was begun for it: name (the binding's alias), connector_id, connection_id, scopes
    (what the provider asked for, empty for a claims challenge), authorization_id, launch_url,
    handoff_token and expires_at. A client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. The old grant keeps working until the step-up succeeds, and the same call
    works afterwards in the same session. While that step-up is open, calls refused for the same access
    send no second event.
    Persistent text sessions also emit `conversation_updated` with conversation_id and a complete
    message snapshot: id, command_id, question_id, role, text, state, response_started_at,
    state_started_at, finished_at, duration_ms, saved, persistence_error and attachments. Each
    tool_calling attachment has tool_call_id, name, title, status, phase, summary, immutable started_at,
    execution_started_at, finished_at and duration_ms. A plugin_authorization attachment asks the end
    user to connect a plugin the reply needed, with plugin_id, title, authorize_url, text, thumb_url and
    title_link: a client shows it as a button opening authorize_url. Once the user finishes that login
    the message is sent again with the attachment's status set to connected. A connector_authorization
    attachment asks the end user to connect a connector binding the reply needed with their own account,
    with name (the binding's alias), connector_id, connection_id, authorization_id, title, launch_url,
    handoff_token and expires_at: a client opens launch_url in a popup and posts it handoff_token, as
    for createAuthorization. Once the user finishes that login the message is sent again with status
    connected and no handoff_token, and the agent carries on by itself. Activity states are thinking,
    queued, tools, writing, completed, failed and cancelled. tool_started includes tool_call_id, tool,
    turn_id and started_at, and pre_speech when the tool's connector binding sets one in its policy;
    tool_ran also includes tool_call_id.
    A respond command carrying command_id emits command_accepted with a nested command receipt
    (command_id, user_message_id, assistant_message_id, state, duplicate). Personal persistent text
    sessions require this ID. A retry with the same text returns the existing IDs without invoking the
    model again; reuse with different text emits an error. Commands with IDs currently accept text only.
    After restart an interrupted command is reported, not rerun.
    An `interrupt` command carrying `command_id` stops that command and emits `command_stopped` with its
    terminal receipt. A stop arriving after its command finished replays that command's receipt and
    leaves the command running now alone; an unknown command is reported as an error. Without
    `command_id` the frame stops whichever reply is current, which is what a caller with no command to
    name means by it.
    A `decision` frame is one judgement the conversation made, carrying the same fields as a CallEvent.
    Together they are why the call went the way it did, and they are also written down, so a finished
    call replays them from `/v1/agents/calls/{id}/events`.
    Two frames are only sent when asked for, because they are far more frequent than the rest and most
    consumers want neither. `interim=true` adds `hearing`, which is a transcript revision as it arrives
    rather than a settled turn. `decisions=false` drops `decision`.
    `replay_pending_tools=true` opts a durable tool host into replay of external tool calls still
    awaiting results in a live voice session. Completed, cancelled and timed-out requests are excluded
    at snapshot time. Replays retain their tool and turn IDs and may duplicate live delivery; the host
    must persist execution receipts and refuse to repeat uncertain writes. Ordinary status watchers
    should leave this disabled. Persistent text command recovery is unchanged.
    The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, `interrupt`
    (optionally naming a `command_id`), `instructions` or `close` to act on the session. `instructions`
    is server-side only: from an end user's device it changes nothing and is answered with an `error`
    frame, `context` `command`, as `updateSession` refuses it. A `tool_call` is the only frame that must
    be answered: everything else is a report. Tool calls made by durable personal commands carry
    `command_id` and `turn_id`; their result must repeat both values so a result cannot be adopted by
    another command or turn.
    A call to a tool declared with an `approval` waits for a person. The client reports their answer
    with `tool_approval` (`tool_call_id`, `command_id`, `turn_id`, `allowed`, and optionally a `summary`
    shown when they declined), before it answers the call with `tool_result`.
    `tool_result.output` is a string, or an array of parts `[{type: text|image_url, ...}]`. An image has
    an `image_url` object containing `url` (HTTP(S) or data URI), optionally with `detail` of `auto`,
    `low` or `high`. One socket message is at most 5 MB.
    `respond` may carry `images: [{url, detail}]`. These schedule the vision skill; the conversation
    receives the question and later the findings, without raw images. Video capture uses task-correlated
    `get_video_frames` tool requests and `tool_result` replies. Frames are never attached automatically
    to conversational turns.

    Args:
        id (str):
        interim (bool | Unset):  Default: False.
        decisions (bool | Unset):  Default: True.
        replay_pending_tools (bool | Unset):  Default: False.

    Raises:
        errors.UnexpectedStatus: If the server returns an undocumented status code and Client.raise_on_unexpected_status is True.
        httpx.TimeoutException: If the request takes longer than Client.timeout.

    Returns:
        Any | ErrorResponse
    """

    return (
        await asyncio_detailed(
            id=id,
            client=client,
            interim=interim,
            decisions=decisions,
            replay_pending_tools=replay_pending_tools,
        )
    ).parsed
