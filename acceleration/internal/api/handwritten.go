package api

import (
	"net/http"
	"reflect"
	"strings"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channels"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/mcpevents"
	"github.com/danielgtaylor/huma/v2"
)

// documentHandWritten declares the routes served by hand rather than by Huma: the four
// sockets, which an operation cannot express past the upgrade, the logs, exports and imports,
// which stream, and the plugin callback, which a browser arrives at from the provider. They
// are in the spec so a reader and a client generator know they exist, and so withServerSide
// reads their marks the way it reads every other operation's: a route nothing declares is
// the one place a default that refuses could fail open. Handler registers what serves them.
func documentHandWritten(api huma.API) {
	document := api.OpenAPI()
	registry := document.Components.Schemas
	document.AddOperation(&huma.Operation{
		OperationID: "watchSession",
		// The frames a client sends, which an upgrade gives OpenAPI no body to name them in.
		Extensions: map[string]any{
			clientAccessibleExtension: true,
			"x-client-frames": []*huma.Schema{
				registry.Schema(reflect.TypeFor[SessionRespondCommand](), true, ""),
				registry.Schema(reflect.TypeFor[ToolResultCommand](), true, ""),
				registry.Schema(reflect.TypeFor[ToolApprovalCommand](), true, ""),
			},
		},
		Method:  http.MethodGet,
		Path:    "/v1/agents/sessions/{id}/events",
		Summary: "Watch the conversation and answer the model's tool calls",
		Description: "A WebSocket, which OpenAPI cannot describe past the upgrade. Frames are JSON objects " +
			"carrying a `type` and the fields of that event.\n" +
			"The server sends what the conversation did: `joined`, `heard`, `responding`, " +
			"`response_delta`, `responded` (pending_work remains true while tools or delegated work " +
			"are outstanding), `spoke`, `turn`, `decision`, `delegated`, `task_settled` (files " +
			"lists what the work's code handed back, each a name, mime_type, url and size, uploaded " +
			"to a persistent conversation's channel and attached to the reply), " +
			"`task_cancelled`, `tool_call`, `tool_ran`, `transferred`, `pressed`, `looked_up`, " +
			"`backchannel`, `interrupted`, `overlap_decided`, `conversation_compacted`, " +
			"`models_changed`, `error` and `left`.\n" +
			"`connector_unavailable` names an optional connector binding the session opened " +
			"without: name (its alias), connector_id and reason, one of no_selection, " +
			"shared_session, caller_unverified, connection_unavailable, provider_mismatch, " +
			"needs_reauthorization, not_connected, open_failed, tool_unavailable and " +
			"selection_dropped (a fork's or a reopened chat's selection for an alias its config no longer " +
			"declares). " +
			"Every watcher is sent each one when it attaches.\n" +
			"`connector_scope_required` says a connector tool call was refused because the caller's " +
			"own connection lacks access the provider asked for (insufficient_scope or a claims " +
			"challenge), and a step-up consent was begun for it: name (the binding's alias), " +
			"connector_id, connection_id, scopes (what the provider asked for, empty for a claims " +
			"challenge), authorization_id, launch_url, handoff_token and expires_at. A client opens " +
			"launch_url in a popup and posts it handoff_token, as for createAuthorization. The old " +
			"grant keeps working until the step-up succeeds, and the same call works afterwards in " +
			"the same session. While that step-up is open, calls refused for the same access send " +
			"no second event.\n" +
			"Persistent text sessions also emit `conversation_updated` with conversation_id and a " +
			"complete message snapshot: id, command_id, question_id, role, text, state, " +
			"response_started_at, state_started_at, finished_at, duration_ms, saved, " +
			"persistence_error and attachments. Each tool_calling attachment has tool_call_id, name, " +
			"title, status, phase, summary, immutable started_at, execution_started_at, finished_at " +
			"and duration_ms. A plugin_authorization attachment asks the end user to connect a plugin " +
			"the reply needed, with plugin_id, title, authorize_url, text, thumb_url and title_link: " +
			"a client shows it as a button opening authorize_url. Once the user finishes that login " +
			"the message is sent again with the attachment's status set to connected. " +
			"A connector_authorization attachment asks the end user to connect a connector binding " +
			"the reply needed with their own account, with name (the binding's alias), connector_id, " +
			"connection_id, authorization_id, title, launch_url, handoff_token and expires_at: a " +
			"client opens launch_url in a popup and posts it handoff_token, as for createAuthorization. " +
			"Once the user finishes that login the message is sent again with status connected and " +
			"no handoff_token, and the agent carries on by itself. Activity states are thinking, queued, tools, writing, completed, " +
			"failed and cancelled. tool_started includes tool_call_id, tool, turn_id and started_at, and " +
			"pre_speech when the tool's connector binding sets one in its policy; " +
			"tool_ran also includes tool_call_id.\n" +
			"A respond command carrying command_id emits command_accepted with a nested command " +
			"receipt (command_id, user_message_id, assistant_message_id, state, duplicate). Personal " +
			"persistent text sessions require this ID. A retry with the same text returns the " +
			"existing IDs without invoking the model again; reuse with different text emits an " +
			"error. Commands with IDs currently accept text only. After restart an interrupted " +
			"command is reported, not rerun.\n" +
			"An `interrupt` command carrying `command_id` stops that command and emits " +
			"`command_stopped` with its terminal receipt. A stop arriving after its command finished " +
			"replays that command's receipt and leaves the command running now alone; an unknown " +
			"command is reported as an error. Without `command_id` the frame stops whichever reply " +
			"is current, which is what a caller with no command to name means by it.\n" +
			"A `decision` frame is one judgement the conversation made, carrying the same fields as " +
			"a CallEvent. Together they are why the call went the way it did, and they are also " +
			"written down, so a finished call replays them from `/v1/agents/calls/{id}/events`.\n" +
			"Two frames are only sent when asked for, because they are far more frequent than the " +
			"rest and most consumers want neither. `interim=true` adds `hearing`, which is a " +
			"transcript revision as it arrives rather than a settled turn. `decisions=false` drops " +
			"`decision`.\n" +
			"`replay_pending_tools=true` opts a durable tool host into replay of external tool calls " +
			"still awaiting results in a live voice session. Completed, cancelled and timed-out " +
			"requests are excluded at snapshot time. Replays retain their tool and turn IDs and may " +
			"duplicate live delivery; the host must persist execution receipts and refuse to repeat " +
			"uncertain writes. Ordinary status watchers should leave this disabled. Persistent text " +
			"command recovery is unchanged.\n" +
			"The client sends `tool_result` to answer a `tool_call`, and `say`, `respond`, " +
			"`interrupt` (optionally naming a `command_id`), `instructions` or `close` to act on the " +
			"session. `instructions` is server-side only: from an end user's device it changes " +
			"nothing and is answered with an `error` frame, `context` `command`, as `updateSession` " +
			"refuses it. A `tool_call` is the only frame that must be answered: everything else is a " +
			"report. Tool calls made by durable personal commands carry `command_id` and `turn_id`; " +
			"their result must repeat both values so a result cannot be adopted by another command " +
			"or turn.\n" +
			"A call to a tool declared with an `approval` waits for a person. The client reports " +
			"their answer with `tool_approval` (`tool_call_id`, `command_id`, `turn_id`, `allowed`, " +
			"and optionally a `summary` shown when they declined), before it answers the call with " +
			"`tool_result`.\n" +
			"`tool_result.output` is a string, or an array of parts `[{type: text|image_url, ...}]`. " +
			"An image has an `image_url` object containing `url` (HTTP(S) or data URI), optionally " +
			"with `detail` of `auto`, `low` or `high`. One socket message is at most 5 MB.\n" +
			"`respond` may carry `images: [{url, detail}]`. These schedule the vision skill; the " +
			"conversation receives the question and later the findings, without raw images. Video " +
			"capture uses task-correlated `get_video_frames` tool requests and `tool_result` " +
			"replies. Frames are never attached automatically to conversational turns.",
		Parameters: []*huma.Param{
			{Name: "id", In: "path", Description: "The session, as returned when it was created.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "interim", In: "query", Description: "Also send `hearing` frames, which are the words as they are revised.", Schema: &huma.Schema{Type: huma.TypeBoolean, Default: false}},
			{Name: "decisions", In: "query", Description: "Send `decision` frames.", Schema: &huma.Schema{Type: huma.TypeBoolean, Default: true}},
			{Name: "replay_pending_tools", In: "query", Description: "Replay pending live voice tool requests to a durable tool host.", Schema: &huma.Schema{Type: huma.TypeBoolean, Default: false}},
		},
		Responses: map[string]*huma.Response{
			"101": {Description: "The socket is open"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "openSocketSession",
		// A device holds a voice conversation over this socket when there is no call, as it
		// creates one with POST /v1/agents/sessions, which is client-accessible too.
		Extensions: map[string]any{clientAccessibleExtension: true},
		Method:     http.MethodGet,
		Path:       "/v1/agents/socket",
		Summary:    "Hold a voice conversation over the socket itself, with no call",
		Description: "A WebSocket, which OpenAPI cannot describe past the upgrade. Text frames are JSON " +
			"objects carrying a `type`.\n" +
			"The client's first frame is `start`, with `session` (a `CreateSessionRequest`) and an " +
			"optional `sample_rate`, 16000 when left out. `call_id` may be left out: the router makes " +
			"one up for the records. A `text` session is refused, because the socket carries audio. " +
			"A field that `createSession` refuses from an end user's device is refused here too: " +
			"`history` and `instructions` are server-side only.\n" +
			"The server answers `session`, with the `Session` and the `sample_rate` in use. Then " +
			"binary frames are PCM16 mono at that rate in both directions: the caller's audio in, " +
			"and the agent's speech out at the pace it would be heard on a call. A `cleared` frame " +
			"says speech already sent was thrown away because the caller cut in. Tool calls and " +
			"every other event go over the session's events socket, as they do for a call.\n" +
			"A refused start is an `error` frame with `error`, the message, and the socket closes. " +
			"An `error` frame for a field refused from a device also carries `code` and " +
			"`error_type`, the `code` and `type` that `createSession` answers the same field with.\n" +
			"The session lasts as long as the socket. Closing the socket, or sending `stop`, ends " +
			"the conversation. A conversation that ends closes the socket.",
		Responses: map[string]*huma.Response{
			"101": {Description: "The socket is open"},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "streamModality",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/stream",
		Summary:     "Route one modality over a socket, for a pipeline running elsewhere",
		Description: "A WebSocket, which OpenAPI cannot describe past the upgrade. This is the routing the " +
			"agent does, offered a piece at a time: a caller running its own pipeline sends audio or " +
			"text and gets transcripts, audio or completions back, and the request is failed over " +
			"and billed exactly as it would be inside a session.\n" +
			"Every socket opens with a `start` frame. It names either a `config_id`, a stored router " +
			"config to take the options from, or the options outright; naming both overrides that " +
			"config field by field. What it may carry is the modality's own option block - " +
			"`SttOptions` for speech-to-text, `TtsOptions` for a voice, `LlmOptions` for a model, " +
			"`StsOptions` for a speech-to-speech model - plus `agent_id` and `call_id` to attribute " +
			"the work to a conversation and `tags` to bill it.\n" +
			"Speech-to-text then takes binary PCM at the `sample_rate` the start frame named, 16 kHz " +
			"mono by default, and returns `transcript` frames. Text-to-speech takes `speak` frames " +
			"and returns binary audio with `synthesis_complete` between utterances: each audio frame " +
			"opens with a little-endian header of a uint32 sample rate, a uint16 channel count and " +
			"two reserved bytes, followed by PCM16 samples. Language models take `respond` frames, " +
			"each naming an `id` and anything from `LlmOptions` for that one response along with its " +
			"`tools`, and return `delta`, `reasoning_delta` and one `complete` per response; a " +
			"`complete` reports the `status` the response ended in, what it cost in tokens and how " +
			"long the caller waited for the first of them. `messages[].content` is a string, or an " +
			"array of parts `[{type: text|image_url, ...}]`. Assistant messages replay `tool_calls: " +
			"[{id, name, arguments, signature}]`, with arguments encoded as a JSON string and " +
			"optional opaque signature. Tool-result messages use `role: tool` and `tool_call_id` to " +
			"correlate their content. Completed tool calls include the same replay fields; " +
			"incomplete responses additionally report `incomplete_reason` (such as " +
			"`max_output_tokens`). The LLM `started` frame advertises `tool_history: true` when tool " +
			"replay is supported. Content parts use `[{type: text|image_url, ...}]` with images on " +
			"`image_url: {url, detail}`. Use `vlm` to select image-capable models. A model that does " +
			"not accept images is refused with an `error` frame naming the model and the modality, " +
			"before anything is billed. An `interrupt` frame naming `response_ids` abandons " +
			"responses still being generated, which still settle and are still billed for what they " +
			"produced before being cut off.\n" +
			"Speech-to-speech takes binary PCM at the `sample_rate` the start frame named, 16 kHz " +
			"mono by default, and returns the model's own voice as binary audio alongside JSON " +
			"frames: `speech_started` and `speech_stopped` when the model's own detector hears the " +
			"caller begin and finish, `input_transcript` and `output_transcript` for what it heard " +
			"and said, `response_started` and `response_complete` around each reply, and `tool_call` " +
			"and `tool_cancel` when it wants a function run. Each audio frame opens with a " +
			"sixteen-byte little-endian header of a uint32 sample rate, a uint16 channel count, a " +
			"uint16 header version, a uint32 generation and a uint32 chunk index, so a client can " +
			"drop audio from a reply that `response_complete` has already reported interrupted. " +
			"Mid-stream it takes `text` to inject a typed turn, `instructions` and `tools` to change " +
			"either where the model allows it, `frame` with an `image_url` for a model that sees, " +
			"`tool_result` with `tool_call_id` and `output` or `error`, and `interrupt` with an " +
			"optional `played_ms` saying how much of the reply the listener heard. What the routed " +
			"model cannot do is refused with an `error` frame rather than dropped. All four report " +
			"failures as `error` frames and end with `closed`.\n" +
			"Search is answered at `/v1/search` rather than here: one question and its answer need " +
			"no socket held open between them. Memory and phone are recorded rather than routed, so " +
			"they are not served either.",
		Parameters: []*huma.Param{
			{Name: "modality", In: "path", Description: "Which kind of model to route.", Required: true, Schema: registry.Schema(reflect.TypeFor[Modality](), true, "")},
		},
		Responses: map[string]*huma.Response{
			"101": {Description: "The socket is open"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "dispatchCalls",
		Method:      http.MethodGet,
		Path:        "/v1/dispatch",
		Summary:     "Wait for inbound calls to answer, as a worker",
		Description: "A WebSocket, which OpenAPI cannot describe past the upgrade. The worker connects here " +
			"and waits, rather than being called, because the agent runs in the customer's own " +
			"process and this service cannot reach into it.\n" +
			"The socket opens with a `ready` frame naming the worker, and a `call` or `message` " +
			"frame arrives for each piece of work handed to it, each carrying the `work_id` that " +
			"names it. The worker answers `done` with that `work_id`, and an `error` when it could " +
			"not be done, which is what frees its room for the next piece. It also sends `load` so " +
			"an operator can see what each worker is under, and `ping` to time the round trip " +
			"itself.\n" +
			"Work goes to whichever of a customer's workers is holding the least of what it said it " +
			"can hold, so a worker that never reports `done` is one the router cannot tell is busy.\n" +
			"A `message` written to a running session whose agent sets `dispatch.text` carries that " +
			"`session_id`, and a `command_id` when it was sent as a durable command. The model has " +
			"not answered it: the worker does, by creating a response on that session with a " +
			"server-side credential and the same `command_id` and text.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
			"user's device. This is the clearest case of why: a worker is offered other people's " +
			"callers, so anything that can open this socket can answer for the whole app. The auth " +
			"type has no query parameter, so a browser cannot open one at all.",
		Parameters: []*huma.Param{
			{Name: "capacity", In: "query", Description: "How much work this worker takes at once, calls and messages together.", Schema: &huma.Schema{Type: huma.TypeInteger, Minimum: bound(1), Default: 4}},
			{Name: "active", In: "query", Description: "How much work this worker is still running from before it reconnected, which the router " +
				"counts against its capacity until each piece is reported `done`. Sending this at all, " +
				"even as 0, is what says the worker reports `done`; one that leaves it out is held only " +
				"to the depth of its queue.", Schema: &huma.Schema{Type: huma.TypeInteger, Minimum: bound(0)}},
			{Name: "handles", In: "query", Description: "The kinds of work this worker accepts, comma separated. Absent is both. A worker that " +
				"only answers in writing says `message`, so a caller is never left listening to a phone " +
				"it would have dropped.", Schema: &huma.Schema{Type: huma.TypeString, Examples: []any{"call,message"}}},
		},
		Responses: map[string]*huma.Response{
			"101": {Description: "The socket is open"},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "pluginOAuthCallback",
		Method:      http.MethodGet,
		Path:        "/v1/agents/plugins/callback",
		Summary:     "Finish a plugin login",
		Description: "The provider redirects here with a code. The path is unauthenticated because the " +
			"browser arrives from the identity provider, and the state is the secret.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "code", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "state", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "error", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"302": {Description: "The browser is sent back to the agent editor"},
		},
	})
	// The consent flow's browser routes (authorizations.go). Unauthenticated, as the plugin
	// callback is: a browser carries no credential of this API's, and what each one checks
	// instead is in its description.
	document.AddOperation(&huma.Operation{
		OperationID: "getConnectorLaunchPage",
		Method:      http.MethodGet,
		Path:        connectorLaunchPath + "{id}",
		Summary:     "The page a consent starts on",
		Description: "The launch_url of an authorization, opened in a popup by the dashboard. The page " +
			"waits for the dashboard's origin to post the handoff token, trades it for the provider's " +
			"authorize URL and goes there. Unauthenticated because a browser opens it; the page " +
			"names no attempt and is the same for every one.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "id", In: "path", Description: "The authorization.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The launch page", Content: map[string]*huma.MediaType{"text/html": {Schema: &huma.Schema{Type: huma.TypeString}}}},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "handOffConnectorLaunch",
		Method:      http.MethodPost,
		Path:        connectorLaunchPath + "{id}",
		Summary:     "Bind a consent to this browser",
		Description: "What the launch page posts: `{\"handoff_token\": ...}`, from the router's own origin " +
			"only. It sets an HttpOnly cookie the callback requires, so the consent can finish only in " +
			"this browser, and answers `{\"authorization_url\": ...}`, the provider's authorize URL. " +
			"A handoff token is traded once: a second handoff for the same consent is a 400. " +
			"Unauthenticated because a browser sends it; the handoff token is the secret.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "id", In: "path", Description: "The authorization.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The cookie is set and the authorize URL returned"},
			"400": {Ref: "#/components/responses/BadRequest"},
			"403": {Description: "Not from the launch page's origin, or not this consent's handoff token"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "finishConnectorConsent",
		Method:      http.MethodGet,
		Path:        ConnectorCallbackPath,
		Summary:     "Finish a consent",
		Description: "The redirect URI a provider sends the browser back to. The state must name an " +
			"open consent, the browser must hold the cookie the handoff set, and the consent is " +
			"used once. The router then exchanges the code and sends the browser to the dashboard " +
			"with connection_id and status: connected, denied, failed, or account_mismatch when a " +
			"reconnect came back with another provider account and the old grant was kept. " +
			"Unauthenticated because the browser arrives from the provider.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "state", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "code", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "iss", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "error", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"302": {Description: "The browser is sent back to the dashboard"},
			"400": {Ref: "#/components/responses/BadRequest"},
			"403": {Description: "The consent was started in another browser"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "getConnectorClientMetadata",
		Method:      http.MethodGet,
		Path:        ConnectorClientMetadataPath,
		Summary:     "The router's OAuth client metadata",
		Description: "The OAuth Client ID Metadata Document a provider that supports it fetches, " +
			"at the URL that is the router's client_id. Served only when ROUTER_PUBLIC_URL is " +
			"https. Unauthenticated because the provider fetches it.",
		Security: []map[string][]string{},
		Responses: map[string]*huma.Response{
			"200": {Description: "The document", Content: map[string]*huma.MediaType{"application/json": {Schema: &huma.Schema{Type: huma.TypeObject}}}},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receiveConnectorEvent",
		Method:      http.MethodPost,
		Path:        connectorEventsPath + "{connector_id}",
		Summary:     "Receive a connector's provider event",
		Description: "Where a provider delivers the events of a built-in connector: Slack's tokens_revoked " +
			"and app_uninstalled to the operator's Slack app's Request URL, for one. Unauthenticated " +
			"because the provider is not a customer: each request is checked by the verifier the " +
			"connector's manifest names (channel.verifier) against the operator's secret, and an " +
			"unsigned or stale one changes nothing. A URL verification is answered with its " +
			"challenge as text/plain. A signal that a grant ended moves every connection of that " +
			"account to needs_reauthorization; a message goes to the channel bridge. The body is " +
			"at most 256 KiB. No SDK wraps it: only a provider calls it.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "connector_id", In: "path", Description: "A built-in connector id such as slack.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The event is taken, or a URL verification's challenge, echoed", Content: map[string]*huma.MediaType{"text/plain": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"401": {Description: "The request is not signed by the provider, or its signed timestamp is more than the manifest's max_age from now"},
			"404": {Description: "This connector takes no events here"},
			"413": {Description: "The event is over 256 KiB"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receiveProviderAppEvent",
		Method:      http.MethodPost,
		Path:        providerAppEventsPath + "{connector_id}/{provider_app_id}",
		Summary:     "Receive a provider app's event",
		Description: "Where a provider delivers the events of one customer's provider app: the Request URL " +
			"of a customer's Slack app, for one. Unauthenticated because the provider is not a customer: " +
			"each request is checked by the verifier the connector's manifest names " +
			"(channel.verifier) against that app's own signing secret, so an event signed for another " +
			"app is refused and changes nothing. A URL verification is answered with its challenge as " +
			"text/plain. A signal that a grant ended moves the app's customer's connections of that " +
			"account to needs_reauthorization, unless they connected after the event. A message goes " +
			"to the channel bridge, which writes it into the thread channel of its external thread in " +
			"Stream Chat; a retried delivery is dropped. The body is at most 256 KiB. No SDK wraps it: " +
			"only a provider calls it.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "connector_id", In: "path", Description: "The connector the provider app is of, such as slack_bot.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "provider_app_id", In: "path", Description: "The provider's id for the app, such as a Slack app id.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The event is taken, or a URL verification's challenge, echoed", Content: map[string]*huma.MediaType{"text/plain": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"401": {Description: "The request is not signed with the app's signing secret, or its signed timestamp is more than the manifest's max_age from now"},
			"404": {Description: "No such provider app, or its connector takes no events here"},
			"413": {Description: "The event is over 256 KiB"},
		},
	})
	// The direct-call proxy is one route for every method a provider's API takes, so it is one
	// operation per method. Its path runs on past {path}, which a Huma operation cannot route.
	errorBody := map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}
	for _, method := range proxyMethods {
		var body *huma.RequestBody
		if method == http.MethodPost || method == http.MethodPut || method == http.MethodPatch {
			// Any body of any media type, forwarded as it came with its Content-Type.
			body = &huma.RequestBody{
				Description: "The body for the provider, of any media type, at most 1 MiB.",
				Content:     map[string]*huma.MediaType{"*/*": {Schema: &huma.Schema{Type: huma.TypeString, Format: "binary"}}},
			}
		}
		document.AddOperation(&huma.Operation{
			OperationID: "proxyConnection" + method[:1] + strings.ToLower(method[1:]),
			Method:      method,
			Path:        connectionProxyPath + "{path}",
			Summary:     "Call a connection's provider directly (" + method + ")",
			Description: "Forwards the request to the connector's api_base with path appended, and answers " +
				"with the provider's answer as it came: status, headers and body. The request goes as it " +
				"came, but for the router's own credentials and caller headers (Authorization, " +
				"X-Api-Key, Stream-Auth-Type, X-Stream-*, X-Customer-Id) and query parameters (api_key, " +
				"token, customer_id, user_id), which never reach the provider; the connection's own " +
				"credential is added instead. On a 401 the credential is renewed and the request sent " +
				"once more when the scheme can renew it. A provider's 429 and Retry-After come back as " +
				"they are, and the connection's calls are then refused with a 429 here until that " +
				"Retry-After passes. A path with a dot segment, which would leave api_base, is " +
				"refused. The body is at most 1 MiB. Point a provider's own SDK at this URL as its base " +
				"URL, with a server-side token as its token and X-Api-Key and Stream-Auth-Type as extra " +
				"headers. An app-owned connection is the app's backend's; a user-owned one is reached " +
				"only by a backend acting for that user (X-Stream-User-Id). Each call that is sent " +
				"leaves one proxy_call audit row.\n\n" +
				"Server-side only: it needs a server-side token, so it cannot be reached from an end " +
				"user's device.",
			Parameters: []*huma.Param{
				{Name: "id", In: "path", Description: "The connection.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
				{Name: "path", In: "path", Description: "The provider's path under api_base, as escaped on the wire. It may hold slashes, such as chat.postMessage or repos/octo/hello/issues. A generated client escapes a slash in it to %2F, so it reaches a single-segment path only, such as chat.postMessage; for a longer one, point the provider's own SDK or an HTTP client at the URL.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			},
			RequestBody: body,
			Responses: map[string]*huma.Response{
				"200": {Description: "The provider's answer, as it came. It may have any status, a 401 or a 429 included."},
				"400": {Ref: "#/components/responses/BadRequest"},
				"401": {Ref: "#/components/responses/Unauthorized"},
				"403": {Ref: "#/components/responses/Forbidden"},
				"404": {Ref: "#/components/responses/NotFound"},
				"409": {Description: "The connection is not connected", Content: errorBody},
				"413": {Description: "The body is over 1 MiB", Content: errorBody},
				"429": {Description: "The provider asked to wait: retry after the Retry-After header's seconds. A 429 the provider answered itself comes back as it came.", Content: errorBody},
				"503": {Description: "The call did not reach the provider, or its answer did not come back", Content: errorBody},
			},
		})
	}
	document.AddOperation(&huma.Operation{
		OperationID: "answerProviderAppHandshake",
		Method:      http.MethodGet,
		Path:        providerAppEventsPath + "{connector_id}/{provider_app_id}",
		Summary:     "Answer a provider app's handshake",
		Description: "Where a provider checks a provider app's events URL before it delivers to it: Meta's " +
			"Verify Token check of a customer's WhatsApp webhook, for one. Unauthenticated because the " +
			"provider is not a customer. Only a connector whose manifest declares channel.handshake " +
			"answers it; the verify token is the provider app's id, the one in the URL, so nothing is " +
			"stored for it, and every delivery is still verified with the app's own secret. With " +
			"hub.mode subscribe, hub.verify_token the provider app's id and hub.challenge digits only, " +
			"the challenge is echoed as text/plain. Any other connector, an unknown provider app, or a " +
			"deployment without connectors answers 405 as for any method a route does not serve. No " +
			"SDK wraps it: only a provider calls it.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "connector_id", In: "path", Description: "The connector the provider app is of, such as whatsapp.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "provider_app_id", In: "path", Description: "The provider's id for the app, such as a Meta app id.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.mode", In: "query", Description: "subscribe.", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.verify_token", In: "query", Description: "The provider app's id.", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.challenge", In: "query", Description: "Digits to echo.", Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The challenge, echoed", Content: map[string]*huma.MediaType{"text/plain": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"404": {Description: "The query is not this URL's handshake: another mode or token, or a challenge that is not digits"},
			"405": {Description: "This connector, or this deployment, answers no handshake here"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "getPluginLogo",
		Method:      http.MethodGet,
		Path:        "/v1/agents/plugins/{plugin_id}/logo",
		Summary:     "A plugin's logo",
		Description: "The image a card uses to show which plugin it is asking about, as an SVG. The path " +
			"is unauthenticated because what draws it is an `<img>` in a chat client or a browser, " +
			"which has no credential of this API's to send, and because the catalog is the same " +
			"built-in list for every customer, so there is nothing of anybody's here.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "plugin_id", In: "path", Description: "A built-in catalog id such as slack or linear.", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The logo", Content: map[string]*huma.MediaType{"image/svg+xml": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receivePluginEvent",
		Method:      http.MethodPost,
		Path:        "/v1/agents/plugins/events/{token}",
		Summary:     "Receive a plugin's MCP event",
		Description: "Where a plugin's MCP server delivers the events an agent subscribed to, signed " +
			"with Standard Webhooks. The path is unauthenticated because the server is not a " +
			"customer: the token names the subscription and its secret signs each delivery. A " +
			"verification is answered with its challenge, and an event opens a text conversation.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "token", In: "path", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "A verification's challenge, echoed, or an event already taken"},
			"202": {Description: "The event is taken, and a conversation is opening for it"},
			"401": {Description: "The delivery is not signed with the subscription's secret"},
			"410": {Description: "There is no such subscription any more; stop delivering to it"},
			"413": {Description: "The delivery is over 256 KiB"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receiveConnectionEvent",
		Method:      http.MethodPost,
		Path:        mcpevents.Path + "{token}",
		Summary:     "Receive a connection's MCP event",
		Description: "Where a connection's MCP server delivers the events an agent config's binding " +
			"subscribed to, signed with Standard Webhooks (MCP Events, a draft). The path is " +
			"unauthenticated because the server is not a customer: the token names the subscription, " +
			"and each delivery is checked against that subscription's own secret, never a provider " +
			"app's. A verification is answered with its challenge, and an event opens a text " +
			"conversation from the config.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "token", In: "path", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "A verification's challenge, echoed, or an event already taken"},
			"202": {Description: "The event is taken, and a conversation is opening for it"},
			"400": {Description: "The delivery is not JSON, or not an event this subscription is for"},
			"401": {Description: "The delivery is not signed with the subscription's secret"},
			"410": {Description: "There is no such subscription, or its connection or declaration is gone; stop delivering to it"},
			"413": {Description: "The delivery is over 256 KiB"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "verifyChannelHook",
		Method:      http.MethodGet,
		Path:        channels.HookPath + "{token}",
		Summary:     "Answer a channel provider's webhook check",
		Description: "What WhatsApp asks for before it will deliver: the verify token the line was " +
			"connected with, answered with the challenge it sent, as text. Unauthenticated " +
			"because Meta is not a customer; the token in the path names the line.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "token", In: "path", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.mode", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.verify_token", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "hub.challenge", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The challenge, echoed", Content: map[string]*huma.MediaType{"text/plain": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"403": {Description: "That is not this line's verify token"},
			"410": {Description: "No line is connected at this address"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receiveChannelMessage",
		Method:      http.MethodPost,
		Path:        channels.HookPath + "{token}",
		Summary:     "Receive a message on a channel",
		Description: "Where WhatsApp, Telnyx and Linq deliver what somebody wrote to one of the app's " +
			"lines. Unauthenticated because the provider is not a customer: the token in the path " +
			"names the line, and each delivery is checked against the signing secret that line " +
			"was connected with. A message that has not been seen before earns a turn from " +
			"whichever agent names the number under `channels`, and what the agent says goes " +
			"back over the channel rather than in this response.",
		Security: []map[string][]string{},
		Parameters: []*huma.Param{
			{Name: "token", In: "path", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"202": {Description: "The delivery is taken, and the agent is answering it"},
			"400": {Description: "The delivery is not one this provider sends"},
			"401": {Description: "The delivery is not signed with this line's secret"},
			"410": {Description: "No line is connected at this address; stop delivering to it"},
			"413": {Description: "The delivery is over 256 KiB"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "receiveDLCReport",
		Method:      http.MethodPost,
		Path:        dlc.HookPath,
		Summary:     "Receive a 10DLC registration report",
		Description: "Where Telnyx reports on the brands and campaigns this router registered. " +
			"Unauthenticated because the vendor is not a customer: each report is checked " +
			"against the vendor's Ed25519 signature, and then only names the campaign to ask the " +
			"vendor about, so a report cannot say a campaign was approved that was not.",
		Security: []map[string][]string{},
		Responses: map[string]*huma.Response{
			"204": {Description: "The report is taken"},
			"401": {Description: "The report is not signed by the vendor"},
			"410": {Description: "This deployment registers nothing"},
			"413": {Description: "The report is over 64 KiB"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "listAgentLogs",
		Method:      http.MethodGet,
		Path:        "/v1/agents/logs",
		Summary:     "Latest structured agent logs, with backward cursor pagination",
		Parameters: []*huma.Param{
			{Name: "config_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "session_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "user_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "severity", In: "query", Description: "The least serious level to show, not the only one: warn is warnings and errors.", Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{"info", "warn", "error"}}},
			{Name: "source", In: "query", Description: "Comma-separated user/agent/tool/system sources.", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "q", In: "query", Schema: &huma.Schema{Type: huma.TypeString, MaxLength: itemLimit(256)}},
			{Name: "from", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Format: "date-time"}},
			{Name: "to", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Format: "date-time"}},
			{Name: "cursor", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "limit", In: "query", Schema: &huma.Schema{Type: huma.TypeInteger, Minimum: bound(1), Maximum: bound(250), Default: 250}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "Newest first; resume cursor marks the history/live handoff.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[AgentLogPage](), true, "")}}},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "getAgentLog",
		Method:      http.MethodGet,
		Path:        "/v1/agents/logs/{id}",
		Summary:     "Redacted structured log details scoped to the customer",
		Parameters: []*huma.Param{
			{Name: "id", In: "path", Required: true, Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "One log with full safe metadata.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[AgentLog](), true, "")}}},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
			"404": {Ref: "#/components/responses/NotFound"},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "streamAgentLogs",
		Method:      http.MethodGet,
		Path:        "/v1/agents/logs/stream",
		Summary:     "Read-only SSE of durable logs with replay",
		Description: "logs events carry AgentLog arrays in ingestion order; checkpoint events advance the " +
			"resume cursor; reset requires reloading history. Last-Event-ID resumes on reconnect. No " +
			"agent controls are accepted.",
		Parameters: []*huma.Param{
			{Name: "config_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "session_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "user_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "severity", In: "query", Description: "The least serious level to show, not the only one: warn is warnings and errors.", Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{"info", "warn", "error"}}},
			{Name: "source", In: "query", Description: "Comma-separated user/agent/tool/system sources.", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "q", In: "query", Schema: &huma.Schema{Type: huma.TypeString, MaxLength: itemLimit(256)}},
			{Name: "from", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Format: "date-time"}},
			{Name: "to", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Format: "date-time"}},
			{Name: "cursor", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "Last-Event-ID", In: "header", Schema: &huma.Schema{Type: huma.TypeString}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "SSE events logs, checkpoint and reset.", Content: map[string]*huma.MediaType{"text/event-stream": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "exportData",
		Method:      http.MethodGet,
		Path:        "/v1/data/export",
		Summary:     "Everything this app has, as newline-delimited JSON",
		Description: "One line per row, `{\"table\": ..., \"row\": {...}}`, and a last line `{\"cursor\": ..., " +
			"\"customer\": ..., \"at\": ...}`. The cursor comes last because it is also what says the " +
			"export finished: a stream that broke halfway has no cursor line, so half a copy cannot " +
			"be mistaken for a whole one. The rows are read at one moment rather than stitched " +
			"together, and exporting starts recording changes so that `listDataChanges` carries on " +
			"from exactly where this left off.\n" +
			"Only the calling app's rows are here, and credentials are not: an API key secret, an " +
			"OAuth access token and an OAuth refresh token stay with the deployment that holds them, " +
			"so an imported plugin connection has to be authorized again. The audio behind a voice " +
			"sample and a call recording lives in an object bucket rather than in this database; the " +
			"rows naming those objects are here, and copying the bucket is yours to do.\n" +
			"Server-side only, and refused entirely when the router runs with " +
			"ROUTER_AUTH_MODE=noauth, where naming a customer is all it takes to be one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The export, streamed", Content: map[string]*huma.MediaType{"text/plain": {Schema: &huma.Schema{Type: huma.TypeString}}}},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"503": {Description: "This deployment has no database to export from.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "importData",
		Method:      http.MethodPost,
		Path:        "/v1/data/import",
		Summary:     "Write an export, or a batch of changes, into this deployment",
		Description: "Takes what `exportData` produced, and the same lines with a `change` in place of a " +
			"`row` for what `listDataChanges` returned. Every row is written under the calling app " +
			"whatever the file says, so an export from one app cannot be imported into another's " +
			"rows, and rows belonging to a customer through a parent are only written where that " +
			"parent is the caller's.\n" +
			"Importing is idempotent: the same export applied twice leaves what applying it once " +
			"would.\n" +
			"Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.",
		RequestBody: &huma.RequestBody{
			Required: true,
			Content: map[string]*huma.MediaType{
				"application/octet-stream": {Schema: &huma.Schema{Type: huma.TypeString, Format: "binary"}},
			},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "What was written", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[DataImport](), true, "")}}},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"503": {Description: "This deployment has no database to import into.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
		},
	})
	document.AddOperation(&huma.Operation{
		OperationID: "listDataChanges",
		Method:      http.MethodGet,
		Path:        "/v1/data/changes",
		Summary:     "What has happened to this app's rows since a cursor",
		Description: "Oldest first, for replaying onto the deployment that took the export. A change is only " +
			"returned once every transaction older than it has committed, so following the cursor " +
			"never steps over a row, and a change carries the row as it now reads rather than the " +
			"columns that changed, so applying one twice is the same as applying it once.\n" +
			"Changes are only recorded for an app that has exported, and only for as long as the " +
			"deployment's retention window. A cursor older than what is still kept is answered 410, " +
			"which means export again.\n" +
			"Server-side only, and refused when the router runs with ROUTER_AUTH_MODE=noauth.",
		Parameters: []*huma.Param{
			{Name: "after", In: "query", Description: "The cursor the last page ended at.", Schema: &huma.Schema{Type: huma.TypeInteger, Format: "int64"}},
			{Name: "limit", In: "query", Schema: &huma.Schema{Type: huma.TypeInteger, Minimum: bound(1), Maximum: bound(1000), Default: 500}},
		},
		Responses: map[string]*huma.Response{
			"200": {Description: "The changes since the cursor", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[DataChangePage](), true, "")}}},
			"400": {Ref: "#/components/responses/BadRequest"},
			"401": {Ref: "#/components/responses/Unauthorized"},
			"403": {Ref: "#/components/responses/Forbidden"},
			"410": {Description: "The changes since that cursor are no longer kept, so export again.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
			"503": {Description: "This deployment has no database to read changes from.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[ErrorResponse](), true, "")}}},
		},
	})
}

// bound is a numeric limit on a parameter.
func bound(value float64) *float64 {
	return &value
}

// AgentLogPage is the AgentLogPage schema.
type AgentLogPage struct {
	Coverage     string     `json:"coverage"`
	DroppedLogs  int64      `json:"dropped_logs"`
	HasMore      bool       `json:"has_more"`
	Items        []AgentLog `json:"items" nullable:"false"`
	NextCursor   string     `json:"next_cursor"`
	ResumeCursor string     `json:"resume_cursor"`
}

// DataChangePage is the DataChangePage schema.
type DataChangePage struct {
	CaughtUp *bool        `json:"caught_up,omitempty" doc:"Nothing else has happened yet, which is when a switchover is safe: point your SDKs at the new deployment, wait for this to be true once more, and stop."`
	Changes  []DataChange `json:"changes" nullable:"false"`
	Cursor   int64        "json:\"cursor\" doc:\"What to pass as `after` next time.\""
}

// DataImport is the DataImport schema.
type DataImport struct {
	Cursor *int64            `json:"cursor,omitempty" doc:"The cursor the export named, to ask the other deployment for changes from."`
	Rows   int64             `json:"rows" doc:"How many rows were written."`
	Tables *map[string]int64 `json:"tables,omitempty" doc:"How many of them went into each table."`
}
