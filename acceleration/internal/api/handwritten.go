package api

import (
	"net/http"
	"reflect"

	"github.com/danielgtaylor/huma/v2"
)

// documentHandWritten declares the routes served by hand rather than by Huma: the three
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
			"Persistent text sessions also emit `conversation_updated` with conversation_id and a " +
			"complete message snapshot: id, command_id, question_id, role, text, state, " +
			"response_started_at, state_started_at, finished_at, duration_ms, saved, " +
			"persistence_error and attachments. Each tool_calling attachment has tool_call_id, name, " +
			"title, status, phase, summary, immutable started_at, execution_started_at, finished_at " +
			"and duration_ms. Activity states are thinking, queued, tools, writing, completed, " +
			"failed and cancelled. tool_started includes tool_call_id, tool, turn_id and started_at; " +
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
			"session. A `tool_call` is the only frame that must be answered: everything else is a " +
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
	document.AddOperation(&huma.Operation{
		OperationID: "listAgentLogs",
		Method:      http.MethodGet,
		Path:        "/v1/agents/logs",
		Summary:     "Latest structured agent logs, with backward cursor pagination",
		Parameters: []*huma.Param{
			{Name: "config_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "session_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "user_id", In: "query", Schema: &huma.Schema{Type: huma.TypeString}},
			{Name: "severity", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{"info", "error"}}},
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
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
			{Name: "severity", In: "query", Schema: &huma.Schema{Type: huma.TypeString, Enum: []any{"info", "error"}}},
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
			"503": {Description: "Log storage is unavailable.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
			"503": {Description: "This deployment has no database to export from.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
			"503": {Description: "This deployment has no database to import into.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
			"410": {Description: "The changes since that cursor are no longer kept, so export again.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
			"503": {Description: "This deployment has no database to read changes from.", Content: map[string]*huma.MediaType{"application/json": {Schema: registry.Schema(reflect.TypeFor[Error](), true, "")}}},
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
