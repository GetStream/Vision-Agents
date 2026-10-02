package api

import (
	"context"
	"net/http"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/danielgtaylor/huma/v2"
)

// noCalls and noTranscripts are what the call paths say on a deployment that cannot
// answer them: a call is only remembered if there is somewhere to remember it, and what
// was said lives in Stream Chat rather than here.
const (
	noCalls       = "calls are not available: no database configured"
	noTranscripts = "transcripts are not available: no chat credentials configured"
	unknownCall   = "no such call"
	noStreamKeys  = "joining is not available: no stream credentials configured"
)

// listenerTokenValidity is how long a browser's token lasts. A call outliving it is a call
// nobody is still on, which is the same bet the Python examples make.
const listenerTokenValidity = time.Hour

// defaultCallType is what a call is joined as when nothing said otherwise. It matches the
// session default, which is what created the call in the first place.
const defaultCallType = "agent"

type CallDirection string

const (
	Inbound  CallDirection = "inbound"
	Outbound CallDirection = "outbound"
)

// Valid indicates whether the value is a known member of the CallDirection enum.
func (e CallDirection) Valid() bool {
	switch e {
	case Inbound:
		return true
	case Outbound:
		return true
	default:
		return false
	}
}

type Call struct {
	Id           string             `json:"id" doc:"The session that ran the call, which is what it is held by."`
	CallId       string             `json:"call_id"`
	AgentId      string             `json:"agent_id" doc:"Which agent ran it, and where its transcript is kept."`
	ConfigId     *string            `json:"config_id,omitempty"`
	CampaignId   *string            `json:"campaign_id,omitempty"`
	ContactId    *string            `json:"contact_id,omitempty"`
	UserId       *string            `json:"user_id,omitempty" doc:"Who the agent spoke to, as the client's own token named them. Empty for a call the customer's backend opened, and for telephony, where the number is the name."`
	FromNumber   *string            `json:"from_number,omitempty"`
	ToNumber     *string            `json:"to_number,omitempty"`
	Direction    CallDirection      `json:"direction" enum:"inbound,outbound"`
	StartedAt    time.Time          `json:"started_at"`
	EndedAt      *time.Time         `json:"ended_at,omitempty" doc:"Absent while the call is still running."`
	Stt          *string            `json:"stt,omitempty" doc:"The transcription target the call ran with, after a session's overrides were folded into whatever config it named. This is what was asked for rather than what each turn resolved to: a shortcut is several models and routing fails over between them, so per-turn providers are in the request rows."`
	Tts          *string            `json:"tts,omitempty" doc:"The voice target, on the same terms as stt."`
	Sts          *string            `json:"sts,omitempty" doc:"The speech-to-speech target, for a native call, on the same terms as stt."`
	Llm          *string            `json:"llm,omitempty" doc:"The target that held the conversation."`
	Subagent     *string            `json:"subagent,omitempty" doc:"The slower target delegated work ran on. Empty means nothing was delegated, which also means the skills below were never offered."`
	SttUsed      *string            `json:"stt_used,omitempty" doc:"The provider/model that transcribed, once routing picked one. Empty until somebody has been heard, and the last one that served if routing failed over."`
	TtsUsed      *string            `json:"tts_used,omitempty" doc:"The provider/model that spoke, on the same terms as stt_used."`
	StsUsed      *string            `json:"sts_used,omitempty" doc:"The provider/model that held a native call, on the same terms as stt_used."`
	LlmUsed      *string            `json:"llm_used,omitempty" doc:"The provider/model that held the conversation."`
	SubagentUsed *string            `json:"subagent_used,omitempty" doc:"The provider/model delegated work ran on. Empty when nothing was handed over, or when the thinking target was never reached."`
	Voice        *string            `json:"voice,omitempty" doc:"The voice the call asked for, in the provider's own terms. Empty means the provider's default."`
	VoiceUsed    *string            `json:"voice_used,omitempty" doc:"The voice that spoke, which is the provider's default when none was asked for. Known only while the call is running."`
	Mode         *SessionMode       `json:"mode,omitempty"`
	Instructions *string            `json:"instructions,omitempty" doc:"What the agent was told to be on this call."`
	Skills       *[]string          `json:"skills,omitempty" doc:"What the fast model could hand to the subagent. The instructions behind each name are in the skill registry."`
	Summary      *string            `json:"summary,omitempty" doc:"What a model made of the call, written once it was over."`
	Usage        *CallUsage         `json:"usage,omitempty"`
	ReviewScore  *int               `json:"review_score,omitempty" doc:"How well the agent handled it, from 1 to 5."`
	ReviewNotes  *string            `json:"review_notes,omitempty"`
	Tags         *map[string]string `json:"tags,omitempty"`
}

type SessionMode string

const (
	SessionModeCascade SessionMode = "cascade"
	SessionModeNative  SessionMode = "native"
	SessionModeText    SessionMode = "text"
)

// Valid indicates whether the value is a known member of the SessionMode enum.
func (e SessionMode) Valid() bool {
	switch e {
	case SessionModeCascade:
		return true
	case SessionModeNative:
		return true
	case SessionModeText:
		return true
	default:
		return false
	}
}

func (SessionMode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SessionMode", "How the session hears and speaks: a transcriber, a conversation model and a voice; one "+
		"speech-to-speech model; or in writing.",
		string(SessionModeCascade), string(SessionModeNative), string(SessionModeText))
}

type CallUsage struct {
	InputTokens       int64 `json:"input_tokens" doc:"Every prompt the models read, the cached part included."`
	CachedInputTokens int64 `json:"cached_input_tokens" doc:"The part of those prompts a provider served from its own cache."`
	OutputTokens      int64 `json:"output_tokens" doc:"Everything the models generated, reasoning included."`
	CostMicros        int64 `json:"cost_micros" doc:"Millionths of a dollar, priced from the providers' configured rates."`
	Requests          int64 `json:"requests" doc:"How many calls to a model it took, transcription and speech included."`
}

func (*CallUsage) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the call spent, summed over every request it made. Counted once the call is over, so " +
		"it is absent while one is still running. Requests that failed are included: a model that " +
		"read the prompt and then fell over is still billed for it."
	return schema
}

type TranscriptMessage struct {
	Speaker   string    `json:"speaker" doc:"Who said it, the agent under its own user id."`
	Name      *string   `json:"name,omitempty" doc:"That speaker's display name, when they have one."`
	Agent     *bool     `json:"agent,omitempty" doc:"Whether the agent said it rather than somebody it was talking to. It is what the line was stored as, so it holds however the agent was named."`
	Text      string    `json:"text"`
	CreatedAt time.Time `json:"created_at"`
}

type TimelineEntry struct {
	TurnId             string             `json:"turn_id"`
	StartedAt          time.Time          `json:"started_at"`
	CadenceMs          *float64           `json:"cadence_ms,omitempty" doc:"Last transcript revision to a stable turn ready for the flow controller."`
	DecisionMs         *float64           `json:"decision_ms,omitempty" doc:"Stable turn to the main model request, including flow and queueing."`
	ModelToFirstTextMs *float64           `json:"model_to_first_text_ms,omitempty" doc:"Main model request to the first text delta admitted to the voice pipeline."`
	TextToTtsMs        *float64           `json:"text_to_tts_ms,omitempty" doc:"First text delta to the first TTS request."`
	TtsToAudioMs       *float64           `json:"tts_to_audio_ms,omitempty" doc:"First TTS request to the first audio chunk published to the edge."`
	ModelCalls         *[]ModelCallTiming `json:"model_calls,omitempty" doc:"Individual model requests for this turn, including flow and delegated work."`
	Heard              *string            `json:"heard,omitempty" doc:"What the caller said, when it can be matched to this exchange."`
	Said               *string            `json:"said,omitempty" doc:"What the agent answered."`
	RoundtripMs        *float64           `json:"roundtrip_ms,omitempty" doc:"Last transcript revision to first audio published; includes cadence settling."`
	SttLatencyMs       *float64           `json:"stt_latency_ms,omitempty" doc:"The provider's decode time for the transcript that settled the turn." nullable:"true"`
	LlmTtftMs          *float64           `json:"llm_ttft_ms,omitempty" doc:"The wait between asking the model and its first token." nullable:"true"`
	TtsTtfbMs          *float64           `json:"tts_ttfb_ms,omitempty" doc:"The wait between sending the first sentence and the first audio." nullable:"true"`
	SpeechEndToAudioMs *float64           `json:"speech_end_to_audio_ms,omitempty" doc:"Last input audio to first output audio, estimated using provider STT processing time plus roundtrip. It excludes network transport and playback." nullable:"true"`
	AudioOutMs         *float64           `json:"audio_out_ms,omitempty" doc:"How much the agent spoke."`
	Interrupted        *bool              `json:"interrupted,omitempty" doc:"Whether the caller talked over the answer."`
}

type ModelCallTiming struct {
	OperationId  *string   `json:"operation_id,omitempty" doc:"Response ID for this operation; retries can share an ID."`
	Purpose      *string   `json:"purpose,omitempty" doc:"reply, flow or subagent."`
	StartedAt    time.Time `json:"started_at"`
	Provider     string    `json:"provider"`
	Model        string    `json:"model"`
	TtftMs       *float64  `json:"ttft_ms,omitempty" doc:"Request to first token."`
	DurationMs   *float64  `json:"duration_ms,omitempty" doc:"Request to completed response or failed create."`
	InputTokens  *int64    `json:"input_tokens,omitempty"`
	OutputTokens *int64    `json:"output_tokens,omitempty"`
	Success      bool      `json:"success"`
}

type CallEvent struct {
	At          time.Time    `json:"at"`
	Kind        DecisionKind `json:"kind"`
	Reason      string       `json:"reason" doc:"Why the conversation chose it, in words."`
	TurnId      *string      `json:"turn_id,omitempty" doc:"The exchange it was about, which lines it up against that turn's timings."`
	Participant *string      `json:"participant,omitempty" doc:"Who it concerned."`
	Said        *string      `json:"said,omitempty" doc:"What was heard, what the agent decided to say, or what the subagent came back with."`
	LatencyMs   *float64     `json:"latency_ms,omitempty" doc:"What the flow controller took to rule, or what the subagent took to answer. Zero where nothing was asked."`
}

type DecisionKind string

const (
	DecisionKindAnswer      DecisionKind = "answer"
	DecisionKindAsk         DecisionKind = "ask"
	DecisionKindBackchannel DecisionKind = "backchannel"
	DecisionKindCompact     DecisionKind = "compact"
	DecisionKindDelegate    DecisionKind = "delegate"
	DecisionKindFail        DecisionKind = "fail"
	DecisionKindIgnore      DecisionKind = "ignore"
	DecisionKindInterrupt   DecisionKind = "interrupt"
	DecisionKindQueue       DecisionKind = "queue"
	DecisionKindSettle      DecisionKind = "settle"
	DecisionKindShorten     DecisionKind = "shorten"
	DecisionKindSupersede   DecisionKind = "supersede"
	DecisionKindWait        DecisionKind = "wait"
)

// Valid indicates whether the value is a known member of the DecisionKind enum.
func (e DecisionKind) Valid() bool {
	switch e {
	case DecisionKindAnswer:
		return true
	case DecisionKindAsk:
		return true
	case DecisionKindBackchannel:
		return true
	case DecisionKindCompact:
		return true
	case DecisionKindDelegate:
		return true
	case DecisionKindFail:
		return true
	case DecisionKindIgnore:
		return true
	case DecisionKindInterrupt:
		return true
	case DecisionKindQueue:
		return true
	case DecisionKindSettle:
		return true
	case DecisionKindShorten:
		return true
	case DecisionKindSupersede:
		return true
	case DecisionKindWait:
		return true
	default:
		return false
	}
}

func (DecisionKind) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "DecisionKind", "What a conversation decided. Asking puts a settled turn to the flow controller; waiting "+
		"leaves it because the caller has not finished; ignoring drops speech meant for somebody "+
		"else; answering replies to it; queueing holds it until the agent has stopped talking; "+
		"interrupting abandons the reply being spoken and shortening ends it early; a "+
		"backchannel is a listening noise that never reaches the model; superseding drops a "+
		"ruling about words that have since changed; compacting replaces old history with a "+
		"summary; delegating hands work to the subagent and settling is that work coming back, "+
		"answered or not.",
		string(DecisionKindAsk), string(DecisionKindWait), string(DecisionKindIgnore), string(DecisionKindAnswer), string(DecisionKindQueue), string(DecisionKindInterrupt), string(DecisionKindShorten), string(DecisionKindBackchannel), string(DecisionKindSupersede), string(DecisionKindCompact), string(DecisionKindDelegate), string(DecisionKindSettle), string(DecisionKindFail))
}

type CallTokenRequest struct {
	UserId   *string `json:"user_id,omitempty" doc:"Who the browser joins as. Somebody watching a call is not the agent, so this defaults to a listener of its own rather than to the agent's user."`
	UserName *string `json:"user_name,omitempty" doc:"The name the other participants see. Defaults to the user id."`
}

type CallToken struct {
	ApiKey    string    `json:"api_key" doc:"The Stream app the call is in, which the browser SDK joins against."`
	Token     string    `json:"token"`
	UserId    string    `json:"user_id"`
	UserName  string    `json:"user_name"`
	CallId    string    `json:"call_id" doc:"The Stream call to join, which is not the id this call is held by here."`
	CallType  string    `json:"call_type"`
	ExpiresAt time.Time `json:"expires_at"`
}

type ChatTokenRequest struct {
	AgentId  string  `json:"agent_id" doc:"Whose conversation to read. This is the session's agent id, which is what names the channel it is written to."`
	UserId   *string `json:"user_id,omitempty" doc:"Who the browser reads as. Somebody watching is not the agent, so this defaults to a reader of its own rather than to the agent's user."`
	UserName *string `json:"user_name,omitempty" doc:"The name shown against anything they write. Defaults to the user id."`
}

type ChatToken struct {
	ApiKey      string    `json:"api_key" doc:"The Stream app the channel is in, which the browser SDK connects to."`
	Token       string    `json:"token"`
	UserId      string    `json:"user_id"`
	UserName    string    `json:"user_name"`
	ChannelType string    `json:"channel_type" doc:"Always agent, which is the type a conversation is written to."`
	ChannelId   string    `json:"channel_id" doc:"The channel holding the conversation, which is the agent id."`
	ExpiresAt   time.Time `json:"expires_at"`
}

type listCallsRequest struct {
	AgentID    optionalParam[string]    `query:"agent_id" doc:"Narrow to one agent."`
	CampaignID optionalParam[string]    `query:"campaign_id" doc:"Narrow to the calls one campaign placed."`
	Running    bool                     `query:"running" doc:"Only calls that have not ended." default:"false"`
	From       optionalParam[time.Time] `query:"from" doc:"Only calls that started at or after this, inclusive."`
	To         optionalParam[time.Time] `query:"to" doc:"Only calls that started before this, exclusive."`
	Limit      int                      `query:"limit" default:"50"`
}

type callListResponse struct {
	Body []Call
}

type getCallRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type callResponse struct {
	Body Call
}

type getCallTranscriptRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type transcriptMessageListResponse struct {
	Body []TranscriptMessage
}

type getCallTimelineRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type timelineEntryListResponse struct {
	Body []TimelineEntry
}

type getCallEventsRequest struct {
	ID    string `path:"id" doc:"The resource, as returned when it was created."`
	Limit int    `query:"limit" doc:"How many to return, oldest first." default:"1000"`
}

type callEventListResponse struct {
	Body []CallEvent
}

type createCallTokenRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body *CallTokenRequest
}

type callTokenResponse struct {
	Body CallToken
}

type createChatTokenRequest struct {
	Body ChatTokenRequest
}

type chatTokenResponse struct {
	Body ChatToken
}

func (s *Server) registerCalls(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listCalls",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls",
		Summary:     "The calls the calling customer has run",
		Description: "A session lives in memory and is gone when the process is, so a call is " +
			"recorded as it starts and again as it ends. This is what answers what happened " +
			"yesterday, and what is happening now after a restart.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's calls, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listCalls)
	huma.Register(api, huma.Operation{
		OperationID: "getCall",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}",
		Summary:     "One call, with whatever was made of it afterwards",
		Responses: map[string]*huma.Response{
			"200": {Description: "The call"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCall)
	huma.Register(api, huma.Operation{
		OperationID: "getCallTranscript",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/transcript",
		Summary:     "What was said on a call",
		Description: "Read back from the chat channel the conversation was written to as it happened, " +
			"rather than copied into a second place that could disagree with it.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The conversation, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallTranscript)
	huma.Register(api, huma.Operation{
		OperationID: "getCallTimeline",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/timeline",
		Summary:     "The call as it unfolded, said and measured together",
		Description: "Each exchange with what was said in it and what it cost the caller in waiting: " +
			"how long the answer took to start, how much the agent spoke, and whether it was " +
			"talked over.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The exchanges, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallTimeline)
	huma.Register(api, huma.Operation{
		OperationID: "getCallEvents",
		Method:      http.MethodGet,
		Path:        "/v1/agents/calls/{id}/events",
		Summary:     "What the conversation decided, and why",
		Description: "A timeline says what a call cost the caller in waiting. This says why the call " +
			"went the way it did: why the agent waited rather than answering, why it read " +
			"something as not meant for it, why it stopped mid-sentence. Read in order they " +
			"are the reasoning behind the conversation, which is the only thing that " +
			"explains a call that surprised somebody. A call still running reports the same " +
			"decisions live on the session socket.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The judgements, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getCallEvents)
	huma.Register(api, huma.Operation{
		OperationID: "createCallToken",
		Method:      http.MethodPost,
		Path:        "/v1/agents/calls/{id}/token",
		Summary:     "What a browser needs to join this call",
		Description: "Mints a Stream token so a person can join the call from a browser and talk to " +
			"the agent, and says which call to join with it. The secret stays here: the " +
			"browser is handed a token that expires, never the key that signs one.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The credentials, and the call they are for"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.createCallToken)
	huma.Register(api, huma.Operation{
		OperationID: "createChatToken",
		Method:      http.MethodPost,
		Path:        "/v1/agents/chat-token",
		Summary:     "What a browser needs to read an agent's conversation",
		Description: "An agent writes what was said into the Stream Chat channel agent:{agent_id}, so " +
			"a client that can read that channel needs no transcript API. This mints the " +
			"token to read it with, and adds the reader to the channel, since a conversation " +
			"they are not a member of is one they cannot watch.\n" +
			"The secret stays here, the same as for a call token: the browser is handed " +
			"something that expires.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The credentials, and the channel they are for"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createChatToken)
}

// listCalls returns the calling customer's calls, newest first.
func (s *Server) listCalls(ctx context.Context, request *listCallsRequest) (*callListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}

	filter := store.CallFilter{
		AgentID:    request.AgentID.Value,
		CampaignID: request.CampaignID.Value,
		Running:    request.Running,
		Limit:      request.Limit,
	}
	if request.From.Set {
		filter.From = request.From.Value
	}
	if request.To.Set {
		filter.To = request.To.Value
	}

	stored, err := s.store.CustomerCalls(ctx, customerID, filter)
	if err != nil {
		return nil, err
	}

	listed := make([]Call, 0, len(stored))
	for _, call := range stored {
		listed = append(listed, callOf(call))
	}
	return &callListResponse{Body: listed}, nil
}

// getCall returns one call and whatever was made of it afterwards.
func (s *Server) getCall(ctx context.Context, request *getCallRequest) (*callResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}

	call, err := s.store.Call(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCall)
	}
	rendered := callOf(call)
	s.attachUsed(ctx, customerID, call, &rendered)
	s.attachUsage(ctx, customerID, call, &rendered)
	return &callResponse{Body: rendered}, nil
}

// attachUsage totals what the call spent, once there is a total to give. A running call is
// left without one: the sum would be read again on every poll and be wrong by a turn each
// time, and what a conversation cost is a question asked after it, not during.
func (s *Server) attachUsage(ctx context.Context, customerID string, call store.Call, rendered *Call) {
	if call.EndedAt == nil || s.store == nil {
		return
	}

	spent, err := s.store.CallUsage(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		s.logger.Error("could not read what a call spent", "call", call.ID, "error", err)
		return
	}
	rendered.Usage = &CallUsage{
		InputTokens:       spent.InputTokens,
		CachedInputTokens: spent.CachedInputTokens,
		OutputTokens:      spent.OutputTokens,
		CostMicros:        spent.CostMicros,
		Requests:          spent.Requests,
	}
}

// createCallToken mints what a browser needs to join a call and talk to the agent.
//
// The token is signed here rather than fetched, so this makes no network calls, and the
// user is not registered either: the coordinator does that when the browser connects. The
// call type comes from the running session when there is one, because only the session
// knows what it joined as.
func (s *Server) createCallToken(ctx context.Context, request *createCallTokenRequest) (*callTokenResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return nil, huma.Error400BadRequest(noStreamKeys)
	}

	call, err := s.store.Call(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCall)
	}

	var wanted CallTokenRequest
	if request.Body != nil {
		wanted = *request.Body
	}
	userID := value(wanted.UserId)
	if userID == "" {
		userID = "listener-" + call.ID
	}
	userName := value(wanted.UserName)
	if userName == "" {
		userName = userID
	}

	callType := defaultCallType
	if s.sessions != nil {
		if found, running := s.sessions.Get(call.ID, OwnerFrom(ctx)); running {
			callType = found.Spec().CallType
		}
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, err
	}
	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return &callTokenResponse{Body: CallToken{
		ApiKey:    s.streamKey,
		Token:     token,
		UserId:    userID,
		UserName:  userName,
		CallId:    call.CallID,
		CallType:  callType,
		ExpiresAt: expiresAt,
	}}, nil
}

// createChatToken mints what a browser needs to read an agent's conversation.
//
// The transcript is already a Stream Chat channel, so a client that can reach it needs no
// transcript API and sees a reply while it is still being written. Reading it means being
// in it: the reader is added to the channel here, because a token alone opens nothing.
func (s *Server) createChatToken(ctx context.Context, request *createChatTokenRequest) (*chatTokenResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.streamKey == "" || s.streamSecret == "" {
		return nil, huma.Error400BadRequest(noStreamKeys)
	}

	agentID := strings.TrimSpace(request.Body.AgentId)
	if agentID == "" {
		return nil, huma.Error400BadRequest("an agent id is required, since it names the channel")
	}

	userID := value(request.Body.UserId)
	if userID == "" {
		// Somebody reading a conversation is not the agent, and two readers of the same
		// one are not each other, which is why this is per customer rather than shared.
		userID = "reader-" + customerID
	}
	userName := value(request.Body.UserName)
	if userName == "" {
		userName = userID
	}

	client, err := getstream.NewClient(s.streamKey, s.streamSecret)
	if err != nil {
		return nil, err
	}

	if _, err := client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{
		Users: map[string]getstream.UserRequest{
			agentID: {ID: agentID},
			userID:  {ID: userID, Name: &userName},
		},
	}); err != nil {
		return nil, err
	}

	// The channel is created by whoever holds the conversation, which for an agent nobody
	// has spoken to yet is nobody. Creating it here means a reader can watch it before
	// the first word rather than polling until it exists.
	if _, err := client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.GetOrCreateChannelRequest{
			Data: &getstream.ChannelInput{
				CreatedByID: &agentID,
				Members:     []getstream.ChannelMemberRequest{{UserID: userID}},
			},
		}); err != nil {
		return nil, err
	}

	// Members in creation data are ignored when the channel already exists.
	// Add the reader explicitly before minting a token for a members-only channel.
	if _, err := client.Chat().UpdateChannel(ctx, chatlog.ChannelType, agentID,
		&getstream.UpdateChannelRequest{AddMembers: []getstream.ChannelMemberRequest{{UserID: userID}}}); err != nil {
		return nil, err
	}

	expiresAt := time.Now().UTC().Add(listenerTokenValidity)
	token, err := client.CreateToken(userID, getstream.WithExpiration(listenerTokenValidity))
	if err != nil {
		return nil, err
	}

	return &chatTokenResponse{Body: ChatToken{
		ApiKey:      s.streamKey,
		Token:       token,
		UserId:      userID,
		UserName:    userName,
		ChannelType: chatlog.ChannelType,
		ChannelId:   agentID,
		ExpiresAt:   expiresAt,
	}}, nil
}

// getCallTranscript returns what was said, read back out of the channel it was written to
// while the call was happening.
func (s *Server) getCallTranscript(ctx context.Context, request *getCallTranscriptRequest) (*transcriptMessageListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}
	if s.transcripts == nil {
		return nil, huma.Error400BadRequest(noTranscripts)
	}

	call, err := s.store.Call(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCall)
	}

	said, err := s.transcripts.Transcript(ctx, call.AgentID)
	if err != nil {
		return nil, err
	}

	messages := make([]TranscriptMessage, 0, len(said))
	for _, line := range said {
		messages = append(messages, TranscriptMessage{
			Speaker:   line.Speaker,
			Name:      optional(line.Name),
			Agent:     &line.Agent,
			Text:      line.Text,
			CreatedAt: line.At,
		})
	}
	return &transcriptMessageListResponse{Body: messages}, nil
}

// getCallEvents returns what the conversation decided on one call, oldest first.
func (s *Server) getCallEvents(ctx context.Context, request *getCallEventsRequest) (*callEventListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}

	call, err := s.store.Call(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCall)
	}

	stored, err := s.store.CallEvents(
		ctx, customerID, call.CallID, call.StartedAt, call.EndedAt, request.Limit)
	if err != nil {
		return nil, err
	}

	decisions := make([]CallEvent, 0, len(stored))
	for _, decided := range stored {
		decisions = append(decisions, CallEvent{
			At:          decided.At,
			Kind:        DecisionKind(decided.Kind),
			Reason:      decided.Reason,
			TurnId:      optional(decided.TurnID),
			Participant: optional(decided.Participant),
			Said:        optional(decided.Said),
			LatencyMs:   decided.LatencyMs,
		})
	}
	return &callEventListResponse{Body: decisions}, nil
}

// getCallTimeline returns the call as it unfolded: each exchange with what was said in it
// and what the caller waited for it.
func (s *Server) getCallTimeline(ctx context.Context, request *getCallTimelineRequest) (*timelineEntryListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noCalls)
	}

	call, err := s.store.Call(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownCall)
	}

	turns, err := s.store.CallTurns(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		return nil, err
	}
	models, err := s.store.CallModelCalls(ctx, customerID, call.AgentID, call.CallID, call.StartedAt, call.EndedAt)
	if err != nil {
		return nil, err
	}

	// The transcript is worth having but not worth failing over: the timings are the
	// part of this view that only this service holds.
	var said []chatlog.Spoken
	if s.transcripts != nil {
		said, err = s.transcripts.Transcript(ctx, call.AgentID)
		if err != nil {
			s.logger.Error("could not read the transcript for a timeline",
				"call", call.ID, "error", err)
		}
	}

	return &timelineEntryListResponse{Body: timelineOf(turns, said, models)}, nil
}

// timelineOf pairs each exchange with the lines said during it.
//
// A turn starts when the transcript settled, and the messages are written as the call
// goes, so a line belongs to the last turn that had started when it was stored. Within a
// turn the first line is what the caller said and the last is what the agent answered,
// which is what a two-line exchange always is.
func timelineOf(turns []store.Turn, said []chatlog.Spoken, models []store.Request) []TimelineEntry {
	modelCalls := make(map[string][]ModelCallTiming)
	for _, request := range models {
		if request.TurnID == "" {
			continue
		}
		modelCalls[request.TurnID] = append(modelCalls[request.TurnID], ModelCallTiming{
			OperationId:  optional(request.OperationID),
			Purpose:      optional(request.Purpose),
			StartedAt:    request.StartedAt,
			Provider:     request.Provider,
			Model:        request.Model,
			TtftMs:       request.LatencyMs,
			DurationMs:   request.DurationMs,
			InputTokens:  &request.InputTokens,
			OutputTokens: &request.OutputTokens,
			Success:      request.Success,
		})
	}
	timeline := make([]TimelineEntry, 0, len(turns))
	for index, turn := range turns {
		entry := TimelineEntry{
			TurnId:             turn.TurnID,
			StartedAt:          turn.StartedAt,
			CadenceMs:          turn.CadenceMs,
			DecisionMs:         turn.DecisionMs,
			ModelToFirstTextMs: turn.ModelToFirstTextMs,
			TextToTtsMs:        turn.TextToTTSMs,
			TtsToAudioMs:       turn.TTSToAudioMs,
			RoundtripMs:        turn.RoundtripMs,
			SttLatencyMs:       turn.STTLatencyMs,
			LlmTtftMs:          turn.LLMTTFTMs,
			TtsTtfbMs:          turn.TTSTTFBMs,
			SpeechEndToAudioMs: turn.SpeechEndToAudioMs,
			AudioOutMs:         turn.AudioOutMs,
			Interrupted:        &turn.Interrupted,
		}
		if calls := modelCalls[turn.TurnID]; len(calls) > 0 {
			entry.ModelCalls = &calls
		}

		var until time.Time
		if index+1 < len(turns) {
			until = turns[index+1].StartedAt
		}
		if spoken := within(said, turn.StartedAt, until); len(spoken) > 0 {
			entry.Heard = optional(spoken[0].Text)
			if len(spoken) > 1 {
				entry.Said = optional(spoken[len(spoken)-1].Text)
			}
		}
		timeline = append(timeline, entry)
	}
	return timeline
}

// within returns the lines stored in a half-open window. A zero end is the rest of them.
func within(said []chatlog.Spoken, from, until time.Time) []chatlog.Spoken {
	var inside []chatlog.Spoken
	for _, line := range said {
		if line.At.Before(from) {
			continue
		}
		if !until.IsZero() && !line.At.Before(until) {
			break
		}
		inside = append(inside, line)
	}
	return inside
}

// callOf renders a call for the wire.
func callOf(call store.Call) Call {
	rendered := Call{
		Id:        call.ID,
		CallId:    call.CallID,
		AgentId:   call.AgentID,
		Direction: CallDirection(call.Direction),
		StartedAt: call.StartedAt,
		EndedAt:   call.EndedAt,
	}
	rendered.ConfigId = optional(call.ConfigID)
	rendered.CampaignId = optional(call.CampaignID)
	rendered.ContactId = optional(call.ContactID)
	rendered.UserId = optional(call.UserID)
	rendered.FromNumber = optional(call.FromNumber)
	rendered.ToNumber = optional(call.ToNumber)
	rendered.Stt = optional(call.STT)
	rendered.Tts = optional(call.TTS)
	rendered.Sts = optional(call.STS)
	rendered.Llm = optional(call.LLM)
	rendered.Subagent = optional(call.Subagent)
	rendered.Voice = optional(call.Voice)
	mode := SessionModeCascade
	if call.STS != "" {
		mode = SessionModeNative
	}
	rendered.Mode = &mode
	rendered.Instructions = optional(call.Instructions)
	rendered.Summary = optional(call.Summary)
	rendered.ReviewNotes = optional(call.ReviewNotes)
	rendered.ReviewScore = call.ReviewScore
	if len(call.Skills) > 0 {
		skills := call.Skills
		rendered.Skills = &skills
	}
	if len(call.Tags) > 0 {
		tags := call.Tags
		rendered.Tags = &tags
	}
	return rendered
}

// attachUsed fills in the provider/models routing picked, which a shortcut does not name.
//
// A live session knows the current selection before any request row has been written. A
// finished call has only the request rows, and those also cover a live call after the
// first turn. Failover can leave more than one model per modality; the one that still
// satisfies the target is the one shown.
func (s *Server) attachUsed(ctx context.Context, customerID string, call store.Call, rendered *Call) {
	if s.sessions != nil {
		if found, ok := s.sessions.Get(call.ID, OwnerFrom(ctx)); ok {
			// The row is written off the request path, so what a running session is on
			// now is read from it rather than from a row a swap may not have reached yet.
			spec := found.Spec()
			rendered.Stt = optional(spec.STTTarget)
			rendered.Tts = optional(spec.TTSTarget)
			rendered.Llm = optional(spec.LLMTarget)
			rendered.Sts = optional(spec.STSTarget)
			rendered.Subagent = optional(spec.SubagentTarget)
			asked, voiceUsed := found.Voice()
			rendered.Voice = optional(asked)
			rendered.VoiceUsed = optional(voiceUsed)
			mode := SessionMode(found.Mode())
			rendered.Mode = &mode
			stt, llm, tts, subagent := found.Resolved()
			rendered.SttUsed = optional(stt)
			rendered.LlmUsed = optional(llm)
			rendered.TtsUsed = optional(tts)
			rendered.SubagentUsed = optional(subagent)
			rendered.StsUsed = optional(found.Speech())
		}
	}

	if filledUsed(rendered) || s.store == nil {
		return
	}

	used, err := s.store.CallUsedModels(ctx, customerID, call.AgentID, call.StartedAt, call.EndedAt)
	if err != nil {
		s.logger.Error("could not read the models a call used", "call", call.ID, "error", err)
		return
	}

	if value(rendered.Sts) != "" {
		rendered.StsUsed = firstUsed(rendered.StsUsed, matchUsed(value(rendered.Sts), namesOf(used, "sts"), s.candidateNames(ctx, routing.STS, value(rendered.Sts))))
	} else {
		rendered.SttUsed = firstUsed(rendered.SttUsed, matchUsed(value(rendered.Stt), namesOf(used, "stt"), s.candidateNames(ctx, routing.STT, value(rendered.Stt))))
		rendered.TtsUsed = firstUsed(rendered.TtsUsed, matchUsed(value(rendered.Tts), namesOf(used, "tts"), s.candidateNames(ctx, routing.TTS, value(rendered.Tts))))
		rendered.LlmUsed = firstUsed(rendered.LlmUsed, matchUsed(value(rendered.Llm), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Llm))))
	}
	rendered.SubagentUsed = firstUsed(rendered.SubagentUsed, matchUsed(value(rendered.Subagent), namesOf(used, "llm"), s.candidateNames(ctx, routing.LLM, value(rendered.Subagent))))
}

func filledUsed(call *Call) bool {
	if value(call.Sts) != "" {
		return call.StsUsed != nil && (value(call.Subagent) == "" || call.SubagentUsed != nil)
	}
	return call.SttUsed != nil && call.TtsUsed != nil && call.LlmUsed != nil && call.SubagentUsed != nil
}

func firstUsed(existing *string, used string) *string {
	if existing != nil {
		return existing
	}
	return optional(used)
}

func namesOf(used []store.UsedModel, modality string) []string {
	var names []string
	for _, model := range used {
		if model.Modality == modality {
			names = append(names, model.Provider+"/"+model.Model)
		}
	}
	return names
}

// matchUsed picks which of the used provider/models served a target. used is most-recent
// first. candidates are the names that target can resolve to; when they are known only a
// used model that is still a candidate is returned, which is how the voice model and the
// thinking model are told apart when both wrote llm rows.
func matchUsed(asked string, used []string, candidates []string) string {
	if asked == "" || len(used) == 0 {
		return ""
	}
	if len(candidates) > 0 {
		allowed := make(map[string]struct{}, len(candidates))
		for _, name := range candidates {
			allowed[name] = struct{}{}
		}
		for _, name := range used {
			if _, ok := allowed[name]; ok {
				return name
			}
		}
		return ""
	}
	for _, name := range used {
		if name == asked {
			return name
		}
	}
	if strings.Contains(asked, "/") {
		return ""
	}
	return used[0]
}

func (s *Server) candidateNames(ctx context.Context, modality routing.Modality, target string) []string {
	if target == "" {
		return nil
	}
	router, ok := s.routers[modality]
	if !ok {
		return nil
	}
	candidates, err := router.Resolve(ctx, target, nil)
	if err != nil {
		return nil
	}
	names := make([]string, 0, len(candidates))
	for _, candidate := range candidates {
		names = append(names, candidate.Config.Name())
	}
	return names
}
