package api

import (
	"context"
	"errors"
	"net/http"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/danielgtaylor/huma/v2"
)

type CreateSessionRequest struct {
	Id               *string                 `json:"id,omitempty" doc:"The id to hold the session by, so a caller can know it before the session exists. It must be a UUID nobody has used for a session before. Omitted, the router generates a UUIDv7."`
	ConversationId   *string                 `json:"conversation_id,omitempty" doc:"Stream Chat CID to resume; returned for persistent text sessions."`
	ContextTruncated *bool                   `json:"context_truncated,omitempty" doc:"Older history was omitted from the model context."`
	CallId           *string                 `json:"call_id,omitempty" doc:"The call to join. Required unless the session is text."`
	Text             *bool                   `json:"text,omitempty" doc:"Hold the conversation in writing rather than on a call. Nothing is transcribed and nothing is spoken, so no call is joined and neither speech target is used. Everything between hearing and answering is unchanged: a text session has the same skills, knowledge and tools a call would have had, and its replies arrive as response_delta and responded events on the session's socket." default:"false"`
	ConfigId         *string                 `json:"config_id,omitempty" doc:"An agent config to start from. Everything else in this request overrides what the config says, so a caller can reuse a configuration and still change one thing about this call."`
	Agent            *string                 `json:"agent,omitempty" doc:"The name of an agent config to start from, as an alternative to config_id. It is what a caller actually knows the agent as: \"docs\" rather than an id they never chose. A name matching nothing is refused rather than silently starting an unconfigured agent, and naming both this and config_id is refused too, since there is no sensible answer when they disagree."`
	Incognito        *bool                   `json:"incognito,omitempty" doc:"Hold the conversation and record nothing about it: no session row, no turns, no transcript, and no Stream Chat channel. The session still works exactly as any other while it is running; it simply cannot be found afterwards, which is the point. Forking one is refused, because there is nothing to fork from." default:"false"`
	Title            *string                 `json:"title,omitempty" doc:"What to call the conversation, for a list a person reads. Never shown to the model: what a conversation is called is a label on it rather than part of it."`
	Description      *string                 `json:"description,omitempty" doc:"A longer note about the conversation, searched alongside the title."`
	ProjectId        *string                 `json:"project_id,omitempty" doc:"What the conversation belongs to. Also recorded as the \"project\" cost tag, so spend breaks down by project without the caller labelling it twice. A tag spelled out in tags wins."`
	Custom           *map[string]interface{} `json:"custom,omitempty" doc:"Anything the caller wants to remember about the session, handed back untouched and never read by the router. Sessions can be queried by these, which is what makes them worth writing."`
	ModelOverwrites  *ModelOverwrites        `json:"model_overwrites,omitempty"`
	CallType         *string                 `json:"call_type,omitempty" default:"default"`
	UserId           *string                 `json:"user_id,omitempty" doc:"Who the agent joins the call as." default:"vision-agent"`
	UserName         *string                 `json:"user_name,omitempty" default:"Vision Agent"`
	AgentId          *string                 `json:"agent_id,omitempty" doc:"Keys transcripts and statistics. Empty means the call id."`
	Instructions     *string                 `json:"instructions,omitempty"`
	Greeting         *string                 `json:"greeting,omitempty" doc:"Said on joining without going through the model. Empty means the agent waits to be spoken to."`
	Navigating       *bool                   `json:"navigating,omitempty" doc:"The agent placed this call, so let recordings finish and answer their menus." default:"false"`
	Llm              *string                 `json:"llm,omitempty" doc:"A provider/model or a capability shortcut. Omit it and the config decides, or llm-fast when there is no config. These carry no schema default on purpose: a generated client that filled one in would send it, and a caller naming a config would silently lose the model it configured."`
	Stt              *string                 `json:"stt,omitempty" doc:"Omit it and the config decides, or en-low-latency when there is no config."`
	Tts              *string                 `json:"tts,omitempty" doc:"Omit it and the config decides, or en-low-latency when there is no config."`
	Sts              *string                 `json:"sts,omitempty" doc:"A speech-to-speech target. Naming one makes this a native session: the model hears and speaks for itself, so no transcriber, conversation model or voice is opened. Omit it and the config decides."`
	Search           *string                 `json:"search,omitempty" doc:"Omit it and the config decides, or search-fast when there is no config."`
	Voice            *string                 `json:"voice,omitempty" doc:"Provider-specific voice id."`
	Languages        *[]string               `json:"languages,omitempty" doc:"Language hints, which narrow the candidates in every modality."`
	Keyterms         *[]string               `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong. Up to 100 terms, and providers that cannot be told about vocabulary ignore them."`
	MaxTokens        *int                    `json:"max_tokens,omitempty"`
	Backchannel      *bool                   `json:"backchannel,omitempty" doc:"Murmur while a participant is still talking, the way a person does." default:"false"`
	MinConfidence    *float64                `json:"min_confidence,omitempty" doc:"How sure the transcriber must be before the agent answers rather than checks what was meant."`
	Tools            *[]SessionTool          `json:"tools,omitempty"`
	ToolTimeoutMs    *int                    `json:"tool_timeout_ms,omitempty" doc:"How long the model waits for a tool result. Zero is the default."`
	Tags             *map[string]string      `json:"tags,omitempty" doc:"Cost labels, carried onto every request the session makes."`
	Memory           *SessionMemory          `json:"memory,omitempty"`
	Phone            *SessionPhone           `json:"phone,omitempty"`
	Video            *SessionVideo           `json:"video,omitempty"`
}

type Session struct {
	ConversationId   *string                 `json:"conversation_id,omitempty" doc:"Stream Chat CID to resume; returned for persistent text sessions."`
	Modality         SessionModality         `json:"modality"`
	ContextTruncated *bool                   `json:"context_truncated,omitempty" doc:"Older history was omitted from the model context."`
	Id               string                  `json:"id"`
	CallId           string                  `json:"call_id" doc:"Empty for a text session, which joins no call."`
	Text             *bool                   `json:"text,omitempty" doc:"The conversation is held in writing rather than on a call."`
	CallType         string                  `json:"call_type"`
	UserId           string                  `json:"user_id"`
	AgentId          string                  `json:"agent_id"`
	State            SessionState            `json:"state"`
	CreatedAt        time.Time               `json:"created_at"`
	Llm              *string                 `json:"llm,omitempty" doc:"The provider and model answering, once routing has picked one."`
	Tts              *string                 `json:"tts,omitempty" doc:"The provider and model speaking."`
	Sts              *string                 `json:"sts,omitempty" doc:"The provider and model holding a native conversation, once routing has picked one."`
	Stt              *string                 `json:"stt,omitempty" doc:"The provider and model transcribing, once somebody has been heard."`
	Subagent         *string                 `json:"subagent,omitempty" doc:"The provider and model delegated work runs on."`
	Voice            *string                 `json:"voice,omitempty" doc:"The voice speaking, in the provider's own terms. It is the provider's default when the session asked for none."`
	Mode             *SessionMode            `json:"mode,omitempty"`
	Video            *SessionVideo           `json:"video,omitempty"`
	Instructions     *string                 `json:"instructions,omitempty"`
	Agent            *string                 `json:"agent,omitempty" doc:"The name the agent was addressed as. Recorded on the session as well as the config id, so renaming a config does not rewrite what older sessions were opened against."`
	ConfigId         *string                 `json:"config_id,omitempty" doc:"The agent config the session ran under, empty for one that spelled itself out."`
	Incognito        *bool                   `json:"incognito,omitempty" doc:"Nothing about this session was recorded. It is reported so a caller can see that what they asked for is what they got, but it is never read back from storage: an incognito session has no row to read it from."`
	Title            *string                 `json:"title,omitempty"`
	Description      *string                 `json:"description,omitempty"`
	ProjectId        *string                 `json:"project_id,omitempty"`
	Custom           *map[string]interface{} `json:"custom,omitempty"`
	ModelOverwrites  *ModelOverwrites        `json:"model_overwrites,omitempty"`
	ForkedFrom       *string                 `json:"forked_from,omitempty" doc:"The session this one continued from, empty for one opened fresh."`
	ClosedAt         *time.Time              `json:"closed_at,omitempty" doc:"When the session ended. Absent while it is still running."`
	LastResponseAt   *time.Time              `json:"last_response_at,omitempty" doc:"When the agent last answered. This is what a most-recently-used ordering of conversations reads, since a session renamed long after it ended has not become more recent."`
}

type ModelOverwritesThinking string

const (
	ModelOverwritesThinkingHigh    ModelOverwritesThinking = "high"
	ModelOverwritesThinkingLow     ModelOverwritesThinking = "low"
	ModelOverwritesThinkingMedium  ModelOverwritesThinking = "medium"
	ModelOverwritesThinkingMinimal ModelOverwritesThinking = "minimal"
	ModelOverwritesThinkingNone    ModelOverwritesThinking = "none"
)

// Valid indicates whether the value is a known member of the ModelOverwritesThinking enum.
func (e ModelOverwritesThinking) Valid() bool {
	switch e {
	case ModelOverwritesThinkingHigh:
		return true
	case ModelOverwritesThinkingLow:
		return true
	case ModelOverwritesThinkingMedium:
		return true
	case ModelOverwritesThinkingMinimal:
		return true
	case ModelOverwritesThinkingNone:
		return true
	default:
		return false
	}
}

type ModelOverwritesVerbosity string

const (
	ModelOverwritesVerbosityHigh   ModelOverwritesVerbosity = "high"
	ModelOverwritesVerbosityLow    ModelOverwritesVerbosity = "low"
	ModelOverwritesVerbosityMedium ModelOverwritesVerbosity = "medium"
)

// Valid indicates whether the value is a known member of the ModelOverwritesVerbosity enum.
func (e ModelOverwritesVerbosity) Valid() bool {
	switch e {
	case ModelOverwritesVerbosityHigh:
		return true
	case ModelOverwritesVerbosityLow:
		return true
	case ModelOverwritesVerbosityMedium:
		return true
	default:
		return false
	}
}

type ModelOverwrites struct {
	Llm             *string                   `json:"llm,omitempty" doc:"A provider/model or a capability shortcut, in place of the config's."`
	Stt             *string                   `json:"stt,omitempty"`
	Tts             *string                   `json:"tts,omitempty"`
	Sts             *string                   `json:"sts,omitempty" doc:"A speech-to-speech target. Naming one here makes the session native even if the config did not, which means no transcriber, model or voice is opened."`
	Search          *string                   `json:"search,omitempty"`
	Thinking        *ModelOverwritesThinking  `json:"thinking,omitempty" doc:"How hard to reason before answering. It becomes the reasoning effort on the request, which is the vocabulary the providers that support one already speak, and means nothing to a model that does not reason." enum:"none,minimal,low,medium,high"`
	Temperature     *float64                  `json:"temperature,omitempty" doc:"How random the answer is. Omitted leaves the provider's own default, which is not the same as zero: zero is a real request for a deterministic model."`
	MaxOutputTokens *int                      `json:"max_output_tokens,omitempty" doc:"Caps the reply, reasoning included. Omitted leaves the provider's default."`
	Verbosity       *ModelOverwritesVerbosity `json:"verbosity,omitempty" doc:"How much detail to give. Dropped for models that do not take it." enum:"low,medium,high"`
}

func (*ModelOverwrites) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What to change about the models for one session, over whatever its agent config decided.\n" +
		"It is one object rather than a dozen fields at the top level because it is one idea: " +
		"everything here overrides the config, and a caller reading a session back wants to see what " +
		"they changed in one place rather than diffed against a config they would have to fetch. " +
		"Only the safe knobs are here. Instructions and tools are not, because a caller able to " +
		"rewrite those could make a session impersonate a different agent."
	return schema
}

type SessionToolExecutor string

const (
	SessionToolExecutorClient SessionToolExecutor = "client"
	SessionToolExecutorServer SessionToolExecutor = "server"
)

// Valid indicates whether the value is a known member of the SessionToolExecutor enum.
func (e SessionToolExecutor) Valid() bool {
	switch e {
	case SessionToolExecutorClient:
		return true
	case SessionToolExecutorServer:
		return true
	default:
		return false
	}
}

type SessionTool struct {
	Name         string                  `json:"name"`
	Description  string                  `json:"description" doc:"What the model is told the tool does, which is the whole of how it decides when to reach for one."`
	Parameters   *map[string]interface{} `json:"parameters,omitempty" doc:"A JSON Schema object describing the arguments."`
	Executor     *SessionToolExecutor    `json:"executor,omitempty" doc:"Who runs it. A client tool runs on a person's device: in a persistent conversation its call is shown as awaiting the device of the person whose command it answers (their user and the command's client_id), with its arguments, which every channel member can read. The caller still answers it over the events socket, once the device has reported. Defaults to server." enum:"server,client"`
	DisplayTitle *string                 `json:"display_title,omitempty" doc:"What a call is doing, in words for the people in the conversation, such as \"Checking your location\". Shown on the reply's ai_tool_call attachment." maxLength:"80"`
}

func (*SessionTool) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "One of the caller's own functions. The model is offered it by name and description; running " +
		"it is the caller's business, over the events socket."
	schema.Properties["executor"].Extensions = map[string]any{"x-enum-varnames": []string{"SessionToolExecutorServer", "SessionToolExecutorClient"}}
	return schema
}

type SessionMemory struct {
	UserId *string            `json:"user_id,omitempty" doc:"Who the memories belong to. Empty means the customer."`
	AppId  *string            `json:"app_id,omitempty" doc:"Separates two deployments sharing one memory account."`
	Filter *map[string]string `json:"filter,omitempty" doc:"The caller's own labels, which narrow recall further. They cannot widen it: a filter is applied alongside the user id, never instead of it."`
}

func (*SessionMemory) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who the session's memories are about. Without a user id nothing is recalled or stored, " +
		"which is the case for a call with nobody identified on it."
	return schema
}

type SessionPhone struct {
	Number       string  `json:"number" doc:"One of the customer's own numbers, written as +15551234567."`
	Vendor       *string `json:"vendor,omitempty" doc:"Who carries an outbound leg."`
	VendorCallId *string `json:"vendor_call_id,omitempty" doc:"The outbound leg, set for a call the agent placed. Without one the agent has no keypad to press at."`
}

func (*SessionPhone) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The number the session acts from, which is what turns transferring on."
	return schema
}

type SessionModality string

const (
	SessionModalityText  SessionModality = "text"
	SessionModalityVideo SessionModality = "video"
	SessionModalityVoice SessionModality = "voice"
)

// Valid indicates whether the value is a known member of the SessionModality enum.
func (e SessionModality) Valid() bool {
	switch e {
	case SessionModalityText:
		return true
	case SessionModalityVideo:
		return true
	case SessionModalityVoice:
		return true
	default:
		return false
	}
}

func (SessionModality) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SessionModality", "How the user took part: text for a conversation held in writing, voice for a call, and "+
		"video once the agent has seen the user's video. It only moves up, from text or voice to "+
		"video.",
		string(SessionModalityText), string(SessionModalityVoice), string(SessionModalityVideo))
}

type SessionState string

const (
	Ended SessionState = "ended"
	Live  SessionState = "live"
)

// Valid indicates whether the value is a known member of the SessionState enum.
func (e SessionState) Valid() bool {
	switch e {
	case Ended:
		return true
	case Live:
		return true
	default:
		return false
	}
}

func (SessionState) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SessionState", "Whether the agent is still in the call.",
		string(Live), string(Ended))
}

type createSessionRequest struct {
	Body CreateSessionRequest
}

type sessionResponse struct {
	Body Session
}

func (s *Server) registerSessionCreate(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "createSession",
		Method:      http.MethodPost,
		Path:        "/v1/agents/sessions",
		Summary:     "Join a call as a voice agent",
		Description: "The whole conversation runs here: the agent joins the call, transcribes what it " +
			"hears, answers it and speaks back, all through the routers. The caller keeps " +
			"the session id and watches the conversation over the events socket.\n" +
			"It returns once the agent is in the call, so a session that comes back is one " +
			"that is already listening. Tools declared here are the caller's own: the model " +
			"asks for them over the events socket and waits for the caller to answer.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The agent is in the call"},
			"409": errorResponse("A session with that id already exists"),
		},
		Errors:       []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusNotFound},
		Extensions:   map[string]any{clientAccessibleExtension: true},
		MaxBodyBytes: largeBody,
	}, s.createSession)
}

// createSession joins a call and returns the session running it.
func (s *Server) createSession(ctx context.Context, request *createSessionRequest) (*sessionResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.sessions == nil {
		return nil, huma.Error404NotFound(noSessions)
	}

	// A config is read before the session is created rather than inside it, so a caller
	// naming one that is not theirs is told so instead of getting a session that quietly
	// ignored it.
	config, failure := s.configFor(ctx, customerID, request.Body.ConfigId, request.Body.Agent)
	if failure != nil {
		if failure.status == notFound {
			return nil, huma.Error404NotFound(failure.message)
		}
		return nil, huma.Error400BadRequest(failure.message)
	}

	spec := specOf(request.Body, customerID, config)
	// Who asked comes from the credential rather than from specOf, which merges the request
	// with the config and so only ever sees what the caller was willing to say about
	// themselves. Both halves are recorded, because the name is only worth what the kind
	// says it is: this pair is what the session is owned by and what every later request
	// for it is matched against.
	spec.Caller = CallerFrom(ctx)
	spec.CallerKind = KindFrom(ctx)
	created, err := s.sessions.Create(ctx, spec)
	if errors.Is(err, session.ErrSessionExists) {
		return nil, huma.Error409Conflict(err.Error())
	}
	if err != nil {
		// Everything that can go wrong here is the caller's spec or a provider that would
		// not start, and both are worth reading rather than a 500 with the detail in a
		// log the caller cannot see.
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &sessionResponse{Body: sessionOf(created)}, nil
}
