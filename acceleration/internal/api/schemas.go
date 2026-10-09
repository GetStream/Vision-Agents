package api

import (
	"encoding/json"
	"reflect"
	"time"

	"github.com/danielgtaylor/huma/v2"
)

// AgentDispatch What the agent leaves to the customer's own server, which waits on /v1/dispatch. Omitted settings are disabled.
type AgentDispatch struct {
	IncomingCall *DispatchSetting `json:"incoming_call,omitempty" doc:"A call to one of the customer's numbers is handed to a dispatch worker. Every inbound call already is, since a number is not tied to an agent config."`
	Text         *DispatchSetting `json:"text,omitempty" doc:"An end user's message is handed to a dispatch worker, with the session it was written to, instead of being answered by the model. The worker answers by creating a response on that session with a server-side credential, passing the message's command_id when it has one; that is the only text the model answers."`
}

func (*AgentDispatch) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What the agent leaves to the customer's own server, which waits on /v1/dispatch. Omitted settings are disabled."
	return schema
}

// AgentLog is the AgentLog schema.
type AgentLog struct {
	AgentId    string                  `json:"agent_id"`
	ConfigId   string                  `json:"config_id"`
	Cursor     *string                 `json:"cursor,omitempty"`
	Details    *map[string]interface{} `json:"details,omitempty"`
	EventType  string                  `json:"event_type"`
	Id         string                  `json:"id"`
	IngestedAt time.Time               `json:"ingested_at"`
	Message    string                  `json:"message"`
	OccurredAt time.Time               `json:"occurred_at"`
	SessionId  string                  `json:"session_id"`
	Severity   AgentLogSeverity        `json:"severity" enum:"info,warn,error"`
	Source     AgentLogSource          `json:"source" enum:"user,agent,tool,system"`
	UserId     *string                 `json:"user_id,omitempty"`
}

// AgentLogSeverity is the AgentLogSeverity schema.
type AgentLogSeverity string

// Defines values for AgentLogSeverity.
const (
	AgentLogSeverityError AgentLogSeverity = "error"
	AgentLogSeverityInfo  AgentLogSeverity = "info"
	AgentLogSeverityWarn  AgentLogSeverity = "warn"
)

// Valid indicates whether the value is a known member of the AgentLogSeverity enum.
func (e AgentLogSeverity) Valid() bool {
	switch e {
	case AgentLogSeverityError:
		return true
	case AgentLogSeverityInfo:
		return true
	case AgentLogSeverityWarn:
		return true
	default:
		return false
	}
}

// AgentLogSource is the AgentLogSource schema.
type AgentLogSource string

// Defines values for AgentLogSource.
const (
	Agent  AgentLogSource = "agent"
	System AgentLogSource = "system"
	Tool   AgentLogSource = "tool"
	User   AgentLogSource = "user"
)

// Valid indicates whether the value is a known member of the AgentLogSource enum.
func (e AgentLogSource) Valid() bool {
	switch e {
	case Agent:
		return true
	case System:
		return true
	case Tool:
		return true
	case User:
		return true
	default:
		return false
	}
}

// AgentMode Whether the agent is spoken to or written to. A voice agent joins a call, transcribes what it hears and speaks its replies. A text agent holds the same conversation in writing, so it uses neither speech target and a session created from it needs no call to join.
type AgentMode string

// Defines values for AgentMode.
const (
	AgentModeText  AgentMode = "text"
	AgentModeVoice AgentMode = "voice"
)

// Valid indicates whether the value is a known member of the AgentMode enum.
func (e AgentMode) Valid() bool {
	switch e {
	case AgentModeText:
		return true
	case AgentModeVoice:
		return true
	default:
		return false
	}
}

// Call is the Call schema.
type Call struct {
	AgentId         string             `json:"agent_id" doc:"Which agent ran it, and where its transcript is kept."`
	CallId          string             `json:"call_id"`
	CampaignId      *string            `json:"campaign_id,omitempty"`
	ConfigId        *string            `json:"config_id,omitempty"`
	ContactId       *string            `json:"contact_id,omitempty"`
	Direction       CallDirection      `json:"direction" enum:"inbound,outbound"`
	EndedAt         *time.Time         `json:"ended_at,omitempty" doc:"Absent while the call is still running."`
	FromNumber      *string            `json:"from_number,omitempty"`
	Id              string             `json:"id" doc:"The session that ran the call, which is what it is held by."`
	Instructions    *string            `json:"instructions,omitempty" doc:"What the agent was told to be on this call."`
	Llm             *string            `json:"llm,omitempty" doc:"The target that held the conversation."`
	LlmUsed         *string            `json:"llm_used,omitempty" doc:"The provider/model that held the conversation."`
	Mode            *SessionMode       `json:"mode,omitempty"`
	ReviewNotes     *string            `json:"review_notes,omitempty"`
	ReviewScore     *int               `json:"review_score,omitempty" doc:"How well the agent handled it, from 1 to 5."`
	Skills          *[]string          `json:"skills,omitempty" doc:"What the fast model could hand to the subagent. The instructions behind each name are in the skill registry."`
	StartedAt       time.Time          `json:"started_at"`
	Sts             *string            `json:"sts,omitempty" doc:"The speech-to-speech target, for a native call, on the same terms as stt."`
	StsUsed         *string            `json:"sts_used,omitempty" doc:"The provider/model that held a native call, on the same terms as stt_used."`
	Stt             *string            `json:"stt,omitempty" doc:"The transcription target the call ran with, after a session's overrides were folded into whatever config it named. This is what was asked for rather than what each turn resolved to: a shortcut is several models and routing fails over between them, so per-turn providers are in the request rows."`
	SttUsed         *string            `json:"stt_used,omitempty" doc:"The provider/model that transcribed, once routing picked one. Empty until somebody has been heard, and the last one that served if routing failed over."`
	ThinkingLlm     *string            `json:"thinking_llm,omitempty" doc:"The target delegated work ran on. Empty means nothing was delegated, which also means the skills below were never offered. A text call names its llm, which runs its skills too."`
	ThinkingLlmUsed *string            `json:"thinking_llm_used,omitempty" doc:"The provider/model delegated work ran on. Empty when nothing was handed over, or when the thinking target was never reached."`
	Summary         *string            `json:"summary,omitempty" doc:"What a model made of the call, written once it was over."`
	Tags            *map[string]string `json:"tags,omitempty"`
	ToNumber        *string            `json:"to_number,omitempty"`
	Tts             *string            `json:"tts,omitempty" doc:"The voice target, on the same terms as stt."`
	TtsUsed         *string            `json:"tts_used,omitempty" doc:"The provider/model that spoke, on the same terms as stt_used."`
	Usage           *CallUsage         `json:"usage,omitempty"`
	UserId          *string            `json:"user_id,omitempty" doc:"Who the agent spoke to, as the client's own token named them. Empty for a call the customer's backend opened, and for telephony, where the number is the name."`
	Voice           *string            `json:"voice,omitempty" doc:"The voice the call asked for, in the provider's own terms. Empty means the provider's default."`
	VoiceUsed       *string            `json:"voice_used,omitempty" doc:"The voice that spoke, which is the provider's default when none was asked for. Known only while the call is running."`
}

func (*Call) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["review_score"].Format = ""
	return schema
}

// CallEvent is the CallEvent schema.
type CallEvent struct {
	At          time.Time    `json:"at"`
	Kind        DecisionKind `json:"kind"`
	LatencyMs   *float64     `json:"latency_ms,omitempty" doc:"What the flow controller took to rule, or what the subagent took to answer. Zero where nothing was asked."`
	Participant *string      `json:"participant,omitempty" doc:"Who it concerned."`
	Reason      string       `json:"reason" doc:"Why the conversation chose it, in words."`
	Said        *string      `json:"said,omitempty" doc:"What was heard, what the agent decided to say, or what the subagent came back with."`
	TurnId      *string      `json:"turn_id,omitempty" doc:"The exchange it was about, which lines it up against that turn's timings."`
}

// ClassifyQuestion is the ClassifyQuestion schema.
type ClassifyQuestion struct {
	Instructions string               `json:"instructions" example:"Is the customer asking for a refund?"`
	Levels       *[]string            `json:"levels,omitempty" doc:"A score's levels, in order, each describing a concrete situation."`
	No           *string              `json:"no,omitempty" doc:"What no means for a noul, where the instructions do not say it."`
	Options      *map[string]string   `json:"options,omitempty" doc:"A choice's options, each with a description of what it covers or an empty string where the name says it. Include one for \"none of these\" whenever the options may not cover an input."`
	Type         ClassifyQuestionType `json:"type"`
	Yes          *string              `json:"yes,omitempty" doc:"What yes means for a noul, where the instructions do not say it."`
}

// ClassifyQuestionType noul is yes or no, answered as the probability of yes. choice picks one of named options. score places the state along ordered levels.
type ClassifyQuestionType string

// Defines values for ClassifyQuestionType.
const (
	Choice ClassifyQuestionType = "choice"
	Noul   ClassifyQuestionType = "noul"
	Score  ClassifyQuestionType = "score"
)

// Valid indicates whether the value is a known member of the ClassifyQuestionType enum.
func (e ClassifyQuestionType) Valid() bool {
	switch e {
	case Choice:
		return true
	case Noul:
		return true
	case Score:
		return true
	default:
		return false
	}
}

func (ClassifyQuestionType) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ClassifyQuestionType", "noul is yes or no, answered as the probability of yes. choice picks one of named options. score places the state along ordered levels.", "noul", "choice", "score")
}

// CommandReceipt is the CommandReceipt schema.
type CommandReceipt struct {
	AssistantMessageId string `json:"assistant_message_id"`
	CommandId          string `json:"command_id"`
	Duplicate          bool   `json:"duplicate" doc:"True when this command already exists and no new inference was started."`
	State              string `json:"state" doc:"Latest locally recorded response state; an interrupted command is never automatically rerun."`
	UserMessageId      string `json:"user_message_id"`
}

// ContentPart is the ContentPart schema.
type ContentPart struct {
	union json.RawMessage
}

// AsTextContentPart returns the union data inside the ContentPart as a TextContentPart
func (t ContentPart) AsTextContentPart() (TextContentPart, error) {
	var body TextContentPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromTextContentPart overwrites any union data inside the ContentPart as the provided TextContentPart
func (t *ContentPart) FromTextContentPart(v TextContentPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// AsImageContentPart returns the union data inside the ContentPart as a ImageContentPart
func (t ContentPart) AsImageContentPart() (ImageContentPart, error) {
	var body ImageContentPart
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromImageContentPart overwrites any union data inside the ContentPart as the provided ImageContentPart
func (t *ContentPart) FromImageContentPart(v ImageContentPart) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

func (t ContentPart) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *ContentPart) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// CreateSessionRequest is the CreateSessionRequest schema.
type CreateSessionRequest struct {
	Agent             *string                    `json:"agent,omitempty" doc:"The name of an agent config to start from, as an alternative to config_id. It is what a caller actually knows the agent as: \"docs\" rather than an id they never chose. A name matching nothing is refused rather than silently starting an unconfigured agent, and naming both this and config_id is refused too, since there is no sensible answer when they disagree."`
	AgentId           *string                    `json:"agent_id,omitempty" doc:"Keys transcripts and statistics. Empty means the call id."`
	Backchannel       *bool                      `json:"backchannel,omitempty" doc:"Murmur while a participant is still talking, the way a person does." default:"false"`
	CallId            *string                    `json:"call_id,omitempty" doc:"The call to join. Required unless the session is text."`
	CallType          *string                    `json:"call_type,omitempty" default:"default"`
	ConnectorBindings *[]SessionConnectorBinding `json:"connector_bindings,omitempty" maxItems:"64" doc:"The connection to use for each of the agent config's connector bindings chosen per session (connection.type session), by its alias. Each must be the verified caller's own connection to the binding's connector: an end user's, or the one a backend names with X-Stream-User-Id, never an anonymous caller's or a guest's. A binding with a fixed connection cannot be given one here, and an alias the config does not declare is refused. A session binding given none here uses the caller's own connection to its connector when exactly one of theirs is connected. Otherwise, with none connected or more than one, a required binding fails the session; an optional one is left out and reported with a connector_unavailable event (no_selection). A fork chooses the same connections again, against the config as it is then and the caller asking for the fork."`
	ConfigId          *string                    `json:"config_id,omitempty" doc:"An agent config to start from. Everything else in this request overrides what the config says, so a caller can reuse a configuration and still change one thing about this call."`
	ContextTruncated  *bool                      `json:"context_truncated,omitempty" doc:"Older history was omitted from the model context."`
	ConversationId    *string                    `json:"conversation_id,omitempty" doc:"Stream Chat CID to resume; returned for persistent text sessions."`
	Custom            *map[string]interface{}    `json:"custom,omitempty" doc:"Anything the caller wants to remember about the session, handed back untouched and never read by the router. Sessions can be queried by these, which is what makes them worth writing."`
	Description       *string                    `json:"description,omitempty" doc:"A longer note about the conversation, searched alongside the title."`
	Greeting          *string                    `json:"greeting,omitempty" doc:"Said on joining without going through the model. Empty means the agent waits to be spoken to."`
	History           *[]HistoryMessage          `json:"history,omitempty" doc:"The conversation so far, for a backend that keeps its own: a thread in its own Slack app, say, that outlives any one session. Send it when a session closed and the thread goes on: open a new session with the thread's messages here, oldest first, then send the message to answer to the responses endpoint. The model is handed them before the first response, as a resumed conversation's history is. They are recorded nowhere, as turns, transcript or Chat messages, so add incognito to keep nothing at all. Up to 100 messages and 60000 characters of text, the most a session reads back of a conversation the router kept; more is refused rather than cut. Not with conversation_id, which reads the history the router kept. Server-side only: a device sending it is refused with a 403, because an assistant message puts words in the agent's mouth." maxItems:"100"`
	Id                *string                    `json:"id,omitempty" doc:"The id to hold the session by, so a caller can know it before the session exists. It must be a UUID nobody has used for a session before. Omitted, the router generates a UUIDv7."`
	Incognito         *bool                      `json:"incognito,omitempty" doc:"Hold the conversation and record nothing about it: no session row, no turns, no transcript, and no Stream Chat channel. The session still works exactly as any other while it is running; it simply cannot be found afterwards, which is the point. Forking one is refused, because there is nothing to fork from." default:"false"`
	Instructions      *string                    `json:"instructions,omitempty" doc:"The system prompt, over what the config says. Server-side only: a device sending it is refused with a 403, as it is on updateSession, because what the agent is told to be is the backend's to decide."`
	Keyterms          *[]string                  `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong. Up to 100 terms, and providers that cannot be told about vocabulary ignore them."`
	Languages         *[]string                  `json:"languages,omitempty" doc:"Language hints, which narrow the candidates in every modality."`
	Llm               *string                    `json:"llm,omitempty" doc:"A provider/model or a capability shortcut. Omit it and the config decides, or llm-fast when there is no config. These carry no schema default on purpose: a generated client that filled one in would send it, and a caller naming a config would silently lose the model it configured."`
	MaxTokens         *int                       `json:"max_tokens,omitempty"`
	Memory            *SessionMemory             `json:"memory,omitempty"`
	MinConfidence     *float64                   `json:"min_confidence,omitempty" doc:"How sure the transcriber must be before the agent answers rather than checks what was meant."`
	ModelOverwrites   *ModelOverwrites           `json:"model_overwrites,omitempty"`
	Navigating        *bool                      `json:"navigating,omitempty" doc:"The agent placed this call, so let recordings finish and answer their menus." default:"false"`
	Phone             *SessionPhone              `json:"phone,omitempty"`
	ProjectId         *string                    `json:"project_id,omitempty" doc:"What the conversation belongs to. Also recorded as the \"project\" cost tag, so spend breaks down by project without the caller labelling it twice. A tag spelled out in tags wins."`
	Search            *string                    `json:"search,omitempty" doc:"Omit it and the config decides, or search-fast when there is no config."`
	Sts               *string                    `json:"sts,omitempty" doc:"A speech-to-speech target. Naming one makes this a native session: the model hears and speaks for itself, so no transcriber, conversation model or voice is opened. Omit it and the config decides."`
	Stt               *string                    `json:"stt,omitempty" doc:"Omit it and the config decides, or en-low-latency when there is no config."`
	Tags              *map[string]string         `json:"tags,omitempty" doc:"Cost labels, carried onto every request the session makes."`
	Text              *bool                      `json:"text,omitempty" doc:"Hold the conversation in writing rather than on a call. Nothing is transcribed and nothing is spoken, so no call is joined and neither speech target is used. Everything between hearing and answering is unchanged: a text session has the same skills, knowledge and tools a call would have had, and its replies arrive as response_delta and responded events on the session's socket." default:"false"`
	Title             *string                    `json:"title,omitempty" doc:"What to call the conversation, for a list a person reads, until the router names a persistent one for what was said. Never shown to the model: what a conversation is called is a label on it rather than part of it."`
	ToolTimeoutMs     *int                       `json:"tool_timeout_ms,omitempty" doc:"How long the model waits for a tool result. Zero is the default."`
	Tools             *[]SessionTool             `json:"tools,omitempty"`
	Tts               *string                    `json:"tts,omitempty" doc:"Omit it and the config decides, or en-low-latency when there is no config."`
	UserId            *string                    `json:"user_id,omitempty" doc:"Who the agent joins the call as." default:"vision-agent"`
	UserName          *string                    `json:"user_name,omitempty" default:"Vision Agent"`
	Video             *SessionVideo              `json:"video,omitempty"`
	Voice             *string                    `json:"voice,omitempty" doc:"Provider-specific voice id."`
}

func (*CreateSessionRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_tokens"].Format = ""
	schema.Properties["tool_timeout_ms"].Format = ""
	return schema
}

// SessionConnectorBinding is the connection a session uses for one of its config's connector
// bindings. The alias pattern is AgentConnectorBinding's, so a name no binding could have is
// refused before the session looks for it. 64 is the most bindings a config holds
// (AgentConnectorBinding), so the most a session can choose for.
type SessionConnectorBinding struct {
	Name         string `json:"name" pattern:"^[a-z]([a-z0-9_-]{0,61}[a-z0-9-])?$" doc:"The binding's alias in the agent config."`
	ConnectionId string `json:"connection_id" minLength:"1" doc:"The caller's own connection to the binding's connector."`
}

func (*SessionConnectorBinding) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The connection a session uses for one of its agent config's connector bindings " +
		"chosen per session. Only a reference: the credential stays sealed on the connection."
	schema.AdditionalProperties = false
	return schema
}

// DataChangeOp is the DataChangeOp schema.
type DataChangeOp string

// Defines values for DataChangeOp.
const (
	Delete DataChangeOp = "delete"
	Insert DataChangeOp = "insert"
	Update DataChangeOp = "update"
)

// Valid indicates whether the value is a known member of the DataChangeOp enum.
func (e DataChangeOp) Valid() bool {
	switch e {
	case Delete:
		return true
	case Insert:
		return true
	case Update:
		return true
	default:
		return false
	}
}

// DataPolicy What a caller requires of what happens to what they send: the audio they had transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a description: a request naming one is only routed to a model whose declared handling meets it, and if none does the request is refused rather than sent somewhere that does not.
type DataPolicy struct {
	AllowTraining *bool   `json:"allow_training,omitempty" doc:"False requires a provider that has said it does not train on what it is sent. Omitting this asks nothing. A provider that has published nothing either way counts as not having said no."`
	Retention     *string `json:"retention,omitempty" doc:"The longest a provider may keep this audio - none, or a duration such as 30d or 24h. Omitting it asks nothing." example:"none"`
}

func (*DataPolicy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a caller requires of what happens to what they send: the audio they had transcribed, or the text they had spoken and the voice speaking it. This is a requirement rather than a description: a request naming one is only routed to a model whose declared handling meets it, and if none does the request is refused rather than sent somewhere that does not."
	return schema
}

// DispatchSetting Whether this kind of work is left to the customer's own dispatch worker.
type DispatchSetting string

// Defines values for DispatchSetting.
const (
	Disabled DispatchSetting = "disabled"
	Enabled  DispatchSetting = "enabled"
)

// Valid indicates whether the value is a known member of the DispatchSetting enum.
func (e DispatchSetting) Valid() bool {
	switch e {
	case Disabled:
		return true
	case Enabled:
		return true
	default:
		return false
	}
}

// Harness Which harness the agent's sessions run: what hands work to the subagent, loads skills, compacts the conversation and starts the sandbox. Set on the agent, never on a session. Omit it for the default, the only one there is.
type Harness string

// Defines values for Harness.
const (
	Default Harness = "default"
)

// Valid indicates whether the value is a known member of the Harness enum.
func (e Harness) Valid() bool {
	switch e {
	case Default:
		return true
	default:
		return false
	}
}

// HistoryMessage is one message of a conversation the caller kept itself. Its name, text
// and created_at are what a TranscriptMessage calls them; a transcript line is not reused
// whole because its speaker and created_at are required, and a caller's thread may know
// neither.
//
// The 256 and 60000 here and the 100 on CreateSessionRequest.history are
// conversation.MaxAuthorName, MaxHistoryRunes and MaxHistoryMessages, which a tag cannot
// name; session.Spec.Normalize refuses past the constants whatever the tags say.
type HistoryMessage struct {
	CreatedAt *time.Time  `json:"created_at,omitempty" doc:"When it was said. The model is shown it beside a person's message, so it can tell an hour ago from just now."`
	Name      *string     `json:"name,omitempty" doc:"Who said it, when several people share the thread. The model is shown it as a label, never as who is asking now." maxLength:"256"`
	Role      HistoryRole `json:"role"`
	Text      string      `json:"text" minLength:"1" maxLength:"60000"`
}

// HistoryRole user is what a person said, assistant what the agent answered.
type HistoryRole string

// Defines values for HistoryRole.
const (
	HistoryRoleAssistant HistoryRole = "assistant"
	HistoryRoleUser      HistoryRole = "user"
)

func (HistoryRole) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "HistoryRole", "user is what a person said, assistant what the agent answered. These are the only turns a resumed conversation hands the model; instructions say anything a system message would.", "user", "assistant")
	// AgentLogSource has user too, and oapi-codegen prefixes every constant of two enums
	// sharing a value, which would rename the Go SDK's User, Agent, System and Tool. Naming
	// these keeps them, as ConnectionOwnerType does.
	registry.Map()["HistoryRole"].Extensions = map[string]any{
		"x-enum-varnames": []string{"HistoryRoleUser", "HistoryRoleAssistant"},
	}
	return ref
}

// ImageContentPart is the ImageContentPart schema.
type ImageContentPart struct {
	ImageUrl ImageSource          `json:"image_url"`
	Type     ImageContentPartType `json:"type" enum:"image_url"`
}

// ImageContentPartType is the ImageContentPartType schema.
type ImageContentPartType string

// Defines values for ImageContentPartType.
const (
	ImageUrl ImageContentPartType = "image_url"
)

// Valid indicates whether the value is a known member of the ImageContentPartType enum.
func (e ImageContentPartType) Valid() bool {
	switch e {
	case ImageUrl:
		return true
	default:
		return false
	}
}

// ImageGenerationStatus Whether the pictures were drawn. A failed generation says why in error_code and error.
type ImageGenerationStatus string

// Defines values for ImageGenerationStatus.
const (
	ImageGenerationStatusCompleted ImageGenerationStatus = "completed"
	ImageGenerationStatusFailed    ImageGenerationStatus = "failed"
)

// Valid indicates whether the value is a known member of the ImageGenerationStatus enum.
func (e ImageGenerationStatus) Valid() bool {
	switch e {
	case ImageGenerationStatusCompleted:
		return true
	case ImageGenerationStatusFailed:
		return true
	default:
		return false
	}
}

func (ImageGenerationStatus) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "ImageGenerationStatus", "Whether the pictures were drawn. A failed generation says why in error_code and error.", "completed", "failed")
}

// ImageOptions Where to draw and what the picture should be. A size, shape, seed, negative prompt or format narrows the candidates to the models that declared it, so it is either honoured or the request is refused.
type ImageOptions struct {
	AspectRatio    *string                   `json:"aspect_ratio,omitempty" doc:"The shape, for the models that are asked for one rather than a size." pattern:"^[0-9]+:[0-9]+$" example:"1:1"`
	N              *int                      `json:"n,omitempty" doc:"How many pictures to draw." minimum:"1" maximum:"4" default:"1"`
	NegativePrompt *string                   `json:"negative_prompt,omitempty" doc:"What to keep out of the picture."`
	OutputFormat   *ImageOptionsOutputFormat `json:"output_format,omitempty" doc:"The encoding, on a model that can be asked for one." enum:"png,jpeg"`
	Providers      *[]string                 `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands."`
	Seed           *int64                    `json:"seed,omitempty" doc:"Draws the same picture again from the same prompt, on a model that reads one." minimum:"0"`
	Size           *string                   `json:"size,omitempty" doc:"Width by height in pixels, for the models that take a size." pattern:"^[0-9]+x[0-9]+$" example:"1024x1024"`
	Target         *string                   `json:"target,omitempty" doc:"A provider/model or a capability shortcut. Defaults to image-fast." example:"image-fast"`
}

func (*ImageOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["n"].Format = ""
	schema.Description = "Where to draw and what the picture should be. A size, shape, seed, negative prompt or format narrows the candidates to the models that declared it, so it is either honoured or the request is refused."
	return schema
}

// ImageOptionsOutputFormat is the ImageOptionsOutputFormat schema.
type ImageOptionsOutputFormat string

// Defines values for ImageOptionsOutputFormat.
const (
	Jpeg ImageOptionsOutputFormat = "jpeg"
	Png  ImageOptionsOutputFormat = "png"
)

// Valid indicates whether the value is a known member of the ImageOptionsOutputFormat enum.
func (e ImageOptionsOutputFormat) Valid() bool {
	switch e {
	case Jpeg:
		return true
	case Png:
		return true
	default:
		return false
	}
}

// ImageSource is the ImageSource schema.
type ImageSource struct {
	Detail *ImageSourceDetail `json:"detail,omitempty" enum:"auto,low,high"`
	Url    string             `json:"url" doc:"Absolute HTTP(S) URL or base64 image data URI."`
}

// ImageSourceDetail is the ImageSourceDetail schema.
type ImageSourceDetail string

// Defines values for ImageSourceDetail.
const (
	ImageSourceDetailAuto ImageSourceDetail = "auto"
	ImageSourceDetailHigh ImageSourceDetail = "high"
	ImageSourceDetailLow  ImageSourceDetail = "low"
)

// Valid indicates whether the value is a known member of the ImageSourceDetail enum.
func (e ImageSourceDetail) Valid() bool {
	switch e {
	case ImageSourceDetailAuto:
		return true
	case ImageSourceDetailHigh:
		return true
	case ImageSourceDetailLow:
		return true
	default:
		return false
	}
}

// KnowledgeDocument is the KnowledgeDocument schema.
type KnowledgeDocument struct {
	Source string `json:"source" doc:"Where the document came from, as a reader would recognise it. Passage ids are keyed by it, so posting the same source again replaces what it wrote before." example:"pricing.md"`
	Text   string `json:"text" doc:"The document, whole. It is cut into passages here."`
}

// KnowledgePassage is the KnowledgePassage schema.
type KnowledgePassage struct {
	Id     string `json:"id"`
	Source string `json:"source" doc:"The heading or file the passage sits under."`
	Text   string `json:"text"`
}

// LlmOptions How this config answers. The names are the response parameters the router already speaks rather than a second vocabulary for the same things. The system prompt is not among them: what the model answers under belongs to the agent asking, not to the config that decides where the asking goes.
type LlmOptions struct {
	Format          *LlmOptionsFormat          `json:"format,omitempty" doc:"Whether the answer is prose or a JSON object." enum:"text,json_object"`
	MaxOutputTokens *int                       `json:"max_output_tokens,omitempty"`
	Metadata        *map[string]string         `json:"metadata,omitempty" doc:"Passed to the provider untouched, for the providers that store it."`
	PromptCacheKey  *string                    `json:"prompt_cache_key,omitempty" doc:"What a cached prompt prefix is keyed by. Requests sharing a key and a prefix are read from the cache rather than charged in full."`
	Providers       *[]string                  `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands. A response that fails is answered by the next entry that will have it."`
	ReasoningEffort *LlmOptionsReasoningEffort `json:"reasoning_effort,omitempty" doc:"How long the model may think before answering, on the models that think." enum:"minimal,low,medium,high"`
	Store           *bool                      `json:"store,omitempty" doc:"Keep the response on the provider so a later one can continue from it."`
	Target          *string                    `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"llm-fast"`
	Temperature     *float32                   `json:"temperature,omitempty"`
	ToolChoice      *string                    `json:"tool_choice,omitempty" doc:"auto, none, required, or the name of a tool the model must call. Which tools exist is per-request, since they change with the turn."`
	Verbosity       *LlmOptionsVerbosity       `json:"verbosity,omitempty" enum:"low,medium,high"`
}

func (*LlmOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_output_tokens"].Format = ""
	schema.Description = "How this config answers. The names are the response parameters the router already speaks rather than a second vocabulary for the same things. The system prompt is not among them: what the model answers under belongs to the agent asking, not to the config that decides where the asking goes."
	return schema
}

// MessageContent is the MessageContent schema.
type MessageContent struct {
	union json.RawMessage
}

// AsMessageContent0 returns the union data inside the MessageContent as a MessageContent0
func (t MessageContent) AsMessageContent0() (MessageContent0, error) {
	var body MessageContent0
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromMessageContent0 overwrites any union data inside the MessageContent as the provided MessageContent0
func (t *MessageContent) FromMessageContent0(v MessageContent0) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

// AsMessageContent1 returns the union data inside the MessageContent as a MessageContent1
func (t MessageContent) AsMessageContent1() (MessageContent1, error) {
	var body MessageContent1
	err := json.Unmarshal(t.union, &body)
	return body, err
}

// FromMessageContent1 overwrites any union data inside the MessageContent as the provided MessageContent1
func (t *MessageContent) FromMessageContent1(v MessageContent1) error {
	b, err := json.Marshal(v)
	t.union = b
	return err
}

func (t MessageContent) MarshalJSON() ([]byte, error) {
	b, err := t.union.MarshalJSON()
	return b, err
}

func (t *MessageContent) UnmarshalJSON(b []byte) error {
	err := t.union.UnmarshalJSON(b)
	return err
}

// MessageContent0 is the MessageContent0 schema.
type MessageContent0 = string

// MessageContent1 is the MessageContent1 schema.
type MessageContent1 = []ContentPart

// Modality What kind of work was done. The first seven are routed across providers; sts is speech to speech, one native audio model in place of a transcriber, a text model and a voice. lcm is a large classifier model: it answers a question about a piece of text with a typed value and the probability behind it rather than with prose, which is what a guardrail asks before a reply is spoken. image is pictures drawn from a prompt. Memory, knowledge and phone are recorded but not routed, since there is one memory store, one knowledge base and one vendor per number, so the provider paths do not serve them while the statistics paths do.
type Modality string

// Defines values for Modality.
const (
	Image     Modality = "image"
	Knowledge Modality = "knowledge"
	Lcm       Modality = "lcm"
	Llm       Modality = "llm"
	Memory    Modality = "memory"
	Phone     Modality = "phone"
	Search    Modality = "search"
	Sts       Modality = "sts"
	Stt       Modality = "stt"
	Tts       Modality = "tts"
)

// Valid indicates whether the value is a known member of the Modality enum.
func (e Modality) Valid() bool {
	switch e {
	case Image:
		return true
	case Knowledge:
		return true
	case Lcm:
		return true
	case Llm:
		return true
	case Memory:
		return true
	case Phone:
		return true
	case Search:
		return true
	case Sts:
		return true
	case Stt:
		return true
	case Tts:
		return true
	default:
		return false
	}
}

func (Modality) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Modality", "What kind of work was done. The first seven are routed across providers; sts is speech to speech, one native audio model in place of a transcriber, a text model and a voice. lcm is a large classifier model: it answers a question about a piece of text with a typed value and the probability behind it rather than with prose, which is what a guardrail asks before a reply is spoken. image is pictures drawn from a prompt. Memory, knowledge and phone are recorded but not routed, since there is one memory store, one knowledge base and one vendor per number, so the provider paths do not serve them while the statistics paths do.", "stt", "tts", "llm", "sts", "search", "lcm", "image", "memory", "knowledge", "phone")
}

// Provider is the Provider schema.
type Provider struct {
	Benchmark   *ProviderBenchmark `json:"benchmark,omitempty"`
	Description *string            `json:"description,omitempty" doc:"What the model is good at, and what that costs in speed or money. Empty if the deployment wrote none."`
	Health      ProviderHealth     `json:"health"`
	Languages   []string           `json:"languages" nullable:"false"`
	Model       string             `json:"model" example:"eleven_flash_v2_5"`
	Price       *ProviderPrice     `json:"price,omitempty"`
	Provider    string             `json:"provider" example:"elevenlabs"`
	Realtime    bool               `json:"realtime"`
	Tier        Tier               `json:"tier"`
	UsageShare  *float64           `json:"usage_share,omitempty" doc:"This model's share of the modality's requests over the last seven days, across every customer, from 0 to 1. It is how popular the model is, and is 0 when nothing was served or the deployment keeps no statistics."`
}

// RecordingSource Where the audio to work on comes from. A URL is what every vendor's batch API takes and what anything longer than a clip should use; inline bytes save a caller with a short local file from having to host it somewhere first.
type RecordingSource struct {
	Audio *[]byte `json:"audio,omitempty" doc:"The file itself, base64. For clips - a long recording belongs behind a URL." format:"byte"`
	Url   *string `json:"url,omitempty" doc:"A fetchable audio or video file."`
}

func (*RecordingSource) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Where the audio to work on comes from. A URL is what every vendor's batch API takes and what anything longer than a clip should use; inline bytes save a caller with a short local file from having to host it somewhere first."
	return schema
}

// Route is the Route schema.
type Route struct {
	Candidates  []Candidate `json:"candidates" doc:"The models the shortcut resolves to right now, best first." nullable:"false"`
	Description string      `json:"description"`
	Id          string      `json:"id" doc:"The shortcut, which is what a config or request names as its target." example:"llm-fast"`
	Title       string      `json:"title" example:"Fast conversational"`
}

// RouterConfig is the RouterConfig schema.
type RouterConfig struct {
	CreatedAt time.Time      `json:"created_at"`
	Id        string         `json:"id"`
	Llm       *LlmOptions    `json:"llm,omitempty"`
	Name      string         `json:"name"`
	Search    *SearchOptions `json:"search,omitempty"`
	Sts       *StsOptions    `json:"sts,omitempty"`
	Stt       *SttOptions    `json:"stt,omitempty"`
	Tts       *TtsOptions    `json:"tts,omitempty"`
	UpdatedAt time.Time      `json:"updated_at"`
}

// Sandbox Where the subagent may run code it writes. Only the subagent is offered it: running code takes seconds, and the model holding the conversation has none to spare. Omit it and the subagent works everything out in its head.
type Sandbox string

// Defines values for Sandbox.
const (
	Daytona Sandbox = "daytona"
)

// Valid indicates whether the value is a known member of the Sandbox enum.
func (e Sandbox) Valid() bool {
	switch e {
	case Daytona:
		return true
	default:
		return false
	}
}

// SandboxOptions How the sandbox is built and how long code may run in it. Only meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run.
type SandboxOptions struct {
	Cpu       *int      `json:"cpu,omitempty" doc:"CPUs for the sandbox. Zero is the provider's default." minimum:"0" maximum:"16"`
	DiskGb    *int      `json:"disk_gb,omitempty" doc:"Disk for the sandbox, in GiB. Zero is the provider's default." minimum:"0" maximum:"100"`
	Image     *string   `json:"image,omitempty" doc:"The container image to build on, which must have Python, such as python:3.13-slim-bookworm. Empty with anything else set is a slim Python 3.13 image." maxLength:"256"`
	MemoryGb  *int      `json:"memory_gb,omitempty" doc:"Memory for the sandbox, in GiB. Zero is the provider's default." minimum:"0" maximum:"64"`
	Setup     *[]string `json:"setup,omitempty" doc:"Shell commands run once on top of the image when it is built, such as installing packages. The provider keeps the built image, so only the first sandbox from a given setup waits for it." maxItems:"32"`
	TimeoutMs *int      `json:"timeout_ms,omitempty" doc:"How long one run of code may take, at most 30 minutes. Zero is 30 seconds. A run is still bounded by the deadline of the skill it was written for." minimum:"0" maximum:"1800000"`
}

func (*SandboxOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How the sandbox is built and how long code may run in it. Only meaningful with a sandbox. Omit it for the provider's own Python sandbox and a 30 second run."
	schema.AdditionalProperties = false
	schema.Properties["setup"].Items.MaxLength = itemLimit(2048)
	return schema
}

// Session is the Session schema.
type Session struct {
	Agent            *string                 `json:"agent,omitempty" doc:"The name the agent was addressed as. Recorded on the session as well as the config id, so renaming a config does not rewrite what older sessions were opened against."`
	AgentId          string                  `json:"agent_id"`
	CallId           string                  `json:"call_id" doc:"Empty for a text session, which joins no call."`
	CallType         string                  `json:"call_type"`
	ClosedAt         *time.Time              `json:"closed_at,omitempty" doc:"When the session ended. Absent while it is still running."`
	ConfigId         *string                 `json:"config_id,omitempty" doc:"The agent config the session ran under, empty for one that spelled itself out."`
	ContextTruncated *bool                   `json:"context_truncated,omitempty" doc:"Older history was omitted from the model context."`
	ConversationId   *string                 `json:"conversation_id,omitempty" doc:"Stream Chat CID to resume; returned for persistent text sessions."`
	CreatedAt        time.Time               `json:"created_at"`
	Custom           *map[string]interface{} `json:"custom,omitempty"`
	Description      *string                 `json:"description,omitempty"`
	ForkedFrom       *string                 `json:"forked_from,omitempty" doc:"The session this one continued from, empty for one opened fresh."`
	Id               string                  `json:"id"`
	Incognito        *bool                   `json:"incognito,omitempty" doc:"Nothing about this session was recorded. It is reported so a caller can see that what they asked for is what they got, but it is never read back from storage: an incognito session has no row to read it from."`
	Instructions     *string                 `json:"instructions,omitempty"`
	LastResponseAt   *time.Time              `json:"last_response_at,omitempty" doc:"When the agent last answered. This is what a most-recently-used ordering of conversations reads, since a session renamed long after it ended has not become more recent."`
	Llm              *string                 `json:"llm,omitempty" doc:"The provider and model answering, once routing has picked one."`
	Modality         SessionModality         `json:"modality"`
	Mode             *SessionMode            `json:"mode,omitempty"`
	ModelOverwrites  *ModelOverwrites        `json:"model_overwrites,omitempty"`
	ProjectId        *string                 `json:"project_id,omitempty"`
	State            SessionState            `json:"state"`
	Sts              *string                 `json:"sts,omitempty" doc:"The provider and model holding a native conversation, once routing has picked one."`
	Stt              *string                 `json:"stt,omitempty" doc:"The provider and model transcribing, once somebody has been heard."`
	ThinkingLlm      *string                 `json:"thinking_llm,omitempty" doc:"The provider and model delegated work runs on."`
	Text             *bool                   `json:"text,omitempty" doc:"The conversation is held in writing rather than on a call."`
	Title            *string                 `json:"title,omitempty"`
	Tts              *string                 `json:"tts,omitempty" doc:"The provider and model speaking."`
	UserId           string                  `json:"user_id"`
	Video            *SessionVideo           `json:"video,omitempty"`
	Voice            *string                 `json:"voice,omitempty" doc:"The voice speaking, in the provider's own terms. It is the provider's default when the session asked for none."`
}

// SessionMemory Who the session's memories are about. Without a user id nothing is recalled or stored, which is the case for a call with nobody identified on it.
type SessionMemory struct {
	AppId  *string            `json:"app_id,omitempty" doc:"Separates two deployments sharing one memory account."`
	Filter *map[string]string `json:"filter,omitempty" doc:"The caller's own labels, which narrow recall further. They cannot widen it: a filter is applied alongside the user id, never instead of it."`
	UserId *string            `json:"user_id,omitempty" doc:"Who the memories belong to. Empty means the customer."`
}

func (*SessionMemory) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Who the session's memories are about. Without a user id nothing is recalled or stored, which is the case for a call with nobody identified on it."
	return schema
}

// SessionModality How the user took part: text for a conversation held in writing, voice for a call, and video once the agent has seen the user's video. It only moves up, from text or voice to video.
type SessionModality string

// Defines values for SessionModality.
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
	return namedEnum(registry, "SessionModality", "How the user took part: text for a conversation held in writing, voice for a call, and video once the agent has seen the user's video. It only moves up, from text or voice to video.", "text", "voice", "video")
}

// SessionMode How the session hears and speaks: a transcriber, a conversation model and a voice; one speech-to-speech model; or in writing.
type SessionMode string

// Defines values for SessionMode.
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
	return namedEnum(registry, "SessionMode", "How the session hears and speaks: a transcriber, a conversation model and a voice; one speech-to-speech model; or in writing.", "cascade", "native", "text")
}

// SessionPhone The number the session acts from, which is what turns transferring on.
type SessionPhone struct {
	Number       string  `json:"number" doc:"One of the customer's own numbers, written as +15551234567."`
	Vendor       *string `json:"vendor,omitempty" doc:"Who carries an outbound leg."`
	VendorCallId *string `json:"vendor_call_id,omitempty" doc:"The outbound leg, set for a call the agent placed. Without one the agent has no keypad to press at."`
}

func (*SessionPhone) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "The number the session acts from, which is what turns transferring on."
	return schema
}

// SessionRespondCommand is the SessionRespondCommand schema.
type SessionRespondCommand struct {
	CommandId *string                   `json:"command_id,omitempty" doc:"Required for personal persistent text conversations; reuse on retries. Text only when present." pattern:"^[A-Za-z0-9_-]{1,128}$"`
	Images    *[]ImageSource            `json:"images,omitempty"`
	Text      string                    `json:"text"`
	Type      SessionRespondCommandType `json:"type" enum:"respond"`
}

// SessionRespondCommandType is the SessionRespondCommandType schema.
type SessionRespondCommandType string

// Defines values for SessionRespondCommandType.
const (
	Respond SessionRespondCommandType = "respond"
)

// Valid indicates whether the value is a known member of the SessionRespondCommandType enum.
func (e SessionRespondCommandType) Valid() bool {
	switch e {
	case Respond:
		return true
	default:
		return false
	}
}

// SessionSettingsRequestThinking is the SessionSettingsRequestThinking schema.
type SessionSettingsRequestThinking string

// Defines values for SessionSettingsRequestThinking.
const (
	SessionSettingsRequestThinkingHigh    SessionSettingsRequestThinking = "high"
	SessionSettingsRequestThinkingLow     SessionSettingsRequestThinking = "low"
	SessionSettingsRequestThinkingMedium  SessionSettingsRequestThinking = "medium"
	SessionSettingsRequestThinkingMinimal SessionSettingsRequestThinking = "minimal"
	SessionSettingsRequestThinkingNone    SessionSettingsRequestThinking = "none"
)

// Valid indicates whether the value is a known member of the SessionSettingsRequestThinking enum.
func (e SessionSettingsRequestThinking) Valid() bool {
	switch e {
	case SessionSettingsRequestThinkingHigh:
		return true
	case SessionSettingsRequestThinkingLow:
		return true
	case SessionSettingsRequestThinkingMedium:
		return true
	case SessionSettingsRequestThinkingMinimal:
		return true
	case SessionSettingsRequestThinkingNone:
		return true
	default:
		return false
	}
}

// SessionSettingsRequestVerbosity is the SessionSettingsRequestVerbosity schema.
type SessionSettingsRequestVerbosity string

// Defines values for SessionSettingsRequestVerbosity.
const (
	SessionSettingsRequestVerbosityHigh   SessionSettingsRequestVerbosity = "high"
	SessionSettingsRequestVerbosityLow    SessionSettingsRequestVerbosity = "low"
	SessionSettingsRequestVerbosityMedium SessionSettingsRequestVerbosity = "medium"
)

// Valid indicates whether the value is a known member of the SessionSettingsRequestVerbosity enum.
func (e SessionSettingsRequestVerbosity) Valid() bool {
	switch e {
	case SessionSettingsRequestVerbosityHigh:
		return true
	case SessionSettingsRequestVerbosityLow:
		return true
	case SessionSettingsRequestVerbosityMedium:
		return true
	default:
		return false
	}
}

// SessionState Whether the agent is still in the call.
type SessionState string

// Defines values for SessionState.
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
	return namedEnum(registry, "SessionState", "Whether the agent is still in the call.", "live", "ended")
}

// SessionTool One of the caller's own functions. The model is offered it by name and description; running it is the caller's business, over the events socket.
type SessionTool struct {
	Approval     *SessionToolApproval    `json:"approval,omitempty"`
	Description  string                  `json:"description" doc:"What the model is told the tool does, which is the whole of how it decides when to reach for one."`
	DisplayTitle *string                 `json:"display_title,omitempty" doc:"What a call is doing, in words for the people in the conversation, such as \"Checking your location\". Shown on the reply's ai_tool_call attachment." maxLength:"80"`
	Executor     *SessionToolExecutor    `json:"executor,omitempty" doc:"Who runs it. A client tool runs on a person's device: in a persistent conversation its call is shown as awaiting the device of the person whose command it answers (their user and the command's client_id), with its arguments, which every channel member can read. The caller still answers it over the events socket, once the device has reported. Defaults to server." enum:"server,client"`
	Name         string                  `json:"name"`
	Parameters   *map[string]interface{} `json:"parameters,omitempty" doc:"A JSON Schema object describing the arguments."`
}

func (*SessionTool) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["executor"].Extensions = map[string]any{"x-enum-varnames": []any{"SessionToolExecutorServer", "SessionToolExecutorClient"}}
	schema.Description = "One of the caller's own functions. The model is offered it by name and description; running it is the caller's business, over the events socket."
	return schema
}

// SessionToolExecutor is the SessionToolExecutor schema.
type SessionToolExecutor string

// Defines values for SessionToolExecutor.
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

// SessionToolApproval Says a person must allow each call before it runs. In a persistent conversation the call's ai_tool_call attachment opens as awaiting_approval, addressed to the person whose command it answers (and, for a client tool, their install), and carries this question for their client to ask. The caller collects the answer and reports it over the events socket with tool_approval: allowed, the call goes on as it would have (awaiting_client for a client tool, running otherwise); declined, it is cancelled. The caller still answers the call with tool_result either way. Every channel member can read the question.
type SessionToolApproval struct {
	AllowTitle     *string `json:"allow_title,omitempty" doc:"The label of the button that allows the call." maxLength:"40"`
	DeclineTitle   *string `json:"decline_title,omitempty" doc:"The label of the button that declines it." maxLength:"40"`
	Message        *string `json:"message,omitempty" doc:"What allowing it shares or does, such as \"Only your city is shared.\"" maxLength:"240"`
	ReasonArgument *string `json:"reason_argument,omitempty" doc:"The argument, a string, in which the model says why it wants this call. Its text (at most 160 characters) is shown as the approval's reason, so it is visible to every channel member even for a server tool." maxLength:"64"`
	Title          string  `json:"title" doc:"The question, such as \"Share your location?\"." maxLength:"80"`
}

func (*SessionToolApproval) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "Says a person must allow each call before it runs. In a persistent conversation the call's ai_tool_call attachment opens as awaiting_approval, addressed to the person whose command it answers (and, for a client tool, their install), and carries this question for their client to ask. The caller collects the answer and reports it over the events socket with tool_approval: allowed, the call goes on as it would have (awaiting_client for a client tool, running otherwise); declined, it is cancelled. The caller still answers the call with tool_result either way. Every channel member can read the question."
	return schema
}

// SessionVideo is the SessionVideo schema.
type SessionVideo struct {
	MaxFrames *int    `json:"max_frames,omitempty" doc:"Number of recent frames captured for a visual task. Default one." minimum:"1" maximum:"8"`
	Source    *string `json:"source,omitempty" doc:"Track or processor source. Omitted requires one unambiguous available source."`
}

func (*SessionVideo) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_frames"].Format = ""
	return schema
}

// Simulation is the Simulation schema.
type Simulation struct {
	Assertion    string             `json:"assertion"`
	CallerStt    *string            `json:"caller_stt,omitempty"`
	CallerTarget *string            `json:"caller_target,omitempty"`
	CallerTts    *string            `json:"caller_tts,omitempty"`
	CallerVoice  *string            `json:"caller_voice,omitempty"`
	ConfigId     string             `json:"config_id"`
	CreatedAt    time.Time          `json:"created_at"`
	Id           string             `json:"id"`
	JudgeTarget  *string            `json:"judge_target,omitempty"`
	MaxTurns     int                `json:"max_turns"`
	Mode         SimulationMode     `json:"mode" enum:"text,audio"`
	Name         string             `json:"name"`
	Scenario     string             `json:"scenario"`
	Tags         *map[string]string `json:"tags,omitempty"`
	UpdatedAt    *time.Time         `json:"updated_at,omitempty"`
	Variations   int                `json:"variations"`
}

func (*Simulation) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_turns"].Format = ""
	schema.Properties["variations"].Format = ""
	return schema
}

// SimulationRequest is the SimulationRequest schema.
type SimulationRequest struct {
	Assertion    string                 `json:"assertion" doc:"What has to be true at the end for the run to have passed."`
	CallerStt    *string                `json:"caller_stt,omitempty" doc:"How the caller hears the agent. Audio simulations only."`
	CallerTarget *string                `json:"caller_target,omitempty" doc:"The model that plays the caller. Empty takes llm-scenario-runner, the deployment's fast-tier default."`
	CallerTts    *string                `json:"caller_tts,omitempty" doc:"How the caller speaks. Audio simulations only."`
	CallerVoice  *string                `json:"caller_voice,omitempty" doc:"The voice the caller speaks in. Audio simulations only."`
	ConfigId     string                 `json:"config_id" doc:"The agent being tested."`
	JudgeTarget  *string                `json:"judge_target,omitempty" doc:"The model that rules on the conversations, named the way any other routing target is. Empty takes llm-judge, the deployment's quality-tier default, since nobody is waiting for it."`
	MaxTurns     *int                   `json:"max_turns,omitempty" doc:"How many times the caller may speak, up to two hundred. It is what stops a caller that never decides it is finished. Twelve when left out."`
	Mode         *SimulationRequestMode `json:"mode,omitempty" doc:"Text hands the agent the words, which tests everything between hearing and answering. Audio generates speech and runs the whole pipeline, so what is judged is what a caller would actually have heard. Text when left out." enum:"text,audio"`
	Name         string                 `json:"name"`
	Scenario     string                 `json:"scenario" doc:"What to ask, in your own words and over as many turns as it takes. This is a brief for the caller rather than a script, so it may describe things that depend on what the agent says back."`
	Tags         *map[string]string     `json:"tags,omitempty"`
	Variations   *int                   `json:"variations,omitempty" doc:"How many ways of asking the same thing one run tries, up to ten. The scenario as written is always the first of them, and one is what left out means."`
}

func (*SimulationRequest) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_turns"].Format = ""
	schema.Properties["variations"].Format = ""
	return schema
}

// Skill is the Skill schema.
type Skill struct {
	CaptureVideo *bool     `json:"capture_video,omitempty" doc:"Capture task-scoped visual evidence before reasoning."`
	ConfigId     string    `json:"config_id"`
	CreatedAt    time.Time `json:"created_at"`
	DeadlineMs   *int64    `json:"deadline_ms,omitempty"`
	Description  string    `json:"description"`
	Id           string    `json:"id"`
	Instructions string    `json:"instructions"`
	Name         string    `json:"name"`
	UpdatedAt    time.Time `json:"updated_at"`
}

// SkillRequest is the SkillRequest schema.
type SkillRequest struct {
	CaptureVideo *bool  `json:"capture_video,omitempty" doc:"Capture task-scoped visual evidence before reasoning."`
	ConfigId     string `json:"config_id" doc:"The agent config this skill belongs to. A skill is not shared: two agents that both need one have one each, so editing either leaves the other alone."`
	DeadlineMs   *int64 `json:"deadline_ms,omitempty" doc:"How long the work may run before it is abandoned. Zero is the default."`
	Description  string `json:"description" doc:"The one line the fast model sees."`
	Instructions string `json:"instructions" doc:"The full prompt, which only the subagent sees."`
	Name         string `json:"name" doc:"How the config names it, which is unique among that config's own skills."`
}

// Speech is the Speech schema.
type Speech struct {
	Audio           *[]byte         `json:"audio,omitempty" doc:"The audio itself, base64, when it was not stored behind a URL." format:"byte"`
	AudioDurationMs *int64          `json:"audio_duration_ms,omitempty"`
	Characters      *int64          `json:"characters,omitempty" doc:"How much text was spoken, which is what it was billed on."`
	CompletedAt     *time.Time      `json:"completed_at,omitempty"`
	CreatedAt       time.Time       `json:"created_at"`
	Error           *string         `json:"error,omitempty"`
	Format          *string         `json:"format,omitempty" doc:"What the audio is encoded as, which is what was asked for." example:"mp3_44100_128"`
	Id              string          `json:"id"`
	Model           *string         `json:"model,omitempty"`
	Provider        *string         `json:"provider,omitempty"`
	Status          RecordingStatus `json:"status"`
	UpdatedAt       time.Time       `json:"updated_at"`
	Url             *string         `json:"url,omitempty" doc:"Where the finished audio is, on a deployment that stores it. Empty means the audio came back inline instead."`
}

// StsOptions How this config holds a conversation with one native audio model, in place of a transcriber, a text model and a voice. What every such model takes is a field here; what only some take is a term, and a request naming a term is routed to a model that declared it or refused, never served by one that ignores it.
type StsOptions struct {
	DataPolicy        *DataPolicy              `json:"data_policy,omitempty"`
	Images            *bool                    `json:"images,omitempty" doc:"The session will send the model frames, so it is routed only to a model that sees, the way vlm routes a text model."`
	InputTranscript   *bool                    `json:"input_transcript,omitempty" doc:"Ask the model to write down what it heard."`
	Instructions      *string                  `json:"instructions,omitempty" doc:"The system prompt the model converses under, sent when the session opens. A stored router config naming one is refused: what is said belongs to the agent holding the conversation, not to the config that decides where it goes."`
	InterruptResponse *bool                    `json:"interrupt_response,omitempty" doc:"Whether the model cuts its own reply off when it hears the caller. Omitting it leaves the vendor's default; false is for a speaker close enough to the microphone that the model would otherwise interrupt itself."`
	Languages         *[]string                `json:"languages,omitempty"`
	OutputTranscript  *bool                    `json:"output_transcript,omitempty" doc:"Ask the model to write down what it said."`
	Overwrites        *map[string]interface{}  `json:"overwrites,omitempty" doc:"Settings for one provider that this vocabulary has no word for, keyed by provider name, for example {\"openai\": {\"eagerness\": \"high\"}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped." example:"{\"openai\": {\"eagerness\": \"high\"}}"`
	PrefixPaddingMs   *int                     `json:"prefix_padding_ms,omitempty" doc:"How much audio before the detected speech is kept, for a silence timer."`
	Providers         *[]string                `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands."`
	SilenceMs         *int                     `json:"silence_ms,omitempty" doc:"How long a pause ends the turn, for a silence timer."`
	Target            *string                  `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"sts-fast"`
	Text              *bool                    `json:"text,omitempty" doc:"The session will inject typed turns."`
	Tools             *bool                    `json:"tools,omitempty" doc:"The session will hand the model functions to call."`
	TurnDetection     *StsOptionsTurnDetection `json:"turn_detection,omitempty" doc:"What decides the caller has finished: a silence timer, a model reading the words, or nothing, which leaves the turns to the caller. Omitting it leaves the vendor's default. Only some models read the words, so semantic is a term." enum:"server_vad,semantic,none"`
	Voice             *string                  `json:"voice,omitempty" doc:"The vendor's own name for a voice, such as marin at OpenAI or Kore at Google. None of these models takes one of your own voices, so the name is passed on as given rather than looked up." example:"marin"`
}

func (*StsOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["prefix_padding_ms"].Format = ""
	schema.Properties["silence_ms"].Format = ""
	schema.Description = "How this config holds a conversation with one native audio model, in place of a transcriber, a text model and a voice. What every such model takes is a field here; what only some take is a term, and a request naming a term is routed to a model that declared it or refused, never served by one that ignores it."
	return schema
}

// SttOptions How this config transcribes, live or from a recording. A field that only means something on one of the two forms says so: a recording has no endpointing to do, and a socket has no file to write subtitles from. A provider that cannot express a term refuses the request rather than dropping it silently.
type SttOptions struct {
	Channels        *int                    `json:"channels,omitempty" doc:"Transcribe a multichannel recording per channel rather than mixed down." minimum:"1"`
	DataPolicy      *DataPolicy             `json:"data_policy,omitempty"`
	DetectLanguage  *bool                   `json:"detect_language,omitempty" doc:"Let the provider identify the language instead of being told it."`
	Diarize         *bool                   `json:"diarize,omitempty" doc:"Label each stretch of speech with who said it."`
	EagerEndOfTurn  *bool                   `json:"eager_end_of_turn,omitempty" doc:"Send a transcript as soon as the model guesses the turn may be over, before it is sure, so a reply can start early. Live only. A model without an eager end of turn transcribes as normal rather than being refused. On by default for en-low-latency and multilingual-low-latency."`
	Endpointing     *Endpointing            `json:"endpointing,omitempty"`
	Entities        *bool                   `json:"entities,omitempty" doc:"Extract named entities from the recording. Recording only."`
	Events          *bool                   `json:"events,omitempty" doc:"Tag non-speech audio events such as laughter or music."`
	Format          *bool                   `json:"format,omitempty" doc:"Punctuation, capitalisation and smart formatting of numbers and dates."`
	Interim         *bool                   `json:"interim,omitempty" doc:"Emit partial transcripts as they firm up, not only final ones. Live only."`
	Keyterms        *[]string               `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong. Up to 100 terms, and providers that cannot be told about vocabulary refuse them."`
	Languages       *[]string               `json:"languages,omitempty" doc:"ISO codes candidates must cover. Empty with detect_language lets the provider decide."`
	MaxSpeakers     *int                    `json:"max_speakers,omitempty" doc:"A hard cap on the speakers diarization may find, not a hint. Providers differ in what they allow, so one asked for more than it supports refuses." minimum:"1" maximum:"32"`
	Mode            *TranscriptionMode      `json:"mode,omitempty"`
	Output          *TranscriptFormat       `json:"output,omitempty"`
	Overwrites      *map[string]interface{} `json:"overwrites,omitempty" doc:"Settings for one provider that this vocabulary has no word for, keyed by provider name, for example {\"deepgram\": {\"eot_threshold\": 0.6}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped." example:"{\"deepgram\": {\"eot_threshold\": 0.6}}"`
	ProfanityFilter *bool                   `json:"profanity_filter,omitempty" doc:"Mask offensive words rather than writing them down. Only some providers can be told to, so a request for it is routed to one of them or refused."`
	Providers       *[]string               `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to the back; unlike a shortcut, this does not reorder on latency, because a caller who wrote an order meant it."`
	Redact          *bool                   `json:"redact,omitempty" doc:"Remove personally identifying information from the transcript."`
	SampleRate      *int                    `json:"sample_rate,omitempty" doc:"Rate of the PCM sent on the socket. Zero means 16 kHz. Live only." example:"16000"`
	SilenceMs       *int                    `json:"silence_ms,omitempty" doc:"How long a pause ends a turn, for silence endpointing. Live only." example:"300"`
	Summary         *bool                   `json:"summary,omitempty" doc:"Summarise the recording, where the provider offers audio intelligence. Recording only."`
	Target          *string                 `json:"target,omitempty" doc:"A provider/model or a capability shortcut such as en-low-latency for the live path or en-recorded for a recording." example:"en-low-latency"`
	UtteranceEndMs  *int                    `json:"utterance_end_ms,omitempty" doc:"How long after the last word an utterance is declared over. Live only."`
	Words           *bool                   `json:"words,omitempty" doc:"Word-level timestamps. Recording only."`
}

func (*SttOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["channels"].Format = ""
	schema.Properties["max_speakers"].Format = ""
	schema.Properties["sample_rate"].Format = ""
	schema.Properties["silence_ms"].Format = ""
	schema.Properties["utterance_end_ms"].Format = ""
	schema.Description = "How this config transcribes, live or from a recording. A field that only means something on one of the two forms says so: a recording has no endpointing to do, and a socket has no file to write subtitles from. A provider that cannot express a term refuses the request rather than dropping it silently."
	return schema
}

// TextContentPart is the TextContentPart schema.
type TextContentPart struct {
	Text string              `json:"text"`
	Type TextContentPartType `json:"type" enum:"text"`
}

// TextContentPartType is the TextContentPartType schema.
type TextContentPartType string

// Defines values for TextContentPartType.
const (
	TextContentPartTypeText TextContentPartType = "text"
)

// Valid indicates whether the value is a known member of the TextContentPartType enum.
func (e TextContentPartType) Valid() bool {
	switch e {
	case TextContentPartTypeText:
		return true
	default:
		return false
	}
}

// ToolApprovalCommand A person's answer to a call awaiting their approval, from a persistent text command. It changes only how the call is shown; the call still needs a tool_result.
type ToolApprovalCommand struct {
	Allowed    bool                    `json:"allowed"`
	CommandId  string                  `json:"command_id"`
	Summary    *string                 `json:"summary,omitempty" doc:"Shown on the declined call, such as \"Location not shared\"." maxLength:"120"`
	ToolCallId string                  `json:"tool_call_id"`
	TurnId     string                  `json:"turn_id"`
	Type       ToolApprovalCommandType `json:"type" enum:"tool_approval"`
}

func (*ToolApprovalCommand) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "A person's answer to a call awaiting their approval, from a persistent text command. It changes only how the call is shown; the call still needs a tool_result."
	return schema
}

// ToolApprovalCommandType is the ToolApprovalCommandType schema.
type ToolApprovalCommandType string

// Defines values for ToolApprovalCommandType.
const (
	ToolApproval ToolApprovalCommandType = "tool_approval"
)

// Valid indicates whether the value is a known member of the ToolApprovalCommandType enum.
func (e ToolApprovalCommandType) Valid() bool {
	switch e {
	case ToolApproval:
		return true
	default:
		return false
	}
}

// ToolResultCommand is the ToolResultCommand schema.
type ToolResultCommand struct {
	CommandId  *string               `json:"command_id,omitempty"`
	Error      *string               `json:"error,omitempty"`
	Output     *MessageContent       `json:"output,omitempty"`
	ToolCallId string                `json:"tool_call_id"`
	TurnId     *string               `json:"turn_id,omitempty"`
	Type       ToolResultCommandType `json:"type" enum:"tool_result"`
}

// ToolResultCommandType is the ToolResultCommandType schema.
type ToolResultCommandType string

// Defines values for ToolResultCommandType.
const (
	ToolResultCommandTypeToolResult ToolResultCommandType = "tool_result"
)

// Valid indicates whether the value is a known member of the ToolResultCommandType enum.
func (e ToolResultCommandType) Valid() bool {
	switch e {
	case ToolResultCommandTypeToolResult:
		return true
	default:
		return false
	}
}

// TtsOptions How this config speaks. A provider that cannot express a term refuses the request rather than dropping it silently, since a voice asked to sound urgent and speaking flatly is worse than one that says it cannot.
type TtsOptions struct {
	ChunkSchedule  *[]int                  `json:"chunk_schedule,omitempty" doc:"Character counts at which a streaming voice flushes audio. Smaller first values start speaking sooner and cost more requests. Live only."`
	DataPolicy     *DataPolicy             `json:"data_policy,omitempty"`
	Emotion        *string                 `json:"emotion,omitempty" doc:"Affect to speak with, for the providers that take one."`
	Format         *string                 `json:"format,omitempty" doc:"Codec, sample rate and bitrate as one name - pcm_16000, mp3_44100_128, ulaw_8000 for telephony." example:"pcm_16000"`
	Languages      *[]string               `json:"languages,omitempty"`
	Overwrites     *map[string]interface{} `json:"overwrites,omitempty" doc:"Settings for one voice provider that this vocabulary has no word for, keyed by provider name, for example {\"elevenlabs\": {\"voice_id\": \"21m00Tcm4TlvDq8ikWAM\"}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped. It is also the only way to steer a live voice per vendor, since a voice id from one library means nothing at another." example:"{\"elevenlabs\": {\"voice_id\": \"21m00Tcm4TlvDq8ikWAM\"}}"`
	Pronunciations *map[string]string      `json:"pronunciations,omitempty" doc:"How to say words the voice gets wrong, keyed by the word."`
	Providers      *[]string               `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to the back."`
	Similarity     *float32                `json:"similarity,omitempty" doc:"How closely a cloned voice tracks its reference."`
	Speed          *float32                `json:"speed,omitempty" doc:"Rate of delivery, 1 being the voice's own. Providers differ in the range they accept, so one asked for a speed outside its own refuses." example:"1"`
	Stability      *float32                `json:"stability,omitempty" doc:"How much the voice may vary between chunks. Higher is flatter and more consistent."`
	Style          *string                 `json:"style,omitempty" doc:"Delivery style, for the providers that name styles rather than emotions."`
	Target         *string                 `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"en-low-latency"`
	Voice          *string                 `json:"voice,omitempty" doc:"A provider's own voice id, or one of your voices by id or by the name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to the provider's library, and with it a name that is not one of yours is refused." example:"custom:receptionist"`
	Volume         *float32                `json:"volume,omitempty" doc:"Loudness, 1 being the voice's own."`
}

func (*TtsOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["chunk_schedule"].Items.Format = ""
	schema.Description = "How this config speaks. A provider that cannot express a term refuses the request rather than dropping it silently, since a voice asked to sound urgent and speaking flatly is worse than one that says it cannot."
	return schema
}

// VideoSource is the VideoSource schema.
type VideoSource struct {
	MaxFrames *int   `json:"max_frames,omitempty" doc:"How many frames to sample, evenly spaced across the clip. Default 8." minimum:"1" maximum:"32"`
	Url       string `json:"url" doc:"Public HTTPS URL or base64 video data URI, such as data:video/mp4;base64,.... At most 50 MB either way. The router fetches a URL itself, and refuses one that resolves to a private or loopback address."`
}

func (*VideoSource) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Properties["max_frames"].Format = ""
	return schema
}

// Voice is the Voice schema.
type Voice struct {
	Bindings    *[]VoiceBinding `json:"bindings,omitempty"`
	CreatedAt   time.Time       `json:"created_at"`
	Description *string         `json:"description,omitempty"`
	Id          string          `json:"id"`
	Name        string          `json:"name"`
	Samples     *[]VoiceSample  `json:"samples,omitempty"`
	UpdatedAt   time.Time       `json:"updated_at"`
}

// CommandID is the CommandID schema.
type CommandID = string

// Cursor is the Cursor schema.
type Cursor = string

// ResourceID is the ResourceID schema.
type ResourceID = string

// SessionID is the SessionID schema.
type SessionID = string

// SessionLimit is the SessionLimit schema.
type SessionLimit = int

func (ContentPart) Schema(registry huma.Registry) *huma.Schema {
	registry.Map()["ContentPart"] = &huma.Schema{OneOf: []*huma.Schema{
		registry.Schema(reflect.TypeFor[TextContentPart](), true, ""),
		registry.Schema(reflect.TypeFor[ImageContentPart](), true, ""),
	}}
	return &huma.Schema{Ref: "#/components/schemas/ContentPart"}
}

func (MessageContent) Schema(registry huma.Registry) *huma.Schema {
	registry.Map()["MessageContent"] = &huma.Schema{OneOf: []*huma.Schema{
		{Type: huma.TypeString},
		{Type: huma.TypeArray, Items: registry.Schema(reflect.TypeFor[ContentPart](), true, "")},
	}}
	return &huma.Schema{Ref: "#/components/schemas/MessageContent"}
}
