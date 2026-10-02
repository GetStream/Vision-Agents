package api

import (
	"context"
	"fmt"
	"net/http"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/danielgtaylor/huma/v2"
)

// noRouterConfigs is what the router config paths say on a deployment without a database.
const noRouterConfigs = "router configs are not available: no database configured"

// unknownRouterConfig is what a caller is told about a config that is not theirs, which is
// the same thing they are told about one that never existed.
const unknownRouterConfig = "no such router config"

type RouterConfig struct {
	Id        string         `json:"id"`
	Name      string         `json:"name"`
	Stt       *SttOptions    `json:"stt,omitempty"`
	Tts       *TtsOptions    `json:"tts,omitempty"`
	Llm       *LlmOptions    `json:"llm,omitempty"`
	Sts       *StsOptions    `json:"sts,omitempty"`
	Search    *SearchOptions `json:"search,omitempty"`
	CreatedAt time.Time      `json:"created_at"`
	UpdatedAt time.Time      `json:"updated_at"`
}

type SttOptions struct {
	Target          *string                 `json:"target,omitempty" doc:"A provider/model or a capability shortcut such as en-low-latency for the live path or en-recorded for a recording." example:"en-low-latency"`
	Providers       *[]string               `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to the back; unlike a shortcut, this does not reorder on latency, because a caller who wrote an order meant it." example:"[\"deepgram\", \"en-low-latency\"]"`
	Languages       *[]string               `json:"languages,omitempty" doc:"ISO codes candidates must cover. Empty with detect_language lets the provider decide." example:"[\"en\"]"`
	DetectLanguage  *bool                   `json:"detect_language,omitempty" doc:"Let the provider identify the language instead of being told it."`
	SampleRate      *int                    `json:"sample_rate,omitempty" doc:"Rate of the PCM sent on the socket. Zero means 16 kHz. Live only." example:"16000"`
	Interim         *bool                   `json:"interim,omitempty" doc:"Emit partial transcripts as they firm up, not only final ones. Live only."`
	Endpointing     *Endpointing            `json:"endpointing,omitempty"`
	SilenceMs       *int                    `json:"silence_ms,omitempty" doc:"How long a pause ends a turn, for silence endpointing. Live only." example:"300"`
	UtteranceEndMs  *int                    `json:"utterance_end_ms,omitempty" doc:"How long after the last word an utterance is declared over. Live only."`
	EagerEndOfTurn  *bool                   `json:"eager_end_of_turn,omitempty" doc:"Send a transcript as soon as the model guesses the turn may be over, before it is sure, so a reply can start early. Live only. A model without an eager end of turn transcribes as normal rather than being refused. On by default for en-low-latency and multilingual-low-latency."`
	Diarize         *bool                   `json:"diarize,omitempty" doc:"Label each stretch of speech with who said it."`
	MaxSpeakers     *int                    `json:"max_speakers,omitempty" doc:"A hard cap on the speakers diarization may find, not a hint. Providers differ in what they allow, so one asked for more than it supports refuses." minimum:"1" maximum:"32"`
	Keyterms        *[]string               `json:"keyterms,omitempty" doc:"Business-specific words the transcriber would otherwise get wrong. Up to 100 terms, and providers that cannot be told about vocabulary refuse them."`
	Format          *bool                   `json:"format,omitempty" doc:"Punctuation, capitalisation and smart formatting of numbers and dates."`
	Redact          *bool                   `json:"redact,omitempty" doc:"Remove personally identifying information from the transcript."`
	Events          *bool                   `json:"events,omitempty" doc:"Tag non-speech audio events such as laughter or music."`
	Channels        *int                    `json:"channels,omitempty" doc:"Transcribe a multichannel recording per channel rather than mixed down." minimum:"1"`
	Words           *bool                   `json:"words,omitempty" doc:"Word-level timestamps. Recording only."`
	Output          *TranscriptFormat       `json:"output,omitempty"`
	Summary         *bool                   `json:"summary,omitempty" doc:"Summarise the recording, where the provider offers audio intelligence. Recording only."`
	Entities        *bool                   `json:"entities,omitempty" doc:"Extract named entities from the recording. Recording only."`
	ProfanityFilter *bool                   `json:"profanity_filter,omitempty" doc:"Mask offensive words rather than writing them down. Only some providers can be told to, so a request for it is routed to one of them or refused."`
	Mode            *TranscriptionMode      `json:"mode,omitempty"`
	DataPolicy      *DataPolicy             `json:"data_policy,omitempty"`
	Overwrites      *map[string]interface{} `json:"overwrites,omitempty" doc:"Settings for one provider that this vocabulary has no word for, keyed by provider name, for example {\"deepgram\": {\"eot_threshold\": 0.6}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped." example:"{\"deepgram\": {\"eot_threshold\": 0.6}}"`
}

func (*SttOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How this config transcribes, live or from a recording. A field that only means something on " +
		"one of the two forms says so: a recording has no endpointing to do, and a socket has no " +
		"file to write subtitles from. A provider that cannot express a term refuses the request " +
		"rather than dropping it silently."
	return schema
}

type TtsOptions struct {
	Target         *string                 `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"en-low-latency"`
	Providers      *[]string               `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, and each is expanded where it stands, so the order given is the order tried. Health only moves a provider that is down to the back." example:"[\"elevenlabs\", \"en-low-latency\"]"`
	Voice          *string                 `json:"voice,omitempty" doc:"A provider's own voice id, or one of your voices by id or by the name you gave it. Prefix it with custom: to mean only the latter: without the prefix a name that is not one of yours is passed through to the provider's library, and with it a name that is not one of yours is refused." example:"custom:receptionist"`
	Languages      *[]string               `json:"languages,omitempty"`
	Speed          *float32                `json:"speed,omitempty" doc:"Rate of delivery, 1 being the voice's own. Providers differ in the range they accept, so one asked for a speed outside its own refuses." example:"1"`
	Volume         *float32                `json:"volume,omitempty" doc:"Loudness, 1 being the voice's own."`
	Emotion        *string                 `json:"emotion,omitempty" doc:"Affect to speak with, for the providers that take one."`
	Style          *string                 `json:"style,omitempty" doc:"Delivery style, for the providers that name styles rather than emotions."`
	Stability      *float32                `json:"stability,omitempty" doc:"How much the voice may vary between chunks. Higher is flatter and more consistent."`
	Similarity     *float32                `json:"similarity,omitempty" doc:"How closely a cloned voice tracks its reference."`
	Format         *string                 `json:"format,omitempty" doc:"Codec, sample rate and bitrate as one name - pcm_16000, mp3_44100_128, ulaw_8000 for telephony." example:"pcm_16000"`
	Pronunciations *map[string]string      `json:"pronunciations,omitempty" doc:"How to say words the voice gets wrong, keyed by the word."`
	ChunkSchedule  *[]int                  `json:"chunk_schedule,omitempty" doc:"Character counts at which a streaming voice flushes audio. Smaller first values start speaking sooner and cost more requests. Live only."`
	DataPolicy     *DataPolicy             `json:"data_policy,omitempty"`
	Overwrites     *map[string]interface{} `json:"overwrites,omitempty" doc:"Settings for one voice provider that this vocabulary has no word for, keyed by provider name, for example {\"elevenlabs\": {\"voice_id\": \"21m00Tcm4TlvDq8ikWAM\"}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped. It is also the only way to steer a live voice per vendor, since a voice id from one library means nothing at another." example:"{\"elevenlabs\": {\"voice_id\": \"21m00Tcm4TlvDq8ikWAM\"}}"`
}

func (*TtsOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How this config speaks. A provider that cannot express a term refuses the request rather " +
		"than dropping it silently, since a voice asked to sound urgent and speaking flatly is worse " +
		"than one that says it cannot."
	return schema
}

type LlmOptionsFormat string

const (
	LlmOptionsFormatJsonObject LlmOptionsFormat = "json_object"
	LlmOptionsFormatText       LlmOptionsFormat = "text"
)

// Valid indicates whether the value is a known member of the LlmOptionsFormat enum.
func (e LlmOptionsFormat) Valid() bool {
	switch e {
	case LlmOptionsFormatJsonObject:
		return true
	case LlmOptionsFormatText:
		return true
	default:
		return false
	}
}

type LlmOptionsReasoningEffort string

const (
	LlmOptionsReasoningEffortHigh    LlmOptionsReasoningEffort = "high"
	LlmOptionsReasoningEffortLow     LlmOptionsReasoningEffort = "low"
	LlmOptionsReasoningEffortMedium  LlmOptionsReasoningEffort = "medium"
	LlmOptionsReasoningEffortMinimal LlmOptionsReasoningEffort = "minimal"
)

// Valid indicates whether the value is a known member of the LlmOptionsReasoningEffort enum.
func (e LlmOptionsReasoningEffort) Valid() bool {
	switch e {
	case LlmOptionsReasoningEffortHigh:
		return true
	case LlmOptionsReasoningEffortLow:
		return true
	case LlmOptionsReasoningEffortMedium:
		return true
	case LlmOptionsReasoningEffortMinimal:
		return true
	default:
		return false
	}
}

type LlmOptionsVerbosity string

const (
	LlmOptionsVerbosityHigh   LlmOptionsVerbosity = "high"
	LlmOptionsVerbosityLow    LlmOptionsVerbosity = "low"
	LlmOptionsVerbosityMedium LlmOptionsVerbosity = "medium"
)

// Valid indicates whether the value is a known member of the LlmOptionsVerbosity enum.
func (e LlmOptionsVerbosity) Valid() bool {
	switch e {
	case LlmOptionsVerbosityHigh:
		return true
	case LlmOptionsVerbosityLow:
		return true
	case LlmOptionsVerbosityMedium:
		return true
	default:
		return false
	}
}

type LlmOptions struct {
	Target          *string                    `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"llm-fast"`
	Providers       *[]string                  `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands. A response that fails is answered by the next entry that will have it." example:"[\"openai/gpt-5-mini\", \"llm-fast\"]"`
	MaxOutputTokens *int                       `json:"max_output_tokens,omitempty"`
	Temperature     *float32                   `json:"temperature,omitempty"`
	ReasoningEffort *LlmOptionsReasoningEffort `json:"reasoning_effort,omitempty" doc:"How long the model may think before answering, on the models that think." enum:"minimal,low,medium,high"`
	Format          *LlmOptionsFormat          `json:"format,omitempty" doc:"Whether the answer is prose or a JSON object." enum:"text,json_object"`
	Verbosity       *LlmOptionsVerbosity       `json:"verbosity,omitempty" enum:"low,medium,high"`
	ToolChoice      *string                    `json:"tool_choice,omitempty" doc:"auto, none, required, or the name of a tool the model must call. Which tools exist is per-request, since they change with the turn."`
	Store           *bool                      `json:"store,omitempty" doc:"Keep the response on the provider so a later one can continue from it."`
	PromptCacheKey  *string                    `json:"prompt_cache_key,omitempty" doc:"What a cached prompt prefix is keyed by. Requests sharing a key and a prefix are read from the cache rather than charged in full."`
	Metadata        *map[string]string         `json:"metadata,omitempty" doc:"Passed to the provider untouched, for the providers that store it."`
}

func (*LlmOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How this config answers. The names are the response parameters the router already speaks " +
		"rather than a second vocabulary for the same things. The system prompt is not among them: " +
		"what the model answers under belongs to the agent asking, not to the config that decides " +
		"where the asking goes."
	return schema
}

type StsOptionsTurnDetection string

const (
	StsOptionsTurnDetectionNone      StsOptionsTurnDetection = "none"
	StsOptionsTurnDetectionSemantic  StsOptionsTurnDetection = "semantic"
	StsOptionsTurnDetectionServerVad StsOptionsTurnDetection = "server_vad"
)

// Valid indicates whether the value is a known member of the StsOptionsTurnDetection enum.
func (e StsOptionsTurnDetection) Valid() bool {
	switch e {
	case StsOptionsTurnDetectionNone:
		return true
	case StsOptionsTurnDetectionSemantic:
		return true
	case StsOptionsTurnDetectionServerVad:
		return true
	default:
		return false
	}
}

type StsOptions struct {
	Target            *string                  `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"sts-fast"`
	Providers         *[]string                `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands." example:"[\"openai\", \"sts-fast\"]"`
	Instructions      *string                  `json:"instructions,omitempty" doc:"The system prompt the model converses under, sent when the session opens. A stored router config naming one is refused: what is said belongs to the agent holding the conversation, not to the config that decides where it goes."`
	Voice             *string                  `json:"voice,omitempty" doc:"The vendor's own name for a voice, such as marin at OpenAI or Kore at Google. None of these models takes one of your own voices, so the name is passed on as given rather than looked up." example:"marin"`
	Languages         *[]string                `json:"languages,omitempty"`
	TurnDetection     *StsOptionsTurnDetection `json:"turn_detection,omitempty" doc:"What decides the caller has finished: a silence timer, a model reading the words, or nothing, which leaves the turns to the caller. Omitting it leaves the vendor's default. Only some models read the words, so semantic is a term." enum:"server_vad,semantic,none"`
	SilenceMs         *int                     `json:"silence_ms,omitempty" doc:"How long a pause ends the turn, for a silence timer."`
	PrefixPaddingMs   *int                     `json:"prefix_padding_ms,omitempty" doc:"How much audio before the detected speech is kept, for a silence timer."`
	InterruptResponse *bool                    `json:"interrupt_response,omitempty" doc:"Whether the model cuts its own reply off when it hears the caller. Omitting it leaves the vendor's default; false is for a speaker close enough to the microphone that the model would otherwise interrupt itself."`
	InputTranscript   *bool                    `json:"input_transcript,omitempty" doc:"Ask the model to write down what it heard."`
	OutputTranscript  *bool                    `json:"output_transcript,omitempty" doc:"Ask the model to write down what it said."`
	Tools             *bool                    `json:"tools,omitempty" doc:"The session will hand the model functions to call."`
	Text              *bool                    `json:"text,omitempty" doc:"The session will inject typed turns."`
	Images            *bool                    `json:"images,omitempty" doc:"The session will send the model frames, so it is routed only to a model that sees, the way vlm routes a text model."`
	DataPolicy        *DataPolicy              `json:"data_policy,omitempty"`
	Overwrites        *map[string]interface{}  `json:"overwrites,omitempty" doc:"Settings for one provider that this vocabulary has no word for, keyed by provider name, for example {\"openai\": {\"eagerness\": \"high\"}}. The provider named parses its own block and refuses a field it does not have, so an overwrite is either sent or reported rather than accepted and dropped." example:"{\"openai\": {\"eagerness\": \"high\"}}"`
}

func (*StsOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How this config holds a conversation with one native audio model, in place of a " +
		"transcriber, a text model and a voice. What every such model takes is a field here; what " +
		"only some take is a term, and a request naming a term is routed to a model that declared it " +
		"or refused, never served by one that ignores it."
	return schema
}

type SearchOptionsContents string

const (
	SearchOptionsContentsHighlights SearchOptionsContents = "highlights"
	SearchOptionsContentsSummary    SearchOptionsContents = "summary"
	SearchOptionsContentsText       SearchOptionsContents = "text"
)

// Valid indicates whether the value is a known member of the SearchOptionsContents enum.
func (e SearchOptionsContents) Valid() bool {
	switch e {
	case SearchOptionsContentsHighlights:
		return true
	case SearchOptionsContentsSummary:
		return true
	case SearchOptionsContentsText:
		return true
	default:
		return false
	}
}

type SearchOptions struct {
	Target         *string                  `json:"target,omitempty" doc:"A provider/model or a capability shortcut." example:"search-fast"`
	Providers      *[]string                `json:"providers,omitempty" doc:"A priority list of where to try, in the order given, which wins over target and depth when it holds anything. Each entry is a provider name, a provider/model or a capability shortcut, expanded where it stands. A search that fails is asked of the next entry that will have it." example:"[\"exa\", \"search-fast\"]"`
	Depth          *SearchDepth             `json:"depth,omitempty"`
	Results        *int                     `json:"results,omitempty" doc:"How many hits to return." minimum:"1"`
	IncludeDomains *[]string                `json:"include_domains,omitempty" doc:"Only answer from these domains."`
	ExcludeDomains *[]string                `json:"exclude_domains,omitempty"`
	Category       *string                  `json:"category,omitempty" doc:"The kind of source to prefer - news, papers, company, github - for the providers that classify their index."`
	MaxAgeHours    *int                     `json:"max_age_hours,omitempty" doc:"How stale a cached page may be. Zero forces a live crawl, which is slower and costs more." minimum:"0"`
	Location       *string                  `json:"location,omitempty" doc:"Country or region to answer from, for queries whose answer depends on where."`
	Contents       *[]SearchOptionsContents `json:"contents,omitempty" doc:"What to return alongside each hit." enum:"text,highlights,summary"`
	OutputSchema   *map[string]interface{}  `json:"output_schema,omitempty" doc:"A JSON schema the answer must fit, for the providers that can be asked to structure what they found."`
}

func (*SearchOptions) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "How this config finds out today's answers."
	return schema
}

type Endpointing string

const (
	EndpointingSemantic Endpointing = "semantic"
	EndpointingSilence  Endpointing = "silence"
)

// Valid indicates whether the value is a known member of the Endpointing enum.
func (e Endpointing) Valid() bool {
	switch e {
	case EndpointingSemantic:
		return true
	case EndpointingSilence:
		return true
	default:
		return false
	}
}

func (Endpointing) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Endpointing", "What decides a turn is over: a long enough pause, or a model reading the words and "+
		"judging the sentence finished.",
		string(EndpointingSilence), string(EndpointingSemantic))
}

type TranscriptFormat string

const (
	Json TranscriptFormat = "json"
	Srt  TranscriptFormat = "srt"
	Vtt  TranscriptFormat = "vtt"
)

// Valid indicates whether the value is a known member of the TranscriptFormat enum.
func (e TranscriptFormat) Valid() bool {
	switch e {
	case Json:
		return true
	case Srt:
		return true
	case Vtt:
		return true
	default:
		return false
	}
}

func (TranscriptFormat) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "TranscriptFormat", "What a finished transcript is rendered as. json carries the words and speakers; srt and "+
		"vtt are subtitle files. Recording only.",
		string(Json), string(Srt), string(Vtt))
}

type TranscriptionMode string

const (
	Smart    TranscriptionMode = "smart"
	Verbatim TranscriptionMode = "verbatim"
)

// Valid indicates whether the value is a known member of the TranscriptionMode enum.
func (e TranscriptionMode) Valid() bool {
	switch e {
	case Smart:
		return true
	case Verbatim:
		return true
	default:
		return false
	}
}

func (TranscriptionMode) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "TranscriptionMode", "How faithfully the transcript follows what was said. verbatim keeps the ums, the "+
		"repetitions and the false starts; smart removes them, tidies the grammar and formats "+
		"the result, which is why it cannot also diarize or time the words - they may no longer "+
		"be the words that were spoken. Almost no provider offers both, so this narrows where a "+
		"request can go.",
		string(Verbatim), string(Smart))
}

type DataPolicy struct {
	AllowTraining *bool   `json:"allow_training,omitempty" doc:"False requires a provider that has said it does not train on what it is sent. Omitting this asks nothing. A provider that has published nothing either way counts as not having said no."`
	Retention     *string `json:"retention,omitempty" doc:"The longest a provider may keep this audio - none, or a duration such as 30d or 24h. Omitting it asks nothing." example:"none"`
}

func (*DataPolicy) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What a caller requires of what happens to what they send: the audio they had transcribed, " +
		"or the text they had spoken and the voice speaking it. This is a requirement rather than a " +
		"description: a request naming one is only routed to a model whose declared handling meets " +
		"it, and if none does the request is refused rather than sent somewhere that does not."
	return schema
}

type SearchDepth string

const (
	Deep     SearchDepth = "deep"
	Fast     SearchDepth = "fast"
	Instant  SearchDepth = "instant"
	Standard SearchDepth = "standard"
)

// Valid indicates whether the value is a known member of the SearchDepth enum.
func (e SearchDepth) Valid() bool {
	switch e {
	case Deep:
		return true
	case Fast:
		return true
	case Instant:
		return true
	case Standard:
		return true
	default:
		return false
	}
}

func (SearchDepth) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "SearchDepth", "How much work a search is worth. instant answers from the index in a few hundred "+
		"milliseconds; deep crawls and reasons over what it finds and can take tens of seconds. "+
		"Providers offer different ladders, so each one maps these four onto its own.",
		string(Instant), string(Fast), string(Standard), string(Deep))
}

type RouterConfigRequest struct {
	Name   string         `json:"name" doc:"What the config is called, which is unique among the customer's own."`
	Stt    *SttOptions    `json:"stt,omitempty"`
	Tts    *TtsOptions    `json:"tts,omitempty"`
	Llm    *LlmOptions    `json:"llm,omitempty"`
	Sts    *StsOptions    `json:"sts,omitempty"`
	Search *SearchOptions `json:"search,omitempty"`
}

type routerConfigListResponse struct {
	Body []RouterConfig
}

type createRouterConfigRequest struct {
	Body RouterConfigRequest
}

type routerConfigResponse struct {
	Body RouterConfig
}

type getRouterConfigRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

type updateRouterConfigRequest struct {
	ID   string `path:"id" doc:"The resource, as returned when it was created."`
	Body RouterConfigRequest
}

type deleteRouterConfigRequest struct {
	ID string `path:"id" doc:"The resource, as returned when it was created."`
}

func (s *Server) registerRouterConfigs(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listRouterConfigs",
		Method:      http.MethodGet,
		Path:        "/v1/router/configs",
		Summary:     "The router configs the calling customer holds",
		Description: "A router config is what an agent config is for a session, for a caller that " +
			"routes one modality at a time: the target, the language and every per-modality " +
			"option, decided once and named. It is separate from an agent config because it " +
			"configures transcribing, speaking, answering and searching on their own, with " +
			"no conversation behind them.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The customer's router configs, newest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.listRouterConfigs)
	huma.Register(api, huma.Operation{
		OperationID: "createRouterConfig",
		Method:      http.MethodPost,
		Path:        "/v1/router/configs",
		Summary:     "Store a named set of per-modality routing options",
		Description: "A modality block that names no target falls back to what a session falls back " +
			"to, so a config only has to say what it wants changed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		DefaultStatus: http.StatusCreated,
		Responses: map[string]*huma.Response{
			"201": {Description: "The config was stored"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.createRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "getRouterConfig",
		Method:      http.MethodGet,
		Path:        "/v1/router/configs/{id}",
		Summary:     "One router config",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "updateRouterConfig",
		Method:      http.MethodPut,
		Path:        "/v1/router/configs/{id}",
		Summary:     "Replace a router config",
		Description: "Every field is written, so the body is what the config now is rather than what " +
			"changed about it. Sockets already open keep the options they were started with.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The config as it now is"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.updateRouterConfig)
	huma.Register(api, huma.Operation{
		OperationID: "deleteRouterConfig",
		Method:      http.MethodDelete,
		Path:        "/v1/router/configs/{id}",
		Summary:     "Delete a router config",
		Description: "The requests that ran under it still name it, so the config stops being usable " +
			"rather than stops having existed.\n" +
			"Server-side only: it needs a server-side token, so it cannot be reached from an " +
			"end user's device.",
		Responses: map[string]*huma.Response{
			"204": {Description: "The config is gone"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.deleteRouterConfig)
}

// listRouterConfigs returns the calling customer's router configs, newest first.
func (s *Server) listRouterConfigs(ctx context.Context, _ *struct{}) (*routerConfigListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	stored, err := s.store.CustomerRouterConfigs(ctx, customerID)
	if err != nil {
		return nil, err
	}

	listed := make([]RouterConfig, 0, len(stored))
	for _, config := range stored {
		listed = append(listed, routerConfigOf(config))
	}
	return &routerConfigListResponse{Body: listed}, nil
}

// createRouterConfig stores a named set of per-modality routing options.
func (s *Server) createRouterConfig(ctx context.Context, request *createRouterConfigRequest) (*routerConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}
	if message, ok := s.routerConfigComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	config := storedRouterConfig(request.Body, customerID)
	if err := s.store.CreateRouterConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &routerConfigResponse{Body: routerConfigOf(config)}, nil
}

// getRouterConfig returns one router config.
func (s *Server) getRouterConfig(ctx context.Context, request *getRouterConfigRequest) (*routerConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	config, err := s.store.RouterConfig(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}
	return &routerConfigResponse{Body: routerConfigOf(config)}, nil
}

// updateRouterConfig replaces a router config with what it now is.
func (s *Server) updateRouterConfig(ctx context.Context, request *updateRouterConfigRequest) (*routerConfigResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}
	if message, ok := s.routerConfigComplaint(request.Body); !ok {
		return nil, huma.Error400BadRequest(message)
	}

	existing, err := s.store.RouterConfig(ctx, customerID, request.ID)
	if err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}

	config := storedRouterConfig(request.Body, customerID)
	config.ID = existing.ID
	config.CreatedAt = existing.CreatedAt
	if err := s.store.UpdateRouterConfig(ctx, &config); err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	return &routerConfigResponse{Body: routerConfigOf(config)}, nil
}

// deleteRouterConfig stops a router config being usable.
func (s *Server) deleteRouterConfig(ctx context.Context, request *deleteRouterConfigRequest) (*struct{}, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest(noRouterConfigs)
	}

	if err := s.store.DeleteRouterConfig(ctx, customerID, request.ID); err != nil {
		return nil, huma.Error404NotFound(unknownRouterConfig)
	}
	return nil, nil
}

// routerConfigComplaint reports what is wrong with a router config, if anything. The
// keyterms are checked here rather than left to the request that uses the config, because
// a config nothing can be routed under is worth hearing about while it is being written
// and not once a socket is open.
func (s *Server) routerConfigComplaint(request RouterConfigRequest) (string, bool) {
	if strings.TrimSpace(request.Name) == "" {
		return "a router config needs a name", false
	}
	if request.Stt != nil && request.Stt.Keyterms != nil && len(*request.Stt.Keyterms) > stt.MaxKeyterms {
		return fmt.Sprintf("a config may name at most %d keyterms", stt.MaxKeyterms), false
	}

	held := sttOptionsOf(request.Stt)
	if err := held.Validate(); err != nil {
		return err.Error(), false
	}
	if message, ok := s.sttComplaint(held); !ok {
		return message, false
	}

	voice := ttsOptionsOf(request.Tts)
	if err := voice.Validate(); err != nil {
		return err.Error(), false
	}
	if message, ok := s.ttsComplaint(voice); !ok {
		return message, false
	}

	conversation := stsOptionsOf(request.Sts)
	if err := conversation.Validate(); err != nil {
		return err.Error(), false
	}
	// A config decides where a conversation goes, not what is said in it. The agent
	// holding it has instructions of its own and sends them when it opens the session,
	// which is the only moment they are known.
	if conversation.Instructions != "" {
		return "a router config carries no instructions: the agent sends its own when it opens the session", false
	}
	if message, ok := s.stsComplaint(conversation); !ok {
		return message, false
	}
	if message, ok := s.chainComplaint(routing.LLM, llmOptionsOf(request.Llm).Providers); !ok {
		return message, false
	}
	if message, ok := s.chainComplaint(routing.Search, searchOptionsOf(request.Search).Providers); !ok {
		return message, false
	}
	return "", true
}

// chainComplaint is the priority list half of sttComplaint, for the modalities whose
// configs hold nothing else only their router can check.
func (s *Server) chainComplaint(modality routing.Modality, providers []string) (string, bool) {
	routed, ok := s.routerFor(Modality(modality))
	if !ok {
		return "", true
	}
	config := routed.Config()
	for _, target := range providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	return "", true
}

// sttComplaint reports what this deployment could never route, which is the half of a
// config's validity that only the router knows.
//
// The same reasoning as the keyterm limit, one step further: a config naming a provider
// this build has never heard of, or a data policy none of its models meet, would fail
// every request made under it. Saying so once, while it is being written, beats saying it
// on every call afterwards.
func (s *Server) sttComplaint(held options.STT) (string, bool) {
	speech, ok := s.routerFor(Modality(routing.STT))
	if !ok {
		return "", true
	}
	config := speech.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no provider this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no provider this deployment offers can serve every option in this config", false
	}
	// An overwrite for a vendor that does not exist is a typo, and the alternative to
	// reporting it is a setting that was stored, sent nowhere and never mentioned again.
	// Being a candidate for one particular request is not asked: which provider a call
	// lands on is the router's business, and a config that prepares for several is doing
	// the right thing.
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no provider for", vendor), false
		}
	}
	return "", true
}

// ttsComplaint is sttComplaint for the voice half, asked of the voice router's own config.
func (s *Server) ttsComplaint(held options.TTS) (string, bool) {
	voice, ok := s.routerFor(Modality(routing.TTS))
	if !ok {
		return "", true
	}
	config := voice.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no voice this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no voice this deployment offers can serve every option in this config", false
	}
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no voice for", vendor), false
		}
	}
	return "", true
}

// stsComplaint is sttComplaint for the speech-to-speech half, asked of that router's own
// config. Frames are checked here too: a config that says it will send them and names no
// model that sees would fail every request made under it.
func (s *Server) stsComplaint(held options.STS) (string, bool) {
	conversing, ok := s.routerFor(Modality(routing.STS))
	if !ok {
		return "", true
	}
	config := conversing.Config()

	for _, target := range held.Providers {
		if !config.Names(target) {
			return fmt.Sprintf(
				"%q is not a provider, a provider/model or a capability shortcut this deployment offers", target), false
		}
	}
	if !config.Meets(held.DataPolicy) {
		return "no speech-to-speech model this deployment offers meets that data policy", false
	}
	if !config.Expresses(held.Terms()) {
		return "no speech-to-speech model this deployment offers can serve every option in this config", false
	}
	if modalities := held.InputModalities(); len(modalities) > 0 && !slices.ContainsFunc(config.Providers, func(provider routing.ProviderConfig) bool {
		return provider.Sees(modalities)
	}) {
		return "no speech-to-speech model this deployment offers sees images", false
	}
	for vendor := range held.Overwrites {
		if !config.Declares(vendor) {
			return fmt.Sprintf("there are overwrites for %q, which this deployment has no speech-to-speech model for", vendor), false
		}
	}
	return "", true
}

// storedRouterConfig turns a request into a row. The customer comes from the trusted
// header rather than the body, the same way an agent config's does.
func storedRouterConfig(request RouterConfigRequest, customerID string) store.RouterConfig {
	config := store.RouterConfig{
		CustomerID: customerID,
		Name:       strings.TrimSpace(request.Name),
		STT:        sttOptionsOf(request.Stt),
		TTS:        ttsOptionsOf(request.Tts),
		LLM:        llmOptionsOf(request.Llm),
		STS:        stsOptionsOf(request.Sts),
		Search:     searchOptionsOf(request.Search),
	}
	config.STT.Keyterms = stt.CleanKeyterms(config.STT.Keyterms)
	return config
}

// routerConfigOf renders a config for the wire.
func routerConfigOf(config store.RouterConfig) RouterConfig {
	rendered := RouterConfig{
		Id:        config.ID,
		Name:      config.Name,
		Stt:       sttOptionsFor(config.STT),
		Tts:       ttsOptionsFor(config.TTS),
		Llm:       llmOptionsFor(config.LLM),
		Sts:       stsOptionsFor(config.STS),
		Search:    searchOptionsFor(config.Search),
		CreatedAt: config.CreatedAt,
		UpdatedAt: config.UpdatedAt,
	}
	return rendered
}

// routerOptions reads a stored config, if one was named, and writes the per-call options
// over it. Everything a config holds is a default; a keyword on the call overrides that
// one field of it.
//
// A config nobody can find is an error rather than an empty default: a caller that named
// one meant it, and transcribing at whatever the fallback happens to be is not what they
// asked for. It is found by id first and by name second, so a caller can say either.
func (s *Server) routerOptions(ctx context.Context, customerID, configID string) (store.RouterConfig, error) {
	if configID == "" {
		return store.RouterConfig{}, nil
	}
	if s.store == nil {
		return store.RouterConfig{}, fmt.Errorf("%s", noRouterConfigs)
	}

	if config, err := s.store.RouterConfig(ctx, customerID, configID); err == nil {
		return config, nil
	}
	config, found, err := s.store.RouterConfigByName(ctx, customerID, configID)
	if err != nil {
		return store.RouterConfig{}, err
	}
	if !found {
		return store.RouterConfig{}, fmt.Errorf("%s: %s", unknownRouterConfig, configID)
	}
	return config, nil
}

// tagsSent are the labels a request is billed with, which are the caller's own and
// nobody else's: a router config says where to route, not who to bill.
func tagsSent(sent *map[string]string) routing.Tags {
	tags := routing.Tags{}
	if sent != nil {
		for key, value := range *sent {
			tags[key] = value
		}
	}
	return tags
}

// These are what a block that names no target falls back to, which is what a session with
// nothing configured falls back to.
const (
	sttDefaultTarget = "en-low-latency"
	ttsDefaultTarget = "en-low-latency"
	llmDefaultTarget = "llm-fast"
	stsDefaultTarget = "sts-fast"
)

// targeted returns the options with a target filled in, since routing has to be told
// where to go and a caller that said nothing meant the usual place.
func targeted(held options.STT) options.STT {
	if held.Target == "" {
		held.Target = sttDefaultTarget
	}
	return held
}

// recordedTarget is where a job with no target of its own goes: the recorded aliases,
// which are the batch models rather than the live ones. A recording streamed at a socket
// would cost more and transcribe worse, so a caller who only said "transcribe this file"
// is not sent there.
func recordedTarget(languages []string) string {
	for _, language := range languages {
		if language != "" && !strings.HasPrefix(language, "en") {
			return "multilingual-recorded"
		}
	}
	return "en-recorded"
}
