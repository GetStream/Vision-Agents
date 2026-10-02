// Package api serves the router's HTTP surface on a chi router. Operations are declared
// in Go with Huma, and the Go structs are the source of truth: api/openapi.yaml is
// rendered from them by cmd/openapi. The handlers written by hand, the sockets and the
// streams, are described in api/legacy.yaml, and generated.go holds the models
// oapi-codegen generates from there; change that file and regenerate rather than editing
// generated.go.
//
// Every routing path is scoped by modality. The server holds one router per modality it
// serves and looks the right one up per request, so adding a modality is a matter of
// passing another router in.
package api

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/netip"
	"runtime/debug"
	"strings"
	"time"

	"github.com/danielgtaylor/huma/v2"
	sentryhttp "github.com/getsentry/sentry-go/http"
	"github.com/go-chi/chi/v5"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/campaign"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chat"
	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge/urls"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/phone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/plugins"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/quota"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/simulation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts/voices"
)

// CustomerHeader names the tenant directly, with no organization around it. It is what a
// local deployment with no proxy and no keys uses, and it is read in noauth and proxy
// modes and ignored entirely in api_key mode.
const CustomerHeader = auth.CustomerHeader

// CustomerParam carries the same identifier on the sockets, because the browser WebSocket
// API cannot set a header.
const CustomerParam = auth.CustomerParam

// customerContextKey holds the customer identifier extracted from the request.
type customerContextKey struct{}

// organizationContextKey holds the organization the customer belongs to, which is what a
// rate limit and a bill are counted against.
type organizationContextKey struct{}

// serverSideContextKey holds whether the caller is a process the customer runs.
type serverSideContextKey struct{}

// callerContextKey holds the end user the request is for and where they made it from, which
// is what a daily limit is counted against.
type callerContextKey struct{}

// kindContextKey holds what sort of caller it is, which is what qualifies the end user's
// name when one person's sessions are kept from another's.
type kindContextKey struct{}

// clientAccessibleExtension is what the spec marks the few operations an end user's device
// may reach with. Everything else is server-side only, and the check reads the mark from
// the operations themselves rather than from a list kept here, so what a generated SDK documents
// and what the server refuses cannot drift apart.
//
// The default is that way round because the two mistakes do not cost the same. An
// operation nobody thought about is refused to a browser, which arrives as a bug report;
// under the old default it was served to one, which arrives as a breach.
const clientAccessibleExtension = "x-client-accessible"

// Options configures a Server. The store and live client are optional; endpoints that
// need them report the dependency as unavailable rather than panicking.
type Options struct {
	// Routers is the router serving each modality. A modality that is absent is a 404.
	Routers map[routing.Modality]routing.Inspector
	Store   *store.Store
	Live    *live.Client
	// Phone serves the telephony paths. Absent when the deployment has no vendors, in
	// which case those paths say so rather than pretending numbers can be bought.
	Phone *phone.Service
	// Sessions runs conversations. Absent when the deployment only inspects routing, in
	// which case the session paths report that there are none rather than 500ing.
	Sessions *session.Manager
	// Streams serves the per-modality sockets, for callers running their own pipeline.
	// Absent when the deployment routes nothing itself.
	Streams *Streams
	// Transcripts reads back what was said on a call. Absent when the deployment has no
	// chat credentials, in which case nothing was written down to read.
	Transcripts *chatlog.Reader
	// Campaigns rings lists of people. Absent without telephony or sessions, in which
	// case a campaign can be written down but not run.
	Campaigns *campaign.Runner
	// Simulations puts an agent through a conversation somebody wrote down and rules on
	// how it went. Absent without sessions or model routing, in which case a simulation
	// can be written down but not run.
	Simulations *simulation.Runner
	// Knowledge fills the bases a config's knowledge_namespace has an agent read from.
	// Absent when the deployment has no knowledge provider, in which case there is nothing
	// to fill and the path says so.
	Knowledge knowledge.Writer
	// KnowledgeURLs keeps those bases filled from pages published elsewhere. Absent
	// without a database or without something that can read a page, in which case the url
	// paths say so rather than accepting a subscription nothing would honour.
	KnowledgeURLs *urls.Service
	// Voices holds the voices customers brought with them. Absent when the deployment has
	// no object storage, in which case there is nowhere to keep a recording and the voice
	// paths say so.
	Voices *voices.Service
	// VoiceLibrary reads the voices the speech providers themselves offer. Absent when no
	// provider that publishes one has a key here, in which case the library path says so
	// rather than answering with an empty catalogue.
	VoiceLibrary *voices.Catalogue
	// Dispatch holds the workers waiting to answer inbound calls. Absent when nothing is
	// meant to answer a phone, in which case the dispatch socket says so rather than
	// accepting a worker whose calls would never arrive.
	Dispatch *dispatch.Pool
	// StreamSecret signs the call events Stream sends. Without it the webhook refuses
	// every request, because an unsigned webhook is anyone who found the URL. It also
	// mints the tokens a browser joins a call with, which is why it never leaves here.
	StreamSecret string
	// StreamKey names the Stream app those tokens are for. A browser needs it to join,
	// so unlike the secret it is meant to be handed out.
	StreamKey string
	// CORSOrigins are the browser origins allowed to call this API directly, which is
	// what a dashboard talking to the router without a proxy in between needs. Empty
	// means no browser may, which is right for a deployment only servers reach.
	CORSOrigins []string
	// PublicURL is where this process is reachable, which plugin OAuth callbacks need.
	PublicURL string
	// DashboardURL is where a finished plugin login sends the browser.
	DashboardURL string
	// AuthMode is how this deployment decided that, which a handler needs when the mode
	// itself is the answer: moving a customer's data is refused outright in noauth,
	// where the tenant is a header rather than something anybody proved. Empty means
	// noauth, matching Auth being absent.
	AuthMode auth.Mode
	// DataRetention is how long a customer moving away has to finish, which is how long
	// their changes are recorded for.
	DataRetention time.Duration
	// Auth decides who a request is from. Absent means noauth, which reads the customer
	// header and takes every caller for that customer's own backend. That is the right
	// default for a server built in code rather than from configuration — a test, or a
	// deployment embedding this package — because there the absence is deliberate, where
	// an unset environment variable is somebody who has not thought about it yet and
	// gets api_key instead.
	Auth auth.Authenticator
	// Quota caps what one end user may spend in a day. Absent means nothing is capped,
	// which is right for a deployment with no Redis to count in and for one whose callers
	// are all backends the customer runs.
	Quota *quota.Limiter
	// Policies holds each organization's and app's budget, data policy and prompt
	// injection setting. Absent without a database, in which case the policy paths say so.
	Policies *policy.Enforcer
	// TrustedProxies are the ranges this deployment's own proxies sit in, and they decide
	// how much of X-Forwarded-For is believed when working out who a request is from.
	// Empty means none of it is, and the connection's own address is used.
	TrustedProxies []netip.Prefix
	Logger         *slog.Logger
}

// Server serves the router's HTTP API.
type Server struct {
	routers       map[routing.Modality]routing.Inspector
	store         *store.Store
	live          *live.Client
	phone         *phone.Service
	sessions      *session.Manager
	streams       *Streams
	transcripts   *chatlog.Reader
	campaigns     *campaign.Runner
	simulations   *simulation.Runner
	knowledge     knowledge.Writer
	pages         *urls.Service
	voices        *voices.Service
	library       *voices.Catalogue
	dispatch      *dispatch.Pool
	streamSecret  string
	streamKey     string
	corsOrigins   []string
	publicURL     string
	dashboardURL  string
	oauth         *plugins.Auth
	authenticator auth.Authenticator
	authMode      auth.Mode
	dataRetention time.Duration
	quota         *quota.Limiter
	policies      *policy.Enforcer
	trusted       []netip.Prefix
	// serverSide matches the requests the spec marks server-side only. It holds no
	// handlers: what is registered on it is the patterns, and matching one is the answer.
	serverSide *http.ServeMux
	upgrader   websocket.Upgrader
	popularity *popularity
	logger     *slog.Logger
}

// Option adjusts the options a server is built from. It exists for the settings a
// deployment supplies as code rather than as configuration, which cannot be written in the
// struct a configuration file is decoded into.
type Option func(*Options)

// WithAuthenticator supplies an authenticator of the deployment's own, which is the whole
// of auth.Custom: the mode names an answer this module does not have, and this is where
// the answer arrives. Anything satisfying auth.Authenticator will do, and auth.Func makes
// one out of a function.
type Provider struct {
	Provider    string             `json:"provider" example:"elevenlabs"`
	Model       string             `json:"model" example:"eleven_flash_v2_5"`
	Description *string            `json:"description,omitempty" doc:"What the model is good at, and what that costs in speed or money. Empty if the deployment wrote none."`
	Languages   []string           `json:"languages"`
	Realtime    bool               `json:"realtime"`
	Tier        Tier               `json:"tier"`
	Health      ProviderHealth     `json:"health"`
	UsageShare  *float64           `json:"usage_share,omitempty" doc:"This model's share of the modality's requests over the last seven days, across every customer, from 0 to 1. It is how popular the model is, and is 0 when nothing was served or the deployment keeps no statistics."`
	Benchmark   *ProviderBenchmark `json:"benchmark,omitempty"`
	Price       *ProviderPrice     `json:"price,omitempty"`
}

type Tier string

const (
	HighQuality Tier = "high-quality"
	LowLatency  Tier = "low-latency"
)

// Valid indicates whether the value is a known member of the Tier enum.
func (e Tier) Valid() bool {
	switch e {
	case HighQuality:
		return true
	case LowLatency:
		return true
	default:
		return false
	}
}

func (Tier) Schema(registry huma.Registry) *huma.Schema {
	return namedEnum(registry, "Tier", "What the model optimises for.",
		string(LowLatency), string(HighQuality))
}

type ProviderHealth struct {
	Available    bool    `json:"available" doc:"False once the error rate crosses the configured threshold."`
	Requests     int64   `json:"requests" doc:"Requests seen in the current health window."`
	Errors       int64   `json:"errors"`
	ErrorRate    float64 `json:"error_rate"`
	LatencyMsAvg float64 `json:"latency_ms_avg"`
}

type ProviderBenchmark struct {
	Elo                   *int     `json:"elo,omitempty" doc:"Speech arena Elo rating of a text-to-speech model." example:"1273"`
	CharactersPerSecond   *float64 `json:"characters_per_second,omitempty" doc:"Characters a text-to-speech model synthesises per second on the vendor's API." example:"115"`
	WordErrorRate         *float64 `json:"word_error_rate,omitempty" doc:"Streaming AA-WER of a speech-to-text model, from 0 to 1." example:"0.027"`
	LatencyMs             *int     `json:"latency_ms,omitempty" doc:"Milliseconds a speech-to-text model takes to its final transcript after speech ends." example:"490"`
	SearchIndex           *int     `json:"search_index,omitempty" doc:"Artificial Analysis Search Index of a search provider, from 0 to 100." example:"74"`
	CostPerTask           *float64 `json:"cost_per_task,omitempty" doc:"US dollars one task of the search benchmark cost, searches and the answering model's tokens together." example:"0.127"`
	IntelligenceIndex     *int     `json:"intelligence_index,omitempty" doc:"Artificial Analysis Intelligence Index of a text model at the reasoning effort the router asks for." example:"33"`
	OutputTokensPerSecond *float64 `json:"output_tokens_per_second,omitempty" doc:"Tokens a text model writes per second on the host the router calls." example:"330"`
}

func (*ProviderBenchmark) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What Artificial Analysis measured for this model, refreshed by hand rather than live. A " +
		"field is absent when the model was not measured on it."
	return schema
}

type ProviderPrice struct {
	PerMillionInputTokens  *float64 `json:"per_million_input_tokens,omitempty" example:"0.75"`
	PerMillionOutputTokens *float64 `json:"per_million_output_tokens,omitempty" example:"3.75"`
}

func (*ProviderPrice) TransformSchema(_ huma.Registry, schema *huma.Schema) *huma.Schema {
	schema.Description = "What this deployment is billed for the model, in US dollars. A rate is absent when the " +
		"model is not billed by that unit."
	return schema
}

type Route struct {
	Id          string      `json:"id" doc:"The shortcut, which is what a config or request names as its target." example:"llm-fast"`
	Title       string      `json:"title" example:"Fast conversational"`
	Description string      `json:"description"`
	Candidates  []Candidate `json:"candidates" doc:"The models the shortcut resolves to right now, best first."`
}

type Candidate struct {
	Provider string         `json:"provider"`
	Model    string         `json:"model"`
	Health   ProviderHealth `json:"health"`
}

type Granularity string

const (
	GranularityDaily  Granularity = "daily"
	GranularityHourly Granularity = "hourly"
)

// Valid indicates whether the value is a known member of the Granularity enum.
func (e Granularity) Valid() bool {
	switch e {
	case GranularityDaily:
		return true
	case GranularityHourly:
		return true
	default:
		return false
	}
}

func (Granularity) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "Granularity", "",
		string(GranularityHourly), string(GranularityDaily))
	registry.Map()["Granularity"].Default = "hourly"
	return ref
}

type StatsBucket struct {
	Provider               string    `json:"provider"`
	Model                  string    `json:"model"`
	Bucket                 time.Time `json:"bucket"`
	AudioMsTotal           int64     `json:"audio_ms_total" doc:"Billable audio, transcribed or produced."`
	CharactersTotal        int64     `json:"characters_total" doc:"Billable text. Zero for providers that bill by audio."`
	InputTokensTotal       int64     `json:"input_tokens_total" doc:"Prompt tokens read, cached ones included. Zero outside llm."`
	CachedInputTokensTotal int64     `json:"cached_input_tokens_total" doc:"The part of the prompt served from the provider's cache."`
	OutputTokensTotal      int64     `json:"output_tokens_total" doc:"Generated tokens, reasoning included. Zero outside llm."`
	ImagesTotal            int64     `json:"images_total" doc:"Pictures drawn. Zero outside image."`
	CostMicrosTotal        int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	RequestCount           int64     `json:"request_count"`
	ErrorCount             int64     `json:"error_count"`
	LatencyP50Ms           *float64  `json:"latency_p50_ms,omitempty" nullable:"true"`
	LatencyP95Ms           *float64  `json:"latency_p95_ms,omitempty" nullable:"true"`
	Uptime                 *float64  `json:"uptime,omitempty" doc:"Successes over total requests in the bucket." nullable:"true"`
}

type TagStatsBucket struct {
	TagKey                 string    `json:"tag_key" example:"project"`
	TagValue               string    `json:"tag_value" example:"moderation"`
	Bucket                 time.Time `json:"bucket"`
	AudioMsTotal           int64     `json:"audio_ms_total"`
	CharactersTotal        int64     `json:"characters_total"`
	InputTokensTotal       int64     `json:"input_tokens_total"`
	CachedInputTokensTotal int64     `json:"cached_input_tokens_total"`
	OutputTokensTotal      int64     `json:"output_tokens_total"`
	ImagesTotal            int64     `json:"images_total"`
	CostMicrosTotal        int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	RequestCount           int64     `json:"request_count"`
	ErrorCount             int64     `json:"error_count"`
	LatencyP50Ms           *float64  `json:"latency_p50_ms,omitempty" nullable:"true"`
	LatencyP95Ms           *float64  `json:"latency_p95_ms,omitempty" nullable:"true"`
	Uptime                 *float64  `json:"uptime,omitempty" nullable:"true"`
}

type TurnStatsBucket struct {
	AgentId          string    `json:"agent_id"`
	Bucket           time.Time `json:"bucket"`
	TurnCount        int64     `json:"turn_count"`
	InterruptedCount int64     `json:"interrupted_count" doc:"Turns a participant talked over before they finished."`
	AudioOutMsTotal  float64   `json:"audio_out_ms_total" doc:"How much speech the agent published in the bucket."`
	SttLatencyP50Ms  *float64  `json:"stt_latency_p50_ms,omitempty" nullable:"true"`
	SttLatencyP95Ms  *float64  `json:"stt_latency_p95_ms,omitempty" nullable:"true"`
	LlmTtftP50Ms     *float64  `json:"llm_ttft_p50_ms,omitempty" nullable:"true"`
	LlmTtftP95Ms     *float64  `json:"llm_ttft_p95_ms,omitempty" nullable:"true"`
	TtsTtfbP50Ms     *float64  `json:"tts_ttfb_p50_ms,omitempty" nullable:"true"`
	TtsTtfbP95Ms     *float64  `json:"tts_ttfb_p95_ms,omitempty" nullable:"true"`
	RoundtripP50Ms   *float64  `json:"roundtrip_p50_ms,omitempty" doc:"Settled transcript to first audio published." nullable:"true"`
	RoundtripP95Ms   *float64  `json:"roundtrip_p95_ms,omitempty" nullable:"true"`
	RoundtripP99Ms   *float64  `json:"roundtrip_p99_ms,omitempty" nullable:"true"`
}

type SpendBucket struct {
	Bucket          time.Time `json:"bucket"`
	Value           string    `json:"value" doc:"The modality or label value this row is for. \"other\" is everything outside the biggest few, and the empty string is spend carrying no such label at all, so a customer that labels only part of its traffic can see which part." example:"support"`
	CostMicrosTotal int64     `json:"cost_micros_total" doc:"Millionths of a dollar, priced from the configured rates."`
	RequestCount    int64     `json:"request_count"`
}

type TagKeySummary struct {
	Key             string            `json:"key" example:"product"`
	ValueCount      int64             `json:"value_count" doc:"How many distinct values the key was used with. One means it is context rather than a breakdown; hundreds mean it identifies something, such as an end customer, and only its largest values are worth a chart."`
	CostMicrosTotal int64             `json:"cost_micros_total"`
	RequestCount    int64             `json:"request_count"`
	Coverage        float64           `json:"coverage" doc:"The share of the window's requests that carry this key, from 0 to 1. A key on half the traffic breaks down half the bill, which is worth knowing before it is read as the whole of it."`
	TopValues       []TagValueSummary `json:"top_values" doc:"The ten largest values, biggest spend first."`
}

type TagValueSummary struct {
	Value           string  `json:"value" example:"support"`
	CostMicrosTotal int64   `json:"cost_micros_total"`
	RequestCount    int64   `json:"request_count"`
	Share           float64 `json:"share" doc:"This value's share of what the key covers, from 0 to 1."`
}

type ActivityGranularity string

const (
	ActivityGranularityDaily   ActivityGranularity = "daily"
	ActivityGranularityMonthly ActivityGranularity = "monthly"
)

// Valid indicates whether the value is a known member of the ActivityGranularity enum.
func (e ActivityGranularity) Valid() bool {
	switch e {
	case ActivityGranularityDaily:
		return true
	case ActivityGranularityMonthly:
		return true
	default:
		return false
	}
}

func (ActivityGranularity) Schema(registry huma.Registry) *huma.Schema {
	ref := namedEnum(registry, "ActivityGranularity", "Separate from Granularity, and coarser, because distinct users cannot be summed: a "+
		"month of them is who came back rather than the sum of its days.",
		string(ActivityGranularityDaily), string(ActivityGranularityMonthly))
	registry.Map()["ActivityGranularity"].Default = "daily"
	return ref
}

type ActivityBucket struct {
	Bucket       time.Time `json:"bucket"`
	ActiveUsers  int64     `json:"active_users" doc:"Distinct end users who opened a session or asked something of an agent in the bucket. A guest who later turned out to be a known user counts as that user.\nA caller that named nobody is not counted, and neither is an anonymous one: an anonymous name is a claim nothing verified, so counting it would make guessing a name enough to inflate this."`
	Sessions     int64     `json:"sessions"`
	Messages     int64     `json:"messages" doc:"Responses the agents produced, which is one per thing asked of them."`
	Calls        int64     `json:"calls"`
	VoiceMinutes float64   `json:"voice_minutes" doc:"How long those calls lasted. One still running counts up to now."`
	PhoneMinutes float64   `json:"phone_minutes" doc:"The part of voice_minutes that arrived over a phone number."`
}

type RollupRequest struct {
	Granularity *Granularity `json:"granularity,omitempty"`
	From        time.Time    `json:"from"`
	To          time.Time    `json:"to"`
}

type RollupResult struct {
	Granularity    Granularity `json:"granularity"`
	BucketsWritten int64       `json:"buckets_written"`
}

type listProvidersRequest struct {
	Modality Modality `path:"modality" doc:"Which kind of model to route."`
}

type providerListResponse struct {
	Body []Provider
}

type listRoutesRequest struct {
	Modality Modality `path:"modality" doc:"Which kind of model to route."`
}

type routeListResponse struct {
	Body []Route
}

type resolveTargetRequest struct {
	Modality Modality                `path:"modality" doc:"Which kind of model to route."`
	Target   string                  `path:"target" doc:"A \"provider/model\" name or a capability shortcut such as en-low-latency." example:"en-low-latency"`
	Language optionalParam[[]string] `query:"language,explode" doc:"Language hints that candidates must cover. Repeat for several." example:"[\"en\"]"`
}

type candidateListResponse struct {
	Body []Candidate
}

type getStatsRequest struct {
	Modality    Modality                   `path:"modality" doc:"Which kind of model to route."`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To          time.Time                  `query:"to" required:"true" doc:"End of the window, exclusive."`
	Tag         optionalParam[[]string]    `query:"tag,explode" doc:"Only count requests carrying every one of these cost labels, each written \"key:value\". Repeat for several. Filtering reads the request rows rather than the rollups, since a rollup bucket no longer knows which labels its requests carried." example:"[\"project:moderation\", \"environment:dev\"]"`
}

type statsBucketListResponse struct {
	Body []StatsBucket
}

type getTagStatsRequest struct {
	Modality    Modality                   `path:"modality" doc:"Which kind of model to route."`
	Key         string                     `query:"key" required:"true" doc:"The cost label to group by." example:"project"`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To          time.Time                  `query:"to" required:"true" doc:"End of the window, exclusive."`
}

type tagStatsBucketListResponse struct {
	Body []TagStatsBucket
}

type getTurnStatsRequest struct {
	AgentID     optionalParam[string]      `query:"agent_id" doc:"Narrow to one agent. Omit for every agent the customer runs."`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To          time.Time                  `query:"to" required:"true" doc:"End of the window, exclusive."`
}

type turnStatsBucketListResponse struct {
	Body []TurnStatsBucket
}

type getSpendRequest struct {
	GroupBy     string                     `query:"group_by" doc:"\"modality\", or the cost label to group by." default:"modality" example:"product"`
	Granularity optionalParam[Granularity] `query:"granularity"`
	From        time.Time                  `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To          time.Time                  `query:"to" required:"true" doc:"End of the window, exclusive."`
	Limit       int                        `query:"limit" doc:"How many values keep a series of their own." minimum:"1" maximum:"50" default:"6"`
	Tag         optionalParam[[]string]    `query:"tag,explode" doc:"Only count requests carrying every one of these cost labels, each written \"key:value\". Repeat for several." example:"[\"product:support\", \"environment:production\"]"`
}

type spendBucketListResponse struct {
	Body []SpendBucket
}

type getTagKeysRequest struct {
	From time.Time               `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To   time.Time               `query:"to" required:"true" doc:"End of the window, exclusive."`
	Tag  optionalParam[[]string] `query:"tag,explode" doc:"Only consider requests carrying every one of these cost labels, each written \"key:value\". Repeat for several." example:"[\"product:support\"]"`
}

type tagKeySummaryListResponse struct {
	Body []TagKeySummary
}

type getActivityRequest struct {
	Granularity optionalParam[ActivityGranularity] `query:"granularity"`
	From        time.Time                          `query:"from" required:"true" doc:"Start of the window, inclusive."`
	To          time.Time                          `query:"to" required:"true" doc:"End of the window, exclusive."`
}

type activityBucketListResponse struct {
	Body []ActivityBucket
}

type runRollupRequest struct {
	Body RollupRequest
}

type rollupResultResponse struct {
	Body RollupResult
}

func (s *Server) registerRouting(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "listProviders",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/providers",
		Summary:     "List the providers configured for a modality and their live health",
		Responses: map[string]*huma.Response{
			"200": {Description: "The configured providers"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listProviders)
	huma.Register(api, huma.Operation{
		OperationID: "listRoutes",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/routes",
		Summary: "List the capability shortcuts offered as a choice, each with the models it " +
			"resolves to",
		Description: "The shortcuts a person picking a model is shown, in the order the deployment " +
			"offers them, so the first is the one a conversation gets by default. Shortcuts " +
			"that exist only for the router's own use are left out, though they can still be " +
			"named as a target.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The offered shortcuts, default first"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.listRoutes)
	huma.Register(api, huma.Operation{
		OperationID: "resolveTarget",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/routes/{target}",
		Summary:     "Resolve a provider name or capability shortcut to a ranked candidate list",
		Responses: map[string]*huma.Response{
			"200": {Description: "Candidates in preference order, best first"},
		},
		Errors: []int{http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.resolveTarget)
	huma.Register(api, huma.Operation{
		OperationID: "getStats",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/stats",
		Summary:     "Aggregated usage for the calling customer",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket, provider and model"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getStats)
	huma.Register(api, huma.Operation{
		OperationID: "getTagStats",
		Method:      http.MethodGet,
		Path:        "/v1/{modality}/stats/tags",
		Summary:     "Aggregated usage broken down by the values of one cost label",
		Description: "What drives the spend. Requests are labelled with whatever keys the customer " +
			"chooses, so asking for key=project returns one row per project per bucket.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and label value"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden, http.StatusNotFound},
	}, s.getTagStats)
	huma.Register(api, huma.Operation{
		OperationID: "getTurnStats",
		Method:      http.MethodGet,
		Path:        "/v1/turns/stats",
		Summary:     "Conversational latency for the calling customer",
		Description: "One row per bucket and agent. A request row measures one provider call; a turn " +
			"measures what the caller felt, from finishing a sentence to hearing the answer " +
			"start, with the transcription, model and voice legs kept apart so a slow " +
			"conversation can be attributed.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and agent"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getTurnStats)
	huma.Register(api, huma.Operation{
		OperationID: "getSpend",
		Method:      http.MethodGet,
		Path:        "/v1/stats/spend",
		Summary:     "What the calling customer spent, grouped",
		Description: "Spend across every modality at once, which is what a bill is. group_by decides " +
			"what the series are: \"modality\" for where the money went, or a cost label for " +
			"what it was spent on.\n" +
			"Only the biggest values keep a series of their own, because a label such as " +
			"customer_id has as many values as the customer has customers. The rest are " +
			"summed into \"other\", and spend carrying no such label at all into the empty " +
			"value, so the rows still add up to the total.\n" +
			"Reads the request rows rather than the rollups, so today's spend is there " +
			"without a rollup having run.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket and group, oldest bucket first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getSpend)
	huma.Register(api, huma.Operation{
		OperationID: "getTagKeys",
		Method:      http.MethodGet,
		Path:        "/v1/stats/tags/keys",
		Summary:     "Which cost labels the calling customer's spend carries",
		Description: "Cost labels are the customer's own, so nothing here knows in advance whether " +
			"spend is broken down by product, by environment or by the end customer it was " +
			"incurred for. This reports the keys in use and what each covers, so a reader " +
			"can be shown the breakdown that means something rather than a list to guess " +
			"from.\n" +
			"A key every request carries with a single value -- environment: production and " +
			"nothing else -- is context rather than a breakdown, and value_count says so.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per label key, largest spend first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getTagKeys)
	huma.Register(api, huma.Operation{
		OperationID: "getActivity",
		Method:      http.MethodGet,
		Path:        "/v1/stats/activity",
		Summary:     "Who used the calling customer's agents, and how much",
		Description: "Sessions opened, responses produced and calls held, counted per bucket, " +
			"alongside how many distinct people were behind them.\n" +
			"Distinct users are counted rather than summed, which is why the granularity " +
			"here is days or months rather than the hours the spend paths take: a month's " +
			"active users are the people who came back, not the sum of its days, so a month " +
			"has to be asked for as a month.",
		Responses: map[string]*huma.Response{
			"200": {Description: "One row per bucket, oldest first"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.getActivity)
	huma.Register(api, huma.Operation{
		OperationID: "runRollup",
		Method:      http.MethodPost,
		Path:        "/v1/stats/rollup",
		Summary:     "Aggregate request rows into a rollup table",
		Description: "Covers every modality and customer in the window. Idempotent: re-running it " +
			"over the same window recomputes those buckets, so a missed run is fixed by " +
			"running it again.",
		Responses: map[string]*huma.Response{
			"200": {Description: "The rollup completed"},
		},
		Errors: []int{http.StatusBadRequest, http.StatusUnauthorized, http.StatusForbidden},
	}, s.runRollup)
}

// It overrides whatever ROUTER_AUTH_MODE asked for, because a deployment that compiled an
// authenticator in meant it.
func WithAuthenticator(authenticator auth.Authenticator) Option {
	return func(options *Options) { options.Auth = authenticator }
}

// NewServer wires the handlers.
func NewServer(options Options, with ...Option) (*Server, error) {
	for _, option := range with {
		option(&options)
	}
	if len(options.Routers) == 0 {
		return nil, errors.New("api: at least one router is required")
	}
	for modality, router := range options.Routers {
		if router == nil {
			return nil, errors.New("api: router for " + string(modality) + " is nil")
		}
	}

	authenticator := options.Auth
	if authenticator == nil {
		// A deployment that names no mode is a local one, where the customer header is
		// the whole of the story.
		var err error
		if authenticator, err = auth.New(auth.NoAuth, nil); err != nil {
			return nil, err
		}
	}

	authMode := options.AuthMode
	if authMode == "" {
		authMode = auth.NoAuth
	}
	// A deployment that names no window still records changes for somebody moving, for
	// as long as the settings say by default.
	retention := options.DataRetention
	if retention <= 0 {
		retention = 7 * 24 * time.Hour
	}

	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	server := &Server{
		routers:       options.Routers,
		store:         options.Store,
		live:          options.Live,
		phone:         options.Phone,
		sessions:      options.Sessions,
		streams:       options.Streams,
		transcripts:   options.Transcripts,
		campaigns:     options.Campaigns,
		simulations:   options.Simulations,
		knowledge:     options.Knowledge,
		pages:         options.KnowledgeURLs,
		voices:        options.Voices,
		library:       options.VoiceLibrary,
		dispatch:      options.Dispatch,
		streamSecret:  options.StreamSecret,
		streamKey:     options.StreamKey,
		corsOrigins:   options.CORSOrigins,
		publicURL:     options.PublicURL,
		dashboardURL:  options.DashboardURL,
		authenticator: authenticator,
		authMode:      authMode,
		dataRetention: retention,
		quota:         options.Quota,
		policies:      options.Policies,
		trusted:       options.TrustedProxies,
		upgrader:      newUpgrader(options.CORSOrigins),
		oauth: &plugins.Auth{
			PublicURL:    options.PublicURL,
			DashboardURL: options.DashboardURL,
		},
		popularity: newPopularity(options.Store, logger),
		logger:     logger,
	}
	serverSide, err := serverSideRoutes(server.newAPI(chi.NewRouter()).OpenAPI())
	if err != nil {
		return nil, err
	}
	server.serverSide = serverSide
	return server, nil
}

// Handler returns the HTTP handler for the whole API.
//
// The three sockets, the answer host and the call hook are registered first, on a router
// the Huma operations are then added to. The sockets are written by hand because a Huma
// operation returns a response object and an upgrade returns a connection, so there is
// nothing for it to hand back. The answer host is written by hand because it serves a
// vendor's XML rather than this API's JSON, and the call hook because both are reached by
// somebody other than a customer: a telephony vendor and Stream.
func (s *Server) Handler() http.Handler {
	mux := chi.NewRouter()
	mux.HandleFunc("GET /v1/agents/logs", s.listAgentLogs)
	mux.HandleFunc("GET /v1/agents/logs/stream", s.streamAgentLogs)
	mux.HandleFunc("GET /v1/agents/logs/{id}", s.getAgentLog)
	mux.HandleFunc("GET /v1/data/export", s.exportData)
	mux.HandleFunc("POST /v1/data/import", s.importData)
	mux.HandleFunc("GET /v1/data/changes", s.listDataChanges)
	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", s.watchSession)
	mux.HandleFunc("GET /v1/agents/socket", s.openSocketSession)
	mux.HandleFunc("GET /v1/{modality}/stream", s.streamModality)
	mux.HandleFunc("GET /v1/dispatch", s.dispatchCalls)
	mux.HandleFunc("GET /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST /v1/phone/answer/{token}", s.answerPhoneCall)
	mux.HandleFunc("POST "+phone.CallHookPath, s.receiveCallEvent)
	mux.HandleFunc("POST "+chat.MessageHookPath, s.receiveMessageEvent)
	mux.HandleFunc("GET "+plugins.CallbackPath, s.finishPluginLogin)
	s.newAPI(mux)
	// Sentry is outermost so it sees panics from every middleware below it, not
	// only from the route handlers.
	//
	// Repanic is false, which is a change in behaviour worth knowing about: this
	// service had no recovery anywhere, so a panic in one request used to take
	// the process down, and the router runs as a single pod -- every call it was
	// carrying went with it. Answering that one request with a 500 and leaving
	// the rest connected is the better trade.
	//
	// WaitForDelivery is false because most of what is served here is a long-
	// lived socket; blocking the handler's return on event delivery would hold
	// the connection open past its use. The flush in cmd/router covers shutdown.
	instrumented := sentryhttp.New(sentryhttp.Options{
		Repanic:         false,
		WaitForDelivery: false,
	})
	return instrumented.Handle(withCORS(s.corsOrigins,
		s.withCustomer(s.withRequestLog(s.withQuota(s.withServerSide(mux))))))
}

// withRequestLog records one line per request served.
//
// It sits inside withCustomer so it can name the caller, and outside withServerSide so a
// refusal is a logged 403 rather than a request that appears not to have arrived. It is
// inside withCORS, which means a preflight goes unlogged: it carries nothing worth routing
// and doubling the volume to record that a browser asked permission is a poor trade.
//
// The path is logged without the query, because a socket names its customer there and a
// vendor names a token, and an access log is the last place either should end up.
//
// A 5xx is logged at error level. An access log at a busy deployment is the one stream
// nobody reads all of, and a server error that only appears in it is a server error nobody
// notices.
//
// A panic is logged with its stack and answered with a 500 here, then panicked again so
// Sentry still reports it. Sentry recovers without writing a status, which net/http sends
// as an empty 200, and a request that panicked would otherwise leave no line at all.
func (s *Server) withRequestLog(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		started := time.Now()
		recorder := &loggedResponse{ResponseWriter: w}
		defer func() {
			recovered := recover()
			if recovered == nil {
				return
			}
			// ErrAbortHandler is net/http's own way of dropping a connection, not a bug.
			if recovered != http.ErrAbortHandler {
				customer, _ := CustomerFrom(r.Context())
				s.logger.Error("a request panicked",
					"method", r.Method, "path", r.URL.Path, "customer", customer,
					"panic", fmt.Sprint(recovered), "stack", string(debug.Stack()))
				if recorder.code == 0 && recorder.written == 0 && !recorder.hijacked {
					http.Error(recorder, `{"error":"internal error"}`, http.StatusInternalServerError)
				}
			}
			panic(recovered)
		}()
		next.ServeHTTP(recorder, r)

		customer, _ := CustomerFrom(r.Context())
		// A socket reports the life of the connection rather than a time to respond,
		// since it is logged once it has closed.
		fields := []any{
			"method", r.Method,
			"path", r.URL.Path,
			"status", recorder.status(),
			"duration", time.Since(started).Round(time.Millisecond),
			"customer", customer,
		}
		if recorder.written > 0 {
			fields = append(fields, "bytes", recorder.written)
		}
		if recorder.status() >= http.StatusInternalServerError {
			s.logger.Error("served a request", fields...)
			return
		}
		s.logger.Info("served a request", fields...)
	})
}

// loggedResponse remembers what was answered so it can be logged once the handler is done.
type loggedResponse struct {
	http.ResponseWriter
	code     int
	written  int64
	hijacked bool
}

// status reports what the caller was told, filling in the two codes a handler can answer
// with without ever naming: writing a body implies a 200, and writing nothing at all is the
// 200 net/http sends when the handler returns.
func (l *loggedResponse) status() int {
	switch {
	case l.code != 0:
		return l.code
	case l.hijacked:
		return http.StatusSwitchingProtocols
	default:
		return http.StatusOK
	}
}

func (l *loggedResponse) WriteHeader(code int) {
	if l.code == 0 {
		l.code = code
	}
	l.ResponseWriter.WriteHeader(code)
}

func (l *loggedResponse) Write(body []byte) (int, error) {
	written, err := l.ResponseWriter.Write(body)
	l.written += int64(written)
	return written, err
}

// Hijack hands the connection over for a socket upgrade, and is declared here rather than
// left to Unwrap so that the upgrade is what gets logged instead of an empty 200.
func (l *loggedResponse) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	conn, buffered, err := http.NewResponseController(l.ResponseWriter).Hijack()
	if err == nil {
		l.hijacked = true
	}
	return conn, buffered, err
}

// Unwrap lets a handler reach the flushing and deadline setting of the writer underneath
// through http.ResponseController, which a streamed response needs.
func (l *loggedResponse) Unwrap() http.ResponseWriter {
	return l.ResponseWriter
}

// unspecifiedRoutes are the hand-written handlers, and whether a client may reach each.
//
// They are named here because they are not Huma operations — one can express neither an
// upgrade nor a stream — so the middleware cannot read their marks off the document. An
// operation the middleware cannot see is the one place an inverted default could fail
// open, so this is the complement's other half rather than a note about sockets. A test
// holds it to naming exactly the operations api/legacy.yaml declares.
var unspecifiedRoutes = map[string]bool{
	"GET /v1/agents/sessions/{id}/events": true,
	"GET /v1/{modality}/stream":           false,
	"GET /v1/dispatch":                    false,
	"GET /v1/agents/logs":                 false,
	"GET /v1/agents/logs/stream":          false,
	"GET /v1/agents/logs/{id}":            false,
	"GET /v1/data/export":                 false,
	"POST /v1/data/import":                false,
	"GET /v1/data/changes":                false,
	// Reached before there is a caller to classify: the browser arrives from the identity
	// provider and the state parameter is the secret.
	"GET /v1/agents/plugins/callback": true,
}

// serverSideRoutes builds the matcher for every operation an end user's device may not
// reach, which is every operation the spec does not mark client-accessible.
//
// The spec's own path templates are the patterns, because OpenAPI writes a parameter as
// {id} and so does ServeMux: a route is registered rather than translated. Matching is
// then the same routing the generated handlers get, so an operation cannot be reached by
// a path that spells it differently.
//
// An operation declaring no security at all is skipped in both directions. It is reached
// before there is a caller to classify — the health check and the plugin redirect, where
// the browser arrives from the identity provider — so there is nobody to refuse.
func serverSideRoutes(document *huma.OpenAPI) (*http.ServeMux, error) {
	operations, err := specifiedOperations(document)
	if err != nil {
		return nil, err
	}

	routes := http.NewServeMux()
	nothing := http.HandlerFunc(func(http.ResponseWriter, *http.Request) {})
	for _, operation := range operations {
		if operation.public || operation.open {
			continue
		}
		routes.Handle(operation.method+" "+operation.path, nothing)
	}
	for route, open := range unspecifiedRoutes {
		if !open {
			routes.Handle(route, nothing)
		}
	}
	return routes, nil
}

// withServerSide refuses the generated operations only a backend may reach.
//
// It sits after withCustomer, because refusing a caller for what it is means having worked
// out what it is first. The three sockets are left out of the embedded spec by being left
// out of generation, so socketRoutes puts them back rather than leaving them to be open by
// omission.
//
// A caller that authenticated and asked for one of these gets a 403 rather than a 401: it
// has already proved who it is, so there is nothing to be learned from a specific answer
// and a caller told "unauthenticated" would go looking for a credential problem it does
// not have. One that never authenticated is left to the handler's own 401.
func (s *Server) withServerSide(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if _, matched := s.serverSide.Handler(r); matched != "" {
			if s.refuseClientSide(w, r) {
				return
			}
		}
		next.ServeHTTP(w, r)
	})
}

// refuseClientSide answers a caller that has authenticated as an end user's device and
// asked for something only a backend may have, and reports whether it did.
func (s *Server) refuseClientSide(w http.ResponseWriter, r *http.Request) bool {
	if _, known := CustomerFrom(r.Context()); !known || ServerSideFrom(r.Context()) {
		return false
	}
	s.logger.Debug("refused a client-side caller a server-side operation",
		"method", r.Method, "path", r.URL.Path)
	writeError(w, http.StatusForbidden, "this operation is server-side only: it needs "+
		auth.AuthTypeHeader+": "+auth.AuthTypeServer+" and a token carrying server: true")
	return true
}

// withCustomer lifts the authenticated principal into the request context so handlers can
// read it without each one reaching into the raw request.
//
// A request that does not authenticate is passed along without one rather than refused
// here. Every handler that needs a customer already reports a 401 when there is none, and
// the paths that legitimately have no customer — the health check, a vendor fetching a call
// plan, the hook Stream signs — are reached by somebody who has no key to present. Failing
// here instead would mean keeping a list of the exceptions in two places.
//
// It also means one 401 for every reason authentication failed. A caller that could tell an
// unknown key from a bad token could use the difference to find out which keys exist.
//
// The one failure it does answer is a caller whose level the app turns away. That caller
// proved who it is, so there is nothing to protect by staying quiet, and the advice it
// needs is the opposite of the advice a 401 gives: its credential is fine and this app
// does not take guests.
//
// The caller is recorded for a backend too, even though a backend is charged no limit. It
// is how a backend says which of its users it is opening a session for, so that the user's
// own device can reach that session afterwards; what keeps the limit off it is that
// withQuota looks at whether the caller is server-side rather than at whether there is one.
func (s *Server) withCustomer(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		principal, err := s.authenticator.Authenticate(r.Context(), r)
		if errors.Is(err, auth.ErrLevelRefused) {
			s.logger.Debug("refused a level of user this app turns away",
				"method", r.Method, "path", r.URL.Path, "kind", principal.Kind)
			writeError(w, http.StatusForbidden, "this app does not accept "+
				"requests from this level of user")
			return
		}
		if err == nil && principal.AppID != "" {
			ctx := context.WithValue(r.Context(), customerContextKey{}, principal.AppID)
			ctx = context.WithValue(ctx, organizationContextKey{}, principal.OrganizationID)
			ctx = context.WithValue(ctx, serverSideContextKey{}, principal.ServerSide)
			ctx = context.WithValue(ctx, kindContextKey{}, principal.Kind)
			ctx = context.WithValue(ctx, callerContextKey{}, routing.Caller{
				UserID: principal.UserID,
				IP:     clientIP(r, s.trusted),
			})
			r = r.WithContext(ctx)
			s.policies.Join(principal.AppID, principal.OrganizationID)
		}
		next.ServeHTTP(w, r)
	})
}

// corsRequestHeaders are the request headers a browser may send.
//
// They cover both ways a caller proves itself, because one deployment's browser is not the
// other's: reached through Stream's proxy a token arrives in Authorization with its kind
// named in Stream-Auth-Type, while a deployment running without a proxy and without keys
// names its tenant in X-Customer-Id instead. X-Stream-Client is what Stream's own clients
// tag themselves with, and arrives whether or not anything here reads it.
//
// A preflight refuses any header it was not asked about, and the browser reports that as a
// blocked request naming only the header, so a list covering one mode alone fails in a way
// that looks like the origin was never allowed.
const corsRequestHeaders = "Authorization, " + auth.AuthTypeHeader + ", " + auth.APIKeyHeader +
	", X-Stream-Client, " + CustomerHeader + ", Content-Type"

// corsMethods are the methods this API serves. PUT belongs here because a live session's
// instructions are replaced with one; PATCH does not, because the spec serves none.
const corsMethods = "GET, POST, PUT, DELETE, OPTIONS"

// withCORS lets a browser at the API from the origins the deployment named.
//
// It exists for a browser app that talks to the router directly rather than through a
// server of its own: an extra hop would only be there to move a header, and the router is
// already the thing that decides who may read a call.
func withCORS(allowed []string, next http.Handler) http.Handler {
	if len(allowed) == 0 {
		return next
	}
	permitted := make(map[string]struct{}, len(allowed))
	for _, origin := range allowed {
		permitted[strings.TrimSpace(origin)] = struct{}{}
	}
	_, anywhere := permitted["*"]

	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		origin := r.Header.Get("Origin")
		_, named := permitted[origin]
		if origin != "" && (named || anywhere) {
			w.Header().Set("Access-Control-Allow-Origin", origin)
			w.Header().Set("Vary", "Origin")
			w.Header().Set("Access-Control-Allow-Headers", corsRequestHeaders)
			w.Header().Set("Access-Control-Allow-Methods", corsMethods)
			w.Header().Set("Access-Control-Max-Age", "600")
		}
		// A preflight asks whether the real request would be allowed and carries nothing
		// worth routing, so it is answered here rather than by a handler that would only
		// report that nothing serves OPTIONS.
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		next.ServeHTTP(w, r)
	})
}

// CustomerFrom returns the customer identifier carried by the request.
func CustomerFrom(ctx context.Context) (string, bool) {
	customerID, ok := ctx.Value(customerContextKey{}).(string)
	return customerID, ok && customerID != ""
}

// OrganizationFrom returns the organization the request's customer belongs to. It is empty
// in a deployment that names a customer without naming an organization.
func OrganizationFrom(ctx context.Context) string {
	organizationID, _ := ctx.Value(organizationContextKey{}).(string)
	return organizationID
}

// ServerSideFrom reports whether the request came from a process the customer runs rather
// than from an end user's device.
func ServerSideFrom(ctx context.Context) bool {
	serverSide, _ := ctx.Value(serverSideContextKey{}).(bool)
	return serverSide
}

// CallerFrom returns the end user the request is for and where they made it from, which is
// what a daily limit is counted against. It is empty for a request nobody authenticated,
// and empty means nothing is counted. A backend acting for a named user has one and is
// still counted nothing, which withQuota decides by asking whether the caller is
// server-side rather than by asking whether there is one.
func CallerFrom(ctx context.Context) routing.Caller {
	caller, _ := ctx.Value(callerContextKey{}).(routing.Caller)
	return caller
}

// KindFrom returns what sort of caller made the request.
func KindFrom(ctx context.Context) auth.Kind {
	kind, _ := ctx.Value(kindContextKey{}).(auth.Kind)
	return kind
}

// OwnerFrom is who the request may reach sessions as: the customer, the end user behind
// it and which sort of caller that is. A backend reaches all of its customer's sessions
// whether or not it names a user; anybody else reaches only what they opened themselves,
// or what their own backend opened in their name.
func OwnerFrom(ctx context.Context) session.Owner {
	customerID, _ := CustomerFrom(ctx)
	return session.Owner{
		CustomerID: customerID,
		UserID:     CallerFrom(ctx).UserID,
		Kind:       KindFrom(ctx),
	}
}

// routerFor returns the router serving a modality, or false when this deployment does not
// serve it.
func (s *Server) routerFor(modality Modality) (routing.Inspector, bool) {
	router, ok := s.routers[routing.Modality(modality)]
	return router, ok
}

// HealthStatus is whether the router is serving, and how each dependency answered.
type HealthStatus struct {
	Status       HealthStatusStatus `json:"status" enum:"ok,degraded"`
	Dependencies map[string]string  `json:"dependencies" doc:"Dependency name to \"ok\" or a failure description." example:"{\"postgres\":\"ok\",\"redis\":\"ok\"}"`
}

// HealthStatusStatus is whether every dependency answered.
type HealthStatusStatus string

const (
	Ok       HealthStatusStatus = "ok"
	Degraded HealthStatusStatus = "degraded"
)

type healthResponse struct {
	Status int
	Body   HealthStatus
}

func (s *Server) registerHealth(api huma.API) {
	huma.Register(api, huma.Operation{
		OperationID: "getHealth",
		Method:      http.MethodGet,
		Path:        "/health",
		Summary:     "Liveness and dependency check",
		Security:    []map[string][]string{},
		Responses: map[string]*huma.Response{
			"200": {Description: "The router is serving"},
			"503": {
				Description: "A dependency is unavailable",
				Content: map[string]*huma.MediaType{
					"application/json": {Schema: &huma.Schema{Ref: "#/components/schemas/HealthStatus"}},
				},
			},
		},
	}, s.getHealth)
}

// getHealth reports whether the router and its dependencies are usable.
func (s *Server) getHealth(ctx context.Context, _ *struct{}) (*healthResponse, error) {
	dependencies := map[string]string{}
	healthy := true

	if s.store == nil {
		dependencies["postgres"] = "not configured"
	} else if err := s.store.Ping(ctx); err != nil {
		dependencies["postgres"] = err.Error()
		healthy = false
	} else {
		dependencies["postgres"] = "ok"
	}

	if s.live == nil {
		dependencies["redis"] = "not configured"
	} else if err := s.live.Ping(ctx); err != nil {
		dependencies["redis"] = err.Error()
		healthy = false
	} else {
		dependencies["redis"] = "ok"
	}

	for modality := range s.routers {
		dependencies[string(modality)] = "ok"
	}

	if !healthy {
		return &healthResponse{
			Status: http.StatusServiceUnavailable,
			Body:   HealthStatus{Status: Degraded, Dependencies: dependencies},
		}, nil
	}
	return &healthResponse{
		Status: http.StatusOK,
		Body:   HealthStatus{Status: Ok, Dependencies: dependencies},
	}, nil
}

// listProviders returns the providers configured for a modality and their live health.
func (s *Server) listProviders(ctx context.Context, request *listProvidersRequest) (*providerListResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, huma.Error404NotFound(unknownModality(request.Modality).Error)
	}

	candidates := router.Providers(ctx)
	shares := s.popularity.shares(ctx, string(request.Modality))
	providers := make([]Provider, 0, len(candidates))
	for _, candidate := range candidates {
		share := shares[candidate.Config.Name()]
		providers = append(providers, Provider{
			Provider:    candidate.Config.Provider,
			Model:       candidate.Config.Model,
			Description: &candidate.Config.Description,
			Languages:   candidate.Config.Languages,
			Realtime:    candidate.Config.Realtime,
			Tier:        tierOf(candidate.Config),
			Health:      providerHealth(candidate.Health),
			UsageShare:  &share,
			Benchmark:   providerBenchmark(candidate.Config.Benchmark),
			Price:       providerPrice(candidate.Config.Price),
		})
	}
	return &providerListResponse{Body: providers}, nil
}

// listRoutes returns the shortcuts offered as a choice and what each resolves to now.
func (s *Server) listRoutes(ctx context.Context, request *listRoutesRequest) (*routeListResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, huma.Error404NotFound(unknownModality(request.Modality).Error)
	}

	config := router.Config()
	offered := config.Offered()
	routes := make([]Route, 0, len(offered))
	for _, name := range offered {
		candidates, err := router.Resolve(ctx, name, nil)
		if err != nil {
			return nil, err
		}
		resolved := make([]Candidate, 0, len(candidates))
		for _, candidate := range candidates {
			resolved = append(resolved, Candidate{
				Provider: candidate.Config.Provider,
				Model:    candidate.Config.Model,
				Health:   providerHealth(candidate.Health),
			})
		}
		alias := config.Aliases[name]
		routes = append(routes, Route{
			Id:          name,
			Title:       alias.Title,
			Description: alias.Description,
			Candidates:  resolved,
		})
	}
	return &routeListResponse{Body: routes}, nil
}

// resolveTarget explains which providers would serve a target, best first.
func (s *Server) resolveTarget(ctx context.Context, request *resolveTargetRequest) (*candidateListResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	router, ok := s.routerFor(request.Modality)
	if !ok {
		return nil, huma.Error404NotFound(unknownModality(request.Modality).Error)
	}

	var languageHints []string
	if request.Language.Set {
		languageHints = request.Language.Value
	}

	candidates, err := router.Resolve(ctx, request.Target, languageHints)
	if err != nil {
		return nil, huma.Error404NotFound(err.Error())
	}

	resolved := make([]Candidate, 0, len(candidates))
	for _, candidate := range candidates {
		resolved = append(resolved, Candidate{
			Provider: candidate.Config.Provider,
			Model:    candidate.Config.Model,
			Health:   providerHealth(candidate.Health),
		})
	}
	return &candidateListResponse{Body: resolved}, nil
}

// getStats returns the calling customer's aggregated usage for one modality.
func (s *Server) getStats(ctx context.Context, request *getStatsRequest) (*statsBucketListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	// Statistics are not limited to the routed modalities: memory and phone are recorded
	// the same way and cost the same customer money.
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.Ptr())
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	granularity := granularityOf(request.Granularity.Ptr())
	buckets, err := s.store.CustomerStats(
		ctx, string(request.Modality), customerID, granularity, request.From, request.To, tags)
	if err != nil {
		return nil, err
	}

	stats := make([]StatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, StatsBucket{
			Provider:               bucket.Provider,
			Model:                  bucket.Model,
			Bucket:                 bucket.Bucket,
			AudioMsTotal:           bucket.AudioMsTotal,
			CharactersTotal:        bucket.CharactersTotal,
			InputTokensTotal:       bucket.InputTokensTotal,
			CachedInputTokensTotal: bucket.CachedInputTokensTotal,
			OutputTokensTotal:      bucket.OutputTokensTotal,
			ImagesTotal:            bucket.ImagesTotal,
			CostMicrosTotal:        bucket.CostMicrosTotal,
			RequestCount:           bucket.RequestCount,
			ErrorCount:             bucket.ErrorCount,
			LatencyP50Ms:           bucket.LatencyP50Ms,
			LatencyP95Ms:           bucket.LatencyP95Ms,
			Uptime:                 bucket.Uptime,
		})
	}
	return &statsBucketListResponse{Body: stats}, nil
}

// getTagStats returns the calling customer's usage broken down by one cost label.
func (s *Server) getTagStats(ctx context.Context, request *getTagStatsRequest) (*tagStatsBucketListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	granularity := granularityOf(request.Granularity.Ptr())
	buckets, err := s.store.CustomerTagStats(ctx, string(request.Modality), customerID,
		request.Key, granularity, request.From, request.To)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	stats := make([]TagStatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, TagStatsBucket{
			TagKey:                 bucket.TagKey,
			TagValue:               bucket.TagValue,
			Bucket:                 bucket.Bucket,
			AudioMsTotal:           bucket.AudioMsTotal,
			CharactersTotal:        bucket.CharactersTotal,
			InputTokensTotal:       bucket.InputTokensTotal,
			CachedInputTokensTotal: bucket.CachedInputTokensTotal,
			OutputTokensTotal:      bucket.OutputTokensTotal,
			ImagesTotal:            bucket.ImagesTotal,
			CostMicrosTotal:        bucket.CostMicrosTotal,
			RequestCount:           bucket.RequestCount,
			ErrorCount:             bucket.ErrorCount,
			LatencyP50Ms:           bucket.LatencyP50Ms,
			LatencyP95Ms:           bucket.LatencyP95Ms,
			Uptime:                 bucket.Uptime,
		})
	}
	return &tagStatsBucketListResponse{Body: stats}, nil
}

// getTurnStats returns the calling customer's conversational latency.
func (s *Server) getTurnStats(ctx context.Context, request *getTurnStatsRequest) (*turnStatsBucketListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	var agentID string
	if request.AgentID.Set {
		agentID = request.AgentID.Value
	}

	granularity := granularityOf(request.Granularity.Ptr())
	buckets, err := s.store.CustomerTurnStats(
		ctx, customerID, agentID, granularity, request.From, request.To)
	if err != nil {
		return nil, err
	}

	stats := make([]TurnStatsBucket, 0, len(buckets))
	for _, bucket := range buckets {
		stats = append(stats, TurnStatsBucket{
			AgentId:          bucket.AgentID,
			Bucket:           bucket.Bucket,
			TurnCount:        bucket.TurnCount,
			InterruptedCount: bucket.InterruptedCount,
			AudioOutMsTotal:  bucket.AudioOutMsTotal,
			SttLatencyP50Ms:  bucket.STTLatencyP50Ms,
			SttLatencyP95Ms:  bucket.STTLatencyP95Ms,
			LlmTtftP50Ms:     bucket.LLMTTFTP50Ms,
			LlmTtftP95Ms:     bucket.LLMTTFTP95Ms,
			TtsTtfbP50Ms:     bucket.TTSTTFBP50Ms,
			TtsTtfbP95Ms:     bucket.TTSTTFBP95Ms,
			RoundtripP50Ms:   bucket.RoundtripP50Ms,
			RoundtripP95Ms:   bucket.RoundtripP95Ms,
			RoundtripP99Ms:   bucket.RoundtripP99Ms,
		})
	}
	return &turnStatsBucketListResponse{Body: stats}, nil
}

// getSpend returns what the calling customer spent, grouped.
func (s *Server) getSpend(ctx context.Context, request *getSpendRequest) (*spendBucketListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.Ptr())
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	buckets, err := s.store.CustomerSpend(ctx, customerID, request.GroupBy,
		granularityOf(request.Granularity.Ptr()), request.From, request.To, request.Limit, tags)
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}

	spend := make([]SpendBucket, 0, len(buckets))
	for _, bucket := range buckets {
		spend = append(spend, SpendBucket{
			Bucket:          bucket.Bucket,
			Value:           bucket.Value,
			CostMicrosTotal: bucket.CostMicrosTotal,
			RequestCount:    bucket.RequestCount,
		})
	}
	return &spendBucketListResponse{Body: spend}, nil
}

// getTagKeys returns which cost labels the calling customer's spend carries.
func (s *Server) getTagKeys(ctx context.Context, request *getTagKeysRequest) (*tagKeySummaryListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	tags, err := parseTagFilter(request.Tag.Ptr())
	if err != nil {
		return nil, huma.Error400BadRequest(err.Error())
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	found, err := s.store.CustomerTagKeys(ctx, customerID, request.From, request.To, tags)
	if err != nil {
		return nil, err
	}

	keys := make([]TagKeySummary, 0, len(found))
	for _, key := range found {
		values := make([]TagValueSummary, 0, len(key.TopValues))
		for _, value := range key.TopValues {
			values = append(values, TagValueSummary{
				Value:           value.Value,
				CostMicrosTotal: value.CostMicrosTotal,
				RequestCount:    value.RequestCount,
				Share:           value.Share,
			})
		}
		keys = append(keys, TagKeySummary{
			Key:             key.Key,
			ValueCount:      key.ValueCount,
			CostMicrosTotal: key.CostMicrosTotal,
			RequestCount:    key.RequestCount,
			Coverage:        key.Coverage,
			TopValues:       values,
		})
	}
	return &tagKeySummaryListResponse{Body: keys}, nil
}

// getActivity returns who used the calling customer's agents, and how much.
func (s *Server) getActivity(ctx context.Context, request *getActivityRequest) (*activityBucketListResponse, error) {
	customerID, ok := CustomerFrom(ctx)
	if !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.To.After(request.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("statistics are not available: no database configured")
	}

	buckets, err := s.store.CustomerActivity(ctx, customerID,
		activityGranularityOf(request.Granularity.Ptr()), request.From, request.To)
	if err != nil {
		return nil, err
	}

	activity := make([]ActivityBucket, 0, len(buckets))
	for _, bucket := range buckets {
		activity = append(activity, ActivityBucket{
			Bucket:       bucket.Bucket,
			ActiveUsers:  bucket.ActiveUsers,
			Sessions:     bucket.Sessions,
			Messages:     bucket.Messages,
			Calls:        bucket.Calls,
			VoiceMinutes: bucket.VoiceMinutes,
			PhoneMinutes: bucket.PhoneMinutes,
		})
	}
	return &activityBucketListResponse{Body: activity}, nil
}

// runRollup aggregates request rows into a rollup table.
func (s *Server) runRollup(ctx context.Context, request *runRollupRequest) (*rollupResultResponse, error) {
	if _, ok := CustomerFrom(ctx); !ok {
		return nil, huma.Error401Unauthorized(missingCustomer().Error)
	}
	if !request.Body.To.After(request.Body.From) {
		return nil, huma.Error400BadRequest("to must be after from")
	}
	if s.store == nil {
		return nil, huma.Error400BadRequest("rollups are not available: no database configured")
	}

	granularity := granularityOf(request.Body.Granularity)
	written, err := s.store.Rollup(ctx, granularity, request.Body.From, request.Body.To)
	if err != nil {
		return nil, err
	}

	return &rollupResultResponse{Body: RollupResult{
		Granularity:    Granularity(granularity),
		BucketsWritten: written,
	}}, nil
}

// parseTagFilter turns repeated "key:value" query parameters into a label filter. A tag
// key never contains a colon, so the first one separates the two.
func parseTagFilter(raw *[]string) (map[string]string, error) {
	if raw == nil || len(*raw) == 0 {
		return nil, nil
	}

	tags := make(map[string]string, len(*raw))
	for _, entry := range *raw {
		key, value, found := strings.Cut(entry, ":")
		if !found || key == "" {
			return nil, fmt.Errorf("tag %q must be written key:value", entry)
		}
		tags[key] = value
	}
	return tags, nil
}

// granularityOf defaults to hourly, matching the spec.
func granularityOf(requested *Granularity) store.Granularity {
	if requested != nil && *requested == GranularityDaily {
		return store.Daily
	}
	return store.Hourly
}

// activityGranularityOf defaults to daily, matching the spec.
func activityGranularityOf(requested *ActivityGranularity) store.ActivityGranularity {
	if requested != nil && *requested == ActivityGranularityMonthly {
		return store.ActivityMonthly
	}
	return store.ActivityDaily
}

// tierOf reports the effective tier, which is low-latency for a model that declares none.
func tierOf(config routing.ProviderConfig) Tier {
	if config.Tier == routing.HighQuality {
		return HighQuality
	}
	return LowLatency
}

func providerHealth(health live.Health) ProviderHealth {
	return ProviderHealth{
		Available:    health.Available,
		Requests:     health.Requests,
		Errors:       health.Errors,
		ErrorRate:    health.ErrorRate(),
		LatencyMsAvg: health.LatencyMsAvg,
	}
}

// providerBenchmark leaves out what was not measured, and is nil for a model with no
// measurements at all.
func providerBenchmark(benchmark routing.Benchmark) *ProviderBenchmark {
	if benchmark == (routing.Benchmark{}) {
		return nil
	}
	measured := func(v float64) *float64 {
		if v == 0 {
			return nil
		}
		return &v
	}
	counted := func(v int) *int {
		if v == 0 {
			return nil
		}
		return &v
	}
	return &ProviderBenchmark{
		Elo:                   counted(benchmark.Elo),
		CharactersPerSecond:   measured(benchmark.CharactersPerSecond),
		WordErrorRate:         measured(benchmark.WordErrorRate),
		LatencyMs:             counted(benchmark.LatencyMs),
		SearchIndex:           counted(benchmark.SearchIndex),
		CostPerTask:           measured(benchmark.CostPerTask),
		IntelligenceIndex:     counted(benchmark.IntelligenceIndex),
		OutputTokensPerSecond: measured(benchmark.OutputTokensPerSecond),
	}
}

// providerPrice is the token rates a model is billed at, and nil for a model not billed
// by the token.
func providerPrice(price routing.Price) *ProviderPrice {
	if price.PerMillionInputTokens == 0 && price.PerMillionOutputTokens == 0 {
		return nil
	}
	return &ProviderPrice{
		PerMillionInputTokens:  &price.PerMillionInputTokens,
		PerMillionOutputTokens: &price.PerMillionOutputTokens,
	}
}

func missingCustomer() Error {
	return Error{Error: "the " + CustomerHeader + " header is required"}
}

func unknownModality(modality Modality) Error {
	return Error{Error: "this deployment does not route " + string(modality)}
}

func badRequest(message string) Error {
	return Error{Error: message}
}
