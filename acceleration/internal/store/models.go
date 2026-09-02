package store

import (
	"time"

	"github.com/uptrace/bun"
)

// Request is one recorded unit of work, stored so billing, cost and health can all be
// derived from the same rows.
type Request struct {
	bun.BaseModel `bun:"table:requests,alias:r"`

	ID       int64  `bun:"id,pk,autoincrement"`
	Modality string `bun:"modality,notnull"`
	// CustomerID owns the request. Every statistic is keyed by it.
	CustomerID string `bun:"customer_id,notnull"`
	// AgentID is the agent the work was done for. Empty outside a conversation.
	AgentID string `bun:"agent_id,nullzero"`
	// CallID is the call the work happened in. Empty outside a conversation.
	CallID string `bun:"call_id,nullzero"`
	// Tags are the customer's own cost labels, aggregated into the tag rollups so spend
	// can be broken down by whatever the customer labelled it with.
	Tags      map[string]string `bun:"tags,type:jsonb,nullzero"`
	Provider  string            `bun:"provider,notnull"`
	Model     string            `bun:"model,notnull"`
	StartedAt time.Time         `bun:"started_at,notnull"`
	// AudioMs is billable audio, transcribed or produced.
	AudioMs int64 `bun:"audio_ms,notnull"`
	// Characters is billable text.
	Characters int64 `bun:"characters,notnull"`
	// InputTokens is the whole prompt an LLM read, cached part included.
	InputTokens int64 `bun:"input_tokens,notnull"`
	// CachedInputTokens is the part of the prompt served from the provider's cache.
	CachedInputTokens int64 `bun:"cached_input_tokens,notnull"`
	// OutputTokens is everything an LLM generated, reasoning included.
	OutputTokens int64    `bun:"output_tokens,notnull"`
	LatencyMs    *float64 `bun:"latency_ms"`
	// CostMicros is millionths of a dollar, priced from the provider's configured rates.
	CostMicros int64  `bun:"cost_micros,notnull"`
	Success    bool   `bun:"success,notnull"`
	ErrorCode  string `bun:"error_code,nullzero"`
}

// Bucket is one aggregated row from stats_hourly or stats_daily.
type Bucket struct {
	Modality               string    `bun:"modality"`
	CustomerID             string    `bun:"customer_id"`
	Provider               string    `bun:"provider"`
	Model                  string    `bun:"model"`
	Bucket                 time.Time `bun:"bucket"`
	AudioMsTotal           int64     `bun:"audio_ms_total"`
	CharactersTotal        int64     `bun:"characters_total"`
	InputTokensTotal       int64     `bun:"input_tokens_total"`
	CachedInputTokensTotal int64     `bun:"cached_input_tokens_total"`
	OutputTokensTotal      int64     `bun:"output_tokens_total"`
	CostMicrosTotal        int64     `bun:"cost_micros_total"`
	RequestCount           int64     `bun:"request_count"`
	ErrorCount             int64     `bun:"error_count"`
	LatencyP50Ms           *float64  `bun:"latency_p50_ms"`
	LatencyP95Ms           *float64  `bun:"latency_p95_ms"`
	Uptime                 *float64  `bun:"uptime"`
}

// Turn is one exchange in a conversation, measured the way the caller experienced it.
// The legs are pointers because a pipeline need not have all of them, and an interrupted
// turn stops partway through.
type Turn struct {
	bun.BaseModel `bun:"table:turns,alias:t"`

	ID         int64  `bun:"id,pk,autoincrement"`
	CustomerID string `bun:"customer_id,notnull"`
	AgentID    string `bun:"agent_id,notnull"`
	CallID     string `bun:"call_id,nullzero"`
	// TurnID is the agent's own identifier for the exchange, unique per agent.
	TurnID    string            `bun:"turn_id,notnull"`
	Tags      map[string]string `bun:"tags,type:jsonb,nullzero"`
	StartedAt time.Time         `bun:"started_at,notnull"`
	// STTLatencyMs is the provider's decode time for the transcript that settled the turn.
	STTLatencyMs *float64 `bun:"stt_latency_ms"`
	// LLMTTFTMs is the wait between asking the model and its first token.
	LLMTTFTMs *float64 `bun:"llm_ttft_ms"`
	// TTSTTFBMs is the wait between sending the first sentence and the first audio.
	TTSTTFBMs *float64 `bun:"tts_ttfb_ms"`
	// RoundtripMs is the whole delay: settled transcript to first audio published.
	RoundtripMs *float64 `bun:"roundtrip_ms"`
	// SpeechEndToAudioMs is voice in to voice out.
	SpeechEndToAudioMs *float64 `bun:"speech_end_to_audio_ms"`
	// AudioOutMs is how much speech the agent published for this turn.
	AudioOutMs *float64 `bun:"audio_out_ms"`
	// AudioDroppedMs is speech that was synthesised for this turn but never published.
	// Set on a turn that was not interrupted, it means the agent cut itself off.
	AudioDroppedMs *float64 `bun:"audio_dropped_ms"`
	Interrupted    bool     `bun:"interrupted,notnull"`
}

// CallEvent is one judgement the conversation made about how to handle a call.
//
// A Turn says what an exchange cost the caller in waiting. This says why the call went
// the way it did: why the agent waited rather than answering, why it read something as
// not meant for it, why it stopped mid-sentence. Read in order they are the reasoning
// behind a conversation, which is the only thing that explains a call that surprised
// somebody.
type CallEvent struct {
	bun.BaseModel `bun:"table:call_events,alias:ce"`

	ID         int64  `bun:"id,pk,autoincrement"`
	CustomerID string `bun:"customer_id,notnull"`
	// CallID is the session the judgement was made in.
	CallID  string    `bun:"call_id,notnull"`
	AgentID string    `bun:"agent_id,notnull"`
	At      time.Time `bun:"at,notnull"`
	// Kind is what was decided.
	Kind string `bun:"kind,notnull"`
	// Reason is why, in words.
	Reason string `bun:"reason,notnull"`
	// TurnID is the exchange it was about, so a judgement lines up with the timings of
	// the turn it produced.
	TurnID      string `bun:"turn_id,nullzero"`
	Participant string `bun:"participant,nullzero"`
	// Said is what was heard, or what the agent decided to say.
	Said string `bun:"said,nullzero"`
	// LatencyMs is what the flow controller took to rule, where anything was asked.
	LatencyMs *float64 `bun:"latency_ms"`
}

// TurnBucket is one aggregated row from turn_stats_hourly or turn_stats_daily.
type TurnBucket struct {
	CustomerID       string    `bun:"customer_id"`
	AgentID          string    `bun:"agent_id"`
	Bucket           time.Time `bun:"bucket"`
	TurnCount        int64     `bun:"turn_count"`
	InterruptedCount int64     `bun:"interrupted_count"`
	AudioOutMsTotal  float64   `bun:"audio_out_ms_total"`
	STTLatencyP50Ms  *float64  `bun:"stt_latency_p50_ms"`
	STTLatencyP95Ms  *float64  `bun:"stt_latency_p95_ms"`
	LLMTTFTP50Ms     *float64  `bun:"llm_ttft_p50_ms"`
	LLMTTFTP95Ms     *float64  `bun:"llm_ttft_p95_ms"`
	TTSTTFBP50Ms     *float64  `bun:"tts_ttfb_p50_ms"`
	TTSTTFBP95Ms     *float64  `bun:"tts_ttfb_p95_ms"`
	RoundtripP50Ms   *float64  `bun:"roundtrip_p50_ms"`
	RoundtripP95Ms   *float64  `bun:"roundtrip_p95_ms"`
	RoundtripP99Ms   *float64  `bun:"roundtrip_p99_ms"`
}

// TagBucket is one aggregated row from stats_tags_hourly or stats_tags_daily: what the
// requests carrying one label cost in one time bucket.
type TagBucket struct {
	Modality               string    `bun:"modality"`
	CustomerID             string    `bun:"customer_id"`
	TagKey                 string    `bun:"tag_key"`
	TagValue               string    `bun:"tag_value"`
	Bucket                 time.Time `bun:"bucket"`
	AudioMsTotal           int64     `bun:"audio_ms_total"`
	CharactersTotal        int64     `bun:"characters_total"`
	InputTokensTotal       int64     `bun:"input_tokens_total"`
	CachedInputTokensTotal int64     `bun:"cached_input_tokens_total"`
	OutputTokensTotal      int64     `bun:"output_tokens_total"`
	CostMicrosTotal        int64     `bun:"cost_micros_total"`
	RequestCount           int64     `bun:"request_count"`
	ErrorCount             int64     `bun:"error_count"`
	LatencyP50Ms           *float64  `bun:"latency_p50_ms"`
	LatencyP95Ms           *float64  `bun:"latency_p95_ms"`
	Uptime                 *float64  `bun:"uptime"`
}

// PhoneNumber is a number this service holds on a customer's behalf.
//
// A released number keeps its row: what it cost while it was held is still part of that
// month's bill after it is gone.
type PhoneNumber struct {
	bun.BaseModel `bun:"table:phone_numbers,alias:pn"`

	ID   int64  `bun:"id,pk,autoincrement"`
	E164 string `bun:"e164,notnull"`
	// Vendor is who it was bought from.
	Vendor  string `bun:"vendor,notnull"`
	Country string `bun:"country,notnull"`
	// Capabilities is what the number can carry: voice, sms, mms, fax.
	Capabilities []string `bun:"capabilities,array"`
	// MonthlyCostMicros is millionths of a dollar per month, charged whether or not the
	// number is used.
	MonthlyCostMicros int64             `bun:"monthly_cost_micros,notnull"`
	CustomerID        string            `bun:"customer_id,notnull"`
	Tags              map[string]string `bun:"tags,type:jsonb,nullzero"`
	// VendorID is the vendor's own identifier, needed to release or reconfigure it.
	VendorID string `bun:"vendor_id,nullzero"`
	// StreamTrunkID is the SIP trunk calls to this number arrive on, empty until it has
	// been attached to one.
	StreamTrunkID string `bun:"stream_trunk_id,nullzero"`
	// StreamCallID and StreamCallType are the Stream call the routing rule puts callers
	// in. They are what an arriving call is recognised by, since a webhook names the call
	// rather than the number.
	StreamCallID   string     `bun:"stream_call_id,nullzero"`
	StreamCallType string     `bun:"stream_call_type,nullzero"`
	PurchasedAt    time.Time  `bun:"purchased_at,notnull"`
	ReleasedAt     *time.Time `bun:"released_at"`
}

// CallBridge is what a vendor is told to do when the person it called picks up.
//
// Three vendors take a URL rather than a call plan when a call is placed, and fetch it on
// answer. This is the answer, parked between the two moments. The token is the whole of the
// fetch's authentication, so a bridge is read once and then gone.
type CallBridge struct {
	bun.BaseModel `bun:"table:call_bridges,alias:cb"`

	// Token is what the vendor puts in the url it fetches, and the only thing proving the
	// fetch is one this service asked for.
	Token      string `bun:"token,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Vendor     string `bun:"vendor,notnull"`
	// TrunkURI is the SIP address the answered leg is transferred to.
	TrunkURI string `bun:"trunk_uri,notnull"`
	// TrunkUsername and TrunkPassword are the trunk's digest credentials, set only for a
	// vendor that can send them.
	TrunkUsername string `bun:"trunk_username,nullzero"`
	TrunkPassword string `bun:"trunk_password,nullzero"`
	// InitialDigits are pressed at the person before the transfer.
	InitialDigits string `bun:"initial_digits,nullzero"`
	// CallID is the Stream call the leg is routed into, kept for the audit trail.
	CallID    string    `bun:"call_id,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	ExpiresAt time.Time `bun:"expires_at,notnull"`
}

// AgentConfig is a named set of the decisions a session is created with.
//
// It holds only what a caller would otherwise repeat on every call. Everything that is
// about one conversation rather than the agent behind it, the call id above all, stays in
// the create request: a config is who the agent is, not which call it is on.
type AgentConfig struct {
	bun.BaseModel `bun:"table:agent_configs,alias:ac"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
	// Mode is whether the agent is spoken to or written to: AgentModeVoice or
	// AgentModeText. A text agent uses neither speech target and joins no call.
	Mode string `bun:"mode,notnull"`
	// STT, TTS, LLM, Subagent and Search are routing targets. Empty leaves the session
	// default.
	STT          string `bun:"stt,notnull"`
	TTS          string `bun:"tts,notnull"`
	Voice        string `bun:"voice,notnull"`
	LLM          string `bun:"llm,notnull"`
	Subagent     string `bun:"subagent,notnull"`
	Search       string `bun:"search,notnull"`
	Instructions string `bun:"instructions,notnull"`
	Greeting     string `bun:"greeting,notnull"`
	// Skills names entries in the skill registry rather than carrying their instructions,
	// so editing a skill changes every config that uses it.
	Skills []string `bun:"skills,type:jsonb"`
	// Plugins names hosted MCP servers this agent is allowed to reach, from the built-in
	// catalog. A name here without a connected row is a plugin that was attached and then
	// the login expired or was revoked.
	Plugins []string `bun:"plugins,type:jsonb"`
	// Keyterms are the business-specific words a transcriber would otherwise get wrong.
	Keyterms []string `bun:"keyterms,type:jsonb"`
	// KnowledgeNamespace is what the agent may look things up in.
	KnowledgeNamespace string            `bun:"knowledge_namespace,notnull"`
	Tags               map[string]string `bun:"tags,type:jsonb"`
	// SyncHash is a fingerprint of the last directory written onto this config. Empty
	// if it was never synced from a directory.
	SyncHash  string     `bun:"sync_hash,notnull"`
	CreatedAt time.Time  `bun:"created_at,notnull"`
	UpdatedAt time.Time  `bun:"updated_at,notnull"`
	DeletedAt *time.Time `bun:"deleted_at"`
}

// How an agent is talked to.
const (
	// AgentModeVoice joins a call, transcribes what it hears and speaks its replies.
	AgentModeVoice = "voice"
	// AgentModeText holds the same conversation in writing, using neither speech target.
	AgentModeText = "text"
)

// Skill is one kind of work worth handing to the slower model, stored so the config that
// owns it can name it.
//
// A skill belongs to one config rather than to the customer: two agents that both need
// the same kind of work have one each, so editing either leaves the other alone.
type Skill struct {
	bun.BaseModel `bun:"table:skills,alias:sk"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// ConfigID is the agent config this skill belongs to.
	ConfigID string `bun:"config_id,notnull"`
	Name     string `bun:"name,notnull"`
	// Description is the one line the fast model sees.
	Description string `bun:"description,notnull"`
	// Instructions is the full prompt, which only the subagent sees.
	Instructions string `bun:"instructions,notnull"`
	// DeadlineMs is how long the work may run before it is abandoned. Zero is the
	// harness's own default.
	DeadlineMs int64      `bun:"deadline_ms,notnull"`
	CreatedAt  time.Time  `bun:"created_at,notnull"`
	UpdatedAt  time.Time  `bun:"updated_at,notnull"`
	DeletedAt  *time.Time `bun:"deleted_at"`
}

// How far a plugin login has got.
const (
	// PluginPending means the browser is still at the provider.
	PluginPending = "pending"
	// PluginConnected means tokens are stored and a session may use them.
	PluginConnected = "connected"
	// PluginFailed means the exchange did not work.
	PluginFailed = "failed"
)

// PluginConnection is one hosted MCP server authorized for one agent config.
type PluginConnection struct {
	bun.BaseModel `bun:"table:agent_plugin_connections,alias:apc"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	ConfigID   string `bun:"config_id,notnull"`
	PluginID   string `bun:"plugin_id,notnull"`
	// InstanceURL is the shop or org hostname for plugins that have no single global URL.
	InstanceURL  string     `bun:"instance_url,notnull"`
	AccessToken  string     `bun:"access_token,notnull"`
	RefreshToken string     `bun:"refresh_token,notnull"`
	ExpiresAt    *time.Time `bun:"expires_at"`
	Status       string     `bun:"status,notnull"`
	// OAuthState, CodeVerifier, ClientID and TokenEndpoint are what the callback needs
	// to finish the login. They are cleared once the connection is connected.
	OAuthState    string     `bun:"oauth_state,notnull"`
	CodeVerifier  string     `bun:"code_verifier,notnull"`
	ClientID      string     `bun:"client_id,notnull"`
	TokenEndpoint string     `bun:"token_endpoint,notnull"`
	CreatedAt     time.Time  `bun:"created_at,notnull"`
	UpdatedAt     time.Time  `bun:"updated_at,notnull"`
	DeletedAt     *time.Time `bun:"deleted_at"`
}

// Where a knowledge url has got to.
const (
	// KnowledgeURLPending means it has been added but not yet read.
	KnowledgeURLPending = "pending"
	// KnowledgeURLIndexed means the page is in the knowledge base and can be looked up.
	KnowledgeURLIndexed = "indexed"
	// KnowledgeURLFailed means it could not be read, and Error says why.
	KnowledgeURLFailed = "failed"
)

// KnowledgeURL is a page an agent's knowledge base is kept filled from.
//
// The passages are in turbopuffer rather than here; this is the subscription to them. It
// carries how many were written because that is what makes removing them exact: the ids
// are the url and a position, so knowing the count is knowing which ids to delete.
type KnowledgeURL struct {
	bun.BaseModel `bun:"table:knowledge_urls,alias:ku"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// Namespace is the knowledge base the page is read into.
	Namespace string `bun:"namespace,notnull"`
	URL       string `bun:"url,notnull"`
	// Title is what the page called itself, as of the last time it was read.
	Title string `bun:"title,notnull"`
	// State is pending, indexed or failed.
	State string `bun:"state,notnull"`
	// Error is why the last read failed, and is empty otherwise.
	Error string `bun:"error,notnull"`
	// Passages is how many the page was last cut into.
	Passages int `bun:"passages,notnull"`
	// LastIndexedAt is when it was last read successfully. Nil means never.
	LastIndexedAt *time.Time `bun:"last_indexed_at"`
	CreatedAt     time.Time  `bun:"created_at,notnull"`
	UpdatedAt     time.Time  `bun:"updated_at,notnull"`
	DeletedAt     *time.Time `bun:"deleted_at"`
}

// What a provider has made of a voice's samples.
const (
	// VoicePending means preparation has been asked for but has not finished.
	VoicePending = "pending"
	// VoiceReady means the provider has the voice and will speak in it.
	VoiceReady = "ready"
	// VoiceFailed means the provider would not take it, and Error says why.
	VoiceFailed = "failed"
)

// Voice is a voice a customer brought with them, as opposed to one from a provider's own
// library.
//
// What a session asks for is this row rather than a provider's id for it, because the
// router fails over between providers mid-call and an id one provider knows means nothing
// to the next. Which id it means is worked out once the provider is chosen.
type Voice struct {
	bun.BaseModel `bun:"table:voices,alias:v"`

	ID          string     `bun:"id,pk"`
	CustomerID  string     `bun:"customer_id,notnull"`
	Name        string     `bun:"name,notnull"`
	Description string     `bun:"description,notnull"`
	CreatedAt   time.Time  `bun:"created_at,notnull"`
	UpdatedAt   time.Time  `bun:"updated_at,notnull"`
	DeletedAt   *time.Time `bun:"deleted_at"`
}

// VoiceSample is one recording of the voice. The audio is in object storage, so the row
// says where it went rather than holding it.
type VoiceSample struct {
	bun.BaseModel `bun:"table:voice_samples,alias:vs"`

	ID      string `bun:"id,pk"`
	VoiceID string `bun:"voice_id,notnull"`
	// ObjectKey locates the audio in the bucket the deployment was given.
	ObjectKey   string `bun:"object_key,notnull"`
	ContentType string `bun:"content_type,notnull"`
	Bytes       int64  `bun:"bytes,notnull"`
	// Transcript is what is said in the recording, which the providers that ask for one
	// clone more faithfully with.
	Transcript string    `bun:"transcript,notnull"`
	CreatedAt  time.Time `bun:"created_at,notnull"`
}

// VoiceBinding is what one provider made of a voice's samples.
type VoiceBinding struct {
	bun.BaseModel `bun:"table:voice_bindings,alias:vb"`

	ID       string `bun:"id,pk"`
	VoiceID  string `bun:"voice_id,notnull"`
	Provider string `bun:"provider,notnull"`
	// ExternalID is what the provider calls this voice, and what a session asks it for.
	ExternalID string `bun:"external_id,notnull"`
	// State is one of VoicePending, VoiceReady or VoiceFailed.
	State     string    `bun:"state,notnull"`
	Error     string    `bun:"error,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
	UpdatedAt time.Time `bun:"updated_at,notnull"`
}

// Which way a call went.
const (
	// Inbound is somebody ringing the agent.
	Inbound = "inbound"
	// Outbound is the agent ringing somebody, which is what a campaign does.
	Outbound = "outbound"
)

// Call is one conversation the service ran, kept so it can be found after the process that
// held it is gone.
//
// What was said is not here: the transcript is in Stream Chat, keyed by AgentID, and the
// timings are in Turn. This row is what ties them together and carries the judgements made
// after the call ended.
type Call struct {
	bun.BaseModel `bun:"table:calls,alias:c"`

	// ID is the session id, which is the handle the caller already holds the call by.
	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	// CallID is the Stream call, and AgentID is the transcript channel.
	CallID  string `bun:"call_id,notnull"`
	AgentID string `bun:"agent_id,notnull"`
	// ConfigID names the agent config the call ran under. Empty when the caller spelled
	// the whole spec out.
	ConfigID   string `bun:"config_id,nullzero"`
	CampaignID string `bun:"campaign_id,nullzero"`
	ContactID  string `bun:"contact_id,nullzero"`
	FromNumber string `bun:"from_number,nullzero"`
	ToNumber   string `bun:"to_number,nullzero"`
	// Direction is inbound or outbound: who rang whom.
	Direction string    `bun:"direction,notnull"`
	StartedAt time.Time `bun:"started_at,notnull"`
	// EndedAt is nil while the call is still running.
	EndedAt *time.Time `bun:"ended_at"`
	// STT, TTS, LLM and Subagent are the targets the call ran with, after a session's
	// overrides were folded into whatever config it named. They are what was asked for
	// rather than what each turn resolved to: a shortcut is several models and routing
	// fails over between them, so per-turn providers live in requests.
	STT      string `bun:"stt,nullzero"`
	TTS      string `bun:"tts,nullzero"`
	LLM      string `bun:"llm,nullzero"`
	Subagent string `bun:"subagent,nullzero"`
	// Instructions is what the agent was told to be on this call.
	Instructions string `bun:"instructions,nullzero"`
	// Skills names what the fast model could hand to the subagent. The instructions
	// behind each name live in the skill registry.
	Skills []string `bun:"skills,type:jsonb,nullzero"`
	// Summary, ReviewScore and ReviewNotes are written after the call by a short model
	// pass over the transcript, so they are empty until it has run.
	Summary     string            `bun:"summary,nullzero"`
	ReviewScore *int              `bun:"review_score"`
	ReviewNotes string            `bun:"review_notes,nullzero"`
	Tags        map[string]string `bun:"tags,type:jsonb,nullzero"`
}

// What a campaign is doing.
const (
	// Draft is a campaign that has never been started.
	Draft = "draft"
	// Running means the runner is working through its contacts.
	Running = "running"
	// Paused means it was stopped partway and can be started again.
	Paused = "paused"
	// Finished means there is nobody left to ring.
	Finished = "finished"
)

// What became of one contact.
const (
	// Pending is somebody who has not been rung yet.
	Pending = "pending"
	// Calling is a call happening now.
	Calling = "calling"
	// Done is somebody who was rung, whatever they said.
	Done = "done"
	// Failed is a call that could not be made.
	Failed = "failed"
)

// Campaign is a list of people to ring and one agent to ring them with.
type Campaign struct {
	bun.BaseModel `bun:"table:campaigns,alias:cp"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
	// ConfigID is the agent that makes the calls.
	ConfigID string `bun:"config_id,notnull"`
	// FromNumber is one of the customer's own, which is what the person sees.
	FromNumber string `bun:"from_number,notnull"`
	// Concurrency is how many of these calls may be happening at once.
	Concurrency int               `bun:"concurrency,notnull"`
	State       string            `bun:"state,notnull"`
	Tags        map[string]string `bun:"tags,type:jsonb,nullzero"`
	CreatedAt   time.Time         `bun:"created_at,notnull"`
	StartedAt   *time.Time        `bun:"started_at"`
	FinishedAt  *time.Time        `bun:"finished_at"`
}

// Contact is one person to ring, and what became of ringing them.
type Contact struct {
	bun.BaseModel `bun:"table:campaign_contacts,alias:cc"`

	ID string `bun:"id,pk"`
	// Seq is the order they are rung in, which is the order they were added.
	Seq        int64  `bun:"seq,autoincrement"`
	CampaignID string `bun:"campaign_id,notnull"`
	ToNumber   string `bun:"to_number,notnull"`
	// Instructions are what to say to this person, added to whatever the config says.
	Instructions string `bun:"instructions,notnull"`
	State        string `bun:"state,notnull"`
	Attempts     int    `bun:"attempts,notnull"`
	// CallID is the calls row this contact became.
	CallID       string    `bun:"call_id,nullzero"`
	VendorCallID string    `bun:"vendor_call_id,nullzero"`
	Error        string    `bun:"error,nullzero"`
	CreatedAt    time.Time `bun:"created_at,notnull"`
}

// Which pipeline a simulation puts the agent through.
const (
	// SimulationText hands the agent the words, which tests everything between hearing
	// and answering without needing a voice.
	SimulationText = "text"
	// SimulationAudio generates speech and runs the whole pipeline, so what the caller
	// hears is what the agent actually said.
	SimulationAudio = "audio"
)

// How a run or one of its conversations ended.
const (
	// SimulationRunning means the conversations are still being had.
	SimulationRunning = "running"
	// SimulationPassed means the judge was satisfied. A run passed only if every one of
	// its cases did.
	SimulationPassed = "passed"
	// SimulationFailed means the judge was not.
	SimulationFailed = "failed"
	// SimulationErrored means it never got as far as a ruling.
	SimulationErrored = "errored"
	// SimulationCancelled means somebody stopped it, or the process did.
	SimulationCancelled = "cancelled"
)

// Why one conversation stopped.
const (
	// EndedComplete is the caller deciding it had asked everything it came to ask.
	EndedComplete = "complete"
	// EndedTurns is the conversation running out of turns.
	EndedTurns = "turns"
	// EndedTimeout is it running out of time.
	EndedTimeout = "timeout"
	// EndedFailed is the agent stopping answering.
	EndedFailed = "failed"
)

// Simulation is a conversation to have with an agent and something that has to be true at
// the end of it.
type Simulation struct {
	bun.BaseModel `bun:"table:simulations,alias:sm"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
	// Mode is SimulationText or SimulationAudio.
	Mode string `bun:"mode,notnull"`
	// ConfigID is the agent being tested.
	ConfigID string `bun:"config_id,notnull"`
	// Scenario is what to ask, in the customer's own words. It is a brief for the caller
	// rather than a script.
	Scenario string `bun:"scenario,notnull"`
	// Assertion is what has to be true at the end for the run to have passed.
	Assertion string `bun:"assertion,notnull"`
	// Variations is how many ways of asking the same thing one run tries.
	Variations int `bun:"variations,notnull"`
	// JudgeTarget and CallerTarget are routing targets like any other. Empty takes the
	// default.
	JudgeTarget  string `bun:"judge_target,notnull"`
	CallerTarget string `bun:"caller_target,notnull"`
	// CallerTTS, CallerSTT and CallerVoice are how the caller speaks and listens in an
	// audio simulation, and mean nothing in a text one.
	CallerTTS   string `bun:"caller_tts,notnull"`
	CallerSTT   string `bun:"caller_stt,notnull"`
	CallerVoice string `bun:"caller_voice,notnull"`
	// MaxTurns bounds one conversation.
	MaxTurns  int               `bun:"max_turns,notnull"`
	Tags      map[string]string `bun:"tags,type:jsonb,nullzero"`
	CreatedAt time.Time         `bun:"created_at,notnull"`
	UpdatedAt time.Time         `bun:"updated_at,notnull"`
	DeletedAt *time.Time        `bun:"deleted_at"`
}

// SimulationRun is one press of Run, and the parent of however many conversations the
// variations asked for.
//
// What was run is copied onto the row rather than referenced, because editing a simulation
// must not rewrite what an old run tested.
type SimulationRun struct {
	bun.BaseModel `bun:"table:simulation_runs,alias:smr"`

	ID           string `bun:"id,pk"`
	CustomerID   string `bun:"customer_id,notnull"`
	SimulationID string `bun:"simulation_id,notnull"`
	State        string `bun:"state,notnull"`
	// Cases, Passed and Failed are the tally, so the log can list a run without reading
	// its conversations.
	Cases       int        `bun:"cases,notnull"`
	Passed      int        `bun:"passed,notnull"`
	Failed      int        `bun:"failed,notnull"`
	Mode        string     `bun:"mode,notnull"`
	ConfigID    string     `bun:"config_id,notnull"`
	Scenario    string     `bun:"scenario,notnull"`
	Assertion   string     `bun:"assertion,notnull"`
	JudgeTarget string     `bun:"judge_target,notnull"`
	Error       string     `bun:"error,nullzero"`
	StartedAt   time.Time  `bun:"started_at,notnull"`
	FinishedAt  *time.Time `bun:"finished_at"`
}

// SimulationCase is one conversation. With variations off a run has one of these; with
// them expanded, ten, each asking the same thing a different way.
type SimulationCase struct {
	bun.BaseModel `bun:"table:simulation_cases,alias:smc"`

	ID    string `bun:"id,pk"`
	RunID string `bun:"run_id,notnull"`
	// Variation is which way of asking this was, and the order they are listed in.
	Variation int `bun:"variation,notnull"`
	// Scenario is the wording this case used.
	Scenario string `bun:"scenario,notnull"`
	State    string `bun:"state,notnull"`
	// CallID is the session that held the conversation, written as soon as it exists so a
	// run in progress can be watched.
	CallID string `bun:"call_id,nullzero"`
	// Transcript is what was said, oldest first.
	Transcript []SimulationLine `bun:"transcript,type:jsonb"`
	Turns      int              `bun:"turns,notnull"`
	// Passed, Verdict and Score are the judge's ruling. A nil Passed means it never got to
	// rule, which is not the same as having ruled against.
	Passed  *bool  `bun:"passed"`
	Verdict string `bun:"verdict,nullzero"`
	Score   *int   `bun:"score"`
	// Ended is why the conversation stopped.
	Ended      string     `bun:"ended,notnull"`
	Error      string     `bun:"error,nullzero"`
	StartedAt  time.Time  `bun:"started_at,notnull"`
	FinishedAt *time.Time `bun:"finished_at"`
}

// SimulationLine is one thing said, in the order it was said.
type SimulationLine struct {
	// Caller is true when the simulated caller said it rather than the agent.
	Caller bool   `json:"caller"`
	Text   string `json:"text"`
	// Intended is what the agent meant to say, where that differs from what the caller
	// heard. Only an audio simulation has both, and the difference is the whole point of
	// running one.
	Intended string    `json:"intended,omitempty"`
	At       time.Time `json:"at"`
}

// SimulationRunFilter narrows which runs are listed. Every field is optional, and an empty
// filter is the customer's most recent runs.
type SimulationRunFilter struct {
	CustomerID   string
	SimulationID string
	State        string
	// Limit caps how many come back. Zero leaves the store's own default.
	Limit int
}

// CallFilter narrows which calls are listed. Every field is optional, and an empty filter
// is the customer's most recent calls.
type CallFilter struct {
	AgentID    string
	CampaignID string
	// Running limits the list to calls that have not ended.
	Running bool
	// From and To bound when the call started. A zero time is unbounded.
	From time.Time
	To   time.Time
	// Limit caps how many come back. Zero leaves the store's own default.
	Limit int
}

// Granularity selects which rollup table to read or write.
type Granularity string

const (
	Hourly Granularity = "hourly"
	Daily  Granularity = "daily"
)

// table returns the rollup table for the granularity.
func (g Granularity) table() string {
	if g == Daily {
		return "stats_daily"
	}
	return "stats_hourly"
}

// tagTable returns the tag rollup table for the granularity.
func (g Granularity) tagTable() string {
	if g == Daily {
		return "stats_tags_daily"
	}
	return "stats_tags_hourly"
}

// turnTable returns the turn rollup table for the granularity.
func (g Granularity) turnTable() string {
	if g == Daily {
		return "turn_stats_daily"
	}
	return "turn_stats_hourly"
}

// truncateUnit returns the Postgres date_trunc unit for the granularity.
func (g Granularity) truncateUnit() string {
	if g == Daily {
		return "day"
	}
	return "hour"
}

// Valid reports whether the granularity is one this store knows.
func (g Granularity) Valid() bool {
	return g == Hourly || g == Daily
}

// Organization owns apps. It exists so a bill and a rate limit have something to name that
// outlives any one app.
type Organization struct {
	bun.BaseModel `bun:"table:organizations,alias:o"`

	ID        string    `bun:"id,pk"`
	Name      string    `bun:"name,notnull"`
	CreatedAt time.Time `bun:"created_at,notnull"`
}

// App is the tenant. Its id is what every other table's customer_id holds.
type App struct {
	bun.BaseModel `bun:"table:apps,alias:ap"`

	ID             string    `bun:"id,pk"`
	OrganizationID string    `bun:"organization_id,notnull"`
	Name           string    `bun:"name,notnull"`
	CreatedAt      time.Time `bun:"created_at,notnull"`
}

// APIKey is one credential belonging to an app. Several may be live at once, because
// rotation without downtime is: create the new key, deploy it, watch the old one go quiet,
// revoke it. One key per app makes every rotation an outage, so rotation stops happening.
type APIKey struct {
	bun.BaseModel `bun:"table:api_keys,alias:k"`

	ID     string `bun:"id,pk"`
	AppID  string `bun:"app_id,notnull"`
	Name   string `bun:"name,notnull"`
	Env    string `bun:"environment,notnull"`
	Sealed []byte `bun:"secret_sealed,notnull"`
	// KEKVersion names which key encryption key sealed this row.
	KEKVersion int        `bun:"kek_version,notnull"`
	Last4      string     `bun:"last4,notnull"`
	CreatedAt  time.Time  `bun:"created_at,notnull"`
	CreatedBy  string     `bun:"created_by,notnull"`
	ExpiresAt  *time.Time `bun:"expires_at"`
	LastUsedAt *time.Time `bun:"last_used_at"`
	RevokedAt  *time.Time `bun:"revoked_at"`
	RevokedBy  string     `bun:"revoked_by,notnull"`
}
