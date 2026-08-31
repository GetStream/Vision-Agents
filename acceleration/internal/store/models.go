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
	AudioOutMs  *float64 `bun:"audio_out_ms"`
	Interrupted bool     `bun:"interrupted,notnull"`
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
	// STT, TTS, LLM and Subagent are routing targets. Empty leaves the session default.
	STT          string `bun:"stt,notnull"`
	TTS          string `bun:"tts,notnull"`
	Voice        string `bun:"voice,notnull"`
	LLM          string `bun:"llm,notnull"`
	Subagent     string `bun:"subagent,notnull"`
	Instructions string `bun:"instructions,notnull"`
	Greeting     string `bun:"greeting,notnull"`
	// Skills names entries in the skill registry rather than carrying their instructions,
	// so editing a skill changes every config that uses it.
	Skills []string `bun:"skills,type:jsonb"`
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

// Skill is one kind of work worth handing to the slower model, stored so a config can name
// it and several configs can share it.
type Skill struct {
	bun.BaseModel `bun:"table:skills,alias:sk"`

	ID         string `bun:"id,pk"`
	CustomerID string `bun:"customer_id,notnull"`
	Name       string `bun:"name,notnull"`
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
