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
	StreamTrunkID string     `bun:"stream_trunk_id,nullzero"`
	PurchasedAt   time.Time  `bun:"purchased_at,notnull"`
	ReleasedAt    *time.Time `bun:"released_at"`
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
