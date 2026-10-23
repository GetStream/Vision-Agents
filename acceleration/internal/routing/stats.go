package routing

import (
	"context"
	"log/slog"
	"maps"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// statQueueSize bounds how far the writer may fall behind before stats are dropped.
const statQueueSize = 512

// statWriteTimeout bounds a single write so a stuck backend cannot wedge the writer.
const statWriteTimeout = 5 * time.Second

// unnamedTierEvery is how often a service tier the price list does not name is logged for
// one model, so a busy one is not logged on every row.
const unnamedTierEvery = 10 * time.Minute

// Owner is who a unit of work belongs to and how the customer wants it labelled. It
// travels with a session so every row the session produces is attributed the same way.
type Owner struct {
	// CustomerID owns the request. Every statistic is keyed by it.
	CustomerID string
	// Caller is the end user the work was done for, when it was asked for by a device
	// rather than by the customer's own backend. It is what daily limits are counted
	// against, and it is not recorded: a stat row is what the customer is billed for, and
	// which of their users asked is theirs to know rather than ours to keep.
	Caller Caller
	// AgentID is the agent the work was done for. Empty outside a conversation.
	AgentID string
	// CallID is the call the work happened in. Empty outside a conversation.
	CallID string
	// Tags are the customer's own cost labels, carried onto every row so spend can be
	// broken down by whatever the customer cares about.
	Tags Tags
}

// Stat is one recorded unit of work. What a unit is depends on the modality: a completed
// turn for speech-to-text, a completed synthesis for text-to-speech, a completed
// completion for an LLM, plus rows for sessions that never got off the ground.
//
// The provider, model, modality and cost are filled in by the Recorder, so a caller
// cannot record a row that disagrees with the routing decision it came from, or with the
// price.
type Stat struct {
	Owner
	StartedAt   time.Time
	OperationID string
	Purpose     string
	TurnID      string
	DurationMs  float64
	Usage
	// InputParts is what an LLM's prompt was made of. Zero for every other modality.
	InputParts store.InputParts
	// Meters are quantities Usage has no field for, priced like the rest and stored with
	// the row.
	Meters pricing.Quantities
	// Attributes are what can choose the price, such as the service tier the vendor served
	// at, or explain the row. They are stored with it.
	Attributes map[string]string
	// LatencyMs is how long the customer waited for the work to be useful.
	LatencyMs float64
	// CostMicros overrides the priced amount for work that is not billed by the units in
	// Usage, such as a phone number's monthly charge, where the vendor quotes the price
	// outright rather than a rate to multiply. Zero leaves the pricing to the rates.
	CostMicros   int64
	Success      bool
	ErrorCode    string
	ErrorMessage string
	// Fatal marks a failure that ended the session for a reason the provider gives every
	// caller, such as an account it refuses. It makes the provider unavailable for one
	// health window, however much it served before.
	Fatal bool
}

// ErrorCancelled is the ErrorCode of work its caller gave up on, whether before the provider
// answered or by closing the stream it had opened: a reply superseded by newer words, the
// slower of two hedged requests, or one cut off by the caller speaking. It is not successful,
// but the provider did nothing wrong, so it is kept out of its health. What it generated
// before it was cut off is still billed.
const ErrorCancelled = store.ErrorCancelled

// Recorder writes stats to Postgres and Redis off the request path. A conversation must
// never wait on a database, so recording is asynchronous and stats are the thing that
// gets dropped when the backend cannot keep up.
type Recorder struct {
	modality Modality
	store    *store.Store
	live     *live.Client
	pricer   pricing.Pricer
	logger   *slog.Logger
	// gate is where the tags a customer's policies lay over every row come from. Nil lays
	// none.
	gate Gate

	queue chan queued
	done  chan struct{}

	// mu keeps a Record from sending on the queue once Close has closed it.
	mu        sync.Mutex
	closed    bool
	closeOnce sync.Once
	// dropped counts stats lost to a full queue or to a Record after Close, reported once on
	// shutdown.
	dropped atomic.Int64
	// tiersWarned is when each service tier the price list does not name was last logged,
	// by provider, model and tier.
	tiersWarned sync.Map
}

// NewRecorder starts the background writer. The pricer prices every row; nil prices
// nothing, and logs every row as billable work with no price.
func NewRecorder(modality Modality, pgStore *store.Store, liveClient *live.Client, pricer pricing.Pricer, logger *slog.Logger) *Recorder {
	return newRecorder(modality, pgStore, liveClient, nil, pricer, logger)
}

func newRecorder(modality Modality, pgStore *store.Store, liveClient *live.Client, gate Gate, pricer pricing.Pricer, logger *slog.Logger) *Recorder {
	if pricer == nil {
		// An empty list cannot be refused.
		pricer, _ = pricing.NewList("none", nil, logger)
	}
	r := &Recorder{
		modality: modality,
		store:    pgStore,
		live:     liveClient,
		pricer:   pricer,
		logger:   logger,
		gate:     gate,
		queue:    make(chan queued, statQueueSize),
		done:     make(chan struct{}),
	}
	go r.run()
	return r
}

// Quote is what that usage of the provider costs, priced as Record prices a row.
func (r *Recorder) Quote(config ProviderConfig, usage Usage) pricing.Quote {
	return r.QuoteStat(config, Stat{Usage: usage})
}

// QuoteStat is what the stat's row costs, its meters and attributes included, priced as
// Record prices it.
func (r *Recorder) QuoteStat(config ProviderConfig, entry Stat) pricing.Quote {
	return quote(r.pricer, r.modality, config, entry.Attributes, row(r.modality, config, entry).Quantities())
}

// Record queues a stat, pricing what its row holds and dropping it if the writer is too
// far behind.
func (r *Recorder) Record(config ProviderConfig, entry Stat) {
	if r.store == nil && r.live == nil {
		return
	}

	r.warnUnnamedTier(config, entry.Attributes["service_tier"])
	request := row(r.modality, config, entry)
	priced := quote(r.pricer, r.modality, config, entry.Attributes, request.Quantities())
	request.CostMicros, request.OutputCostMicros, request.Billable = priced.CostMicros, priced.OutputCostMicros, priced.Billable
	if entry.CostMicros != 0 {
		// Quoted outright by the vendor, such as a number's monthly charge, which no rate
		// prices: what it charged is spend. Generated tokens keep their share of it.
		request.CostMicros = entry.CostMicros
		request.OutputCostMicros = min(priced.OutputCostMicros, entry.CostMicros)
		request.Billable = priced.Billable || entry.CostMicros > 0
	}

	// A caller still streaming at shutdown records after Close; that stat is dropped.
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.closed {
		r.dropped.Add(1)
		return
	}
	select {
	case r.queue <- queued{request: request, fatal: entry.Fatal}:
	default:
		r.dropped.Add(1)
	}
}

// warnUnnamedTier logs a service tier the vendor served at that the provider's price does
// not name, which is priced at the default rates: how a vendor renaming a tier shows up.
// Rows are recorded from many goroutines, so the time is swapped in rather than read and
// then written.
func (r *Recorder) warnUnnamedTier(config ProviderConfig, tier string) {
	if tier == "" || tier == "default" {
		return
	}
	if _, named := config.Price.ServiceTiers[tier]; named {
		return
	}
	key := [3]string{config.Provider, config.Model, tier}
	now := time.Now()
	for {
		last, seen := r.tiersWarned.LoadOrStore(key, now)
		if !seen {
			break
		}
		if now.Sub(last.(time.Time)) < unnamedTierEvery {
			return
		}
		if r.tiersWarned.CompareAndSwap(key, last, now) {
			break
		}
	}
	r.logger.Warn("the vendor served a tier the price list does not name, priced at the default rates",
		"modality", r.modality, "provider", config.Provider, "model", config.Model, "service_tier", tier)
}

// row is the request a stat records, before it is priced: what it is priced by is what it
// stores.
func row(modality Modality, config ProviderConfig, entry Stat) store.Request {
	request := store.Request{
		Modality:          string(modality),
		CustomerID:        entry.CustomerID,
		AgentID:           entry.AgentID,
		CallID:            entry.CallID,
		Tags:              entry.Tags,
		Provider:          config.Provider,
		Model:             config.Model,
		StartedAt:         entry.StartedAt,
		OperationID:       entry.OperationID,
		Purpose:           entry.Purpose,
		TurnID:            entry.TurnID,
		AudioMs:           entry.AudioMs,
		Characters:        entry.Characters,
		InputTokens:       entry.InputTokens,
		CachedInputTokens: entry.CachedInputTokens,
		OutputTokens:      entry.OutputTokens,
		InputParts:        entry.InputParts,
		Images:            entry.Images,
		Meters:            entry.meters(),
		Attributes:        entry.Attributes,
		Success:           entry.Success,
		ErrorCode:         entry.ErrorCode,
		ErrorMessage:      store.SafeLogText(entry.ErrorMessage),
	}
	if entry.LatencyMs > 0 {
		latency := entry.LatencyMs
		request.LatencyMs = &latency
	}
	if entry.DurationMs > 0 {
		duration := entry.DurationMs
		request.DurationMs = &duration
	}
	return request
}

// meters is what the row stores beyond its typed columns: the stat's own meters, and the
// usage that has no column.
func (s Stat) meters() pricing.Quantities {
	if len(s.Meters) == 0 && s.Pixels == 0 && s.Requests == 0 {
		return nil
	}
	meters := maps.Clone(s.Meters)
	if meters == nil {
		meters = pricing.Quantities{}
	}
	if s.Pixels != 0 {
		meters[pricing.Pixels] += s.Pixels
	}
	if s.Requests != 0 {
		meters[pricing.Requests] += s.Requests
	}
	return meters
}

// Close drains the queue and stops the writer.
func (r *Recorder) Close() {
	r.closeOnce.Do(func() {
		r.mu.Lock()
		r.closed = true
		close(r.queue)
		r.mu.Unlock()
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped stats because the writer fell behind or they were recorded after Close",
				"modality", r.modality, "count", dropped)
		}
	})
}

func (r *Recorder) run() {
	defer close(r.done)

	for next := range r.queue {
		ctx, cancel := context.WithTimeout(context.Background(), statWriteTimeout)
		r.write(ctx, next)
		cancel()
	}
}

// queued is a stat on its way to the writer: the row, and what only health reads.
type queued struct {
	request store.Request
	fatal   bool
}

func (r *Recorder) write(ctx context.Context, next queued) {
	request := next.request
	if r.gate != nil {
		// Only the tags are wanted here. A refusal is the budget, which the work being
		// recorded has already got past.
		admission, _ := r.gate.Admit(ctx, request.CustomerID)
		request.Tags = admission.Labelled(request.Tags)
	}
	if r.store != nil {
		if err := r.store.RecordRequest(ctx, &request); err != nil {
			r.logger.Error("could not record request", "error", err)
		}
	}

	if r.live != nil {
		usage := liveUsage(request)
		usage.Fatal = next.fatal
		if err := r.live.RecordRequest(ctx, usage); err != nil {
			r.logger.Error("could not update live counters", "error", err)
		}
	}
}

// liveUsage is what a request adds to the live counters. A cancelled one is marked so, and the
// counters keep it out of provider health while still counting what it generated.
func liveUsage(request store.Request) live.Usage {
	var latencyMs float64
	if request.LatencyMs != nil {
		latencyMs = *request.LatencyMs
	}
	return live.Usage{
		Modality:          request.Modality,
		CustomerID:        request.CustomerID,
		Provider:          request.Provider,
		Model:             request.Model,
		LatencyMs:         latencyMs,
		AudioMs:           request.AudioMs,
		Characters:        request.Characters,
		InputTokens:       request.InputTokens,
		CachedInputTokens: request.CachedInputTokens,
		OutputTokens:      request.OutputTokens,
		CostMicros:        request.CostMicros,
		Success:           request.Success,
		Cancelled:         !measuresProvider(request),
	}
}

// measuresProvider reports whether a request says anything about its provider's health.
// One its caller cancelled does not: it neither failed nor was served, and how long it ran
// is how long it was left to, not how long the provider took. It is still billed for what
// it generated before it was cut off, so it is kept in the customer's spend and the request
// log.
func measuresProvider(request store.Request) bool {
	return request.ErrorCode != ErrorCancelled
}
