package routing

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// statQueueSize bounds how far the writer may fall behind before stats are dropped.
const statQueueSize = 512

// statWriteTimeout bounds a single write so a stuck backend cannot wedge the writer.
const statWriteTimeout = 5 * time.Second

// Owner is who a unit of work belongs to and how the customer wants it labelled. It
// travels with a session so every row the session produces is attributed the same way.
type Owner struct {
	// CustomerID owns the request. Every statistic is keyed by it.
	CustomerID string
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
// cannot record a row that disagrees with the routing decision it came from.
type Stat struct {
	Owner
	StartedAt time.Time
	Usage
	// LatencyMs is how long the customer waited for the work to be useful.
	LatencyMs float64
	// CostMicros overrides the priced amount for work that is not billed by the units in
	// Usage, such as a phone number's monthly charge, where the vendor quotes the price
	// outright rather than a rate to multiply. Zero leaves the pricing to the rates.
	CostMicros int64
	Success    bool
	ErrorCode  string
}

// Recorder writes stats to Postgres and Redis off the request path. A conversation must
// never wait on a database, so recording is asynchronous and stats are the thing that
// gets dropped when the backend cannot keep up.
type Recorder struct {
	modality Modality
	store    *store.Store
	live     *live.Client
	logger   *slog.Logger

	queue chan store.Request
	done  chan struct{}

	closeOnce sync.Once
	// dropped counts stats lost to a full queue, reported once on shutdown.
	dropped atomic.Int64
}

// NewRecorder starts the background writer.
func NewRecorder(modality Modality, pgStore *store.Store, liveClient *live.Client, logger *slog.Logger) *Recorder {
	r := &Recorder{
		modality: modality,
		store:    pgStore,
		live:     liveClient,
		logger:   logger,
		queue:    make(chan store.Request, statQueueSize),
		done:     make(chan struct{}),
	}
	go r.run()
	return r
}

// Record queues a stat, pricing it from the provider's configured rates and dropping it
// if the writer is too far behind.
func (r *Recorder) Record(config ProviderConfig, entry Stat) {
	if r.store == nil && r.live == nil {
		return
	}

	cost := config.Price.CostMicros(entry.Usage)
	if entry.CostMicros != 0 {
		cost = entry.CostMicros
	}

	request := store.Request{
		Modality:          string(r.modality),
		CustomerID:        entry.CustomerID,
		AgentID:           entry.AgentID,
		CallID:            entry.CallID,
		Tags:              entry.Tags,
		Provider:          config.Provider,
		Model:             config.Model,
		StartedAt:         entry.StartedAt,
		AudioMs:           entry.AudioMs,
		Characters:        entry.Characters,
		InputTokens:       entry.InputTokens,
		CachedInputTokens: entry.CachedInputTokens,
		OutputTokens:      entry.OutputTokens,
		CostMicros:        cost,
		Success:           entry.Success,
		ErrorCode:         entry.ErrorCode,
	}
	if entry.LatencyMs > 0 {
		latency := entry.LatencyMs
		request.LatencyMs = &latency
	}

	select {
	case r.queue <- request:
	default:
		r.dropped.Add(1)
	}
}

// Close drains the queue and stops the writer.
func (r *Recorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped stats because the writer fell behind",
				"modality", r.modality, "count", dropped)
		}
	})
}

func (r *Recorder) run() {
	defer close(r.done)

	for request := range r.queue {
		ctx, cancel := context.WithTimeout(context.Background(), statWriteTimeout)
		r.write(ctx, request)
		cancel()
	}
}

func (r *Recorder) write(ctx context.Context, request store.Request) {
	if r.store != nil {
		if err := r.store.RecordRequest(ctx, &request); err != nil {
			r.logger.Error("could not record request", "error", err)
		}
	}

	if r.live != nil {
		var latencyMs float64
		if request.LatencyMs != nil {
			latencyMs = *request.LatencyMs
		}
		err := r.live.RecordRequest(ctx, live.Usage{
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
		})
		if err != nil {
			r.logger.Error("could not update live counters", "error", err)
		}
	}
}
