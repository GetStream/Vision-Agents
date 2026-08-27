package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// How the decision writer behaves under load. A call makes several judgements a second,
// which is too many to write one at a time, and none of them is worth making a caller
// wait for.
const (
	decisionQueueSize   = 1024
	decisionBatchSize   = 64
	decisionFlushEvery  = time.Second
	decisionWriteTimout = 5 * time.Second
)

// decisionRecorder writes the conversation's reasoning to Postgres off the call's path.
//
// It batches, because a judgement is small and frequent and a round trip each would leave
// the writer permanently behind. It drops rather than blocks, because an account of a call
// is worth less than the call: a database having a bad minute must cost somebody some
// detail in a dashboard and nobody a conversation.
type decisionRecorder struct {
	store  *store.Store
	owner  routing.Owner
	logger *slog.Logger

	queue chan store.CallEvent
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newDecisionRecorder(pgStore *store.Store, owner routing.Owner, logger *slog.Logger) *decisionRecorder {
	r := &decisionRecorder{
		store:  pgStore,
		owner:  owner,
		logger: logger,
		queue:  make(chan store.CallEvent, decisionQueueSize),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

// Record queues one judgement, dropping it if the writer is too far behind.
func (r *decisionRecorder) Record(decided Decided) {
	at := decided.At
	if at.IsZero() {
		at = time.Now()
	}

	row := store.CallEvent{
		CustomerID:  r.owner.CustomerID,
		CallID:      r.owner.CallID,
		AgentID:     r.owner.AgentID,
		At:          at.UTC(),
		Kind:        decided.Kind,
		Reason:      decided.Reason,
		TurnID:      decided.TurnID,
		Participant: decided.Participant.ID,
		Said:        decided.Text,
		LatencyMs:   measured(decided.LatencyMs),
	}

	select {
	case r.queue <- row:
	default:
		r.dropped.Add(1)
	}
}

// Close drains the queue and stops the writer.
func (r *decisionRecorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped decisions because the writer fell behind", "count", dropped)
		}
	})
}

// run gathers judgements into batches, writing when one is full or when the oldest in it
// has waited long enough that somebody watching a live call would notice.
func (r *decisionRecorder) run() {
	defer close(r.done)

	ticker := time.NewTicker(decisionFlushEvery)
	defer ticker.Stop()

	batch := make([]store.CallEvent, 0, decisionBatchSize)
	for {
		select {
		case row, open := <-r.queue:
			if !open {
				r.write(batch)
				return
			}
			batch = append(batch, row)
			if len(batch) >= decisionBatchSize {
				r.write(batch)
				batch = batch[:0]
			}

		case <-ticker.C:
			if len(batch) > 0 {
				r.write(batch)
				batch = batch[:0]
			}
		}
	}
}

func (r *decisionRecorder) write(batch []store.CallEvent) {
	if len(batch) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), decisionWriteTimout)
	defer cancel()

	if err := r.store.RecordCallEvents(ctx, batch); err != nil {
		r.logger.Error("could not record what the conversation decided", "error", err)
	}
}
