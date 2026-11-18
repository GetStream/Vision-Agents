package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// How the decision writer behaves under load. A call makes several judgements a second,
// which is too many to write one at a time, and none of them is worth making a caller
// wait for.
const (
	decisionQueueSize   = 1024
	decisionBatchSize   = 64
	decisionFlushEvery  = time.Second
	decisionWriteTimout = 5 * time.Second
	// decisionDrainTimeout bounds how long Close waits for the queue to be written, the way
	// turnDrainTimeout does: two writes, the one in flight and what is left for the rest.
	// Batches not written by then are dropped and counted.
	decisionDrainTimeout = 2 * decisionWriteTimout
)

// decisionRecorder keeps the conversation's reasoning in Redis for the hour somebody may
// read the call back in, off the call's path.
//
// It batches, because a judgement is small and frequent and a round trip each would leave
// the writer permanently behind. It drops rather than blocks, because an account of a call
// is worth less than the call: Redis having a bad minute must cost somebody some detail in
// a dashboard and nobody a conversation.
type decisionRecorder struct {
	live   *live.Client
	owner  routing.Owner
	logger *slog.Logger

	queue chan live.CallEvent
	done  chan struct{}

	// send stores one batch; it is Redis unless a test says otherwise.
	send func(ctx context.Context, batch []live.CallEvent) error

	// ctx is the writer's, cancelled when Close gives up on what is still queued.
	ctx          context.Context
	cancel       context.CancelFunc
	drainTimeout time.Duration

	// closing guards the queue against a send after Close: a floor check is reported from
	// the transcriber's and the flow controller's goroutines, which may outlast the call, and
	// a send on a closed channel panics even from a select with a default.
	closing   sync.RWMutex
	closed    bool
	closeOnce sync.Once
	dropped   atomic.Int64
}

func newDecisionRecorder(liveClient *live.Client, owner routing.Owner, logger *slog.Logger) *decisionRecorder {
	return buildDecisionRecorder(liveClient, owner, logger).start()
}

// buildDecisionRecorder is a recorder that has not started writing, so a test can swap how.
func buildDecisionRecorder(liveClient *live.Client, owner routing.Owner, logger *slog.Logger) *decisionRecorder {
	r := &decisionRecorder{
		live:   liveClient,
		owner:  owner,
		logger: logger,
		queue:  make(chan live.CallEvent, decisionQueueSize),
		done:   make(chan struct{}),

		drainTimeout: decisionDrainTimeout,
	}
	r.send = liveClient.RecordCallEvents
	r.ctx, r.cancel = context.WithCancel(context.Background())
	return r
}

func (r *decisionRecorder) start() *decisionRecorder {
	go r.run()
	return r
}

// Record queues one judgement, dropping it if the writer is too far behind.
func (r *decisionRecorder) Record(decided Decided) {
	at := decided.At
	if at.IsZero() {
		at = time.Now()
	}

	row := live.CallEvent{
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
	r.enqueue(row)
}

// enqueue queues a row, dropping it if the writer is too far behind, and letting it go if the
// recorder has been closed.
func (r *decisionRecorder) enqueue(row live.CallEvent) {
	r.closing.RLock()
	defer r.closing.RUnlock()
	if r.closed {
		return
	}
	select {
	case r.queue <- row:
	default:
		r.dropped.Add(1)
	}
}

// RecordFloorCheck queues one of the chorus-asr model's checks on the floor, about turn when
// that was known as it was asked, dropping it if the writer is too far behind. It never
// waits: a check may be reported on the flow controller's goroutine.
func (r *decisionRecorder) RecordFloorCheck(turn string, check live.FloorCheck) {
	row := live.CallEvent{
		CustomerID: r.owner.CustomerID,
		CallID:     r.owner.CallID,
		AgentID:    r.owner.AgentID,
		At:         check.At,
		Kind:       "floor_check",
		TurnID:     turn,
		FloorCheck: &check,
	}
	r.enqueue(row)
}

// Close drains the queue and stops the writer, giving the writes drainTimeout between
// them. Judgements still unwritten then are dropped and counted with the rest.
func (r *decisionRecorder) Close() {
	r.closeOnce.Do(func() {
		r.closing.Lock()
		r.closed = true
		close(r.queue)
		r.closing.Unlock()
		drainWithin(r.done, r.drainTimeout, r.cancel)
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

	batch := make([]live.CallEvent, 0, decisionBatchSize)
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

func (r *decisionRecorder) write(batch []live.CallEvent) {
	if len(batch) == 0 {
		return
	}
	if r.ctx.Err() != nil {
		r.dropped.Add(int64(len(batch)))
		return
	}
	ctx, cancel := context.WithTimeout(r.ctx, decisionWriteTimout)
	defer cancel()

	if err := r.send(ctx, batch); err != nil {
		r.logger.Error("could not record what the conversation decided", "error", err)
		if ctx.Err() != nil {
			r.dropped.Add(int64(len(batch)))
		}
	}
}
