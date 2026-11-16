package session

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// invocationQueueSize bounds how far the invocation writer may fall behind before rows are
// dropped. The session writer's recordQueueSize: a connector call is rarer than a turn's
// items, so a queue that holds those holds these. Unverified, not measured.
const invocationQueueSize = 1024

// invocationRecorder writes one store.ConnectorInvocation for each connector tool call off
// the call's path (T29, AI-856).
//
// The same trade as the session writer (sessionRecorder): the model must never wait on a
// database to read a tool's answer, and a call must not fail because its row could not be
// written. So Record never blocks and never fails; a row the writer cannot keep up with is
// dropped and counted, and one the store refuses is logged.
//
// An incognito session's rows reach it too, without the session's id: the row says a
// credential was used, which the connection's owner is owed, and nothing in it ties the use
// to the conversation (Spec.Incognito). No session's row holds the call's arguments or
// result.
type invocationRecorder struct {
	store  *store.Store
	logger *slog.Logger

	queue chan store.ConnectorInvocation
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newInvocationRecorder(pgStore *store.Store, logger *slog.Logger) *invocationRecorder {
	r := &invocationRecorder{
		store:  pgStore,
		logger: logger,
		queue:  make(chan store.ConnectorInvocation, invocationQueueSize),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

// Record queues one call's row. A nil recorder, on a deployment with connectors off, records
// nothing.
func (r *invocationRecorder) Record(row store.ConnectorInvocation) {
	if r == nil {
		return
	}
	select {
	case r.queue <- row:
	default:
		r.dropped.Add(1)
	}
}

// Close drains the queue and stops the writer.
func (r *invocationRecorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped connector invocations because the writer fell behind", "count", dropped)
		}
	})
}

func (r *invocationRecorder) run() {
	defer close(r.done)
	for row := range r.queue {
		ctx, cancel := context.WithTimeout(context.Background(), recordWriteTimeout)
		if err := r.store.RecordConnectorInvocation(ctx, &row); err != nil {
			r.logger.Error("could not record a connector invocation", "connection", row.ConnectionID, "error", err)
		}
		cancel()
	}
}
