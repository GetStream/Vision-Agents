package session

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// callQueueSize bounds how far the call writer may fall behind before rows are dropped.
const callQueueSize = 128

// callWriteTimeout bounds a single write so a stuck database cannot wedge the writer.
const callWriteTimeout = 5 * time.Second

// callWrite is either a call that began or one that ended. Both are the same piece of
// work to the writer: something happened to a conversation, and Postgres should hear
// about it whenever it can.
type callWrite struct {
	// started is the row a beginning call writes, nil when this is an ending.
	started *store.Call
	// id and at are which call ended and when.
	id string
	at time.Time
}

// callRecorder writes calls to Postgres off the conversation's path.
//
// A caller must never wait on a database to be answered, and a call must not fail to
// happen because the row recording it could not be written. So recording is asynchronous,
// and rows are what gets dropped when the writer cannot keep up.
type callRecorder struct {
	store  *store.Store
	logger *slog.Logger

	queue chan callWrite
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newCallRecorder(pgStore *store.Store, logger *slog.Logger) *callRecorder {
	r := &callRecorder{
		store:  pgStore,
		logger: logger,
		queue:  make(chan callWrite, callQueueSize),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

// Started queues the row for a call that has just joined.
func (r *callRecorder) Started(row store.Call) {
	r.queueWrite(callWrite{started: &row})
}

// Ended queues the time a call left.
func (r *callRecorder) Ended(id string, at time.Time) {
	r.queueWrite(callWrite{id: id, at: at})
}

// Close drains the queue and stops the writer.
func (r *callRecorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped calls because the writer fell behind", "count", dropped)
		}
	})
}

func (r *callRecorder) queueWrite(write callWrite) {
	select {
	case r.queue <- write:
	default:
		r.dropped.Add(1)
	}
}

func (r *callRecorder) run() {
	defer close(r.done)

	for write := range r.queue {
		ctx, cancel := context.WithTimeout(context.Background(), callWriteTimeout)
		if write.started != nil {
			if err := r.store.StartCall(ctx, write.started); err != nil {
				r.logger.Error("could not record the call starting", "error", err)
			}
		} else if err := r.store.FinishCall(ctx, write.id, write.at); err != nil {
			r.logger.Error("could not record the call ending", "error", err)
		}
		cancel()
	}
}

// row is what a session looks like to somebody reading it back later.
func row(created *Session) store.Call {
	spec := created.spec
	call := store.Call{
		ID:         created.id,
		CustomerID: spec.CustomerID,
		CallID:     spec.CallID,
		AgentID:    spec.AgentID,
		ConfigID:   spec.ConfigID,
		CampaignID: spec.CampaignID,
		ContactID:  spec.ContactID,
		Direction:  store.Inbound,
		StartedAt:  created.created.UTC(),
		Tags:       spec.Tags,
		// The spec here has already had a config folded into it, so these are what the
		// call actually ran with rather than what either side asked for on its own.
		STT:          spec.STTTarget,
		TTS:          spec.TTSTarget,
		LLM:          spec.LLMTarget,
		Subagent:     spec.SubagentTarget,
		Instructions: spec.prompt(),
	}
	for _, skill := range created.skills.Skills {
		call.Skills = append(call.Skills, skill.Name)
	}
	if spec.Phone != nil {
		call.FromNumber = spec.Phone.Number
		call.ToNumber = spec.Phone.To
		// A vendor call id means this process placed the call rather than answered it.
		if spec.Phone.VendorCallID != "" {
			call.Direction = store.Outbound
		}
	}
	return call
}
