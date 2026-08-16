package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// memoryQueueSize bounds how many exchanges may wait to be remembered before they are
// dropped. Forgetting a turn costs the next conversation a little context; blocking the
// current one costs the participant a silence.
const memoryQueueSize = 64

// memoryTimeout bounds one call to the memory store.
const memoryTimeout = 10 * time.Second

// memoryProvider names the model row a memory call is recorded as, so what memory costs
// is reported next to what the models cost.
const memoryModel = "v3"

// memoryWriter hands finished exchanges to the memory store off the conversation's path.
type memoryWriter struct {
	store    memory.Store
	scope    memory.Scope
	owner    routing.Owner
	recorder *routing.Recorder
	logger   *slog.Logger

	queue chan []llm.Message
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newMemoryWriter(
	store memory.Store,
	scope memory.Scope,
	owner routing.Owner,
	recorder *routing.Recorder,
	logger *slog.Logger,
) *memoryWriter {
	w := &memoryWriter{
		store:    store,
		scope:    scope,
		owner:    owner,
		recorder: recorder,
		logger:   logger,
		queue:    make(chan []llm.Message, memoryQueueSize),
		done:     make(chan struct{}),
	}
	go w.run()
	return w
}

// Remember queues an exchange, dropping it if the writer is too far behind.
func (w *memoryWriter) Remember(messages []llm.Message) {
	if len(messages) == 0 {
		return
	}

	select {
	case w.queue <- messages:
	default:
		w.dropped.Add(1)
	}
}

// Recall asks what is already known about the participant. A failure is not fatal: an
// agent that has forgotten is worse than one that never knew, but neither is a broken
// call, so this reports the problem and carries on with nothing.
func (w *memoryWriter) Recall(ctx context.Context, limit int) []memory.Memory {
	started := time.Now()
	recalled, err := w.store.Recall(ctx, memory.Query{Scope: w.scope, Limit: limit})
	w.record(started, err)
	if err != nil {
		w.logger.Error("could not recall what is known about this participant", "error", err)
		return nil
	}
	return recalled
}

// Close drains the queue and stops the writer.
func (w *memoryWriter) Close() {
	w.closeOnce.Do(func() {
		close(w.queue)
		<-w.done
		if w.recorder != nil {
			w.recorder.Close()
		}
		if dropped := w.dropped.Load(); dropped > 0 {
			w.logger.Warn("dropped exchanges the memory writer could not keep up with", "count", dropped)
		}
	})
}

func (w *memoryWriter) run() {
	defer close(w.done)

	for messages := range w.queue {
		ctx, cancel := context.WithTimeout(context.Background(), memoryTimeout)
		started := time.Now()
		err := w.store.Remember(ctx, w.scope, messages)
		w.record(started, err)
		if err != nil {
			w.logger.Error("could not remember an exchange", "error", err)
		}
		cancel()
	}
}

// record files the call as a request row, so memory appears in cost reporting under the
// same customer and labels as the models the conversation used.
func (w *memoryWriter) record(started time.Time, err error) {
	if w.recorder == nil {
		return
	}

	stat := routing.Stat{
		Owner:     w.owner,
		StartedAt: started.UTC(),
		LatencyMs: routing.MsSince(started),
		Success:   err == nil,
	}
	if err != nil {
		stat.ErrorCode = "memory_failed"
	}
	w.recorder.Record(routing.ProviderConfig{
		Provider: w.store.Provider(),
		Model:    memoryModel,
	}, stat)
}
