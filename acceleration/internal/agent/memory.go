package agent

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// memoryQueueSize bounds how many exchanges may wait to be remembered before they are
// dropped. Forgetting a turn costs the next conversation a little context; blocking the
// current one costs the participant a silence.
const memoryQueueSize = 64

// memoryTimeout bounds one call to the memory store.
const memoryTimeout = 10 * time.Second

// memoryDrainTimeout bounds how long Close waits for the queue to be written, the way
// turnDrainTimeout does: two writes, the one in flight and what is left for the rest. A
// mem0 that never answers would otherwise hold a call's teardown for memoryQueueSize
// writes of memoryTimeout each. Exchanges not written by then are dropped and counted.
const memoryDrainTimeout = 2 * memoryTimeout

// memoryWriter hands finished exchanges to the memory store off the conversation's path. The
// store is the agent's owner's, so every call is recorded under the same customer and labels
// as the models the conversation used.
type memoryWriter struct {
	store memory.Store
	scope memory.Scope
	// incognito recalls but never writes, so nothing said in the session is kept.
	incognito bool
	logger    *slog.Logger

	queue chan []llm.Message
	done  chan struct{}

	// ctx is the writer's, cancelled when Close gives up on what is still queued.
	ctx          context.Context
	cancel       context.CancelFunc
	drainTimeout time.Duration

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newMemoryWriter(store memory.Store, scope memory.Scope, incognito bool, logger *slog.Logger) *memoryWriter {
	w := &memoryWriter{
		store:     store,
		scope:     scope,
		incognito: incognito,
		logger:    logger,
		queue:     make(chan []llm.Message, memoryQueueSize),
		done:      make(chan struct{}),

		drainTimeout: memoryDrainTimeout,
	}
	w.ctx, w.cancel = context.WithCancel(context.Background())
	go w.run()
	return w
}

// Remember queues an exchange, dropping it if the writer is too far behind.
func (w *memoryWriter) Remember(messages []llm.Message) {
	if w.incognito || len(messages) == 0 {
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
	recalled, err := w.store.Recall(ctx, memory.Query{Scope: w.scope, Limit: limit})
	if err != nil {
		w.logger.Error("could not recall what is known about this participant", "error", err)
		return nil
	}
	return recalled
}

// Close drains the queue and stops the writer, giving the writes drainTimeout between
// them. Exchanges still unwritten then are dropped and counted with the rest.
func (w *memoryWriter) Close() {
	w.closeOnce.Do(func() {
		close(w.queue)
		drainWithin(w.done, w.drainTimeout, w.cancel)
		if dropped := w.dropped.Load(); dropped > 0 {
			w.logger.Warn("dropped exchanges the memory writer could not keep up with", "count", dropped)
		}
	})
}

func (w *memoryWriter) run() {
	defer close(w.done)

	for messages := range w.queue {
		if w.ctx.Err() != nil {
			w.dropped.Add(1)
			continue
		}
		ctx, cancel := context.WithTimeout(w.ctx, memoryTimeout)
		if err := w.store.Remember(ctx, w.scope, messages); err != nil {
			w.logger.Error("could not remember an exchange", "error", err)
			if ctx.Err() != nil {
				w.dropped.Add(1)
			}
		}
		cancel()
	}
}
