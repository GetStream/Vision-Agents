package chatdelivery

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// Lease is how far a claim pushes a queued message's next attempt: no other router takes it
// while this one answers it. It outlasts the longest answer, a thread channel's, which waits
// up to askTimeout (2 min) for the thread's turn and up to another for the reply
// (api.answerThread), plus a minute for closing the session it opened. A lease that ran out
// mid-answer would have a second router answer the message again.
const Lease = 5 * time.Minute

// look is the longest wait between looks for due messages, so a message another router
// queued or leased and then stopped with is answered within about a lease and a look.
const look = time.Minute

// defaultPoll is the shortest wait between looks while messages are queued, so rows another
// router is claiming at that moment are not looked for in a busy loop.
const defaultPoll = time.Second

// inFlight is how many messages one router answers at once. An answer holds its slot for up
// to Lease, mostly waiting on a model; 64 is a choice, not a measurement, four times the
// event forwarder's 16 sends of 15 s at most.
const inFlight = 64

// defaultRetries are the waits before each attempt after the first: the inbound queue's
// schedule, the last one ten minutes after the one before.
var defaultRetries = []time.Duration{2 * time.Second, 10 * time.Second, 30 * time.Second, 2 * time.Minute, 10 * time.Minute}

// recordTimeout bounds writing what came of an attempt.
const recordTimeout = 10 * time.Second

// Answer answers one queued message, returning when its answer has ended. An error is an
// attempt a later one may get through.
type Answer func(ctx context.Context, delivery store.ChatDelivery) error

// Options configures a Worker. Store and Answer are required.
type Options struct {
	Store  *store.Store
	Answer Answer
	// Retries are the waits before each attempt after the first. Nil is defaultRetries.
	Retries []time.Duration
	// Poll is the shortest wait between looks while messages are queued. Zero is defaultPoll.
	Poll time.Duration
	// Lease is how far a claim pushes a message's next attempt. Zero is Lease. Tests shorten it.
	Lease  time.Duration
	Logger *slog.Logger
}

// Worker answers the Stream Chat messages queued in chat_deliveries, on every router: whichever
// claims one answers it.
type Worker struct {
	store   *store.Store
	answer  Answer
	retries []time.Duration
	poll    time.Duration
	lease   time.Duration
	logger  *slog.Logger

	// kick wakes the loop: a slot freed.
	kick chan struct{}
	// slots holds one token for each answer running.
	slots   chan struct{}
	ctx     context.Context
	cancel  context.CancelFunc
	working sync.WaitGroup
}

// NewWorker validates the options and returns a Worker. It claims nothing until Start.
func NewWorker(options Options) (*Worker, error) {
	if options.Store == nil || options.Answer == nil {
		return nil, stack.Wrap(errors.New("chatdelivery: a store and an answer are required"))
	}
	worker := &Worker{
		store:   options.Store,
		answer:  options.Answer,
		retries: options.Retries,
		poll:    options.Poll,
		lease:   options.Lease,
		logger:  options.Logger,
		kick:    make(chan struct{}, 1),
		slots:   make(chan struct{}, inFlight),
	}
	if worker.retries == nil {
		worker.retries = defaultRetries
	}
	if worker.poll <= 0 {
		worker.poll = defaultPoll
	}
	if worker.lease <= 0 {
		worker.lease = Lease
	}
	if worker.logger == nil {
		worker.logger = slog.Default()
	}
	worker.ctx, worker.cancel = context.WithCancel(context.Background())
	return worker, nil
}

// Start runs the loop until Close. It looks at start, for the messages a router queued before
// it stopped, then when an answer ends, at the first message due, and at least once a look.
func (w *Worker) Start() {
	w.working.Add(1)
	go func() {
		defer w.working.Done()
		timer := time.NewTimer(0)
		defer timer.Stop()
		for {
			wait, again := w.answerDue()
			timer.Stop()
			if again {
				timer.Reset(wait)
			}
			select {
			case <-w.ctx.Done():
				return
			case <-timer.C:
			case <-w.kick:
			}
		}
	}()
}

// Close stops claiming and waits for the answers in flight, so one is recorded rather than
// left for its lease to run out and another router to answer it again.
func (w *Worker) Close() {
	w.cancel()
	w.working.Wait()
}

// answerDue claims as many due messages as there are free slots and answers each on its own
// goroutine, until none is due or no slot is free. It returns how long to wait before looking
// again, with again false when only an ending answer needs to wake the loop.
func (w *Worker) answerDue() (wait time.Duration, again bool) {
	for {
		if w.ctx.Err() != nil {
			return 0, false
		}
		free := cap(w.slots) - len(w.slots)
		if free == 0 {
			return 0, false
		}
		now := time.Now()
		claimed, err := w.store.ClaimChatDeliveries(w.ctx, now, free, now.Add(w.lease))
		if err != nil {
			if w.ctx.Err() == nil {
				w.logger.Error("could not take the chat deliveries due", "error", err)
			}
			return w.poll, true
		}
		for _, delivery := range claimed {
			w.slots <- struct{}{}
			w.working.Add(1)
			go func() {
				defer w.working.Done()
				defer func() {
					<-w.slots
					select {
					case w.kick <- struct{}{}:
					default:
					}
				}()
				w.attempt(delivery)
			}()
		}
		if len(claimed) < free {
			break
		}
	}
	next, queued, err := w.store.NextChatDeliveryAt(w.ctx)
	switch {
	case err != nil:
		if w.ctx.Err() == nil {
			w.logger.Error("could not find when the next chat delivery is due", "error", err)
		}
		return w.poll, true
	case !queued:
		return look, true
	}
	return min(max(time.Until(next), w.poll), look), true
}

// attempt answers a claimed message once and records what came of it: answered, due again
// after the next wait, or given up when the waits ran out.
func (w *Worker) attempt(delivery store.ChatDelivery) {
	log := w.logger.With("seq", delivery.Seq, "stream_app", delivery.StreamApp, "message", delivery.MessageID)
	log.Info("answering a queued chat delivery", "attempts", delivery.Attempts)
	err := w.answer(w.ctx, delivery)
	if err != nil && w.ctx.Err() != nil {
		// Close cut it short: the lease runs out and a router answers it again.
		log.Info("left a queued chat delivery for another router", "error", err)
		return
	}
	// Recorded even while Close waits: an answered message left leased is answered again.
	writing, done := context.WithTimeout(context.WithoutCancel(w.ctx), recordTimeout)
	defer done()
	attempts := delivery.Attempts + 1
	var changed bool
	switch {
	case err == nil:
		changed, err = w.store.FinishChatDelivery(writing, delivery, time.Now())
		log.Info("answered a queued chat delivery", "attempts", attempts)
	case attempts <= len(w.retries):
		next := time.Now().Add(w.retries[attempts-1])
		log.Info("a queued chat delivery failed and is answered again", "attempts", attempts, "next", next, "error", err)
		changed, err = w.store.RetryChatDelivery(writing, delivery, next)
	default:
		log.Warn("gave up a queued chat delivery", "attempts", attempts, "error", err)
		changed, err = w.store.FailChatDelivery(writing, delivery, time.Now())
	}
	switch {
	case err != nil:
		log.Error("could not record a queued chat delivery's attempt", "error", err)
	case !changed:
		log.Warn("a queued chat delivery's lease ran out before its attempt was recorded")
	}
}
