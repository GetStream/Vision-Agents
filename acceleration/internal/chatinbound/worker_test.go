//go:build integration

package chatinbound

import (
	"bytes"
	"context"
	"errors"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// WorkerSuite is the queue's worker against Postgres: what it answers, tries again, gives up,
// and what Close leaves. The answer is a function of each test's own.
type WorkerSuite struct {
	suite.Suite
	ctx   context.Context
	store *store.Store
}

func TestWorkerSuite(t *testing.T) {
	suite.Run(t, new(WorkerSuite))
}

func (s *WorkerSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN not set")
	}
	s.ctx = context.Background()
	// A database of this suite's own, as the worker claims every row in it.
	db, err := store.Open(testenv.Database(dsn, "chatinbound"))
	s.Require().NoError(err)
	s.store = db
	s.Require().NoError(db.Migrate(s.ctx))
}

func (s *WorkerSuite) SetupTest() {
	// A row an earlier test left queued would be the next test's worker's to answer.
	_, err := s.store.DB().NewDelete().Model((*store.ChatInbound)(nil)).Where("TRUE").Exec(s.ctx)
	s.Require().NoError(err)
}

func (s *WorkerSuite) TearDownSuite() {
	if s.store != nil {
		s.Require().NoError(s.store.Close())
	}
}

func (s *WorkerSuite) TestAQueuedDeliveryIsAnsweredOnce() {
	answers := &answers{}
	queued := s.queue()

	s.start(Options{Answer: answers.answer(nil)})

	s.Eventually(func() bool { return s.row(queued).State == store.ChatInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return answers.of(queued) > 1 }, 200*time.Millisecond, 10*time.Millisecond)
	s.Equal(1, answers.of(queued))
}

// Each wait comes before its attempt, and once they run out the delivery is given up, with a
// warning that names it and no message text.
func (s *WorkerSuite) TestADeliveryThatKeepsFailingIsTriedAfterEachWaitThenGivenUp() {
	answers := &answers{}
	var logged syncBuffer
	queued := s.queue()

	s.start(Options{
		Answer:  answers.answer(func(int) error { return errors.New("nobody could answer") }),
		Retries: []time.Duration{100 * time.Millisecond, 300 * time.Millisecond},
		Logger:  slog.New(slog.NewTextHandler(&logged, nil)),
	})

	s.Eventually(func() bool { return s.row(queued).State == store.ChatInboundFailed }, 5*time.Second, 10*time.Millisecond)
	row := s.row(queued)
	s.Equal(3, row.Attempts)
	s.NotNil(row.FinishedAt)
	at := answers.times(queued)
	s.Require().Len(at, 3)
	s.GreaterOrEqual(at[1].Sub(at[0]), 100*time.Millisecond)
	s.GreaterOrEqual(at[2].Sub(at[1]), 300*time.Millisecond)
	s.Contains(logged.String(), `level=WARN msg="gave up a queued chat inbound message" seq=`)
	s.Contains(logged.String(), "message="+queued)
}

func (s *WorkerSuite) TestADeliveryThatFailedOnceIsAnsweredOnTheNextTry() {
	answers := &answers{}
	queued := s.queue()

	s.start(Options{
		Answer: answers.answer(func(attempt int) error {
			if attempt == 1 {
				return errors.New("nobody could answer")
			}
			return nil
		}),
		Retries: []time.Duration{50 * time.Millisecond},
	})

	s.Eventually(func() bool { return s.row(queued).State == store.ChatInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(1, s.row(queued).Attempts)
	s.Equal(2, answers.of(queued))
}

// Close waits for the answer in flight and records it, so no other router answers it again
// once the lease runs out, and claims nothing after.
func (s *WorkerSuite) TestCloseWaitsForTheAnswerInFlightAndRecordsIt() {
	answers := &answers{}
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	queued := s.queue()
	worker := s.start(Options{Answer: answers.answer(func(int) error {
		started <- struct{}{}
		<-release
		return nil
	})})
	<-started

	closed := make(chan struct{})
	go func() {
		worker.Close()
		close(closed)
	}()
	s.Never(func() bool { return isClosed(closed) }, 200*time.Millisecond, 10*time.Millisecond, "Close did not wait for the answer")
	later := s.queue()
	close(release)

	s.Eventually(func() bool { return isClosed(closed) }, 5*time.Second, 10*time.Millisecond)
	s.Equal(store.ChatInboundDone, s.row(queued).State)
	s.Never(func() bool { return answers.of(later) > 0 }, 200*time.Millisecond, 10*time.Millisecond, "a closed worker claimed a delivery")
}

// The router's stop signal ends the worker's claims: the answer it took before goes on and is
// recorded, and a delivery queued after the signal stays queued, with no attempt, until another
// router answers it.
func (s *WorkerSuite) TestAfterTheStopSignalAWorkerFinishesItsAnswerAndClaimsNoOther() {
	answers := &answers{}
	release := make(chan struct{})
	started := make(chan struct{}, 1)
	queued := s.queue()
	stopping, stop := context.WithCancel(s.ctx)
	worker := s.startUntil(stopping, Options{Answer: answers.answer(func(int) error {
		started <- struct{}{}
		<-release
		return nil
	})})
	<-started

	stop()
	later := s.queue()
	worker.Kick()
	close(release)

	s.Eventually(func() bool { return s.row(queued).State == store.ChatInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool {
		row := s.row(later)
		return answers.of(later) > 0 || row.Attempts != 0 || row.NextAttemptAt.After(time.Now())
	}, 300*time.Millisecond, 10*time.Millisecond, "a stopped worker claimed a delivery")

	s.start(Options{Answer: answers.answer(nil)})

	s.Eventually(func() bool { return s.row(later).State == store.ChatInboundDone }, 5*time.Second, 10*time.Millisecond)
	s.Equal(1, answers.of(later))
}

// Running counts the answers in flight, for a draining router to wait for, until each ends.
func (s *WorkerSuite) TestRunningCountsTheAnswersInFlight() {
	release := make(chan struct{})
	started := make(chan struct{}, 2)
	first, second := s.queue(), s.queue()
	worker := s.start(Options{Answer: func(context.Context, store.ChatInbound) error {
		started <- struct{}{}
		<-release
		return nil
	}})
	<-started
	<-started

	s.Equal(2, worker.Running())
	close(release)

	s.Require().Eventually(func() bool {
		return s.row(first).State == store.ChatInboundDone && s.row(second).State == store.ChatInboundDone
	}, 5*time.Second, 10*time.Millisecond)
	s.Require().Eventually(func() bool { return worker.Running() == 0 }, 5*time.Second, 10*time.Millisecond)
}

// An answer Close cut short is no attempt: the row waits for its lease to run out, and a
// router answers it then.
func (s *WorkerSuite) TestAnAnswerCloseCutShortIsLeftForItsLease() {
	started := make(chan struct{}, 1)
	queued := s.queue()
	worker := s.start(Options{Answer: func(ctx context.Context, _ store.ChatInbound) error {
		started <- struct{}{}
		<-ctx.Done()
		return ctx.Err()
	}})
	<-started

	worker.Close()

	row := s.row(queued)
	s.Equal(store.ChatInboundQueued, row.State)
	s.Zero(row.Attempts)
	s.True(row.NextAttemptAt.After(time.Now().Add(Lease-time.Minute)), "the lease was let go of: %s", row.NextAttemptAt)
}

// One router answers at most inFlight messages at once; the next is left unclaimed, for
// another router to take, until a slot frees.
func (s *WorkerSuite) TestAtMostInFlightAnswersRunAtOnce() {
	release := make(chan struct{})
	var mu sync.Mutex
	started := 0
	for range inFlight + 1 {
		s.queue()
	}
	s.start(Options{Answer: func(context.Context, store.ChatInbound) error {
		mu.Lock()
		started++
		mu.Unlock()
		<-release
		return nil
	}})
	count := func() int {
		mu.Lock()
		defer mu.Unlock()
		return started
	}

	s.Eventually(func() bool { return count() == inFlight }, 5*time.Second, 10*time.Millisecond)
	s.Never(func() bool { return count() > inFlight }, 300*time.Millisecond, 10*time.Millisecond)
	unclaimed, err := s.store.DB().NewSelect().Model((*store.ChatInbound)(nil)).Where("next_attempt_at <= now()").Count(s.ctx)
	s.Require().NoError(err)
	s.Equal(1, unclaimed)
	close(release)
	s.Eventually(func() bool { return count() == inFlight+1 }, 5*time.Second, 10*time.Millisecond)
}

// A lease shorter than the answer has the message answered twice, which is why Lease
// outlasts the longest answer; the late attempt changes nothing and says so.
func (s *WorkerSuite) TestAnAnswerThatOutlastsItsLeaseIsAnsweredAgainAndSaysSo() {
	answers := &answers{}
	var logged syncBuffer
	queued := s.queue()

	s.start(Options{
		Answer: answers.answer(func(attempt int) error {
			if attempt == 1 {
				time.Sleep(300 * time.Millisecond)
			}
			return nil
		}),
		Lease:  100 * time.Millisecond,
		Poll:   10 * time.Millisecond,
		Logger: slog.New(slog.NewTextHandler(&logged, nil)),
	})

	s.Eventually(func() bool {
		return strings.Contains(logged.String(), "lease ran out before its attempt was recorded")
	}, 5*time.Second, 10*time.Millisecond)
	s.Equal(2, answers.of(queued))
	s.Equal(store.ChatInboundDone, s.row(queued).State)
}

// Two routers' workers answer every message once between them, and close while answering.
func (s *WorkerSuite) TestTwoWorkersAnswerEachDeliveryOnce() {
	answers := &answers{}
	var queued []string
	for range 40 {
		queued = append(queued, s.queue())
	}
	answer := answers.answer(func(int) error {
		time.Sleep(5 * time.Millisecond)
		return nil
	})
	first, second := s.start(Options{Answer: answer}), s.start(Options{Answer: answer})

	s.Eventually(func() bool { return answers.total() == len(queued) }, 10*time.Second, 10*time.Millisecond)
	var closing sync.WaitGroup
	for _, worker := range []*Worker{first, second} {
		closing.Go(worker.Close)
	}
	closing.Wait()
	for _, id := range queued {
		s.Equal(1, answers.of(id), id)
		s.Equal(store.ChatInboundDone, s.row(id).State, id)
	}
}

// A Kick never waits for the loop: the endpoint kicks on its request, before the worker has
// started and after it closed as well, and a Kick that waited would hold that request forever.
func (s *WorkerSuite) TestAKickBeforeStartOrAfterCloseReturnsAtOnce() {
	worker, err := NewWorker(Options{Store: s.store, Answer: (&answers{}).answer(nil)})
	s.Require().NoError(err)
	kicked := make(chan struct{})
	go func() {
		defer close(kicked)
		worker.Kick()
		worker.Kick()
		worker.Start(s.ctx)
		worker.Close()
		worker.Kick()
		worker.Kick()
	}()

	s.Eventually(func() bool { return isClosed(kicked) }, 5*time.Second, time.Millisecond)
}

// A worker with no store or no answer could only fail later, on its first message.
func (s *WorkerSuite) TestAWorkerNeedsAStoreAndAnAnswer() {
	_, err := NewWorker(Options{Answer: (&answers{}).answer(nil)})
	s.Error(err)
	_, err = NewWorker(Options{Store: s.store})
	s.Error(err)
}

// With every slot taken the loop waits for an answer to end rather than look again.
func (s *WorkerSuite) TestAWorkerWithEverySlotTakenWaitsForOneToFree() {
	s.queue()
	worker, err := NewWorker(Options{Store: s.store, Answer: (&answers{}).answer(nil)})
	s.Require().NoError(err)
	for range inFlight {
		worker.slots <- struct{}{}
	}

	_, again := worker.answerDue(s.ctx)

	s.False(again)
}

// A look that could not read the queue is tried again a poll on, so a database that was away
// for a moment does not leave the queue unanswered.
func (s *WorkerSuite) TestALookThatFailedIsTriedAgainAPollOn() {
	closed, err := store.Open(testenv.Database(os.Getenv("ROUTER_POSTGRES_DSN"), "chatinbound"))
	s.Require().NoError(err)
	s.Require().NoError(closed.Close())
	worker, err := NewWorker(Options{Store: closed, Answer: (&answers{}).answer(nil), Poll: 30 * time.Millisecond})
	s.Require().NoError(err)

	wait, again := worker.answerDue(s.ctx)

	s.True(again)
	s.Equal(30*time.Millisecond, wait)
}

// With nothing queued a worker looks again a look later, for a message another router queues.
func (s *WorkerSuite) TestAWorkerWithNothingQueuedLooksAgainALookLater() {
	worker, err := NewWorker(Options{Store: s.store, Answer: (&answers{}).answer(nil), Poll: 30 * time.Millisecond})
	s.Require().NoError(err)

	wait, again := worker.answerDue(s.ctx)

	s.True(again)
	s.Equal(look, wait)
}

// A worker deletes the rows past their keep at its first look, and not again within the hour.
func (s *WorkerSuite) TestTheWorkerDeletesRowsPastTheirKeepOnceAnHour() {
	worker, err := NewWorker(Options{Store: s.store, Answer: (&answers{}).answer(nil)})
	s.Require().NoError(err)
	first := s.finished(s.queue(), 25*time.Hour)

	worker.answerDue(s.ctx)

	s.False(s.exists(first), "an answered message's row is kept a day")
	second := s.finished(s.queue(), 25*time.Hour)

	worker.answerDue(s.ctx)

	s.True(s.exists(second), "the next prune is an hour after the first")
}

// queue queues a delivery due now and returns its message id.
func (s *WorkerSuite) queue() string {
	id := "message-" + uuid.NewString()
	queued, err := s.store.QueueChatInbound(s.ctx, store.ChatInbound{
		StreamApp: 1, MessageID: id, Deployment: true, Body: []byte(`{}`), NextAttemptAt: time.Now(),
	})
	s.Require().NoError(err)
	s.Require().True(queued)
	return id
}

// start runs a worker on the suite's store, looking every 10 ms, until the test ends.
func (s *WorkerSuite) start(options Options) *Worker {
	return s.startUntil(s.ctx, options)
}

// startUntil is start whose worker claims until stopping ends, as a router's until its stop
// signal.
func (s *WorkerSuite) startUntil(stopping context.Context, options Options) *Worker {
	options.Store = s.store
	if options.Poll == 0 {
		options.Poll = 10 * time.Millisecond
	}
	worker, err := NewWorker(options)
	s.Require().NoError(err)
	worker.Start(stopping)
	s.T().Cleanup(worker.Close)
	return worker
}

// finished records a queued delivery answered ago, and returns its message id.
func (s *WorkerSuite) finished(messageID string, ago time.Duration) string {
	_, err := s.store.DB().NewUpdate().Model((*store.ChatInbound)(nil)).
		Set("state = ?", store.ChatInboundDone).Set("finished_at = ?", time.Now().Add(-ago)).
		Where("message_id = ?", messageID).Exec(s.ctx)
	s.Require().NoError(err)
	return messageID
}

func (s *WorkerSuite) exists(messageID string) bool {
	exists, err := s.store.DB().NewSelect().Model((*store.ChatInbound)(nil)).Where("message_id = ?", messageID).Exists(s.ctx)
	s.Require().NoError(err)
	return exists
}

func (s *WorkerSuite) row(messageID string) store.ChatInbound {
	var row store.ChatInbound
	s.Require().NoError(s.store.DB().NewSelect().Model(&row).Where("message_id = ?", messageID).Scan(s.ctx))
	return row
}

// answers records each attempt at each message, and when it was made.
type answers struct {
	mu sync.Mutex
	at map[string][]time.Time
}

// answer is an Answer that records the attempt, then fails it with what outcome says for the
// message's nth attempt, or answers it when outcome is nil.
func (a *answers) answer(outcome func(attempt int) error) Answer {
	return func(_ context.Context, delivery store.ChatInbound) error {
		a.mu.Lock()
		if a.at == nil {
			a.at = map[string][]time.Time{}
		}
		a.at[delivery.MessageID] = append(a.at[delivery.MessageID], time.Now())
		attempt := len(a.at[delivery.MessageID])
		a.mu.Unlock()
		if outcome == nil {
			return nil
		}
		return outcome(attempt)
	}
}

func (a *answers) times(id string) []time.Time {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]time.Time(nil), a.at[id]...)
}

func (a *answers) of(id string) int { return len(a.times(id)) }

func (a *answers) total() int {
	a.mu.Lock()
	defer a.mu.Unlock()
	total := 0
	for _, at := range a.at {
		total += len(at)
	}
	return total
}

func isClosed(done <-chan struct{}) bool {
	select {
	case <-done:
		return true
	default:
		return false
	}
}

// syncBuffer is a log the worker's goroutines write while the test reads it.
type syncBuffer struct {
	mu  sync.Mutex
	buf bytes.Buffer
}

func (b *syncBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.Write(p)
}

func (b *syncBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}
