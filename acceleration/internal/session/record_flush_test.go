//go:build integration

package session

import (
	"context"
	"crypto/rand"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// RecordFlushSuite is the session writer's Flush, against Postgres.
type RecordFlushSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	customer string
	records  *sessionRecorder
}

func TestRecordFlushSuite(t *testing.T) {
	suite.Run(t, new(RecordFlushSuite))
}

func (s *RecordFlushSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "record_flush"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *RecordFlushSuite) SetupTest() {
	s.customer = "flush-" + rand.Text()
	s.records = newSessionRecorder(s.store, slog.New(slog.DiscardHandler))
	s.T().Cleanup(s.records.Close)
}

func (s *RecordFlushSuite) TestFlushesWaitingOnASlowWriterLeaveRoomForRows() {
	release := s.holdWriter()
	waiters := recordQueueSize + 64
	errs := make(chan error, waiters)
	var started, finished sync.WaitGroup
	for range waiters {
		started.Add(1)
		finished.Go(func() {
			started.Done()
			errs <- s.records.Flush(s.bounded())
		})
	}
	started.Wait()
	s.Never(func() bool { return len(s.records.queue) > 0 }, 200*time.Millisecond, 5*time.Millisecond,
		"the waiting flushes took room in the queue")

	later := s.row()
	s.records.Opened(later, nil)
	s.Zero(s.records.Dropped(), "a row queued behind the waiting flushes was dropped")

	release()
	finished.Wait()
	close(errs)
	for err := range errs {
		s.Require().NoError(err)
	}
	s.Require().NoError(s.records.Flush(s.bounded()))
	s.True(s.stored(later.ID), "a row queued behind the waiting flushes is not stored")
}

func (s *RecordFlushSuite) TestEachFlushWaitsForTheRowItsCallerQueuedBeforeIt() {
	release := s.holdWriter()
	const callers = 32
	missing := make(chan string, callers)
	var started, finished sync.WaitGroup
	for range callers {
		row := s.row()
		started.Add(1)
		finished.Go(func() {
			s.records.Opened(row, nil)
			started.Done()
			if err := s.records.Flush(s.bounded()); err != nil {
				missing <- err.Error()
				return
			}
			if !s.stored(row.ID) {
				missing <- row.ID
			}
		})
	}
	// Every row is queued behind the held one before the writer moves, so each Flush
	// reaches it with rows still to write.
	started.Wait()
	release()
	finished.Wait()
	close(missing)
	var unwritten []string
	for id := range missing {
		unwritten = append(unwritten, id)
	}
	s.Empty(unwritten, "a Flush returned before the row queued ahead of it was written")
}

func (s *RecordFlushSuite) TestAFlushReturnsOnceTheItemsQueuedBeforeItAreWritten() {
	row := s.row()
	s.records.Opened(row, nil)
	response := store.AgentResponse{ID: rand.Text(), SessionID: row.ID, CustomerID: s.customer, Said: "Where is my order?"}
	s.records.Responding(response)
	for ordinal, text := range []string{"Where is my order?", "It ships today.", "Thanks."} {
		s.records.Item(store.AgentResponseItem{ResponseID: response.ID, Ordinal: ordinal, SessionID: row.ID,
			Kind: store.ItemSaid, Text: text, At: time.Now().UTC()})
	}

	s.Require().NoError(s.records.Flush(s.bounded()))

	// Base 4f22418b8: all three, at once.
	items, err := s.store.SessionItems(s.ctx, s.customer, row.ID, response.ID, 10, nil)
	s.Require().NoError(err)
	s.Len(items, 3, "the items still batching in the writer are not written")
}

func (s *RecordFlushSuite) TestAFlushWaitingAsTheWriterClosesReturnsOnceItsRowIsWritten() {
	release := s.holdWriter()
	row := s.row()
	s.records.Opened(row, nil)
	flushed := make(chan error, 1)
	go func() { flushed <- s.records.Flush(s.bounded()) }()
	closed := make(chan struct{})
	go func() {
		s.records.Close()
		close(closed)
	}()

	release()
	<-closed
	// Base 4f22418b8 answered nil with the row stored when the Flush reached the queue
	// before Close did, and panicked when it came after.
	s.Require().NoError(<-flushed)
	s.True(s.stored(row.ID), "the row queued before the Flush is not stored")
}

func (s *RecordFlushSuite) TestAFlushAfterTheWriterClosedReturnsAtOnce() {
	s.records.Close()

	// Base 4f22418b8 panicked here: the marker was sent on the closed queue.
	s.NoError(s.records.Flush(s.bounded()))
}

func (s *RecordFlushSuite) TestAFlushGivesUpWhenItsCallerDoes() {
	s.holdWriter()
	ctx, cancel := context.WithTimeout(s.ctx, 100*time.Millisecond)
	defer cancel()

	start := time.Now()
	s.ErrorIs(s.records.Flush(ctx), context.DeadlineExceeded)
	// The writer itself gives up on the held row only after recordWriteTimeout.
	s.Less(time.Since(start), recordWriteTimeout/2, "the Flush waited for the writer rather than its caller")
}

// holdWriter stalls the writer on a session row another transaction is inserting, until the
// returned release rolls that transaction back. The writer gives up on its own after
// recordWriteTimeout, so a test releases well before that.
func (s *RecordFlushSuite) holdWriter() (release func()) {
	held := s.row()
	tx, err := s.store.DB().BeginTx(s.ctx, nil)
	s.Require().NoError(err)
	rollback := sync.OnceFunc(func() { s.NoError(tx.Rollback()) })
	s.T().Cleanup(rollback)
	_, err = tx.ExecContext(s.ctx, "INSERT INTO agent_sessions (id, customer_id, state) VALUES (?, ?, ?)",
		held.ID, held.CustomerID, store.SessionRunning)
	s.Require().NoError(err)

	s.records.Opened(held, nil)
	s.Require().Eventually(func() bool { return len(s.records.queue) == 0 }, settleFor, 5*time.Millisecond,
		"the writer never took the held row")
	return rollback
}

// bounded is a context a broken Flush cannot hang the suite past.
func (s *RecordFlushSuite) bounded() context.Context {
	ctx, cancel := context.WithTimeout(s.ctx, 10*time.Second)
	s.T().Cleanup(cancel)
	return ctx
}

func (s *RecordFlushSuite) row() store.AgentSession {
	return store.AgentSession{ID: rand.Text(), CustomerID: s.customer, State: store.SessionRunning, CreatedAt: time.Now().UTC()}
}

// stored says whether the session's row is in Postgres. It is called from the callers'
// goroutines, so a failure to read is not stored rather than a failed check.
func (s *RecordFlushSuite) stored(id string) bool {
	rows, err := s.store.QuerySessions(s.ctx, s.customer, store.SessionFilter{Limit: 100})
	if err != nil {
		return false
	}
	for _, row := range rows {
		if row.ID == id {
			return true
		}
	}
	return false
}
