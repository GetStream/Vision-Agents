package session

import (
	"context"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// recordQueueSize bounds how far the session writer may fall behind before rows are
// dropped. Larger than the call queue because items arrive per turn rather than per call.
const recordQueueSize = 1024

// recordWriteTimeout bounds a single write so a stuck database cannot wedge the writer.
const recordWriteTimeout = 5 * time.Second

// itemFlushEvery is how long a partial batch of items waits for company.
//
// Items arrive in bursts -- a question, then nothing while a tool runs, then several at once
// -- so batching only on a full buffer would leave the last few items of a quiet turn
// unwritten until the next turn happened. Short enough that a reader refreshing the page
// sees the turn they just watched.
const itemFlushEvery = 250 * time.Millisecond

// itemBatchSize is how many items are written in one statement.
const itemBatchSize = 64

// recordWrite is one thing that happened to a session that Postgres should hear about.
//
// A single queue rather than one per kind, because the order matters and separate queues
// would race: a response row written before the session row it points at violates the
// foreign key, and items written before their response do the same.
type recordWrite struct {
	// session is a session that opened or was renamed, nil otherwise.
	session *store.AgentSession
	// closed is the session that ended, and closedAt when.
	closed   string
	closedAt time.Time
	// response is a turn that began, nil otherwise.
	response *store.AgentResponse
	// finished is the response that ended, with how it ended.
	finished   string
	status     string
	failure    string
	finishedAt time.Time
	// items are things that happened during a turn.
	items []store.AgentResponseItem
	// flushed is closed once everything queued before it has been written.
	flushed chan struct{}
}

// recorder is what a session needs of the writer behind it, which is less than the writer
// offers: starting and stopping it belongs to the manager that owns it. Narrowing it here
// also says plainly what a session may record, which is the list a reader of the event
// mapping wants.
type recorder interface {
	// Responding says a turn began.
	Responding(row store.AgentResponse)
	// Responded says how it ended.
	Responded(id, status, failure string, at time.Time)
	// Item says one thing that happened during it.
	Item(item store.AgentResponseItem)
	// Flush waits until everything said so far has been written, which is what reading the
	// conversation back straight after it happened needs.
	Flush(ctx context.Context) error
}

// sessionRecorder writes sessions, turns and items to Postgres off the conversation's path.
//
// Same trade as the call recorder and for the same reason: a caller must never wait on a
// database to be answered, and a conversation must not fail to happen because the row
// recording it could not be written. So recording is asynchronous and rows are what gets
// dropped when the writer cannot keep up.
//
// An incognito session never reaches this writer at all. That is deliberate: a flag checked
// here would be one place, and one place that forgot to check it would be a conversation
// kept against its caller's wishes. The session simply never hands anything over.
type sessionRecorder struct {
	store  *store.Store
	logger *slog.Logger

	queue chan recordWrite
	done  chan struct{}

	closeOnce sync.Once
	dropped   atomic.Int64
}

func newSessionRecorder(pgStore *store.Store, logger *slog.Logger) *sessionRecorder {
	r := &sessionRecorder{
		store:  pgStore,
		logger: logger,
		queue:  make(chan recordWrite, recordQueueSize),
		done:   make(chan struct{}),
	}
	go r.run()
	return r
}

// Opened queues the row for a session that has just started.
func (r *sessionRecorder) Opened(row store.AgentSession) {
	r.queueWrite(recordWrite{session: &row})
}

// Closed queues the time a session ended.
func (r *sessionRecorder) Closed(id string, at time.Time) {
	r.queueWrite(recordWrite{closed: id, closedAt: at})
}

// Responding queues a turn that has just begun.
func (r *sessionRecorder) Responding(row store.AgentResponse) {
	r.queueWrite(recordWrite{response: &row})
}

// Responded queues how a turn ended.
func (r *sessionRecorder) Responded(id, status, failure string, at time.Time) {
	r.queueWrite(recordWrite{finished: id, status: status, failure: failure, finishedAt: at})
}

// Item queues one thing that happened during a turn.
func (r *sessionRecorder) Item(item store.AgentResponseItem) {
	r.queueWrite(recordWrite{items: []store.AgentResponseItem{item}})
}

// Flush waits for the writer to catch up with everything queued before it. Unlike the other
// writes it waits for room rather than being dropped, because the caller is about to read
// what it wrote.
func (r *sessionRecorder) Flush(ctx context.Context) error {
	flushed := make(chan struct{})
	select {
	case r.queue <- recordWrite{flushed: flushed}:
	case <-ctx.Done():
		return ctx.Err()
	}
	select {
	case <-flushed:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// Dropped reports how many writes were thrown away, for a test or a health check that wants
// to know the writer is keeping up.
func (r *sessionRecorder) Dropped() int64 { return r.dropped.Load() }

// Close drains the queue and stops the writer.
func (r *sessionRecorder) Close() {
	r.closeOnce.Do(func() {
		close(r.queue)
		<-r.done
		if dropped := r.dropped.Load(); dropped > 0 {
			r.logger.Warn("dropped session records because the writer fell behind", "count", dropped)
		}
	})
}

func (r *sessionRecorder) queueWrite(write recordWrite) {
	select {
	case r.queue <- write:
	default:
		r.dropped.Add(1)
	}
}

func (r *sessionRecorder) run() {
	defer close(r.done)

	// Items are the only writes that batch, so they collect here while everything else goes
	// straight through. Anything that is not an item flushes them first: a response marked
	// finished before its items were written would be read back as a turn that did nothing.
	pending := make([]store.AgentResponseItem, 0, itemBatchSize)
	flush := time.NewTicker(itemFlushEvery)
	defer flush.Stop()

	for {
		select {
		case write, open := <-r.queue:
			if !open {
				r.writeItems(pending)
				return
			}
			if len(write.items) > 0 {
				pending = append(pending, write.items...)
				if len(pending) >= itemBatchSize {
					r.writeItems(pending)
					pending = pending[:0]
				}
				continue
			}
			if len(pending) > 0 {
				r.writeItems(pending)
				pending = pending[:0]
			}
			r.write(write)
		case <-flush.C:
			if len(pending) > 0 {
				r.writeItems(pending)
				pending = pending[:0]
			}
		}
	}
}

func (r *sessionRecorder) write(write recordWrite) {
	ctx, cancel := context.WithTimeout(context.Background(), recordWriteTimeout)
	defer cancel()

	switch {
	case write.flushed != nil:
		// The items pending ahead of it were written before this was reached.
		close(write.flushed)
	case write.session != nil:
		if err := r.store.SaveSession(ctx, write.session); err != nil {
			r.logger.Error("could not record the session starting", "error", err)
		}
	case write.closed != "":
		if err := r.store.CloseSession(ctx, write.closed, write.closedAt); err != nil {
			r.logger.Error("could not record the session ending", "error", err)
		}
	case write.response != nil:
		if err := r.store.StartResponse(ctx, write.response); err != nil {
			r.logger.Error("could not record the turn starting", "error", err)
		}
	case write.finished != "":
		if err := r.store.FinishResponse(ctx, write.finished, write.status, write.failure, write.finishedAt); err != nil {
			r.logger.Error("could not record the turn ending", "error", err)
		}
	}
}

func (r *sessionRecorder) writeItems(items []store.AgentResponseItem) {
	if len(items) == 0 {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), recordWriteTimeout)
	defer cancel()

	// The slice is copied because the caller reuses its buffer, and the write outlives the
	// call that handed it over only by as long as this statement takes.
	batch := append([]store.AgentResponseItem(nil), items...)
	if err := r.store.AppendResponseItems(ctx, batch); err != nil {
		r.logger.Error("could not record what a turn did", "error", err, "items", len(batch))
	}
}

// sessionRow is what a session looks like to somebody reading it back later.
//
// Deliberately not the call row. That one is a call: it carries numbers, a direction and the
// review a finished call gets, none of which a text conversation has. This one carries what
// the caller said about the conversation, which is what they will look for it by.
func sessionRow(created *Session) store.AgentSession {
	spec := created.spec
	row := store.AgentSession{
		ID:              created.id,
		CustomerID:      spec.CustomerID,
		ConfigID:        spec.ConfigID,
		AgentName:       spec.AgentName,
		AgentID:         spec.AgentID,
		ConversationID:  spec.ConversationID,
		CallID:          spec.CallID,
		CallType:        spec.CallType,
		Title:           spec.Title,
		Description:     spec.Description,
		Project:         spec.Project,
		Custom:          spec.Custom,
		ModelOverwrites: spec.ModelOverwrites,
		ForkedFrom:      spec.ForkedFrom,
		State:           store.SessionRunning,
		CreatedAt:       created.created.UTC(),
	}
	// Whose the session is comes from the credential rather than from the spec's UserID,
	// which is who the agent joined the call as. Recording the agent's own id as the owner
	// would make every session look like it belonged to the agent.
	row.UserID = spec.Caller.UserID
	row.CallerKind = string(spec.CallerKind)
	return row
}
