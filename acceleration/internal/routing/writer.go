package routing

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/usage"
)

const (
	// defaultQueueSize is about 20 MB of rows.
	defaultQueueSize    = 20_000
	defaultWriteTimeout = 5 * time.Second
	// batchSize is the most rows one transaction writes, and one round trip counts live.
	batchSize = 200
	// A failed batch is tried again after retryFirst, doubling up to retryMost.
	retryFirst = 100 * time.Millisecond
	retryMost  = 30 * time.Second
	// A batch that keeps failing is logged at most every errorLogEvery.
	errorLogEvery = 10 * time.Second
	// unattributedLogEvery spaces the error naming a row with no key.
	unattributedLogEvery = time.Minute
	// lookupTimeout bounds asking the gate whose one customer's rows are.
	lookupTimeout = time.Second
	// takenRetries is how many times in a row a batch is tried again at once after an id it
	// drew was taken, before it waits as after any other failure.
	takenRetries = 10
)

// Unattributed is the key a row is written under when neither a key nor an origin started it.
const Unattributed = "unattributed"

// Writer writes the recorders' rows off the request path, in two stages. The first updates the
// live counters and provider health as rows arrive, so quotas, spend and routing never wait on
// Postgres; the second writes the rows to Postgres in batches, trying again until they are
// written. A conversation never waits on either: a row is queued at once.
//
// QueueSize bounds the rows waiting across both stages. A row is lost only when they are full,
// and it still reaches the live counters; when the process stops before it could be written;
// or when Postgres refuses it on its own (invalid). Each loss is logged in full and counted.
//
// With a usage service, a row is written to be shipped, and the shard's requests table is the
// outbox: the shipper sends each written batch at once, and the sweeper whatever it could not.
type Writer struct {
	store        *store.Store
	live         *live.Client
	logger       *slog.Logger
	gate         Gate
	shipper      *shipper
	written      func([]Acked)
	writeTimeout time.Duration
	queueSize    int64

	// queue is what the first stage counts, and prepared what the second writes.
	queue    chan queued
	prepared chan queued
	// held is how many rows are kept to be written: queued, prepared or being written.
	held atomic.Int64
	// mu keeps an enqueue from sending on the queue once Close has closed it.
	mu     sync.RWMutex
	closed bool
	// ctx ends when Close's context does: whatever is not written by then is lost. closeBy is
	// that context's deadline, in Unix nanoseconds, zero until Close.
	ctx       context.Context
	cancel    context.CancelFunc
	closeBy   atomic.Int64
	done      chan struct{}
	closeOnce sync.Once

	// inFlight is how many rows are being written, and oldest when the first of them was
	// queued, in Unix nanoseconds.
	inFlight atomic.Int64
	oldest   atomic.Int64

	// When each throttled log line was last made, in Unix nanoseconds.
	errorLogged        atomic.Int64
	liveErrorLogged    atomic.Int64
	unattributedLogged atomic.Int64
	unresolvedLogged   atomic.Int64
}

// WriterOptions configures a Writer. Store and Live are optional: without either a row is
// not recorded there.
type WriterOptions struct {
	Store  *store.Store
	Live   *live.Client
	Logger *slog.Logger
	// Gate says whose a row is, its organization and that organization's tier, for a row whose
	// recorder has no gate of its own. Optional.
	Gate Gate
	// Usage is the usage service the written rows are shipped to. Nil: rows are written with
	// to_ship false and never shipped.
	Usage *usage.Client
	// Source names this process in the events it ships (see Source).
	Source string
	// Shipped is told which rows the usage service has, a row possibly more than once (see
	// Acked). Optional.
	Shipped func(acked []Acked)
	// Written is told which rows the shard has as each batch commits, a row it already had
	// included: the ledger a deployment with no usage service reads its spend from. Optional.
	Written func(written []Acked)
	// Settle is how long a written batch may wait for the shipper before it is left to the
	// sweeper, which sends what started that long ago; defaults to 30 s, the sweeper's.
	Settle time.Duration
	// QueueSize defaults to 20,000 rows.
	QueueSize int
	// WriteTimeout bounds one batch; defaults to 5 s.
	WriteTimeout time.Duration
}

// NewWriter starts the goroutines that count and write what the recorders queue.
func NewWriter(options WriterOptions) *Writer {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	size := options.QueueSize
	if size <= 0 {
		size = defaultQueueSize
	}
	timeout := options.WriteTimeout
	if timeout <= 0 {
		timeout = defaultWriteTimeout
	}
	ctx, cancel := context.WithCancel(context.Background())
	w := &Writer{
		store:        options.Store,
		live:         options.Live,
		logger:       logger,
		gate:         options.Gate,
		written:      options.Written,
		writeTimeout: timeout,
		queueSize:    int64(size),
		queue:        make(chan queued, size),
		prepared:     make(chan queued, size),
		ctx:          ctx,
		cancel:       cancel,
		done:         make(chan struct{}),
	}
	if options.Usage != nil && options.Store != nil {
		settle := options.Settle
		if settle <= 0 {
			settle = defaultSettle
		}
		w.shipper = newShipper(&shipping{
			store: options.Store, usage: options.Usage, source: options.Source, shipped: options.Shipped, logger: logger,
		}, settle)
	}
	writers.Store(w, struct{}{})
	go w.count()
	go w.write()
	return w
}

// Close stops taking rows and writes what is queued until ctx is done; what is left is lost.
// What is written and not shipped by then waits in the shard for the sweeper.
func (w *Writer) Close(ctx context.Context) {
	w.closeOnce.Do(func() {
		if deadline, ok := ctx.Deadline(); ok {
			w.closeBy.Store(deadline.UnixNano())
		}
		w.mu.Lock()
		w.closed = true
		close(w.queue)
		w.mu.Unlock()
		select {
		case <-w.done:
		case <-ctx.Done():
			w.cancel()
			<-w.done
		}
		w.cancel()
		if w.shipper != nil {
			w.shipper.close(ctx)
		}
		writers.Delete(w)
	})
}

// queued is a row, the gate its tags come from (looked up by the writer so the request path
// never does it), and what only provider health reads. keep is false for a row the stages
// have no room for, which is counted live and then lost.
type queued struct {
	request    store.Request
	gate       Gate
	fatal      bool
	unmeasured bool
	at         time.Time
	keep       bool
}

// live is what the row adds to the live counters. A row that measures nothing its provider did
// is spent as a cancelled request is, and kept out of health like one.
func (row queued) live() live.Usage {
	usage := liveUsage(row.request)
	usage.Fatal = row.fatal
	usage.Cancelled = usage.Cancelled || row.unmeasured
	return usage
}

// records reports whether a row has anywhere to go.
func (w *Writer) records() bool {
	return w.store != nil || w.live != nil
}

// enqueue queues a row without waiting, and reports whether it will be written. A row past
// QueueSize is still queued to be counted live, then lost.
func (w *Writer) enqueue(row queued) bool {
	row.at = time.Now()
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w.closed {
		w.lose([]store.Request{row.request}, lostShutdown, nil)
		return false
	}
	row.keep = w.held.Add(1) <= w.queueSize
	if !row.keep {
		w.held.Add(-1)
	}
	select {
	case w.queue <- row:
		return row.keep
	default:
		// Only Redis being slow holds the first stage back this far.
		if row.keep {
			w.held.Add(-1)
		}
		w.lose([]store.Request{row.request}, lostQueueFull, nil)
		return false
	}
}

// pending is how many rows are in memory and not written.
func (w *Writer) pending() int64 {
	return w.held.Load()
}

// waited is how long the oldest row being written has been waiting. A row still queued is
// newer than every row being written, so it is never the oldest.
func (w *Writer) waited() time.Duration {
	oldest := w.oldest.Load()
	if oldest == 0 {
		return 0
	}
	return time.Since(time.Unix(0, oldest))
}

// takeBatch takes first and whatever else is waiting on rows, up to batchSize, without waiting.
func takeBatch(first queued, rows <-chan queued) []queued {
	batch := []queued{first}
	for len(batch) < batchSize {
		select {
		case row, ok := <-rows:
			if !ok {
				return batch
			}
			batch = append(batch, row)
		default:
			return batch
		}
	}
	return batch
}

// count is the first stage: the live counters, as rows arrive, then the rows kept to the second.
func (w *Writer) count() {
	defer close(w.prepared)
	for first := range w.queue {
		batch := takeBatch(first, w.queue)
		w.countLive(batch)
		var lost []store.Request
		for _, row := range batch {
			switch {
			case !row.keep:
				lost = append(lost, row.request)
			case w.store == nil:
				w.held.Add(-1)
			default:
				// Never waits: prepared has room for every row held.
				w.prepared <- row
			}
		}
		if len(lost) > 0 {
			w.lose(lost, lostQueueFull, nil)
		}
	}
}

// countLive adds the rows to the live counters and provider health in one round trip.
func (w *Writer) countLive(batch []queued) {
	if w.live == nil {
		return
	}
	usages := make([]live.Usage, len(batch))
	for i, row := range batch {
		usages[i] = row.live()
	}
	ctx, cancel := context.WithTimeout(w.ctx, w.writeTimeout)
	defer cancel()
	if err := w.live.RecordRequests(ctx, usages...); err != nil && w.throttled(&w.liveErrorLogged, errorLogEvery) {
		w.logger.Error("could not update live counters", "rows", len(batch), "error", err)
	}
}

// write is the second stage: the rows to Postgres, a batch at a time.
func (w *Writer) write() {
	defer close(w.done)
	for first := range w.prepared {
		batch := takeBatch(first, w.prepared)
		w.oldest.Store(batch[0].at.UnixNano())
		w.inFlight.Store(int64(len(batch)))
		w.writeBatch(batch)
		w.inFlight.Store(0)
		w.oldest.Store(0)
		w.held.Add(-int64(len(batch)))
	}
}

// writeBatch writes the rows until they are written or Close gives up on them.
func (w *Writer) writeBatch(batch []queued) {
	rows := make([]pending, len(batch))
	for i, next := range batch {
		rows[i] = pending{request: w.prepare(next.request), gate: next.gate}
		if next.gate == nil {
			rows[i].gate = w.gate
		}
	}
	if w.ctx.Err() != nil {
		w.lose(requestsOf(rows), lostShutdown, nil)
		return
	}
	w.writeRows(rows)
}

// pending is a row being written, the gate that says whose it is, and whether it has.
type pending struct {
	request    store.Request
	gate       Gate
	attributed bool
}

func requestsOf(rows []pending) []store.Request {
	requests := make([]store.Request, len(rows))
	for i, row := range rows {
		requests[i] = row.request
	}
	return requests
}

// attempt bounds one try at a batch by WriteTimeout, and by Close's deadline once it has one:
// a Postgres query stops at its context's deadline, not when the context is cancelled.
func (w *Writer) attempt() (context.Context, context.CancelFunc) {
	deadline := time.Now().Add(w.writeTimeout)
	if by := w.closeBy.Load(); by != 0 && by < deadline.UnixNano() {
		deadline = time.Unix(0, by)
	}
	return context.WithDeadline(w.ctx, deadline)
}

// prepare fills what a row is written with when its recorder left it empty.
func (w *Writer) prepare(request store.Request) store.Request {
	// With no usage service, nothing would ever ship the row.
	request.ToShip = w.shipper != nil
	if request.APIKey == "" {
		request.APIKey = Unattributed
		usageUnattributed.Inc()
		// Every site that owns work names a key or an origin, so this is one that was missed;
		// the row is what tells which.
		if w.throttled(&w.unattributedLogged, unattributedLogEvery) {
			w.logger.Error("usage row recorded with no api key or origin",
				"usage_id", request.UsageID, "customer_id", request.CustomerID, "modality", request.Modality,
				"provider", request.Provider, "model", request.Model, "purpose", request.Purpose,
				"agent", request.AgentID, "session", request.SessionID)
		}
	}
	if request.OrganizationTier == "" {
		request.OrganizationTier = usage.DefaultTier
	}
	// A recorder's row has had RecordedUser applied; one that slipped past would be refused by
	// the usage service whole, so it is never written with the id.
	request.UserID = RecordedUser(request.UserID)
	return request
}

// Attributor is a Gate that can say whose work a row is after the work, without keeping a
// failed lookup as the answer the next request is admitted on. The writer asks one in place
// of Admit.
type Attributor interface {
	Attribute(ctx context.Context, customerID string) (Admission, error)
}

// attribute lays what the gate says over each row it has not answered for yet: the tags, and
// the organization when it knows one. It is asked here rather than with the live counters
// because it reads Postgres, which the counters must not wait on, and again before every try
// the batch gets. Each customer of a gate is asked once, each lookup bounded on its own. Once
// one has run out of time, an Attributor is still asked for the rest, with the context ended,
// so what it has cached still applies and only a customer it would have to read fails, rather
// than each lookup running out of time of its own; another gate is not asked. It returns how
// many rows a lookup failed for: when the write then succeeds they are written without their
// organization, and counted in router_usage_unresolved_total.
func (w *Writer) attribute(rows []pending) int {
	ctx, cancel := w.attempt()
	defer cancel()
	type asked struct {
		gate     Gate
		customer string
	}
	type answer struct {
		admission Admission
		err       error
	}
	answers := map[asked]answer{}
	failed := 0
	for i := range rows {
		row := &rows[i]
		if row.gate == nil || row.attributed {
			continue
		}
		if _, attributes := row.gate.(Attributor); !attributes && ctx.Err() != nil {
			failed++
			continue
		}
		// Only a pointer gate is a map key: another may hold what cannot be compared.
		key, shared := asked{gate: row.gate, customer: row.request.CustomerID}, reflect.ValueOf(row.gate).Kind() == reflect.Pointer
		got, known := answer{}, false
		if shared {
			got, known = answers[key]
		}
		if !known {
			got.admission, got.err = w.lookup(ctx, row.gate, row.request.CustomerID)
			if shared {
				answers[key] = got
			}
			if errors.Is(got.err, context.DeadlineExceeded) {
				cancel()
			}
		}
		if got.err != nil {
			failed++
			continue
		}
		row.attributed = true
		row.request.Tags = got.admission.Labelled(row.request.Tags)
		if got.admission.OrganizationID != "" {
			row.request.OrganizationID = got.admission.OrganizationID
		}
		if got.admission.OrganizationTier != "" {
			row.request.OrganizationTier = got.admission.OrganizationTier
		}
	}
	return failed
}

// lookup asks a gate whose a customer's work is, within lookupTimeout.
func (w *Writer) lookup(ctx context.Context, gate Gate, customerID string) (Admission, error) {
	ctx, cancel := context.WithTimeout(ctx, lookupTimeout)
	defer cancel()
	if attributor, ok := gate.(Attributor); ok {
		admission, err := attributor.Attribute(ctx, customerID)
		if err != nil && ctx.Err() != nil {
			// A driver need not say it was the deadline.
			err = errors.Join(err, ctx.Err())
		}
		return admission, err
	}
	// A refusal is the budget, which the work being recorded has already got past.
	admission, _ := gate.Admit(ctx, Owner{CustomerID: customerID})
	return admission, nil
}

// writeRows writes rows until they are written, trying again after a failure that may pass
// and one row at a time after one that cannot.
func (w *Writer) writeRows(rows []pending) {
	delay := retryFirst
	taken := 0
	for {
		unknown := w.attribute(rows)
		requests := requestsOf(rows)
		ctx, cancel := w.attempt()
		written, err := w.store.RecordRequests(ctx, requests)
		cancel()
		if err == nil || errors.Is(err, store.ErrRequestLogsMissing) {
			if err != nil {
				w.logger.Warn("usage rows written without some of their agent log rows", "error", err)
			}
			if unknown > 0 {
				usageUnresolved.Add(float64(unknown))
				if w.throttled(&w.unresolvedLogged, errorLogEvery) {
					w.logger.Warn("usage rows written without their organization, which could not be looked up", "rows", unknown)
				}
			}
			usageWritten.Add(float64(len(written)))
			usageDuplicates.Add(float64(len(rows) - len(written)))
			if w.written != nil {
				w.written(ackedOf(requests))
			}
			if w.shipper != nil {
				w.shipper.offer(writtenRows(requests, written))
			}
			return
		}
		if store.Unwritable(err) {
			// One row Postgres refuses must not keep the rest of the batch out.
			if len(rows) > 1 {
				for i := range rows {
					w.writeRows(rows[i : i+1])
				}
				return
			}
			w.lose(requests, lostInvalid, err)
			return
		}
		// RecordRequests has moved the sequence past the id, so the next try draws a free one.
		if store.IDTaken(err) && taken < takenRetries && w.ctx.Err() == nil {
			taken++
			continue
		}

		usageWriteErrors.Inc()
		if w.throttled(&w.errorLogged, errorLogEvery) {
			w.logger.Error("could not write usage rows, trying again", "rows", len(rows), "retry_in", delay, "error", err)
		}
		select {
		case <-time.After(delay):
		case <-w.ctx.Done():
			w.lose(requests, lostShutdown, err)
			return
		}
		delay = min(delay*2, retryMost)
		taken = 0
	}
}

// ackedOf is the rows as the shard has them.
func ackedOf(requests []store.Request) []Acked {
	acked := make([]Acked, len(requests))
	for i, request := range requests {
		acked[i] = Acked{UsageID: request.UsageID, OrganizationID: request.OrganizationID, CostMicros: request.CostMicros}
	}
	return acked
}

// writtenRows is the rows a batch wrote, each usage id once. A row whose id was already in the
// shard is left out: the row there is the one to ship, and the sweeper sends it if it has to.
func writtenRows(requests []store.Request, written []uuid.UUID) []store.Request {
	ids := make(map[uuid.UUID]bool, len(written))
	for _, id := range written {
		ids[id] = true
	}
	rows := make([]store.Request, 0, len(written))
	for _, request := range requests {
		if ids[request.UsageID] {
			delete(ids, request.UsageID)
			rows = append(rows, request)
		}
	}
	return rows
}

// lose logs every row in full, which is what replaying them reads, and counts them, which is
// what the lost alert reads.
func (w *Writer) lose(rows []store.Request, reason string, cause error) {
	for _, request := range rows {
		cost := max(request.CostMicros, 0)
		usageLost.WithLabelValues(reason).Inc()
		usageLostMicros.Add(float64(cost))
		row, err := json.Marshal(request)
		if err != nil {
			w.logger.Error("usage row lost", "reason", reason, "usage_id", request.UsageID, "encode_error", err)
			continue
		}
		args := []any{"reason", reason, "usage_id", request.UsageID, "row", string(row)}
		if cause != nil {
			args = append(args, "error", cause)
		}
		w.logger.Error("usage row lost", args...)
	}
}

// throttled reports whether every has passed since last, and if so moves last to now.
func (w *Writer) throttled(last *atomic.Int64, every time.Duration) bool {
	now := time.Now().UnixNano()
	previous := last.Load()
	if previous != 0 && now-previous < every.Nanoseconds() {
		return false
	}
	return last.CompareAndSwap(previous, now)
}
