package routing

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"maps"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	"github.com/getsentry/sentry-go"

	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
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
	// A batch that keeps failing is logged at most every errorLogEvery, and failures and losses
	// reach Sentry at most every reportEvery each.
	errorLogEvery = 10 * time.Second
	reportEvery   = time.Minute
	// unattributedLogEvery spaces the warning naming a row with no key, which before every
	// owner names one is every row.
	unattributedLogEvery = time.Minute
	// lookupTimeout bounds asking the gate whose one customer's rows are.
	lookupTimeout = time.Second
	// takenRetries is how many times in a row a batch is tried again at once after an id it
	// drew was taken, before it waits as after any other failure.
	takenRetries = 10
)

// unattributed is the key a row is written under when neither a key nor an origin started it.
const unattributed = "unattributed"

// defaultTier is the tier a row is written under when its organization's is not known: the
// only tier name the router knows.
const defaultTier = "default"

// Writer writes the recorders' rows off the request path, in two stages. The first updates the
// live counters and provider health as rows arrive, so quotas, spend and routing never wait on
// Postgres; the second writes the rows to Postgres in batches, trying again until they are
// written. A conversation never waits on either: a row is queued at once.
//
// QueueSize bounds the rows waiting across both stages. A row is lost only when they are full,
// and it still reaches the live counters; when the process stops before it could be written;
// or when Postgres refuses it on its own (invalid). Each loss is logged in full and counted.
type Writer struct {
	store        *store.Store
	live         *live.Client
	logger       *slog.Logger
	hub          *sentry.Hub
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

	// When each throttled log line and report was last made, in Unix nanoseconds.
	errorLogged        atomic.Int64
	liveErrorLogged    atomic.Int64
	errorReported      atomic.Int64
	lossReported       atomic.Int64
	unattributedLogged atomic.Int64
	unresolvedLogged   atomic.Int64
	// lost is what was lost and not reported yet.
	lost lossTally
}

// WriterOptions configures a Writer. Store and Live are optional: without either a row is
// not recorded there.
type WriterOptions struct {
	Store  *store.Store
	Live   *live.Client
	Logger *slog.Logger
	// Hub is where failed writes and lost rows are reported. Nil is Sentry's current hub.
	Hub *sentry.Hub
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
	hub := options.Hub
	if hub == nil {
		hub = sentry.CurrentHub()
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
		hub:          hub,
		writeTimeout: timeout,
		queueSize:    int64(size),
		queue:        make(chan queued, size),
		prepared:     make(chan queued, size),
		ctx:          ctx,
		cancel:       cancel,
		done:         make(chan struct{}),
	}
	writers.Store(w, struct{}{})
	go w.count()
	go w.write()
	return w
}

// Close stops taking rows and writes what is queued until ctx is done; what is left is lost.
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
		writers.Delete(w)
		// What the last minute lost is not left waiting for a report the process may not live
		// to send.
		w.lost.stop()
		w.reportLosses()
	})
}

// queued is a row, the gate its tags come from (looked up by the writer so the request path
// never does it), and what only provider health reads. keep is false for a row the stages
// have no room for, which is counted live and then lost.
type queued struct {
	request store.Request
	gate    Gate
	fatal   bool
	at      time.Time
	keep    bool
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
		usages[i] = liveUsage(row.request)
		usages[i].Fatal = row.fatal
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
	request.ToShip = false
	if request.APIKey == "" {
		request.APIKey = unattributed
		usageUnattributed.Inc()
		if w.throttled(&w.unattributedLogged, unattributedLogEvery) {
			w.logger.Warn("usage row recorded with no api key or origin",
				"modality", request.Modality, "provider", request.Provider,
				"purpose", request.Purpose, "agent", request.AgentID)
		}
	}
	if request.OrganizationTier == "" {
		request.OrganizationTier = defaultTier
	}
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
	admission, _ := gate.Admit(ctx, customerID)
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
		if w.throttled(&w.errorReported, reportEvery) {
			event := sentry.NewEvent()
			event.Level = sentry.LevelError
			event.Message = "could not write usage rows"
			event.Fingerprint = []string{"usage_write_failed"}
			event.Tags = map[string]string{"component": "usage_writer"}
			event.Contexts = map[string]sentry.Context{"usage": {"rows": len(rows), "error": err.Error()}}
			w.hub.CaptureEvent(event)
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

// lose logs every row in full, which is what replaying them reads, counts them, and reports
// the loss: at once if nothing was reported in the last reportEvery, otherwise once it has
// passed, with everything lost meanwhile.
func (w *Writer) lose(rows []store.Request, reason string, cause error) {
	var micros int64
	for _, request := range rows {
		cost := max(request.CostMicros, 0)
		micros += cost
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
	w.lost.add(reason, len(rows), micros)
	if w.throttled(&w.lossReported, reportEvery) {
		w.reportLosses()
		return
	}
	w.lost.later(time.Unix(0, w.lossReported.Load()).Add(reportEvery), w.reportLosses)
}

// reportLosses sends what was lost since the last report to Sentry, if anything was.
func (w *Writer) reportLosses() {
	byReason, micros := w.lost.take()
	if len(byReason) == 0 {
		return
	}
	w.lossReported.Store(time.Now().UnixNano())
	rows := 0
	reasons := make(map[string]any, len(byReason))
	for reason, count := range byReason {
		rows += count
		reasons[reason] = count
	}
	event := sentry.NewEvent()
	event.Level = sentry.LevelError
	event.Message = "usage rows lost"
	event.Fingerprint = []string{"usage_alert", "lost"}
	event.Tags = map[string]string{"usage_alert": "lost"}
	if len(byReason) == 1 {
		for reason := range byReason {
			event.Tags["reason"] = reason
		}
	}
	event.Contexts = map[string]sentry.Context{
		"usage":          {"rows": rows, "cost_micros": micros},
		"lost_by_reason": reasons,
	}
	w.hub.CaptureEvent(event)
}

// lossTally is what was lost since it was last reported, and the report waiting to send it.
type lossTally struct {
	mu      sync.Mutex
	rows    map[string]int
	micros  int64
	pending *time.Timer
	stopped bool
}

func (t *lossTally) add(reason string, rows int, micros int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.rows == nil {
		t.rows = map[string]int{}
	}
	t.rows[reason] += rows
	t.micros += micros
}

// take empties the tally and returns what it held. A report waiting to send it is not needed
// any more.
func (t *lossTally) take() (map[string]int, int64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	rows, micros := maps.Clone(t.rows), t.micros
	clear(t.rows)
	t.micros = 0
	if t.pending != nil {
		t.pending.Stop()
		t.pending = nil
	}
	return rows, micros
}

// later runs report at at, unless a report is already waiting or the tally is stopped.
func (t *lossTally) later(at time.Time, report func()) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.pending != nil || t.stopped {
		return
	}
	t.pending = time.AfterFunc(time.Until(at), report)
}

// stop keeps any report from waiting again; whatever is lost is left to take.
func (t *lossTally) stop() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.stopped = true
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
