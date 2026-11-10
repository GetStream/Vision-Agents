// Package activity counts the requests each app makes, by area of the API and by client, and
// writes the sums to customer_activity (store.RecordActivity).
//
// Every authenticated request is one, so the counting is in memory and the writing is not on
// the request path: a minute's requests for one app, area, client and day are one row's
// update. Nothing here fails a request: counting it is not what the caller asked for.
package activity

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The areas of the API a request is counted under.
const (
	AreaDashboardReads = "dashboard_reads"
	AreaAgentSetup     = "agent_setup"
	AreaConversations  = "conversations"
	AreaWorkers        = "workers"
	AreaVoiceVideo     = "voice_video"
	AreaModelAPI       = "model_api"
	AreaPhoneSMS       = "phone_sms"
	AreaConnectors     = "connectors"
	AreaScheduling     = "scheduling"
	AreaEvals          = "evals"
	AreaData           = "data"
	// AreaHooksIn is a delivery a vendor or Stream sends to a hook.
	AreaHooksIn = "hooks_in"
	// AreaWebhooksOut is a delivery the router sends to a customer's own endpoint.
	AreaWebhooksOut = "webhooks_out"
	// AreaOther is a route no area names, so a new route is counted rather than lost.
	AreaOther = "other"
)

// DefaultInterval is how often the sums are written. A row is then written at most once a
// minute per app, area, client and day, which is the rate the activity record was asked to
// keep to; a router that stops loses at most that much, and not even that when it stops
// cleanly (Close).
const DefaultInterval = time.Minute

// DefaultCapacity is how many rows one process holds between writes. There are 14 areas and
// 4 clients, so it is every area and client of about 180 apps in one minute. A judgement, not
// a measurement: past it a request is dropped and counted, and the write is brought forward.
const DefaultCapacity = 10_000

// writeTimeout bounds one write, well inside the interval so a slow one is over before the
// next is due.
const writeTimeout = 10 * time.Second

// Request is one request to count.
type Request struct {
	OrganizationID string
	AppID          string
	Area           string
	// Client is one of the audit sources a client may name itself as (store.AuditSource).
	Client string
	At     time.Time
	// Refused is an answer of 400 or more, and Code the error code it carried, if any.
	Refused bool
	Code    string
}

// Options configures a Recorder.
type Options struct {
	// Store is where the sums are written. Required.
	Store *store.Store
	// Interval is how often they are. DefaultInterval when unset.
	Interval time.Duration
	// Capacity is how many rows are held between writes. DefaultCapacity when unset.
	Capacity int
	Logger   *slog.Logger
}

// Recorder sums requests in memory and writes them every interval, and once more on Close.
type Recorder struct {
	db       *store.Store
	interval time.Duration
	capacity int
	logger   *slog.Logger

	mu      sync.Mutex
	pending map[key]*store.ActivityDelta
	// dropped is how many requests found no room since the last write.
	dropped int64

	// full asks the writer to write now, when pending has reached capacity.
	full      chan struct{}
	stop      chan struct{}
	done      chan struct{}
	started   atomic.Bool
	closeOnce sync.Once
}

// key is the row a request is counted in.
type key struct {
	appID, area, client string
	day                 time.Time
}

// New returns a Recorder that holds what it is given until it is started.
func New(options Options) (*Recorder, error) {
	if options.Store == nil {
		return nil, errors.New("activity: a store is required")
	}
	if options.Interval <= 0 {
		options.Interval = DefaultInterval
	}
	if options.Capacity <= 0 {
		options.Capacity = DefaultCapacity
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	return &Recorder{
		db:       options.Store,
		interval: options.Interval,
		capacity: options.Capacity,
		logger:   options.Logger,
		pending:  map[key]*store.ActivityDelta{},
		full:     make(chan struct{}, 1),
		stop:     make(chan struct{}),
		done:     make(chan struct{}),
	}, nil
}

// Start begins writing every interval, until Close. It does not stop with a context, so the
// requests a router serves while it drains are counted too.
func (r *Recorder) Start() {
	if !r.started.CompareAndSwap(false, true) {
		return
	}
	go r.run()
}

// Close stops the writer and writes what is still held.
func (r *Recorder) Close() {
	r.closeOnce.Do(func() {
		close(r.stop)
		if r.started.Load() {
			<-r.done
		}
		r.write()
	})
}

// Record counts one request. A request naming no app is nobody's and is not counted.
func (r *Recorder) Record(request Request) {
	if request.AppID == "" {
		return
	}
	at := request.At.UTC()
	delta := store.ActivityDelta{
		OrganizationID: request.OrganizationID,
		AppID:          request.AppID,
		Area:           request.Area,
		Client:         request.Client,
		Day:            time.Date(at.Year(), at.Month(), at.Day(), 0, 0, 0, 0, time.UTC),
		Requests:       1,
		FirstAt:        at,
		LastAt:         at,
	}
	if request.Refused {
		delta.Refused = 1
		delta.LastRefusalCode = request.Code
		delta.LastRefusalAt = at
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	r.keep(delta)
	if len(r.pending) >= r.capacity {
		select {
		case r.full <- struct{}{}:
		default:
		}
	}
}

// keep adds delta to what is held, or drops it when there is no room for another row. The
// caller holds mu.
func (r *Recorder) keep(delta store.ActivityDelta) {
	row := key{delta.AppID, delta.Area, delta.Client, delta.Day}
	if held, found := r.pending[row]; found {
		held.Add(delta)
		return
	}
	if len(r.pending) >= r.capacity {
		r.dropped += delta.Requests
		return
	}
	r.pending[row] = &delta
}

func (r *Recorder) run() {
	defer close(r.done)
	ticker := time.NewTicker(r.interval)
	defer ticker.Stop()
	for {
		select {
		case <-r.stop:
			return
		case <-ticker.C:
		case <-r.full:
		}
		r.write()
	}
}

// write writes what is held. What a failed write held is kept for the next one, as far as
// there is room: a database that did not answer this minute may next minute.
func (r *Recorder) write() {
	r.mu.Lock()
	pending, dropped := r.pending, r.dropped
	r.pending, r.dropped = map[key]*store.ActivityDelta{}, 0
	r.mu.Unlock()

	if dropped > 0 {
		r.logger.Warn("dropped requests from the activity record: more rows than it holds",
			"dropped", dropped, "capacity", r.capacity)
	}
	if len(pending) == 0 {
		return
	}
	deltas := make([]store.ActivityDelta, 0, len(pending))
	for _, delta := range pending {
		deltas = append(deltas, *delta)
	}
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()
	if err := r.db.RecordActivity(ctx, deltas); err != nil {
		r.logger.Error("could not write the activity record", "rows", len(deltas), "error", err)
		r.mu.Lock()
		defer r.mu.Unlock()
		for _, delta := range deltas {
			r.keep(delta)
		}
	}
}
