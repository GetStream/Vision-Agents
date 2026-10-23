package routing

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// grantCountQueue is how many written batches wait at most to be counted toward their grants.
const grantCountQueue = 1_000

var (
	grantCountErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_grant_count_errors_total",
		Help:      "Attempts to count written usage rows toward their grants in the shared database that failed and are tried again",
	})
	grantCountsDropped = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_grant_counts_dropped_total",
		Help:      "Written usage rows the grant counter had no room for, or could not count before the process stopped, which the reconcile counts",
	})
)

// grantCounter counts the rows a shard took toward their grants in the shared database, on a
// goroutine of its own, so a slow or unreachable shared database never holds up writing rows to
// the shards. Each batch is tried until it lands, and then counted is told its rows. The queue is
// bounded: a batch past it is dropped, logged at ERROR and left to the reconcile, as is what is
// left when the process stops. Until counted hears of a row, or the policies find it counted by
// the reconcile, they keep what they hold against its grants.
type grantCounter struct {
	store   *store.Store
	logger  *slog.Logger
	counted func([]Acked)
	timeout time.Duration

	queue chan countJob
	// mu keeps an offer from sending on the queue once close has closed it.
	mu     sync.RWMutex
	closed bool
	ctx    context.Context
	cancel context.CancelFunc
	done   chan struct{}

	logged throttle
}

func newGrantCounter(db *store.Store, counted func([]Acked), timeout time.Duration, logger *slog.Logger) *grantCounter {
	ctx, cancel := context.WithCancel(context.Background())
	c := &grantCounter{
		store: db, logger: logger, counted: counted, timeout: timeout,
		queue: make(chan countJob, grantCountQueue), ctx: ctx, cancel: cancel, done: make(chan struct{}),
	}
	go c.run()
	return c
}

// countJob is one written batch's uses, and the rows counted is told of once they land.
type countJob struct {
	uses []store.GrantUse
	rows []Acked
}

// offer queues the uses to be counted, and never waits. With no uses the rows have nothing to
// count, and counted is told at once.
func (c *grantCounter) offer(uses []store.GrantUse, rows []Acked) {
	if len(uses) == 0 {
		if c.counted != nil && len(rows) > 0 {
			c.counted(rows)
		}
		return
	}
	c.mu.RLock()
	queued := false
	if !c.closed {
		select {
		case c.queue <- countJob{uses: uses, rows: rows}:
			queued = true
		default:
		}
	}
	c.mu.RUnlock()
	if queued {
		return
	}
	grantCountsDropped.Add(float64(len(uses)))
	if c.logged.due("dropped", errorLogEvery) {
		c.logger.Error("could not queue usage rows to be counted toward their grants, which the reconcile counts",
			"uses", len(uses), "queued", len(c.queue))
	}
}

func (c *grantCounter) run() {
	defer close(c.done)
	for job := range c.queue {
		c.count(job)
	}
}

// count counts one batch, trying again until it lands or the counter is stopped.
func (c *grantCounter) count(job countJob) {
	uses := job.uses
	delay := retryFirst
	for {
		ctx, cancel := context.WithTimeout(c.ctx, c.timeout)
		_, err := c.store.AddGrantUses(ctx, uses)
		cancel()
		if err == nil {
			if c.counted != nil {
				c.counted(job.rows)
			}
			return
		}
		grantCountErrors.Inc()
		if c.logged.due("failed", errorLogEvery) {
			c.logger.Error("could not count usage rows toward their grants, trying again", "uses", len(uses), "retry_in", delay, "error", err)
		}
		select {
		case <-time.After(delay):
		case <-c.ctx.Done():
			grantCountsDropped.Add(float64(len(uses)))
			c.logger.Warn("usage rows written and not counted toward their grants, which the reconcile counts", "uses", len(uses))
			return
		}
		delay = min(delay*2, retryMost)
	}
}

// close stops taking batches and counts what is queued until ctx is done; what is left is
// dropped, for the reconcile.
func (c *grantCounter) close(ctx context.Context) {
	c.mu.Lock()
	if c.closed {
		c.mu.Unlock()
		return
	}
	c.closed = true
	close(c.queue)
	c.mu.Unlock()
	select {
	case <-c.done:
	case <-ctx.Done():
		c.cancel()
		<-c.done
	}
	c.cancel()
}