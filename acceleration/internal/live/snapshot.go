package live

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
	"github.com/redis/rueidis"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

// DefaultRefresh is how often a router reads every model's health again.
const DefaultRefresh = time.Second

// refreshTimeout bounds one refresh, so a slow Redis leaves the last snapshot answering rather
// than piling refreshes up behind it.
const refreshTimeout = time.Second

// warnEvery spaces the warning a failing refresh logs.
const warnEvery = time.Minute

// The reads healthReads tells apart.
const (
	readRefresh = "refresh"
	readMiss    = "miss"
)

// clients is every Client of this process with a snapshot, which the age gauge looks at.
var clients sync.Map

var (
	healthReads = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "health_reads_total",
		Help:      "Round trips to Redis for provider health: the background refresh, or a walk's models missing from the snapshot, read together",
	}, []string{"kind"})
	healthRefreshErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "health_refresh_errors_total",
		Help:      "Background reads of provider health that failed and left the last snapshot answering",
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "health_snapshot_age_seconds",
		Help:      "How long ago provider health was last read in full; past the health window a model reads as unmeasured",
	}, snapshotAge)
)

func snapshotAge() float64 {
	var oldest time.Duration
	clients.Range(func(key, _ any) bool {
		oldest = max(oldest, key.(*Client).snapshotAge())
		return true
	})
	return oldest.Seconds()
}

// Model is one provider and model within a modality, whose health is kept.
type Model struct {
	Modality string
	Provider string
	Model    string
}

func (m Model) key() string { return healthKey(m.Modality, m.Provider, m.Model) }

// counts is one model's health hash.
type counts struct {
	requests       int64
	errors         int64
	latencyMsTotal float64
	fatalAtMs      int64
	// readAt is when Redis last answered for the model. What this router records itself
	// leaves it alone, so a snapshot Redis has stopped answering for still goes stale.
	readAt time.Time
}

func parseCounts(entries map[string]string) counts {
	return counts{
		requests:       parseInt(entries["requests"]),
		errors:         parseInt(entries["errors"]),
		latencyMsTotal: parseFloat(entries["latency_ms_total"]),
		fatalAtMs:      parseInt(entries["fatal_at_ms"]),
	}
}

// snapshot is every model's health as this router last read it, refreshed in the background,
// so a routing decision reads memory and never Redis.
type snapshot struct {
	mu sync.Mutex
	// models is every model the snapshot keeps, by key: the ones tracked and the ones asked
	// about. counts holds those Redis has answered for.
	models map[string]Model
	counts map[string]counts
	// refreshedAt is when every model was last read together, or when the client was made.
	refreshedAt time.Time
	warnedAt    time.Time
}

// Track has the next refresh read the models, so a router's first walk over them finds them
// in the snapshot.
func (c *Client) Track(models ...Model) {
	c.snapshot.mu.Lock()
	defer c.snapshot.mu.Unlock()
	for _, model := range models {
		c.snapshot.models[model.key()] = model
	}
}

// Healths is the health of each model, in order, from the snapshot. The ones missing from it
// are read in one pipelined round trip and kept, so the refresh reads them from then on. A
// model whose read failed, or which Redis has not answered for within the health window, is
// unmeasured: available, with no history, as a failed read of Health is taken to be. The
// error is the failed read's.
func (c *Client) Healths(ctx context.Context, models []Model) ([]Health, error) {
	healths := make([]Health, len(models))
	var missing []int
	c.snapshot.mu.Lock()
	now := c.now()
	for i, model := range models {
		found, ok := c.snapshot.counts[model.key()]
		if !ok {
			missing = append(missing, i)
			continue
		}
		healths[i] = c.judge(model, found, now)
	}
	c.snapshot.mu.Unlock()
	if len(missing) == 0 {
		return healths, nil
	}

	unread := make([]Model, len(missing))
	for j, i := range missing {
		unread[j] = models[i]
	}
	read, err := c.read(ctx, unread, readMiss)
	now = c.now()
	for _, i := range missing {
		healths[i] = unmeasured(models[i])
		if found, ok := read[models[i].key()]; ok {
			healths[i] = c.judge(models[i], found, now)
		}
	}
	return healths, err
}

// read reads the models' health in one pipelined round trip, which rueidis splits per node,
// and keeps what Redis answered in the snapshot.
func (c *Client) read(ctx context.Context, models []Model, kind string) (map[string]counts, error) {
	unique := map[string]Model{}
	for _, model := range models {
		unique[model.key()] = model
	}
	keys := slices.Sorted(maps.Keys(unique))
	commands := make([]rueidis.Completed, len(keys))
	for i, key := range keys {
		commands[i] = c.shard.B().Hgetall().Key(key).Build()
	}
	healthReads.WithLabelValues(kind).Inc()
	responses := c.shard.DoMulti(ctx, commands...)

	read := make(map[string]counts, len(keys))
	var failed error
	at := c.now()
	for i, response := range responses {
		entries, err := response.AsStrMap()
		if err != nil {
			if failed == nil {
				failed = err
			}
			continue
		}
		found := parseCounts(entries)
		found.readAt = at
		read[keys[i]] = found
	}

	c.snapshot.mu.Lock()
	for key, found := range read {
		c.snapshot.models[key] = unique[key]
		c.snapshot.counts[key] = found
	}
	c.snapshot.mu.Unlock()
	if failed != nil {
		return read, stack.Wrap(fmt.Errorf("live: read health of %d of %d models: %w", len(keys)-len(read), len(keys), failed))
	}
	return read, nil
}

// Refresh reads every model the snapshot keeps again now, as the background loop does every
// Options.Refresh. A failure keeps the last snapshot, which goes stale model by model as the
// health window passes.
func (c *Client) Refresh(ctx context.Context) error {
	c.snapshot.mu.Lock()
	models := slices.Collect(maps.Values(c.snapshot.models))
	c.snapshot.mu.Unlock()
	if len(models) > 0 {
		ctx, cancel := context.WithTimeout(ctx, refreshTimeout)
		defer cancel()
		if _, err := c.read(ctx, models, readRefresh); err != nil {
			healthRefreshErrors.Inc()
			c.snapshot.mu.Lock()
			warn := c.now().Sub(c.snapshot.warnedAt) >= warnEvery
			if warn {
				c.snapshot.warnedAt = c.now()
			}
			c.snapshot.mu.Unlock()
			if warn {
				c.options.Logger.Warn("could not refresh provider health, the last snapshot answers",
					"models", len(models), "age", c.snapshotAge(), "error", err)
			}
			return err
		}
	}
	c.snapshot.mu.Lock()
	c.snapshot.refreshedAt = c.now()
	c.snapshot.mu.Unlock()
	return nil
}

// refreshing refreshes every Options.Refresh until stop is closed.
func (c *Client) refreshing() {
	defer close(c.stopped)
	ticker := time.NewTicker(c.options.Refresh)
	defer ticker.Stop()
	for {
		select {
		case <-c.stop:
			return
		case <-ticker.C:
			_ = c.Refresh(context.Background())
		}
	}
}

func (c *Client) snapshotAge() time.Duration {
	c.snapshot.mu.Lock()
	defer c.snapshot.mu.Unlock()
	return c.now().Sub(c.snapshot.refreshedAt)
}

// observe adds what this router recorded to its own snapshot, so a model that just failed here
// is ranked by it before the next refresh brings the same from Redis. Only models already in
// the snapshot are changed: one it has not read yet is read when it is first asked about.
func (c *Client) observe(usages []Usage) {
	c.snapshot.mu.Lock()
	defer c.snapshot.mu.Unlock()
	for _, usage := range usages {
		if usage.Cancelled || usage.Private {
			continue
		}
		key := healthKey(usage.Modality, usage.Provider, usage.Model)
		found, ok := c.snapshot.counts[key]
		if !ok {
			continue
		}
		found.requests++
		found.latencyMsTotal += usage.LatencyMs
		if !usage.Success {
			found.errors++
		}
		if usage.Fatal {
			found.fatalAtMs = c.now().UnixMilli()
		}
		c.snapshot.counts[key] = found
	}
}

// judge is a model's health from its counts. One Redis has not answered for within the window
// is unmeasured.
func (c *Client) judge(model Model, found counts, now time.Time) Health {
	if !found.readAt.IsZero() && now.Sub(found.readAt) > c.options.Window {
		return unmeasured(model)
	}
	health := Health{Provider: model.Provider, Model: model.Model, Available: true}
	health.Requests = found.requests
	health.Errors = found.errors
	if health.Requests > 0 {
		health.LatencyMsAvg = found.latencyMsTotal / float64(health.Requests)
	}
	// Too few requests are not a rate: until there are enough, an unseen provider and one that
	// failed once look alike.
	health.Available = health.Requests < c.options.MinRequests || health.ErrorRate() < c.options.MaxErrorRate
	// A fatal failure is one every caller would get, so no number of successes outweighs it.
	if found.fatalAtMs > 0 && now.Sub(time.UnixMilli(found.fatalAtMs)) < c.options.Window {
		health.Available = false
	}
	return health
}

// unmeasured is a model with no history, which is available.
func unmeasured(model Model) Health {
	return Health{Provider: model.Provider, Model: model.Model, Available: true}
}
