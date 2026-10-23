package routing

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The reasons a usage row is lost.
const (
	lostQueueFull = "queue_full"
	lostShutdown  = "shutdown"
	// lostInvalid is a row Postgres refuses however often it is sent, written to the log in
	// full so it can be put right and replayed.
	lostInvalid = "invalid"
)

// writers is every open Writer of this process, which the queue gauges add up.
var writers sync.Map

var (
	usageWritten = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_written_total",
		Help:      "Usage rows written to Postgres",
	})
	usageDuplicates = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_duplicates_total",
		Help:      "Usage rows not written because a row with their usage id already was",
	})
	usageWriteErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_write_errors_total",
		Help:      "Batches of usage rows that failed to write and are being tried again",
	})
	usageLost = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_lost_total",
		Help:      "Usage rows lost: each is revenue, and is logged in full as usage row lost",
	}, []string{"reason"})
	usageLostMicros = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_lost_micros_total",
		Help:      "What the lost usage rows cost, in millionths of a dollar",
	})
	usageUnattributed = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_unattributed_total",
		Help:      "Usage rows written with no API key or origin, under the key unattributed",
	})
	usageUnresolved = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_unresolved_total",
		Help:      "Usage rows written without their organization because looking it up failed",
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "usage_queue_depth",
		Help:      "Usage rows in memory and not yet written: what a crash would lose",
	}, func() float64 {
		var depth int64
		writers.Range(func(key, _ any) bool {
			depth += key.(*Writer).pending()
			return true
		})
		return float64(depth)
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "usage_oldest_pending_seconds",
		Help:      "How long the oldest usage row being written has waited since it was recorded",
	}, func() float64 {
		var oldest time.Duration
		writers.Range(func(key, _ any) bool {
			oldest = max(oldest, key.(*Writer).waited())
			return true
		})
		return oldest.Seconds()
	})
)
