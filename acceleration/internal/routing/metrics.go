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

// writers is every open Writer of this process, which the queue gauges add up, and sweepers
// every running Sweeper, which the unshipped gauges do.
var writers, sweepers sync.Map

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
	usageShipped = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_shipped_total",
		Help:      "Usage rows the usage service has, accepted now or before; a row the shipper and the sweeper both sent, or sent again after it was not marked shipped, counts each time",
	})
	usageShipErrors = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_ship_errors_total",
		Help:      "Shipments of usage rows that failed, left for the sweeper to send again",
	})
	usageShipDeferred = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_ship_deferred_total",
		Help:      "Written usage rows the shipper had no room for, or that waited for it past the settle time, left for the sweeper to send",
	})
	usageRejected = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_rejected_total",
		Help:      "Usage rows the usage service refused, kept with their ship_error until reshipped",
	})
	usageUnowned = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "usage_unowned_total",
		Help:      "Usage rows a sweep left unshipped because their app has joined no organization yet",
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
	}, oldestPending)
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "usage_unshipped_rows",
		Help:      "This shard's usage rows not yet in the usage service; every router of the shard reports the same count",
	}, func() float64 {
		var rows int64
		sweepers.Range(func(key, _ any) bool {
			rows = max(rows, key.(*Sweeper).unshipped.Load())
			return true
		})
		return float64(rows)
	})
	_ = promauto.NewGaugeFunc(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "usage_unshipped_oldest_seconds",
		Help:      "How long ago the work of this shard's oldest usage row not yet in the usage service started; every router of the shard reports it",
	}, unshippedOldest)
)

// oldestPending is how long the oldest row any writer of the process is writing has waited.
func oldestPending() float64 {
	var oldest time.Duration
	writers.Range(func(key, _ any) bool {
		oldest = max(oldest, key.(*Writer).waited())
		return true
	})
	return oldest.Seconds()
}

// unshippedOldest is how long ago the oldest unshipped row the last count found started.
func unshippedOldest() float64 {
	var oldest time.Duration
	sweepers.Range(func(key, _ any) bool {
		oldest = max(oldest, time.Duration(key.(*Sweeper).oldest.Load()))
		return true
	})
	return oldest.Seconds()
}
