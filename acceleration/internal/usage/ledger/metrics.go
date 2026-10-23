package ledger

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The ledger's metrics. No label carries an organization.
var (
	// WindowsUpserted counts window rows written by ingests.
	WindowsUpserted = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_windows_upserted_total",
		Help: "Window rows inserted or added to by recorded events.",
	})
	mirrorSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "usage_mirror_seconds",
		Help:    "Time to write one batch of window totals to Redis.",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 13),
	})
	// MirrorFailures counts organizations whose totals could not be written to Redis, once
	// per attempt.
	MirrorFailures = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_mirror_failures_total",
		Help: "Writes of one organization's window totals to Redis that failed, retries included.",
	})
	// MirrorGivenUp counts window totals dropped without reaching Redis: retried for as long
	// as the mirror retries, or still waiting when the service stopped. Each stays stale in
	// Redis until its next write or a read through the service.
	MirrorGivenUp = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_mirror_given_up_total",
		Help: "Window totals dropped unwritten to Redis, after the retries or at shutdown.",
	})
)
