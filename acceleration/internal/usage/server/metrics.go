package server

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The service's metrics, beside the ledger's. No label carries an organization.
var (
	eventsReceived = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_events_received_total",
		Help: "Events in the batches producers shipped, whatever became of them.",
	})
	eventsWritten = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_events_written_total",
		Help: "Events recorded for the first time.",
	})
	eventsDuplicate = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_events_duplicate_total",
		Help: "Events already recorded, sent again.",
	})
	// EventsRejected counts events refused, by reason: each one is revenue lost until the
	// producer is fixed and ships it again.
	EventsRejected = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "usage_events_rejected_total",
		Help: "Events refused for breaking the contract, by reason.",
	}, []string{"reason"})
	// BatchesRefused counts batches refused whole, by status: none of their events is
	// recorded until the producer ships them right.
	BatchesRefused = promauto.NewCounterVec(prometheus.CounterOpts{
		Name: "usage_batches_refused_total",
		Help: "Batches answered 4xx, none of their events recorded, by status.",
	}, []string{"status"})
	// IngestErrors counts batches answered with a 5xx.
	IngestErrors = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_ingest_errors_total",
		Help: "Batches the service could not record, answered 5xx.",
	})
	ingestSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "usage_ingest_seconds",
		Help:    "Time to record one batch, Redis included.",
		Buckets: prometheus.ExponentialBuckets(0.002, 2, 12),
	})
	windowReads = promauto.NewCounter(prometheus.CounterOpts{
		Name: "usage_window_reads_total",
		Help: "Subjects read through /v1/windows.",
	})
	standingSeconds = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "usage_standing_seconds",
		Help:    "Time to answer one standings read.",
		Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
	})
	// EventLag is how long after it happened a router's event was recorded. Other producers'
	// events are left out: rollups and reconcile jobs ship late by design.
	EventLag = promauto.NewHistogram(prometheus.HistogramOpts{
		Name:    "usage_event_lag_seconds",
		Help:    "received_at - occurred_at of each event recorded from a router (source router/...).",
		Buckets: []float64{0.5, 1, 2, 5, 10, 30, 60, 120, 300, lagAlertSeconds, 900, 1800, 3600, 21600, 86400},
	})
)
