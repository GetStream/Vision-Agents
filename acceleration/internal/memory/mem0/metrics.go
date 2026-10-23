package mem0

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// op is the call (search, add, list, batch_delete), status mem0's HTTP status or "none"
// when it never answered, and outcome one of the outcome* constants: all fixed sets.
var (
	calls = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "mem0_calls_total",
		Help:      "Calls to mem0, by operation, mem0's HTTP status and how the call ended",
	}, []string{"op", "status", "outcome"})
	callSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "mem0_call_seconds",
		Help:      "How long a call to mem0 took, by operation and how it ended",
		// 50 ms to 12.8 s, past the 10 s a call is given (defaultTimeout).
		Buckets: prometheus.ExponentialBuckets(0.05, 2, 9),
	}, []string{"op", "outcome"})
	emptyLists = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "mem0_empty_lists_total",
		Help:      "Lists of memories to delete that mem0 answered with none",
	})
)
