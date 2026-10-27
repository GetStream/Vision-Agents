package mem0

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/alerts"
)

// op is the call (search, add, list, delete), status mem0's HTTP status or "none"
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
		Help:      "Deletions whose first list of memories to delete mem0 answered with none",
	})
)

// Alerts are the conditions of mem0 refusing the router, one alert per operation and status at
// most every limit of the loop that runs them: any HTTP error, a spent quota (429) and a revoked
// key (401) alike, since each leaves recall and remember failing. It does not page: a call goes
// on without its memories.
func Alerts(limit time.Duration) []alerts.Condition {
	return []alerts.Condition{{Name: "mem0_refused", Level: alerts.Warn,
		Items: alerts.IncreasedPer(calls, limit, []string{"op", "status"}, map[string]string{"outcome": outcomeHTTPError})}}
}
