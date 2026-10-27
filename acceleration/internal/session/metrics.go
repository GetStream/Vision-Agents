package session

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// connector is store.ConnectorLabel of the call's connector; the tool name is left out, since
// a custom connector's tools are named by its provider, without a bound.
var (
	// outcome is "ok" for a call that answered, else its row's error_type (store.Invocation*).
	connectorCalls = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "connector_calls_total",
		Help:      "Connector tool calls a session ran or refused, by how they ended: ok or the error_type of their invocation row (session dispatcher.record)",
	}, []string{"connector", "outcome"})
	connectorCallSeconds = promauto.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "connector_call_seconds",
		Help:      "How long a connector tool call took, the latency_ms of its invocation row, whatever its outcome (session dispatcher.record)",
		// 25 ms to 51.2 s, past the 30 s a binding's timeout_ms may be (api configs.go).
		Buckets: prometheus.ExponentialBuckets(0.025, 2, 12),
	}, []string{"connector"})
)
