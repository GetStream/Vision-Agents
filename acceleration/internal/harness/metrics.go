package harness

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// reason is parseFlow's, a fixed set.
var invalidRulings = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "agent_flow_invalid_rulings_total",
	Help:      "Flow controller answers refused and replaced by the fallback ruling, by what was wrong: decode, disposition or floor",
}, []string{"reason"})
