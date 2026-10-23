package keypool

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// unnamedProvider is the provider of a pool no FromEnv has named: one from New, or from
// keys passed in that no provider reads from a variable.
const unnamedProvider = "unknown"

// provider is the variable a pool's keys are read from (MEM0_API_KEY), a constant in the
// code, and status the refused response's HTTP status (refused): both fixed sets. A key
// is never a label.
var refusedAttempts = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "keypool_refused_attempts_total",
	Help:      "Requests and handshakes a provider refused on a pooled key, each moved to the next key or, on the last, handed back to fail over; by the variable the keys come from and the refusal's HTTP status",
}, []string{"provider", "status"})
