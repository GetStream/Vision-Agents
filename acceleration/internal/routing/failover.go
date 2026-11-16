package routing

import (
	"errors"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The stages a failover is counted at: in Walk, a candidate that failed to start, or one
// that started and failed in its use; and a provider that failed after its session had
// started, which another candidate took over from.
const (
	failoverStart   = "start"
	failoverUse     = "use"
	failoverSession = "session"
)

// The reasons a failover is counted under. Only vendor is the vendor failing, which is what
// an alert reads: request is a setting the provider refused (ErrRequest), the same for every
// vendor asked; customer_key is work on the customer's own key or their own model (custom,
// their endpoint and key), whose revoked key or spent account is theirs; cancelled is a candidate that failed once the caller's context had
// ended.
const (
	failoverVendor      = "vendor"
	failoverRequest     = "request"
	failoverCustomerKey = "customer_key"
	failoverCancelled   = "cancelled"
)

var failovers = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "failovers_total",
	Help:      "Times the router moved work off a provider that failed, by the provider that failed, counted once another candidate has been started in its place: a failure nothing replaced (the last candidate, a walk the caller left) is no failover. Stage start is a candidate Walk could not start, use one that started and failed in Walk's use, session one that failed after its session started and another took over (an STT session the vendor refused, a voice lost mid-call to another provider or key, an LLM response that failed on the session's model or a fallback). By reason: request for a setting the provider refused, customer_key for work on the customer's own key or their own model (custom), cancelled once the caller's context had ended, otherwise vendor",
}, []string{"modality", "provider", "stage", "reason"})

// failover is a candidate the walk moved past on a failure.
type failover struct {
	config ProviderConfig
	stage  string
	reason string
}

// CountHandover counts a provider that failed after its session started, err being what it
// said, once another candidate has taken the work over.
func CountHandover(modality Modality, config ProviderConfig, err error) {
	countFailover(modality, failover{config: config, stage: failoverSession, reason: failoverReason(config, err, false)})
}

// failoverReason is what a candidate's failure is counted under, ended when the caller's
// context had ended by then.
func failoverReason(config ProviderConfig, err error, ended bool) string {
	switch {
	case errors.Is(err, ErrRequest):
		return failoverRequest
	case ended:
		return failoverCancelled
	case config.BYOK || config.Provider == CustomProvider:
		return failoverCustomerKey
	default:
		return failoverVendor
	}
}

// countFailover counts moved by provider alone: a model label would grow the series with
// the catalog, and router_requests_total already tells a provider's models apart.
func countFailover(modality Modality, moved failover) {
	failovers.WithLabelValues(string(modality), moved.config.Provider, moved.stage, moved.reason).Inc()
}
