package routing

import (
	"errors"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

// The stages a failover is counted at: in Walk, a candidate that failed to start, or one
// that started and failed in its use; and a provider that failed after its session had
// started, which the router asks another candidate to take over from.
const (
	failoverStart   = "start"
	failoverUse     = "use"
	failoverSession = "session"
)

// The reasons a failover is counted under. Only vendor is the vendor failing, which is what
// an alert reads: request is a setting the provider refused (ErrRequest), the same for every
// vendor asked; customer_key is work on the customer's own key, whose revoked key or spent
// account is theirs; cancelled is a candidate that failed once the caller's context had
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
	Help:      "Times the router moved work off a provider that failed, by the provider and model that failed. In Walk, stage start is a candidate that failed to start and use one that started and failed in its use, counted when the next candidate is started: the last candidate failing, or a walk the caller left, is no failover. Stage session is a provider that failed after its session started (an STT session the vendor refused, a voice lost mid-call, an LLM response that failed), counted when another candidate is asked to take it over, whether or not one does. By reason: request for a setting the provider refused, customer_key for work on the customer's own key, cancelled once the caller's context had ended, otherwise vendor",
}, []string{"modality", "provider", "model", "stage", "reason"})

// failover is a candidate the walk moved past on a failure.
type failover struct {
	config ProviderConfig
	stage  string
	reason string
}

// CountHandover counts a provider that failed after its session started, err being what it
// said, as the router asks another candidate to take the work over.
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
	case config.BYOK:
		return failoverCustomerKey
	default:
		return failoverVendor
	}
}

func countFailover(modality Modality, moved failover) {
	model := moved.config.Model
	if moved.config.Provider == CustomProvider {
		// The model is a name each customer chose: unbounded as a label.
		model = CustomProvider
	}
	failovers.WithLabelValues(string(modality), moved.config.Provider, model, moved.stage, moved.reason).Inc()
}
