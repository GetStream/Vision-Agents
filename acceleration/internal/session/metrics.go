package session

import (
	"errors"
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/policy"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
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

var (
	sessionsActive = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "sessions_active",
		Help:      "Sessions this process holds, from Manager.Create holding one until it has closed; sum across pods for a shard",
	})
	// failed, fault and stage are what EndReasons says of the reason, so an alert reads them
	// rather than a list of reasons that grows. Each reason is one series.
	sessionEnds = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "session_ends_total",
		Help: "Sessions ended, once each, by reason (session.EndReasons: the end_reason a session row stores, or a " +
			"create_* reason for one never held), whether that is a failure, whose fault (none, caller, router, vendor, " +
			"carrier, customer_config, customer_credit) and at which stage (starting, ringing, conversation)",
	}, []string{"reason", "failed", "fault", "stage"})
)

// The series exist from the start, so a rate over the first failure reads it.
func init() {
	for _, info := range EndReasons {
		endCounter(info.Reason)
	}
}

// endCounter is the router_session_ends_total series of reason.
func endCounter(reason EndReason) prometheus.Counter {
	info, _ := reason.Info()
	return sessionEnds.WithLabelValues(string(reason), strconv.FormatBool(info.Failed), string(info.Fault), string(info.Stage))
}

// endLabel is the reason router_session_ends_total counts the session's end under. An agent
// that dropped out, or a model that answered nothing, came first: what ends the session after
// it is a consequence.
func (s *Session) endLabel() EndReason {
	reason := s.endReason()
	switch {
	case s.agentLeft.Load(), s.modelErred.Load() && !s.replied.Load():
		return s.failedFor()
	case reason != "":
		return reason
	case s.Halted():
		return EndShutdown
	}
	return EndClosed
}

// failedFor is the reason a session whose agent dropped out, or whose model answered nothing,
// ends for: whose doing its last model failure was. An agent that left with none since its
// last answer left for a failure of the router's own.
func (s *Session) failedFor() EndReason {
	switch s.fault() {
	case FaultCustomerCredit:
		return EndModelNoCredit
	case FaultCustomerConfig:
		return EndModelKeyRefused
	case FaultVendor:
		return EndModelFailed
	}
	return EndAgentFailed
}

// kept is the reason the session's row keeps: the one noted, so a row keeps what it kept
// before the reasons said whose fault an end was, and otherwise the one it is counted under.
func (s *Session) kept() EndReason {
	if reason := s.endReason(); reason != "" {
		return reason
	}
	return s.endLabel()
}

// counted takes the session's end into the session metrics if the manager held it. Called
// once, from close.
func (s *Session) counted() {
	if s.active.Load() {
		sessionsActive.Dec()
		endCounter(s.endLabel()).Inc()
	}
}

// fault is whose doing the session's last model failure since its last answer was, empty
// for none.
func (s *Session) fault() Fault {
	fault, _ := s.failure.Load().(Fault)
	return fault
}

// createFailure is the reason a session Create gave up on, by whose fault err is.
func createFailure(err error) EndReason {
	fault, _ := faultOf(err)
	switch fault {
	case FaultCustomerCredit:
		return EndCreateNoCredit
	case FaultCustomerConfig:
		return EndCreateConfigRefused
	case FaultVendor:
		return EndCreateVendorFailed
	}
	return EndCreateFailed
}

// faultOf is whose doing a failure was, and false when err does not say. Money that could not
// be checked is the router's, money that ran out the customer's; a setting a vendor refused or
// a model the customer's policies do not allow is the customer's configuration.
func faultOf(err error) (Fault, bool) {
	switch {
	case policy.Transient(err), errors.Is(err, agent.ErrPlaybackFull):
		return FaultRouter, true
	case policy.Code(err) != "":
		return FaultCustomerCredit, true
	case errors.Is(err, routing.ErrRequest), errors.Is(err, routing.ErrModelNotAllowed):
		return FaultCustomerConfig, true
	case errors.Is(err, routing.ErrVendorsFailed):
		return FaultVendor, true
	}
	return "", false
}

// modelCallFault is whose doing a model request that failed was: the customer's when it ran
// on their own key or their own model, as router_requests_total's customer_key outcome counts
// it, and the vendor's otherwise.
func modelCallFault(call agent.ModelCall) Fault {
	if call.ErrorCode == routing.ErrorKeyRefused || call.Provider == routing.CustomProvider {
		return FaultCustomerConfig
	}
	return FaultVendor
}

// observeTurn notes, for endLabel, a turn the session's model failed and a turn the agent
// answered. lost is the last turn whose model request failed, which the agent still
// reports as responded, saying that the reply was lost; one agent's events share it. A
// greeting is said without the caller asking, and a turn the prompt injection screen
// refused is the caller's doing, not the model's.
func (s *Session) observeTurn(event Event, lost *string) {
	switch typed := event.(type) {
	case agent.ModelCall:
		switch {
		case !typed.Success && typed.ErrorCode != string(llm.StatusCancelled):
			*lost = typed.TurnID
			s.failure.Store(modelCallFault(typed))
		case typed.Success && typed.TurnID == *lost:
			// Another provider answered the request the first one failed.
			*lost = ""
		}
	case agent.Responded:
		if typed.Purpose != agent.PurposeGreeting && typed.TurnID != *lost {
			s.replied.Store(true)
			s.failure.Store(Fault(""))
		}
	case agent.Error:
		if (typed.Context == "llm" || typed.Context == "sts") && !errors.Is(typed.Err, policy.ErrPromptInjection) {
			s.modelErred.Store(true)
			// What the error says wins over what the request that failed said; one that says
			// nothing, with no request failed before it, is the vendor's.
			if fault, known := faultOf(typed.Err); known {
				s.failure.Store(fault)
			} else if s.fault() == "" {
				s.failure.Store(FaultVendor)
			}
		}
	}
}
