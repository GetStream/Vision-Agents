package session

import (
	"strconv"

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

// The reasons router_session_ends_total counts beside the EndReason values, for an end
// nothing gave a reason for. None is stored on the session's row.
const (
	// endClosed is a session closed with no reason said: by its caller (Manager.Close), by
	// the agent's next session in the same call (supersede), by a reopened conversation
	// nobody was watching (takeOver), or by a chat left unwatched past its grace.
	endClosed = "closed"
	// endShutdown is a session the router ended as it went away (Manager.Shutdown).
	endShutdown = "shutdown"
	// endCreateFailed is a session Create made and then gave up on before it was held, such
	// as one whose call refused the agent.
	endCreateFailed = "create_failed"
	// endAgentFailed is a session whose agent left the call by itself, not by hanging up or
	// transferring: its speech-to-speech session ended and could not be replaced, or its
	// playback queue filled. Counted when the session closes, whatever ends it then.
	endAgentFailed = "agent_failed"
)

// endReasons is every reason label router_session_ends_total has, and whether it is a
// failure: a session that never got going (nobody joined it in time, Create gave up on it)
// or whose agent dropped out. A call its callee did not answer or a router shutting down is
// no failure of the session.
var endReasons = map[string]bool{
	string(EndAgentHangup):    false,
	string(EndTransferred):    false,
	string(TimeoutHangup):     false,
	string(TimeoutStart):      true,
	string(TimeoutUnanswered): false,
	string(EndVoicemail):      false,
	endClosed:                 false,
	endShutdown:               false,
	endCreateFailed:           true,
	endAgentFailed:            true,
}

var (
	sessionsActive = promauto.NewGauge(prometheus.GaugeOpts{
		Namespace: "router",
		Name:      "sessions_active",
		Help:      "Sessions this process holds, from Manager.Create holding one until it has closed; sum across pods for a shard",
	})
	// failed is "true" for the reasons endReasons marks as failures, so an alert reads it
	// rather than a list of reasons that grows.
	sessionEnds = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "session_ends_total",
		Help:      "Sessions ended, once each, by reason: an end_reason a session row stores, or closed, shutdown, create_failed or agent_failed",
	}, []string{"reason", "failed"})
)

// The series exist from the start, so a rate over the first failure reads it.
func init() {
	for reason := range endReasons {
		endCounter(reason)
	}
}

// endCounter is the router_session_ends_total series of reason.
func endCounter(reason string) prometheus.Counter {
	return sessionEnds.WithLabelValues(reason, strconv.FormatBool(endReasons[reason]))
}

// endLabel is the reason router_session_ends_total counts the session's end under. An agent
// that dropped out came first: what ends the session after it is a consequence.
func (s *Session) endLabel() string {
	reason := s.endReason()
	switch {
	case s.agentLeft.Load():
		return endAgentFailed
	case reason != "":
		return string(reason)
	case s.Halted():
		return endShutdown
	}
	return endClosed
}

// counted takes the session's end into the session metrics if the manager held it. Called
// once, from close.
func (s *Session) counted() {
	if s.active.Load() {
		sessionsActive.Dec()
		endCounter(s.endLabel()).Inc()
	}
}
