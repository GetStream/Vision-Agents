package outcome

import (
	"slices"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The labels a call is counted under, beside the outcomes it is stored with.
const (
	// dropped is an answered call its agent left without ending (AGENTS.md). It is never
	// stored: the stored outcome of such a call stays completed.
	dropped     = "dropped"
	inbound     = "inbound"
	outbound    = "outbound"
	otherVendor = "other"
)

var callOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "phone_call_outcomes_total",
	Help: "Phone calls, each counted once when its leg first ends, by vendor (one the router knows, else other), direction " +
		"(inbound, or outbound for a call the router placed; a transfer's leg is not counted) and outcome: the stored " +
		"one (completed, busy, no_answer, failed, canceled, unknown; an inbound call is completed, or failed when the router refused it), or dropped for an " +
		"answered call whose agent left it without ending it (internal/phone/outcome/AGENTS.md)",
}, []string{"vendor", "direction", "outcome"})

// endsTheCall are the session end reasons (session.EndReason, internal/session/timeout.go) of a
// session that ended its call or followed the caller's leaving: agent_hangup and voicemail end
// the call at Stream, transferred leaves the caller with a person, and hangup comes after the
// caller left. Written out because session imports this package.
var endsTheCall = []string{"agent_hangup", "transferred", "voicemail", "hangup"}

// EndsTheCall reports whether a session that ended for reason ended its call, or followed the
// caller's leaving, rather than leaving a caller on the line.
func EndsTheCall(reason string) bool { return slices.Contains(endsTheCall, reason) }

// leftOnTheLine reports whether an answered call that ended at ended was dropped by its agent:
// the call's session had closed by then, for a reason that does not end the call. A session
// still open when the leg ended (nil ClosedAt) saw the phone leave first, as a caller hanging
// up, or a network cutting the call, looks.
func leftOnTheLine(session store.AgentSession, ended time.Time) bool {
	return session.ClosedAt != nil && !session.ClosedAt.After(ended) && !EndsTheCall(session.EndReason)
}
