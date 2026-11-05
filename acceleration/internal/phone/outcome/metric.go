package outcome

import (
	"slices"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The labels a call is counted under, beside the outcomes it is stored with.
const (
	// dropped is an answered call its agent left without ending (AGENTS.md). It is never
	// stored: the stored outcome of such a call stays completed.
	dropped = "dropped"
	// refused is a dial the vendor refused: it never rang, and has no row.
	refused     = "refused"
	inbound     = "inbound"
	outbound    = "outbound"
	otherVendor = "other"
)

var callOutcomes = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "phone_call_outcomes_total",
	Help: "Phone calls, each counted once when its leg first ends, by vendor (one the router knows, else other), direction " +
		"(inbound, or outbound for a call the router placed; a transfer's leg is not counted) and outcome: the stored " +
		"one (completed, busy, no_answer, failed, canceled, unknown; an inbound call is completed, or failed when the " +
		"router refused it), dropped for an answered call whose agent left it without ending it, or refused for a dial " +
		"the vendor refused, counted when it refused (internal/phone/outcome/AGENTS.md)",
}, []string{"vendor", "direction", "outcome"})

var sipResponses = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "phone_call_sip_responses_total",
	Help: "Placed phone calls by the final SIP response to their INVITE, from the vendors that report one, each call " +
		"counted once, at its first final report: code is a response code of RFC 3261, any other as its class (4xx) " +
		"(internal/phone/outcome/AGENTS.md)",
}, []string{"vendor", "direction", "code"})

var postDialDelay = promauto.NewHistogramVec(prometheus.HistogramOpts{
	Namespace: "router",
	Name:      "phone_post_dial_delay_seconds",
	Help: "How long a placed phone call waited for a first sign of how it goes: from the INVITE to its first " +
		"response other than 100 Trying or a challenge (RFC 6076 §4.3) on the router's own bridge, or from the dial to " +
		"the vendor's first ringing report (internal/phone/outcome/AGENTS.md)",
	// ITU-T E.721's post-selection delay targets fall on bounds: a mean of 3 s, 5 s and 8 s and a 95th
	// percentile of 6 s, 8 s and 11 s, for a local, a toll and an international call.
	Buckets: []float64{0.5, 1, 2, 3, 5, 6, 8, 11, 15, 20, 30},
}, []string{"vendor", "direction"})

var shortCalls = promauto.NewCounterVec(prometheus.CounterOpts{
	Namespace: "router",
	Name:      "phone_short_calls_total",
	Help: "Answered phone calls the person was on for less than 6 s, counted when router_phone_call_outcomes_total " +
		"counts them (internal/phone/outcome/AGENTS.md)",
}, []string{"vendor", "direction"})

// shortCall is the shortest an answered call lasts and is not short: Twilio's
// pstn_short_duration, https://www.twilio.com/docs/voice/voice-insights/api/call/details-call-tags.
const shortCall = 6 * time.Second

// isShort reports whether a call answered for d was short. A negative d is a join that Stream
// dated after the leg's end, which says nothing of how long the call lasted.
func isShort(d time.Duration) bool { return d >= 0 && d < shortCall }

// sipCodes are the response codes of RFC 3261 §21, the codes a SIP response is counted under;
// any other final code is counted as its class, so the label stays a fixed set.
var sipCodes = []int{
	200,
	300, 301, 302, 305, 380,
	400, 401, 402, 403, 404, 405, 406, 407, 408, 410, 413, 414, 415, 416, 420, 421, 423,
	480, 481, 482, 483, 484, 485, 486, 487, 488, 491, 493,
	500, 501, 502, 503, 504, 505, 513,
	600, 603, 604, 606,
}

// sipCodeLabel is the label a final SIP response is counted under, false for a code that is no
// final response: RFC 3261 §7.2 puts those in the classes 2xx to 6xx.
func sipCodeLabel(code int) (string, bool) {
	switch {
	case code < 200 || code > 699:
		return "", false
	case slices.Contains(sipCodes, code):
		return strconv.Itoa(code), true
	}
	return strconv.Itoa(code/100) + "xx", true
}

// countReport counts what a vendor's report on a placed call says of its carrier. before is
// the call as the report found it, under the row's lock, which makes each count once a call:
// the first ringing report times the post-dial delay, and the first final report gives the
// SIP response and the bridge's post-dial delay.
func (k *Keeper) countReport(before Call, report Report, at time.Time) {
	vendor := k.vendorLabel(before.Vendor)
	// Only a call with no outcome can still start ringing: the leg's end decides one.
	if report.Ringing && before.Outcome == "" && before.VendorStatus != report.Word {
		postDialDelay.WithLabelValues(vendor, outbound).Observe(at.Sub(before.PlacedAt).Seconds())
	}
	if report.Outcome == "" || finalReportKept(before) {
		return
	}
	if code, ok := sipCodeLabel(report.SIPCode); ok {
		sipResponses.WithLabelValues(vendor, outbound, code).Inc()
	}
	if report.PostDialDelay > 0 {
		postDialDelay.WithLabelValues(vendor, outbound).Observe(report.PostDialDelay.Seconds())
	}
}

// finalReportKept reports whether a final report reached the call before: only a final report
// decides an outcome before the leg ends, and one that decided it at or after the end was the
// end. decide dates every outcome it sets. A vendor that posts two final reports after the leg
// ended has both counted.
func finalReportKept(c Call) bool {
	if c.Outcome == "" {
		return false
	}
	return c.EndedAt == nil || c.DecidedAt.Before(*c.EndedAt)
}

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
