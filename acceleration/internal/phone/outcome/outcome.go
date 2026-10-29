// Package outcome keeps how each phone call the router placed ended. The vendor's reports and
// the call's own events (its phone joining and leaving) arrive in any order; decide is the one
// place that turns them into an outcome.
package outcome

import "time"

// Outcome is how a call ended. Empty is not decided yet.
type Outcome string

const (
	Completed Outcome = "completed"
	Busy      Outcome = "busy"
	NoAnswer  Outcome = "no_answer"
	Failed    Outcome = "failed"
	Canceled  Outcome = "canceled"
	// Unknown is a call that ended never joined, from a vendor that does not report.
	Unknown Outcome = "unknown"
)

// Unanswered reports whether the person never picked up.
func (o Outcome) Unanswered() bool {
	switch o {
	case Busy, NoAnswer, Failed, Canceled:
		return true
	}
	return false
}

// Key names a call by its Stream call and the numbers at either end, so a transfer into the
// same Stream call is a different call.
type Key struct {
	CustomerID string
	// StreamAppPK is the Stream app the call is in. Zero is the deployment's own app.
	StreamAppPK int64
	CallType    string
	CallID      string
	Local       string
	Remote      string
}

// Call is one placed call and what is known of how it ended.
type Call struct {
	Key
	Vendor string
	// Reports is whether the vendor posts a final status for the call.
	Reports      bool
	VendorCallID string
	PlacedAt     time.Time
	// AnsweredAt is when the person's phone joined the Stream call, and EndedAt when its leg
	// ended. Nil until then.
	AnsweredAt *time.Time
	EndedAt    *time.Time
	Outcome    Outcome
	// VendorStatus is the vendor's own word for the call, kept for debugging.
	VendorStatus string
	DecidedAt    *time.Time
}

// Duration is how long the person was on the call, known once it was answered and has ended.
func (c Call) Duration() (time.Duration, bool) {
	if c.AnsweredAt == nil || c.EndedAt == nil {
		return 0, false
	}
	return c.EndedAt.Sub(*c.AnsweredAt), true
}

// NobodyAnswered reports whether the call ended without the person picking up. A vendor's
// busy for a call whose phone joined is not that.
func (c Call) NobodyAnswered() bool {
	return c.Outcome.Unanswered() && c.AnsweredAt == nil
}

type eventKind int

const (
	reportedEvent eventKind = iota
	answeredEvent
	endedEvent
)

// event is one thing learned about a call. reported and word are a vendor's report's: reported
// is empty when it is not final, and word is the vendor's own.
type event struct {
	kind     eventKind
	reported Outcome
	word     string
	at       time.Time
}

// decide is c once e is known. Every rule of what a call's outcome is lives here.
func decide(c Call, e event) Call {
	was := c.Outcome
	switch e.kind {
	case reportedEvent:
		// The vendor's first final report is kept, and its word with it.
		if c.Outcome == "" {
			c.VendorStatus = e.word
			c.Outcome = e.reported
		}
	case answeredEvent:
		if c.AnsweredAt == nil {
			c.AnsweredAt = &e.at
		}
	case endedEvent:
		if c.EndedAt == nil {
			c.EndedAt = &e.at
		}
		if c.Outcome == "" {
			c.Outcome = Unknown
			if c.Reports {
				c.Outcome = NoAnswer
			}
		}
	}
	// Stream saw the phone join, which outranks whatever the vendor said. A call still under
	// way stays undecided until it ends.
	if c.AnsweredAt != nil && c.Outcome != "" {
		c.Outcome = Completed
	}
	if c.Outcome != was {
		c.DecidedAt = &e.at
	}
	return c
}
