package harness

import (
	"context"
	"errors"
	"time"
)

// Hearer is a flow controller that hears the speaker as well as reading their words. It is
// not a Judge: a Judge is asked named questions about written state, while a Hearer's model
// is asked about the audio in the words it was trained on, which only its server writes, and
// answers with a ruling.
type Hearer interface {
	// Hear rules on turn: its Disposition and Floor, and Tool, Cannot and Again.
	Hear(ctx context.Context, turn FlowTurn) (Decided, error)
}

// Audio is a recording of the speaker, encoded as a Hearer's server takes it.
type Audio struct {
	// Type is Data's media type.
	Type string
	Data []byte
	// Duration is how long it lasts.
	Duration time.Duration
}

// HearDeadline is how long a Hearer has to rule on turn: a third of the flow controller's
// own deadline, which leaves the judge or the model the other two thirds to decide as they
// would without it.
func HearDeadline(turn FlowTurn) time.Duration {
	if turn.Unfinished && turn.Speaking {
		return overlapDeadline / 3
	}
	return FlowDeadline / 3
}

// runHearer asks the hearer about one candidate and reports its ruling, or false when it
// did not rule in time.
func (f *flow) runHearer(asked *candidate) bool {
	ctx, cancel := context.WithTimeout(asked.ctx, HearDeadline(asked.turn))
	defer cancel()
	ruling, err := f.hearer.Hear(ctx, asked.turn)
	if errors.Is(asked.ctx.Err(), context.Canceled) {
		// Newer words replaced these, or the conversation is over.
		return true
	}
	if err != nil {
		f.logger.Debug("the hearer did not rule, the flow controller will", "candidate", asked.turn.ID, "error", err)
		return false
	}
	if f.forget(asked.turn.ID) == nil {
		return true
	}
	ruling.CandidateID, ruling.TookMs, ruling.Ruler = asked.turn.ID, millis(time.Since(asked.askedAt)), Heard
	f.emitter.Send(ruling)
	return true
}
