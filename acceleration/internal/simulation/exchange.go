package simulation

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/session"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// transport is the agent under test and the way one thing the caller says reaches it.
//
// Both kinds of simulation hold the same session and have the same conversation. What
// differs is only how a line crosses: in writing it is handed over, and out loud it is
// spoken, carried as audio, transcribed, answered, spoken back and transcribed again. The
// loop above this does not know which it is watching.
type transport interface {
	// Session is the conversation being held, so a case can say which call it was.
	Session() *session.Session
	// Opening is what the agent said before anybody said anything to it. An agent that
	// answers the phone greets whoever called, and a caller deciding what to say next has
	// to have heard it.
	Opening() string
	// Say delivers one line and returns the agent's answer, once it has finished saying
	// it rather than once it has started. The answer is a line rather than a string
	// because out loud there are two of them: what the agent meant and what was heard.
	Say(ctx context.Context, text string) (store.SimulationLine, error)
	Close() error
}

// exchange plays a scenario out.
//
// It stops when the caller decides it has asked everything it came to ask, when the turns
// run out, or when the clock does. Whatever was said before it stopped is still returned:
// a conversation that ran out of turns is a failure with evidence, not a lost one.
func exchange(ctx context.Context, persona *caller, over transport, limit int) (said, string, error) {
	var so said
	if opening := over.Opening(); opening != "" {
		so = append(so, store.SimulationLine{Text: opening, At: time.Now().UTC()})
	}

	for so.turns() < limit {
		next, err := persona.next(ctx, so)
		if err != nil {
			return so, store.EndedFailed, err
		}
		if next.Done {
			return so, store.EndedComplete, nil
		}

		so = append(so, store.SimulationLine{Caller: true, Text: next.Say, At: time.Now().UTC()})

		reply, err := over.Say(ctx, next.Say)
		if err != nil {
			if ctx.Err() != nil {
				return so, store.EndedTimeout, err
			}
			return so, store.EndedFailed, err
		}
		reply.At = time.Now().UTC()
		so = append(so, reply)
	}
	return so, store.EndedTurns, nil
}
