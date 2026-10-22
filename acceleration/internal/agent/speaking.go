package agent

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// utterance is one synthesis on its way to the caller, kept so that the moment it starts
// being heard can be reported with what it says (Speaking).
type utterance struct {
	agent       *Agent
	turnID      string
	synthesisID string
	// text is what the voice has been sent for it. Agent.mu guards it.
	text string
	// published is set once any of its audio went to the edge.
	published atomic.Bool
	// pulledAt is when the track took its first audible frame, in Unix nanoseconds, zero
	// until it has. The edge may report that more than once.
	pulledAt atomic.Int64
	// mu is held while Speaking is sent, so the Spoke after it cannot overtake it.
	mu        sync.Mutex
	announced bool
}

func (u *utterance) FirstFrameQueued(time.Time) {}

// FirstAudiblePulled runs on the track's goroutine, or the voice's for an edge that cannot
// say, and neither may wait on a consumer, so Speaking is sent from a goroutine of its own.
func (u *utterance) FirstAudiblePulled(at time.Time) {
	if u.pulledAt.CompareAndSwap(0, at.UnixNano()) {
		go u.agent.announce(u, at)
	}
}

// marks is what the edge is told to report the utterance's first frames to, nil once the
// track has taken them.
func (u *utterance) marks() PlayoutMarks {
	if u == nil || u.pulledAt.Load() != 0 {
		return nil
	}
	return u
}

// bothMarks tells the turn's measurement and the utterance the same moments.
type bothMarks [2]PlayoutMarks

func (m bothMarks) FirstFrameQueued(at time.Time) {
	m[0].FirstFrameQueued(at)
	m[1].FirstFrameQueued(at)
}

func (m bothMarks) FirstAudiblePulled(at time.Time) {
	m[0].FirstAudiblePulled(at)
	m[1].FirstAudiblePulled(at)
}

// withUtterance adds an utterance's marks to a turn's. Either may be nil.
func withUtterance(turn PlayoutMarks, heard *utterance) PlayoutMarks {
	marks := heard.marks()
	switch {
	case marks == nil:
		return turn
	case turn == nil:
		return marks
	}
	return bothMarks{turn, marks}
}

// announce reports an utterance as being heard, once. A time the track reported wins over
// at, which is for an edge that reports none, or an utterance that finished before the
// track got to it.
func (a *Agent) announce(heard *utterance, at time.Time) {
	heard.mu.Lock()
	defer heard.mu.Unlock()
	if heard.announced {
		return
	}
	heard.announced = true
	if pulled := heard.pulledAt.Load(); pulled != 0 {
		at = time.Unix(0, pulled)
	}
	a.mu.Lock()
	text := heard.text
	a.mu.Unlock()
	a.emitter.Send(Speaking{TurnID: heard.turnID, SynthesisID: heard.synthesisID, Text: text, At: at})
}

// synthesize sends a request to the voice, keeping what each utterance is sent. It is kept
// first, since a voice may send audio back before Synthesize returns.
func (a *Agent) synthesize(voice *ttsrouter.Session, turnID string, request tts.Request) error {
	a.sending(turnID, request.ID, request.Text)
	err := voice.Synthesize(request)
	if err != nil && request.Final {
		a.takeUtterance(request.ID)
	}
	return err
}

// sending adds text to what an utterance has been sent, opening it if it is new.
func (a *Agent) sending(turnID, synthesisID, text string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	heard := a.heard[synthesisID]
	if heard == nil {
		heard = &utterance{agent: a, turnID: turnID, synthesisID: synthesisID}
		a.heard[synthesisID] = heard
	}
	heard.text += text
}

// utteranceOf is the utterance a synthesis is, nil for one the agent has no record of.
func (a *Agent) utteranceOf(synthesisID string) *utterance {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.heard[synthesisID]
}

// takeUtterance forgets an utterance that has finished, and returns it.
func (a *Agent) takeUtterance(synthesisID string) *utterance {
	a.mu.Lock()
	defer a.mu.Unlock()
	heard := a.heard[synthesisID]
	delete(a.heard, synthesisID)
	return heard
}

// heardBefore reports an utterance that is about to be reported spoken as being heard, if
// the track has not yet said so: a short one behind speech already queued finishes before
// the track gets to it.
func (a *Agent) heardBefore(heard *utterance) {
	if heard != nil && heard.published.Load() {
		a.announce(heard, time.Now())
	}
}
