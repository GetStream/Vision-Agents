package agent

import (
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// finishedKept is how many finished syntheses the agent remembers, so a delta sent for one
// after it settled does not open it again. Only the last few can be sent to: a turn's
// syntheses settle within it.
const finishedKept = 64

// MessageSpan is where an utterance is in the text of the Chat message its reply is written
// into, in Unicode code points: from Start up to End. The voice is sent other words than
// the message holds (a direction a voice acts, "last four digits, 1234" for "last four
// 1234", sentences without the space between them), so a page revealing the message as it
// is heard goes by this rather than by how much the voice said.
type MessageSpan struct {
	Start, End int
}

// utterance is one synthesis on its way to the caller, kept so that the moment it starts
// being heard can be reported with what it says (Speaking), and how much of it was heard
// if it is cut off (Interrupted).
type utterance struct {
	agent       *Agent
	turnID      string
	synthesisID string
	// text is what the voice has been sent for it, and message where that is in the
	// reply's message. Agent.mu guards both.
	text    string
	message MessageSpan
	// published is set once any of its audio went to the edge.
	published atomic.Bool
	// publishedUs is how much of its audio went to the edge, in microseconds.
	publishedUs atomic.Int64
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
		go u.agent.announce(u, at, 0)
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

// sent notes a chunk of it published to the edge, and reports whether it was the first.
// It runs once per chunk on the goroutine publishing, and costs two atomic operations.
func (u *utterance) sent(durationMs float64) (first bool) {
	if u == nil {
		return false
	}
	u.publishedUs.Add(int64(durationMs * 1000))
	return !u.published.Swap(true)
}

// heardBy is how much of it had been heard by now, in milliseconds: from when the track
// took its first audible frame, but never more than was published. False for one the track
// has not started on.
func (u *utterance) heardBy(now time.Time) (float64, bool) {
	if u == nil {
		return 0, false
	}
	pulled := u.pulledAt.Load()
	if pulled == 0 {
		return 0, false
	}
	heard := float64(now.UnixNano()-pulled) / 1e6
	return max(0, min(heard, float64(u.publishedUs.Load())/1000)), true
}

// bothMarks tells the turn's measurement and the utterance the same moments. The turn's
// are its first reply frame and close with the turn; an utterance's are each synthesis's
// own, so the two are kept apart rather than folded into one.
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
// track got to it; for that one, delayMs is how long after at it is expected to be heard.
func (a *Agent) announce(heard *utterance, at time.Time, delayMs float64) {
	heard.mu.Lock()
	defer heard.mu.Unlock()
	if heard.announced {
		return
	}
	heard.announced = true
	if pulled := heard.pulledAt.Load(); pulled != 0 {
		at, delayMs = time.Unix(0, pulled), 0
	}
	a.mu.Lock()
	text, message := heard.text, heard.message
	a.mu.Unlock()
	a.emitter.Send(Speaking{TurnID: heard.turnID, SynthesisID: heard.synthesisID, Text: text, Message: &message, At: at, DelayMs: delayMs})
}

// span is where the utterance is in its reply's message, nil for one the agent has no
// record of. It is read once the utterance is taken, when nothing changes it any more.
func (u *utterance) span() *MessageSpan {
	if u == nil {
		return nil
	}
	return &u.message
}

// synthesize sends a request to the voice, keeping what each utterance is sent and where
// that is in the reply's message. It is kept first, since a voice may send audio back
// before Synthesize returns. A request the voice refuses leaves nothing more of the
// utterance to be heard, so it is let go of.
func (a *Agent) synthesize(voice *ttsrouter.Session, turnID string, request tts.Request, message MessageSpan) error {
	a.sending(turnID, request.ID, request.Text, message)
	err := voice.Synthesize(request)
	if err != nil {
		a.takeUtterance(request.ID)
	}
	return err
}

// sending adds text to what an utterance has been sent, opening it if it is new, and
// stretches it to the end of message: a streaming voice's utterance is a reply's sentences
// one after another. One that already finished is not opened again: a delta sent after it
// settled is never heard, and would be kept forever.
func (a *Agent) sending(turnID, synthesisID, text string, message MessageSpan) {
	a.mu.Lock()
	defer a.mu.Unlock()
	heard := a.heard[synthesisID]
	if heard == nil {
		if _, done := a.finished[synthesisID]; done {
			return
		}
		heard = &utterance{agent: a, turnID: turnID, synthesisID: synthesisID, message: message}
		a.heard[synthesisID] = heard
	}
	heard.text += text
	heard.message.End = max(heard.message.End, message.End)
}

// utteranceLocked is the utterance a synthesis is, nil for one the agent has no record of.
// Agent.mu must be held.
func (a *Agent) utteranceLocked(synthesisID string) *utterance {
	return a.heard[synthesisID]
}

// takeUtterance forgets an utterance that has finished, and returns it.
func (a *Agent) takeUtterance(synthesisID string) *utterance {
	a.mu.Lock()
	defer a.mu.Unlock()
	heard := a.heard[synthesisID]
	delete(a.heard, synthesisID)
	if _, done := a.finished[synthesisID]; !done {
		a.finished[synthesisID] = struct{}{}
		a.finishedOrder = append(a.finishedOrder, synthesisID)
		if len(a.finishedOrder) > finishedKept {
			delete(a.finished, a.finishedOrder[0])
			a.finishedOrder = a.finishedOrder[1:]
		}
	}
	return heard
}

// forgetUtterancesLocked lets go of every utterance and of what finished, once what was
// saying them is gone: a voice lost, a speech-to-speech session ended, the agent closed. A
// voice that replaces a lost one may say the same synthesis again. Agent.mu must be held.
func (a *Agent) forgetUtterancesLocked() {
	clear(a.heard)
	clear(a.finished)
	a.finishedOrder = a.finishedOrder[:0]
}

// cutShort is the utterance of a turn being heard when it was cut off, how many
// milliseconds of it had been, and where all it had been sent is in the reply's message.
type cutShort struct {
	synthesisID string
	heardMs     float64
	message     *MessageSpan
}

// cut says so on an interruption, which is left saying nothing of it when cut is empty.
func (cut cutShort) cut(interrupted Interrupted) Interrupted {
	if cut.synthesisID != "" {
		interrupted.SynthesisID, interrupted.HeardMs, interrupted.Message = cut.synthesisID, cut.heardMs, cut.message
	}
	return interrupted
}

// cutShortLocked is the utterance of a turn being heard when it was cut off, empty for a
// turn none of whose open utterances the track had started on. Of several, it is the one
// the track took last. Agent.mu must be held.
func (a *Agent) cutShortLocked(turnID string, now time.Time) cutShort {
	var cut *utterance
	var latest int64
	for _, heard := range a.heard {
		if pulled := heard.pulledAt.Load(); heard.turnID == turnID && pulled > latest {
			cut, latest = heard, pulled
		}
	}
	if cut == nil {
		return cutShort{}
	}
	heardMs, _ := cut.heardBy(now)
	message := cut.message
	return cutShort{synthesisID: cut.synthesisID, heardMs: heardMs, message: &message}
}

// heardBefore reports an utterance that is about to be reported spoken as being heard, if
// the track has not yet said so: a short one behind speech already queued finishes before
// the track gets to it. ahead is how many milliseconds of speech are queued ahead of it,
// which is how long until it is heard.
func (a *Agent) heardBefore(heard *utterance, ahead func() float64) {
	if heard == nil || !heard.published.Load() {
		return
	}
	delayMs := 0.0
	if heard.pulledAt.Load() == 0 && ahead != nil {
		delayMs = ahead()
	}
	a.announce(heard, time.Now(), delayMs)
}

// queuedAhead is how many milliseconds of the speech waiting at the edge are ahead of the
// last ownMs of it, zero for an edge that does not count its queue.
func queuedAhead(stats PlayoutStats, counted bool, ownMs float64) float64 {
	if !counted {
		return 0
	}
	return max(0, float64(stats.Waiting.Microseconds())/1000-ownMs)
}

// messageText is a piece of the reply's text without directions as its message has it. The
// message starts at the reply's first word, and where each sentence is in it is counted
// from there.
func (a *Agent) messageText(plain string) string {
	if a.spoken.Len() == 0 {
		return strings.TrimLeftFunc(plain, unicode.IsSpace)
	}
	return plain
}

// sentencesOf gives the chunker more of the reply, keeping it for a voice that acts
// directions so that its sentences can be found in the message.
func (a *Agent) sentencesOf(text string) []sentence {
	if a.performs {
		a.performed = append(a.performed, []rune(text)...)
	}
	return a.chunk.Add(text)
}

// inMessage is where a sentence the chunker cut is in the reply's message. A voice that does
// not act directions is given the message's own text, so it is where the chunker found it.
// One that does is given the reply with its directions, and a position in that is how much
// of the message the text before it makes once they are taken out.
func (a *Agent) inMessage(cut sentence) MessageSpan {
	if !a.performs {
		return MessageSpan{Start: cut.start, End: cut.end}
	}
	return MessageSpan{Start: a.plainBefore(cut.start), End: a.plainBefore(cut.end)}
}

// plainBefore is how many characters of the message the first at characters of what a
// voice that acts directions was given come to.
func (a *Agent) plainBefore(at int) int {
	before := tts.Unperformed(string(a.performed[:min(at, len(a.performed))]))
	return utf8.RuneCountInString(strings.TrimLeftFunc(before, unicode.IsSpace))
}

// appendSpoken adds a line the agent says of its own accord to the reply, a filler or an
// apology, and returns where it is in the reply's message.
func (a *Agent) appendSpoken(line string) MessageSpan {
	start := utf8.RuneCountInString(a.spoken.String())
	a.spoken.WriteString(line)
	return MessageSpan{Start: start, End: start + utf8.RuneCountInString(line)}
}
