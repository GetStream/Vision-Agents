package agent

import (
	"cmp"
	"crypto/rand"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// linePrefix starts a line id, apart from the turn and murmur stamps.
const linePrefix = "line-"

// ActionKind names what the conversation decided to do next.
type ActionKind string

const (
	// ActAsk puts a settled turn to the flow controller.
	ActAsk ActionKind = "ask"
	// ActWait leaves a turn alone because the caller has not finished the thought.
	ActWait ActionKind = "wait"
	// ActIgnore drops a turn that was not meant for the agent.
	ActIgnore ActionKind = "ignore"
	// ActAnswer replies to a turn.
	ActAnswer ActionKind = "answer"
	// ActQueue holds a turn back until the agent has stopped talking.
	ActQueue ActionKind = "queue"
	// ActHold leaves a turn the agent owes unanswered while the caller is talking, so the
	// agent does not take the floor from somebody who still has it.
	ActHold ActionKind = "hold"
	// ActInterrupt abandons the reply being spoken.
	ActInterrupt ActionKind = "interrupt"
	// ActShorten stops the model adding more while letting the audio already sent finish.
	ActShorten ActionKind = "shorten"
	// ActBackchannel makes a short listening noise.
	ActBackchannel ActionKind = "backchannel"
	// ActCheckIn speaks to a caller who has heard nothing for a while: whether they need
	// anything else, or that the work they are waiting on is still going.
	ActCheckIn ActionKind = "checkin"
	// ActSupersede abandons a ruling asked for about words that have since changed.
	ActSupersede ActionKind = "supersede"
	// ActCompact replaces an old stretch of history with a summary.
	ActCompact ActionKind = "compact"
	// ActDelegate hands a piece of work to the subagent.
	ActDelegate ActionKind = "delegate"
	// ActSettle is delegated work coming back, answered or not.
	ActSettle ActionKind = "settle"
	// ActFail is a judgement that could not be made.
	ActFail ActionKind = "fail"
)

const (
	// interruptGrace is how much longer the turn after an overlap is given to hold still.
	// Somebody talking over somebody else is as often a connection running late as it is a
	// change of mind, and words that arrive late are answered half-said if the next turn
	// is settled at the usual pace.
	interruptGrace = 150 * time.Millisecond
	// defaultPatience is how long the same unfinished words are waited on before the
	// caller is asked what they meant instead.
	defaultPatience = 3 * time.Second
	// callerHold is how recently the caller's words must have changed for them to still
	// hold the floor. A revision younger than this means somebody is mid-utterance, so a
	// turn the agent owes waits rather than talking over them.
	callerHold = 1500 * time.Millisecond
	// rulingWithin is how long the flow controller takes at most to rule on words, its own
	// deadline (harness flowDeadline).
	rulingWithin = 3 * time.Second
)

const (
	// overlapPrefix marks a provisional candidate, put to the controller while the caller
	// is still talking, so it is never confused with a settled one.
	overlapPrefix = "over-"
	// overlapAsks caps how many provisional asks one utterance earns. After that the
	// settled ask decides, as it does when nobody talks over the agent.
	overlapAsks = 3
	// overlapReaskWords is how many more words must arrive before the same utterance is
	// asked about again, which bounds the cost of somebody talking over the agent at length.
	overlapReaskWords = 3
)

// What the model is told about a turn it is answering with a question rather than an
// answer. There are two reasons to ask, and they are not the same question.
const (
	ambiguousNote = "The caller addressed you, but their meaning is ambiguous. " +
		"Ask one short clarifying question."
	unfinishedNote = "The caller has gone quiet, and you may not have heard all of what " +
		"they meant to say. If what you heard is enough to act on, answer it; otherwise ask " +
		"briefly for the part that is missing."
)

// Action is one thing the conversation decided to do. It says what and why; carrying it
// out is the agent's part, because doing any of these means touching a provider session.
type Action struct {
	Kind ActionKind
	// Reason is why, in words, for the log and for whoever is watching the call.
	Reason string
	// Candidate is the turn being answered or held, for the kinds that have one.
	Candidate candidate
	// Participant is who the action concerns.
	Participant stt.Participant
	// Text is what was heard, or the murmur to make.
	Text string
	// Compose is what the model is asked to write a line for, when the words are only
	// settled at the moment they are said.
	Compose string
	// TurnID is the reply being interrupted, shortened or abandoned.
	TurnID string
	// Clarify is what the model is told when the turn is owed a short question rather than
	// an answer. Empty on a turn that is simply answered.
	Clarify string
	// LatencyMs is what the flow controller took to rule. Zero where nothing was asked.
	LatencyMs float64
	// Unheard says none of the interrupted reply has gone out, and it is only abandoned so.
	Unheard bool
	// LineID is the caller's line a turn queued, held or answered was heard as, so a
	// watcher can say of that line that it waits for the agent to finish.
	LineID string
	// Unanswered is what the same participant said in a turn queued before this one, which
	// this answer is for too. Empty when nothing was waiting.
	Unanswered string
	// Err is why a judgement could not be made.
	Err error
}

// floor is what the agent is doing at the moment a judgement has to be made.
//
// It is passed in rather than held, because it belongs to the machinery that speaks: the
// conversation decides what should happen, and only the agent knows whether a synthesis
// is still in flight.
type floor struct {
	// Quiet reports whether the agent has stopped talking.
	Quiet bool
	// Recovering reports that a lost voice is still being replaced.
	Recovering bool
	// Speaking is the reply currently allowed to make audio, empty when there is none.
	Speaking string
	// Talking reports whether the agent is writing or saying a reply, which Quiet cannot
	// say on its own because a tool that is running also leaves the agent not quiet.
	Talking bool
	// Working names the work the caller is waiting on, a tool or the subagent, empty
	// when nothing is running.
	Working string
	// Owed reports whether something came back that the caller has not been told yet.
	Owed bool
	// Composing reports whether a line the agent chose to say is still being written, which
	// another would only repeat.
	Composing bool
	// LastSpokeAt is when the agent last published audio.
	LastSpokeAt time.Time
	// LastHeardAt is when anyone on the call was last transcribed.
	LastHeardAt time.Time
	// LastParticipant is who the agent was last talking to.
	LastParticipant stt.Participant
	// Asked reports that the agent's last reply asked the caller something, so a silence
	// after it is an answer that has not come.
	Asked bool
	// Unheard is the caller turn being answered while none of its reply has gone out.
	Unheard unheardReply
}

// active is when anything last happened on the call, either end of it. It is zero on a
// call where nobody has spoken and the agent has yet to say a word.
func (f floor) active() time.Time {
	if f.LastHeardAt.After(f.LastSpokeAt) {
		return f.LastHeardAt
	}
	return f.LastSpokeAt
}

// converse holds the judgements a conversation makes.
//
// Every one of them used to live next to the code that acted on it, which meant the
// reasoning behind a call was spread across six functions and only visible as debug lines
// nobody had switched on. They are gathered here instead: when a caller has finished
// speaking, whether what they said was meant for the agent, what to do about hearing them
// while talking, when a silence needs filling, and when the history has grown expensive
// enough to summarise. Each one is logged once and reported as a Decided event, so a call
// can be followed rather than guessed at.
type converse struct {
	cadence *cadence
	duplex  *duplex
	emitter *Emitter
	logger  *slog.Logger
	// patience is how long the same unfinished words are waited on before the caller is
	// asked what they meant.
	patience time.Duration
	// undecidedFor is how long words that have held still keep a turn nobody asked for
	// waiting: patience and the rulings it takes, after which their ruling is taken as lost.
	undecidedFor time.Duration
	// record keeps the trail after the process holding it is gone. Nil when the
	// deployment has nowhere to put it, in which case the decisions are still logged and
	// still reported live.
	record func(Decided)

	mu sync.Mutex
	// candidates are settled turns waiting on the flow controller.
	candidates map[string]candidate
	// queued is a relevant turn heard while the agent chose to finish speaking.
	queued *queuedCandidate
	// delegated is the turn each piece of delegated work was asked for in, so what comes
	// back lands against the exchange that wanted it. A subagent's result names the task
	// and not the turn, and by the time it arrives the conversation has usually moved on.
	delegated map[string]string
	// waiting is the unfinished words each participant is being given longer to finish.
	waiting map[string]unfinished
	// wentOn is when each participant last went on talking before hearing any of the reply
	// to what they had said, until a turn of theirs is answered. The flow controller is
	// told, because the next part that sounds complete may be no more the end than the last.
	wentOn map[string]time.Time
	// overlaps names the participant each provisional ask still with the controller is
	// about, so a ruling can find what was asked.
	overlaps map[string]string
	// overlapping is what has been asked about each participant's in-progress utterance,
	// so a caller talking over the agent at length is asked about at a bounded rate
	// rather than on every revision.
	overlapping map[string]overlapState
	// reported is the last ask and the last wait written down, so words that have not
	// changed are not judged out loud again on every retry.
	reported map[ActionKind]judged
	// lines is the line each participant's words are going into, from the first of them
	// until they are heard or let go, as the transcript keeps one being written per speaker.
	lines map[string]string
	// grown is the participants whose open line is one already heard that they carried on
	// with: it is that line whole, so nothing in it is a repeat of another.
	grown map[string]bool
	// lastHeard is the turn each participant's last heard line was heard on, which a
	// transcriber going over that run of speech sends again in front of what is new.
	lastHeard map[string]string
	// repeated is what each participant's open line opens with that was already heard as
	// a line before it, decided when the line opens and kept for it: decided again on
	// every revision, a line could show its first words and then lose them.
	repeated map[string]string
	// queuedSaid is what a turn queued before it was heard has put in a participant's
	// open line. Words that follow in a run of speech of their own go after it rather
	// than over it: the turn is still owed, and its words are still the line's.
	queuedSaid map[string]string
}

// overlapState is what has been asked about a participant's in-progress utterance.
type overlapState struct {
	participant stt.Participant
	// text is the caller's evolving words as of the last observation.
	text string
	// askedText is what the most recent provisional ask was about, and askedWords how
	// many words that was.
	askedText  string
	askedWords int
	// asks is how many provisional asks this utterance has earned.
	asks int
	// inflight is the provisional candidate still with the controller, if any, and turnID
	// the reply it was asked against.
	inflight string
	turnID   string
}

// queuedCandidate is a turn held back until the agent stops talking.
type queuedCandidate struct {
	candidate candidate
	clarify   string
	// heard says the turn was already reported as what the caller said. One queued before
	// it was ruled on is reported once it is answered.
	heard bool
	// line is the line the turn was heard as, or for one not yet heard, the line it was
	// being said in when it was queued, and said is what that line showed of it then.
	line string
	said string
}

// unfinished is a thought the controller keeps wanting to wait on, and when the waiting
// for it began.
type unfinished struct {
	text  string
	since time.Time
}

// judged is what a judgement was about: who was speaking and what they said.
type judged struct {
	participant string
	text        string
}

func newConverse(
	settling *cadence,
	listening *duplex,
	emitter *Emitter,
	record func(Decided),
	patience time.Duration,
	logger *slog.Logger,
) *converse {
	if patience <= 0 {
		patience = defaultPatience
	}
	if logger == nil {
		logger = slog.Default()
	}
	return &converse{
		cadence:      settling,
		duplex:       listening,
		emitter:      emitter,
		logger:       logger,
		patience:     patience,
		undecidedFor: patience + 2*rulingWithin + callerHold,
		record:       record,
		candidates:   map[string]candidate{},
		delegated:    map[string]string{},
		waiting:      map[string]unfinished{},
		wentOn:       map[string]time.Time{},
		overlaps:     map[string]string{},
		overlapping:  map[string]overlapState{},
		reported:     map[ActionKind]judged{},
		lines:        map[string]string{},
		grown:        map[string]bool{},
		lastHeard:    map[string]string{},
		queuedSaid:   map[string]string{},
		repeated:     map[string]string{},
	}
}

// Observe takes a transcript revision and says what it changed.
//
// Two things can come of hearing more words: they may be worth acknowledging out loud,
// and they may make a ruling already in flight about the words as they were pointless.
func (c *converse) Observe(transcript stt.Transcript, state floor) []Action {
	var actions []Action

	if phrase := c.duplex.Heard(transcript.Participant, transcript.Text, state.Quiet); phrase != "" {
		actions = append(actions, c.decide(Action{
			Kind:        ActBackchannel,
			Reason:      "the caller has said enough that letting them know you are still there is worth it",
			Participant: transcript.Participant,
			Text:        phrase,
		}))
	}

	superseded, saying := c.cadence.Observe(transcript)
	if saying != "" {
		c.mu.Lock()
		released, grown := c.forgetQueuedGrownLocked(transcript.Participant, transcript.Utterance, saying)
		if grown {
			c.grown[transcript.Participant.ID] = true
		}
		if _, open := c.lines[transcript.Participant.ID]; !open {
			c.repeated[transcript.Participant.ID] = c.repeatOfLocked(
				transcript.Participant.ID, saying, c.cadence.Repeating(transcript.Participant.ID))
		}
		line := c.lineLocked(transcript.Participant.ID)
		shown := c.lineTextLocked(transcript.Participant.ID, saying)
		c.mu.Unlock()
		if released != "" {
			c.cadence.ReleaseHeld(released)
		}
		// The words so far may be only the ones already heard, said again on the way
		// to more: there is nothing of this line to show until the more arrives.
		if shown != "" {
			c.emitter.Send(Hearing{
				Participant: transcript.Participant,
				Text:        shown,
				Language:    transcript.Language,
				LineID:      line,
			})
		}
		if stale, ok := c.stale(transcript, saying, state); ok {
			actions = append(actions, stale)
		} else {
			actions = append(actions, c.overlap(transcript, saying, state)...)
		}
	}
	if superseded != "" {
		c.mu.Lock()
		delete(c.candidates, superseded)
		c.mu.Unlock()
		actions = append(actions, c.supersede(transcript.Participant, superseded))
	}

	return actions
}

// overlap asks the controller whether the agent should give up the floor to a caller who
// is talking over it, while their words are still in progress.
//
// It is asked from transcript revisions rather than from settled turns, so the floor can be
// ruled on while the caller is still speaking rather than only once they pause. The cost is
// bounded: only while the agent holds the floor, at most overlapAsks per utterance, and only
// once the words have grown by overlapReaskWords since the last ask. An ask still in flight
// is replaced rather than waited on, so "okay" does not occupy the controller while the
// caller has already said "wait, make it six".
func (c *converse) overlap(transcript stt.Transcript, saying string, state floor) []Action {
	if state.Quiet || state.Speaking == "" || transcript.Final() {
		return nil
	}
	// A murmur is meant to overlap, so hearing the caller carry on is not a reason to stop:
	// there is no reply to abandon.
	if strings.HasPrefix(state.Speaking, backchannelPrefix) {
		return nil
	}
	if overlapNoise(saying) {
		return nil
	}

	participantID := transcript.Participant.ID
	fields := strings.Fields(words(saying))
	count := len(fields)
	// One word is too little to rule on unless it is itself an interruption, and Gemma ruled a
	// child's "Mom" a stop. The next revision is a fraction of a second away.
	if count == 1 && !interruptsAlone[strings.ToLower(fields[0])] {
		return nil
	}

	c.mu.Lock()
	seen := c.overlapping[participantID]
	var superseded string
	if seen.askedText != "" && !revisesTranscript(seen.askedText, saying) {
		// A new utterance rather than more of the one asked about: the ask in flight is
		// about words that are gone, and the count starts again.
		superseded = seen.inflight
		seen = overlapState{}
	}
	seen.participant = transcript.Participant
	seen.text = saying
	ask := seen.asks < overlapAsks &&
		(seen.asks == 0 || count-seen.askedWords >= overlapReaskWords)
	var id string
	if ask {
		if seen.inflight != "" {
			superseded = seen.inflight
		}
		id = overlapPrefix + turnStamp()
		seen.inflight, seen.turnID = id, state.Speaking
		seen.askedText, seen.askedWords = saying, count
		seen.asks++
		c.overlaps[id] = participantID
	}
	c.overlapping[participantID] = seen
	if superseded != "" {
		delete(c.overlaps, superseded)
	}
	c.mu.Unlock()

	var actions []Action
	if superseded != "" {
		actions = append(actions, c.supersede(transcript.Participant, superseded))
	}
	if !ask {
		return actions
	}
	return append(actions, c.decide(Action{
		Kind:   ActAsk,
		Reason: "the caller is talking over the agent, asking whether it should stop",
		Candidate: candidate{
			ID:          id,
			Participant: transcript.Participant,
			Utterance:   transcript.Utterance,
			Text:        saying,
			Language:    transcript.Language,
			Unfinished:  true,
		},
		Participant: transcript.Participant,
		Text:        saying,
	}))
}

// stale abandons the reply to a caller's turn once the same caller says more before any of it
// has gone out: it answers words they moved on from, over them. What they say next is answered
// with those words in the history. A murmur or a lone word that is no interruption is somebody
// waiting to be answered, and leaves it.
func (c *converse) stale(transcript stt.Transcript, saying string, state floor) (Action, bool) {
	if state.Unheard.turnID == "" || transcript.Participant.ID != state.Unheard.participant ||
		!takesFloor(unrepeated(saying, c.cadence.Repeating(transcript.Participant.ID))) {
		return Action{}, false
	}
	c.mu.Lock()
	c.wentOn[transcript.Participant.ID] = time.Now()
	c.mu.Unlock()
	return c.decide(Action{
		Kind:        ActInterrupt,
		Reason:      "the caller said more before hearing any of the reply, so it answers words they have moved on from",
		Participant: transcript.Participant,
		TurnID:      state.Unheard.turnID,
		Text:        saying,
		Unheard:     true,
	}), true
}

// takesFloor reports whether words are more than a murmur: a cough, "mm", a lone non-interruption,
// or nothing but assent ("okay okay", "uh huh", "got it"), which is somebody listening.
func takesFloor(saying string) bool {
	if overlapNoise(saying) {
		return false
	}
	fields := strings.Fields(strings.ToLower(words(saying)))
	if len(fields) == 1 {
		return interruptsAlone[fields[0]]
	}
	return slices.ContainsFunc(fields, func(field string) bool {
		_, murmur := murmurs[field]
		return !murmur
	})
}

// supersede abandons a ruling asked for about words that have since changed.
func (c *converse) supersede(participant stt.Participant, candidateID string) Action {
	return c.decide(Action{
		Kind:        ActSupersede,
		Reason:      "the caller said more, so the ruling in flight is about words that have changed",
		Participant: participant,
		TurnID:      candidateID,
	})
}

// Settled registers a turn whose words have stopped changing and asks for a ruling on it.
func (c *converse) Settled(ready candidate, state floor) Action {
	c.mu.Lock()
	if !state.Quiet {
		if seen, ok := c.overlapping[ready.Participant.ID]; ok && seen.inflight != "" {
			line := c.lines[ready.Participant.ID]
			said := c.lineTextLocked(ready.Participant.ID, ready.Text)
			c.mu.Unlock()
			// Retrying cadence here used to re-Settle every 700ms for as long as the
			// overlap Create sat behind a Cerebras generate, so a follow-up never left
			// the loop. Queue it once; Waiting answers it when the floor is quiet.
			if !c.cadence.Hold(ready) {
				return Action{}
			}
			c.mu.Lock()
			c.queuedSaid[ready.Participant.ID] = said
			c.mu.Unlock()
			c.hold(queuedCandidate{candidate: ready, line: line, said: said})
			return c.decide(Action{
				Kind:        ActQueue,
				Reason:      "the floor is already being asked about, so the settled turn waits",
				Candidate:   ready,
				Participant: ready.Participant,
				Text:        ready.Text,
				LineID:      line,
			})
		}
	}
	ready.OverAgent = !state.Quiet
	c.candidates[ready.ID] = ready
	c.mu.Unlock()

	reason := "the words held still, asking whether they are meant for the agent"
	if !state.Quiet {
		reason = "the words held still while the agent was talking, asking who should have the floor"
	}
	return c.decide(Action{
		Kind:        ActAsk,
		Reason:      reason,
		Candidate:   ready,
		Participant: ready.Participant,
		Text:        ready.Text,
	})
}

// Unasked forgets a turn the flow controller could not be asked about, so the caller gets
// another chance at it rather than losing it to a failure they had no part in.
func (c *converse) Unasked(candidateID string) {
	c.mu.Lock()
	delete(c.candidates, candidateID)
	c.forgetOverlapLocked(candidateID)
	c.mu.Unlock()
	c.cadence.Resolve(candidateID, true)
}

// forgetOverlapLocked drops a provisional ask and returns what has been asked about the
// participant it concerned, or false when the id is not a provisional ask still in flight.
// The caller holds the lock.
func (c *converse) forgetOverlapLocked(candidateID string) (overlapState, bool) {
	participantID, ok := c.overlaps[candidateID]
	if !ok {
		return overlapState{}, false
	}
	delete(c.overlaps, candidateID)
	seen := c.overlapping[participantID]
	if seen.inflight == candidateID {
		seen.inflight = ""
		c.overlapping[participantID] = seen
	}
	return seen, true
}

// Ruled turns the flow controller's answer into what the agent should do about it.
//
// The actions come back in the order they must run: a turn taken from the agent is
// interrupted before it is answered, and a turn the agent talks through is held before
// the reply it is waiting on is cut short.
func (c *converse) Ruled(ruling harness.Decided, state floor) (actions []Action) {
	c.mu.Lock()
	seen, provisional := c.forgetOverlapLocked(ruling.CandidateID)
	c.mu.Unlock()
	if provisional {
		return c.overlapRuled(ruling, seen, state)
	}

	c.mu.Lock()
	ready, known := c.candidates[ruling.CandidateID]
	delete(c.candidates, ruling.CandidateID)
	c.mu.Unlock()
	if !known {
		c.logger.Debug("a ruling arrived for a turn that has moved on",
			"candidate", ruling.CandidateID)
		return nil
	}
	if ready.OverAgent {
		defer func() {
			settledOverAgent.WithLabelValues(wordCount(ready.Said()), strconv.FormatBool(ready.Final),
				floorOutcome(actions)).Inc()
		}()
	}

	if ruling.Err != nil || !ruling.Valid() {
		if ready.Unfinished {
			// A ruling that cannot be read must not cost the caller their turn, so the
			// words go back to settling and are put again once they have.
			c.cadence.Resolve(ruling.CandidateID, true)
			return []Action{c.decide(Action{
				Kind:        ActFail,
				Reason:      "the flow controller did not answer, giving the caller longer to finish",
				Candidate:   ready,
				Participant: ready.Participant,
				Text:        ready.Text,
				LatencyMs:   ruling.TookMs,
				Err:         ruling.Error(),
			})}
		}
		// A finished thought is still the caller's turn. Putting it back to
		// settling is how a slow controller never
		// answers and never records what they said.
		ruling.Disposition = harness.Respond
		if !ruling.Floor.Valid() {
			// Nobody ruled on who has the floor. Holding a finished thought until the agent
			// stops talking cost callers up to ten seconds, so they are given it: noise
			// spoken over the agent is still set aside below.
			ruling.Floor = harness.Stop
		}
	}

	clarify, clarified := "", ""
	if ruling.Disposition == harness.Clarify {
		clarify = ambiguousNote
		clarified = "the caller addressed the agent but their meaning is ambiguous"
	}

	if ruling.Disposition == harness.Wait {
		if c.patient(ready) {
			c.cadence.Resolve(ruling.CandidateID, true)
			return []Action{c.decide(Action{
				Kind:        ActWait,
				Reason:      "the caller has not finished the thought",
				Candidate:   ready,
				Participant: ready.Participant,
				Text:        ready.Text,
				LatencyMs:   ruling.TookMs,
			})}
		}
		// Waiting again would be waiting for good, and a thought that never arrives is as
		// likely to be one the transcriber mangled as one the caller abandoned. Either way
		// the answer is to say something rather than to keep listening to silence.
		clarify = unfinishedNote
		clarified = "the caller went quiet on an unfinished thought, so what was heard may not be what was said"
	}

	queue := !state.Quiet && !overlapNoise(ready.Text) && ruling.Disposition != harness.Ignore &&
		(state.Recovering || ruling.Floor != harness.Stop)
	var resolved bool
	if queue {
		resolved = c.cadence.Hold(ready)
	} else {
		resolved = c.cadence.Resolve(ruling.CandidateID, false)
	}
	if !resolved {
		c.logger.Debug("not acting on a ruling, the words changed while it was being made",
			"candidate", ruling.CandidateID)
		return nil
	}

	// The words are dealt with, so the next unfinished thought starts its own wait, and any
	// provisional asks about this utterance are done with.
	c.mu.Lock()
	delete(c.waiting, ready.Participant.ID)
	if seen, ok := c.overlapping[ready.Participant.ID]; ok {
		delete(c.overlaps, seen.inflight)
		delete(c.overlapping, ready.Participant.ID)
	}
	c.mu.Unlock()

	ignore := ""
	switch {
	case !state.Quiet && overlapNoise(ready.Text):
		ignore = "a cough or other non-speech noise overlapping the reply, so the agent keeps talking"
	case ruling.Disposition == harness.Ignore:
		ignore = "this was background speech or meant for somebody else"
	}
	if ignore != "" {
		// The transcript lets go of the words it was writing, and so does the line. What is
		// let go of is this line, not the one heard before it.
		shown, line := c.closeLine(ready.Participant.ID, ready.Text, false)
		return []Action{c.decide(Action{
			Kind:        ActIgnore,
			Reason:      ignore,
			Candidate:   ready,
			Participant: ready.Participant,
			Text:        shown,
			LatencyMs:   ruling.TookMs,
			LineID:      line,
		})}
	}

	// The turn counts, which is the point at which it becomes what the caller said rather
	// than what they might have been saying.
	shown, line := c.closeLine(ready.Participant.ID, ready.Text, true)
	var unanswered string
	if !queue {
		unanswered = c.forgetQueuedAnsweredOver(ready)
	}
	c.mu.Lock()
	released := c.forgetQueuedRevisedByLocked(ready)
	c.mu.Unlock()
	if released != "" {
		c.cadence.ReleaseHeld(released)
	}
	c.emitter.Send(Heard{
		Participant: ready.Participant,
		Text:        shown,
		Language:    ready.Language,
		LineID:      line,
	})

	c.mu.Lock()
	delete(c.wentOn, ready.Participant.ID)
	c.mu.Unlock()

	answer := Action{
		Kind:        ActAnswer,
		Reason:      "a complete thought addressed to the agent",
		Candidate:   ready,
		Participant: ready.Participant,
		Text:        ready.Text,
		Clarify:     clarify,
		LatencyMs:   ruling.TookMs,
		LineID:      line,
		Unanswered:  unanswered,
	}
	if clarify != "" {
		answer.Reason = clarified
	}

	if state.Quiet {
		return []Action{c.decide(answer)}
	}

	// Somebody spoke over the agent, so who keeps the floor has to be settled before the
	// turn can be dealt with at all.
	c.emitter.Send(OverlapDecided{
		TurnID:      state.Speaking,
		Participant: ready.Participant,
		Action:      string(ruling.Floor),
	})
	// Two people were talking at once, whichever of them gives way. The turn after that is
	// given longer to settle, because a caller heard over the agent is a caller whose audio
	// may be arriving late.
	c.cadence.Grace(interruptGrace)

	switch ruling.Floor {
	case harness.Stop:
		answering := answer
		if queue {
			answering.Kind = ActQueue
			answering.Reason = "the voice is being replaced, so the caller's answer waits until it is ready"
			c.hold(queuedCandidate{candidate: ready, clarify: clarify, heard: true, line: line})
		}
		return []Action{
			c.decide(Action{
				Kind:        ActInterrupt,
				Reason:      "the caller corrected or interrupted the agent, so the reply is abandoned",
				Participant: ready.Participant,
				TurnID:      state.Speaking,
				LatencyMs:   ruling.TookMs,
			}),
			c.decide(answering),
		}

	case harness.Shorten:
		held := answer
		held.Kind = ActQueue
		held.Reason = "the caller added to what they asked, so the reply in flight is cut short"
		c.hold(queuedCandidate{candidate: ready, clarify: clarify, heard: true, line: line})
		return []Action{
			c.decide(held),
			c.decide(Action{
				Kind:        ActShorten,
				Reason:      "the answer being spoken has become too long for what was asked",
				Participant: ready.Participant,
				TurnID:      state.Speaking,
				LatencyMs:   ruling.TookMs,
			}),
		}

	default:
		held := answer
		held.Kind = ActQueue
		held.Reason = "a brief acknowledgement or unrelated speech, so the agent finishes what it was saying"
		c.hold(queuedCandidate{candidate: ready, clarify: clarify, heard: true, line: line})
		return []Action{c.decide(held)}
	}
}

// wordCount buckets a turn's words: one word is what is most often misruled over the agent.
func wordCount(text string) string {
	switch n := len(strings.Fields(words(text))); {
	case n <= 1:
		return "1"
	case n <= 3:
		return "2-3"
	}
	return "4+"
}

// floorOutcome is what came of a ruling: the first thing decided, or the shortening queued
// ahead of, and "none" when the words moved on first.
func floorOutcome(actions []Action) string {
	outcome := "none"
	for i, action := range actions {
		if i == 0 || action.Kind == ActShorten {
			outcome = string(action.Kind)
		}
	}
	return outcome
}

// overlapRuled turns a provisional ruling — made while the caller was still talking — into
// what the agent should do about the floor.
//
// Unlike a settled ruling it never answers: the words have not settled, so when they do they
// go through Settled and Ruled with a quiet floor and yield the ordinary answer. It is checked
// against the reply it was asked about, because revisions arrive faster than a controller
// round trip and the reply may have ended by itself. A ruling about words that have since
// changed never gets this far: Observe forgets the ask when it supersedes it.
func (c *converse) overlapRuled(ruling harness.Decided, seen overlapState, state floor) []Action {
	if ruling.Err != nil || !ruling.Valid() {
		c.logger.Debug("dropping a provisional ruling that could not be read",
			"candidate", ruling.CandidateID, "error", ruling.Error())
		return nil
	}
	if state.Quiet || state.Speaking != seen.turnID {
		c.logger.Debug("not acting on a provisional ruling, the moment has passed",
			"candidate", ruling.CandidateID, "quiet", state.Quiet,
			"speaking", state.Speaking, "asked", seen.turnID)
		return nil
	}

	// Background speech does not stop the agent, whatever floor the controller returned.
	floorDecision := ruling.Floor
	if ruling.Disposition == harness.Ignore {
		floorDecision = harness.Continue
	}
	if floorDecision == harness.Continue {
		c.logger.Debug("a provisional ruling let the agent keep the floor",
			"candidate", ruling.CandidateID, "floor", ruling.Floor)
		return nil
	}

	c.emitter.Send(OverlapDecided{
		TurnID:      seen.turnID,
		Participant: seen.participant,
		Action:      string(floorDecision),
	})
	if floorDecision == harness.Shorten {
		return []Action{c.decide(Action{
			Kind:        ActShorten,
			Reason:      "the caller is adding to what they asked, so the reply in flight is cut short",
			Participant: seen.participant,
			TurnID:      seen.turnID,
			LatencyMs:   ruling.TookMs,
		})}
	}

	// The caller has the floor, so what was asked about this utterance is done with.
	c.mu.Lock()
	delete(c.overlapping, seen.participant.ID)
	c.mu.Unlock()
	c.cadence.Grace(interruptGrace)
	return []Action{c.decide(Action{
		Kind:        ActInterrupt,
		Reason:      "the caller is still talking and has already said enough to take the floor, so the reply is abandoned",
		Participant: seen.participant,
		TurnID:      seen.turnID,
		LatencyMs:   ruling.TookMs,
	})}
}

// idle invites a caller who has gone quiet back into the conversation, because a silence
// that nobody breaks is how a call ends by accident rather than because it was over.
func (c *converse) idle(state floor, participant stt.Participant) []Action {
	if !c.duplex.Idle(state.active(), state.Quiet, state.Asked) {
		return nil
	}
	if state.Asked {
		return []Action{c.decide(Action{
			Kind:        ActCheckIn,
			Reason:      "the caller has not answered the agent's question, checking they are still there",
			Participant: participant,
			Compose:     answerPurpose,
		})}
	}
	return []Action{c.decide(Action{
		Kind:        ActCheckIn,
		Reason:      "nobody has said anything for a while, asking whether there is anything else",
		Participant: participant,
		Compose:     idlePurpose,
	})}
}

// update tells a caller waiting on work that it is still going, once they have heard
// nothing for long enough to wonder.
func (c *converse) update(state floor, participant stt.Participant) []Action {
	if state.Talking {
		return nil
	}
	if !c.duplex.Update(state.Working, state.LastSpokeAt) {
		return nil
	}
	return []Action{c.decide(Action{
		Kind:        ActCheckIn,
		Reason:      "work the caller was promised is still running and they have heard nothing for a while",
		Participant: participant,
		Compose:     updatePurpose,
	})}
}

// Tick decides whether a long listening or thinking gap needs filling, so an agent that
// is busy does not sound like a dead line.
func (c *converse) Tick(state floor) []Action {
	participant, _, hearing := c.cadence.Active()
	if !hearing {
		participant = state.LastParticipant
		switch {
		case state.Composing:
			return nil
		case state.Owed:
			// The answer is on its way to them, and asking whether there is anything
			// else would talk over the thing they asked for.
			return nil
		case state.Working != "":
			return c.update(state, participant)
		}
		return c.idle(state, participant)
	}

	phrase := c.duplex.Presence(participant, state.LastSpokeAt, state.Quiet)
	if phrase == "" {
		return nil
	}

	return []Action{c.decide(Action{
		Kind:        ActBackchannel,
		Reason:      "the caller has been talking a while without hearing anything back",
		Participant: participant,
		Text:        phrase,
	})}
}

// Listening reports whether somebody has evolving words younger than callerHold, which is
// a caller who still holds the floor. A turn the agent owes waits while they do.
func (c *converse) Listening() bool {
	_, revised, hearing := c.cadence.Active()
	return hearing && time.Since(revised) < callerHold
}

// Undecided reports whether somebody has said something the agent has not yet dealt with:
// words still being gathered or put to the controller, or a turn queued behind the reply. A
// turn nobody asked for waits for them, because it would be said over a caller who spoke
// first and answer none of what they said. Every ruling but a wait settles the words, and a
// wait runs out with patience; words that have held still for longer than that and the
// rulings it takes are taken to have lost theirs, and no longer hold up what is owed.
func (c *converse) Undecided() bool {
	if _, revised, hearing := c.cadence.Active(); hearing && time.Since(revised) < c.undecidedFor {
		return true
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.queued != nil
}

// Waiting hands back the turn held while the agent finished speaking, now that it has.
//
// A caller who has started talking again keeps the floor: the turn stays queued, so the
// agent does not answer over the top of them. The hold is reported once rather than on
// every retry.
func (c *converse) Waiting(state floor) (Action, bool) {
	if !state.Quiet {
		return Action{}, false
	}

	c.mu.Lock()
	if c.queued == nil {
		c.mu.Unlock()
		return Action{}, false
	}
	// A copy: what is queued changes with the line it is in, under the lock.
	held := *c.queued
	listening := c.Listening()
	if !listening {
		if !c.cadence.CommitHeld(held.candidate.ID) {
			c.queued = nil
			c.mu.Unlock()
			return Action{}, false
		}
		c.queued = nil
	}
	c.mu.Unlock()

	line := held.line
	if listening {
		if line == "" {
			c.mu.Lock()
			line = c.lines[held.candidate.Participant.ID]
			c.mu.Unlock()
		}
		c.decide(Action{
			Kind:        ActHold,
			Reason:      "the caller is talking, so the turn the agent owes waits",
			Candidate:   held.candidate,
			Participant: held.candidate.Participant,
			Text:        held.candidate.Text,
			LineID:      line,
		})
		return Action{}, false
	}
	if !held.heard {
		// Answering it is what makes it count, as Ruled says of a turn ruled on, and a
		// transcript waits on this to close the caller's line: without it their next words
		// are written over this one.
		var shown string
		shown, line = c.closeQueuedLine(held)
		c.emitter.Send(Heard{
			Participant: held.candidate.Participant,
			Text:        shown,
			Language:    held.candidate.Language,
			LineID:      line,
		})
	}
	return c.decide(Action{
		Kind:        ActAnswer,
		Reason:      "the agent has stopped talking, so the turn that was waiting can be answered",
		Candidate:   held.candidate,
		Participant: held.candidate.Participant,
		Text:        held.candidate.Text,
		Clarify:     held.clarify,
		LineID:      line,
	}), true
}

// Compact decides whether the conversation has grown expensive enough to summarise.
//
// What counts as expensive is the harness's to judge, because it is the harness that
// watches the prefix cache stop paying and the harness that runs the summary. What the
// conversation owns is that it happened at all, which is the kind of thing somebody
// reading back a call needs to see: a history replaced part way through explains an agent
// that forgot a detail from ten minutes ago.
func (c *converse) Compact(current *harness.Harness, history []llm.Message, inputTokens, cachedTokens int64) error {
	if current == nil {
		return nil
	}
	started, err := current.MaybeCompact(history, inputTokens, cachedTokens)
	if err != nil {
		return err
	}
	if !started {
		return nil
	}
	c.decide(Action{
		Kind:   ActCompact,
		Reason: "the history is long and keeping it verbatim has stopped paying for itself",
	})
	return nil
}

// Delegating records that the model handed work to the subagent, which is a decision the
// model made rather than one this made, and belongs in the same trail regardless.
func (c *converse) Delegating(taskID, skill, prompt, turnID string) {
	c.mu.Lock()
	c.delegated[taskID] = turnID
	c.mu.Unlock()

	c.decide(Action{
		Kind:   ActDelegate,
		Reason: "the model handed " + skill + " to the subagent and carried on talking",
		TurnID: turnID,
		Text:   prompt,
	})
}

// Delegated records delegated work coming back.
//
// Without it the trail says work went out and never what became of it, so a call where the
// subagent ran out of time reads exactly like one where it answered and the answer was
// never spoken. What it took is recorded too: the caller spent that long being kept
// company, which is the cost of having asked.
func (c *converse) Delegated(result harness.Result) {
	c.mu.Lock()
	turnID := c.delegated[result.TaskID]
	delete(c.delegated, result.TaskID)
	c.mu.Unlock()

	// A task that needs something from the caller has a question instead of an answer, and
	// the question is the useful half of what came back.
	text := result.Text
	if text == "" {
		text = result.Question
	}

	c.decide(Action{
		Kind:      ActSettle,
		Reason:    settlement(result),
		TurnID:    turnID,
		Text:      text,
		LatencyMs: result.ElapsedMs,
	})
}

// settlement says in words what became of a piece of delegated work.
func settlement(result harness.Result) string {
	switch result.State {
	case harness.Done:
		if result.Question != "" {
			return "the subagent needs the caller asked something before it can answer " + result.Skill
		}
		if result.Text == "" {
			return "the subagent finished " + result.Skill + " with nothing to say"
		}
		return "the subagent answered " + result.Skill
	case harness.Cancelled:
		return "the subagent's " + result.Skill + " was abandoned: " + result.Reason
	}
	if result.Err != nil {
		return "the subagent could not answer " + result.Skill + ": " + result.Err.Error()
	}
	return "the subagent could not answer " + result.Skill
}

// inPartsSpan bounds how long a thought given in parts holds the floor for the caller.
const inPartsSpan = 2 * time.Minute

// patient reports whether an unfinished thought is still worth waiting on.
//
// The controller is asked about the same words every retry for as long as the caller says
// nothing more, and it answers the same way every time, so waiting is a loop that only the
// caller can end. Somebody who has gone quiet mid-sentence has usually finished and been
// misheard, and at that point asking them is better than listening to silence.
// inParts reports whether the participant is giving a thought in parts: they went on
// talking before hearing any of the reply to what they had said, and have not been
// answered since. It lapses after inPartsSpan.
func (c *converse) inParts(participant string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	since, ok := c.wentOn[participant]
	return ok && time.Since(since) <= inPartsSpan
}

// waitedOn says the same words were waited on before.
func (c *converse) waitedOn(ready candidate) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	waiting, known := c.waiting[ready.Participant.ID]
	return known && sameWords(waiting.text, ready.Text)
}

func (c *converse) patient(ready candidate) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	waiting, known := c.waiting[ready.Participant.ID]
	if !known || !sameWords(waiting.text, ready.Text) {
		c.waiting[ready.Participant.ID] = unfinished{text: ready.Text, since: time.Now()}
		return true
	}
	return time.Since(waiting.since) < c.patience
}

// forgetQueuedRevisedByLocked drops the turn queued behind a reply when the words being dealt
// with now are the same run of speech, restated or grown. A caller who kept talking after a
// turn was queued is answered on everything they said, and answering the queued part again
// once the agent stops is answering them twice, late, to less than they asked. Words that are
// not a revision of it leave it queued: they may be the only copy of what it said. The caller
// holds the lock.
func (c *converse) forgetQueuedRevisedByLocked(ready candidate) string {
	held := c.queued
	if held == nil || held.candidate.ID == ready.ID || held.candidate.Participant.ID != ready.Participant.ID {
		return ""
	}
	if !revisesQueued(held.candidate, ready) {
		return ""
	}
	c.logger.Debug("dropping a queued turn the caller has since said again in full",
		"candidate", held.candidate.ID, "text", held.candidate.Text, "answered", ready.Text)
	c.queued = nil
	return held.candidate.ID
}

// revisesQueued reports whether ready is the queued turn's run of speech again, restated or
// grown, rather than a second utterance: the same utterance number when both have one, the
// same words so far otherwise.
func revisesQueued(held, ready candidate) bool {
	if held.Utterance != 0 && ready.Utterance != 0 {
		return held.Utterance == ready.Utterance
	}
	return revisesTranscript(held.Text, ready.Text)
}

// forgetQueuedAnsweredOver drops the turn the same participant has queued once a newer turn of
// theirs is answered, and returns its words for that answer to take in. Answered on its own
// after it, the older turn undoes the newer: a caller's correction is answered, then their
// first request again, and the agent goes back to asking about it. Another
// participant's queued turn is still theirs to be answered.
//
// Only words the newer turn does not say are taken in. A revision that kept less of the run
// ("okay wait" after "okay wait make it six") owes the rest of it; any other revision says it
// all and is left to forgetQueuedRevisedByLocked. A newer utterance that opens with the queued
// words says them already.
func (c *converse) forgetQueuedAnsweredOver(ready candidate) string {
	c.mu.Lock()
	held := c.queued
	if held == nil || held.candidate.Participant.ID != ready.Participant.ID {
		c.mu.Unlock()
		return ""
	}
	said := held.candidate.Said()
	switch {
	case !revisesQueued(held.candidate, ready):
		if opensWith(ready.Text, said) {
			said = ""
		}
	case opensWith(held.candidate.Text, ready.Text) &&
		len(wordList(held.candidate.Text)) > len(wordList(ready.Text)):
		said = unrepeated(held.candidate.Text, ready.Text)
	default:
		c.mu.Unlock()
		return ""
	}
	c.queued = nil
	c.mu.Unlock()
	c.cadence.ReleaseHeld(held.candidate.ID)
	if !held.heard {
		// The line it waited in was let go of, so it is heard on one of its own, as when
		// it is answered alone (Waiting).
		shown, line := c.closeQueuedLine(*held)
		c.emitter.Send(Heard{
			Participant: held.candidate.Participant,
			Text:        shown,
			Language:    held.candidate.Language,
			LineID:      line,
		})
	}
	staleQueuedTurns.Inc()
	c.logger.Info("dropped a queued turn the caller has said more after",
		"turn", held.candidate.ID, "participant", held.candidate.Participant.ID,
		"age_ms", time.Since(held.candidate.ReadyAt).Milliseconds())
	return said
}

// forgetQueuedGrownLocked drops the queued turn once the caller's words still in progress have
// grown past it. A caller who paused mid-sentence while the agent talked had the first part
// queued; the rest goes to the controller with it once it settles. Answering the queued part
// when the agent stops answers half a sentence and cuts into the rest of the caller's long
// turn. Words that only start the same way are a new turn
// and leave it queued. The caller holds the lock.
func (c *converse) forgetQueuedGrownLocked(participant stt.Participant, utterance int64, saying string) (released string, grown bool) {
	held := c.queued
	if held == nil || held.candidate.Participant.ID != participant.ID {
		return "", false
	}
	if held.candidate.Utterance != 0 && utterance != 0 {
		if held.candidate.Utterance != utterance {
			return "", false
		}
	} else if !growsTranscript(held.candidate.Text, saying) {
		return "", false
	}
	c.logger.Debug("dropping a queued turn the caller is still adding to",
		"candidate", held.candidate.ID, "text", held.candidate.Text, "saying", saying)
	c.queued = nil
	// What is being said now carries its words on, so they are not put in front again.
	delete(c.queuedSaid, participant.ID)
	// The words it was heard as are the start of what is being said now, so they are that
	// line growing rather than a second one beside it.
	if _, open := c.lines[participant.ID]; held.line != "" && !open {
		c.lines[participant.ID] = held.line
		grown = true
	}
	return held.candidate.ID, grown
}

// lineLocked is the line a participant's words are going into, opened by the first of them.
// It is random rather than a stamp because the transcript stores the line under it, and a
// message id is the app's to keep unique across every channel. The caller holds the lock.
func (c *converse) lineLocked(participantID string) string {
	if line, open := c.lines[participantID]; open {
		return line
	}
	line := newLine()
	c.lines[participantID] = line
	return line
}

func newLine() string {
	return linePrefix + strings.ToLower(rand.Text())
}

// closeLine closes the line a participant's words were going into, once they are ruled on,
// and returns what it says of them and its id. One the words were heard on is what the
// participant's next line repeats, if the transcriber goes over that run of speech again.
func (c *converse) closeLine(participantID, text string, heard bool) (shown, line string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.closeLineOnLocked(participantID, text, heard)
}

func (c *converse) closeLineOnLocked(participantID, text string, heard bool) (shown, line string) {
	shown = c.shownLocked(participantID, text)
	line = c.closeLineLocked(participantID)
	if heard {
		c.lastHeard[participantID] = text
	}
	// A turn queued unheard in this line goes with it: heard, its words are part of what
	// the line says; let go of, they are not, and are heard under a line of their own once
	// answered, since this one's message is gone.
	if held := c.queued; held != nil && !held.heard && line != "" && held.line == line {
		if heard {
			held.heard = true
		} else {
			held.line = newLine()
		}
	}
	return shown, line
}

// closeQueuedLine is closeLine for a turn queued before it was heard. The line it was being
// said in may have been let go of since, and the line open now be the participant's next
// one: that is left to go on, and the turn is heard under a line of its own, saying what
// its line showed of it.
func (c *converse) closeQueuedLine(held queuedCandidate) (shown, line string) {
	participantID := held.candidate.Participant.ID
	c.mu.Lock()
	defer c.mu.Unlock()
	if open := c.lines[participantID]; held.line != "" && open != held.line {
		return cmp.Or(held.said, held.candidate.Text), held.line
	}
	return c.closeLineOnLocked(participantID, held.candidate.Text, true)
}

func (c *converse) closeLineLocked(participantID string) string {
	line := c.lines[participantID]
	delete(c.lines, participantID)
	delete(c.grown, participantID)
	delete(c.queuedSaid, participantID)
	delete(c.repeated, participantID)
	return line
}

// shownLocked is what a participant's line says of a turn's words: those words, without
// the ones already heard as a line before it, unless the line is that one carried on with.
// A turn that came back to nothing but those words is said whole, as a line must say
// something. The caller holds the lock.
func (c *converse) shownLocked(participantID, text string) string {
	if shown := c.lineTextLocked(participantID, text); shown != "" {
		return shown
	}
	return text
}

func (c *converse) lineTextLocked(participantID, text string) string {
	if c.grown[participantID] {
		return text
	}
	shown := unrepeated(text, c.repeated[participantID])
	if held := c.queuedSaid[participantID]; held != "" && shown != "" {
		shown = joinOverlapping(held, shown)
	}
	return shown
}

// repeatOfLocked is how much of the line last heard a new line opening on text says again.
// repeats is what cadence found the transcriber restating, and only words heard as a line
// count:
//   - repeats that end the line last heard are those words again;
//   - text that opens with the whole line last heard went back over all of it;
//   - anything else restated was acted on but never heard, such as a turn queued while the
//     agent talked, and is the new line's own.
//
// With nothing restated, nothing is a repeat. The caller holds the lock.
func (c *converse) repeatOfLocked(participantID, text, repeats string) string {
	heard := c.lastHeard[participantID]
	switch {
	case repeats == "" || heard == "":
		return ""
	case endsWith(heard, repeats):
		return repeats
	case opensWith(text, heard):
		return heard
	}
	return ""
}

// joinOverlapping puts more after held, once: words more opens with that held already ends
// with are the transcriber going back over them, not the caller saying them twice.
func joinOverlapping(held, more string) string {
	if opensWith(more, held) {
		return more
	}
	before, after := wordList(held), wordList(more)
	for overlap := min(len(before), len(after)); overlap > 0; overlap-- {
		if slices.Equal(before[len(before)-overlap:], after[:overlap]) {
			if rest := unrepeated(more, strings.Join(after[:overlap], " ")); rest != "" {
				return held + " " + rest
			}
			return held
		}
	}
	return held + " " + more
}

// hold keeps a turn until the agent has stopped talking. Only one is kept: a caller who
// has said two more things while being talked over is owed an answer to the last of them.
func (c *converse) hold(held queuedCandidate) {
	c.mu.Lock()
	released := ""
	if c.queued != nil {
		c.logger.Debug("dropping a turn that never got answered",
			"candidate", c.queued.candidate.ID, "text", c.queued.candidate.Text)
		if c.queued.candidate.ID != held.candidate.ID {
			released = c.queued.candidate.ID
		}
	}
	c.queued = &held
	c.mu.Unlock()
	if released != "" {
		c.cadence.ReleaseHeld(released)
	}
}

// decide records one judgement and hands it back, so a caller can write
// `return c.decide(...)` and have the reporting be part of making the decision rather
// than something to remember afterwards.
func (c *converse) decide(action Action) Action {
	if !c.worthReporting(action) {
		return action
	}
	at := time.Now()

	c.logger.Info("the conversation decided",
		"decision", action.Kind, "reason", action.Reason,
		"participant", action.Participant.ID, "turn", action.TurnID,
		"candidate", action.Candidate.ID, "text", action.Text,
		"took_ms", action.LatencyMs, "error", action.Err)

	turnID := action.TurnID
	if turnID == "" {
		turnID = action.Candidate.ID
	}
	decided := Decided{
		At:          at,
		Kind:        string(action.Kind),
		Reason:      action.Reason,
		TurnID:      turnID,
		Participant: action.Participant,
		Text:        action.Text,
		LatencyMs:   action.LatencyMs,
		LineID:      action.LineID,
	}
	c.emitter.Send(decided)
	if c.record != nil {
		c.record(decided)
	}
	return action
}

// overlapNoise reports whether settled words overlapping the agent's reply are a cough or
// similar non-speech, which the flow controller often classifies as a new request.
// interruptsAlone are the words that take the floor from the agent on their own. "Way" is
// how the transcriber first hears "wait".
var interruptsAlone = map[string]bool{
	"wait": true, "way": true, "stop": true, "no": true, "nope": true, "hold": true,
	"hang": true, "actually": true, "sorry": true, "excuse": true, "pardon": true,
	"hello": true, "hey": true,
}

func overlapNoise(text string) bool {
	t := strings.ToLower(words(text))
	if t == "" {
		return false
	}
	if strings.Contains(t, "cough") || strings.Contains(t, "ahem") {
		return true
	}
	return murmurs[t]
}

// murmurs are the words of someone listening rather than taking the floor. True marks the ones
// that are noise said alone (overlapNoise); the rest are assent, which only a run of murmurs is.
// The punctuation is gone (words), so "uh-huh" is "uhhuh".
var murmurs = map[string]bool{
	"huh": true, "uh": true, "mm": true, "hm": true, "hmm": true,
	"okay": false, "ok": false, "yeah": false, "yes": false, "yep": false, "yup": false,
	"uhhuh": false, "mhm": false, "mmhmm": false, "right": false, "sure": false, "alright": false,
	"got": false, "it": false, "gotcha": false, "thank": false, "thanks": false, "you": false,
}

// worthReporting says whether a judgement tells a reader anything the last one did not,
// and remembers it when it does.
//
// A caller who has stopped mid-thought is asked about and waited on again every retry,
// which is the same two judgements over and over about words that have not changed. The
// first pair says everything the rest do, and the rest are only volume: in the log, in the
// dashboard's trail and in the table behind it. Anything else decided means the words moved
// on, so what was remembered is dropped and the same sentence said again is written down
// again.
func (c *converse) worthReporting(action Action) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if action.Kind != ActAsk && action.Kind != ActWait && action.Kind != ActHold {
		clear(c.reported)
		return true
	}

	subject := judged{participant: action.Participant.ID, text: action.Text}
	if c.reported[action.Kind] == subject {
		return false
	}
	c.reported[action.Kind] = subject
	return true
}
