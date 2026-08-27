package agent

import (
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

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
	// ActInterrupt abandons the reply being spoken.
	ActInterrupt ActionKind = "interrupt"
	// ActShorten stops the model adding more while letting the audio already sent finish.
	ActShorten ActionKind = "shorten"
	// ActBackchannel makes a short listening noise.
	ActBackchannel ActionKind = "backchannel"
	// ActCheckIn asks a caller who has gone quiet whether they need anything else.
	ActCheckIn ActionKind = "checkin"
	// ActSupersede abandons a ruling asked for about words that have since changed.
	ActSupersede ActionKind = "supersede"
	// ActCompact replaces an old stretch of history with a summary.
	ActCompact ActionKind = "compact"
	// ActDelegate hands a piece of work to the subagent.
	ActDelegate ActionKind = "delegate"
	// ActFail is a judgement that could not be made.
	ActFail ActionKind = "fail"
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
	// TurnID is the reply being interrupted, shortened or abandoned.
	TurnID string
	// Supersede names an earlier turn whose delegated work is no longer wanted, because
	// the caller has moved on from what asked for it.
	Supersede string
	// Clarify asks for a short clarifying question rather than an answer.
	Clarify bool
	// LatencyMs is what the flow controller took to rule. Zero where nothing was asked.
	LatencyMs float64
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
	// Speaking is the reply currently allowed to make audio, empty when there is none.
	Speaking string
	// Delegating reports whether the subagent is still working on something.
	Delegating bool
	// LastSpokeAt is when the agent last published audio.
	LastSpokeAt time.Time
	// LastHeardAt is when anyone on the call was last transcribed.
	LastHeardAt time.Time
	// LastParticipant is who the agent was last talking to.
	LastParticipant stt.Participant
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
	// record keeps the trail after the process holding it is gone. Nil when the
	// deployment has nowhere to put it, in which case the decisions are still logged and
	// still reported live.
	record func(Decided)

	mu sync.Mutex
	// candidates are settled turns waiting on the flow controller.
	candidates map[string]candidate
	// queued is a relevant turn heard while the agent chose to finish speaking.
	queued *queuedCandidate
	// lastCandidate owns delegated work from the last relevant caller turn.
	lastCandidate string
}

// queuedCandidate is a turn held back until the agent stops talking.
type queuedCandidate struct {
	candidate candidate
	clarify   bool
}

func newConverse(
	settling *cadence,
	listening *duplex,
	emitter *Emitter,
	record func(Decided),
	logger *slog.Logger,
) *converse {
	if logger == nil {
		logger = slog.Default()
	}
	return &converse{
		cadence:    settling,
		duplex:     listening,
		emitter:    emitter,
		logger:     logger,
		record:     record,
		candidates: map[string]candidate{},
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
		c.emitter.Send(Hearing{
			Participant: transcript.Participant,
			Text:        saying,
			Language:    transcript.Language,
		})
	}
	if superseded != "" {
		c.mu.Lock()
		delete(c.candidates, superseded)
		c.mu.Unlock()
		actions = append(actions, c.decide(Action{
			Kind:        ActSupersede,
			Reason:      "the caller said more, so the ruling in flight is about words that have changed",
			Participant: transcript.Participant,
			TurnID:      superseded,
		}))
	}

	return actions
}

// Settled registers a turn whose words have stopped changing and asks for a ruling on it.
func (c *converse) Settled(ready candidate, state floor) Action {
	c.mu.Lock()
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
	c.mu.Unlock()
	c.cadence.Resolve(candidateID, true)
}

// Ruled turns the flow controller's answer into what the agent should do about it.
//
// The actions come back in the order they must run: a turn taken from the agent is
// interrupted before it is answered, and a turn the agent talks through is held before
// the reply it is waiting on is cut short.
func (c *converse) Ruled(ruling harness.Decided, state floor) []Action {
	c.mu.Lock()
	ready, known := c.candidates[ruling.CandidateID]
	delete(c.candidates, ruling.CandidateID)
	c.mu.Unlock()
	if !known {
		c.logger.Debug("a ruling arrived for a turn that has moved on",
			"candidate", ruling.CandidateID)
		return nil
	}

	if ruling.Err != nil || !ruling.Valid() {
		// A ruling that cannot be read must not cost the caller their turn, so the words
		// go back to settling and are put again once they have.
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

	if ruling.Disposition == harness.Wait {
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

	if !c.cadence.Resolve(ruling.CandidateID, false) {
		c.logger.Debug("not acting on a ruling, the words changed while it was being made",
			"candidate", ruling.CandidateID)
		return nil
	}

	if ruling.Disposition == harness.Ignore {
		return []Action{c.decide(Action{
			Kind:        ActIgnore,
			Reason:      "this was background speech or meant for somebody else",
			Candidate:   ready,
			Participant: ready.Participant,
			Text:        ready.Text,
			LatencyMs:   ruling.TookMs,
		})}
	}

	// The turn counts, which is the point at which it becomes what the caller said rather
	// than what they might have been saying.
	c.emitter.Send(Heard{
		Participant: ready.Participant,
		Text:        ready.Text,
		Language:    ready.Language,
	})

	c.mu.Lock()
	previous := c.lastCandidate
	c.lastCandidate = ready.ID
	c.mu.Unlock()
	if previous == ready.ID {
		previous = ""
	}

	clarify := ruling.Disposition == harness.Clarify
	answer := Action{
		Kind:        ActAnswer,
		Reason:      "a complete thought addressed to the agent",
		Candidate:   ready,
		Participant: ready.Participant,
		Text:        ready.Text,
		Supersede:   previous,
		Clarify:     clarify,
		LatencyMs:   ruling.TookMs,
	}
	if clarify {
		answer.Reason = "the caller addressed the agent but their meaning is ambiguous"
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

	switch ruling.Floor {
	case harness.Stop:
		return []Action{
			c.decide(Action{
				Kind:        ActInterrupt,
				Reason:      "the caller corrected or interrupted the agent, so the reply is abandoned",
				Participant: ready.Participant,
				TurnID:      state.Speaking,
				LatencyMs:   ruling.TookMs,
			}),
			c.decide(answer),
		}

	case harness.Shorten:
		held := answer
		held.Kind = ActQueue
		held.Reason = "the caller added to what they asked, so the reply in flight is cut short"
		c.hold(ready, clarify)
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
		c.hold(ready, clarify)
		return []Action{c.decide(held)}
	}
}

// idle invites a caller who has gone quiet back into the conversation, because a silence
// that nobody breaks is how a call ends by accident rather than because it was over.
func (c *converse) idle(state floor, participant stt.Participant) []Action {
	phrase := c.duplex.Idle(state.active(), state.Quiet)
	if phrase == "" {
		return nil
	}
	return []Action{c.decide(Action{
		Kind:        ActCheckIn,
		Reason:      "nobody has said anything for a while, asking whether there is anything else",
		Participant: participant,
		Text:        phrase,
	})}
}

// Tick decides whether a long listening or thinking gap needs filling, so an agent that
// is busy does not sound like a dead line.
func (c *converse) Tick(state floor) []Action {
	participant, hearing := c.cadence.Active()
	if !hearing {
		participant = state.LastParticipant
	}
	if !hearing && !state.Delegating {
		return c.idle(state, participant)
	}

	phrase := c.duplex.Presence(participant, state.LastSpokeAt, state.Quiet)
	if phrase == "" {
		return nil
	}

	reason := "the caller has been talking a while without hearing anything back"
	if !hearing {
		reason = "work the caller was promised is still running and they have heard nothing for a while"
	}
	return []Action{c.decide(Action{
		Kind:        ActBackchannel,
		Reason:      reason,
		Participant: participant,
		Text:        phrase,
	})}
}

// Waiting hands back the turn held while the agent finished speaking, now that it has.
func (c *converse) Waiting(state floor) (Action, bool) {
	if !state.Quiet {
		return Action{}, false
	}

	c.mu.Lock()
	held := c.queued
	c.queued = nil
	c.mu.Unlock()
	if held == nil {
		return Action{}, false
	}

	return c.decide(Action{
		Kind:        ActAnswer,
		Reason:      "the agent has stopped talking, so the turn that was waiting can be answered",
		Candidate:   held.candidate,
		Participant: held.candidate.Participant,
		Text:        held.candidate.Text,
		Clarify:     held.clarify,
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
func (c *converse) Delegating(skill, prompt, turnID string) {
	c.decide(Action{
		Kind:   ActDelegate,
		Reason: "the model handed " + skill + " to the subagent and carried on talking",
		TurnID: turnID,
		Text:   prompt,
	})
}

// hold keeps a turn until the agent has stopped talking. Only one is kept: a caller who
// has said two more things while being talked over is owed an answer to the last of them.
func (c *converse) hold(ready candidate, clarify bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.queued != nil {
		c.logger.Debug("dropping a turn that never got answered",
			"candidate", c.queued.candidate.ID, "text", c.queued.candidate.Text)
	}
	c.queued = &queuedCandidate{candidate: ready, clarify: clarify}
}

// decide records one judgement and hands it back, so a caller can write
// `return c.decide(...)` and have the reporting be part of making the decision rather
// than something to remember afterwards.
func (c *converse) decide(action Action) Action {
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
	}
	c.emitter.Send(decided)
	if c.record != nil {
		c.record(decided)
	}
	return action
}
