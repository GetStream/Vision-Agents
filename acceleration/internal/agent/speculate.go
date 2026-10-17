package agent

import (
	"context"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
)

// speculation is a reply started while the flow controller is still deciding whether the
// words were meant for the agent.
//
// Deciding is a model round trip of its own, and the reply used to wait for it, so every
// answered turn paid for the two one after the other. A speculative reply is asked for
// beside the ruling instead and held on this side of pump, where a guardrail holds one
// too: nothing the model writes is heard, written down or acted on until an answer for the
// same words releases it. Any other ruling drops it, which costs the tokens of a reply
// nobody hears.
type speculation struct {
	ready   candidate
	turn    harness.Turn
	harness *harness.Harness
	// history is how long the conversation was when the reply was built on it. A turn that
	// changed it in the meantime is not what the reply answers.
	history   int
	startedAt time.Time
	ctx       context.Context
	cancel    context.CancelFunc
	// release says what became of the ruling, and is sent on exactly once: true speaks the
	// reply, false drops it. Whoever takes the speculation out of the map sends it.
	release chan bool
	// unified says the reply opens with the tag that rules on it, so no flow controller
	// was asked.
	unified bool
}

// speculate starts the reply to a settled turn before the flow controller has ruled on it.
//
// It only does so for a turn the agent would simply answer: nothing of the agent's own is
// still going out, the words are the caller's and finished, and no guardrail has to see
// them before the model does. Only the newest words are worth a reply, so whatever was
// speculated on before them is dropped: it is about to be superseded anyway.
//
// It runs before the ruling is asked for, so a ruling that comes back at once still finds
// the reply it is about.
//
// With UnifiedReplies the reply also rules on the words, and it reports true: nobody else
// is asked, so the caller must not ask the flow controller as well.
func (a *Agent) speculate(current *harness.Harness, ready candidate, speaking, anotherVoice bool) bool {
	unified := a.options.UnifiedReplies
	if !a.options.SpeculativeReplies && !unified {
		return false
	}
	if a.options.Text || a.options.Guardrail != nil || speaking || (anotherVoice && !unified) ||
		ready.Unfinished || strings.TrimSpace(ready.Text) == "" {
		return false
	}

	a.mu.Lock()
	if a.closed || a.harness != current || a.replies == nil || a.generating || a.switching.Load() {
		a.mu.Unlock()
		return false
	}
	note := a.duplex.Note(ready.Confidence)
	if unified {
		parts := []string{harness.UnifiedNote}
		if anotherVoice {
			parts = append(parts, harness.AnotherVoiceNote)
		}
		if note != "" {
			parts = append(parts, note)
		}
		note = strings.Join(parts, "\n\n")
	}
	stale := a.takeSpeculationsLocked()
	ctx, cancel := context.WithCancel(a.ctx)
	spec := &speculation{
		ready:   ready,
		harness: current,
		history: len(a.history),
		turn: harness.Turn{
			ID:           ready.ID,
			Instructions: a.instructions(),
			History:      append(a.replayLocked(), a.userTurnLocked(ready.Text, nil)),
			Note:         note,
		},
		startedAt: time.Now(),
		ctx:       ctx,
		cancel:    cancel,
		release:   make(chan bool, 1),
		unified:   unified,
	}
	a.speculations[ready.ID] = spec
	a.pumps.Add(1)
	a.mu.Unlock()

	for _, dropped := range stale {
		dropped.drop()
	}
	go a.runSpeculation(spec)
	return unified
}

// runSpeculation asks for the reply and holds it until the ruling says what to do with it.
// Once released it is drained exactly as startReply drains a reply asked for afterwards.
func (a *Agent) runSpeculation(spec *speculation) {
	defer a.pumps.Done()

	stream, err := spec.harness.Respond(spec.ctx, spec.turn)
	if spec.unified {
		a.ruleFromLead(spec, stream, err)
	}

	var speak bool
	select {
	case speak = <-spec.release:
	case <-spec.ctx.Done():
	}
	if !speak {
		if stream != nil {
			stream.Close()
		}
		spec.harness.Forget()
		return
	}

	turnID := spec.ready.ID
	defer a.finishGenerate(turnID)
	if err != nil {
		if spec.ctx.Err() != nil {
			return
		}
		a.mu.Lock()
		replies := a.replies
		a.mu.Unlock()
		replies <- llm.ResponseFailed{ResponseID: turnID, Err: err, Context: "llm"}
		replies <- llm.ResponseCompleted{Response: llm.Response{ID: turnID, Status: llm.StatusFailed}}
		return
	}

	a.mu.Lock()
	a.streams[turnID] = stream
	_, abandoned := a.abandoned[turnID]
	closed := a.closed
	a.mu.Unlock()
	if abandoned || closed {
		stream.Close()
	}
	a.pump(turnID, stream)
}

// adoptSpeculation answers a turn with the reply already started for it, and reports
// whether there was one to use. It is not used when the answer carries a note the reply
// was not written with, when the conversation moved on while the flow controller decided,
// or when something else took the floor in the meantime: those are answered the ordinary
// way, from the start.
func (a *Agent) adoptSpeculation(ready candidate, note string) bool {
	a.mu.Lock()
	spec, known := a.speculations[ready.ID]
	if !known {
		a.mu.Unlock()
		return false
	}
	delete(a.speculations, ready.ID)
	usable := (note == "" || spec.unified && note == ambiguousNote) && !a.closed && !a.generating && !a.switching.Load() &&
		a.harness == spec.harness && len(a.history) == spec.history && ready.Text == spec.ready.Text
	if !usable {
		a.mu.Unlock()
		spec.drop()
		return false
	}
	a.history = append(a.history, a.userTurnLocked(ready.Text, nil))
	a.speakingTurn = ready.ID
	a.generating = true
	a.toolRounds = 0
	a.toolReply = false
	a.asked = options.LLM{}
	a.sender = routing.Caller{}
	a.lastParticipant = ready.Participant
	a.generatingCancel[ready.ID] = spec.cancel
	a.mu.Unlock()

	// The model is counted as starting now, when the ruling came back, so the decision is
	// still measured in full and the head start shows up as a shorter wait for the model.
	adopted := time.Now()
	a.turns.begin(ready.ID, ready.Participant, ready.ReadyAt, ready.RevisedAt, ready.STTLatencyMs)
	a.turns.modelStarted(ready.ID, adopted)
	a.emitter.Send(Responding{TurnID: ready.ID, Participant: ready.Participant, Prompt: ready.Text})
	a.logger.Debug("answering with a reply started before the ruling",
		"turn", ready.ID, "head_start_ms", adopted.Sub(spec.startedAt).Milliseconds())
	spec.release <- true
	return true
}

// dropSpeculation drops the reply started for a turn the ruling did not answer.
func (a *Agent) dropSpeculation(candidateID string) {
	if candidateID == "" {
		return
	}
	a.mu.Lock()
	spec, known := a.speculations[candidateID]
	delete(a.speculations, candidateID)
	a.mu.Unlock()
	if known {
		spec.drop()
	}
}

// dropSpeculations drops every reply still waiting on a ruling. A pipeline being released
// waits for every reply goroutine, and a held one would otherwise wait for a ruling that
// is never coming.
func (a *Agent) dropSpeculations() {
	a.mu.Lock()
	stale := a.takeSpeculationsLocked()
	a.mu.Unlock()
	for _, spec := range stale {
		spec.drop()
	}
}

// takeSpeculationsLocked empties the held replies and returns them for dropping. The
// caller holds the lock.
func (a *Agent) takeSpeculationsLocked() []*speculation {
	stale := make([]*speculation, 0, len(a.speculations))
	for id, spec := range a.speculations {
		delete(a.speculations, id)
		stale = append(stale, spec)
	}
	return stale
}

// drop lets go of a held reply. Only whoever took it out of the map calls this, so it
// happens once.
func (s *speculation) drop() {
	s.release <- false
	s.cancel()
}

// ruleFromLead reads the tag a unified reply opened with and reports it as the ruling on
// its words. The tag is taken off the reply, which is then spoken or dropped exactly as a
// speculative one is once a controller has ruled.
func (a *Agent) ruleFromLead(spec *speculation, stream *llm.Stream, err error) {
	if err != nil {
		if spec.ctx.Err() == nil {
			spec.harness.Rule(harness.Decided{CandidateID: spec.ready.ID, Err: err,
				TookMs: float64(time.Since(spec.startedAt).Microseconds()) / 1000})
		}
		return
	}
	lead := stream.Lead(harness.UnifiedLead)
	disposition, from, to := harness.ParseUnified(lead)
	stream.Cut(from, to)
	if spec.ctx.Err() != nil {
		return
	}
	took := float64(time.Since(spec.startedAt).Microseconds()) / 1000
	a.logger.Debug("the reply ruled on its own turn",
		"candidate", spec.ready.ID, "disposition", disposition, "lead", lead, "took_ms", took)
	spec.harness.Rule(harness.Decided{
		CandidateID: spec.ready.ID,
		Disposition: disposition,
		Floor:       harness.Continue,
		TookMs:      took,
	})
}
