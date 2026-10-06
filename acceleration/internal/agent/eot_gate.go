package agent

import (
	"context"
	"math"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const maxEOTParticipants = 16

type eotGate struct {
	candidateID   string
	participantID string
	ready         candidate
	turn          harness.FlowTurn
	primary       bool
	pipeline      *pipeline
	harness       *harness.Harness
	ctx           context.Context
	cancel        context.CancelFunc
	approved      bool
	held          *harness.Decided
}

func (a *Agent) eotCandidateCurrentLocked(gate *eotGate) bool {
	_, ok := a.eotCandidateSnapshotLocked(gate)
	return ok
}

func (a *Agent) eotCandidateSnapshotLocked(gate *eotGate) (candidate, bool) {
	a.converse.mu.Lock()
	ready, ok := a.converse.candidates[gate.candidateID]
	a.converse.mu.Unlock()
	if !ok || ready.ID != gate.ready.ID || ready.Participant.ID != gate.ready.Participant.ID ||
		ready.Text != gate.ready.Text {
		return candidate{}, false
	}
	return a.cadence.candidateSnapshot(gate.ready)
}

type eotResult struct {
	gate    *eotGate
	score   EOTScore
	err     error
	latency time.Duration
}

func (a *Agent) cancelEOTPreviews(gates []*eotGate) {
	for _, gate := range gates {
		_ = gate.harness.CancelDecision(gate.candidateID)
		a.cancelPreview(gate.candidateID)
	}
}

func (a *Agent) retainEOTAudio(participantID string, sampleRate, channels int, samples []int16) {
	if a.options.EOT == nil || participantID == "" || sampleRate != eotSampleRate || channels != 1 || len(samples) == 0 {
		return
	}
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return
	}
	ring := a.audioHistory[participantID]
	if ring == nil {
		if len(a.audioHistory) >= maxEOTParticipants {
			a.mu.Unlock()
			return
		}
		ring = newPCM16LERing()
		a.audioHistory[participantID] = ring
	}
	a.mu.Unlock()
	ring.append(samples)
}

func (a *Agent) eotAudioSnapshot(participantID string) []byte {
	a.mu.Lock()
	ring := a.audioHistory[participantID]
	a.mu.Unlock()
	if ring == nil {
		return nil
	}
	return ring.snapshot()
}

func (a *Agent) hasEOTAudio(participantID string) bool {
	a.mu.Lock()
	ring := a.audioHistory[participantID]
	a.mu.Unlock()
	if ring == nil {
		return false
	}
	ring.mu.Lock()
	defer ring.mu.Unlock()
	count := ring.next
	if ring.full {
		count = len(ring.sample)
	}
	return count >= eotMinSamples
}

func (a *Agent) detachEOTAudioLocked() []*pcm16leRing {
	rings := make([]*pcm16leRing, 0, len(a.audioHistory))
	for id, ring := range a.audioHistory {
		rings = append(rings, ring)
		delete(a.audioHistory, id)
	}
	return rings
}

func clearEOTAudio(rings []*pcm16leRing) {
	for _, ring := range rings {
		ring.clear()
	}
}

func (a *Agent) registerEOTGateLocked(ready candidate, current *harness.Harness, p *pipeline) *eotGate {
	if a.options.EOT == nil || current == nil || p == nil || p.native || p.eotResults == nil ||
		a.closed || a.switching.Load() || a.harness != current || a.pipe != p {
		return nil
	}
	for id, old := range a.eotGates {
		if old.participantID == ready.Participant.ID {
			delete(a.eotGates, id)
			old.cancel()
		}
	}
	ctx, cancel := context.WithTimeout(p.ctx, eotGateLimit)
	gate := &eotGate{
		candidateID:   ready.ID,
		participantID: ready.Participant.ID,
		ready:         ready,
		pipeline:      p,
		harness:       current,
		ctx:           ctx,
		cancel:        cancel,
	}
	a.eotGates[ready.ID] = gate
	return gate
}

func (a *Agent) startEOT(gate *eotGate, pcm []byte) {
	if gate == nil || len(pcm) == 0 {
		return
	}
	a.mu.Lock()
	if a.eotGates[gate.candidateID] != gate || a.closed || gate.pipeline.ctx.Err() != nil {
		a.mu.Unlock()
		return
	}
	// The WaitGroup Add is serialized with releasePipeline under Agent.mu.
	a.running.Add(1)
	gate.pipeline.running.Add(1)
	a.mu.Unlock()

	go func() {
		defer a.running.Done()
		defer gate.pipeline.running.Done()
		started := time.Now()
		score, err := a.options.EOT.Score(gate.ctx, gate.candidateID, pcm)
		result := eotResult{gate: gate, score: score, err: err, latency: time.Since(started)}
		select {
		case gate.pipeline.eotResults <- result:
		case <-gate.pipeline.ctx.Done():
		}
	}()
}

func (a *Agent) cancelEOTGate(candidateID string) {
	a.mu.Lock()
	gate := a.eotGates[candidateID]
	delete(a.eotGates, candidateID)
	a.mu.Unlock()
	if gate != nil {
		gate.cancel()
	}
}

// cancelEOTGatesLocked removes every acoustic join owned by p. A nil p means all gates.
// The caller holds Agent.mu; returned semantic results can be released after unlocking.
func (a *Agent) cancelEOTGatesLocked(p *pipeline) []*eotGate {
	var canceled []*eotGate
	for id, gate := range a.eotGates {
		if p != nil && gate.pipeline != p {
			continue
		}
		delete(a.eotGates, id)
		gate.cancel()
		if !a.closed {
			a.converse.Unasked(gate.candidateID)
		}
		canceled = append(canceled, gate)
	}
	return canceled
}

func (a *Agent) cancelParticipantEOTLocked(participant stt.Participant) []*eotGate {
	var canceled []*eotGate
	for id, gate := range a.eotGates {
		if gate.participantID == participant.ID {
			delete(a.eotGates, id)
			gate.cancel()
			canceled = append(canceled, gate)
		}
	}
	a.converse.ForgetParticipant(participant)
	return canceled
}

func (a *Agent) primaryEOTEligible(gate *eotGate) bool {
	if gate == nil || !gate.primary || gate.ready.Unfinished || a.options.Text ||
		(a.options.Guardrail != nil && a.options.Guardrail.Policy().Mode == guardrail.ModeBlocking) {
		return false
	}
	a.mu.Lock()
	eligible := a.eotPrimaryStateLocked(gate)
	a.mu.Unlock()
	if !eligible || a.speechPending() {
		return false
	}
	a.mu.Lock()
	eligible = a.eotPrimaryStateLocked(gate)
	a.mu.Unlock()
	return eligible && !a.speechPending()
}

func (a *Agent) eotPrimaryStateLocked(gate *eotGate) bool {
	ready, current := a.eotCandidateSnapshotLocked(gate)
	return a.eotGates[gate.candidateID] == gate && gate.pipeline == a.pipe &&
		gate.harness == a.harness && !a.closed && !a.switching.Load() &&
		gate.pipeline.ctx.Err() == nil && !gate.ready.Unfinished && !a.options.Text &&
		(a.options.Guardrail == nil || a.options.Guardrail.Policy().Mode != guardrail.ModeBlocking) &&
		current && !a.generating && a.utterances == 0 &&
		a.pendingTools == 0 && !a.anotherVoiceLocked(ready)
}

func (a *Agent) refreshedPrimaryFlowTurnLocked(gate *eotGate, speechPending bool) harness.FlowTurn {
	turn := gate.turn
	turn.Instructions = a.instructions()
	turn.History = llm.OmitImages(append([]llm.Message(nil), a.history...))
	turn.Speaking = a.generating || a.utterances > 0 || a.pendingTools > 0
	turn.Reply = a.saying
	if turn.Reply == "" {
		turn.Reply = lastAssistantSaid(turn.History)
	}
	ready, current := a.eotCandidateSnapshotLocked(gate)
	if !current {
		ready = gate.ready
	}
	turn.AnotherVoice = a.anotherVoiceLocked(ready)
	turn.Speaking = turn.Speaking || speechPending
	return turn
}

func (a *Agent) fallbackPrimaryEOT(gate *eotGate, reason string, latency time.Duration) {
	speechPending := a.speechPending()
	a.mu.Lock()
	if a.eotGates[gate.candidateID] != gate || gate.pipeline != a.pipe || gate.harness != a.harness {
		a.mu.Unlock()
		return
	}
	// Let the swap/close path own the join while it invalidates the candidate. In
	// particular, do not remove a gate between switching being raised and its
	// cancellation/resettle step.
	if a.switching.Load() || gate.pipeline.ctx.Err() != nil {
		a.mu.Unlock()
		return
	}
	if a.closed || !a.eotCandidateCurrentLocked(gate) {
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		a.mu.Unlock()
		return
	}
	turn := a.refreshedPrimaryFlowTurnLocked(gate, speechPending)
	delete(a.eotGates, gate.candidateID)
	gate.cancel()
	err := gate.harness.Decide(turn)
	a.mu.Unlock()
	a.logger.Info("primary EOT fell back to semantic flow", "candidate", gate.candidateID,
		"reason", reason, "latency_ms", float64(latency)/float64(time.Millisecond))
	if err != nil {
		a.cancelPreview(gate.candidateID)
		a.converse.Unasked(gate.candidateID)
		a.fail(err, "flow")
	}
}

func (a *Agent) consumeEOTResult(result eotResult, current *harness.Harness, p *pipeline) {
	a.mu.Lock()
	gate := result.gate
	if gate == nil || a.eotGates[gate.candidateID] != gate ||
		gate.pipeline != p || gate.harness != current || a.harness != current {
		a.mu.Unlock()
		return
	}
	if gate.primary && (a.switching.Load() || gate.pipeline.ctx.Err() != nil) {
		// The transition path owns invalidating and resettling this candidate.
		a.mu.Unlock()
		return
	}
	if result.err != nil {
		if gate.primary {
			a.mu.Unlock()
			a.fallbackPrimaryEOT(gate, "service_unavailable", result.latency)
			return
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		held := gate.held
		a.mu.Unlock()
		a.logger.Debug("acoustic endpoint score unavailable; using the flow decision",
			"candidate", gate.candidateID, "reason", result.err.Error())
		if held != nil {
			a.act(a.converse.Ruled(*held, a.floor()))
		}
		return
	}
	if math.IsNaN(result.score.Probability) || math.IsInf(result.score.Probability, 0) ||
		result.score.Probability < 0 || result.score.Probability > 1 {
		if gate.primary {
			a.mu.Unlock()
			a.fallbackPrimaryEOT(gate, "invalid_score", result.latency)
			return
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		held := gate.held
		a.mu.Unlock()
		a.logger.Debug("acoustic endpoint score unavailable; using the flow decision",
			"candidate", gate.candidateID, "reason", "invalid_score")
		if held != nil {
			a.act(a.converse.Ruled(*held, a.floor()))
		}
		return
	}
	threshold := a.options.EOTThreshold
	if gate.primary {
		a.logger.Info("primary EOT score received", "candidate", gate.candidateID,
			"probability", result.score.Probability, "threshold", threshold,
			"samples", result.score.Samples, "latency_ms", float64(result.latency)/float64(time.Millisecond))
	} else {
		a.logger.Debug("acoustic endpoint score received", "candidate", gate.candidateID,
			"probability", result.score.Probability, "threshold", threshold,
			"samples", result.score.Samples, "latency_ms", float64(result.latency)/float64(time.Millisecond))
	}
	if result.score.Probability < threshold {
		if gate.primary {
			a.mu.Unlock()
			if !a.primaryEOTEligible(gate) {
				a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency)
				return
			}
			a.mu.Lock()
			if !a.eotPrimaryStateLocked(gate) {
				a.mu.Unlock()
				a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency)
				return
			}
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		a.mu.Unlock()
		_ = current.CancelDecision(gate.candidateID)
		wait := harness.Decided{
			CandidateID: gate.candidateID,
			Disposition: harness.Wait,
			Floor:       harness.Continue,
			TookMs:      float64(result.latency) / float64(time.Millisecond),
		}
		a.act(a.converse.Ruled(wait, a.floor()))
		return
	}
	if gate.primary {
		a.mu.Unlock()
		if !a.primaryEOTEligible(gate) {
			a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency)
			return
		}
		a.mu.Lock()
		if !a.eotPrimaryStateLocked(gate) {
			a.mu.Unlock()
			a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency)
			return
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		a.mu.Unlock()
		decision := harness.Decided{
			CandidateID: gate.candidateID,
			Disposition: harness.Respond,
			Floor:       harness.Continue,
			TookMs:      float64(result.latency) / float64(time.Millisecond),
		}
		a.act(a.converse.Ruled(decision, a.floor()))
		return
	}
	gate.approved = true
	held := gate.held
	if held != nil {
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
	}
	a.mu.Unlock()
	if held != nil {
		a.act(a.converse.Ruled(*held, a.floor()))
	}
}

func (a *Agent) decideWithEOT(p *pipeline, current *harness.Harness, ready candidate, turn harness.FlowTurn, pcm []byte) {
	if len(pcm) == 0 || a.options.EOT == nil {
		if a.options.EOTMode == EOTModePrimary {
			reason := "no_audio_window"
			if a.options.EOT == nil {
				reason = "endpoint_unconfigured"
			}
			a.logger.Info("primary EOT using semantic fallback", "candidate", ready.ID, "reason", reason)
		}
		if err := current.Decide(turn); err != nil {
			a.cancelPreview(ready.ID)
			a.converse.Unasked(ready.ID)
			a.fail(err, "flow")
		}
		return
	}

	a.mu.Lock()
	if a.closed || a.harness != current || a.pipe != p || a.switching.Load() {
		a.mu.Unlock()
		return
	}
	gate := a.registerEOTGateLocked(ready, current, p)
	primary := a.options.EOTMode == EOTModePrimary
	var err error
	if gate != nil {
		gate.primary = primary
		gate.turn = turn
	}
	if !primary || gate == nil {
		err = current.Decide(turn)
	}
	if err != nil && gate != nil {
		delete(a.eotGates, ready.ID)
		gate.cancel()
	}
	a.mu.Unlock()
	if err != nil {
		a.cancelPreview(ready.ID)
		a.converse.Unasked(ready.ID)
		a.fail(err, "flow")
		return
	}
	if gate != nil {
		a.startEOT(gate, pcm)
	}
}

func canPassWithoutEOT(decision harness.Decided, floorQuiet bool) bool {
	if !decision.Valid() {
		return false
	}
	return decision.Disposition == harness.Wait || decision.Disposition == harness.Ignore ||
		decision.Disposition == harness.Clarify ||
		(!floorQuiet && (decision.Floor == harness.Stop || decision.Floor == harness.Shorten))
}

func (a *Agent) decideFromHarness(p *pipeline, current *harness.Harness, decision harness.Decided) {
	floorQuiet := a.floor().Quiet
	a.mu.Lock()
	gate := a.eotGates[decision.CandidateID]
	if gate == nil || gate.pipeline != p || gate.harness != current {
		a.mu.Unlock()
		a.act(a.converse.Ruled(decision, a.floor()))
		return
	}
	if gate.approved || canPassWithoutEOT(decision, floorQuiet) {
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		a.mu.Unlock()
		a.act(a.converse.Ruled(decision, a.floor()))
		return
	}
	copy := decision
	gate.held = &copy
	a.mu.Unlock()
}
