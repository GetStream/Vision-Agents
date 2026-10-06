package agent

import (
	"context"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const maxEOTParticipants = 16

type eotGate struct {
	candidateID   string
	participantID string
	pipeline      *pipeline
	harness       *harness.Harness
	ctx           context.Context
	cancel        context.CancelFunc
	approved      bool
	held          *harness.Decided
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

func (a *Agent) consumeEOTResult(result eotResult, current *harness.Harness, p *pipeline) {
	a.mu.Lock()
	gate := result.gate
	if gate == nil || a.eotGates[gate.candidateID] != gate ||
		gate.pipeline != p || gate.harness != current || a.harness != current {
		a.mu.Unlock()
		return
	}
	if result.err != nil {
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
	threshold := a.options.EOTThreshold
	a.logger.Debug("acoustic endpoint score received", "candidate", gate.candidateID,
		"probability", result.score.Probability, "threshold", threshold,
		"samples", result.score.Samples, "latency_ms", float64(result.latency)/float64(time.Millisecond))
	if result.score.Probability < threshold {
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
	err := current.Decide(turn)
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
