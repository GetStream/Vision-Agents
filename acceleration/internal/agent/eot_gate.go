package agent

import (
	"context"
	"errors"
	"math"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

const maxEOTParticipants = 16

type eotGate struct {
	candidateID         string
	participantID       string
	ready               candidate
	turn                harness.FlowTurn
	primary             bool
	pipeline            *pipeline
	harness             *harness.Harness
	ctx                 context.Context
	cancel              context.CancelFunc
	approved            bool
	held                *harness.Decided
	snapshotDiagnostics eotSnapshotDiagnostics
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
	gate            *eotGate
	score           EOTScore
	err             error
	errorClass      eotFailureClass
	attempts        int
	budgetExhausted bool
	latency         time.Duration
	snapshotOrdinal uint64
}

func (a *Agent) cancelEOTPreviews(gates []*eotGate) {
	for _, gate := range gates {
		_ = gate.harness.CancelDecision(gate.candidateID)
		a.cancelPreview(gate.candidateID)
	}
}

func (a *Agent) retainEOTAudio(participantID string, sampleRate, channels int, samples []int16) {
	a.retainEOTAudioTimed(participantID, sampleRate, channels, samples, AudioTiming{})
}

func (a *Agent) retainEOTAudioTimed(participantID string, sampleRate, channels int, samples []int16, timing AudioTiming) {
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
	ring.appendTimed(samples, timing)
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

func (a *Agent) eotScoringSnapshot(participantID string) (eotScoringSnapshot, bool) {
	a.mu.Lock()
	ring := a.audioHistory[participantID]
	a.mu.Unlock()
	if ring == nil {
		return eotScoringSnapshot{}, false
	}
	return ring.scoringSnapshot()
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
	budget := eotGateLimit
	if a.options.EOTMode == EOTModePrimary {
		budget = eotPrimaryLimit
	}
	ctx, cancel := context.WithTimeout(p.ctx, budget)
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

func (a *Agent) startEOT(gate *eotGate, snapshot eotScoringSnapshot) {
	if gate == nil || len(snapshot.pcm) == 0 {
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
	diagnostics := snapshot.claim()
	a.eotSnapshotOrdinal++
	diagnostics.ordinal = a.eotSnapshotOrdinal
	gate.snapshotDiagnostics = diagnostics
	snapshotOrdinal := gate.snapshotDiagnostics.ordinal
	a.mu.Unlock()
	pcm := snapshot.pcm

	go func() {
		defer a.running.Done()
		defer gate.pipeline.running.Done()
		score, err, attempts, failureClass, latency, budgetExhausted := scoreEOTAttempts(gate.ctx, a.options.EOT, gate.candidateID, pcm, gate.primary)
		result := eotResult{
			gate: gate, score: score, err: err, errorClass: failureClass,
			attempts: attempts, budgetExhausted: budgetExhausted, latency: latency,
			snapshotOrdinal: snapshotOrdinal,
		}
		select {
		case gate.pipeline.eotResults <- result:
		case <-gate.pipeline.ctx.Done():
		}
		a.logEOTAudioSnapshot(gate.snapshotDiagnostics)
	}()
}

func (a *Agent) logEOTAudioSnapshot(snapshot eotSnapshotDiagnostics) {
	timing := snapshot.timing
	const milliseconds = float64(time.Millisecond)
	a.logger.Info("EOT audio snapshot diagnostics",
		"timing_valid", snapshot.timingValid,
		"snapshot_ordinal", snapshot.ordinal,
		"generation", snapshot.generation,
		"generation_advance", snapshot.generationAdvance,
		"snapshot_generation_unchanged", snapshot.snapshotGenerationUnchanged,
		"samples", snapshot.samples,
		"clock_epoch", timing.Epoch,
		"pts_ms", float64(timing.PTS)/milliseconds,
		"source_age_valid", snapshot.sourceAgeValid,
		"source_age_ms", float64(snapshot.sourceAge)/milliseconds,
		"append_age_valid", snapshot.appendAgeValid,
		"append_age_ms", float64(snapshot.appendAge)/milliseconds,
		"timestamp_gap_delta_ms", float64(snapshot.timestampGapDelta)/milliseconds,
		"timestamp_gap_ms", float64(timing.TimestampGap)/milliseconds,
		"timestamp_only_gap_delta_ms", float64(snapshot.timestampOnlyGapDelta)/milliseconds,
		"timestamp_only_gap_ms", float64(timing.TimestampOnlyGap)/milliseconds,
		"sequence_loss_delta", snapshot.sequenceLossDelta,
		"sequence_loss", timing.SequenceLoss,
		"clock_resets_delta", snapshot.clockResetsDelta,
		"clock_resets", timing.ClockResets,
		"ambiguous_gaps_delta", snapshot.ambiguousGapsDelta,
		"ambiguous_gaps", timing.AmbiguousGaps,
		"overlap_delta_ms", float64(snapshot.overlapDelta)/milliseconds,
		"overlap_ms", float64(timing.Overlap)/milliseconds,
		"tail_100ms_samples", snapshot.tail100ms.samples,
		"tail_100ms_rms", snapshot.tail100ms.rms,
		"tail_100ms_peak", snapshot.tail100ms.peak,
		"tail_100ms_zero_fraction", snapshot.tail100ms.zeroFraction,
		"tail_500ms_samples", snapshot.tail500ms.samples,
		"tail_500ms_rms", snapshot.tail500ms.rms,
		"tail_500ms_peak", snapshot.tail500ms.peak,
		"tail_500ms_zero_fraction", snapshot.tail500ms.zeroFraction,
		"tail_1000ms_samples", snapshot.tail1000ms.samples,
		"tail_1000ms_rms", snapshot.tail1000ms.rms,
		"tail_1000ms_peak", snapshot.tail1000ms.peak,
		"tail_1000ms_zero_fraction", snapshot.tail1000ms.zeroFraction,
	)
}

func scoreEOTAttempts(ctx context.Context, client *EOTClient, requestID string, pcm []byte, primary bool) (EOTScore, error, int, eotFailureClass, time.Duration, bool) {
	started := time.Now()
	budget := eotGateLimit
	if primary {
		budget = eotPrimaryLimit
	}
	maxDeadline := started.Add(budget)
	if parentDeadline, ok := ctx.Deadline(); !ok || parentDeadline.After(maxDeadline) {
		bounded, cancel := context.WithDeadline(ctx, maxDeadline)
		defer cancel()
		ctx = bounded
	}
	maxAttempts := 1
	if primary {
		maxAttempts += eotPrimaryRetryLimit
	}
	var lastScore EOTScore
	var lastErr error
	var lastClass eotFailureClass
	attempts := 0
	budgetExhausted := false
	for attempt := 0; attempt < maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			if errors.Is(err, context.DeadlineExceeded) {
				budgetExhausted = true
				if lastErr == nil {
					lastErr = &eotAttemptError{class: eotFailureTimeout}
					lastClass = eotFailureTimeout
				}
			} else {
				lastErr = &eotAttemptError{class: eotFailureCanceled}
				lastClass = eotFailureCanceled
			}
			break
		}

		attemptLimit := eotGateLimit
		if attempt > 0 {
			attemptLimit = eotPrimaryRetryWindow
		}
		if deadline, ok := ctx.Deadline(); ok {
			remaining := time.Until(deadline)
			if remaining < attemptLimit {
				attemptLimit = remaining
			}
		}
		if attemptLimit <= 0 {
			budgetExhausted = true
			if lastErr == nil {
				lastErr = &eotAttemptError{class: eotFailureTimeout}
				lastClass = eotFailureTimeout
			}
			break
		}

		attemptCtx, cancel := context.WithTimeout(ctx, attemptLimit)
		lastScore, lastErr = client.Score(attemptCtx, requestID, pcm)
		cancel()
		attempts++
		if lastErr == nil {
			return lastScore, nil, attempts, "", time.Since(started), false
		}
		var retryAfter time.Duration
		var hasRetryAfter bool
		lastClass, retryAfter, hasRetryAfter = eotErrorMetadata(lastErr)
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			budgetExhausted = true
			if lastClass == eotFailureCanceled {
				lastErr = &eotAttemptError{class: eotFailureTimeout}
				lastClass = eotFailureTimeout
			}
			break
		}
		if errors.Is(ctx.Err(), context.Canceled) {
			lastErr = &eotAttemptError{class: eotFailureCanceled}
			lastClass = eotFailureCanceled
			break
		}
		failure, typed := lastErr.(*eotAttemptError)
		if !primary || ctx.Err() != nil || attempts >= maxAttempts ||
			!typed || !failure.retryable() {
			break
		}

		delay := 25 * time.Millisecond
		if attempt == 1 {
			delay = 50 * time.Millisecond
		}
		if hasRetryAfter && retryAfter > delay {
			delay = retryAfter
		}
		if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < delay+eotPrimaryMinRetryWindow {
			break
		}
		if !waitEOTRetry(ctx, delay, nil) {
			if errors.Is(ctx.Err(), context.DeadlineExceeded) {
				budgetExhausted = true
			} else {
				lastErr = &eotAttemptError{class: eotFailureCanceled}
				lastClass = eotFailureCanceled
			}
			return EOTScore{}, lastErr, attempts, lastClass, time.Since(started), budgetExhausted
		}
		if errors.Is(ctx.Err(), context.DeadlineExceeded) {
			budgetExhausted = true
			break
		}
		if errors.Is(ctx.Err(), context.Canceled) {
			lastErr = &eotAttemptError{class: eotFailureCanceled}
			lastClass = eotFailureCanceled
			break
		}
		if deadline, ok := ctx.Deadline(); ok && time.Until(deadline) < eotPrimaryMinRetryWindow {
			break
		}
	}
	if lastErr == nil {
		lastErr = &eotAttemptError{class: eotFailureUnknown}
		lastClass = eotFailureUnknown
	}
	return EOTScore{}, lastErr, attempts, lastClass, time.Since(started), budgetExhausted
}

func waitEOTRetry(ctx context.Context, delay time.Duration, onWait func()) bool {
	timer := time.NewTimer(delay)
	defer timer.Stop()
	if onWait != nil {
		onWait()
	}
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
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
	turn.History = a.heardHistoryLocked()
	turn.Speaking = a.generating || a.utterances > 0 || a.pendingTools > 0
	turn.Reply = a.saidLocked(turn.History)
	ready, current := a.eotCandidateSnapshotLocked(gate)
	if !current {
		ready = gate.ready
	}
	turn.AnotherVoice = a.anotherVoiceLocked(ready)
	turn.Speaking = turn.Speaking || speechPending
	turn.Unheard = turn.Speaking && a.unheardLocked()
	return turn
}

func (a *Agent) fallbackPrimaryEOT(gate *eotGate, reason string, latency time.Duration, attempts int, failureClass eotFailureClass, budgetExhausted bool, snapshotOrdinal uint64) {
	speechPending := a.speechPending()
	a.mu.Lock()
	if a.eotGates[gate.candidateID] != gate || gate.pipeline != a.pipe || gate.harness != a.harness {
		a.mu.Unlock()
		return
	}
	if errors.Is(gate.ctx.Err(), context.Canceled) {
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
	attrs := []any{"candidate", gate.candidateID, "snapshot_ordinal", snapshotOrdinal,
		"reason", reason, "attempts", attempts,
		"aggregate_latency_ms", float64(latency) / float64(time.Millisecond),
		"budget_exhausted", budgetExhausted}
	if failureClass != "" {
		attrs = append(attrs, "error_class", string(failureClass))
	}
	a.logger.Info("primary EOT fell back to semantic flow", attrs...)
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
	if errors.Is(gate.ctx.Err(), context.Canceled) {
		// A superseded or otherwise canceled candidate must never trigger fallback.
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
			a.fallbackPrimaryEOT(gate, "service_unavailable", result.latency, result.attempts, result.errorClass, result.budgetExhausted, result.snapshotOrdinal)
			return
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		held := gate.held
		a.mu.Unlock()
		a.logger.Debug("acoustic endpoint score unavailable; using the flow decision",
			"candidate", gate.candidateID, "error_class", string(result.errorClass),
			"snapshot_ordinal", result.snapshotOrdinal,
			"attempts", result.attempts,
			"aggregate_latency_ms", float64(result.latency)/float64(time.Millisecond),
			"budget_exhausted", result.budgetExhausted)
		if held != nil {
			a.rule(*held)
		}
		return
	}
	if math.IsNaN(result.score.Probability) || math.IsInf(result.score.Probability, 0) ||
		result.score.Probability < 0 || result.score.Probability > 1 {
		if gate.primary {
			a.mu.Unlock()
			a.fallbackPrimaryEOT(gate, "invalid_score", result.latency, result.attempts, eotFailureInvalidResponse, result.budgetExhausted, result.snapshotOrdinal)
			return
		}
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		held := gate.held
		a.mu.Unlock()
		a.logger.Debug("acoustic endpoint score unavailable; using the flow decision",
			"candidate", gate.candidateID, "snapshot_ordinal", result.snapshotOrdinal,
			"reason", "invalid_score")
		if held != nil {
			a.rule(*held)
		}
		return
	}
	threshold := a.options.EOTThreshold
	if gate.primary {
		a.logger.Info("primary EOT score received", "candidate", gate.candidateID,
			"snapshot_ordinal", result.snapshotOrdinal,
			"probability", result.score.Probability, "threshold", threshold,
			"samples", result.score.Samples, "attempts", result.attempts,
			"aggregate_latency_ms", float64(result.latency)/float64(time.Millisecond),
			"budget_exhausted", result.budgetExhausted)
	} else {
		a.logger.Debug("acoustic endpoint score received", "candidate", gate.candidateID,
			"snapshot_ordinal", result.snapshotOrdinal,
			"probability", result.score.Probability, "threshold", threshold,
			"samples", result.score.Samples, "attempts", result.attempts,
			"aggregate_latency_ms", float64(result.latency)/float64(time.Millisecond),
			"budget_exhausted", result.budgetExhausted)
	}
	if result.score.Probability < threshold {
		if gate.primary {
			a.mu.Unlock()
			if !a.primaryEOTEligible(gate) {
				a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency, result.attempts, "", false, result.snapshotOrdinal)
				return
			}
			a.mu.Lock()
			if !a.eotPrimaryStateLocked(gate) {
				a.mu.Unlock()
				a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency, result.attempts, "", false, result.snapshotOrdinal)
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
		if gate.primary {
			a.rulePrimaryEOTLow(wait)
			return
		}
		a.rule(wait)
		return
	}
	if gate.primary {
		a.mu.Unlock()
		if !a.primaryEOTEligible(gate) {
			a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency, result.attempts, "", false, result.snapshotOrdinal)
			return
		}
		a.mu.Lock()
		if !a.eotPrimaryStateLocked(gate) {
			a.mu.Unlock()
			a.fallbackPrimaryEOT(gate, "candidate_or_floor_changed", result.latency, result.attempts, "", false, result.snapshotOrdinal)
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
		// A score this high is the model sure the caller has finished, so the reply to them is
		// held for a shorter silence. It is only remembered until the ruling has been carried out.
		if a.replyConfidentScore > 0 && result.score.Probability >= a.replyConfidentScore {
			a.mu.Lock()
			a.confident[gate.candidateID] = struct{}{}
			a.mu.Unlock()
		}
		a.ruleByScore(decision)
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
		a.rule(*held)
	}
}

func (a *Agent) decideWithEOT(p *pipeline, current *harness.Harness, ready candidate, turn harness.FlowTurn, snapshot eotScoringSnapshot) {
	if len(snapshot.pcm) == 0 || a.options.EOT == nil {
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
		a.startEOT(gate, snapshot)
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
	if a.addresseeRuled(decision) {
		return
	}
	floorQuiet := a.floor().Quiet
	a.mu.Lock()
	gate := a.eotGates[decision.CandidateID]
	if gate == nil || gate.pipeline != p || gate.harness != current {
		a.mu.Unlock()
		a.rule(decision)
		return
	}
	if gate.approved || canPassWithoutEOT(decision, floorQuiet) {
		delete(a.eotGates, gate.candidateID)
		gate.cancel()
		a.mu.Unlock()
		a.rule(decision)
		return
	}
	copy := decision
	gate.held = &copy
	a.mu.Unlock()
}

// eotAudioUnchanged reports whether nothing has been heard from a participant since the audio
// last put to the acoustic scorer was copied, so asking again would score the same window.
// It copies nothing.
func (a *Agent) eotAudioUnchanged(participantID string) bool {
	a.mu.Lock()
	ring := a.audioHistory[participantID]
	a.mu.Unlock()
	if ring == nil {
		return false
	}
	ring.mu.Lock()
	defer ring.mu.Unlock()
	return ring.hasScoredSnapshot && ring.generation == ring.lastScoredGeneration
}

// waitForFreshAudio leaves a candidate the acoustic score already ruled unfinished until
// there is something new to score, looking again after primaryEOTLowRetry. When the patience
// for the words runs out first, the wait ends as it would on a score: the same words are
// answered with a question.
func (a *Agent) waitForFreshAudio(ready candidate, deadline time.Time) {
	remaining := time.Until(deadline)
	if remaining > 0 {
		a.converse.unaskedAfter(ready.ID, min(primaryEOTLowRetry, remaining))
		return
	}
	a.rulePrimaryEOTLow(harness.Decided{
		CandidateID: ready.ID,
		Disposition: harness.Wait,
		Floor:       harness.Continue,
	})
}
