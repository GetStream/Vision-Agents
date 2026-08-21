package caller

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/transport"
)

// DefaultTurnHangoverMS is how long the agent must be silent before the next caller turn.
const DefaultTurnHangoverMS = 1200

// DefaultClosingGraceMS is how long to wait for the agent to start another turn after the
// last scripted one before treating the call as finished.
const DefaultClosingGraceMS = 8000

// closingWait bounds how long the call stays open after the last scripted turn.
const closingWait = 25 * time.Second

// turnSettle bounds the same wait mid-call, before the next scripted turn plays. It is shorter
// than closingWait so one stuck turn cannot eat the whole scenario duration budget.
const turnSettle = 12 * time.Second

// turnSettleGrace is how long to wait for the agent to resume before treating a mid-call turn as
// finished. Measured across the hc-* recordings, an agent pausing inside its own reply comes back
// 20-660 ms after the hangover expires in five calls and 2240 ms in a sixth; the multi-second gaps
// in those calls all span a caller turn, which this wait never sees. Keeping it well under
// ClosingGraceMS matters because it is paid on every turn and the 10-turn coherence scenarios have
// a 240 s budget. A turn dropped for "overlap" in metrics.json is this wait coming up short.
const turnSettleGrace = 4 * time.Second

const paceInterval = 20 * time.Millisecond
const substantiveReplyMin = 2500 * time.Millisecond
const maxAgentJitterFrames = 5
const decodedSilenceThreshold = 0.000001

// Event marks when a scripted turn was played.
type Event struct {
	TurnID       string `json:"turn_id"`
	Kind         string `json:"kind"`
	RecStartMs   int    `json:"rec_start_ms"`
	RecEndMs     int    `json:"rec_end_ms"`
	BargeIn      bool   `json:"barge_in"`
	Overlap      bool   `json:"overlap"`
	Text         bool   `json:"caller_speech"`
	OverlapSound string `json:"overlap_sound,omitempty"`
}

// Result is the dual-leg recording plus turn timing.
type Result struct {
	Caller    []int16
	Agent     []int16
	Rate      int
	Events    []Event
	StartedAt time.Time
	// FirstFrameAt and LastFrameAt anchor recording time to wall time. Recording time advances
	// one 20 ms frame per pacer tick and a Go ticker drops ticks under load, so the two clocks
	// drift apart over a long call. Tool timestamps are wall time and have to be mapped across.
	FirstFrameAt time.Time
	LastFrameAt  time.Time
	// InboundDropped is how many agent frames the transport had to discard.
	InboundDropped int
	RequestedSNRDB float64
	MeasuredSNRDB  float64
}

// DurationMS is the recorded call length in recording time.
func (r Result) DurationMS() int {
	if r.Rate <= 0 {
		return 0
	}
	return len(r.Caller) * 1000 / r.Rate
}

// SampleMs maps a wall-clock instant onto recording time by interpolating between the first and
// last frame the pacer sent. Without the anchors it falls back to the offset from StartedAt,
// which is what a Result built by hand in a test gets.
func (r Result) SampleMs(t time.Time) int {
	wall := r.LastFrameAt.Sub(r.FirstFrameAt).Milliseconds()
	if r.FirstFrameAt.IsZero() || wall <= 0 {
		if r.StartedAt.IsZero() {
			return 0
		}
		return int(t.Sub(r.StartedAt).Milliseconds())
	}
	return int(t.Sub(r.FirstFrameAt).Milliseconds() * int64(r.DurationMS()) / wall)
}

// ClockDriftMS is how far recording time fell behind wall time over the call.
func (r Result) ClockDriftMS() int {
	if r.FirstFrameAt.IsZero() || r.LastFrameAt.IsZero() {
		return 0
	}
	return int(r.LastFrameAt.Sub(r.FirstFrameAt).Milliseconds()) - r.DurationMS()
}

// Engine plays scripted turns onto a Media pipe and records both legs.
type Engine struct {
	Audio          map[string][]int16
	Logger         *slog.Logger
	Threshold      float64
	TurnHangoverMS int
	ClosingGraceMS int
}

type clipJob struct {
	pcm     []int16
	startMs chan int
	endMs   chan int
}

// Play runs the caller script until turns finish or the context is done.
func (e Engine) Play(ctx context.Context, sc scenario.Scenario, media transport.Media) (Result, error) {
	if e.Logger == nil {
		e.Logger = slog.Default()
	}
	if e.Threshold == 0 {
		e.Threshold = audio.DefaultSpeechThreshold
	}
	if e.ClosingGraceMS == 0 {
		e.ClosingGraceMS = DefaultClosingGraceMS
	}
	if e.TurnHangoverMS == 0 {
		e.TurnHangoverMS = DefaultTurnHangoverMS
	}

	startedAt := time.Now()

	var mu sync.Mutex
	maxN := audio.Rate * max(sc.MaxDurationS, 60)
	callerRec := make([]int16, 0, maxN)
	agentRec := make([]int16, 0, maxN)
	var agentFrames []transport.Frame
	agentLive := false
	hasSpoken := false
	agentLiveAt := time.Time{}
	agentSegmentStartedAt := time.Time{}
	lastAgentSegment := time.Duration(0)
	agentStarted := make(chan struct{}, 8)
	agentEnded := make(chan struct{}, 8)
	hangover := time.Duration(e.TurnHangoverMS) * time.Millisecond
	grace := time.Duration(e.ClosingGraceMS) * time.Millisecond
	var firstFrameAt, lastFrameAt time.Time

	markSilent := func() {
		if !agentLive {
			return
		}
		agentLive = false
		if !agentSegmentStartedAt.IsZero() {
			lastAgentSegment = time.Since(agentSegmentStartedAt)
		}
		select {
		case agentEnded <- struct{}{}:
		default:
		}
	}

	recvDone := make(chan struct{})
	go func() {
		defer close(recvDone)
		ticker := time.NewTicker(paceInterval)
		defer ticker.Stop()
		for {
			select {
			case <-ctx.Done():
				return
			case frame, ok := <-media.Recv():
				if !ok {
					return
				}
				mu.Lock()
				agentFrames = append(agentFrames, frame)
				energy := audio.FrameEnergy(frame.PCM)
				if energy >= e.Threshold {
					if !agentLive {
						agentLive = true
						hasSpoken = true
						agentSegmentStartedAt = time.Now()
						select {
						case agentStarted <- struct{}{}:
						default:
						}
					}
					agentLiveAt = time.Now()
				} else if agentLive && !agentLiveAt.IsZero() && time.Since(agentLiveAt) >= hangover {
					markSilent()
				}
				mu.Unlock()
			case <-ticker.C:
				mu.Lock()
				if agentLive && !agentLiveAt.IsZero() && time.Since(agentLiveAt) >= hangover {
					markSilent()
				}
				mu.Unlock()
			}
		}
	}()

	isLive := func() bool {
		mu.Lock()
		defer mu.Unlock()
		return agentLive
	}
	spoken := func() bool {
		mu.Lock()
		defer mu.Unlock()
		return hasSpoken
	}
	drain := func(ch <-chan struct{}) {
		for {
			select {
			case <-ch:
			default:
				return
			}
		}
	}
	waitCh := func(ch <-chan struct{}, timeout time.Duration) bool {
		timer := time.NewTimer(timeout)
		defer timer.Stop()
		select {
		case <-ctx.Done():
			return false
		case <-ch:
			return true
		case <-timer.C:
			return false
		}
	}
	waitSilence := func() {
		for isLive() && ctx.Err() == nil {
			drain(agentEnded)
			if !isLive() {
				return
			}
			waitCh(agentEnded, 2*time.Second)
		}
	}
	waitAgentTurn := func(startTimeout time.Duration, first bool) bool {
		if isLive() {
			waitSilence()
			return true
		}
		if first && spoken() {
			return true
		}
		drain(agentStarted)
		drain(agentEnded)
		if isLive() {
			waitSilence()
			return true
		}
		if !waitCh(agentStarted, startTimeout) {
			if isLive() {
				waitSilence()
				return true
			}
			return false
		}
		waitSilence()
		return true
	}
	waitInsideAgent := func(delay time.Duration) {
		deadline := time.Now().Add(12 * time.Second)
		for ctx.Err() == nil && time.Now().Before(deadline) {
			if !isLive() {
				drain(agentStarted)
				if !isLive() && !waitCh(agentStarted, time.Until(deadline)) {
					return
				}
			}
			drain(agentEnded)
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return
			case <-agentEnded:
				timer.Stop()
				continue
			case <-timer.C:
				if isLive() {
					return
				}
			}
		}
	}
	segmentDuration := func() time.Duration {
		mu.Lock()
		defer mu.Unlock()
		return lastAgentSegment
	}

	// settle keeps waiting through short filler phrases, but does not add the full resume grace
	// after a substantive reply. This keeps long coherence scripts inside their call budget.
	settle := func(total, wait time.Duration, responseObserved bool) {
		deadline := time.Now().Add(total)
		drain(agentStarted)
		drain(agentEnded)
		for ctx.Err() == nil {
			if isLive() {
				waitSilence()
				responseObserved = true
			}
			if responseObserved && segmentDuration() > substantiveReplyMin {
				return
			}
			remaining := time.Until(deadline)
			if remaining <= 0 || !waitCh(agentStarted, min(remaining, wait)) {
				return
			}
			responseObserved = true
		}
	}

	jobs := make(chan clipJob, 1)
	stopPacer := make(chan struct{})
	pacerDone := make(chan struct{})
	var stopOnce sync.Once
	stop := func() {
		stopOnce.Do(func() {
			close(stopPacer)
			<-pacerDone
		})
	}
	defer stop()

	var bed []int16
	requestedSNR := 0.0
	measuredSNR := 0.0
	if sc.Noise != "" && sc.Noise != "none" {
		requestedSNR = sc.SNRDB
		if requestedSNR == 0 {
			requestedSNR = 10
		}
		var callerSpeech []int16
		for _, text := range sc.SpeechTexts() {
			callerSpeech = append(callerSpeech, e.Audio[text]...)
		}
		bed = audio.ScaleNoiseForSignalSNR(
			audio.NoiseNamed(sc.Noise, audio.Rate*5, 42), callerSpeech, requestedSNR,
		)
		measuredSNR = audio.MeasuredSNRDB(callerSpeech, bed)
	}

	var sendFail = make(chan error, 1)
	go func() {
		defer close(pacerDone)
		tick := time.NewTicker(paceInterval)
		defer tick.Stop()
		var job *clipJob
		off := 0
		frame := make([]int16, audio.FrameSamples)
		silence := make([]int16, audio.FrameSamples)
		for {
			select {
			case <-ctx.Done():
				return
			case <-stopPacer:
				return
			case <-tick.C:
			}
			if job == nil {
				select {
				case j := <-jobs:
					job = &j
					off = 0
				default:
				}
			}
			clear(frame)
			if job != nil {
				if off == 0 {
					mu.Lock()
					start := len(callerRec) * 1000 / audio.Rate
					mu.Unlock()
					job.startMs <- start
				}
				n := copy(frame, job.pcm[off:])
				off += n
			}
			if len(bed) > 0 {
				mu.Lock()
				pos := len(callerRec)
				mu.Unlock()
				audio.Add(frame, bed, pos)
			}
			if err := media.Send(frame); err != nil {
				select {
				case sendFail <- err:
				default:
				}
				return
			}
			mu.Lock()
			if firstFrameAt.IsZero() {
				firstFrameAt = time.Now()
			}
			lastFrameAt = time.Now()
			callerRec = append(callerRec, frame...)
			for len(agentFrames) > maxAgentJitterFrames && audio.FrameEnergy(agentFrames[0].PCM) <= decodedSilenceThreshold {
				agentFrames = agentFrames[1:]
			}
			if len(agentFrames) > 0 {
				agentRec = append(agentRec, agentFrames[0].PCM...)
				agentFrames = agentFrames[1:]
			} else {
				agentRec = append(agentRec, silence...)
			}
			end := len(callerRec) * 1000 / audio.Rate
			mu.Unlock()
			if job != nil && off >= len(job.pcm) {
				job.endMs <- end
				job = nil
			}
		}
	}()

	playClip := func(pcm []int16) (int, int, error) {
		if len(pcm) == 0 {
			mu.Lock()
			ms := len(callerRec) * 1000 / audio.Rate
			mu.Unlock()
			return ms, ms, nil
		}
		startCh := make(chan int, 1)
		endCh := make(chan int, 1)
		select {
		case jobs <- clipJob{pcm: pcm, startMs: startCh, endMs: endCh}:
		case err := <-sendFail:
			return 0, 0, err
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		var start, end int
		select {
		case start = <-startCh:
		case err := <-sendFail:
			return 0, 0, err
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		select {
		case end = <-endCh:
		case err := <-sendFail:
			return 0, 0, err
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		return start, end, nil
	}

	var events []Event
	finish := func() Result {
		stop()
		select {
		case <-recvDone:
		case <-time.After(200 * time.Millisecond):
		}
		mu.Lock()
		defer mu.Unlock()
		n := len(callerRec)
		if len(agentRec) > n {
			n = len(agentRec)
		}
		return Result{
			InboundDropped: media.Dropped(),
			Caller:         audio.PadRight(callerRec, n),
			Agent:          audio.PadRight(agentRec, n),
			Rate:           audio.Rate,
			Events:         events,
			StartedAt:      startedAt,
			FirstFrameAt:   firstFrameAt,
			LastFrameAt:    lastFrameAt,
			RequestedSNRDB: requestedSNR,
			MeasuredSNRDB:  measuredSNR,
		}
	}

	for i, turn := range sc.Turns {
		if err := ctx.Err(); err != nil {
			return finish(), err
		}
		kind := turn.Trigger.Kind
		if kind == "" {
			kind = scenario.TriggerAfterAgent
		}
		switch kind {
		case scenario.TriggerAfterAgent:
			responseObserved := false
			if i == 0 {
				responseObserved = waitAgentTurn(8*time.Second, true)
			} else {
				responseObserved = waitAgentTurn(12*time.Second, false)
			}
			settle(turnSettle, turnSettleGrace, responseObserved)
			delay := time.Duration(turn.Trigger.DelayMS) * time.Millisecond
			if delay == 0 {
				delay = 400 * time.Millisecond
			}
			time.Sleep(delay)
		case scenario.TriggerBargeIn, scenario.TriggerDuringAgent:
			wait := time.Duration(turn.Trigger.AfterMS) * time.Millisecond
			if wait == 0 {
				wait = 800 * time.Millisecond
			}
			waitInsideAgent(wait)
		case scenario.TriggerImmediate:
		}

		var pcm []int16
		switch {
		case turn.OverlapSound != "":
			pcm = audio.NoiseNamed(turn.OverlapSound, 0, int64(i+1))
			if len(pcm) == 0 {
				pcm = audio.Cough(int64(i + 1))
			}
		case turn.Text != "":
			cached, ok := e.Audio[turn.Text]
			if !ok {
				return finish(), fmt.Errorf("caller: missing audio for turn %s", turn.ID)
			}
			pcm = cached
		}
		overlap := kind != scenario.TriggerBargeIn && turn.OverlapSound == "" && isLive()
		recStart, recEnd, err := playClip(pcm)
		if err != nil {
			return finish(), err
		}
		events = append(events, Event{
			TurnID:       turn.ID,
			Kind:         kind,
			RecStartMs:   recStart,
			RecEndMs:     recEnd,
			BargeIn:      kind == scenario.TriggerBargeIn || turn.OverlapSound != "",
			Overlap:      overlap,
			Text:         turn.Text != "",
			OverlapSound: turn.OverlapSound,
		})
	}

	settle(closingWait, grace, false)
	time.Sleep(400 * time.Millisecond)

	stop()

	select {
	case err := <-sendFail:
		return finish(), err
	default:
	}
	return finish(), nil
}
