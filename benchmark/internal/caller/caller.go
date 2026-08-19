package caller

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/benchmark/internal/audio"
	"github.com/GetStream/Vision-Agents/benchmark/internal/scenario"
	"github.com/GetStream/Vision-Agents/benchmark/internal/telephony"
)

// DefaultTurnHangoverMS is how long the agent must be silent before the next caller turn.
const DefaultTurnHangoverMS = 1200

const closingWait = 15 * time.Second

const paceInterval = 20 * time.Millisecond

// Event marks when a scripted turn was played.
type Event struct {
	TurnID     string
	Kind       string
	RecStartMs int
	RecEndMs   int
	BargeIn    bool
	Overlap    bool
}

// Result is the dual-leg recording plus turn timing.
type Result struct {
	Caller    []int16
	Agent     []int16
	Rate      int
	Events    []Event
	StartedAt time.Time
}

// Engine plays scripted turns onto a Media pipe and records both legs.
type Engine struct {
	Audio          map[string][]int16
	Logger         *slog.Logger
	Threshold      float64
	TurnHangoverMS int
}

type clipJob struct {
	pcm     []int16
	startMs chan int
	endMs   chan int
}

// Play runs the caller script until turns finish or the context is done.
func (e Engine) Play(ctx context.Context, sc scenario.Scenario, media telephony.Media) (Result, error) {
	if e.Logger == nil {
		e.Logger = slog.Default()
	}
	if e.Threshold == 0 {
		e.Threshold = audio.DefaultSpeechThreshold
	}
	if e.TurnHangoverMS == 0 {
		e.TurnHangoverMS = DefaultTurnHangoverMS
	}

	startedAt := time.Now()

	var mu sync.Mutex
	var callerRec []int16
	var agentRec []int16
	agentLive := false
	hasSpoken := false
	agentLiveAt := time.Time{}
	agentStarted := make(chan struct{}, 8)
	agentEnded := make(chan struct{}, 8)
	hangover := time.Duration(e.TurnHangoverMS) * time.Millisecond

	markSilent := func() {
		if !agentLive {
			return
		}
		agentLive = false
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
				for len(agentRec) < len(callerRec) {
					agentRec = append(agentRec, 0)
				}
				agentRec = append(agentRec, frame.PCM...)
				energy := audio.FrameEnergy(frame.PCM)
				if energy >= e.Threshold {
					if !agentLive {
						agentLive = true
						hasSpoken = true
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
	waitAgentTurn := func(startTimeout time.Duration, first bool) {
		if isLive() {
			waitSilence()
			return
		}
		if first && spoken() {
			return
		}
		drain(agentStarted)
		drain(agentEnded)
		if isLive() {
			waitSilence()
			return
		}
		if !waitCh(agentStarted, startTimeout) {
			if isLive() {
				waitSilence()
			}
			return
		}
		waitSilence()
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
	if sc.Noise != "" && sc.Noise != "none" {
		n := audio.TelnyxRate * max(sc.MaxDurationS, 60)
		bed = audio.ScaleNoiseForSNR(audio.NoiseNamed(sc.Noise, n, 42), snrOrDefault(sc.SNRDB))
	}

	go func() {
		defer close(pacerDone)
		tick := time.NewTicker(paceInterval)
		defer tick.Stop()
		var job *clipJob
		off := 0
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
			frame := make([]int16, audio.FrameSamples)
			if job != nil {
				if off == 0 {
					mu.Lock()
					start := len(callerRec) * 1000 / audio.TelnyxRate
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
				frame = audio.Add(frame, bed, pos)
			}
			_ = media.Send(frame)
			mu.Lock()
			callerRec = append(callerRec, frame...)
			end := len(callerRec) * 1000 / audio.TelnyxRate
			for len(agentRec) < len(callerRec) {
				agentRec = append(agentRec, 0)
			}
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
			ms := len(callerRec) * 1000 / audio.TelnyxRate
			mu.Unlock()
			return ms, ms, nil
		}
		startCh := make(chan int, 1)
		endCh := make(chan int, 1)
		select {
		case jobs <- clipJob{pcm: pcm, startMs: startCh, endMs: endCh}:
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		var start, end int
		select {
		case start = <-startCh:
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		select {
		case end = <-endCh:
		case <-ctx.Done():
			return 0, 0, ctx.Err()
		}
		return start, end, nil
	}

	var events []Event

	for i, turn := range sc.Turns {
		if err := ctx.Err(); err != nil {
			return Result{}, err
		}
		kind := turn.Trigger.Kind
		if kind == "" {
			kind = scenario.TriggerAfterAgent
		}
		switch kind {
		case scenario.TriggerAfterAgent:
			if i == 0 {
				waitAgentTurn(8*time.Second, true)
			} else {
				waitAgentTurn(12*time.Second, false)
			}
			delay := time.Duration(turn.Trigger.DelayMS) * time.Millisecond
			if delay == 0 {
				delay = 400 * time.Millisecond
			}
			time.Sleep(delay)
		case scenario.TriggerBargeIn, scenario.TriggerDuringAgent:
			if !isLive() {
				drain(agentStarted)
				if !isLive() {
					waitCh(agentStarted, 8*time.Second)
				}
			}
			wait := time.Duration(turn.Trigger.AfterMS) * time.Millisecond
			if wait == 0 {
				wait = 800 * time.Millisecond
			}
			time.Sleep(wait)
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
				return Result{}, fmt.Errorf("caller: missing audio for turn %s", turn.ID)
			}
			pcm = cached
		}
		overlap := kind != scenario.TriggerBargeIn && turn.OverlapSound == "" && isLive()
		recStart, recEnd, err := playClip(pcm)
		if err != nil {
			return Result{}, err
		}
		events = append(events, Event{
			TurnID:     turn.ID,
			Kind:       kind,
			RecStartMs: recStart,
			RecEndMs:   recEnd,
			BargeIn:    kind == scenario.TriggerBargeIn || turn.OverlapSound != "",
			Overlap:    overlap,
		})
	}

	drain(agentStarted)
	drain(agentEnded)
	if !isLive() {
		waitCh(agentStarted, closingWait)
	}
	waitSilence()
	time.Sleep(400 * time.Millisecond)

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
		Caller:    audio.PadRight(callerRec, n),
		Agent:     audio.PadRight(agentRec, n),
		Rate:      audio.TelnyxRate,
		Events:    events,
		StartedAt: startedAt,
	}, nil
}

func snrOrDefault(snr float64) float64 {
	if snr == 0 {
		return 10
	}
	return snr
}
