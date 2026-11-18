// Package echocancel takes the agent's own voice out of what a caller's line sends back.
//
// A phone with no echo cancellation of its own, a speakerphone above all, plays the agent's
// voice into its microphone, and the line carries it back to the agent, where the transcriber
// writes it as the caller's words and barge-in takes it for the caller talking over the agent.
// This removes it with AEC3 (github.com/GetStream/goaec3), WebRTC's echo canceller: the voice
// as it goes out on the call is the reference, and each listened-to line is cleaned of it
// before anything hears that line.
//
// Everything here is 16 kHz mono PCM in 10 ms frames, what the agent hears and AEC3 runs at.
package echocancel

import (
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/goaec3"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"

	"github.com/GetStream/Vision-Agents/acceleration/internal/echocancel/echodelay"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
)

const (
	// SampleRate is the rate both directions are cancelled at: the rate the agent hears
	// participants at, and the lowest AEC3 runs at. A phone line carries nothing above 4 kHz.
	SampleRate = 16_000
	// FrameSamples is one 10 ms frame, the only frame AEC3 takes.
	FrameSamples = SampleRate / 100
	// FrameDuration is how long a frame plays for.
	FrameDuration = 10 * time.Millisecond
	// playedFrames is how much of the agent's voice a line holds that its canceller has not
	// taken yet: a second. The voice is taken on the line's own goroutine as the line's audio
	// arrives, so a line whose audio stops for longer drops the oldest of it, and the
	// canceller, whose voice and line no longer line up, is replaced.
	playedFrames = 100
)

const (
	// filterWindow and filterShift are AEC3's matched filter, the one that finds the echo's
	// delay, in 4 ms blocks: each filter covers 32 blocks, 128 ms, and starts 24 blocks, 96 ms,
	// after the one before it (kMatchedFilterWindowSizeSubBlocks and
	// kMatchedFilterAlignmentShiftSizeSubBlocks in WebRTC's aec3_common.h).
	filterWindow = echodelay.Min
	filterShift  = 96 * time.Millisecond
)

// Config is how a call's echo is cancelled.
type Config struct {
	// MaxDelay is the longest an echo is looked for after the voice went out, from
	// echodelay.Min to echodelay.Max. Zero is echodelay.Default.
	MaxDelay time.Duration
}

// filters is how many matched filters cover maxDelay.
func filters(maxDelay time.Duration) int {
	return 1 + int((maxDelay-filterWindow+filterShift-1)/filterShift)
}

// aec3Config is the AEC3 configuration for cfg: WebRTC's default with the echo's delay looked
// for up to cfg.MaxDelay.
func aec3Config(cfg Config) (goaec3.Config, error) {
	maxDelay := cfg.MaxDelay
	if maxDelay == 0 {
		maxDelay = echodelay.Default
	}
	if err := echodelay.Check(maxDelay); err != nil {
		return goaec3.Config{}, err
	}
	config := goaec3.DefaultConfig()
	config.Delay.NumFilters = filters(maxDelay)
	return config, nil
}

// Check reports whether cfg is one a call can be cancelled with, by building a canceller with
// it: AEC3 refuses some configs its own bounds allow.
func Check(cfg Config) error {
	_, err := NewCall(cfg)
	return err
}

// Every series is one value per line, observed when it closes, so a bucket counts lines.
var (
	lineERLE = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_echo_return_loss_enhancement_db",
		Help:      "How much quieter the echo canceller made the agent's voice coming back on a caller's line (ERLE), in dB, as AEC3 last estimated it when the line closed",
		Buckets:   []float64{0, 3, 6, 10, 15, 20, 25, 30, 40},
	})
	lineERL = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_echo_return_loss_db",
		Help:      "How much quieter the agent's voice was on its way back on a caller's line than when it went out (ERL), in dB, as AEC3 last estimated it when the line closed. AEC3 holds its estimate from -30 to 20 dB, so 20 is 20 or more",
		Buckets:   []float64{-20, -10, -5, 0, 5, 10, 15, 19.9},
	})
	lineDelay = promauto.NewHistogram(prometheus.HistogramOpts{
		Namespace: "router",
		Name:      "agent_echo_delay_seconds",
		Help:      "How long after it went out the agent's voice came back as echo on a caller's line, as AEC3 last estimated it when the line closed",
		Buckets:   []float64{0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1, 1.2},
	})
	linesCancelled = promauto.NewCounterVec(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_echo_lines_total",
		Help:      "Callers' lines the echo canceller listened to, by whether the agent's voice went out while it did, without which there is no echo to measure",
	}, []string{"voiced"})
	voiceDropped = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_echo_voice_dropped_frames_total",
		Help:      "10 ms frames of the agent's voice a caller's line let go before its canceller took them, because the line's audio stopped for more than a second",
	})
	gapFrames = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_echo_gap_frames_total",
		Help:      "10 ms frames missing from a caller's audio, by its timestamps, that the echo canceller stepped over as silence to keep the agent's voice lined up with its echo",
	})
	cancellerResets = promauto.NewCounter(prometheus.CounterOpts{
		Namespace: "router",
		Name:      "agent_echo_canceller_resets_total",
		Help:      "Echo cancellers replaced on a caller's line because its audio stopped for longer than the agent's voice is held for, after which the two no longer line up",
	})
)

// Call is the agent's voice on one call, as it goes out, and the lines it is cancelled from.
type Call struct {
	config goaec3.Config

	mu    sync.Mutex
	lines []*Line
	// listening counts lines and expected the participants who will have one, both read
	// without the lock by whoever decides whether to keep voice.
	listening, expected atomic.Int32

	// partial is the start of the next frame of voice. Only Played touches it.
	partial  [FrameSamples]int16
	buffered int
}

// NewCall returns a Call that cancels echo as cfg says. It builds one canceller to check the
// config with, so a call that could not cancel any line is refused here.
func NewCall(cfg Config) (*Call, error) {
	config, err := aec3Config(cfg)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	if _, err := goaec3.New(config, SampleRate, 1, 1); err != nil {
		return nil, stack.Wrap(fmt.Errorf("echocancel: %w", err))
	}
	return &Call{config: config}, nil
}

// Played is the agent's voice as it went out, 16 kHz mono, and silence as zeros while it
// says nothing: AEC3 lines the voice up with its echo by when each frame went out, so the
// silences between replies count as much as the replies. It is called on one goroutine,
// the one that sends the voice, and does not allocate.
func (c *Call) Played(samples []int16) {
	for len(samples) > 0 {
		n := copy(c.partial[c.buffered:], samples)
		samples = samples[n:]
		c.buffered += n
		if c.buffered < FrameSamples {
			return
		}
		c.buffered = 0
		c.mu.Lock()
		for _, line := range c.lines {
			line.played(&c.partial)
		}
		c.mu.Unlock()
	}
}

// Listen starts cancelling the agent's voice from one more line. The line is cleaned on its
// own goroutine, alongside Played, and is closed when it ends.
func (c *Call) Listen() (*Line, error) {
	canceller, err := goaec3.New(c.config, SampleRate, 1, 1)
	if err != nil {
		return nil, stack.Wrap(fmt.Errorf("echocancel: %w", err))
	}
	line := &Line{
		call:      c,
		canceller: canceller,
		render:    [][]float32{make([]float32, FrameSamples)},
		capture:   [][]float32{make([]float32, FrameSamples)},
	}
	c.mu.Lock()
	c.lines = append(c.lines, line)
	c.listening.Add(1)
	c.mu.Unlock()
	return line, nil
}

// Expect says a participant whose line will be cleaned arrived (1) or left (-1). Their line
// opens only once their audio does, and the voice going out before then is still what their
// echo will be of.
func (c *Call) Expect(delta int) { c.expected.Add(int32(delta)) }

// Wanted reports whether the voice is of use: a line is being cleaned or a participant whose
// line will be is in the call.
func (c *Call) Wanted() bool { return c.listening.Load() > 0 || c.expected.Load() > 0 }

// Line is one caller's line being cleaned of the agent's voice. Clean and Close are called on
// one goroutine, the one the line's audio arrives on.
type Line struct {
	call      *Call
	canceller *goaec3.EchoCanceller

	// mu guards the voice played and not yet taken, written by Played and taken by Clean.
	mu sync.Mutex
	// voice holds the frames, waiting the oldest first from next, count of them.
	voice       [playedFrames][FrameSamples]int16
	next, count int
	// dropped counts the frames let go unread since the voice was last taken.
	dropped int
	// voiced is whether any of the voice played was audible, and detached whether the line
	// is no longer played any.
	voiced, detached bool

	// taking is the voice taken from the queue, so its lock is not held through AEC3.
	taking [playedFrames][FrameSamples]int16
	// render and capture are the frames AEC3 is handed.
	render, capture [][]float32
	// pending is the start of the next frame of the line, held until it is whole, and
	// silence a frame of nothing, which stands for one missing from the line.
	pending  [FrameSamples]int16
	buffered int
	silence  [FrameSamples]int16
	closed   bool
}

// played queues one frame of the agent's voice for the line's canceller.
func (l *Line) played(frame *[FrameSamples]int16) {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.count == playedFrames {
		l.next = (l.next + 1) % playedFrames
		l.count--
		l.dropped++
		voiceDropped.Inc()
	}
	l.voice[(l.next+l.count)%playedFrames] = *frame
	l.count++
	if !l.voiced {
		for _, sample := range frame {
			if sample != 0 {
				l.voiced = true
				break
			}
		}
	}
}

// take moves the voice queued since the last call into taking, and says how many frames, and
// whether any were dropped before they could be taken.
func (l *Line) take() (int, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	n := l.count
	for i := range n {
		l.taking[i] = l.voice[(l.next+i)%playedFrames]
	}
	dropped := l.dropped > 0
	l.next, l.count, l.dropped = 0, 0, 0
	return n, dropped
}

// Clean takes the agent's voice out of a chunk of the line, 16 kHz mono, of any length.
//
// AEC3 cleans whole 10 ms frames, so what is returned is every frame the chunk completes,
// with the end of the chunk that does not fill one held until the next. A chunk of whole
// frames following one, which is what a call's 20 ms packets are, is cleaned in place and
// returned without delay or allocation; any other is returned in a new slice, starting with
// what was held. Either way the result is the caller's.
func (l *Line) Clean(samples []int16) []int16 {
	if l.buffered == 0 && len(samples)%FrameSamples == 0 {
		for at := 0; at < len(samples); at += FrameSamples {
			l.cleanFrame(samples[at : at+FrameSamples])
		}
		return samples
	}
	frames := (l.buffered + len(samples)) / FrameSamples
	if frames == 0 {
		l.buffered += copy(l.pending[l.buffered:], samples)
		return nil
	}
	out := make([]int16, frames*FrameSamples)
	used := copy(out[copy(out, l.pending[:l.buffered]):], samples)
	l.buffered = copy(l.pending[:], samples[used:])
	for at := 0; at < len(out); at += FrameSamples {
		l.cleanFrame(out[at : at+FrameSamples])
	}
	return out
}

// cleanFrame hands AEC3 the voice played since the frame before, then cleans this one in
// place.
func (l *Line) cleanFrame(frame []int16) {
	n, dropped := l.take()
	if dropped {
		l.reset()
	}
	for i := range n {
		toFloat(l.render[0], l.taking[i][:])
		// Neither call fails: the frames are built here to the size the canceller was made for.
		_ = l.canceller.AnalyzeRender(l.render)
	}
	toFloat(l.capture[0], frame)
	_ = l.canceller.ProcessCapture(l.capture)
	fromFloat(frame, l.capture[0])
}

// Missed steps the line over audio missing from it, d long by its timestamps: a caller's
// phone that sends nothing while its user is quiet (Opus DTX), or packets lost past what the
// decoder conceals. The agent's voice went on going out meanwhile, so each missing 10 ms is
// cleaned as silence, which keeps the voice and the line lined up; a gap longer than the
// voice a line holds replaces the canceller instead, as nothing lines up after it.
func (l *Line) Missed(d time.Duration) {
	frames := int((d + FrameDuration/2) / FrameDuration)
	if frames <= 0 {
		return
	}
	if frames >= playedFrames {
		l.reset()
		return
	}
	gapFrames.Add(float64(frames))
	for range frames {
		l.cleanFrame(l.silence[:])
		clear(l.silence[:])
	}
}

// reset replaces the canceller with a new one, which finds the echo afresh.
func (l *Line) reset() {
	canceller, err := goaec3.New(l.call.config, SampleRate, 1, 1)
	if err != nil {
		// NewCall built one with the same config.
		return
	}
	l.canceller = canceller
	cancellerResets.Inc()
}

// Metrics are what AEC3 last estimated of a line's echo.
type Metrics struct {
	// ERL is how much quieter the echo is than the voice that went out, and ERLE how much
	// quieter the canceller made it, both in dB.
	ERL, ERLE float64
	// Delay is how long after the voice went out its echo came back.
	Delay time.Duration
}

// Metrics is what AEC3 estimates of the line's echo now.
func (l *Line) Metrics() Metrics {
	m := l.canceller.Metrics()
	return Metrics{ERL: m.EchoReturnLoss, ERLE: m.EchoReturnLossEnhancement,
		Delay: time.Duration(m.DelayMs) * time.Millisecond}
}

// Detach stops playing the line any more of the agent's voice, as the line has ended though
// whoever cleans it has not closed it yet. It may be called from any goroutine, more than once.
func (l *Line) Detach() {
	c := l.call
	c.mu.Lock()
	defer c.mu.Unlock()
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.detached {
		return
	}
	l.detached = true
	for i, line := range c.lines {
		if line == l {
			c.lines = append(c.lines[:i], c.lines[i+1:]...)
			c.listening.Add(-1)
			break
		}
	}
}

// Close stops cancelling the line, counts it and, if the agent's voice went out while it was
// listened to, observes what AEC3 last estimated of its echo. It returns those estimates.
func (l *Line) Close() Metrics {
	l.Detach()

	metrics := l.Metrics()
	l.mu.Lock()
	voiced, closed := l.voiced, l.closed
	l.closed = true
	l.mu.Unlock()
	if closed {
		return metrics
	}
	linesCancelled.WithLabelValues(strconv.FormatBool(voiced)).Inc()
	if voiced {
		lineERLE.Observe(metrics.ERLE)
		lineERL.Observe(metrics.ERL)
		lineDelay.Observe(metrics.Delay.Seconds())
	}
	return metrics
}

// toFloat writes PCM as AEC3's samples, in [-1, 1).
func toFloat(dst []float32, src []int16) {
	for i, s := range src {
		dst[i] = float32(s) / 32768
	}
}

// fromFloat writes AEC3's samples back as PCM, rounded and clipped.
func fromFloat(dst []int16, src []float32) {
	for i, s := range src {
		v := math.Round(float64(s) * 32768)
		dst[i] = int16(max(math.MinInt16, min(math.MaxInt16, v)))
	}
}
