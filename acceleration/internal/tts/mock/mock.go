// Package mock implements the tts.TTS contract without a vendor behind it, for load
// tests that want a cascaded conversation without paying for one.
//
// It says nothing: the text decides how long the speech is, and what comes back is a tone
// at that length. What it does keep is the timing a listener feels — a wait before the
// first chunk, then audio arriving in pieces faster than it plays, and barge-in dropping
// the rest — so the speaker queue and the turn-taking downstream have the same work to do
// as they would with a real voice.
package mock

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "mock"

// DefaultModel is what the catalogue declares.
const DefaultModel = "mock-tts"

// DefaultSampleRate is the rate the tone is generated at, which is what the real
// providers here produce.
const DefaultSampleRate = 24_000

const (
	// defaultCharsPerSecond is how fast the voice speaks, which is what turns a sentence
	// into a duration. Fifteen is an unhurried conversational pace.
	defaultCharsPerSecond = 15.0
	// defaultTimeToFirstByteMs is the wait before the first chunk, in the range the
	// streaming vendors answer in.
	defaultTimeToFirstByteMs = 150
	// defaultRealtimeFactor is how much faster than playback the audio is generated.
	defaultRealtimeFactor = 5.0
	// chunkMs is how much speech one chunk carries.
	chunkMs = 200
	// toneHz is the pitch of the tone, low enough to sit where a voice does.
	toneHz = 220
	// toneLevel keeps the tone well under full scale, so a listener measuring energy
	// sees speech rather than a clipped square.
	toneLevel = 0.2
	// eventBuffer is the consumer's slack, matching what the real providers allow.
	eventBuffer = 64
)

// Options configures the provider.
type Options struct {
	Model string
	Voice string
	// CharsPerSecond is how fast the voice speaks, TimeToFirstByteMs the wait before the
	// first chunk, and RealtimeFactor how much faster than playback the rest arrives.
	CharsPerSecond    float64
	TimeToFirstByteMs int
	RealtimeFactor    float64
	SampleRate        int
	Logger            *slog.Logger
}

// TTS is a voice that produces a tone of the right length instead of speech.
type TTS struct {
	model             string
	voice             string
	charsPerSecond    float64
	timeToFirstByteMs int
	realtimeFactor    float64
	sampleRate        int
	logger            *slog.Logger

	emitter *tts.Emitter

	// ctx is the session, cancelled by Close, and interrupted is the utterance in
	// flight, cancelled by barge-in.
	ctx         context.Context
	stop        context.CancelFunc
	mu          sync.Mutex
	interrupted context.CancelFunc
	// speaking counts the utterances still generating, so Close waits for them rather
	// than closing the channel under them.
	speaking sync.WaitGroup
	// pending is the text of each utterance so far, keyed by synthesis id, since a
	// sentence arrives as deltas and is only said once it is closed.
	pending map[string]string
}

// New returns an unstarted provider.
func New(options Options) (*TTS, error) {
	if options.CharsPerSecond < 0 || options.RealtimeFactor < 0 {
		return nil, errors.New("mock: a speaking pace cannot be negative")
	}

	provider := &TTS{
		model:             options.Model,
		voice:             options.Voice,
		charsPerSecond:    options.CharsPerSecond,
		timeToFirstByteMs: options.TimeToFirstByteMs,
		realtimeFactor:    options.RealtimeFactor,
		sampleRate:        options.SampleRate,
		logger:            options.Logger,
		emitter:           tts.NewEmitter(eventBuffer),
		pending:           map[string]string{},
	}
	if provider.model == "" {
		provider.model = DefaultModel
	}
	if provider.charsPerSecond == 0 {
		provider.charsPerSecond = defaultCharsPerSecond
	}
	if provider.timeToFirstByteMs == 0 {
		provider.timeToFirstByteMs = defaultTimeToFirstByteMs
	}
	if provider.realtimeFactor == 0 {
		provider.realtimeFactor = defaultRealtimeFactor
	}
	if provider.sampleRate == 0 {
		provider.sampleRate = DefaultSampleRate
	}
	if provider.logger == nil {
		provider.logger = slog.Default()
	}
	return provider, nil
}

// Start opens the session. There is nothing upstream, so it only says so.
func (t *TTS) Start(ctx context.Context) error {
	t.ctx, t.stop = context.WithCancel(context.WithoutCancel(ctx))
	t.emitter.Send(tts.Connected{Provider: ProviderName, Model: t.model, At: time.Now()})
	return nil
}

// Synthesize accumulates a sentence and speaks it once the turn closes it.
func (t *TTS) Synthesize(request tts.Request) error {
	if t.ctx == nil {
		return errors.New("mock: the session is not open")
	}
	if request.ID == "" {
		request.ID = fmt.Sprintf("mock-%d", time.Now().UnixNano())
	}

	t.mu.Lock()
	said := t.pending[request.ID] + request.Text
	if !request.Final {
		t.pending[request.ID] = said
		t.mu.Unlock()
		return nil
	}
	delete(t.pending, request.ID)
	ctx, interrupt := context.WithCancel(t.ctx)
	if t.interrupted != nil {
		t.interrupted()
	}
	t.interrupted = interrupt
	t.mu.Unlock()

	t.speaking.Add(1)
	go func() {
		defer t.speaking.Done()
		defer interrupt()
		t.speak(ctx, request.ID, said)
	}()
	return nil
}

// Interrupt drops the audio of the utterance in flight, which is barge-in.
func (t *TTS) Interrupt() error {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.interrupted != nil {
		t.interrupted()
		t.interrupted = nil
	}
	return nil
}

// Events carries the audio back.
func (t *TTS) Events() <-chan tts.Event { return t.emitter.Events() }

// Close ends the session, once the utterances in flight have settled.
func (t *TTS) Close() error {
	if t.stop != nil {
		t.stop()
	}
	t.speaking.Wait()
	t.emitter.Send(tts.Disconnected{Provider: ProviderName, Model: t.model, Clean: true, At: time.Now()})
	t.emitter.Close()
	return nil
}

// Provider is the stable provider name used in stats.
func (t *TTS) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (t *TTS) Model() string { return t.model }

// Streaming reports that deltas are accepted, as every socket provider here does.
func (t *TTS) Streaming() bool { return true }

// Prompt is empty: there is no voice to describe to a model writing for it.
func (t *TTS) Prompt() string { return "" }

// Performs reports that bracketed directions are read out rather than acted, so the agent
// takes them out of the text first, as it would for most real voices.
func (t *TTS) Performs() bool { return false }

// Voice is the speaker this session was opened with.
func (t *TTS) Voice() string { return t.voice }

// SampleRate is the rate the audio comes back at.
func (t *TTS) SampleRate() int { return t.sampleRate }

// speak streams the tone one utterance amounts to, faster than it plays, and settles it.
func (t *TTS) speak(ctx context.Context, id, said string) {
	started := time.Now()
	t.emitter.Send(tts.SynthesisStarted{
		SynthesisID: id,
		Provider:    ProviderName,
		Model:       t.model,
		Voice:       t.voice,
		At:          started,
	})

	if !t.wait(ctx, time.Duration(t.timeToFirstByteMs)*time.Millisecond) {
		t.settle(id, said, started, 0, true)
		return
	}
	firstByteAt := time.Now()

	durationMs := float64(len([]rune(said))) / t.charsPerSecond * 1000
	var saidMs float64
	for index := 0; saidMs < durationMs; index++ {
		piece := math.Min(chunkMs, durationMs-saidMs)
		t.emitter.Send(tts.AudioChunk{
			SynthesisID: id,
			Index:       index,
			Audio:       tone(t.sampleRate, saidMs, piece),
		})
		saidMs += piece
		if !t.wait(ctx, time.Duration(piece/t.realtimeFactor)*time.Millisecond) {
			break
		}
	}

	complete := tts.SynthesisComplete{
		SynthesisID:       id,
		Provider:          ProviderName,
		Model:             t.model,
		Characters:        int64(len(said)),
		AudioDurationMs:   saidMs,
		TimeToFirstByteMs: float64(firstByteAt.Sub(started).Microseconds()) / 1000,
		SynthesisTimeMs:   float64(time.Since(started).Microseconds()) / 1000,
		Interrupted:       saidMs < durationMs,
	}
	t.emitter.Send(complete)
}

// settle reports an utterance that produced nothing, which is one cut off before its
// first chunk.
func (t *TTS) settle(id, said string, started time.Time, durationMs float64, interrupted bool) {
	t.emitter.Send(tts.SynthesisComplete{
		SynthesisID:     id,
		Provider:        ProviderName,
		Model:           t.model,
		Characters:      int64(len(said)),
		AudioDurationMs: durationMs,
		SynthesisTimeMs: float64(time.Since(started).Microseconds()) / 1000,
		Interrupted:     interrupted,
	})
}

// wait sleeps unless the utterance is abandoned first, which it reports.
func (t *TTS) wait(ctx context.Context, how time.Duration) bool {
	timer := time.NewTimer(how)
	defer timer.Stop()
	select {
	case <-timer.C:
		return true
	case <-ctx.Done():
		return false
	}
}

// tone generates one chunk of the carrier, continuing the phase the chunks before it left
// off at so the pieces join without a click.
func tone(sampleRate int, atMs, durationMs float64) audio.PcmData {
	samples := make([]int16, int(durationMs*float64(sampleRate)/1000))
	phase := 2 * math.Pi * toneHz * atMs / 1000
	step := 2 * math.Pi * toneHz / float64(sampleRate)
	for i := range samples {
		samples[i] = int16(math.Sin(phase+float64(i)*step) * toneLevel * math.MaxInt16)
	}
	return audio.PcmData{Samples: samples, SampleRate: sampleRate, Channels: 1}
}
