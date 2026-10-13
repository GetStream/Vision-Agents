// Package mock implements the stt.STT contract without a vendor behind it, for load
// tests that want a cascaded conversation without paying for one.
//
// It hears rather than recognises: the energy of every frame is measured, a run of loud
// frames is an utterance and the quiet after it ends the turn, which is the same shape a
// real transcriber's events arrive in. What comes back is a fixed sentence, so nothing
// about the words is a measurement — the point is the audio path, the turn boundaries and
// the work per frame, not the transcript.
package mock

import (
	"context"
	"errors"
	"log/slog"
	"math"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "mock"

// DefaultModel is what the catalogue declares.
const DefaultModel = "mock-stt"

// DefaultTranscript is what every settled turn says.
const DefaultTranscript = "what are your opening hours on a saturday"

const (
	// defaultEndOfTurnMs is the quiet after speech that settles a turn. It is near what
	// Flux's own endpointing uses, so a mocked conversation has a real one's rhythm.
	defaultEndOfTurnMs = 600
	// defaultPartialMs is how much speech goes by between in-progress transcripts.
	defaultPartialMs = 400
	// defaultSpeechLevel is the RMS, as a fraction of full scale, above which a frame is
	// speech. Comfort noise and an empty room sit well below it.
	defaultSpeechLevel = 0.02
	// minSpeechMs is the shortest run of loud frames that counts as a turn, so a cough or
	// a door does not produce a transcript.
	minSpeechMs = 300
	// eventBuffer is the consumer's slack, matching what the real providers allow.
	eventBuffer = 64
)

// Options configures the provider.
type Options struct {
	Model string
	// Transcript is what a settled turn says. Empty uses DefaultTranscript.
	Transcript string
	// EndOfTurnMs is the quiet that ends a turn, and PartialMs how often an in-progress
	// transcript is emitted while the caller is still speaking.
	EndOfTurnMs int
	PartialMs   int
	// SpeechLevel is the RMS, from 0 to 1, above which a frame counts as speech.
	SpeechLevel float64
	Logger      *slog.Logger
}

// STT is a transcriber that reports turns without recognising any words.
type STT struct {
	model       string
	transcript  string
	endOfTurnMs int
	partialMs   int
	speechLevel float64
	logger      *slog.Logger

	emitter *stt.Emitter

	// started guards against audio arriving before the session is open, which is a
	// caller mistake rather than something to recover from.
	started bool
	// speakingMs and silenceMs are the run the frames so far amount to, and utterance
	// counts the turns this session has settled.
	speakingMs float64
	silenceMs  float64
	saidMs     float64
	utterance  int64
	speaker    stt.Participant
}

// New returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.SpeechLevel < 0 || options.SpeechLevel > 1 {
		return nil, errors.New("mock: speech level is a fraction of full scale, from 0 to 1")
	}

	provider := &STT{
		model:       options.Model,
		transcript:  options.Transcript,
		endOfTurnMs: options.EndOfTurnMs,
		partialMs:   options.PartialMs,
		speechLevel: options.SpeechLevel,
		logger:      options.Logger,
		emitter:     stt.NewEmitter(eventBuffer),
	}
	if provider.model == "" {
		provider.model = DefaultModel
	}
	if provider.transcript == "" {
		provider.transcript = DefaultTranscript
	}
	if provider.endOfTurnMs == 0 {
		provider.endOfTurnMs = defaultEndOfTurnMs
	}
	if provider.partialMs == 0 {
		provider.partialMs = defaultPartialMs
	}
	if provider.speechLevel == 0 {
		provider.speechLevel = defaultSpeechLevel
	}
	if provider.logger == nil {
		provider.logger = slog.Default()
	}
	return provider, nil
}

// Start opens the session. There is nothing upstream, so it only says so.
func (s *STT) Start(context.Context) error {
	s.started = true
	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.model, At: time.Now()})
	return nil
}

// ProcessAudio measures one frame and reports a turn when the speech in it ends.
//
// It is called from the session's own goroutine, one frame at a time, which is what lets
// the run of speech be plain fields rather than guarded state.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if !s.started {
		return errors.New("mock: the session is not open")
	}
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return err
	}

	s.speaker = participant
	durationMs := pcm.DurationMs()

	if level(pcm) < s.speechLevel {
		if s.speakingMs == 0 {
			return nil
		}
		s.silenceMs += durationMs
		if s.silenceMs >= float64(s.endOfTurnMs) {
			s.settle()
		}
		return nil
	}

	s.speakingMs += durationMs + s.silenceMs
	s.silenceMs = 0
	if s.speakingMs-s.saidMs >= float64(s.partialMs) {
		s.saidMs = s.speakingMs
		s.say(stt.ModeReplacement, s.speakingMs)
	}
	return nil
}

// Events carries the transcripts back.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close ends the session.
func (s *STT) Close() error {
	s.emitter.Send(stt.Disconnected{Provider: ProviderName, Model: s.model, Clean: true, At: time.Now()})
	s.emitter.Close()
	return nil
}

// Provider is the stable provider name used in stats.
func (s *STT) Provider() string { return ProviderName }

// Model is the model identifier used in stats.
func (s *STT) Model() string { return s.model }

// settle ends the turn the frames so far amount to. A run too short to be speech is
// dropped rather than transcribed, so the quiet between turns stays quiet.
func (s *STT) settle() {
	spoken := s.speakingMs
	s.speakingMs, s.silenceMs, s.saidMs = 0, 0, 0
	if spoken < minSpeechMs {
		return
	}
	s.utterance++
	s.say(stt.ModeFinal, spoken)
}

// say emits a transcript for the utterance in progress. An in-progress one belongs to the
// turn about to settle, which is the one after the last settled.
func (s *STT) say(mode stt.Mode, spokenMs float64) {
	utterance := s.utterance
	text := s.transcript
	if mode != stt.ModeFinal {
		utterance++
		text = s.heard(spokenMs)
	}
	s.emitter.Send(stt.Transcript{
		Participant:     s.speaker,
		Mode:            mode,
		Utterance:       utterance,
		Text:            text,
		Confidence:      1,
		Provider:        ProviderName,
		Model:           s.model,
		AudioDurationMs: spokenMs,
	})
}

// heard is the sentence as far as the caller has got: a word for every partialMs of
// speech, and the whole of it once they have talked that long.
//
// What matters is not the pace but that the words keep changing while somebody is
// talking. The router waits for a transcript to hold still before it decides the turn is
// over, so a mock whose every partial was the finished sentence would have it answer
// while the caller was still mid-breath, and the wait it measures would be nobody's.
func (s *STT) heard(spokenMs float64) string {
	words := strings.Fields(s.transcript)
	said := int(spokenMs) / s.partialMs
	if said >= len(words) {
		return s.transcript
	}
	return strings.Join(words[:said], " ")
}

// level is the root mean square of a frame, as a fraction of full scale. It is the work
// this provider does per frame, and it is deliberately the same arithmetic a real client
// does before it writes the frame to its socket.
func level(pcm stt.PcmData) float64 {
	if len(pcm.Samples) == 0 {
		return 0
	}
	var sum float64
	for _, sample := range pcm.Samples {
		value := float64(sample)
		sum += value * value
	}
	return math.Sqrt(sum/float64(len(pcm.Samples))) / math.MaxInt16
}
