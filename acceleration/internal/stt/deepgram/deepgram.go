// Package deepgram implements the stt.STT contract on top of Deepgram Flux.
//
// Flux runs turn detection server-side and reports it through TurnInfo events, so
// callers do not need a separate turn detector. Interim transcripts arrive as full
// replacements rather than deltas, which is why Update maps to stt.ModeReplacement.
package deepgram

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"sync"
	"time"

	msginterfaces "github.com/deepgram/deepgram-go-sdk/v3/pkg/api/listen/v2/websocket/interfaces"
	clientinterfaces "github.com/deepgram/deepgram-go-sdk/v3/pkg/client/interfaces"
	interfacesv2 "github.com/deepgram/deepgram-go-sdk/v3/pkg/client/interfaces/v2"
	listenv2 "github.com/deepgram/deepgram-go-sdk/v3/pkg/client/listen/v2"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "deepgram"

// DefaultModel is the English Flux model.
const DefaultModel = "flux-general-en"

// MultilingualModel is the Flux model that accepts language hints.
const MultilingualModel = "flux-general-multi"

// defaultEagerEotThreshold matches the Python plugin: eager turn detection is
// meaningless without a threshold, so supply one when the caller only asks for eagerness.
const defaultEagerEotThreshold = 0.5

// Options configures the provider. Only APIKey is required, and it falls back to
// DEEPGRAM_API_KEY.
type Options struct {
	APIKey string
	Model  string
	// LanguageHints is only valid with MultilingualModel.
	LanguageHints []string
	// EagerTurnDetection asks Flux for early end-of-turn signals.
	EagerTurnDetection bool
	EagerEotThreshold  float64
	EotThreshold       float64
	EotTimeoutMs       int
	Keyterms           []string
	Logger             *slog.Logger
}

// STT is a Deepgram Flux speech-to-text session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	client *listenv2.WSCallback

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last handed to Deepgram, so latency can be
	// reported as the delay between sending audio and hearing about it.
	lastAudioAt time.Time
	started     bool
	closed      bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv("DEEPGRAM_API_KEY")
	}
	if options.APIKey == "" {
		return nil, errors.New("deepgram: api key is required (set DEEPGRAM_API_KEY)")
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if len(options.LanguageHints) > 0 && options.Model != MultilingualModel {
		return nil, fmt.Errorf("deepgram: language hints require model %s, got %s", MultilingualModel, options.Model)
	}
	if options.EagerTurnDetection && options.EagerEotThreshold == 0 {
		options.EagerEotThreshold = defaultEagerEotThreshold
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STT{
		options: options,
		logger:  logger.With("provider", ProviderName, "model", options.Model),
		emitter: stt.NewEmitter(64),
	}, nil
}

// Start opens the Flux WebSocket. It returns once the connection is established.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("deepgram: already started")
	}
	s.started = true
	s.mu.Unlock()

	transcription := &interfacesv2.FluxTranscriptionOptions{
		Model:             s.options.Model,
		Encoding:          "linear16",
		SampleRate:        stt.SampleRate,
		EotThreshold:      s.options.EotThreshold,
		EagerEotThreshold: s.options.EagerEotThreshold,
		EotTimeoutMs:      s.options.EotTimeoutMs,
		Keyterm:           s.options.Keyterms,
		LanguageHint:      s.options.LanguageHints,
	}

	client, err := listenv2.NewWSUsingCallback(
		ctx,
		s.options.APIKey,
		&clientinterfaces.ClientOptionsV2{},
		transcription,
		&callbacks{stt: s},
	)
	if err != nil {
		return fmt.Errorf("deepgram: create client: %w", err)
	}
	if !client.Connect() {
		return errors.New("deepgram: connect failed")
	}

	s.client = client
	return nil
}

// ProcessAudio streams one chunk of audio. The participant labels any transcript that
// results from it.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("deepgram: %w", err)
	}

	s.mu.Lock()
	client, closed := s.client, s.closed
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("deepgram: session closed")
	}
	if client == nil {
		return errors.New("deepgram: not started")
	}

	if _, err := client.Write(pcm.Bytes()); err != nil {
		return fmt.Errorf("deepgram: write audio: %w", err)
	}
	return nil
}

// Events returns transcripts and turn boundaries.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close tears down the session and closes the event channel.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	client := s.client
	s.mu.Unlock()

	if client != nil {
		client.Stop()
	}
	s.emitter.Close()
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// TurnDetection reports true: Flux detects turns itself.
func (s *STT) TurnDetection() bool { return true }

// Client exposes the underlying Deepgram client so callers can use Flux features this
// abstraction deliberately does not cover, such as mid-session Configure.
func (s *STT) Client() *listenv2.WSCallback { return s.client }

// isClosing reports whether Close has started, so teardown noise can be ignored.
func (s *STT) isClosing() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.closed
}

// snapshot returns the current speaker and how long ago audio was last sent.
func (s *STT) snapshot() (stt.Participant, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	return s.participant, latencyMs
}

func (s *STT) handleTurnInfo(turn *msginterfaces.TurnInfoResponse) {
	participant, latencyMs := s.snapshot()
	text := strings.TrimSpace(turn.Transcript)

	switch turn.EventType {
	case msginterfaces.TurnEventStartOfTurn:
		s.emitter.Send(stt.TurnStarted{Participant: participant, Confidence: turn.EndOfTurnConfidence})
	case msginterfaces.TurnEventTurnResumed:
		// Speech continued after an eager end-of-turn, so the turn is live again.
		s.emitter.Send(stt.TurnStarted{Participant: participant, Confidence: turn.EndOfTurnConfidence})
	case msginterfaces.TurnEventUpdate:
		s.sendTranscript(participant, turn, text, stt.ModeReplacement, latencyMs)
	case msginterfaces.TurnEventEagerEndOfTurn:
		s.sendTranscript(participant, turn, text, stt.ModeReplacement, latencyMs)
		s.sendTurnEnded(participant, turn, true)
	case msginterfaces.TurnEventEndOfTurn:
		s.sendTranscript(participant, turn, text, stt.ModeFinal, latencyMs)
		s.sendTurnEnded(participant, turn, false)
	default:
		s.logger.Debug("unhandled turn event", "event", turn.EventType)
	}
}

func (s *STT) sendTranscript(
	participant stt.Participant,
	turn *msginterfaces.TurnInfoResponse,
	text string,
	mode stt.Mode,
	latencyMs float64,
) {
	if text == "" {
		return
	}

	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Text:             text,
		Confidence:       turn.EndOfTurnConfidence,
		Language:         firstLanguage(turn.Languages),
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
		AudioDurationMs:  audioWindowMs(turn),
	})
}

func (s *STT) sendTurnEnded(participant stt.Participant, turn *msginterfaces.TurnInfoResponse, eager bool) {
	s.emitter.Send(stt.TurnEnded{
		Participant: participant,
		Confidence:  turn.EndOfTurnConfidence,
		Eager:       eager,
		DurationMs:  audioWindowMs(turn),
	})
}

// audioWindowMs converts the turn's audio window, reported in seconds, to milliseconds.
func audioWindowMs(turn *msginterfaces.TurnInfoResponse) float64 {
	window := turn.AudioWindowEnd - turn.AudioWindowStart
	if window <= 0 {
		return 0
	}
	return window * 1000
}

func firstLanguage(languages []string) string {
	if len(languages) == 0 {
		return ""
	}
	return languages[0]
}

// callbacks adapts the SDK's FluxMessageCallback to STT. It is a separate type because
// the SDK requires a Close(*CloseResponse) method, which would collide with STT.Close.
type callbacks struct {
	stt *STT
}

func (c *callbacks) Open(*msginterfaces.OpenResponse) error { return nil }

func (c *callbacks) Connected(response *msginterfaces.ConnectedResponse) error {
	c.stt.logger.Debug("flux session ready", "request_id", response.RequestID)
	c.stt.emitter.Send(stt.Connected{
		Provider: ProviderName,
		Model:    c.stt.options.Model,
		At:       time.Now(),
	})
	return nil
}

func (c *callbacks) TurnInfo(response *msginterfaces.TurnInfoResponse) error {
	c.stt.handleTurnInfo(response)
	return nil
}

func (c *callbacks) ConfigureSuccess(*msginterfaces.ConfigureSuccessResponse) error { return nil }

func (c *callbacks) ConfigureFailure(*msginterfaces.ConfigureFailureResponse) error {
	c.stt.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    c.stt.options.Model,
		Err:      errors.New("configure rejected"),
		Context:  "configure",
	})
	return nil
}

func (c *callbacks) FatalError(response *msginterfaces.FatalErrorResponse) error {
	c.stt.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    c.stt.options.Model,
		Err:      fmt.Errorf("%s: %s", response.Code, response.Description),
		Context:  "fatal",
		Fatal:    true,
	})
	return nil
}

func (c *callbacks) Close(*msginterfaces.CloseResponse) error {
	c.stt.emitter.Send(stt.Disconnected{
		Provider: ProviderName,
		Model:    c.stt.options.Model,
		Clean:    true,
		At:       time.Now(),
	})
	return nil
}

func (c *callbacks) Error(response *msginterfaces.ErrorResponse) error {
	// Stopping the client makes its read loop report the socket we just closed. We asked
	// for that, so it is not a failure to bill or to hold against the provider's health.
	if c.stt.isClosing() {
		c.stt.logger.Debug("error during teardown", "code", response.ErrCode, "message", response.ErrMsg)
		return nil
	}

	c.stt.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    c.stt.options.Model,
		Err:      fmt.Errorf("%s: %s", response.ErrCode, response.ErrMsg),
		Context:  "transport",
	})
	return nil
}

func (c *callbacks) UnhandledEvent(data []byte) error {
	c.stt.logger.Debug("unhandled flux message", "payload", string(data))
	return nil
}
