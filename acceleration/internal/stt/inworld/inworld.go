// Package inworld implements the stt.STT contract on top of Inworld's STT-1 over their
// bidirectional streaming socket.
//
// Inworld's endpoint is a gateway in front of several vendors as well as their own model,
// and which one a session opens on is the modelId. Only the first-party model is
// configured here: the others are already providers of their own in this router, and
// reaching Deepgram through somebody else's gateway would route around the health and the
// cost accounting that make a fallback list mean anything.
//
// The session is configured by the first frame, which has to be wrapped in a
// transcribeConfig key. An unwrapped one is not refused so much as dropped: the socket
// closes with no error frame at all, which is why the wrapper is a type here rather than a
// field set at the call site.
//
// Audio goes up base64-encoded inside a JSON frame, and everything the server says comes
// back wrapped in a result envelope. A transcription restates the turn so far rather than
// carrying only the words that are new, so the interim ones are replacements. Turn
// detection is the server's by default; the turn settles on isFinal.
package inworld

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "inworld"

// DefaultModel is Inworld's own model, which is the one this provider serves.
const DefaultModel = "inworld-stt-1"

// modelVendor qualifies the model on the wire. The gateway fronts several vendors and
// names each model by the one that serves it, while routing here names the provider
// separately, so the prefix is added on the way out rather than carried around and
// written twice in every config that picks this model.
const modelVendor = "inworld/"

// DefaultURL is the bidirectional streaming socket.
const DefaultURL = "wss://api.inworld.ai/stt/v1/transcribe:streamBidirectional"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "INWORLD_API_KEY"

// encodingLinear16 is the only encoding this endpoint streams, and it is the samples
// stt.PcmData already holds.
const encodingLinear16 = "LINEAR16"

// Control frames. They carry nothing, so they go up as the literal they are rather than
// through a struct whose only job would be to marshal to an empty object.
const (
	frameEndTurn     = `{"endTurn":{}}`
	frameCloseStream = `{"closeStream":{}}`
)

// Options configures the provider. APIKey falls back to INWORLD_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong, sent as the custom
	// vocabulary to bias recognition toward. The gateway refuses terms containing
	// characters such as #, / or @, so they are dropped rather than sent.
	Keyterms []string
	// LanguageHints are the languages to expect. The field takes one ISO code, which for
	// this model also settles which script the output is written in, so the first hint is
	// the one that is sent.
	LanguageHints []string
	// EndOfTurnConfidenceThreshold is how sure the model has to be that the caller has
	// finished. Higher is fewer premature turns and a slower boundary. Zero leaves the
	// server's own default.
	EndOfTurnConfidenceThreshold float64
	// MinEndOfTurnSilenceMs is the least silence there has to be before a turn the model
	// is confident about is settled. Zero leaves the server's own default.
	MinEndOfTurnSilenceMs int
	// VadThreshold is how loud speech has to be to count as speech. A pointer because
	// zero is a request rather than a default: it turns the server's turn detection off
	// altogether and leaves the boundaries to endTurn, which nothing here sends mid-call.
	VadThreshold *float64
	// InactivityTimeoutSeconds stops the transcription when the caller has been silent
	// that long. Zero leaves the session open.
	InactivityTimeoutSeconds int
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a frame sent to the server. Each field is its own kind of frame, and
// exactly one of them is set.
type clientMessage struct {
	TranscribeConfig *transcribeConfig `json:"transcribeConfig,omitempty"`
	AudioChunk       *audioChunk       `json:"audioChunk,omitempty"`
}

// transcribeConfig configures the session. It has to be the first frame.
type transcribeConfig struct {
	ModelID          string `json:"modelId"`
	AudioEncoding    string `json:"audioEncoding"`
	SampleRateHertz  int    `json:"sampleRateHertz"`
	NumberOfChannels int    `json:"numberOfChannels"`
	Language         string `json:"language,omitempty"`
	// Prompts is what the rest of the router calls keyterms. It is a soft bias rather
	// than a lock on the output.
	Prompts                      []string            `json:"prompts,omitempty"`
	InactivityTimeoutSeconds     int                 `json:"inactivityTimeoutSeconds,omitempty"`
	EndOfTurnConfidenceThreshold float64             `json:"endOfTurnConfidenceThreshold,omitempty"`
	InworldSttV1Config           *inworldSttV1Config `json:"inworldSttV1Config,omitempty"`
}

// inworldSttV1Config is the turn detection this model has of its own.
type inworldSttV1Config struct {
	MinEndOfTurnSilenceWhenConfident int      `json:"minEndOfTurnSilenceWhenConfident,omitempty"`
	VadThreshold                     *float64 `json:"vadThreshold,omitempty"`
}

// audioChunk carries the audio, base64-encoded.
type audioChunk struct {
	Content string `json:"content"`
}

// serverMessage is a frame sent by the server. Everything it has to say arrives in the
// result envelope, except a failure, which replaces it.
type serverMessage struct {
	Result *serverResult `json:"result"`
	Error  *serverError  `json:"error"`
}

type serverResult struct {
	Transcription *transcription `json:"transcription"`
	Usage         *usage         `json:"usage"`
	SpeechStarted *speechStarted `json:"speechStarted"`
	SpeechStopped *speechStopped `json:"speechStopped"`
}

type transcription struct {
	// Transcript is the turn so far, restated from its beginning on each result.
	Transcript string `json:"transcript"`
	IsFinal    bool   `json:"isFinal"`
	// SilenceDurationMs is the trailing silence this result was settled on.
	SilenceDurationMs int `json:"silenceDurationMs"`
}

// usage is the billing summary, which the server sends as the stream ends. It is the one
// frame that says the session is over, so Close waits for it rather than for the socket.
type usage struct {
	TranscribedAudioMs int    `json:"transcribedAudioMs"`
	ModelID            string `json:"modelId"`
}

type speechStarted struct {
	StartTimeMs int     `json:"startTimeMs"`
	Confidence  float64 `json:"confidence"`
}

type speechStopped struct {
	SilenceDurationMs int `json:"silenceDurationMs"`
}

type serverError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// STT is one Inworld STT-1 session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// finished is closed when the server has answered the end of the audio stream, which
	// is how Close knows the tail has been transcribed.
	finished     chan struct{}
	finishedOnce sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// utterance counts the runs of speech so far, and ended marks that the current one is
	// over so the next transcript starts a new one.
	utterance int64
	ended     bool
	started   bool
	closed    bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("inworld: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("inworld: url must be ws:// or wss://, got %s", options.URL)
	}
	options.Keyterms = keyterms(stt.CleanKeyterms(options.Keyterms))
	if len(options.Keyterms) > stt.MaxKeyterms {
		return nil, fmt.Errorf("inworld: at most %d keyterms, got %d", stt.MaxKeyterms, len(options.Keyterms))
	}
	if options.VadThreshold != nil && (*options.VadThreshold < 0 || *options.VadThreshold > 1) {
		return nil, fmt.Errorf("inworld: vad threshold must be between 0 and 1, got %v", *options.VadThreshold)
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 15 * time.Second
	}
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 10 * time.Second
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STT{
		options:  options,
		logger:   logger.With("provider", ProviderName, "model", options.Model),
		emitter:  stt.NewEmitter(64),
		finished: make(chan struct{}),
	}, nil
}

// Start dials the socket and sends the frame that configures the session. The server does
// not acknowledge it, so a rejected config arrives later as an error frame, or as the
// socket closing under a config it could not read at all.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("inworld: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Basic " + s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.options.URL, header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("inworld: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("inworld: dial: %w", err)
	}
	s.conn = conn

	if err := s.configure(); err != nil {
		conn.Close()
		return err
	}

	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop()
	return nil
}

// ProcessAudio streams one chunk of audio. The participant labels any transcript that
// results from it.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("inworld: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("inworld: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("inworld: not started")
	}

	frame := clientMessage{
		AudioChunk: &audioChunk{Content: base64.StdEncoding.EncodeToString(pcm.Bytes())},
	}
	if err := s.send(frame); err != nil {
		return fmt.Errorf("inworld: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close ends the turn and the audio stream, waits for the server to transcribe what it
// was holding, then tears the connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	heard := !s.lastAudioAt.IsZero()
	s.mu.Unlock()

	if conn != nil {
		if heard {
			s.flush()
		}
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// Client exposes the underlying WebSocket so callers can use the session directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// configure sends the opening frame, wrapped the way this endpoint insists on.
func (s *STT) configure() error {
	held := &transcribeConfig{
		ModelID:                      modelID(s.options.Model),
		AudioEncoding:                encodingLinear16,
		SampleRateHertz:              stt.SampleRate,
		NumberOfChannels:             1,
		Language:                     firstLanguage(s.options.LanguageHints),
		Prompts:                      s.options.Keyterms,
		InactivityTimeoutSeconds:     s.options.InactivityTimeoutSeconds,
		EndOfTurnConfidenceThreshold: s.options.EndOfTurnConfidenceThreshold,
	}
	if s.options.MinEndOfTurnSilenceMs > 0 || s.options.VadThreshold != nil {
		held.InworldSttV1Config = &inworldSttV1Config{
			MinEndOfTurnSilenceWhenConfident: s.options.MinEndOfTurnSilenceMs,
			VadThreshold:                     s.options.VadThreshold,
		}
	}

	if err := s.send(clientMessage{TranscribeConfig: held}); err != nil {
		return fmt.Errorf("inworld: send config: %w", err)
	}
	return nil
}

// flush ends the turn, tells the server the audio has stopped, and waits for it to report
// what it was still holding. A dead connection must not stop teardown.
//
// Both frames are sent because they mean different things: endTurn settles the transcript
// of a caller who was cut off mid-sentence, and closeStream is what this protocol has
// instead of the client closing its half of a gRPC stream.
func (s *STT) flush() {
	if err := s.write([]byte(frameEndTurn)); err != nil {
		s.logger.Debug("end of turn not delivered", "error", err)
		return
	}
	if err := s.write([]byte(frameCloseStream)); err != nil {
		s.logger.Debug("close not delivered", "error", err)
		return
	}

	select {
	case <-s.finished:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the last words")
	}
}

func (s *STT) send(frame clientMessage) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	return s.write(payload)
}

func (s *STT) write(payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.TextMessage, payload)
}

// readLoop translates server frames into events until the connection ends.
func (s *STT) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			s.done()
			s.handleReadError(err)
			return
		}

		var message serverMessage
		if err := json.Unmarshal(raw, &message); err != nil {
			s.logger.Debug("undecodable frame", "error", err, "payload", string(raw))
			continue
		}
		s.handleMessage(message)
	}
}

func (s *STT) handleReadError(err error) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return
	}

	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(stt.Disconnected{
			Provider: ProviderName,
			Model:    s.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		return
	}
	s.emitter.Send(stt.Error{
		Provider: ProviderName,
		Model:    s.options.Model,
		Err:      err,
		Context:  "read",
		Fatal:    true,
	})
}

func (s *STT) handleMessage(message serverMessage) {
	if message.Error != nil {
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.Error.failure()),
			Context:  "server",
			Fatal:    true,
		})
		return
	}
	if message.Result == nil {
		return
	}

	switch {
	case message.Result.Transcription != nil:
		s.sendTranscript(*message.Result.Transcription)
	case message.Result.SpeechStarted != nil:
		s.startTurn()
	case message.Result.SpeechStopped != nil:
		// Silence after speech, which is what the server settles a turn on. The
		// transcript that settles it follows, and that is the one to report.
	case message.Result.Usage != nil:
		// The stream is over, which is what a Close waiting on the tail is waiting for.
		// It arrives before the socket closes, so waiting for this rather than for the
		// hangup is what keeps a teardown from costing the whole flush timeout.
		s.logger.Debug("session usage",
			"transcribed_audio_ms", message.Result.Usage.TranscribedAudioMs)
		s.done()
	}
}

func (s *STT) sendTranscript(heard transcription) {
	text := strings.TrimSpace(heard.Transcript)
	if text == "" {
		return
	}

	mode := stt.ModeReplacement
	if heard.IsFinal {
		mode = stt.ModeFinal
	}

	participant, utterance, latencyMs := s.snapshot(heard.IsFinal)
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Utterance:        utterance,
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// startTurn marks that the run of speech the server has just heard begin is a new one.
//
// The count moves on the next transcript rather than here, so a turn the server announces
// and then transcribes nothing in does not use up a number.
func (s *STT) startTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ended = true
}

// snapshot returns the current speaker, the turn being spoken and how long ago audio was
// last sent. A final settles the turn, so the next transcript belongs to the next one.
func (s *STT) snapshot(final bool) (stt.Participant, int64, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 || s.ended {
		s.utterance++
		s.ended = false
	}
	if final {
		s.ended = true
	}

	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	return s.participant, s.utterance, latencyMs
}

// done reports that the server has nothing further to say about the audio it was given.
func (s *STT) done() {
	s.finishedOnce.Do(func() { close(s.finished) })
}

// failure is what the server said went wrong.
func (e serverError) failure() string {
	if e.Message == "" {
		return fmt.Sprintf("unknown error (code %d)", e.Code)
	}
	return e.Message
}

// keyterms drops the terms the gateway refuses rather than sending them and losing the
// session. It rejects anything outside letters, digits, spaces and basic punctuation with
// an invalid-argument error, which would cost the call the whole turn rather than the one
// word that was spelt with a slash in it.
func keyterms(terms []string) []string {
	if len(terms) == 0 {
		return nil
	}
	kept := make([]string, 0, len(terms))
	for _, term := range terms {
		if !strings.ContainsAny(term, "#/@|") {
			kept = append(kept, term)
		}
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

// modelID is the model as the gateway names it. A model that already names its vendor is
// left alone, so a third-party model reached deliberately is still addressable.
func modelID(model string) string {
	if strings.Contains(model, "/") {
		return model
	}
	return modelVendor + model
}

// firstLanguage picks the hint to send. The field takes one ISO code, and for this model
// it also settles which script the output is written in, so a list of hints has one
// useful answer in it.
func firstLanguage(hints []string) string {
	if len(hints) == 0 {
		return ""
	}
	return hints[0]
}
