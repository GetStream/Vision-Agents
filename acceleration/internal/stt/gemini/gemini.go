// Package gemini implements the stt.STT contract on top of Gemini 3.5 Transcribe, over
// the Live API's BidiGenerateContent WebSocket.
//
// The Live API is a conversational protocol that happens to expose what it heard. Only
// the listening half is used here: the session is set up with input transcription on, and
// whatever the model would have said back is dropped.
//
// What it heard arrives twice. While the caller is still talking the server sends an
// interim hypothesis roughly every half second, each one restating the turn from its
// beginning rather than adding to the last, so a hypothesis replaces the one before it.
// When the turn finishes it sends the finalized transcript of the whole turn, which is
// what settles it and what the rest of the call is built on.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// DefaultModel is the streaming half of Gemini 3.5 Transcribe. The other half,
// gemini-3.5-transcribe, is a file API and cannot serve a call.
const DefaultModel = "gemini-3.5-transcribe-live"

// DefaultURL is the Live API socket. The key goes on the query string, which is the only
// authentication this endpoint takes.
const DefaultURL = "wss://generativelanguage.googleapis.com/ws/" +
	"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

// audioMimeType describes what ProcessAudio sends. The rate is part of the type rather
// than a field of its own, and a mismatch is transcribed at the wrong speed rather than
// rejected, so it is derived from stt.SampleRate.
var audioMimeType = fmt.Sprintf("audio/pcm;rate=%d", stt.SampleRate)

// apiKeyEnvVar holds the credentials when Options does not. It is the name the Gemini LLM
// provider and the Python side of this repository already use for the same key.
const apiKeyEnvVar = "GOOGLE_API_KEY"

// flushGrace is how long Close waits for the tail when the server has no utterance in
// flight. It only has to cover the caller who spoke in the moment before hanging up, too
// recently for a hypothesis to have arrived: the model finalizes a turn about 1.4s after
// the last word, and a hypothesis of it lands within 700ms of the first.
const flushGrace = 1500 * time.Millisecond

// TranscriptionMode is how faithfully the transcript follows what was said.
type TranscriptionMode string

const (
	// ModeVerbatim writes down every word, the filler and the false starts included. It
	// is what the server does when it is not asked for anything else.
	ModeVerbatim TranscriptionMode = "VERBATIM"
	// ModeSmart drops the filler, resolves a spoken self-correction to whichever answer
	// the caller settled on, and punctuates. It reads better at the cost of no longer
	// being word for word, which is the wrong trade when the words are the record.
	ModeSmart TranscriptionMode = "SMART"
)

// Options configures the provider. APIKey falls back to GOOGLE_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong, sent as the
	// transcriber's custom vocabulary. The API takes up to a thousand of them and
	// recommends no more than a hundred, which is what stt.MaxKeyterms allows.
	Keyterms []string
	// LanguageHints narrow what is expected, as BCP-47 codes such as "en-US" or "es-ES".
	// Empty leaves the model to detect the language itself, including a caller who
	// switches between two of them mid-sentence.
	LanguageHints []string
	// Mode is how much tidying up the transcript gets. Empty leaves it to the server,
	// which is verbatim.
	Mode TranscriptionMode
	// HandshakeTimeout bounds the initial connect and the setup exchange.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the tail of the last utterance.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a frame sent to the Live API. Exactly one field is ever set.
type clientMessage struct {
	Setup         *setup         `json:"setup,omitempty"`
	RealtimeInput *realtimeInput `json:"realtimeInput,omitempty"`
}

// setup is the first frame, which configures the session.
type setup struct {
	Model                   string              `json:"model"`
	GenerationConfig        generationConfig    `json:"generationConfig"`
	InputAudioTranscription *audioTranscription `json:"inputAudioTranscription,omitempty"`
}

type generationConfig struct {
	ResponseModalities []string `json:"responseModalities"`
}

// audioTranscription configures the transcriber: the vocabulary to lean on, the languages
// to expect and how much of a tidy-up to apply. Sending it at all is what asks for
// transcription, so an empty one still has a job to do.
type audioTranscription struct {
	LanguageCodes    []string `json:"languageCodes,omitempty"`
	CustomVocabulary []string `json:"customVocabulary,omitempty"`
	Mode             string   `json:"mode,omitempty"`
}

// realtimeInput carries audio, or the note that there will be no more of it.
type realtimeInput struct {
	Audio *blob `json:"audio,omitempty"`
	// AudioStreamEnd tells the server the microphone is off, which makes it transcribe
	// what it is still holding rather than waiting for more.
	AudioStreamEnd bool `json:"audioStreamEnd,omitempty"`
}

type blob struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

// serverMessage is a frame sent by the Live API.
type serverMessage struct {
	SetupComplete *json.RawMessage `json:"setupComplete"`
	ServerContent *serverContent   `json:"serverContent"`
	GoAway        *goAway          `json:"goAway"`
}

type serverContent struct {
	// InterimInputTranscription is what the caller seems to be saying, restated in full
	// each time and superseded by the next one.
	InterimInputTranscription *transcription `json:"interimInputTranscription"`
	// InputTranscription is the finalized transcript of a turn, sent once the server has
	// committed to it and immediately before it reports the turn over.
	InputTranscription *transcription `json:"inputTranscription"`
	TurnComplete       bool           `json:"turnComplete"`
	GenerationComplete bool           `json:"generationComplete"`
	Interrupted        bool           `json:"interrupted"`
}

type transcription struct {
	Text string `json:"text"`
}

type goAway struct {
	TimeLeft string `json:"timeLeft"`
}

// STT is a Gemini 3.5 Transcribe session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// settled receives when a turn boundary arrives, so Close can wait for the tail of
	// the last one rather than cutting it off. Buffered and never blocked on, so the
	// boundaries nobody is waiting for during the call cost nothing.
	settled chan struct{}

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// hypothesis is the latest interim text, kept so a turn cut off before the server
	// finalized it still settles on the words that were heard.
	hypothesis string
	// utterance counts the runs of speech seen so far, and ended marks that the current
	// one is over so the next transcript starts a new one.
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
		return nil, fmt.Errorf("gemini: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("gemini: url must be ws:// or wss://, got %s", options.URL)
	}
	if len(options.Keyterms) > stt.MaxKeyterms {
		return nil, fmt.Errorf("gemini: at most %d keyterms, got %d", stt.MaxKeyterms, len(options.Keyterms))
	}
	switch options.Mode {
	case "", ModeVerbatim, ModeSmart:
	default:
		return nil, fmt.Errorf("gemini: mode must be %s or %s, got %s",
			ModeVerbatim, ModeSmart, options.Mode)
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 30 * time.Second
	}
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 5 * time.Second
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STT{
		options: options,
		logger:  logger.With("provider", ProviderName, "model", options.Model),
		emitter: stt.NewEmitter(64),
		settled: make(chan struct{}, 1),
	}, nil
}

// Start dials the Live API and completes the setup exchange. It returns once the server
// reports it is ready for audio.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("gemini: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	conn, response, err := dialer.DialContext(ctx, s.endpoint(), nil)
	if err != nil {
		if response != nil {
			return fmt.Errorf("gemini: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("gemini: dial: %w", err)
	}
	s.conn = conn

	if err := s.handshake(); err != nil {
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
		return fmt.Errorf("gemini: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("gemini: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("gemini: not started")
	}

	frame := clientMessage{RealtimeInput: &realtimeInput{Audio: &blob{
		Data:     base64.StdEncoding.EncodeToString(pcm.Bytes()),
		MimeType: audioMimeType,
	}}}
	if err := s.send(frame); err != nil {
		return fmt.Errorf("gemini: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close ends the audio stream, waits for what is still being transcribed, then tears the
// connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	conn := s.conn
	heard := !s.lastAudioAt.IsZero()
	// An outstanding hypothesis is the server holding an utterance it has not finalized,
	// which is the tail worth waiting the full timeout for. With nothing outstanding the
	// last turn is already settled and no further boundary is coming, so waiting that
	// long would spend it in full on every hangup.
	patience := s.options.FlushTimeout
	if s.hypothesis == "" {
		patience = min(flushGrace, s.options.FlushTimeout)
	}
	s.mu.Unlock()

	if conn != nil && heard {
		s.flush(patience)
	}

	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()

	if conn != nil {
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// Client exposes the underlying WebSocket so callers can use the Live API directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// endpoint is the socket with the key attached, which is how this API authenticates.
func (s *STT) endpoint() string {
	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + "key=" + url.QueryEscape(s.options.APIKey)
}

// handshake configures the session and waits for the server to accept it.
func (s *STT) handshake() error {
	frame := clientMessage{Setup: &setup{
		Model: "models/" + s.options.Model,
		// Text, because a spoken reply is neither wanted nor free. What the model would
		// have said is dropped either way; asking for audio would only bill for it.
		GenerationConfig:        generationConfig{ResponseModalities: []string{"TEXT"}},
		InputAudioTranscription: s.transcription(),
	}}
	if err := s.send(frame); err != nil {
		return fmt.Errorf("gemini: send setup: %w", err)
	}

	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("gemini: read setup: %w", err)
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("gemini: read setup: %w", err)
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return fmt.Errorf("gemini: read setup: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("gemini: decode setup: %w", err)
	}
	if message.SetupComplete == nil {
		return fmt.Errorf("gemini: setup rejected: %s", strings.TrimSpace(string(raw)))
	}
	return nil
}

// transcription is how the session is asked to listen. It is never nil: an empty one is
// what turns transcription on, and every field in it is a bias the caller may not want.
func (s *STT) transcription() *audioTranscription {
	return &audioTranscription{
		LanguageCodes:    s.options.LanguageHints,
		CustomVocabulary: stt.CleanKeyterms(s.options.Keyterms),
		Mode:             string(s.options.Mode),
	}
}

// flush tells the server the audio has stopped and waits for it to settle the turn, so
// the tail of a call is not lost. A dead connection must not stop teardown.
func (s *STT) flush(patience time.Duration) {
	// Forget the boundaries of turns that are already over, so the wait below is for one
	// that comes after the audio stopped.
	select {
	case <-s.settled:
	default:
	}

	if err := s.send(clientMessage{RealtimeInput: &realtimeInput{AudioStreamEnd: true}}); err != nil {
		s.logger.Debug("audio stream end not delivered", "error", err)
		return
	}

	select {
	case <-s.settled:
	case <-time.After(patience):
		s.logger.Debug("timed out waiting for the last words", "patience", patience)
	}
}

func (s *STT) send(frame clientMessage) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return err
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.TextMessage, payload)
}

// readLoop translates server frames into events until the connection ends.
func (s *STT) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
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
	// A go-away is the server saying the session is nearly out of time. It is worth
	// knowing about in a log, but the router reacts to the disconnection that follows,
	// not to the warning.
	if message.GoAway != nil {
		s.logger.Info("the session will be cut off soon", "time_left", message.GoAway.TimeLeft)
		return
	}
	if message.ServerContent == nil {
		return
	}
	server := *message.ServerContent

	if server.InterimInputTranscription != nil {
		s.heard(server.InterimInputTranscription.Text)
	}
	if server.InputTranscription != nil {
		s.settle(server.InputTranscription.Text)
	}
	if server.TurnComplete || server.GenerationComplete || server.Interrupted {
		s.reachedTheEndOfATurn()
	}
}

// heard reports what the caller seems to be saying, which is worth showing at once: it
// arrives while they are still talking, seconds before the turn is finalized.
//
// Each hypothesis restates the turn so far, so it replaces its predecessor rather than
// adding to it. Appending them would spell the sentence out several times over.
func (s *STT) heard(text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	s.mu.Lock()
	s.hypothesis = text
	s.mu.Unlock()

	participant, latencyMs := s.snapshot()
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeReplacement,
		Utterance:        s.utteranceID(),
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// settle reports the turn the server has committed to, which supersedes the hypotheses
// before it. In smart mode this is also where the tidied-up wording arrives, so it is not
// necessarily the last hypothesis with a full stop on the end.
func (s *STT) settle(text string) {
	text = strings.TrimSpace(text)
	if text == "" {
		return
	}

	s.mu.Lock()
	s.hypothesis = ""
	s.mu.Unlock()

	participant, latencyMs := s.snapshot()
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             stt.ModeFinal,
		Utterance:        s.utteranceID(),
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})

	s.endUtterance()
	s.reachedABoundary()
}

// reachedTheEndOfATurn settles whatever the server left outstanding.
//
// It normally finalizes a turn before saying it is over, so there is nothing here to do
// but let a waiting Close know. A turn cut short is the exception, and words heard before
// the cut are still words that were said.
func (s *STT) reachedTheEndOfATurn() {
	s.mu.Lock()
	outstanding := s.hypothesis
	s.mu.Unlock()

	if outstanding != "" {
		s.settle(outstanding)
		return
	}
	s.reachedABoundary()
}

// reachedABoundary tells a waiting Close that the server has finished a turn.
func (s *STT) reachedABoundary() {
	select {
	case s.settled <- struct{}{}:
	default:
	}
}

// endUtterance marks the current run of speech as over.
//
// The count moves on the next transcript rather than here, so a final and the delta that
// starts the next turn are one boundary between utterances rather than two.
func (s *STT) endUtterance() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ended = true
}

// utteranceID numbers the run of speech the next transcript belongs to.
func (s *STT) utteranceID() int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 || s.ended {
		s.utterance++
		s.ended = false
	}
	return s.utterance
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
