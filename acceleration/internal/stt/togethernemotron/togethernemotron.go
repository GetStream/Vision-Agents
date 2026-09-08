// Package togethernemotron implements the stt.STT contract on top of NVIDIA's Nemotron
// ASR streaming models as Together AI serves them, over their realtime WebSocket.
//
// Two models share the weights' architecture and this provider. Nemotron 3 ASR is
// English-only and is what NVIDIA recommend for an English call; Nemotron 3.5 ASR adds
// language-ID prompt conditioning and transcribes 40 language-locales, detecting the
// language rather than being told it. Which one a session opens on is the Model option.
//
// It is a separate provider from togetherparakeet, which is the other NVIDIA speech model
// on the same socket. The wire protocol is shared but the models are not, and a provider
// name is what routing has to name to pick one.
//
// The wire protocol is the OpenAI realtime one rather than anything of Together's own, so
// audio goes up base64-encoded inside a JSON frame. What comes back is a delta while the
// caller is still talking and a completed transcript once they pause. Together document
// that each delta can replace the one before it rather than adding to it, which is why
// they are replacements.
package togethernemotron

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "together-nemotron"

// DefaultModel is Nemotron 3 ASR, which is English-only. NVIDIA recommend it over the
// multilingual model when the call is in English, so it is what a session opens on
// unless asked for the other.
const DefaultModel = "nvidia/nemotron-3-asr-streaming-0.6b"

// MultilingualModel is Nemotron 3.5 ASR, which transcribes 40 language-locales from one
// checkpoint and detects the language rather than being told it.
const MultilingualModel = "nvidia/nemotron-3.5-asr-streaming-0.6b"

// DefaultURL is Together's realtime socket.
const DefaultURL = "wss://api.together.ai/v1/realtime"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "TOGETHER_API_KEY"

// audioFormat is the only wire format this endpoint takes, and it names the rate that
// stt.SampleRate already pins everything to.
const audioFormat = "pcm_s16le_16000"

// Client frame types.
const (
	clientTypeAppend = "input_audio_buffer.append"
	clientTypeCommit = "input_audio_buffer.commit"
)

// Server frame types.
const (
	eventSessionCreated = "session.created"
	eventDelta          = "conversation.item.input_audio_transcription.delta"
	eventCompleted      = "conversation.item.input_audio_transcription.completed"
	eventFailed         = "conversation.item.input_audio_transcription.failed"
	eventError          = "error"
)

// flushGrace is how long Close waits for the tail when nothing is outstanding. The last
// utterance is already settled in that case and no further one is coming, so waiting the
// full timeout would spend it in full on every hangup.
const flushGrace = 1500 * time.Millisecond

// Options configures the provider. APIKey falls back to TOGETHER_API_KEY.
type Options struct {
	APIKey string
	// Model is DefaultModel or MultilingualModel; a dedicated endpoint may name its own.
	Model string
	URL   string
	// HandshakeTimeout bounds the initial connect and the wait for the session to open.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a frame sent to the server.
type clientMessage struct {
	Type  string `json:"type"`
	Audio string `json:"audio,omitempty"`
}

// serverMessage is a frame sent by the server.
type serverMessage struct {
	Type string `json:"type"`
	// Delta is the transcript so far. Each one can restate the utterance rather than
	// carrying only what is new.
	Delta string `json:"delta"`
	// Transcript is the settled utterance.
	Transcript string `json:"transcript"`
	// Message is where Together puts a failure. The realtime protocol it mirrors nests
	// one under error instead, so both are read.
	Message string       `json:"message"`
	Error   *serverError `json:"error"`
}

type serverError struct {
	Message string `json:"message"`
}

// STT is a Together AI realtime transcription session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// settled receives when an utterance is transcribed, so Close can wait for the tail
	// of the last one rather than cutting it off. Buffered and never blocked on, so the
	// utterances nobody is waiting for during the call cost nothing.
	settled chan struct{}

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// hypothesis is the latest delta, kept so Close can tell an utterance the server is
	// still working on from one it has already settled.
	hypothesis string
	// run is the segments the server has already settled for the utterance in progress,
	// joined exactly as it sent them. It is not the whole utterance: the deltas of the
	// segment being decoded now go on the end of it.
	run string
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
		return nil, fmt.Errorf("togethernemotron: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("togethernemotron: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 30 * time.Second
	}
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 10 * time.Second
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

// Start dials the socket and waits for the server to open the session.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("togethernemotron: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{
		"Authorization": []string{"Bearer " + s.options.APIKey},
		"OpenAI-Beta":   []string{"realtime=v1"},
	}

	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("togethernemotron: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("togethernemotron: dial: %w", err)
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
		return fmt.Errorf("togethernemotron: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("togethernemotron: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("togethernemotron: not started")
	}

	frame := clientMessage{
		Type:  clientTypeAppend,
		Audio: base64.StdEncoding.EncodeToString(pcm.Bytes()),
	}
	if err := s.send(frame); err != nil {
		return fmt.Errorf("togethernemotron: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close asks the server to transcribe whatever audio it is still holding, waits for it,
// then tears the connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	conn := s.conn
	heard := !s.lastAudioAt.IsZero()
	// An outstanding delta is the server working on an utterance it has not settled, and
	// an unfinished run is a sentence it settled only part of. Either is a tail worth
	// waiting the full timeout for; a call whose last utterance ended is not.
	patience := s.options.FlushTimeout
	if s.hypothesis == "" && s.run == "" {
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

// Client exposes the underlying WebSocket so callers can use the session directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// endpoint is the socket with the session's configuration attached, which is the only
// place this API takes it.
func (s *STT) endpoint() string {
	query := url.Values{}
	// Transcription rather than a conversation: this endpoint serves both, and without
	// saying so the session opens expecting to answer back.
	query.Set("intent", "transcription")
	query.Set("model", s.options.Model)
	query.Set("input_audio_format", audioFormat)

	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + query.Encode()
}

// handshake waits for the server to open the session. Audio sent before that is audio the
// server is not yet listening to.
func (s *STT) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("togethernemotron: read handshake: %w", err)
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("togethernemotron: read handshake: %w", err)
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return fmt.Errorf("togethernemotron: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("togethernemotron: decode handshake: %w", err)
	}
	if message.Type == eventError {
		return fmt.Errorf("togethernemotron: handshake rejected: %s", message.failure())
	}
	if message.Type != eventSessionCreated {
		return fmt.Errorf("togethernemotron: expected %q, got %q", eventSessionCreated, message.Type)
	}
	return nil
}

// flush asks the server to transcribe the audio it has buffered and waits for it to come
// back, so the tail of a call is not lost. A dead connection must not stop teardown.
func (s *STT) flush(patience time.Duration) {
	// Forget the utterances that are already settled, so the wait below is for one that
	// comes after the audio stopped.
	select {
	case <-s.settled:
	default:
	}

	if err := s.send(clientMessage{Type: clientTypeCommit}); err != nil {
		s.logger.Debug("commit not delivered", "error", err)
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
	switch message.Type {
	case eventDelta:
		s.heard(message.Delta)
	case eventCompleted:
		s.settle(message.Transcript)
	case eventFailed:
		// One utterance the server could not transcribe. The session carries on, so this
		// is not fatal, but the words are gone and the caller is owed an answer anyway.
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  "transcription",
		})
		s.reachedABoundary()
	case eventError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  "server",
			Fatal:    true,
		})
	case eventSessionCreated:
		// The handshake already waited for this one, and the server sends it once.
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

// heard reports what the caller seems to be saying, which is worth showing at once: it
// arrives while they are still talking, well before the utterance is settled.
//
// Each delta restates the segment being decoded, not the whole utterance, and it starts
// again from nothing after every completed. So the words the caller can be shown are the
// segments already settled with this delta on the end. Sending the delta on its own is
// how "can you hear me" reaches the caller as "me".
//
// The join is deliberately not trimmed on either side: the server's own spacing is what
// makes "…for for" and " this." into one sentence, and what keeps "pat" and "io." from
// becoming two words.
func (s *STT) heard(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}

	s.mu.Lock()
	s.hypothesis = text
	whole := strings.TrimSpace(s.run + text)
	s.mu.Unlock()

	s.sendTranscript(stt.ModeReplacement, whole)
}

// settle folds a segment the server has finished decoding into the utterance in progress.
//
// A completed frame is the decoder flushing its buffer, not the caller reaching the end of
// what they were saying: it arrives mid-sentence and even mid-word, splitting "patio" into
// "pat" and "io.". So the transcript reported is the whole utterance up to here rather than
// the fragment that just settled, and the utterance only ends when the words read as a
// finished sentence. Ending it on every flush is what turned one question into two, the
// second of which was the last word of the first.
func (s *STT) settle(text string) {
	s.mu.Lock()
	s.run += text
	s.hypothesis = ""
	whole := strings.TrimSpace(s.run)
	ended := endsUtterance(whole)
	if ended {
		s.run = ""
	}
	s.mu.Unlock()

	if whole == "" {
		// Nothing was said in the audio that was committed, but a Close waiting on it has
		// its answer all the same.
		s.reachedABoundary()
		return
	}

	s.sendTranscript(stt.ModeFinal, whole)
	if ended {
		s.endUtterance()
	}
	s.reachedABoundary()
}

// endsUtterance reports whether the words the server has settled read as the end of what
// somebody was saying.
//
// It is the only boundary this protocol offers. The server has no voice activity detection
// and never says a turn is over, so its own sentence punctuation is what distinguishes a
// buffer flush landing mid-sentence from the caller having finished one. A flush that ends
// no sentence leaves the utterance open, so the segment after it is joined on rather than
// published as a turn of its own.
func endsUtterance(text string) bool {
	return strings.HasSuffix(text, ".") ||
		strings.HasSuffix(text, "?") ||
		strings.HasSuffix(text, "!")
}

func (s *STT) sendTranscript(mode stt.Mode, text string) {
	participant, latencyMs := s.snapshot()

	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Utterance:        s.utteranceID(),
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// reachedABoundary tells a waiting Close that the server has finished an utterance.
func (s *STT) reachedABoundary() {
	select {
	case s.settled <- struct{}{}:
	default:
	}
}

// endUtterance marks the current run of speech as over.
//
// The count moves on the next transcript rather than here, so a final and the delta that
// opens the next turn are one boundary between utterances rather than two.
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

// failure is what the server said went wrong, from whichever of the two places it put it.
func (m serverMessage) failure() string {
	if m.Message != "" {
		return m.Message
	}
	if m.Error != nil && m.Error.Message != "" {
		return m.Error.Message
	}
	return "unknown error"
}
