// Package assemblyai implements the stt.STT contract on top of AssemblyAI's v3 streaming
// socket and Universal-3.6 Pro Realtime.
//
// The session is configured by the query string and opened by a Begin frame, which Start
// waits for. Begin also echoes the configuration the server applied, and that echo is
// checked rather than trusted: AssemblyAI ignore a query parameter they do not recognise
// instead of refusing it, so a misspelt model would otherwise transcribe on whatever the
// default happens to be.
//
// Audio goes up as raw PCM in binary frames, each of which has to hold between 50 ms and a
// second of it or the server ends the session. A call delivers 20 ms at a time, so audio
// is gathered here until there is enough to send.
//
// Every Turn frame restates the turn so far rather than appending to it, and the turn is
// settled by the one carrying end_of_turn. Turns are numbered by the server. Terminate
// settles whatever is still being said and is answered by a Termination frame, after
// which nothing else is sent, so that is what Close waits for.
package assemblyai

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "assemblyai"

// DefaultModel is Universal-3.6 Pro Realtime, the streaming model built for voice agents.
const DefaultModel = "universal-3-6-pro"

// DefaultURL is the global streaming socket, which routes each connection to the nearest
// region.
const DefaultURL = "wss://streaming.assemblyai.com/v3/ws"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "ASSEMBLYAI_API_KEY"

// apiKeyHeader carries the key as it is, with no Bearer in front: the server refuses one
// with a prefix as an invalid key. The query string takes a temporary token instead, which
// is for a browser rather than for the router.
const apiKeyHeader = "Authorization"

// The latency and accuracy presets, which set the defaults the turn detector starts from.
const (
	ModeMinLatency  = "min_latency"
	ModeBalanced    = "balanced"
	ModeMaxAccuracy = "max_accuracy"
)

// DefaultMode is the preset a call gets. The server's own, balanced, ended the turn at a
// 390 ms pause mid-sentence in three sessions out of four on the shared fixture, which on
// a call is the agent answering half a sentence. This one did not in four out of four,
// and settled about 300 ms later, still well inside what a conversation puts up with.
const DefaultMode = ModeMaxAccuracy

// Client frame types.
const clientTypeTerminate = "Terminate"

// Server frame types.
const (
	eventBegin           = "Begin"
	eventSpeechStarted   = "SpeechStarted"
	eventTurn            = "Turn"
	eventSpeakerRevision = "SpeakerRevision"
	eventTermination     = "Termination"
	eventError           = "Error"
)

// The keyterms a session takes: a hundred, of up to fifty characters each. The count is
// stt.MaxKeyterms already, and the length is enforced here.
const (
	maxKeyterms     = 100
	maxKeytermRunes = 50
)

// The size of one binary frame, which the server refuses outside these bounds by ending
// the session.
const (
	minChunkMs = 50
	maxChunkMs = 1000
)

// Options configures the provider. APIKey falls back to ASSEMBLYAI_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong.
	Keyterms []string
	// LanguageHints steer the model toward these languages, as ISO codes, while still
	// letting it switch between them. Empty leaves it to switch across all it knows.
	LanguageHints []string
	// Mode is ModeMinLatency, ModeBalanced or ModeMaxAccuracy. Empty means DefaultMode.
	Mode string
	// MinTurnSilenceMs is the silence after which the model checks whether the turn is
	// over, and MaxTurnSilenceMs the silence after which it is over whatever the words
	// say. Zero leaves the preset's own.
	MinTurnSilenceMs int
	MaxTurnSilenceMs int
	// HandshakeTimeout bounds the initial connect and the wait for the session to open.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// serverMessage is a frame sent by the server. Every kind has a type and few share any
// other field, so one struct reads all of them.
type serverMessage struct {
	Type string `json:"type"`
	// Configuration is what Begin says the session was opened with.
	Configuration sessionConfiguration `json:"configuration"`
	// TurnOrder numbers the turns from zero, and Transcript restates the turn so far.
	TurnOrder  int64  `json:"turn_order"`
	EndOfTurn  bool   `json:"end_of_turn"`
	Transcript string `json:"transcript"`
	// ErrorCode is also the code the socket closes with, and Error the reason.
	ErrorCode int    `json:"error_code"`
	Error     string `json:"error"`
}

// sessionConfiguration is the part of Begin's echo this package asked for.
type sessionConfiguration struct {
	Model string `json:"model"`
	Mode  string `json:"mode"`
}

// STT is one Universal-3.6 Pro Realtime session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes, since a websocket connection allows only one writer, and
	// guards pending.
	writeMu sync.Mutex
	// pending is audio not yet sent because there is less of it than a frame has to hold.
	pending []byte

	// terminated is closed when the server has said its last, so Close can wait for the
	// tail of the call rather than cutting it off.
	terminated chan struct{}
	endOnce    sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	started     bool
	closed      bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("assemblyai: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("assemblyai: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.Mode == "" {
		options.Mode = DefaultMode
	}
	switch options.Mode {
	case ModeMinLatency, ModeBalanced, ModeMaxAccuracy:
	default:
		return nil, fmt.Errorf("assemblyai: mode must be %s, %s or %s, got %s",
			ModeMinLatency, ModeBalanced, ModeMaxAccuracy, options.Mode)
	}
	options.Keyterms = stt.CleanKeyterms(options.Keyterms)
	if len(options.Keyterms) > maxKeyterms {
		return nil, fmt.Errorf("assemblyai: at most %d keyterms, got %d",
			maxKeyterms, len(options.Keyterms))
	}
	for _, term := range options.Keyterms {
		if len([]rune(term)) > maxKeytermRunes {
			return nil, fmt.Errorf("assemblyai: keyterms are at most %d characters, got %q",
				maxKeytermRunes, term)
		}
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
		options:    options,
		logger:     logger.With("provider", ProviderName, "model", options.Model),
		emitter:    stt.NewEmitter(64),
		terminated: make(chan struct{}),
	}, nil
}

// Start dials the socket and waits for the server to open the session.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("assemblyai: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{apiKeyHeader: []string{s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("assemblyai: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("assemblyai: dial: %w", err)
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
		return fmt.Errorf("assemblyai: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("assemblyai: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("assemblyai: not started")
	}

	if err := s.sendAudio(pcm.Bytes()); err != nil {
		return fmt.Errorf("assemblyai: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close sends what audio is left, asks the server to settle the turn in progress, waits
// for it to finish, then tears the connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()

	if conn != nil {
		s.terminate()
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
// place this API takes it at connect time.
func (s *STT) endpoint() string {
	query := url.Values{}
	query.Set("speech_model", s.options.Model)
	query.Set("sample_rate", strconv.Itoa(stt.SampleRate))
	query.Set("encoding", "pcm_s16le")
	query.Set("mode", s.options.Mode)
	// Both lists go as JSON arrays inside one parameter, which is how this endpoint reads
	// a list.
	if len(s.options.Keyterms) > 0 {
		query.Set("keyterms_prompt", jsonList(s.options.Keyterms))
	}
	if len(s.options.LanguageHints) > 0 {
		query.Set("language_codes", jsonList(s.options.LanguageHints))
	}
	if s.options.MinTurnSilenceMs > 0 {
		query.Set("min_turn_silence", strconv.Itoa(s.options.MinTurnSilenceMs))
	}
	if s.options.MaxTurnSilenceMs > 0 {
		query.Set("max_turn_silence", strconv.Itoa(s.options.MaxTurnSilenceMs))
	}

	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + query.Encode()
}

// handshake waits for the server to open the session and checks it opened the one that
// was asked for. A rejected key or an invalid parameter arrives here as an Error frame
// rather than as a refused upgrade.
func (s *STT) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("assemblyai: read handshake: %w", err)
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("assemblyai: read handshake: %w", err)
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return fmt.Errorf("assemblyai: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("assemblyai: decode handshake: %w", err)
	}
	switch message.Type {
	case eventBegin:
	case eventError:
		return fmt.Errorf("assemblyai: session refused: %s", message.failure())
	default:
		return fmt.Errorf("assemblyai: expected %q, got %q", eventBegin, message.Type)
	}

	applied := message.Configuration
	if applied.Model != s.options.Model {
		return fmt.Errorf("assemblyai: asked for model %s and the server opened %s",
			s.options.Model, applied.Model)
	}
	if applied.Mode != s.options.Mode {
		return fmt.Errorf("assemblyai: asked for mode %s and the server applied %s",
			s.options.Mode, applied.Mode)
	}
	return nil
}

// sendAudio adds samples to what is waiting and sends it once there is enough for a
// frame, in frames no longer than the server takes.
func (s *STT) sendAudio(samples []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()

	s.pending = append(s.pending, samples...)
	minBytes, maxBytes := chunkBytes(minChunkMs), chunkBytes(maxChunkMs)
	sent := 0
	for len(s.pending)-sent >= minBytes {
		size := min(len(s.pending)-sent, maxBytes)
		if err := s.conn.WriteMessage(websocket.BinaryMessage, s.pending[sent:sent+size]); err != nil {
			return err
		}
		sent += size
	}
	s.pending = s.pending[:copy(s.pending, s.pending[sent:])]
	return nil
}

// terminate sends the audio still waiting, padded with silence to the shortest frame the
// server takes, asks for the session to end, and waits for the server's last word. A dead
// connection must not stop teardown.
func (s *STT) terminate() {
	s.writeMu.Lock()
	var err error
	if len(s.pending) > 0 {
		tail := make([]byte, max(len(s.pending), chunkBytes(minChunkMs)))
		copy(tail, s.pending)
		s.pending = nil
		err = s.conn.WriteMessage(websocket.BinaryMessage, tail)
	}
	if err == nil {
		err = s.conn.WriteMessage(websocket.TextMessage, []byte(`{"type":"`+clientTypeTerminate+`"}`))
	}
	s.writeMu.Unlock()
	if err != nil {
		s.logger.Debug("terminate not delivered", "error", err)
		return
	}

	select {
	case <-s.terminated:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the last words")
	}
}

// readLoop translates server frames into events until the connection ends.
func (s *STT) readLoop() {
	defer s.ended()
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
	case eventTurn:
		mode := stt.ModeReplacement
		if message.EndOfTurn {
			mode = stt.ModeFinal
		}
		s.sendTranscript(message, mode)
	case eventTermination:
		s.ended()
	case eventError:
		// The server closes the socket straight after, with the same code.
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  eventError,
			Fatal:    true,
		})
	case eventBegin, eventSpeechStarted, eventSpeakerRevision:
		// Begin was the handshake, SpeechStarted only says a Turn is coming, and speaker
		// revisions only arrive on a session that asked for speaker labels.
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

func (s *STT) sendTranscript(message serverMessage, mode stt.Mode) {
	text := strings.TrimSpace(message.Transcript)
	if text == "" {
		return
	}

	s.mu.Lock()
	participant := s.participant
	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	s.mu.Unlock()

	s.emitter.Send(stt.Transcript{
		Participant: participant,
		Mode:        mode,
		// turn_order counts from zero and Utterance from one.
		Utterance:        message.TurnOrder + 1,
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// ended tells a waiting Close that nothing more is coming.
func (s *STT) ended() {
	s.endOnce.Do(func() { close(s.terminated) })
}

// failure is what the server said went wrong, with the code it will close the socket with.
func (m serverMessage) failure() string {
	if m.Error == "" {
		return fmt.Sprintf("error %d", m.ErrorCode)
	}
	return fmt.Sprintf("%s (code %d)", m.Error, m.ErrorCode)
}

// chunkBytes is how many bytes of audio last this long, at two bytes a sample.
func chunkBytes(ms int) int {
	return 2 * stt.SampleRate * ms / 1000
}

// jsonList renders a list the way this endpoint reads one inside a query parameter.
func jsonList(items []string) string {
	encoded, err := json.Marshal(items)
	if err != nil {
		// A list of strings always encodes.
		panic(err)
	}
	return string(encoded)
}
