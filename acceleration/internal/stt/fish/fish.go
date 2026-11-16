// Package fish implements the stt.STT contract on top of Fish Audio's realtime
// transcriber, transcribe-stream-1.
//
// Every message either way is one MessagePack map in a binary frame. The session is set
// up by a config message, which Start sends and then waits for session_started to accept.
// Audio goes up as raw PCM inside audio messages.
//
// The server decides where a turn ends and, in its auto mode, whether the turn is the
// caller's at all: noise, a television and other voices open turns too, and are drafted
// like speech. So a turn is reported only once user_turn_taken says it is the caller's,
// and from then on each draft restates it until committed settles it. A turn that never
// becomes the caller's is never reported, rather than reported and then left unsettled.
// audio_done settles whatever is still open and is answered by session_done, which is
// what Close waits for.
package fish

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"
	"github.com/vmihailenco/msgpack/v5"

	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "fish"

// DefaultModel is Fish's realtime transcriber.
const DefaultModel = "transcribe-stream-1"

// DefaultURL is the realtime transcription socket.
const DefaultURL = "wss://api.fish.audio/v1/asr/realtime"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "FISH_API_KEY"

// The turn detection modes a call can run in. The server's third, manual, ends a turn
// only when the client says so, and nothing here would.
const (
	// ModeAuto ends a turn when the speaker has finished rather than on a silence alone,
	// and tells the caller's turns from other sounds.
	ModeAuto = "auto"
	// ModeVAD ends every turn after a fixed silence.
	ModeVAD = "vad"
)

// The silence_timeout_ms each mode takes. In auto mode it is only the safety net that
// ends a turn whatever its words say; in vad mode it is what ends every turn. The server
// refuses a value outside its mode's range by closing the session.
var silenceRange = map[string][2]int{
	ModeAuto: {3000, 60000},
	ModeVAD:  {300, 60000},
}

// The keyterms a session takes: a hundred, of up to thirty-two characters each.
const (
	maxKeyterms     = 100
	maxKeytermRunes = 32
)

// keepAlivePeriod is how often the session is pinged. The server closes a session it has
// heard nothing from for 30 seconds, and a muted caller sends no audio.
const keepAlivePeriod = 10 * time.Second

// Client message types.
const (
	clientTypeConfig    = "config"
	clientTypeAudio     = "audio"
	clientTypeConfigure = "configure"
	clientTypeAudioDone = "audio_done"
)

// Server message types.
const (
	eventSessionStarted = "session_started"
	eventSpeechStart    = "speech_start"
	eventUserTurnTaken  = "user_turn_taken"
	eventDraft          = "draft"
	eventEagerEndOfTurn = "eager_end_of_turn"
	eventTurnResumed    = "turn_resumed"
	eventBackchannel    = "user_backchannel"
	eventCommitted      = "committed"
	eventConfigured     = "configured"
	eventProgress       = "progress"
	eventError          = "error"
	eventSessionDone    = "session_done"
)

// soundTag is a sound the model heard rather than a word, such as [laughter]. The set is
// open, so it is matched by its brackets rather than by name.
var soundTag = regexp.MustCompile(`\[[^\[\]]*\]`)

// Options configures the provider. APIKey falls back to FISH_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong.
	Keyterms []string
	// Mode is ModeAuto or ModeVAD. Empty means ModeAuto.
	Mode string
	// SilenceTimeoutMs is the silence after which a turn ends: in auto mode whatever its
	// words say, in vad mode always. Zero leaves the mode's own.
	SilenceTimeoutMs int
	// Eager asks for eager_end_of_turn, the transcript sent as soon as the model guesses
	// the turn may be over.
	Eager bool
	// HandshakeTimeout bounds the initial connect and the wait for the session to open.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a message sent to the server. Every kind has a type and takes few of
// the rest, so one struct writes all of them.
type clientMessage struct {
	Type          string          `msgpack:"type"`
	Audio         *audioFormat    `msgpack:"audio,omitempty"`
	TurnDetection *turnDetection  `msgpack:"turn_detection,omitempty"`
	Results       *results        `msgpack:"results,omitempty"`
	Context       *sessionContext `msgpack:"context,omitempty"`
	Data          []byte          `msgpack:"data,omitempty"`
}

type audioFormat struct {
	Format     string `msgpack:"format"`
	SampleRate int    `msgpack:"sample_rate"`
	Channels   int    `msgpack:"channels"`
}

type turnDetection struct {
	Mode             string `msgpack:"mode"`
	Eager            bool   `msgpack:"eager"`
	SilenceTimeoutMs int    `msgpack:"silence_timeout_ms,omitempty"`
}

// results asks for drafts, which are the interim transcripts, and turns diarization off,
// which the server has on by default and which writes speaker tags into the text.
type results struct {
	Drafts  bool `msgpack:"drafts"`
	Diarize bool `msgpack:"diarize"`
}

type sessionContext struct {
	Terms []string `msgpack:"terms"`
}

// serverMessage is a message sent by the server. Every kind has a type and few share any
// other field, so one struct reads all of them.
type serverMessage struct {
	Type string `msgpack:"type"`
	// TurnIndex numbers the turns from zero, and every message about a turn carries it.
	TurnIndex int64 `msgpack:"turn_index"`
	// Text restates the turn so far on a draft, and is the turn on committed.
	Text         string `msgpack:"text"`
	LanguageCode string `msgpack:"language_code"`
	AudioStartMs int64  `msgpack:"audio_start_ms"`
	AudioEndMs   int64  `msgpack:"audio_end_ms"`
	// UserTurn is whether a committed turn was the caller's. Fish say to treat it as true
	// when it is missing.
	UserTurn *bool `msgpack:"user_turn"`
	// OK is whether a configure was applied.
	OK          bool   `msgpack:"ok"`
	ErrorType   string `msgpack:"error_type"`
	Message     string `msgpack:"message"`
	Recoverable bool   `msgpack:"recoverable"`
	RequestID   string `msgpack:"request_id"`
}

// STT is one transcribe-stream-1 session.
type STT struct {
	options Options
	keys    *keypool.Pool
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes, since a websocket connection allows only one writer.
	writeMu sync.Mutex

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
	// taken are the open turns known to be the caller's.
	taken   map[int64]bool
	started bool
	closed  bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	keys := keypool.FromEnv(options.APIKey, apiKeyEnvVar).Pin()
	if keys.Len() == 0 {
		return nil, stack.Wrap(fmt.Errorf("fish: api key is required (set %s)", apiKeyEnvVar))
	}
	options.APIKey = keys.Peek()
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, stack.Wrap(fmt.Errorf("fish: url must be ws:// or wss://, got %s", options.URL))
	}
	if options.Mode == "" {
		options.Mode = ModeAuto
	}
	bounds, ok := silenceRange[options.Mode]
	if !ok {
		return nil, stack.Wrap(fmt.Errorf("fish: mode must be %s or %s, got %s", ModeAuto, ModeVAD, options.Mode))
	}
	if options.SilenceTimeoutMs != 0 && (options.SilenceTimeoutMs < bounds[0] || options.SilenceTimeoutMs > bounds[1]) {
		return nil, stack.Wrap(fmt.Errorf("fish: silence timeout must be between %d and %d ms in %s mode, got %d",
			bounds[0], bounds[1], options.Mode, options.SilenceTimeoutMs))
	}
	keyterms, err := validKeyterms(options.Keyterms)
	if err != nil {
		return nil, err
	}
	options.Keyterms = keyterms
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
		keys:       keys,
		logger:     logger.With("provider", ProviderName, "model", options.Model),
		emitter:    stt.NewEmitter(64),
		terminated: make(chan struct{}),
		taken:      map[int64]bool{},
	}, nil
}

// Start dials the socket, configures the session and waits for the server to open it.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return stack.Wrap(errors.New("fish: already started"))
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	conn, response, _, err := s.keys.Dial(ctx, dialer, func(key string) (string, http.Header) {
		return s.options.URL, http.Header{
			"Authorization": []string{"Bearer " + key},
			"model":         []string{s.options.Model},
		}
	})
	if err != nil {
		if response != nil {
			return stack.Wrap(fmt.Errorf("fish: dial: %w (http %d)", err, response.StatusCode))
		}
		return stack.Wrap(fmt.Errorf("fish: dial: %w", err))
	}
	s.conn = conn

	if err := s.handshake(); err != nil {
		conn.Close()
		return err
	}

	s.emitter.Send(stt.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop()
	go s.keepAlive()
	return nil
}

// ProcessAudio streams one chunk of audio. The participant labels any transcript that
// results from it.
func (s *STT) ProcessAudio(pcm stt.PcmData, participant stt.Participant) error {
	if err := pcm.Validate(stt.SampleRate); err != nil {
		return fmt.Errorf("fish: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("fish: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("fish: not started")
	}

	if err := s.send(clientMessage{Type: clientTypeAudio, Data: pcm.Bytes()}); err != nil {
		return fmt.Errorf("fish: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close asks the server to settle the turn in progress, waits for it to finish, then
// tears the connection down.
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
		s.finish()
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// SetKeyterms implements stt.Retuner: configure takes a new list on the open session, and
// it replaces the one the session started with. A list the server refuses is reported as
// an error event when it says so.
func (s *STT) SetKeyterms(terms []string) error {
	terms, err := validKeyterms(terms)
	if err != nil {
		return err
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.mu.Unlock()
	if closed {
		return stack.Wrap(errors.New("fish: session closed"))
	}
	if !started || s.conn == nil {
		return stack.Wrap(errors.New("fish: not started"))
	}

	if terms == nil {
		terms = []string{}
	}
	if err := s.send(clientMessage{Type: clientTypeConfigure, Context: &sessionContext{Terms: terms}}); err != nil {
		return stack.Wrap(fmt.Errorf("fish: configure keyterms: %w", err))
	}
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// Metering is the router's count of the audio it streams: Fish bill a session by how long
// it is open, which a call streaming all the time matches, and a turn says nothing of
// either.
func (s *STT) Metering() stt.Metering { return stt.MeteredByRouter }

// Client exposes the underlying WebSocket so callers can use the session directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// config is the session's first message, which the server has to receive within ten
// seconds of the connection opening.
func (s *STT) config() clientMessage {
	message := clientMessage{
		Type:  clientTypeConfig,
		Audio: &audioFormat{Format: "pcm_s16le", SampleRate: stt.SampleRate, Channels: 1},
		TurnDetection: &turnDetection{
			Mode:             s.options.Mode,
			Eager:            s.options.Eager,
			SilenceTimeoutMs: s.options.SilenceTimeoutMs,
		},
		Results: &results{Drafts: true},
	}
	if len(s.options.Keyterms) > 0 {
		message.Context = &sessionContext{Terms: s.options.Keyterms}
	}
	return message
}

// handshake configures the session and waits for the server to open it. A config the
// server refuses arrives as an error message, and the socket closes straight after.
func (s *STT) handshake() error {
	if err := s.send(s.config()); err != nil {
		return stack.Wrap(fmt.Errorf("fish: write config: %w", err))
	}
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return stack.Wrap(fmt.Errorf("fish: read handshake: %w", err))
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return stack.Wrap(fmt.Errorf("fish: read handshake: %w", err))
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return stack.Wrap(fmt.Errorf("fish: read handshake: %w", err))
	}

	var message serverMessage
	if err := msgpack.Unmarshal(raw, &message); err != nil {
		return stack.Wrap(fmt.Errorf("fish: decode handshake: %w", err))
	}
	switch message.Type {
	case eventSessionStarted:
		s.logger.Debug("session started", "request_id", message.RequestID)
		return nil
	case eventError:
		return stack.Wrap(fmt.Errorf("fish: session refused: %s", message.failure()))
	default:
		return stack.Wrap(fmt.Errorf("fish: expected %q, got %q", eventSessionStarted, message.Type))
	}
}

// send writes one message. Every message is a MessagePack map in a binary frame.
func (s *STT) send(message clientMessage) error {
	encoded, err := msgpack.Marshal(message)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(websocket.BinaryMessage, encoded)
}

// finish says no more audio is coming and waits for the server's last word, which settles
// the turn a caller cut off mid-sentence left open. A dead connection must not stop
// teardown.
func (s *STT) finish() {
	if err := s.send(clientMessage{Type: clientTypeAudioDone}); err != nil {
		s.logger.Debug("audio_done not delivered", "error", err)
		return
	}

	select {
	case <-s.terminated:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the last words")
	}
}

// keepAlive pings the socket until the session ends. A ping counts as the client being
// there, and the server answers it.
func (s *STT) keepAlive() {
	ticker := time.NewTicker(keepAlivePeriod)
	defer ticker.Stop()
	for {
		select {
		case <-s.terminated:
			return
		case <-ticker.C:
			if err := s.conn.WriteControl(websocket.PingMessage, nil, time.Now().Add(keepAlivePeriod)); err != nil {
				s.logger.Debug("keepalive ping failed", "error", err)
			}
		}
	}
}

// readLoop translates server messages into events until the connection ends.
func (s *STT) readLoop() {
	defer s.ended()
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			s.handleReadError(err)
			return
		}

		var message serverMessage
		if err := msgpack.Unmarshal(raw, &message); err != nil {
			s.logger.Debug("undecodable message", "error", err)
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
	case eventSpeechStart:
		// A turn opened in vad mode is the caller's by definition, and gets no
		// user_turn_taken to say so.
		if s.options.Mode == ModeVAD {
			s.take(message.TurnIndex)
		}
	case eventUserTurnTaken:
		s.take(message.TurnIndex)
	case eventDraft, eventEagerEndOfTurn:
		if s.isTaken(message.TurnIndex) {
			s.sendTranscript(message, stt.ModeReplacement)
		}
	case eventCommitted:
		s.mu.Lock()
		delete(s.taken, message.TurnIndex)
		s.mu.Unlock()
		if message.UserTurn == nil || *message.UserTurn {
			s.sendTranscript(message, stt.ModeFinal)
		}
	case eventConfigured:
		if !message.OK {
			s.emitter.Send(stt.Error{
				Provider: ProviderName,
				Model:    s.options.Model,
				Err:      fmt.Errorf("configure refused: %s", message.Message),
				Context:  eventConfigured,
			})
		}
	case eventError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  eventError,
			Fatal:    !message.Recoverable,
		})
	case eventSessionDone:
		s.ended()
	case eventTurnResumed, eventBackchannel, eventProgress:
		// A resumed turn carries on under the number it had, and a backchannel turn
		// commits as the caller's only if they go on to take it.
	default:
		s.logger.Debug("unhandled message", "type", message.Type)
	}
}

func (s *STT) take(turn int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.taken[turn] = true
}

func (s *STT) isTaken(turn int64) bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.taken[turn]
}

func (s *STT) sendTranscript(message serverMessage, mode stt.Mode) {
	text := words(message.Text)
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
		// turn_index counts from zero and Utterance from one.
		Utterance:        message.TurnIndex + 1,
		Text:             text,
		Language:         message.LanguageCode,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
		AudioDurationMs:  float64(message.AudioEndMs - message.AudioStartMs),
	})
}

// ended tells a waiting Close that nothing more is coming.
func (s *STT) ended() {
	s.endOnce.Do(func() { close(s.terminated) })
}

// failure is what the server said went wrong, by the type Fish say to match on.
func (m serverMessage) failure() string {
	if m.Message == "" {
		return m.ErrorType
	}
	return fmt.Sprintf("%s: %s", m.ErrorType, m.Message)
}

// words is the text without its sound tags, which are sounds the model heard rather than
// something said: a turn that is only [laughter] has nothing in it to answer.
func words(text string) string {
	return strings.Join(strings.Fields(soundTag.ReplaceAllString(text, " ")), " ")
}

// validKeyterms is the list cleaned and held to what a session takes.
func validKeyterms(terms []string) ([]string, error) {
	terms = stt.CleanKeyterms(terms)
	if len(terms) > maxKeyterms {
		return nil, stack.Wrap(fmt.Errorf("fish: at most %d keyterms, got %d", maxKeyterms, len(terms)))
	}
	for _, term := range terms {
		if len([]rune(term)) > maxKeytermRunes {
			return nil, stack.Wrap(fmt.Errorf("fish: keyterms are at most %d characters, got %q",
				maxKeytermRunes, term))
		}
	}
	return terms, nil
}
