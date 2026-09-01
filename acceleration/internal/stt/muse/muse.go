// Package muse implements the stt.STT contract on top of Meta's Muse Voice Transcribe.
//
// The session is configured by the first frame rather than by the query string or a
// header: the API key travels inside that frame too, which is why Start writes it before
// anything else and why a rejected key surfaces as an error frame rather than as a failed
// dial.
//
// Partials arrive in CUMULATIVE mode, so each one restates the turn from its beginning
// rather than carrying only the words that are new. They are therefore replacements. The
// turn settles on speechComplete, which the server also sends for whatever it was still
// holding when the audio stream ends, so the tail of a call that was cut off is not lost.
package muse

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "muse"

// DefaultModel is the only speech-to-text model Meta serves.
const DefaultModel = "muse-voice-transcribe-1.0"

// DefaultURL is the realtime transcription socket.
const DefaultURL = "wss://api.meta.ai/v1/asr/realtime"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "META_API_KEY"

// The modes the server recognises. ModeEndpointing is the default here because a call
// needs the model to find the turn boundaries; ModePushToTalk leaves that to the caller,
// and ModeDiarization adds speaker labels on top of endpointing.
const (
	ModePushToTalk  = "PUSH_TO_TALK"
	ModeEndpointing = "ENDPOINTING"
	ModeDiarization = "DIARIZATION"
)

// encoding16k matches stt.SampleRate. The server also takes PCM_24KHZ, which it prefers,
// but upsampling audio that arrived at 16kHz would not put back what is not in it.
const encoding16k = "PCM_16KHZ"

// partialModeCumulative makes each partial restate the whole turn, which is what
// stt.ModeReplacement means. The alternative, DELTA, would need the consumer to stitch
// the turn back together.
const partialModeCumulative = "CUMULATIVE"

// Server message types.
const (
	messageSpeechStart    = "speechStart"
	messageTranscript     = "transcript"
	messageSpeechEnd      = "speechEnd"
	messageSpeechComplete = "speechComplete"
	messageSpeaker        = "speaker"
	messageAudioProgress  = "audioProgress"
	messageError          = "error"
)

// controlTypeEndStream tells the server the audio has stopped, which makes it transcribe
// what it is still holding, report it as speechComplete and close the socket.
const controlTypeEndStream = "endStream"

// Options configures the provider. APIKey falls back to META_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Mode is one of ModePushToTalk, ModeEndpointing or ModeDiarization. Empty means
	// ModeEndpointing.
	Mode string
	// Keyterms are the words the model would otherwise get wrong, sent as the vocabulary
	// to bias recognition toward. The API takes up to a hundred, which is stt.MaxKeyterms.
	Keyterms []string
	// LanguageHints are the languages to expect, as ISO codes such as "en", or as the
	// names the API itself uses. The model code-switches regardless; the hints only
	// weight it.
	LanguageHints []string
	// HandshakeTimeout bounds the initial connect.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// authorization carries the credentials, which this API takes in the setup frame rather
// than in a header.
type authorization struct {
	AccessToken string `json:"accessToken"`
}

// setupFrame is the first frame, which configures the session.
type setupFrame struct {
	Authorization authorization `json:"authorization"`
	AudioEncoding string        `json:"audioEncoding"`
	Model         string        `json:"model"`
	Mode          string        `json:"mode"`
	PartialMode   string        `json:"partialMode"`
	// EmitAudioProgress is off: the router has no use for a milestone every chunk.
	EmitAudioProgress bool     `json:"emitAudioProgress"`
	LanguageBias      []string `json:"languageBias,omitempty"`
	Keywords          []string `json:"keywords,omitempty"`
}

// serverMessage is a frame sent by the server.
type serverMessage struct {
	Type string `json:"type"`
	// TurnID numbers the run of speech, and is only on the turn boundary frames.
	TurnID int64 `json:"turnId"`
	// Transcript is the text, on transcript and speechComplete.
	Transcript string `json:"transcript"`
	// Final says the text will not change. speechComplete says the same thing about the
	// turn as a whole.
	Final bool `json:"final"`
	// AudioProcessedMs is how much audio the server has consumed so far, counting from
	// the start of the session rather than of the turn.
	AudioProcessedMs float64 `json:"audioProcessedMs"`
	// Label is the diarised speaker, "A" through to "Z", in ModeDiarization only.
	Label   string `json:"label"`
	Message string `json:"message"`
}

// STT is a Muse Voice Transcribe session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// finished is closed when the server has answered the end of the audio stream and
	// hung up, which is how Close knows the tail has been transcribed.
	finished     chan struct{}
	finishedOnce sync.Once

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// turnStartMs is how much audio had been processed when the current turn began, so
	// the turn's own duration can be told from the session-wide count the server reports.
	turnStartMs float64
	// utterance counts the runs of speech seen so far, and ended marks that the current
	// one is over so the next transcript starts a new one.
	utterance int64
	ended     bool
	// settled marks that the current turn already has a final. A transcript flagged final
	// and the speechComplete behind it report the same settled turn, and emitting both
	// would tell the rest of the call the caller said it twice.
	settled bool
	started bool
	closed  bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("muse: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("muse: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.Mode == "" {
		options.Mode = ModeEndpointing
	}
	switch options.Mode {
	case ModePushToTalk, ModeEndpointing, ModeDiarization:
	default:
		return nil, fmt.Errorf("muse: mode must be one of %s, %s or %s, got %s",
			ModePushToTalk, ModeEndpointing, ModeDiarization, options.Mode)
	}
	options.Keyterms = stt.CleanKeyterms(options.Keyterms)
	if len(options.Keyterms) > stt.MaxKeyterms {
		return nil, fmt.Errorf("muse: at most %d keyterms, got %d", stt.MaxKeyterms, len(options.Keyterms))
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

// Start dials the socket and sends the setup frame. The server does not acknowledge that
// frame, so a rejected key arrives later as an error event rather than from here.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("muse: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	conn, response, err := dialer.DialContext(ctx, s.endpoint(), nil)
	if err != nil {
		if response != nil {
			return fmt.Errorf("muse: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("muse: dial: %w", err)
	}
	s.conn = conn

	if err := s.setup(); err != nil {
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
		return fmt.Errorf("muse: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("muse: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("muse: not started")
	}

	if err := s.write(websocket.BinaryMessage, pcm.Bytes()); err != nil {
		return fmt.Errorf("muse: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close ends the audio stream, waits for the server to transcribe what it was holding,
// then tears down the connection.
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
		s.flush()
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements stt.STT.
func (s *STT) Provider() string { return ProviderName }

// Model implements stt.STT.
func (s *STT) Model() string { return s.options.Model }

// Client exposes the underlying WebSocket so callers can use the API directly.
func (s *STT) Client() *websocket.Conn { return s.conn }

// endpoint is the socket URL with the session identifier the API asks for.
func (s *STT) endpoint() string {
	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + "sessionId=" + url.QueryEscape(uuid.NewString())
}

// setup sends the frame that configures the session and authenticates it.
func (s *STT) setup() error {
	payload, err := json.Marshal(setupFrame{
		Authorization: authorization{AccessToken: "Bearer " + s.options.APIKey},
		AudioEncoding: encoding16k,
		Model:         s.options.Model,
		Mode:          s.options.Mode,
		PartialMode:   partialModeCumulative,
		LanguageBias:  languageBias(s.options.LanguageHints),
		Keywords:      s.options.Keyterms,
	})
	if err != nil {
		return fmt.Errorf("muse: encode setup: %w", err)
	}
	if err := s.write(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("muse: send setup: %w", err)
	}
	return nil
}

// flush tells the server the audio has stopped and waits for it to hang up, which it does
// once it has reported the transcript of everything it was still holding. A dead
// connection must not stop teardown.
func (s *STT) flush() {
	if err := s.write(websocket.TextMessage, []byte(`{"type":"`+controlTypeEndStream+`"}`)); err != nil {
		s.logger.Debug("endStream not delivered", "error", err)
		return
	}

	select {
	case <-s.finished:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the final transcript")
	}
}

func (s *STT) write(messageType int, payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return s.conn.WriteMessage(messageType, payload)
}

// readLoop translates server frames into events until the connection ends.
func (s *STT) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
		if err != nil {
			s.finishedOnce.Do(func() { close(s.finished) })
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
	case messageSpeechStart:
		s.startTurn(message.AudioProcessedMs)
	case messageTranscript:
		mode := stt.ModeReplacement
		if message.Final {
			mode = stt.ModeFinal
		}
		s.sendTranscript(message, mode)
	case messageSpeechComplete:
		s.sendTranscript(message, stt.ModeFinal)
	case messageSpeechEnd:
		// The boundary alone settles nothing: speechComplete carries the text the turn
		// settles on and follows straight after.
	case messageSpeaker:
		// The router labels transcripts from the audio track it fed, so a diarised label
		// is only worth seeing when the two disagree.
		s.logger.Debug("diarised speaker", "label", message.Label)
	case messageAudioProgress:
		s.logger.Debug("audio progress", "audio_processed_ms", message.AudioProcessedMs)
	case messageError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.Message),
			Context:  "server",
			Fatal:    true,
		})
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

func (s *STT) sendTranscript(message serverMessage, mode stt.Mode) {
	text := strings.TrimSpace(message.Transcript)
	if text == "" {
		return
	}
	if mode == stt.ModeFinal && s.alreadySettled() {
		return
	}

	participant, latencyMs, turnMs := s.snapshot(message.AudioProcessedMs)
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Utterance:        s.utteranceID(mode),
		Text:             text,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
		AudioDurationMs:  turnMs,
	})
}

// startTurn opens a new run of speech at the point in the audio the server reports.
func (s *STT) startTurn(audioProcessedMs float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.ended = true
	s.settled = false
	s.turnStartMs = audioProcessedMs
}

// alreadySettled reports whether this turn has had its final already.
func (s *STT) alreadySettled() bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.settled
}

// utteranceID numbers the run of speech the transcript belongs to, and records whether it
// settled that run.
//
// A final marks the run as over, but the count only moves on the next transcript, so the
// final and the speechStart that follows it are one boundary between utterances rather
// than two. A hypothesis arriving after a final belongs to a turn the server never
// announced the start of, so it clears the settled mark that would swallow that turn's own
// final.
func (s *STT) utteranceID(mode stt.Mode) int64 {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 || s.ended {
		s.utterance++
		s.ended = false
	}
	if mode == stt.ModeFinal {
		s.settled = true
		s.ended = true
	} else {
		s.settled = false
	}
	return s.utterance
}

// languageNames are the twenty-five languages the model recognises, keyed by the ISO code
// the rest of the router speaks. The API names them in full instead, so a hint has to be
// translated on the way out or it is silently ignored.
var languageNames = map[string]string{
	"ar": "Arabic",
	"bn": "Bengali",
	"de": "German",
	"en": "English",
	"es": "Spanish",
	"fr": "French",
	"he": "Hebrew",
	"hi": "Hindi",
	"id": "Indonesian",
	"it": "Italian",
	"ja": "Japanese",
	"kn": "Kannada",
	"ko": "Korean",
	"mr": "Marathi",
	"ms": "Malay",
	"nl": "Dutch",
	"pl": "Polish",
	"pt": "Portuguese",
	"ta": "Tamil",
	"te": "Telugu",
	"th": "Thai",
	"tl": "Tagalog",
	"tr": "Turkish",
	"vi": "Vietnamese",
	"zh": "Mandarin Chinese",
}

// languageBias renders the hints the way the API names languages. A hint that is not a
// code this model knows goes up as it was given, so a caller naming a language outright
// is not second-guessed.
func languageBias(hints []string) []string {
	if len(hints) == 0 {
		return nil
	}
	named := make([]string, 0, len(hints))
	for _, hint := range hints {
		if name, ok := languageNames[strings.ToLower(strings.TrimSpace(hint))]; ok {
			named = append(named, name)
			continue
		}
		named = append(named, hint)
	}
	return named
}

// snapshot returns the current speaker, how long ago audio was last sent, and how much
// audio the current turn covers.
func (s *STT) snapshot(audioProcessedMs float64) (stt.Participant, float64, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	turnMs := audioProcessedMs - s.turnStartMs
	if turnMs < 0 {
		turnMs = 0
	}
	return s.participant, latencyMs, turnMs
}
