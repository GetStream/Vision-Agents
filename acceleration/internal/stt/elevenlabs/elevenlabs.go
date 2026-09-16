// Package elevenlabs implements the stt.STT contract on top of ElevenLabs' Scribe v2
// Realtime.
//
// ElevenLabs serve Scribe twice over. This is the streaming half, which is the one a call
// needs; the batch model is more accurate and has features this one does not, diarization
// among them, and it belongs to the recorded path rather than here. A request that asks to
// be told who spoke is therefore routed away from this model rather than served by it, and
// that is deliberate: ElevenLabs say plainly that the realtime model does not diarize.
//
// The session is configured by the query string and acknowledged by a session_started
// frame, which Start waits for. Audio sent before that is audio nobody is listening to.
//
// Audio goes up base64-encoded inside a JSON frame, each of which says whether it also
// commits what has been heard so far. Partials arrive as partial_transcript and restate
// the segment rather than appending to it; the segment settles on committed_transcript,
// which is also what the timestamped and entity frames annotate, so a turn that arrives
// twice is reported once.
package elevenlabs

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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "elevenlabs"

// DefaultModel is Scribe v2 Realtime, the streaming model.
const DefaultModel = "scribe_v2_realtime"

// DefaultURL is the realtime transcription socket.
const DefaultURL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "ELEVENLABS_API_KEY"

// apiKeyHeader carries the credentials. The query string takes a single-use token
// instead, which is for a browser rather than for the router.
const apiKeyHeader = "xi-api-key"

// audioFormat16k matches stt.SampleRate and the samples stt.PcmData holds.
const audioFormat16k = "pcm_16000"

// The strategies for deciding when a segment is settled. CommitOnVAD leaves the boundary
// to the server's voice activity detection, which is what a call wants; CommitManually
// settles only what the client asks it to, chunk by chunk.
const (
	CommitOnVAD    = "vad"
	CommitManually = "manual"
)

// Client frame types.
const clientTypeAudioChunk = "input_audio_chunk"

// Server frame types.
const (
	eventSessionStarted                 = "session_started"
	eventPartialTranscript              = "partial_transcript"
	eventCommittedTranscript            = "committed_transcript"
	eventCommittedTranscriptWithTimings = "committed_transcript_with_timestamps"
	eventCommittedTranscriptEntities    = "committed_transcript_entities"
	eventWarning                        = "warning"
	eventCommitThrottled                = "commit_throttled"
	eventInsufficientAudioActivity      = "insufficient_audio_activity"
)

// The realtime model takes far fewer keyterms than the batch one, and shorter ones: fifty
// of up to twenty characters against a thousand of up to fifty. Both are below what
// stt.MaxKeyterms allows, so they are enforced here.
const (
	maxKeyterms     = 50
	maxKeytermRunes = 20
)

// flushSilenceMs is the silence sent alongside the commit that ends a session.
//
// The only frame this protocol lets a client send is a chunk of audio, and committing the
// tail of a call means sending one with the commit flag set. The field is required, so
// there has to be something in it, and a fiftieth of a second of quiet at the point the
// caller has already hung up changes nothing about what they said.
const flushSilenceMs = 20

// Options configures the provider. APIKey falls back to ELEVENLABS_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Keyterms are the words the model would otherwise get wrong, sent as the terms to
	// bias recognition toward.
	Keyterms []string
	// LanguageHints are the languages to expect, as ISO codes. The first is the language
	// the session is for and the rest are the ones it may switch into.
	LanguageHints []string
	// CommitStrategy is CommitOnVAD or CommitManually. Empty means CommitOnVAD, since a
	// call has no other way of finding where a turn ended.
	CommitStrategy string
	// VadSilenceThresholdSecs is how much silence ends a segment. Zero leaves the
	// server's own default.
	VadSilenceThresholdSecs float64
	// VadThreshold is how loud speech has to be to count as speech. Zero leaves the
	// server's own default.
	VadThreshold float64
	// MinSpeechDurationMs and MinSilenceDurationMs are the shortest run of either the
	// detector will act on. Zero leaves the server's own defaults.
	MinSpeechDurationMs  int
	MinSilenceDurationMs int
	// HandshakeTimeout bounds the initial connect and the wait for the session to open.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	Logger       *slog.Logger
}

// clientMessage is a frame sent to the server. Audio is the only kind there is: the
// commit flag on one of these is what this protocol has instead of a control frame.
type clientMessage struct {
	MessageType string `json:"message_type"`
	AudioBase64 string `json:"audio_base_64"`
	Commit      bool   `json:"commit"`
	SampleRate  int    `json:"sample_rate"`
}

// serverMessage is a frame sent by the server. The transcript frames and the whole family
// of failures share a shape, so one struct reads all of them.
type serverMessage struct {
	MessageType string `json:"message_type"`
	SessionID   string `json:"session_id"`
	// Text is the transcript, on the partial and committed frames. It restates the
	// segment rather than carrying only what is new.
	Text string `json:"text"`
	// LanguageCode is the language the server detected, on the timestamped frame.
	LanguageCode string `json:"language_code"`
	// Error is why a failure frame was sent, and Warning why a warning was.
	Error   string `json:"error"`
	Warning string `json:"warning"`
}

// STT is one Scribe v2 Realtime session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// settled receives when a segment is committed, so Close can wait for the tail of the
	// last one rather than cutting it off. Buffered and never blocked on, so the segments
	// nobody is waiting for during the call cost nothing.
	settled chan struct{}

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// utterance counts the runs of speech so far, and committed marks that the current
	// one has settled so the next partial starts a new one.
	utterance int64
	committed bool
	// settledText is the text the current run of speech settled on, so the timestamped
	// and entity frames that annotate it are not reported as the caller saying it again.
	settledText string
	started     bool
	closed      bool
}

// New validates the options and returns an unstarted provider.
func New(options Options) (*STT, error) {
	if options.APIKey == "" {
		options.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if options.APIKey == "" {
		return nil, fmt.Errorf("elevenlabs: api key is required (set %s)", apiKeyEnvVar)
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.URL == "" {
		options.URL = DefaultURL
	}
	if !strings.HasPrefix(options.URL, "ws://") && !strings.HasPrefix(options.URL, "wss://") {
		return nil, fmt.Errorf("elevenlabs: url must be ws:// or wss://, got %s", options.URL)
	}
	if options.CommitStrategy == "" {
		options.CommitStrategy = CommitOnVAD
	}
	switch options.CommitStrategy {
	case CommitOnVAD, CommitManually:
	default:
		return nil, fmt.Errorf("elevenlabs: commit strategy must be %s or %s, got %s",
			CommitOnVAD, CommitManually, options.CommitStrategy)
	}
	options.Keyterms = stt.CleanKeyterms(options.Keyterms)
	if len(options.Keyterms) > maxKeyterms {
		return nil, fmt.Errorf("elevenlabs: at most %d keyterms, got %d",
			maxKeyterms, len(options.Keyterms))
	}
	for _, term := range options.Keyterms {
		if len([]rune(term)) > maxKeytermRunes {
			return nil, fmt.Errorf("elevenlabs: keyterms are at most %d characters, got %q",
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
		return errors.New("elevenlabs: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{apiKeyHeader: []string{s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("elevenlabs: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("elevenlabs: dial: %w", err)
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
		return fmt.Errorf("elevenlabs: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("elevenlabs: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("elevenlabs: not started")
	}

	if err := s.sendAudio(pcm.Bytes(), false); err != nil {
		return fmt.Errorf("elevenlabs: write audio: %w", err)
	}
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close commits whatever the server is still holding, waits for it, then tears the
// connection down.
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

// endpoint is the socket with the session's configuration attached, which is the only
// place this API takes it.
func (s *STT) endpoint() string {
	query := url.Values{}
	query.Set("model_id", s.options.Model)
	query.Set("audio_format", audioFormat16k)
	query.Set("commit_strategy", s.options.CommitStrategy)
	if language := s.options.LanguageHints; len(language) > 0 {
		query.Set("language_code", language[0])
		// The rest are the languages the call may switch into, which is a different
		// question from the one it is in.
		for _, secondary := range language[1:] {
			query.Add("secondary_languages", secondary)
		}
	}
	if s.options.VadSilenceThresholdSecs > 0 {
		query.Set("vad_silence_threshold_secs", seconds(s.options.VadSilenceThresholdSecs))
	}
	if s.options.VadThreshold > 0 {
		query.Set("vad_threshold", seconds(s.options.VadThreshold))
	}
	if s.options.MinSpeechDurationMs > 0 {
		query.Set("min_speech_duration_ms", strconv.Itoa(s.options.MinSpeechDurationMs))
	}
	if s.options.MinSilenceDurationMs > 0 {
		query.Set("min_silence_duration_ms", strconv.Itoa(s.options.MinSilenceDurationMs))
	}
	// Repeated rather than joined: one term per copy is how this endpoint reads a list.
	for _, term := range s.options.Keyterms {
		query.Add("keyterms", term)
	}

	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + query.Encode()
}

// handshake waits for the server to open the session. Audio sent before that is audio the
// server is not yet listening to, and a rejected key arrives here rather than as silence.
func (s *STT) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("elevenlabs: read handshake: %w", err)
	}
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("elevenlabs: read handshake: %w", err)
	}
	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return fmt.Errorf("elevenlabs: read handshake: %w", err)
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return fmt.Errorf("elevenlabs: decode handshake: %w", err)
	}
	if message.MessageType != eventSessionStarted {
		if message.Error != "" {
			return fmt.Errorf("elevenlabs: session refused: %s", message.Error)
		}
		return fmt.Errorf("elevenlabs: expected %q, got %q", eventSessionStarted, message.MessageType)
	}
	return nil
}

// flush commits the audio the server is still holding and waits for the transcript of it,
// so the tail of a call that was cut off is not lost. A dead connection must not stop
// teardown.
func (s *STT) flush() {
	// Forget the segments that are already settled, so the wait below is for one that
	// comes after the audio stopped.
	select {
	case <-s.settled:
	default:
	}

	if err := s.sendAudio(silence(flushSilenceMs), true); err != nil {
		s.logger.Debug("commit not delivered", "error", err)
		return
	}

	select {
	case <-s.settled:
	case <-time.After(s.options.FlushTimeout):
		s.logger.Debug("timed out waiting for the last words")
	}
}

// sendAudio writes one chunk, saying whether it also settles what has been heard so far.
func (s *STT) sendAudio(samples []byte, commit bool) error {
	payload, err := json.Marshal(clientMessage{
		MessageType: clientTypeAudioChunk,
		AudioBase64: base64.StdEncoding.EncodeToString(samples),
		Commit:      commit,
		SampleRate:  stt.SampleRate,
	})
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
	switch message.MessageType {
	case eventPartialTranscript:
		s.heard(message.Text)
	case eventCommittedTranscript, eventCommittedTranscriptWithTimings:
		s.settle(message.Text, message.LanguageCode)
	case eventCommittedTranscriptEntities:
		// The entities found in a segment that has already been reported. Nothing in the
		// shared contract carries them, and the text is the same text.
	case eventSessionStarted:
		// The handshake already waited for this one, and the server sends it once.
	case eventWarning:
		s.logger.Debug("the server warned about this session", "warning", message.Warning)
	case eventCommitThrottled, eventInsufficientAudioActivity:
		// The session carries on: one commit was refused or one segment had nothing in
		// it, which is not the same as the call being over.
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  message.MessageType,
		})
	default:
		// Scribe has a failure frame per reason, from an expired quota to unaccepted
		// terms, and they all carry the reason in the same field. Reading them by that
		// rather than by name means a reason added later is still reported.
		if message.Error == "" {
			s.logger.Debug("unhandled frame", "type", message.MessageType)
			return
		}
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  message.MessageType,
			Fatal:    true,
		})
	}
}

// heard reports what the caller seems to be saying. A partial restates the segment, so it
// is a replacement, and it is the frame that tells a settled segment from the next one
// beginning.
func (s *STT) heard(text string) {
	if strings.TrimSpace(text) == "" {
		return
	}
	s.sendTranscript(text, "", stt.ModeReplacement)
}

// settle reports the segment the server has committed.
//
// The same text arrives more than once when the session was asked for timestamps or
// entities: those frames annotate a committed segment rather than replacing it. Reporting
// each would tell the rest of the call the caller said it twice, so a segment that has
// already settled on this text is only acknowledged.
func (s *STT) settle(text, language string) {
	settled := strings.TrimSpace(text)
	if settled == "" {
		// Nothing was said in the audio that was committed, but a Close waiting on it has
		// its answer all the same.
		s.reachedABoundary()
		return
	}

	if s.alreadySettled(settled) {
		s.reachedABoundary()
		return
	}
	s.sendTranscript(text, language, stt.ModeFinal)
	s.reachedABoundary()
}

func (s *STT) sendTranscript(text, language string, mode stt.Mode) {
	participant, utterance, latencyMs := s.snapshot(text, mode)
	s.emitter.Send(stt.Transcript{
		Participant:      participant,
		Mode:             mode,
		Utterance:        utterance,
		Text:             strings.TrimSpace(text),
		Language:         language,
		Provider:         ProviderName,
		Model:            s.options.Model,
		ProcessingTimeMs: latencyMs,
	})
}

// alreadySettled reports whether this is the text the current run of speech settled on.
func (s *STT) alreadySettled(text string) bool {
	s.mu.Lock()
	defer s.mu.Unlock()

	return s.committed && s.settledText == text
}

// snapshot returns the current speaker, the run of speech this text belongs to, and how
// long ago audio was last sent.
//
// A partial after a commit is the next run of speech beginning, which is the only thing
// this protocol says about where one ends: the segments are the server's and it never
// numbers them.
func (s *STT) snapshot(text string, mode stt.Mode) (stt.Participant, int64, float64) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.utterance == 0 || s.committed {
		s.utterance++
		s.committed = false
		s.settledText = ""
	}
	if mode == stt.ModeFinal {
		s.committed = true
		s.settledText = strings.TrimSpace(text)
	}

	var latencyMs float64
	if !s.lastAudioAt.IsZero() {
		latencyMs = float64(time.Since(s.lastAudioAt).Microseconds()) / 1000
	}
	return s.participant, s.utterance, latencyMs
}

// reachedABoundary tells a waiting Close that the server has finished a segment.
func (s *STT) reachedABoundary() {
	select {
	case s.settled <- struct{}{}:
	default:
	}
}

// failure is what the server said went wrong, from whichever of the two fields it used.
func (m serverMessage) failure() string {
	switch {
	case m.Error != "":
		return m.Error
	case m.Warning != "":
		return m.Warning
	default:
		return "unknown error"
	}
}

// silence is a run of quiet of the given length, as the samples go on the wire.
func silence(ms int) []byte {
	// Two bytes per sample, and the quiet is the zero value of both of them.
	return make([]byte, 2*stt.SampleRate*ms/1000)
}

// seconds renders a threshold without the exponent or the trailing zeroes a default
// format would add.
func seconds(value float64) string {
	return strconv.FormatFloat(value, 'f', -1, 64)
}
