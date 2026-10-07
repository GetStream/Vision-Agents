// Package microsoft implements the stt.STT contract on top of Microsoft's
// MAI-Transcribe-2-Streaming, over the OpenAI Realtime-compatible WebSocket a Microsoft
// Foundry resource serves it on.
//
// The protocol is the realtime one with one addition. A delta carries newly finalized text,
// to be appended exactly as sent, and an intermediate frame - Microsoft's own - restates the
// provisional suffix after the last delta, so the best reading of the turn so far is every
// delta joined plus the latest intermediate. A completed frame is the whole transcript of
// the audio since the previous commit.
//
// What the server will not do is say where a turn ends. Turn detection can only be null:
// there is no server-side voice activity detection and no automatic commit, and nothing is
// completed until the client commits. So this provider commits when the words stop
// changing for TurnGrace, the same boundary the Together providers draw on the same
// protocol, and publishes the completed transcript that comes back as the settled turn.
package microsoft

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "microsoft"

// DefaultModel is the model id Microsoft publish, which is also the name Foundry gives a
// deployment of it unless told otherwise.
const DefaultModel = "MAI-Transcribe-2-Streaming"

// Environment variables, named as Microsoft's own samples name them.
const (
	// apiKeyEnvVar is the Foundry resource's key.
	apiKeyEnvVar = "AZURE_MAI_API_KEY"
	// endpointEnvVar is the resource root, https://<resource>.services.ai.azure.com.
	endpointEnvVar = "AZURE_MAI_ENDPOINT"
	// deploymentEnvVar is the name the model was deployed under in Foundry.
	deploymentEnvVar = "AZURE_MAI_DEPLOYMENT_NAME"
)

// realtimePath is where the socket lives under the resource root.
const realtimePath = "/mai/v1/realtime"

// Client frame types.
const (
	clientTypeUpdate = "session.update"
	clientTypeAppend = "input_audio_buffer.append"
	clientTypeCommit = "input_audio_buffer.commit"
)

// Server frame types.
const (
	eventSessionCreated = "session.created"
	eventSessionUpdated = "session.updated"
	eventCommitted      = "input_audio_buffer.committed"
	eventDelta          = "conversation.item.input_audio_transcription.delta"
	eventIntermediate   = "conversation.item.input_audio_transcription.intermediate"
	eventCompleted      = "conversation.item.input_audio_transcription.completed"
	eventFailed         = "conversation.item.input_audio_transcription.failed"
	eventError          = "error"
)

// flushGrace is how long Close waits for the tail when nothing is outstanding. The last
// utterance is already settled in that case and no further one is coming, so waiting the
// full timeout would spend it in full on every hangup.
const flushGrace = 1500 * time.Millisecond

// defaultTurnGrace is how long the transcript has to stop changing before the turn is
// committed. It is the boundary the Together providers measured on the same protocol, and
// it leaves the settled turn inside the two and a half seconds sttsuite allows.
const defaultTurnGrace = 1200 * time.Millisecond

// Options configures the provider. APIKey, Endpoint and Deployment fall back to
// AZURE_MAI_API_KEY, AZURE_MAI_ENDPOINT and AZURE_MAI_DEPLOYMENT_NAME.
type Options struct {
	// APIKey is sent as the api-key header.
	APIKey string
	// Endpoint is the Foundry resource root, https://<resource>.services.ai.azure.com.
	Endpoint string
	// Deployment is the name the model was deployed under in Foundry. Empty uses Model,
	// which is what Foundry names a deployment by default.
	Deployment string
	// Model is the id reported in stats.
	Model string
	// Language is a code such as "en". Empty lets the model detect it.
	Language string
	// HandshakeTimeout bounds the initial connect and the wait for the session to be
	// configured.
	HandshakeTimeout time.Duration
	// FlushTimeout bounds how long Close waits for the transcript of whatever audio the
	// server is still holding.
	FlushTimeout time.Duration
	// TurnGrace is how long the transcript has to stop changing before the turn is
	// committed. Nothing on this protocol says where a turn ends.
	TurnGrace time.Duration
	Logger    *slog.Logger
}

// clientMessage is a frame sent to the server.
type clientMessage struct {
	Type    string         `json:"type"`
	Audio   string         `json:"audio,omitempty"`
	Session *sessionConfig `json:"session,omitempty"`
}

type sessionConfig struct {
	Type  string       `json:"type"`
	Audio sessionAudio `json:"audio"`
}

type sessionAudio struct {
	Input sessionInput `json:"input"`
}

// sessionInput is how the audio is read. Turn detection and noise reduction are sent as
// null because null is the only value the server takes for either.
type sessionInput struct {
	Format         audioFormat         `json:"format"`
	Transcription  transcriptionConfig `json:"transcription"`
	TurnDetection  *struct{}           `json:"turn_detection"`
	NoiseReduction *struct{}           `json:"noise_reduction"`
}

type audioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate"`
}

type transcriptionConfig struct {
	Model string `json:"model"`
	// Language is null for automatic detection.
	Language *string `json:"language"`
}

// serverMessage is a frame sent by the server.
type serverMessage struct {
	Type string `json:"type"`
	// Delta is newly finalized text, to be appended as it is.
	Delta string `json:"delta"`
	// Intermediate is the provisional text after the last delta, replacing the one before.
	Intermediate string `json:"intermediate"`
	// Transcript is the whole of the audio since the previous commit.
	Transcript string       `json:"transcript"`
	Error      *serverError `json:"error"`
}

type serverError struct {
	Message string `json:"message"`
}

// STT is a MAI-Transcribe-2-Streaming transcription session.
type STT struct {
	options Options
	logger  *slog.Logger
	emitter *stt.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// settled receives when a commit is answered, so Close can wait for the tail of the
	// last one rather than cutting it off. Buffered and never blocked on, so the commits
	// nobody is waiting for during the call cost nothing.
	settled chan struct{}

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label transcripts
	// that arrive asynchronously.
	participant stt.Participant
	// lastAudioAt is when audio was last sent, so latency can be reported as the delay
	// between sending audio and hearing about it.
	lastAudioAt time.Time
	// finalized is the deltas since the last commit, joined exactly as they were sent.
	finalized string
	// intermediate is the latest provisional suffix after finalized.
	intermediate string
	// shown is the last hypothesis sent, so a frame that changes nothing neither repeats
	// it nor holds the turn open.
	shown string
	// uncommitted marks audio sent since the last commit, which is what Close has to ask
	// the server to finish.
	uncommitted bool
	// committing marks a commit the server has not answered yet.
	committing bool
	// turn fires when the transcript stops changing, which is the only end of turn there is.
	turn *time.Timer
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
		return nil, stack.Wrap(fmt.Errorf("microsoft: api key is required (set %s)", apiKeyEnvVar))
	}
	if options.Endpoint == "" {
		options.Endpoint = os.Getenv(endpointEnvVar)
	}
	if options.Endpoint == "" {
		return nil, stack.Wrap(fmt.Errorf("microsoft: endpoint is required (set %s)", endpointEnvVar))
	}
	if _, err := socketURL(options.Endpoint); err != nil {
		return nil, err
	}
	if options.Model == "" {
		options.Model = DefaultModel
	}
	if options.Deployment == "" {
		options.Deployment = os.Getenv(deploymentEnvVar)
	}
	if options.Deployment == "" {
		options.Deployment = options.Model
	}
	if options.HandshakeTimeout == 0 {
		options.HandshakeTimeout = 30 * time.Second
	}
	if options.FlushTimeout == 0 {
		options.FlushTimeout = 10 * time.Second
	}
	if options.TurnGrace == 0 {
		options.TurnGrace = defaultTurnGrace
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

// Start dials the socket, configures the session and waits for the server to accept it.
func (s *STT) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return stack.Wrap(errors.New("microsoft: already started"))
	}
	s.started = true
	s.mu.Unlock()

	endpoint, err := socketURL(s.options.Endpoint)
	if err != nil {
		return err
	}
	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"api-key": []string{s.options.APIKey}}

	conn, response, err := dialer.DialContext(ctx, endpoint, header)
	if err != nil {
		if response != nil {
			return stack.Wrap(fmt.Errorf("microsoft: dial: %w (http %d)", err, response.StatusCode))
		}
		return stack.Wrap(fmt.Errorf("microsoft: dial: %w", err))
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
		return fmt.Errorf("microsoft: %w", err)
	}

	s.mu.Lock()
	closed, started := s.closed, s.started
	s.participant = participant
	s.lastAudioAt = time.Now()
	s.mu.Unlock()

	if closed {
		return errors.New("microsoft: session closed")
	}
	if !started || s.conn == nil {
		return errors.New("microsoft: not started")
	}

	frame := clientMessage{
		Type:  clientTypeAppend,
		Audio: base64.StdEncoding.EncodeToString(pcm.Bytes()),
	}
	if err := s.send(frame); err != nil {
		return fmt.Errorf("microsoft: write audio: %w", err)
	}
	s.mu.Lock()
	s.uncommitted = true
	s.mu.Unlock()
	return nil
}

// Events returns transcript revisions.
func (s *STT) Events() <-chan stt.Event { return s.emitter.Events() }

// Close commits whatever audio the server has not transcribed, waits for it, then tears
// the connection down.
func (s *STT) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	conn := s.conn
	owed := s.uncommitted || s.committing
	// Words heard and not yet completed are a tail worth waiting the full timeout for; a
	// call whose last turn is already out is not.
	patience := s.options.FlushTimeout
	if s.finalized == "" && s.intermediate == "" && !s.committing {
		patience = min(flushGrace, s.options.FlushTimeout)
	}
	if s.turn != nil {
		s.turn.Stop()
	}
	s.mu.Unlock()

	if conn != nil && owed {
		s.flush(patience)
	}

	// Hanging up ends the turn whether or not the server answered in time, so the last
	// thing the caller said is not lost to the teardown.
	s.publishWhatIsLeft()

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

// socketURL is the realtime socket under a resource root given as https, wss, http or ws.
func socketURL(endpoint string) (string, error) {
	parsed, err := url.Parse(strings.TrimSpace(endpoint))
	if err != nil {
		return "", stack.Wrap(fmt.Errorf("microsoft: endpoint: %w", err))
	}
	switch parsed.Scheme {
	case "https", "wss":
		parsed.Scheme = "wss"
	case "http", "ws":
		parsed.Scheme = "ws"
	default:
		return "", stack.Wrap(fmt.Errorf("microsoft: endpoint must be an https:// resource root, got %s", endpoint))
	}
	if parsed.Host == "" || strings.Trim(parsed.Path, "/") != "" || parsed.RawQuery != "" {
		return "", stack.Wrap(fmt.Errorf("microsoft: endpoint must be a resource root such as https://<resource>.services.ai.azure.com, got %s", endpoint))
	}
	parsed.Path = realtimePath
	// Transcription rather than a conversation: the realtime socket serves both.
	parsed.RawQuery = url.Values{"intent": []string{"transcription"}}.Encode()
	return parsed.String(), nil
}

// sessionUpdate is the configuration the session runs on. It cannot change once audio
// has been appended, so it is sent once, before any.
func (s *STT) sessionUpdate() clientMessage {
	var language *string
	if s.options.Language != "" {
		language = &s.options.Language
	}
	return clientMessage{
		Type: clientTypeUpdate,
		Session: &sessionConfig{
			Type: "transcription",
			Audio: sessionAudio{Input: sessionInput{
				Format:        audioFormat{Type: "audio/pcm", Rate: stt.SampleRate},
				Transcription: transcriptionConfig{Model: s.options.Deployment, Language: language},
			}},
		},
	}
}

// handshake waits for the session to open, configures it and waits for the server to
// accept the configuration. Audio sent before that is audio the server is not yet
// listening to.
func (s *STT) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return stack.Wrap(fmt.Errorf("microsoft: read handshake: %w", err))
	}

	opening, err := s.readHandshakeFrame()
	if err != nil {
		return err
	}
	if opening.Type != eventSessionCreated {
		return stack.Wrap(fmt.Errorf("microsoft: expected %q, got %q", eventSessionCreated, opening.Type))
	}

	if err := s.send(s.sessionUpdate()); err != nil {
		return stack.Wrap(fmt.Errorf("microsoft: configure session: %w", err))
	}
	for {
		message, err := s.readHandshakeFrame()
		if err != nil {
			return err
		}
		if message.Type == eventSessionUpdated {
			break
		}
		s.logger.Debug("frame before the session was configured", "type", message.Type)
	}

	if err := s.conn.SetReadDeadline(time.Time{}); err != nil {
		return stack.Wrap(fmt.Errorf("microsoft: read handshake: %w", err))
	}
	return nil
}

// readHandshakeFrame reads one frame of the handshake, failing on an error the server sent.
func (s *STT) readHandshakeFrame() (serverMessage, error) {
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return serverMessage{}, stack.Wrap(fmt.Errorf("microsoft: read handshake: %w", err))
	}
	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return serverMessage{}, stack.Wrap(fmt.Errorf("microsoft: decode handshake: %w", err))
	}
	if message.Type == eventError {
		return serverMessage{}, stack.Wrap(fmt.Errorf("microsoft: handshake rejected: %s", message.failure()))
	}
	return message, nil
}

// flush commits the audio the server is holding and waits for its transcript, so the tail
// of a call is not lost. A dead connection must not stop teardown.
func (s *STT) flush(patience time.Duration) {
	// Forget the commits already answered, so the wait below is for one that comes after
	// the audio stopped.
	select {
	case <-s.settled:
	default:
	}

	s.mu.Lock()
	uncommitted := s.uncommitted
	s.mu.Unlock()
	if uncommitted {
		if err := s.sendCommit(); err != nil {
			s.logger.Debug("commit not delivered", "error", err)
			return
		}
	}

	select {
	case <-s.settled:
	case <-time.After(patience):
		s.logger.Debug("timed out waiting for the last words", "patience", patience)
	}
}

// commitTurn asks the server to settle the turn, once the words have stopped changing.
func (s *STT) commitTurn() {
	s.mu.Lock()
	pending := s.conn != nil && !s.closed && !s.committing && strings.TrimSpace(s.finalized+s.intermediate) != ""
	s.mu.Unlock()
	if !pending {
		return
	}

	if err := s.sendCommit(); err != nil {
		s.logger.Debug("commit not delivered", "error", err)
	}
}

func (s *STT) sendCommit() error {
	s.mu.Lock()
	s.committing = true
	s.uncommitted = false
	s.mu.Unlock()

	if err := s.send(clientMessage{Type: clientTypeCommit}); err != nil {
		s.mu.Lock()
		s.committing = false
		s.mu.Unlock()
		return err
	}
	return nil
}

func (s *STT) send(frame clientMessage) error {
	payload, err := json.Marshal(frame)
	if err != nil {
		return stack.Wrap(err)
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	return stack.Wrap(s.conn.WriteMessage(websocket.TextMessage, payload))
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
		s.mu.Lock()
		s.finalized += message.Delta
		// The suffix was relative to the delta before this one, which this has overtaken.
		s.intermediate = ""
		s.mu.Unlock()
		s.heard()
	case eventIntermediate:
		s.mu.Lock()
		s.intermediate = message.Intermediate
		s.mu.Unlock()
		s.heard()
	case eventCompleted:
		s.settle(message.Transcript)
	case eventFailed:
		// One commit the server could not transcribe. The session carries on, so this is
		// not fatal, but the words are gone and the caller is owed an answer anyway.
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  "transcription",
		})
		s.forgetTurn()
		s.reachedABoundary()
	case eventError:
		s.emitter.Send(stt.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      errors.New(message.failure()),
			Context:  "server",
			Fatal:    true,
		})
	case eventSessionCreated, eventSessionUpdated, eventCommitted:
		// The handshake already waited for the first two, and a commit is answered by the
		// completed transcript that follows it.
	default:
		s.logger.Debug("unhandled frame", "type", message.Type)
	}
}

// heard reports the best reading of the turn so far, which is worth showing at once: it
// arrives while the caller is still talking. The join is not trimmed inside, because the
// server's own spacing is what holds the deltas together.
func (s *STT) heard() {
	s.mu.Lock()
	whole := strings.TrimSpace(s.finalized + s.intermediate)
	changed := whole != s.shown
	s.shown = whole
	s.mu.Unlock()

	if whole == "" || !changed {
		return
	}

	s.sendTranscript(stt.ModeReplacement, whole)
	s.expectMore()
}

// settle publishes the transcript of a commit as the turn.
func (s *STT) settle(transcript string) {
	s.forgetTurn()

	if text := strings.TrimSpace(transcript); text != "" {
		s.sendTranscript(stt.ModeFinal, text)
		s.endUtterance()
	}
	// Nothing said in the committed audio is still an answer to a Close waiting on it.
	s.reachedABoundary()
}

// publishWhatIsLeft settles the words a hangup cut off before the server completed them.
func (s *STT) publishWhatIsLeft() {
	s.mu.Lock()
	whole := strings.TrimSpace(s.finalized + s.intermediate)
	s.mu.Unlock()
	s.forgetTurn()

	if whole == "" {
		return
	}
	s.sendTranscript(stt.ModeFinal, whole)
	s.endUtterance()
}

// forgetTurn clears the turn in progress once it is settled or lost.
func (s *STT) forgetTurn() {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.finalized = ""
	s.intermediate = ""
	s.shown = ""
	s.committing = false
	if s.turn != nil {
		s.turn.Stop()
	}
}

// expectMore restarts the clock on the turn, because a caller whose words are still
// changing has not finished saying them.
func (s *STT) expectMore() {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return
	}
	if s.turn == nil {
		s.turn = time.AfterFunc(s.options.TurnGrace, s.commitTurn)
		return
	}
	s.turn.Reset(s.options.TurnGrace)
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

// reachedABoundary tells a waiting Close that the server has answered a commit.
func (s *STT) reachedABoundary() {
	select {
	case s.settled <- struct{}{}:
	default:
	}
}

// endUtterance marks the current run of speech as over.
//
// The count moves on the next transcript rather than here, so a final and the hypothesis
// that opens the next turn are one boundary between utterances rather than two.
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

// failure is what the server said went wrong.
func (m serverMessage) failure() string {
	if m.Error != nil && m.Error.Message != "" {
		return m.Error.Message
	}
	return "unknown error"
}
