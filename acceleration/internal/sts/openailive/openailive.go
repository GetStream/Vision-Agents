// Package openailive implements the sts.STS contract over OpenAI's GPT-Live API, the
// full-duplex successor to the Realtime API, at wss://api.openai.com/v1/live/sessions.
//
// It shares no code with openairealtime because it shares no events with it. A Live session
// is configured once with session.start; the caller's audio goes up in
// session.input_audio.append and the model's comes back in session.output_audio.delta; and
// there is no response, no commit and no response.done. The model listens and speaks at
// the same time and decides for itself when to do which.
//
// A reply is therefore whatever the model says between two silences, numbered here: one
// opens with the first audio after a quiet spell and settles once the audio has stopped for
// ReplyGap. The caller is taken to have stopped at the last piece of transcript before the
// model began, and a reply the caller was heard over is reported as interrupted, since the
// API has no word for either.
//
// Reasoning and tools are delegated to a backend Responses model the session names. Its
// function calls arrive wrapped in response.event envelopes and their results go back as
// response.item.create followed by response.create, which is the one thing that makes the
// backend carry on. Speech does not wait for the backend: the model can keep talking while a
// tool runs.
package openailive

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats. It is OpenAI's, shared
// with the Realtime models, because it is the same vendor on the same key.
const ProviderName = "openai"

// DefaultModel is the only Live model in the API.
const DefaultModel = "gpt-live-1"

// DefaultBackend is the Responses model delegated work goes to when nobody named one, which
// is the one OpenAI's delegation guide says to start with.
const DefaultBackend = "gpt-5.6-terra"

// DefaultURL is the Live socket. It takes no query parameters: the model goes in the
// session.start frame.
const DefaultURL = "wss://api.openai.com/v1/live/sessions"

// SampleRate is the rate the session is configured at. One format applies both ways, so
// the caller's audio is resampled up to it rather than the model's down.
const SampleRate = 24_000

// DefaultReplyGap is how long the model's audio has to stop before its reply is settled.
const DefaultReplyGap = 800 * time.Millisecond

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "OPENAI_API_KEY"

// modelPrefix is what every Live conversation model is called, and what tells the openai
// factory to build this package rather than openairealtime.
const modelPrefix = "gpt-live-1"

// Client event types.
const (
	eventSessionStart      = "session.start"
	eventSessionUpdate     = "session.update"
	eventSessionClose      = "session.close"
	eventAudioAppend       = "session.input_audio.append"
	eventInstructionAppend = "session.instructions.append"
	eventItemCreate        = "response.item.create"
	eventResponseCreate    = "response.create"
)

// Server event types.
const (
	eventSessionStarted    = "session.started"
	eventSessionClosed     = "session.closed"
	eventAudioDelta        = "session.output_audio.delta"
	eventInputTranscript   = "session.input_transcript.delta"
	eventOutputTranscript  = "session.output_transcript.delta"
	eventDelegationCreated = "session.delegation.created"
	eventResponseEvent     = "response.event"
	eventError             = "error"
)

// Nested Responses event types, inside a response.event envelope.
const (
	nestedOutputItemDone = "response.output_item.done"
	nestedFailed         = "response.failed"
	nestedError          = "error"
)

// closedByConnectionLoss is the one session.closed reason that is not somebody's decision.
const closedByConnectionLoss = "connection_lost"

// Options configures the provider. APIKey falls back to OPENAI_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Voice is one of the Live voices, such as marin or quartz. Empty leaves marin.
	Voice string
	// Instructions go to both the live model and the backend: the one decides what to
	// hand over and the other what to do with it. The live model's are fixed at startup.
	Instructions string
	// Tools are what the backend may call. They can be replaced mid-session.
	Tools []llm.Tool
	// Backend is the Responses model delegated work goes to. Empty is DefaultBackend.
	Backend string
	// ReplyGap is how long the model's audio has to stop before its reply is settled.
	ReplyGap time.Duration
	// HandshakeTimeout bounds the connect and the session.start exchange.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger
}

// clientEvent is a frame sent to the Live API.
type clientEvent struct {
	Type    string         `json:"type"`
	Session *sessionConfig `json:"session,omitempty"`
	Audio   string         `json:"audio,omitempty"`
	Item    *item          `json:"item,omitempty"`
}

// appendEvent adds context to the live model. delegation_id is required even when null,
// which is what names the session rather than a delegated task.
type appendEvent struct {
	Type         string  `json:"type"`
	DelegationID *string `json:"delegation_id"`
	Content      string  `json:"content"`
}

// sessionConfig is the session.start configuration, and the part of it session.update can
// change. The API rejects unknown fields, so nothing here is speculative.
type sessionConfig struct {
	Model        string       `json:"model,omitempty"`
	Instructions string       `json:"instructions,omitempty"`
	Audio        *audioConfig `json:"audio,omitempty"`
	Delegation   *delegation  `json:"delegation,omitempty"`
}

type audioConfig struct {
	Format audioFormat  `json:"format"`
	Output *audioOutput `json:"output,omitempty"`
}

type audioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate"`
}

type audioOutput struct {
	Voice string `json:"voice"`
}

type delegation struct {
	Type      string   `json:"type"`
	Responses *backend `json:"responses,omitempty"`
}

// backend is the Responses configuration delegated work runs under. session.update replaces
// it whole, so it is always sent complete.
type backend struct {
	Model        string `json:"model"`
	Instructions string `json:"instructions,omitempty"`
	Tools        []tool `json:"tools,omitempty"`
	ToolChoice   string `json:"tool_choice,omitempty"`
	// ParallelToolCalls is off so each backend response waits on one answer, which is what
	// lets Answer continue it without counting outstanding calls.
	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`
}

// tool is a function in the Responses shape, which is flat rather than nested.
type tool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// item is a Responses input item: a tool's result or a typed message.
type item struct {
	Type    string    `json:"type"`
	CallID  string    `json:"call_id,omitempty"`
	Output  string    `json:"output,omitempty"`
	Role    string    `json:"role,omitempty"`
	Content []content `json:"content,omitempty"`
}

type content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// serverEvent is a frame sent by the Live API. Only the fields this package reads are here.
type serverEvent struct {
	Type         string          `json:"type"`
	Delta        string          `json:"delta"`
	Reason       string          `json:"reason"`
	DelegationID string          `json:"delegation_id"`
	Event        json.RawMessage `json:"event"`
	Error        *apiError       `json:"error"`
}

// nestedEvent is the Responses event inside a response.event envelope.
type nestedEvent struct {
	Type string      `json:"type"`
	Item *outputItem `json:"item"`
}

type outputItem struct {
	Type      string `json:"type"`
	CallID    string `json:"call_id"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type apiError struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
	Param   string `json:"param"`
}

func (e *apiError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

// STS is a GPT-Live conversation.
type STS struct {
	options Options
	logger  *slog.Logger
	emitter *sts.Emitter

	mu   sync.Mutex
	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// tools is what the backend was last given, which session.update has to send whole.
	tools []llm.Tool
	// participant is the speaker of the most recent audio.
	participant sts.Participant
	// hearing is what the model has written down of the caller since its last reply, and
	// lastHeardAt when the last piece arrived, which stands in for the moment the caller
	// stopped since this API never says.
	hearing     strings.Builder
	listening   bool
	lastHeardAt time.Time
	heardAt     time.Time
	// turn is the reply in flight, generation counts them, and quiet is the timer that
	// settles it once the model stops talking. overheard is whether the caller was heard
	// while it spoke.
	turn       *sts.Turn
	generation int
	quiet      *time.Timer
	overheard  bool
	// muted drops the rest of a reply the caller cut off from this side, since the API
	// has no way to tell the model to stop.
	muted   bool
	started bool
	closed  bool
	// ended is set once the server reported the session closed, so the transport closing
	// after it is not reported a second time.
	ended bool
}

// New validates the options and returns an unstarted provider.
func New(settings Options) (*STS, error) {
	if settings.APIKey == "" {
		settings.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if settings.APIKey == "" {
		return nil, fmt.Errorf("openailive: api key is required (set %s)", apiKeyEnvVar)
	}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if !Serves(settings.Model) {
		return nil, fmt.Errorf("openailive: %s is not a GPT-Live model", settings.Model)
	}
	if settings.URL == "" {
		settings.URL = DefaultURL
	}
	if !strings.HasPrefix(settings.URL, "ws://") && !strings.HasPrefix(settings.URL, "wss://") {
		return nil, fmt.Errorf("openailive: url must be ws:// or wss://, got %s", settings.URL)
	}
	if settings.Backend == "" {
		settings.Backend = DefaultBackend
	}
	if settings.ReplyGap == 0 {
		settings.ReplyGap = DefaultReplyGap
	}
	if settings.HandshakeTimeout == 0 {
		settings.HandshakeTimeout = 30 * time.Second
	}
	logger := settings.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STS{
		options: settings,
		logger:  logger.With("provider", ProviderName, "model", settings.Model),
		emitter: sts.NewEmitter(sts.EmitterBuffer),
		tools:   settings.Tools,
	}, nil
}

// Serves reports whether a model id is a Live conversation model, which is how the openai
// provider decides between this package and the Realtime one.
func Serves(model string) bool { return strings.HasPrefix(model, modelPrefix) }

// CapabilitiesFor is what a Live model can be asked for.
//
// It hears no images and has no turn detector to tune: listening and speaking are the
// model's own business, so neither endpointing nor semantic_turns is anything this package
// could send. Its instructions are fixed at startup; the backend's tools are not. Usage is
// session seconds rather than tokens per reply, so it is not reported here, and OpenAI has
// published no session limit.
func CapabilitiesFor(string) sts.Capabilities {
	return sts.Capabilities{
		Text:             true,
		Tools:            true,
		InputTranscript:  true,
		OutputTranscript: true,
		ToolsMidSession:  true,
	}
}

// Start dials the Live API and sends session.start. It returns once the server reports the
// session started, since audio sent before that is refused.
func (s *STS) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("openailive: already started")
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Bearer " + s.options.APIKey}}
	conn, response, err := dialer.DialContext(ctx, s.options.URL, header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("openailive: dial: %w (http %d)", err, response.StatusCode)
		}
		return fmt.Errorf("openailive: dial: %w", err)
	}
	if err := s.handshake(conn); err != nil {
		conn.Close()
		return err
	}

	s.mu.Lock()
	s.conn = conn
	s.mu.Unlock()

	s.emitter.Send(sts.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop(conn)
	return nil
}

// ProcessAudio streams one chunk of the caller's speech.
func (s *STS) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	if err := pcm.Validate(sts.InputSampleRate); err != nil {
		return fmt.Errorf("openailive: %w", err)
	}

	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()

	pcm = audio.Resample(pcm, SampleRate, 1)
	return s.send(clientEvent{Type: eventAudioAppend, Audio: base64.StdEncoding.EncodeToString(pcm.Bytes())})
}

// SendText hands a typed turn to the backend and runs it. That is where OpenAI sends typed
// input: the live model hears the backend's answer and speaks it.
func (s *STS) SendText(text string, participant sts.Participant) error {
	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()

	if err := s.send(clientEvent{Type: eventItemCreate, Item: &item{
		Type:    "message",
		Role:    "user",
		Content: []content{{Type: "input_text", Text: text}},
	}}); err != nil {
		return err
	}
	return s.send(clientEvent{Type: eventResponseCreate})
}

// SendFrame is refused: the live model hears and speaks only.
func (s *STS) SendFrame(llm.ImagePart) error { return sts.ErrNoImages }

// SetInstructions is refused. The live model's instructions are fixed at startup, and what
// the API offers instead appends to them, which is not the replacement this asks for.
func (s *STS) SetInstructions(string) error { return sts.ErrInstructionsFixed }

// SetTools replaces what the backend may call. The delegation is replaced whole, so the
// rest of it is sent again unchanged.
func (s *STS) SetTools(tools []llm.Tool) error {
	s.mu.Lock()
	s.tools = tools
	s.mu.Unlock()
	return s.send(clientEvent{Type: eventSessionUpdate, Session: &sessionConfig{Delegation: s.delegation(tools)}})
}

// Answer returns a tool's result and continues the backend response, which does not carry
// on by itself.
func (s *STS) Answer(callID string, output string, err error) error {
	if err != nil {
		output = "Error: " + err.Error()
	}
	if sendErr := s.send(clientEvent{Type: eventItemCreate, Item: &item{
		Type:   "function_call_output",
		CallID: callID,
		Output: output,
	}}); sendErr != nil {
		return sendErr
	}
	return s.send(clientEvent{Type: eventResponseCreate})
}

// Prompt asks the model to speak now, guided by the text. It is an appended instruction,
// which is how OpenAI has a Live session greet before the caller speaks, with the part that
// says not to wait spelled out because the model otherwise waits.
func (s *STS) Prompt(text string) error {
	return s.send(appendEvent{
		Type:    eventInstructionAppend,
		Content: text + " Do this now, without waiting for the caller, then listen.",
	})
}

// Interrupt stops forwarding the reply in flight and settles it as cut off. The API has no
// client-side interrupt, so the model is not told: it stops of its own accord when it hears
// the caller, which is what a barge-in sounds like to it anyway.
func (s *STS) Interrupt(int) error {
	s.mu.Lock()
	if s.turn != nil {
		s.muted = true
	}
	s.mu.Unlock()
	s.settleTurn(true)
	return nil
}

// Events returns what the model heard, said and asked for.
func (s *STS) Events() <-chan sts.Event { return s.emitter.Events() }

// Close ends the session, asking the server to finish it first so the seconds it bills
// stop there. A reply still in flight is settled as interrupted, so it is billed once like
// every other.
func (s *STS) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	conn := s.conn
	s.mu.Unlock()

	if conn != nil {
		if err := s.send(clientEvent{Type: eventSessionClose}); err != nil {
			s.logger.Debug("could not close the session gracefully", "error", err)
		}
	}

	s.mu.Lock()
	s.closed = true
	s.mu.Unlock()

	s.settleTurn(true)
	if conn != nil {
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements sts.STS.
func (s *STS) Provider() string { return ProviderName }

// Model implements sts.STS.
func (s *STS) Model() string { return s.options.Model }

// Voice is the voice asked for, or empty when the vendor picks it.
func (s *STS) Voice() string { return s.options.Voice }

// SampleRate is the rate the model speaks at.
func (s *STS) SampleRate() int { return SampleRate }

// Capabilities implements sts.STS.
func (s *STS) Capabilities() sts.Capabilities { return CapabilitiesFor(s.options.Model) }

// handshake sends session.start and waits for session.started. Anything the server rejects
// arrives as an error frame instead.
func (s *STS) handshake(conn *websocket.Conn) error {
	payload, err := json.Marshal(clientEvent{Type: eventSessionStart, Session: s.session()})
	if err != nil {
		return err
	}
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("openailive: send session.start: %w", err)
	}

	if err := conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("openailive: read session.started: %w", err)
	}
	defer conn.SetReadDeadline(time.Time{})

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return fmt.Errorf("openailive: read session.started: %w", err)
		}
		var event serverEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			return fmt.Errorf("openailive: decode session.started: %w", err)
		}
		switch event.Type {
		case eventSessionStarted:
			return nil
		case eventError:
			if event.Error == nil {
				return fmt.Errorf("openailive: session rejected: %s", strings.TrimSpace(string(raw)))
			}
			return fmt.Errorf("openailive: session rejected: %w", event.Error)
		}
	}
}

// session is the session.start configuration: the model, its voice and instructions, audio
// both ways at SampleRate, and the backend delegated work goes to.
func (s *STS) session() *sessionConfig {
	configured := &sessionConfig{
		Model:        s.options.Model,
		Instructions: s.options.Instructions,
		Audio:        &audioConfig{Format: audioFormat{Type: "audio/pcm", Rate: SampleRate}},
		Delegation:   s.delegation(s.options.Tools),
	}
	if s.options.Voice != "" {
		configured.Audio.Output = &audioOutput{Voice: s.options.Voice}
	}
	return configured
}

// delegation is the Responses backend with the given tools.
func (s *STS) delegation(tools []llm.Tool) *delegation {
	configured := &backend{Model: s.options.Backend, Instructions: s.options.Instructions}
	if len(tools) > 0 {
		parallel := false
		configured.ToolChoice = "auto"
		configured.ParallelToolCalls = &parallel
		for _, offered := range tools {
			configured.Tools = append(configured.Tools, tool{
				Type:        "function",
				Name:        offered.Name,
				Description: offered.Description,
				Parameters:  offered.Parameters,
			})
		}
	}
	return &delegation{Type: "responses", Responses: configured}
}

func (s *STS) send(frame any) error {
	s.mu.Lock()
	conn, started, closed := s.conn, s.started, s.closed
	s.mu.Unlock()
	if closed {
		return errors.New("openailive: session closed")
	}
	if !started || conn == nil {
		return errors.New("openailive: not started")
	}

	payload, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("openailive: write: %w", err)
	}
	return nil
}

// readLoop translates server frames into events until the connection ends.
func (s *STS) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			s.handleReadError(err)
			return
		}

		var event serverEvent
		if err := json.Unmarshal(raw, &event); err != nil {
			s.logger.Debug("undecodable frame", "error", err, "payload", string(raw))
			continue
		}
		s.handleMessage(event)
	}
}

// handleReadError reports the connection ending, unless this side or a session.closed
// already accounted for it. A reply cut off by it is settled, because the rest of its audio
// is not coming.
func (s *STS) handleReadError(err error) {
	s.mu.Lock()
	closed, ended := s.closed, s.ended
	s.mu.Unlock()
	if closed || ended {
		return
	}

	s.settleTurn(true)
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(sts.Disconnected{Provider: ProviderName, Model: s.options.Model, Clean: true, At: time.Now()})
		return
	}
	s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: err, Context: "read", Fatal: true})
}

func (s *STS) handleMessage(event serverEvent) {
	switch event.Type {
	case eventAudioDelta:
		raw, err := base64.StdEncoding.DecodeString(event.Delta)
		if err != nil {
			s.logger.Debug("undecodable audio", "error", err)
			return
		}
		s.spoke(raw)
	case eventInputTranscript:
		if event.Delta != "" {
			s.heard(event.Delta)
		}
	case eventOutputTranscript:
		if event.Delta != "" {
			turn := s.currentTurn()
			s.emitter.Send(sts.OutputTranscript{ResponseID: turn.ID, Mode: stt.ModeDelta, Text: event.Delta})
		}
	case eventResponseEvent:
		s.delegated(event)
	case eventDelegationCreated:
		s.logger.Debug("the model delegated work")
	case eventSessionClosed:
		s.mu.Lock()
		s.ended = true
		s.mu.Unlock()
		s.settleTurn(true)
		s.emitter.Send(sts.Disconnected{
			Provider: ProviderName,
			Model:    s.options.Model,
			Reason:   event.Reason,
			Clean:    event.Reason != closedByConnectionLoss,
			At:       time.Now(),
		})
	case eventError:
		failure := errors.New("openailive: unknown error")
		if event.Error != nil {
			failure = event.Error
		}
		s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: failure, Context: "session"})
	}
}

// delegated handles an event of the backend's. A finished function call is forwarded as a
// tool call; a failure is reported; the rest is the backend's own bookkeeping.
func (s *STS) delegated(envelope serverEvent) {
	var nested nestedEvent
	if err := json.Unmarshal(envelope.Event, &nested); err != nil {
		s.logger.Debug("undecodable delegated event", "error", err)
		return
	}

	switch nested.Type {
	case nestedOutputItemDone:
		if nested.Item == nil || nested.Item.Type != "function_call" {
			return
		}
		arguments := nested.Item.Arguments
		if arguments == "" {
			arguments = "{}"
		}
		s.mu.Lock()
		var responseID string
		if s.turn != nil {
			responseID = s.turn.ID
		}
		s.mu.Unlock()
		s.emitter.Send(sts.ToolCall{ResponseID: responseID, CallID: nested.Item.CallID, Name: nested.Item.Name, Arguments: arguments})
	case nestedFailed, nestedError:
		s.emitter.Send(sts.Error{
			Provider: ProviderName,
			Model:    s.options.Model,
			Err:      fmt.Errorf("openailive: delegated work failed: %s", envelope.Event),
			Context:  "delegation",
		})
	}
}

// heard is a piece of what the caller is saying. The first piece since the model last
// spoke is also the only word this API gives that the caller started talking.
func (s *STS) heard(text string) {
	s.mu.Lock()
	first := !s.listening
	s.listening = true
	s.lastHeardAt = time.Now()
	s.hearing.WriteString(text)
	if s.turn != nil {
		s.overheard = true
	}
	participant := s.participant
	s.mu.Unlock()

	if first {
		s.emitter.Send(sts.SpeechStarted{Participant: participant, At: time.Now()})
	}
	s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: stt.ModeDelta, Text: text})
}

// finishHearing settles the caller's turn: the transcript so far becomes the final one, and
// the last piece of it is taken for the moment they stopped, since nothing else says.
func (s *STS) finishHearing() {
	s.mu.Lock()
	if !s.listening {
		s.mu.Unlock()
		return
	}
	s.listening = false
	text := s.hearing.String()
	s.hearing.Reset()
	stoppedAt := s.lastHeardAt
	s.heardAt = stoppedAt
	participant := s.participant
	s.mu.Unlock()

	if text != "" {
		s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: stt.ModeFinal, Text: text})
	}
	s.emitter.Send(sts.SpeechStopped{Participant: participant, At: stoppedAt})
}

// spoke forwards a piece of the model's speech, opening a reply if none is in flight.
func (s *STS) spoke(raw []byte) {
	s.mu.Lock()
	opening := s.turn == nil
	muted := s.muted
	s.mu.Unlock()

	// The rest of a reply the caller cut off from this side is not a new reply, so it is
	// dropped before it can open one. The quiet after it is what ends the mute.
	if muted {
		s.keepAlive(nil)
		return
	}
	if opening {
		s.finishHearing()
	}
	turn := s.currentTurn()
	s.emitter.Send(turn.Chunk(audio.FromBytes(raw, SampleRate, 1)))
}

// currentTurn is the reply in flight, opened now if the model has begun one, which is how
// this API begins every reply. Anything that belongs to the reply keeps it open.
func (s *STS) currentTurn() *sts.Turn {
	s.mu.Lock()
	if s.turn != nil {
		turn := s.turn
		s.mu.Unlock()
		s.keepAlive(turn)
		return turn
	}
	s.generation++
	turn := sts.NewTurn("", s.generation, s.heardAt)
	s.heardAt = time.Time{}
	s.turn = turn
	s.mu.Unlock()

	s.emitter.Send(sts.ResponseStarted{ResponseID: turn.ID, Generation: turn.Generation, At: time.Now()})
	s.keepAlive(turn)
	return turn
}

// keepAlive restarts the quiet timer for the reply in flight, or for a muted remainder when
// turn is nil.
func (s *STS) keepAlive(turn *sts.Turn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.quiet != nil {
		s.quiet.Stop()
	}
	s.quiet = time.AfterFunc(s.options.ReplyGap, func() { s.fellQuiet(turn) })
}

// fellQuiet ends the reply the model stopped talking in, and any mute on a reply cut off
// here, since whatever was being dropped has now all arrived.
func (s *STS) fellQuiet(turn *sts.Turn) {
	s.mu.Lock()
	s.muted = false
	current, overheard := s.turn, s.overheard
	s.mu.Unlock()
	if turn == nil || current != turn {
		return
	}
	s.settleTurn(overheard)
}

// settleTurn closes the reply in flight, if there is one.
func (s *STS) settleTurn(interrupted bool) {
	s.mu.Lock()
	turn := s.turn
	s.turn = nil
	s.overheard = false
	s.mu.Unlock()
	if turn == nil {
		return
	}
	s.emitter.Send(turn.Complete(ProviderName, s.options.Model, interrupted, sts.Usage{}))
}
