// Package phonic implements the sts.STS contract over Phonic's speech-to-speech socket at
// wss://api.phonic.ai/v1/sts/ws, which is its own protocol rather than OpenAI's events.
//
// A session is one config frame answered by ready_to_start_conversation. The caller's audio
// goes up in audio_chunk frames of at most 40 ms, and the model's comes back in audio_chunk
// frames carrying the words they speak. Phonic says where each turn begins and ends:
// user_started_speaking and user_finished_speaking for the caller, assistant_started_speaking
// and assistant_finished_speaking for the model. The model's audio is asked for ahead of real
// time, but assistant_finished_speaking keeps Phonic's own playback clock, so it comes when
// the listener would have heard the last of it.
//
// A barge-in arrives as assistant_finished_speaking followed at once by interrupted_response,
// so a reply that finished is settled only once a moment has passed without the second. The
// API has no way for this side to stop a reply: Interrupt drops the rest of it here, and the
// model stops when it hears the caller.
//
// Phonic bills the minutes of a conversation and reports neither tokens nor time, so the
// session's seconds are counted here from the moment Phonic took the config.
package phonic

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "phonic"

// DefaultModel is Phonic's current model, named in the config frame.
const DefaultModel = "phonic-1"

// DefaultURL is the conversation socket. The model goes in the config frame, not on the URL.
const DefaultURL = "wss://api.phonic.ai/v1/sts/ws"

// SampleRate is the rate the model is asked to speak at. The caller's audio goes up at
// sts.InputSampleRate, which Phonic takes as it is.
const SampleRate = 24_000

// Audio formats, in Phonic's spelling.
const (
	inputFormat  = "pcm_16000"
	outputFormat = "pcm_24000"
)

// MaxDuration is the longest Phonic lets a conversation run, which it ends with close code
// 4505.
const MaxDuration = 2 * time.Hour

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "PHONIC_API_KEY"

// chunkMs is how much audio goes up in one frame. Phonic refuses a frame over 40 ms and
// suggests 20 for headroom.
const chunkMs = 20

// interruptGrace is how long after assistant_finished_speaking an interrupted_response may
// still say the reply was cut off. It came 5 ms after when measured.
const interruptGrace = 300 * time.Millisecond

// Limits Phonic would otherwise end a call with: 5 s for a tool's answer, a minute without a
// frame, three minutes without the caller saying anything. Each is set to the most Phonic
// takes, since a tool may wait on the caller and a call may sit quiet.
const (
	toolTimeoutMs      = 180_000
	socketTimeoutSec   = 300
	silenceEndsCallSec = 3600
)

// Client frame types.
const (
	frameConfig       = "config"
	frameAudio        = "audio_chunk"
	frameSystemPrompt = "update_system_prompt"
	frameTools        = "update_tools_subset"
	frameToolOutput   = "tool_call_output"
	frameReply        = "generate_reply"
)

// Server frame types.
const (
	frameReady             = "ready_to_start_conversation"
	frameCreated           = "conversation_created"
	frameInputText         = "input_text"
	frameInputCancelled    = "input_cancelled"
	frameUserStarted       = "user_started_speaking"
	frameUserFinished      = "user_finished_speaking"
	frameAssistantStarted  = "assistant_started_speaking"
	frameAssistantFinished = "assistant_finished_speaking"
	frameInterrupted       = "interrupted_response"
	frameToolCall          = "tool_call"
	frameToolInterrupted   = "tool_call_interrupted"
	frameChoseNotToRespond = "assistant_chose_not_to_respond"
	frameEndedConversation = "assistant_ended_conversation"
	frameError             = "error"
	frameWarning           = "warning"
)

// reasonEndedConversation is the Disconnected reason when the model hung up.
const reasonEndedConversation = "assistant_ended_conversation"

// Options configures the provider. APIKey falls back to PHONIC_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Voice is a Phonic voice id, such as sabrina or grant. Empty leaves Phonic's default.
	Voice        string
	Instructions string
	Tools        []llm.Tool
	// Languages are what the caller may speak, the first the one the agent starts in. Empty
	// leaves Phonic's default, English.
	Languages []string
	// HandshakeTimeout bounds the connect and the config exchange.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger
}

// config is the first frame, which Phonic validates strictly: an unknown field closes the
// socket, so nothing here is speculative.
type config struct {
	Type                string   `json:"type"`
	Model               string   `json:"model"`
	SystemPrompt        string   `json:"system_prompt,omitempty"`
	VoiceID             string   `json:"voice_id,omitempty"`
	InputFormat         string   `json:"input_format"`
	OutputFormat        string   `json:"output_format"`
	StreamAhead         bool     `json:"stream_ahead_of_real_time"`
	WebsocketTimeoutSec int      `json:"websocket_timeout_sec"`
	NoInputEndSec       int      `json:"no_input_end_conversation_sec"`
	DefaultLanguage     string   `json:"default_language,omitempty"`
	AdditionalLanguages []string `json:"additional_languages,omitempty"`
	Tools               []tool   `json:"tools,omitempty"`
}

// tool is an inline WebSocket tool: Phonic sends tool_call and waits for tool_call_output.
type tool struct {
	Type          string     `json:"type"`
	Schema        toolSchema `json:"tool_schema"`
	OutputTimeout int        `json:"tool_call_output_timeout_ms"`
}

type toolSchema struct {
	Type     string   `json:"type"`
	Function function `json:"function"`
}

// function is OpenAI's function shape. Phonic takes inline tools only as strict.
type function struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters"`
	Strict      bool           `json:"strict"`
}

// clientFrame is every frame after the config. Only the fields its type takes are set.
type clientFrame struct {
	Type          string  `json:"type"`
	Audio         string  `json:"audio,omitempty"`
	SystemPrompt  string  `json:"system_prompt,omitempty"`
	Tools         *[]tool `json:"tools,omitempty"`
	ToolCallID    string  `json:"tool_call_id,omitempty"`
	Output        *string `json:"output,omitempty"`
	SystemMessage string  `json:"system_message,omitempty"`
	UserMessage   string  `json:"user_message,omitempty"`
}

// serverFrame is a frame Phonic sends. Only the fields this package reads are here.
type serverFrame struct {
	Type           string          `json:"type"`
	ConversationID string          `json:"conversation_id"`
	Text           string          `json:"text"`
	Language       string          `json:"language"`
	Audio          string          `json:"audio"`
	ToolCallID     string          `json:"tool_call_id"`
	ToolName       string          `json:"tool_name"`
	Parameters     json.RawMessage `json:"parameters"`
	Error          *apiError       `json:"error"`
	Warning        *apiError       `json:"warning"`
	ParamErrors    json.RawMessage `json:"param_errors"`
}

type apiError struct {
	Message string `json:"message"`
	Code    string `json:"code"`
}

func (e *apiError) Error() string {
	if e.Code != "" {
		return fmt.Sprintf("%s: %s", e.Code, e.Message)
	}
	return e.Message
}

// STS is a Phonic conversation.
type STS struct {
	options Options
	keys    *keypool.Pool
	logger  *slog.Logger
	emitter *sts.Emitter

	mu   sync.Mutex
	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// openedAt is when Phonic took the config, which the seconds it bills are counted from.
	openedAt time.Time
	// participant is the speaker of the most recent audio.
	participant sts.Participant
	// hearing is the words Phonic has written down of the caller's turn so far, and
	// listening whether one is open. heardAt is when the caller last stopped, for the reply
	// that answers them.
	hearing   []string
	listening bool
	heardAt   time.Time
	// typed is the turns sent as text, which Phonic echoes back as input_text: they are not
	// the caller speaking.
	typed []string
	// turn is the reply in flight and generation counts them. finished is a reply whose
	// speech has ended, held for interruptGrace in case it was cut off, and finishing the
	// timer that settles it.
	turn       *sts.Turn
	generation int
	finished   *sts.Turn
	finishing  *time.Timer
	// muted drops the rest of a reply cut off from this side until the next one starts.
	muted   bool
	started bool
	closed  bool
	// ended is set once the model ended the conversation, so the socket closing after it is
	// not reported as a failure.
	ended bool
}

// New validates the options and returns an unstarted provider.
func New(settings Options) (*STS, error) {
	keys := keypool.FromEnv(settings.APIKey, apiKeyEnvVar).Pin()
	if keys.Len() == 0 {
		return nil, stack.Wrap(fmt.Errorf("phonic: api key is required (set %s)", apiKeyEnvVar))
	}
	settings.APIKey = keys.Peek()
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.URL == "" {
		settings.URL = DefaultURL
	}
	if !strings.HasPrefix(settings.URL, "ws://") && !strings.HasPrefix(settings.URL, "wss://") {
		return nil, stack.Wrap(fmt.Errorf("phonic: url must be ws:// or wss://, got %s", settings.URL))
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
		keys:    keys,
		logger:  logger.With("provider", ProviderName, "model", settings.Model),
		emitter: sts.NewEmitter(sts.EmitterBuffer),
	}, nil
}

// CapabilitiesFor is what a Phonic model can be asked for.
//
// It hears no images. Its turn taking is a named patience rather than a silence timer, and
// its VAD settings are not documented to end a turn, so neither endpointing nor
// semantic_turns is declared. It stops speaking when the caller does, so it is not full
// duplex. Instructions and tools both change mid-session. It reports no usage: the session's
// minutes are the bill.
func CapabilitiesFor(string) sts.Capabilities {
	return sts.Capabilities{
		Text:                   true,
		Tools:                  true,
		InputTranscript:        true,
		OutputTranscript:       true,
		InstructionsMidSession: true,
		ToolsMidSession:        true,
		SessionBilled:          true,
		MaxDuration:            MaxDuration,
	}
}

// Start dials Phonic and sends the config. It returns once Phonic is ready for audio.
func (s *STS) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return stack.Wrap(errors.New("phonic: already started"))
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	conn, response, _, err := s.keys.Dial(ctx, dialer, func(key string) (string, http.Header) {
		return s.options.URL, http.Header{"Authorization": []string{"Bearer " + key}}
	})
	if err != nil {
		if response != nil {
			return stack.Wrap(fmt.Errorf("phonic: dial: %w (http %d)", err, response.StatusCode))
		}
		return stack.Wrap(fmt.Errorf("phonic: dial: %w", err))
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()
	if err := s.handshake(conn); err != nil {
		conn.Close()
		return err
	}

	s.mu.Lock()
	s.conn = conn
	s.openedAt = time.Now()
	s.mu.Unlock()

	s.emitter.Send(sts.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop(conn)
	return nil
}

// ProcessAudio streams one chunk of the caller's speech, in pieces Phonic accepts.
func (s *STS) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	if err := pcm.Validate(sts.InputSampleRate); err != nil {
		return fmt.Errorf("phonic: %w", err)
	}
	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()

	raw := pcm.Bytes()
	step := sts.InputSampleRate * chunkMs / 1000 * 2
	for at := 0; at < len(raw); at += step {
		piece := raw[at:min(at+step, len(raw))]
		if err := s.send(clientFrame{Type: frameAudio, Audio: base64.StdEncoding.EncodeToString(piece)}); err != nil {
			return err
		}
	}
	return nil
}

// SendText asks for a reply to a typed turn.
func (s *STS) SendText(text string, participant sts.Participant) error {
	s.mu.Lock()
	s.participant = participant
	s.typed = append(s.typed, text)
	s.mu.Unlock()
	return s.send(clientFrame{Type: frameReply, UserMessage: text})
}

// SendFrame is refused: Phonic hears and speaks only.
func (s *STS) SendFrame(llm.ImagePart) error { return sts.ErrNoImages }

// SetInstructions replaces the system prompt for the replies that follow.
func (s *STS) SetInstructions(text string) error {
	return s.send(clientFrame{Type: frameSystemPrompt, SystemPrompt: text})
}

// SetTools replaces what the model may call.
func (s *STS) SetTools(tools []llm.Tool) error {
	inline := toolsOf(tools)
	if inline == nil {
		inline = []tool{}
	}
	return s.send(clientFrame{Type: frameTools, Tools: &inline})
}

// Answer returns a tool's result. Phonic carries on by itself once it has it.
func (s *STS) Answer(callID string, output string, err error) error {
	if err != nil {
		output = "Error: " + err.Error()
	}
	return s.send(clientFrame{Type: frameToolOutput, ToolCallID: callID, Output: &output})
}

// Prompt asks the model to speak now, guided by the text as a system message.
func (s *STS) Prompt(text string) error {
	return s.send(clientFrame{Type: frameReply, SystemMessage: text})
}

// Interrupt stops forwarding the reply in flight and settles it as cut off. Phonic has no
// way to be told: it stops of its own accord when it hears the caller.
func (s *STS) Interrupt(int) error {
	s.mu.Lock()
	if s.turn != nil {
		s.muted = true
	}
	s.mu.Unlock()
	s.settleFinished(true)
	s.settleTurn(true)
	return nil
}

// Events returns what the model heard, said and asked for.
func (s *STS) Events() <-chan sts.Event { return s.emitter.Events() }

// Close ends the session. A reply that finished is settled as it was; one still in flight is
// settled as cut off; then the session's seconds are reported for the last time.
func (s *STS) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()

	s.settleFinished(false)
	s.settleTurn(true)
	if conn != nil {
		s.billed(true)
		s.writeMu.Lock()
		_ = conn.WriteControl(websocket.CloseMessage, websocket.FormatCloseMessage(websocket.CloseNormalClosure, ""), time.Now().Add(time.Second))
		s.writeMu.Unlock()
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements sts.STS.
func (s *STS) Provider() string { return ProviderName }

// Model implements sts.STS.
func (s *STS) Model() string { return s.options.Model }

// Voice is the voice asked for, or empty when Phonic picks it.
func (s *STS) Voice() string { return s.options.Voice }

// SampleRate is the rate the model speaks at.
func (s *STS) SampleRate() int { return SampleRate }

// Capabilities implements sts.STS.
func (s *STS) Capabilities() sts.Capabilities { return CapabilitiesFor(s.options.Model) }

// handshake sends the config and waits for ready_to_start_conversation. A config Phonic
// refuses is answered with an error frame and the socket closed.
func (s *STS) handshake(conn *websocket.Conn) error {
	payload, err := json.Marshal(s.config())
	if err != nil {
		return stack.Wrap(err)
	}
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return stack.Wrap(fmt.Errorf("phonic: send config: %w", err))
	}
	if err := conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return stack.Wrap(fmt.Errorf("phonic: read ready: %w", err))
	}
	defer conn.SetReadDeadline(time.Time{})

	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			return stack.Wrap(fmt.Errorf("phonic: read ready: %w", err))
		}
		var frame serverFrame
		if err := json.Unmarshal(raw, &frame); err != nil {
			return stack.Wrap(fmt.Errorf("phonic: decode ready: %w", err))
		}
		switch frame.Type {
		case frameReady:
			return nil
		case frameError:
			return stack.Wrap(fmt.Errorf("phonic: config refused: %w", frame.failure()))
		}
	}
}

// config is the first frame: the model, its voice, instructions, tools and languages, audio
// at the router's rates, and the limits Phonic would otherwise end a call with.
func (s *STS) config() config {
	configured := config{
		Type:                frameConfig,
		Model:               s.options.Model,
		SystemPrompt:        s.options.Instructions,
		VoiceID:             s.options.Voice,
		InputFormat:         inputFormat,
		OutputFormat:        outputFormat,
		StreamAhead:         true,
		WebsocketTimeoutSec: socketTimeoutSec,
		NoInputEndSec:       silenceEndsCallSec,
		Tools:               toolsOf(s.options.Tools),
	}
	if len(s.options.Languages) > 0 {
		configured.DefaultLanguage = s.options.Languages[0]
		configured.AdditionalLanguages = s.options.Languages[1:]
	}
	return configured
}

// toolsOf is the tools as inline WebSocket tools. Phonic requires a parameters object.
func toolsOf(tools []llm.Tool) []tool {
	var inline []tool
	for _, offered := range tools {
		parameters := offered.Parameters
		if parameters == nil {
			parameters = map[string]any{"type": "object", "properties": map[string]any{}}
		}
		inline = append(inline, tool{
			Type: "custom_websocket",
			Schema: toolSchema{Type: "function", Function: function{
				Name:        offered.Name,
				Description: offered.Description,
				Parameters:  parameters,
				Strict:      true,
			}},
			OutputTimeout: toolTimeoutMs,
		})
	}
	return inline
}

func (s *STS) send(frame clientFrame) error {
	s.mu.Lock()
	conn, closed := s.conn, s.closed
	s.mu.Unlock()
	if closed {
		return stack.Wrap(errors.New("phonic: session closed"))
	}
	if conn == nil {
		return stack.Wrap(errors.New("phonic: not started"))
	}

	payload, err := json.Marshal(frame)
	if err != nil {
		return stack.Wrap(err)
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return stack.Wrap(fmt.Errorf("phonic: write: %w", err))
	}
	return nil
}

// readLoop translates server frames into events until the connection ends.
func (s *STS) readLoop(conn *websocket.Conn) {
	defer s.Close()
	for {
		_, raw, err := conn.ReadMessage()
		if err != nil {
			s.handleReadError(err)
			return
		}
		var frame serverFrame
		if err := json.Unmarshal(raw, &frame); err != nil {
			s.logger.Debug("undecodable frame", "error", err, "payload", string(raw))
			continue
		}
		s.handleMessage(frame)
	}
}

// handleReadError reports the connection ending, unless this side ended it. Phonic closes
// with a code of its own and a reason when it ends a conversation, such as 4505 at its
// length limit, so anything but a normal close is a failure.
func (s *STS) handleReadError(err error) {
	s.mu.Lock()
	closed, ended := s.closed, s.ended
	s.mu.Unlock()
	if closed || ended {
		return
	}

	s.settleFinished(false)
	s.settleTurn(true)
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(sts.Disconnected{Provider: ProviderName, Model: s.options.Model, Clean: true, At: time.Now()})
		return
	}
	s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: err, Context: "read", Fatal: true})
}

func (s *STS) handleMessage(frame serverFrame) {
	switch frame.Type {
	case frameCreated:
		s.logger.Debug("conversation created", "conversation", frame.ConversationID)
	case frameUserStarted:
		s.userStarted()
	case frameInputText:
		s.heard(frame.Text, frame.Language)
	case frameUserFinished:
		s.userFinished()
	case frameInputCancelled:
		s.mu.Lock()
		s.listening = false
		s.hearing = nil
		s.mu.Unlock()
	case frameChoseNotToRespond:
		s.finishHearing()
		s.mu.Lock()
		s.heardAt = time.Time{}
		s.mu.Unlock()
	case frameAssistantStarted:
		s.settleFinished(false)
		s.mu.Lock()
		s.muted = false
		s.mu.Unlock()
		s.replying()
	case frameAudio:
		s.spoke(frame)
	case frameAssistantFinished:
		s.assistantFinished()
	case frameInterrupted:
		s.settleFinished(true)
		s.settleTurn(true)
	case frameToolCall:
		s.toolCall(frame)
	case frameToolInterrupted:
		s.emitter.Send(sts.ToolCancel{CallIDs: []string{frame.ToolCallID}})
	case frameEndedConversation:
		s.settleFinished(false)
		s.settleTurn(false)
		s.mu.Lock()
		s.ended = true
		s.mu.Unlock()
		s.emitter.Send(sts.Disconnected{Provider: ProviderName, Model: s.options.Model, Reason: reasonEndedConversation, Clean: true, At: time.Now()})
		_ = s.Close()
	case frameError:
		s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: frame.failure(), Context: "session"})
	case frameWarning:
		if frame.Warning != nil {
			s.logger.Warn("phonic warned", "code", frame.Warning.Code, "message", frame.Warning.Message)
		}
	}
}

// failure is what an error frame says, with the parameters it names.
func (f serverFrame) failure() error {
	if f.Error == nil {
		return errors.New("phonic: unknown error")
	}
	if len(f.ParamErrors) > 0 && string(f.ParamErrors) != "null" {
		return fmt.Errorf("%w %s", f.Error, f.ParamErrors)
	}
	return f.Error
}

// userStarted opens the caller's turn. A turn already open is settled first.
func (s *STS) userStarted() {
	s.finishHearing()
	s.mu.Lock()
	s.listening = true
	participant := s.participant
	s.mu.Unlock()
	s.emitter.Send(sts.SpeechStarted{Participant: participant, At: time.Now()})
}

// userFinished is the caller's turn ending, which is what the reply's wait is timed from.
// Its last word can still follow, so the transcript is settled when the reply begins.
func (s *STS) userFinished() {
	now := time.Now()
	s.mu.Lock()
	s.heardAt = now
	participant := s.participant
	s.mu.Unlock()
	s.emitter.Send(sts.SpeechStopped{Participant: participant, At: now})
}

// heard is a word of the caller's, or the echo of a typed turn, which is dropped.
func (s *STS) heard(text, language string) {
	s.mu.Lock()
	if len(s.typed) > 0 && s.typed[0] == text {
		s.typed = s.typed[1:]
		s.mu.Unlock()
		return
	}
	s.listening = true
	if len(s.hearing) > 0 && !strings.HasPrefix(text, " ") {
		text = " " + text
	}
	s.hearing = append(s.hearing, text)
	participant := s.participant
	s.mu.Unlock()
	s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: stt.ModeDelta, Text: text, Language: language})
}

// finishHearing settles the caller's turn into a final transcript.
func (s *STS) finishHearing() {
	s.mu.Lock()
	if !s.listening {
		s.mu.Unlock()
		return
	}
	s.listening = false
	text := strings.TrimSpace(strings.Join(s.hearing, ""))
	s.hearing = nil
	participant := s.participant
	s.mu.Unlock()
	if text != "" {
		s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: stt.ModeFinal, Text: text})
	}
}

// replying is the reply in flight, opened now if there is none. Opening one settles the
// caller's turn and takes the moment they stopped as the start of the wait.
func (s *STS) replying() *sts.Turn {
	s.mu.Lock()
	if s.turn != nil {
		turn := s.turn
		s.mu.Unlock()
		return turn
	}
	s.mu.Unlock()
	s.finishHearing()

	s.mu.Lock()
	s.generation++
	turn := sts.NewTurn("", s.generation, s.heardAt)
	s.heardAt = time.Time{}
	s.turn = turn
	s.mu.Unlock()
	s.emitter.Send(sts.ResponseStarted{ResponseID: turn.ID, Generation: turn.Generation, At: time.Now()})
	return turn
}

// spoke forwards a piece of the model's speech and the words it carries, unless the reply
// was cut off from this side.
func (s *STS) spoke(frame serverFrame) {
	s.mu.Lock()
	muted := s.muted
	s.mu.Unlock()
	if muted {
		return
	}
	raw, err := base64.StdEncoding.DecodeString(frame.Audio)
	if err != nil {
		s.logger.Debug("undecodable audio", "error", err)
		return
	}
	turn := s.replying()
	if len(raw) > 0 {
		s.emitter.Send(turn.Chunk(audio.FromBytes(raw, SampleRate, 1)))
	}
	if frame.Text != "" {
		turn.Said(frame.Text)
		s.emitter.Send(sts.OutputTranscript{ResponseID: turn.ID, Mode: stt.ModeDelta, Text: frame.Text})
	}
}

// assistantFinished holds the reply that ended for interruptGrace, since a barge-in is
// reported as its end followed by interrupted_response.
func (s *STS) assistantFinished() {
	s.mu.Lock()
	turn := s.turn
	s.turn = nil
	s.muted = false
	if turn == nil {
		s.mu.Unlock()
		return
	}
	previous := s.finished
	s.finished = turn
	if s.finishing != nil {
		s.finishing.Stop()
	}
	s.finishing = time.AfterFunc(interruptGrace, func() { s.settleFinished(false) })
	s.mu.Unlock()
	if previous != nil {
		s.complete(previous, false)
	}
}

// settleFinished settles the reply held after it ended, if there is one.
func (s *STS) settleFinished(interrupted bool) {
	s.mu.Lock()
	turn := s.finished
	s.finished = nil
	if s.finishing != nil {
		s.finishing.Stop()
		s.finishing = nil
	}
	s.mu.Unlock()
	if turn != nil {
		s.complete(turn, interrupted)
	}
}

// settleTurn settles the reply in flight, if there is one.
func (s *STS) settleTurn(interrupted bool) {
	s.mu.Lock()
	turn := s.turn
	s.turn = nil
	s.mu.Unlock()
	if turn != nil {
		s.complete(turn, interrupted)
	}
}

// complete reports a reply settled, and the session's seconds so far with it.
func (s *STS) complete(turn *sts.Turn, interrupted bool) {
	s.emitter.Send(turn.Complete(ProviderName, s.options.Model, interrupted, sts.Usage{}))
	s.billed(false)
}

// toolCall forwards a call the model made, against the reply it made it in. A call can come
// before the reply has said anything, and opens it.
func (s *STS) toolCall(frame serverFrame) {
	arguments := string(frame.Parameters)
	if arguments == "" || arguments == "null" {
		arguments = "{}"
	}
	turn := s.replying()
	s.emitter.Send(sts.ToolCall{ResponseID: turn.ID, CallID: frame.ToolCallID, Name: frame.ToolName, Arguments: arguments})
}

// billed reports the seconds the session has been open. Phonic reports none, so they are
// counted here from when it took the config, which is when its conversation began.
func (s *STS) billed(final bool) {
	s.mu.Lock()
	openedAt := s.openedAt
	s.mu.Unlock()
	if openedAt.IsZero() {
		return
	}
	s.emitter.Send(sts.SessionUsage{Seconds: int64(math.Round(time.Since(openedAt).Seconds())), Final: final})
}
