// Package openairealtime implements the sts.STS contract over the OpenAI Realtime API's
// WebSocket, and over the two vendors that speak the same events: xAI's Voice Agent API and
// Alibaba's Qwen realtime models.
//
// The three agree on everything that arrives: the caller's audio goes up in
// input_audio_buffer.append frames, and speech, transcripts, tool calls and the end of a
// reply come back under the same event names. Where they disagree is how a session is
// configured, so the setup frame is spelled per vendor and the rest is shared.
//
// A reply is a response. The model opens one when its own turn detector decides the caller
// has finished, streams audio and a transcript of it, and closes it with response.done,
// which carries what it cost. A reply the caller talked over ends the same way with a
// status saying so, and that status is the one thing that means barge-in here.
package openairealtime

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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Dialect is how a vendor spells the session configuration. The events they send back are
// the same; the frame that sets a session up is not.
type Dialect string

const (
	// DialectOpenAI is the generally available Realtime API: the session names its type
	// and nests the audio formats, the transcriber and the turn detector under audio.
	DialectOpenAI Dialect = "openai"
	// DialectXAI keeps the audio formats nested and everything else at the top level,
	// which is the shape xAI documents as OpenAI-compatible.
	DialectXAI Dialect = "xai"
	// DialectQwen names its formats as words, pcm16 and pcm24, and its transcriber by a
	// model of its own.
	DialectQwen Dialect = "qwen"
)

// Vendor is one endpoint that speaks this protocol, and what differs about it.
type Vendor struct {
	// Provider is the stable name used in routing config and stats.
	Provider string
	// URL is the socket, without the model, which goes on the query string.
	URL string
	// Model is what a session is opened on when the options name none.
	Model string
	// APIKeyEnvVar holds the credentials when Options does not.
	APIKeyEnvVar string
	// InputSampleRate is the rate the vendor wants fed. The caller's audio arrives at
	// sts.InputSampleRate and is resampled up where these differ.
	InputSampleRate int
	// OutputSampleRate is the rate the model speaks at.
	OutputSampleRate int
	Dialect          Dialect
}

// The vendors this package reaches.
var (
	OpenAI = Vendor{
		Provider:         "openai",
		URL:              "wss://api.openai.com/v1/realtime",
		Model:            "gpt-realtime-2",
		APIKeyEnvVar:     "OPENAI_API_KEY",
		InputSampleRate:  24_000,
		OutputSampleRate: 24_000,
		Dialect:          DialectOpenAI,
	}
	XAI = Vendor{
		Provider:         "xai",
		URL:              "wss://api.x.ai/v1/realtime",
		Model:            "grok-voice-think-fast-2.0",
		APIKeyEnvVar:     "XAI_API_KEY",
		InputSampleRate:  24_000,
		OutputSampleRate: 24_000,
		Dialect:          DialectXAI,
	}
	Qwen = Vendor{
		Provider:         "qwen",
		URL:              "wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
		Model:            "qwen3.5-omni-plus-realtime",
		APIKeyEnvVar:     "DASHSCOPE_API_KEY",
		InputSampleRate:  16_000,
		OutputSampleRate: 24_000,
		Dialect:          DialectQwen,
	}
)

// qwenTranscriber is the transcription model Qwen's realtime sessions take, which is the
// only one they document.
const qwenTranscriber = "gummy-realtime-v1"

// openaiTranscriber is the transcriber OpenAI sessions are given when nobody named one.
// OpenAI refuses a transcription block with no model in it, where xAI takes the empty
// block as yes.
const openaiTranscriber = "gpt-4o-mini-transcribe"

// Client event types.
const (
	eventSessionUpdate  = "session.update"
	eventAudioAppend    = "input_audio_buffer.append"
	eventImageAppend    = "input_image_buffer.append"
	eventItemCreate     = "conversation.item.create"
	eventItemTruncate   = "conversation.item.truncate"
	eventResponseCreate = "response.create"
	eventResponseCancel = "response.cancel"
)

// Server event types. Two of them come in an older spelling as well, which Qwen still uses.
const (
	eventSessionCreated    = "session.created"
	eventSessionUpdated    = "session.updated"
	eventError             = "error"
	eventSpeechStarted     = "input_audio_buffer.speech_started"
	eventSpeechStopped     = "input_audio_buffer.speech_stopped"
	eventCommitted         = "input_audio_buffer.committed"
	eventInputDelta        = "conversation.item.input_audio_transcription.delta"
	eventInputCompleted    = "conversation.item.input_audio_transcription.completed"
	eventResponseCreated   = "response.created"
	eventOutputItemAdded   = "response.output_item.added"
	eventAudioDelta        = "response.output_audio.delta"
	eventAudioDeltaOld     = "response.audio.delta"
	eventTranscriptDelta   = "response.output_audio_transcript.delta"
	eventTranscriptDone    = "response.output_audio_transcript.done"
	eventTranscriptDeltaV1 = "response.audio_transcript.delta"
	eventTranscriptDoneV1  = "response.audio_transcript.done"
	eventArgumentsDone     = "response.function_call_arguments.done"
	eventResponseDone      = "response.done"
)

// cancelNotActive is what the server says about a cancel that found nothing to cancel,
// which is not news to anyone.
const cancelNotActive = "response_cancel_not_active"

// Options configures the provider. APIKey falls back to the vendor's environment variable.
type Options struct {
	Vendor Vendor
	APIKey string
	Model  string
	URL    string
	// Voice is the vendor's own name for one. Empty leaves the vendor's default.
	Voice string
	// Instructions is the system prompt the model converses under.
	Instructions string
	// Tools are what the model may call. Empty offers none.
	Tools []llm.Tool
	// TurnDetection is one of the options.Turn modes. Empty leaves the vendor default.
	TurnDetection string
	// SilenceMs and PrefixPaddingMs tune a silence timer.
	SilenceMs       *int
	PrefixPaddingMs *int
	// Threshold is the silence timer's activation threshold, which only this family of
	// vendors takes as a number.
	Threshold *float64
	// Eagerness is how quickly a semantic turn detector decides: low, medium, high or
	// auto. OpenAI only.
	Eagerness string
	// InterruptResponse is whether the model cuts its own reply off when it hears the
	// caller. Nil leaves the vendor default.
	InterruptResponse *bool
	// InputTranscript asks the model to write down what it heard. Qwen always does.
	InputTranscript bool
	// TranscriptionModel names the transcriber, on the vendors that let one be chosen.
	TranscriptionModel string
	// HandshakeTimeout bounds the connect and the configuration exchange.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger
}

// clientEvent is a frame sent to the server. Type says which of the fields matter.
type clientEvent struct {
	Type     string           `json:"type"`
	Session  *session         `json:"session,omitempty"`
	Audio    string           `json:"audio,omitempty"`
	Image    string           `json:"image,omitempty"`
	Item     *item            `json:"item,omitempty"`
	Response *responseRequest `json:"response,omitempty"`
	// ItemID, ContentIndex and AudioEndMs truncate an assistant item to what was heard.
	ItemID       string `json:"item_id,omitempty"`
	ContentIndex *int   `json:"content_index,omitempty"`
	AudioEndMs   *int   `json:"audio_end_ms,omitempty"`
}

// session is the configuration, with every dialect's fields on it and omitempty deciding
// which are sent.
type session struct {
	Type                    string         `json:"type,omitempty"`
	OutputModalities        []string       `json:"output_modalities,omitempty"`
	Modalities              []string       `json:"modalities,omitempty"`
	Instructions            string         `json:"instructions,omitempty"`
	Voice                   string         `json:"voice,omitempty"`
	Audio                   *audioConfig   `json:"audio,omitempty"`
	InputAudioFormat        string         `json:"input_audio_format,omitempty"`
	OutputAudioFormat       string         `json:"output_audio_format,omitempty"`
	InputAudioTranscription *transcription `json:"input_audio_transcription,omitempty"`
	TurnDetection           *turnDetection `json:"turn_detection,omitempty"`
	Tools                   []tool         `json:"tools,omitempty"`
}

type audioConfig struct {
	Input  *audioInput  `json:"input,omitempty"`
	Output *audioOutput `json:"output,omitempty"`
}

type audioInput struct {
	Format        *audioFormat   `json:"format,omitempty"`
	Transcription *transcription `json:"transcription,omitempty"`
	TurnDetection *turnDetection `json:"turn_detection,omitempty"`
}

type audioOutput struct {
	Format *audioFormat `json:"format,omitempty"`
	Voice  string       `json:"voice,omitempty"`
}

type audioFormat struct {
	Type string `json:"type"`
	Rate int    `json:"rate,omitempty"`
}

type transcription struct {
	Model string `json:"model,omitempty"`
}

type turnDetection struct {
	Type              string   `json:"type"`
	Threshold         *float64 `json:"threshold,omitempty"`
	PrefixPaddingMs   *int     `json:"prefix_padding_ms,omitempty"`
	SilenceDurationMs *int     `json:"silence_duration_ms,omitempty"`
	Eagerness         string   `json:"eagerness,omitempty"`
	CreateResponse    *bool    `json:"create_response,omitempty"`
	InterruptResponse *bool    `json:"interrupt_response,omitempty"`
}

type tool struct {
	Type        string         `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// item is one conversation item the client adds: a typed turn, an image, or what a tool
// produced.
type item struct {
	Type    string    `json:"type"`
	Role    string    `json:"role,omitempty"`
	Content []content `json:"content,omitempty"`
	CallID  string    `json:"call_id,omitempty"`
	Output  string    `json:"output,omitempty"`
}

type content struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

type responseRequest struct {
	Instructions string `json:"instructions,omitempty"`
}

// serverEvent is a frame from the server, with the fields every event type this package
// reads might carry.
type serverEvent struct {
	Type       string      `json:"type"`
	ResponseID string      `json:"response_id"`
	ItemID     string      `json:"item_id"`
	Delta      string      `json:"delta"`
	Transcript string      `json:"transcript"`
	CallID     string      `json:"call_id"`
	Name       string      `json:"name"`
	Arguments  string      `json:"arguments"`
	Response   *response   `json:"response"`
	Item       *serverItem `json:"item"`
	Error      *apiError   `json:"error"`
}

type response struct {
	ID            string         `json:"id"`
	Status        string         `json:"status"`
	StatusDetails *statusDetails `json:"status_details"`
	Usage         *usage         `json:"usage"`
}

type statusDetails struct {
	Type   string    `json:"type"`
	Reason string    `json:"reason"`
	Error  *apiError `json:"error"`
}

// UnmarshalJSON reads the details as the object OpenAI sends or as the bare string xAI
// sends in its place. A frame that failed to decode over this one field would be dropped
// whole, and response.done is the frame that settles a reply.
func (d *statusDetails) UnmarshalJSON(raw []byte) error {
	if len(raw) > 0 && raw[0] == '"' {
		var reason string
		if err := json.Unmarshal(raw, &reason); err != nil {
			return err
		}
		*d = statusDetails{Reason: reason}
		return nil
	}
	type plain statusDetails
	var decoded plain
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*d = statusDetails(decoded)
	return nil
}

type usage struct {
	InputTokens        int64         `json:"input_tokens"`
	OutputTokens       int64         `json:"output_tokens"`
	InputTokenDetails  *tokenDetails `json:"input_token_details"`
	OutputTokenDetails *tokenDetails `json:"output_token_details"`
}

type tokenDetails struct {
	CachedTokens int64 `json:"cached_tokens"`
	TextTokens   int64 `json:"text_tokens"`
	AudioTokens  int64 `json:"audio_tokens"`
}

type serverItem struct {
	ID   string `json:"id"`
	Type string `json:"type"`
}

type apiError struct {
	Type    string `json:"type"`
	Code    string `json:"code"`
	Message string `json:"message"`
}

func (e *apiError) Error() string {
	if e.Code != "" {
		return e.Code + ": " + e.Message
	}
	return e.Message
}

// STS is one realtime session.
type STS struct {
	options      Options
	capabilities sts.Capabilities
	logger       *slog.Logger
	emitter      *sts.Emitter

	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	mu sync.Mutex
	// participant is the speaker of the most recent audio, used to label what the model
	// heard and who it heard start and stop.
	participant sts.Participant
	// heardAt is when the model decided the caller had finished, which is what the reply
	// that follows is timed from. Zero once a reply has claimed it.
	heardAt time.Time
	// turn is the reply in flight, and generation counts them.
	turn       *sts.Turn
	generation int
	// itemID is the assistant item being spoken, which is what a truncate names.
	itemID string
	// speaking is whether the model's detector currently hears the caller, and heardSoFar
	// the latest restatement of what they have said. xAI reports the transcript of a turn
	// as it grows, each report restating it from the beginning, and only the one after
	// speech stops is settled; OpenAI reports once, after the stop.
	speaking   bool
	heardSoFar string
	started    bool
	closed     bool
}

// New validates the settings and returns an unstarted provider.
func New(settings Options) (*STS, error) {
	if settings.Vendor.Provider == "" {
		return nil, errors.New("openairealtime: a vendor is required")
	}
	if settings.APIKey == "" && settings.Vendor.APIKeyEnvVar != "" {
		settings.APIKey = os.Getenv(settings.Vendor.APIKeyEnvVar)
	}
	if settings.APIKey == "" {
		return nil, fmt.Errorf("%s: api key is required (set %s)", settings.Vendor.Provider, settings.Vendor.APIKeyEnvVar)
	}
	if settings.Model == "" {
		settings.Model = settings.Vendor.Model
	}
	if settings.URL == "" {
		settings.URL = settings.Vendor.URL
	}
	if !strings.HasPrefix(settings.URL, "ws://") && !strings.HasPrefix(settings.URL, "wss://") {
		return nil, fmt.Errorf("%s: url must be ws:// or wss://, got %s", settings.Vendor.Provider, settings.URL)
	}
	if settings.Vendor.InputSampleRate <= 0 || settings.Vendor.OutputSampleRate <= 0 {
		return nil, fmt.Errorf("%s: the vendor's sample rates are required", settings.Vendor.Provider)
	}
	capabilities := CapabilitiesFor(settings.Vendor, settings.Model)
	switch settings.TurnDetection {
	case "", options.TurnServerVAD:
	case options.TurnSemantic:
		if !capabilities.SemanticTurns {
			return nil, fmt.Errorf("%s: %s has no semantic turn detector", settings.Vendor.Provider, settings.Model)
		}
	case options.TurnManual:
		return nil, fmt.Errorf("%s: manual turns are not supported", settings.Vendor.Provider)
	default:
		return nil, fmt.Errorf("%s: unknown turn detection %q", settings.Vendor.Provider, settings.TurnDetection)
	}
	if len(settings.Tools) > 0 && !capabilities.Tools {
		return nil, fmt.Errorf("%s: %s does not call tools", settings.Vendor.Provider, settings.Model)
	}
	if settings.HandshakeTimeout == 0 {
		settings.HandshakeTimeout = 30 * time.Second
	}
	logger := settings.Logger
	if logger == nil {
		logger = slog.Default()
	}

	return &STS{
		options:      settings,
		capabilities: capabilities,
		logger:       logger.With("provider", settings.Vendor.Provider, "model", settings.Model),
		emitter:      sts.NewEmitter(sts.EmitterBuffer),
	}, nil
}

// CapabilitiesFor is what a model at a vendor can be asked for. It is a table rather than
// something learned from the session, because routing checks it before a session exists.
func CapabilitiesFor(vendor Vendor, model string) sts.Capabilities {
	switch vendor.Dialect {
	case DialectOpenAI:
		return sts.Capabilities{
			InputModalities:        []string{options.ModalityImage},
			Text:                   true,
			Tools:                  true,
			InputTranscript:        true,
			OutputTranscript:       true,
			SemanticTurns:          true,
			Endpointing:            true,
			InstructionsMidSession: true,
			ToolsMidSession:        true,
			Usage:                  true,
			MaxDuration:            60 * time.Minute,
		}
	case DialectXAI:
		return sts.Capabilities{
			Text:                   true,
			Tools:                  true,
			InputTranscript:        true,
			OutputTranscript:       true,
			Endpointing:            true,
			InstructionsMidSession: true,
			ToolsMidSession:        true,
		}
	case DialectQwen:
		// Qwen's realtime models take neither a typed turn nor a tool, and a session is
		// configured once, before its first audio.
		return sts.Capabilities{
			InputModalities:  []string{options.ModalityImage},
			InputTranscript:  true,
			OutputTranscript: true,
			Endpointing:      true,
			Usage:            true,
		}
	default:
		return sts.Capabilities{}
	}
}

// Start dials the socket, configures the session and waits for the server to accept the
// configuration. It returns once the model is ready to hear the caller.
func (s *STS) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return fmt.Errorf("%s: already started", s.options.Vendor.Provider)
	}
	s.started = true
	s.mu.Unlock()

	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	header := http.Header{"Authorization": []string{"Bearer " + s.options.APIKey}}
	conn, response, err := dialer.DialContext(ctx, s.endpoint(), header)
	if err != nil {
		if response != nil {
			return fmt.Errorf("%s: dial: %w (http %d)", s.options.Vendor.Provider, err, response.StatusCode)
		}
		return fmt.Errorf("%s: dial: %w", s.options.Vendor.Provider, err)
	}
	s.conn = conn

	if err := s.handshake(); err != nil {
		conn.Close()
		return err
	}

	s.emitter.Send(sts.Connected{Provider: s.options.Vendor.Provider, Model: s.options.Model, At: time.Now()})
	go s.readLoop()
	return nil
}

// ProcessAudio streams one chunk of the caller's speech. The participant labels whatever
// the model hears in it.
func (s *STS) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	if err := pcm.Validate(sts.InputSampleRate); err != nil {
		return fmt.Errorf("%s: %w", s.options.Vendor.Provider, err)
	}
	if err := s.ready(); err != nil {
		return err
	}

	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()

	if pcm.SampleRate != s.options.Vendor.InputSampleRate {
		pcm = audio.Resample(pcm, s.options.Vendor.InputSampleRate, 1)
	}
	return s.send(clientEvent{Type: eventAudioAppend, Audio: base64.StdEncoding.EncodeToString(pcm.Bytes())})
}

// SendText injects a typed turn and asks the model to answer it.
func (s *STS) SendText(text string, participant sts.Participant) error {
	if !s.capabilities.Text {
		return sts.ErrNoText
	}
	if err := s.ready(); err != nil {
		return err
	}

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

// SendFrame offers the model a still image as context for its next reply.
func (s *STS) SendFrame(frame llm.ImagePart) error {
	if !s.capabilities.Accepts(options.ModalityImage) {
		return sts.ErrNoImages
	}
	if err := s.ready(); err != nil {
		return err
	}
	if len(frame.Data) == 0 {
		return fmt.Errorf("%s: a frame has to carry its bytes", s.options.Vendor.Provider)
	}

	if s.options.Vendor.Dialect == DialectQwen {
		return s.send(clientEvent{Type: eventImageAppend, Image: base64.StdEncoding.EncodeToString(frame.Data)})
	}
	return s.send(clientEvent{Type: eventItemCreate, Item: &item{
		Type:    "message",
		Role:    "user",
		Content: []content{{Type: "input_image", ImageURL: frame.DataURI()}},
	}})
}

// SetInstructions changes the system prompt for the replies that follow.
func (s *STS) SetInstructions(text string) error {
	if !s.capabilities.InstructionsMidSession {
		return sts.ErrInstructionsFixed
	}
	if err := s.ready(); err != nil {
		return err
	}
	update := &session{Instructions: text}
	if s.options.Vendor.Dialect == DialectOpenAI {
		update.Type = "realtime"
	}
	return s.send(clientEvent{Type: eventSessionUpdate, Session: update})
}

// SetTools replaces what the model may call.
func (s *STS) SetTools(tools []llm.Tool) error {
	if !s.capabilities.Tools {
		return sts.ErrNoTools
	}
	if !s.capabilities.ToolsMidSession {
		return sts.ErrToolsFixed
	}
	if err := s.ready(); err != nil {
		return err
	}
	update := &session{Tools: toolsOf(tools)}
	if s.options.Vendor.Dialect == DialectOpenAI {
		update.Type = "realtime"
	}
	return s.send(clientEvent{Type: eventSessionUpdate, Session: update})
}

// Answer returns a tool's result and asks the model to carry on, which this protocol does
// not do on its own.
func (s *STS) Answer(callID string, output string, err error) error {
	if !s.capabilities.Tools {
		return sts.ErrNoTools
	}
	if readyErr := s.ready(); readyErr != nil {
		return readyErr
	}
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

// Prompt asks the model to reply now, guided by the text.
func (s *STS) Prompt(text string) error {
	if !s.capabilities.Text {
		return sts.ErrNoText
	}
	if err := s.ready(); err != nil {
		return err
	}
	return s.send(clientEvent{Type: eventResponseCreate, Response: &responseRequest{Instructions: text}})
}

// Interrupt stops the reply in flight and tells the model how much of it was heard, so its
// own record of the conversation ends where the listener's does.
func (s *STS) Interrupt(playedMs int) error {
	if err := s.ready(); err != nil {
		return err
	}

	s.mu.Lock()
	turn, itemID := s.turn, s.itemID
	s.mu.Unlock()
	if turn == nil {
		return nil
	}

	if err := s.send(clientEvent{Type: eventResponseCancel}); err != nil {
		return err
	}
	if itemID == "" {
		return nil
	}
	if playedMs <= 0 {
		playedMs = turn.SentMs()
	}
	index := 0
	return s.send(clientEvent{Type: eventItemTruncate, ItemID: itemID, ContentIndex: &index, AudioEndMs: &playedMs})
}

// Events returns what the model heard, said and asked for.
func (s *STS) Events() <-chan sts.Event { return s.emitter.Events() }

// Close ends the session. A reply still in flight is settled as interrupted, so it is
// billed once like every other.
func (s *STS) Close() error {
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	conn := s.conn
	s.mu.Unlock()

	s.settleTurn(true, sts.Usage{})
	if conn != nil {
		conn.Close()
	}
	s.emitter.Close()
	return nil
}

// Provider implements sts.STS.
func (s *STS) Provider() string { return s.options.Vendor.Provider }

// Model implements sts.STS.
func (s *STS) Model() string { return s.options.Model }

// SampleRate is the rate the model speaks at.
func (s *STS) SampleRate() int { return s.options.Vendor.OutputSampleRate }

// Capabilities implements sts.STS.
func (s *STS) Capabilities() sts.Capabilities { return s.capabilities }

// Client exposes the underlying WebSocket so callers can use the protocol directly.
func (s *STS) Client() *websocket.Conn { return s.conn }

// endpoint is the socket with the model on its query string, which is where every vendor
// here takes it.
func (s *STS) endpoint() string {
	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + "model=" + url.QueryEscape(s.options.Model)
}

// ready reports whether the session can be sent anything.
func (s *STS) ready() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return fmt.Errorf("%s: session closed", s.options.Vendor.Provider)
	}
	if !s.started || s.conn == nil {
		return fmt.Errorf("%s: not started", s.options.Vendor.Provider)
	}
	return nil
}

// handshake waits for the server to open the session, configures it, and waits for the
// server to say the configuration took. Audio sent before that would be heard by a session
// configured some other way.
func (s *STS) handshake() error {
	if err := s.conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("%s: read handshake: %w", s.options.Vendor.Provider, err)
	}
	defer s.conn.SetReadDeadline(time.Time{})

	created, err := s.next()
	if err != nil {
		return err
	}
	if created.Type != eventSessionCreated {
		return fmt.Errorf("%s: expected %s, got %s", s.options.Vendor.Provider, eventSessionCreated, created.Type)
	}

	if err := s.send(clientEvent{Type: eventSessionUpdate, Session: s.session()}); err != nil {
		return fmt.Errorf("%s: send session: %w", s.options.Vendor.Provider, err)
	}
	for {
		event, err := s.next()
		if err != nil {
			return err
		}
		switch event.Type {
		case eventSessionUpdated:
			return nil
		case eventError:
			return fmt.Errorf("%s: session rejected: %w", s.options.Vendor.Provider, event.Error)
		}
	}
}

// next reads one server frame during the handshake.
func (s *STS) next() (serverEvent, error) {
	_, raw, err := s.conn.ReadMessage()
	if err != nil {
		return serverEvent{}, fmt.Errorf("%s: read handshake: %w", s.options.Vendor.Provider, err)
	}
	var event serverEvent
	if err := json.Unmarshal(raw, &event); err != nil {
		return serverEvent{}, fmt.Errorf("%s: decode handshake: %w", s.options.Vendor.Provider, err)
	}
	if event.Type == eventError && event.Error == nil {
		event.Error = &apiError{Message: strings.TrimSpace(string(raw))}
	}
	return event, nil
}

// session is the configuration frame, spelled the way this vendor wants it.
func (s *STS) session() *session {
	configured := &session{Instructions: s.options.Instructions, Tools: toolsOf(s.options.Tools)}
	turns := s.turnDetection()
	// xAI answers the caller on its own only when its detector is told to: a session
	// that names no detector waits to be asked for every reply.
	if s.options.Vendor.Dialect == DialectXAI {
		if turns == nil {
			turns = &turnDetection{Type: "server_vad"}
		}
		answers := true
		turns.CreateResponse = &answers
	}

	switch s.options.Vendor.Dialect {
	case DialectOpenAI:
		configured.Type = "realtime"
		configured.OutputModalities = []string{"audio"}
		configured.Audio = &audioConfig{
			Input: &audioInput{
				Format:        &audioFormat{Type: "audio/pcm", Rate: s.options.Vendor.InputSampleRate},
				Transcription: s.transcription(),
				TurnDetection: turns,
			},
			Output: &audioOutput{
				Format: &audioFormat{Type: "audio/pcm", Rate: s.options.Vendor.OutputSampleRate},
				Voice:  s.options.Voice,
			},
		}
	case DialectXAI:
		configured.Modalities = []string{"text", "audio"}
		configured.Voice = s.options.Voice
		configured.Audio = &audioConfig{
			Input:  &audioInput{Format: &audioFormat{Type: "audio/pcm", Rate: s.options.Vendor.InputSampleRate}},
			Output: &audioOutput{Format: &audioFormat{Type: "audio/pcm", Rate: s.options.Vendor.OutputSampleRate}},
		}
		configured.InputAudioTranscription = s.transcription()
		configured.TurnDetection = turns
	case DialectQwen:
		configured.Modalities = []string{"text", "audio"}
		configured.Voice = s.options.Voice
		configured.InputAudioFormat = "pcm16"
		configured.OutputAudioFormat = "pcm24"
		// Qwen's session always transcribes what it heard, and always with the one
		// transcriber it has, so asking is not optional here.
		configured.InputAudioTranscription = &transcription{Model: qwenTranscriber}
		configured.TurnDetection = turns
	}
	return configured
}

// transcription asks the model to write down what it heard, or leaves the default when
// nobody asked.
func (s *STS) transcription() *transcription {
	if !s.options.InputTranscript {
		return nil
	}
	model := s.options.TranscriptionModel
	if model == "" && s.options.Vendor.Dialect == DialectOpenAI {
		model = openaiTranscriber
	}
	return &transcription{Model: model}
}

// turnDetection is the detector the session was asked for, or nil for the vendor's own.
func (s *STS) turnDetection() *turnDetection {
	switch s.options.TurnDetection {
	case options.TurnSemantic:
		return &turnDetection{
			Type:              "semantic_vad",
			Eagerness:         s.options.Eagerness,
			InterruptResponse: s.options.InterruptResponse,
		}
	case options.TurnServerVAD:
		return &turnDetection{
			Type:              "server_vad",
			Threshold:         s.options.Threshold,
			PrefixPaddingMs:   s.options.PrefixPaddingMs,
			SilenceDurationMs: s.options.SilenceMs,
			InterruptResponse: s.options.InterruptResponse,
		}
	}
	// Nothing was asked for by name. A threshold or a barge-in preference on its own
	// still needs a detector to hang off, and a silence timer is every vendor's default.
	if s.options.Threshold != nil || s.options.SilenceMs != nil || s.options.PrefixPaddingMs != nil || s.options.InterruptResponse != nil {
		return &turnDetection{
			Type:              "server_vad",
			Threshold:         s.options.Threshold,
			PrefixPaddingMs:   s.options.PrefixPaddingMs,
			SilenceDurationMs: s.options.SilenceMs,
			InterruptResponse: s.options.InterruptResponse,
		}
	}
	return nil
}

// toolsOf is the model's shape for what it may call.
func toolsOf(tools []llm.Tool) []tool {
	if len(tools) == 0 {
		return nil
	}
	shaped := make([]tool, 0, len(tools))
	for _, offered := range tools {
		shaped = append(shaped, tool{
			Type:        "function",
			Name:        offered.Name,
			Description: offered.Description,
			Parameters:  offered.Parameters,
		})
	}
	return shaped
}

func (s *STS) send(event clientEvent) error {
	payload, err := json.Marshal(event)
	if err != nil {
		return err
	}

	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := s.conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("%s: write %s: %w", s.options.Vendor.Provider, event.Type, err)
	}
	return nil
}

// readLoop translates server frames into events until the connection ends.
func (s *STS) readLoop() {
	for {
		_, raw, err := s.conn.ReadMessage()
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

// handleReadError settles what the connection was in the middle of. A reply cut off by a
// dead socket is still a reply that was billed for.
func (s *STS) handleReadError(err error) {
	s.mu.Lock()
	closed := s.closed
	s.mu.Unlock()
	if closed {
		return
	}

	s.settleTurn(true, sts.Usage{})
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(sts.Disconnected{
			Provider: s.options.Vendor.Provider,
			Model:    s.options.Model,
			Clean:    true,
			At:       time.Now(),
		})
		return
	}
	s.emitter.Send(sts.Error{
		Provider: s.options.Vendor.Provider,
		Model:    s.options.Model,
		Err:      err,
		Context:  "read",
		Fatal:    true,
	})
}

func (s *STS) handleMessage(event serverEvent) {
	switch event.Type {
	case eventSpeechStarted:
		s.mu.Lock()
		s.speaking = true
		s.mu.Unlock()
		s.emitter.Send(sts.SpeechStarted{Participant: s.speaker(), At: time.Now()})

	case eventSpeechStopped:
		s.stoppedHearing()

	case eventCommitted:
		// A semantic turn detector commits without ever saying speech stopped, so the
		// commit is the moment the caller started waiting. Under a silence timer the stop
		// came first and this adds nothing.
		s.mu.Lock()
		waiting := !s.heardAt.IsZero()
		s.mu.Unlock()
		if !waiting {
			s.stoppedHearing()
		}

	case eventInputDelta:
		if event.Delta != "" {
			s.emitter.Send(sts.InputTranscript{Participant: s.speaker(), Mode: stt.ModeDelta, Text: event.Delta})
		}

	case eventInputCompleted:
		s.heardTranscript(strings.TrimSpace(event.Transcript))

	case eventResponseCreated:
		s.startTurn(event)

	case eventOutputItemAdded:
		if event.Item != nil && event.Item.Type == "message" {
			s.mu.Lock()
			s.itemID = event.Item.ID
			s.mu.Unlock()
		}

	case eventAudioDelta, eventAudioDeltaOld:
		s.speak(event)

	case eventTranscriptDelta, eventTranscriptDeltaV1:
		if event.Delta != "" {
			s.emitter.Send(sts.OutputTranscript{ResponseID: s.responseID(event.ResponseID), Mode: stt.ModeDelta, Text: event.Delta})
		}

	case eventTranscriptDone, eventTranscriptDoneV1:
		if text := strings.TrimSpace(event.Transcript); text != "" {
			s.emitter.Send(sts.OutputTranscript{ResponseID: s.responseID(event.ResponseID), Mode: stt.ModeFinal, Text: text})
		}

	case eventArgumentsDone:
		s.emitter.Send(sts.ToolCall{
			ResponseID: s.responseID(event.ResponseID),
			CallID:     event.CallID,
			Name:       event.Name,
			Arguments:  event.Arguments,
		})

	case eventResponseDone:
		s.finishTurn(event)

	case eventError:
		s.fail(event)

	default:
		if s.logger.Enabled(context.Background(), slog.LevelDebug) {
			s.logger.Debug("unhandled frame", "type", event.Type)
		}
	}
}

// stoppedHearing marks the moment the caller finished, which the next reply is timed from.
// A reply the model opened while the caller was still talking is timed from here too.
func (s *STS) stoppedHearing() {
	now := time.Now()
	s.mu.Lock()
	s.heardAt = now
	s.speaking = false
	if s.turn != nil {
		s.turn.HeardAt(now)
	}
	participant := s.participant
	s.mu.Unlock()
	s.emitter.Send(sts.SpeechStopped{Participant: participant, At: now})
}

// heardTranscript reports what the model wrote down of the caller. While they are still
// talking it is a restatement that the next one supersedes; once they have stopped it is
// the transcript the turn settles on.
func (s *STS) heardTranscript(text string) {
	if text == "" {
		return
	}
	s.mu.Lock()
	mode := stt.ModeFinal
	if s.speaking {
		mode = stt.ModeReplacement
		s.heardSoFar = text
	} else {
		s.heardSoFar = ""
	}
	participant := s.participant
	s.mu.Unlock()
	s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: mode, Text: text})
}

// settleHeard settles a restatement nothing followed. The reply beginning to speak is the
// latest moment the caller's turn can still be open, so what was heard by then is what
// was said.
func (s *STS) settleHeard() {
	s.mu.Lock()
	text := s.heardSoFar
	s.heardSoFar = ""
	participant := s.participant
	s.mu.Unlock()
	if text != "" {
		s.emitter.Send(sts.InputTranscript{Participant: participant, Mode: stt.ModeFinal, Text: text})
	}
}

// startTurn opens the bookkeeping for a reply. A reply that arrives while another is still
// open settles the earlier one first, so nothing is billed twice or not at all.
func (s *STS) startTurn(event serverEvent) {
	id := ""
	if event.Response != nil {
		id = event.Response.ID
	}

	s.settleTurn(true, sts.Usage{})

	s.mu.Lock()
	s.generation++
	turn := sts.NewTurn(id, s.generation, s.heardAt)
	s.heardAt = time.Time{}
	s.itemID = ""
	s.turn = turn
	s.mu.Unlock()

	s.emitter.Send(sts.ResponseStarted{ResponseID: turn.ID, Generation: turn.Generation, At: time.Now()})
}

// speak forwards a piece of the reply's audio.
func (s *STS) speak(event serverEvent) {
	raw, err := base64.StdEncoding.DecodeString(event.Delta)
	if err != nil {
		s.logger.Debug("undecodable audio", "error", err)
		return
	}

	s.mu.Lock()
	if event.ItemID != "" {
		s.itemID = event.ItemID
	}
	turn := s.turn
	if turn == nil {
		// Audio without a response.created before it: the server skipped the opening
		// frame, which is not a reason to drop what the model said.
		s.generation++
		turn = sts.NewTurn(event.ResponseID, s.generation, s.heardAt)
		s.heardAt = time.Time{}
		s.turn = turn
		s.mu.Unlock()
		s.emitter.Send(sts.ResponseStarted{ResponseID: turn.ID, Generation: turn.Generation, At: time.Now()})
	} else {
		s.mu.Unlock()
	}

	s.settleHeard()
	pcm := audio.FromBytes(raw, s.options.Vendor.OutputSampleRate, 1)
	s.emitter.Send(turn.Chunk(pcm))
}

// finishTurn settles the reply the server has closed, with what it cost and whether the
// caller cut it off.
func (s *STS) finishTurn(event serverEvent) {
	var status, reason string
	var cost sts.Usage
	var failure *apiError
	if event.Response != nil {
		status = event.Response.Status
		cost = usageOf(event.Response.Usage)
		if event.Response.StatusDetails != nil {
			reason = event.Response.StatusDetails.Reason
			failure = event.Response.StatusDetails.Error
		}
	}

	if status == "failed" {
		err := error(failure)
		if failure == nil {
			err = errors.New("the response failed")
		}
		s.emitter.Send(sts.Error{
			Provider:   s.options.Vendor.Provider,
			Model:      s.options.Model,
			ResponseID: s.responseID(event.Response.ID),
			Err:        err,
			Context:    "response",
		})
	}

	interrupted := status == "cancelled" || reason == "turn_detected" || reason == "client_cancelled"
	s.settleTurn(interrupted, cost)
}

// settleTurn closes the reply in flight, if there is one, and reports it.
func (s *STS) settleTurn(interrupted bool, cost sts.Usage) {
	s.mu.Lock()
	turn := s.turn
	s.turn = nil
	s.itemID = ""
	s.mu.Unlock()
	if turn == nil {
		return
	}
	s.emitter.Send(turn.Complete(s.options.Vendor.Provider, s.options.Model, interrupted, cost))
}

// fail reports a server error. A cancel that found no reply to cancel is not one.
func (s *STS) fail(event serverEvent) {
	if event.Error == nil {
		return
	}
	if event.Error.Code == cancelNotActive {
		s.logger.Debug("nothing to cancel")
		return
	}
	s.emitter.Send(sts.Error{
		Provider: s.options.Vendor.Provider,
		Model:    s.options.Model,
		Err:      event.Error,
		Context:  "server",
	})
}

// responseID is the id of the reply an event belongs to, which is the event's own when it
// names one and the reply in flight otherwise.
func (s *STS) responseID(named string) string {
	if named != "" {
		return named
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.turn == nil {
		return ""
	}
	return s.turn.ID
}

// speaker is who the most recent audio came from.
func (s *STS) speaker() sts.Participant {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.participant
}

// usageOf is what a reply cost in this package's words.
func usageOf(reported *usage) sts.Usage {
	if reported == nil {
		return sts.Usage{}
	}
	cost := sts.Usage{InputTokens: reported.InputTokens, OutputTokens: reported.OutputTokens}
	if reported.InputTokenDetails != nil {
		cost.CachedInputTokens = reported.InputTokenDetails.CachedTokens
		cost.InputAudioTokens = reported.InputTokenDetails.AudioTokens
	}
	if reported.OutputTokenDetails != nil {
		cost.OutputAudioTokens = reported.OutputTokenDetails.AudioTokens
	}
	return cost
}
