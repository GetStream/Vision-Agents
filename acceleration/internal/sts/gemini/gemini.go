// Package gemini implements the sts.STS contract on top of the Gemini Live API's native
// audio models, over the BidiGenerateContent WebSocket.
//
// It is the same socket the Gemini transcriber under internal/stt opens, with the talking
// half turned on: the session is set up to answer in audio, to write down what it heard and
// what it said, and to call the tools it was given. The two packages share no code on
// purpose. The transcriber never reconnects, because a transcript that resumes mid-turn is
// worse than one that restarts; a conversation is the other way round, so this package owns
// a resume loop that the other must not have.
//
// The Live API has no response ids and no event for the caller falling silent. A reply is
// whatever the model says between one turnComplete and the next, numbered here; the caller
// is taken to have stopped at the last piece of transcript before the model began.
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

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// DefaultModel is Google's highest-quality native audio model.
const DefaultModel = "gemini-3.1-flash-live-preview"

// DefaultURL is the Live API socket. The key goes on the query string, which is the only
// authentication this endpoint takes.
const DefaultURL = "wss://generativelanguage.googleapis.com/ws/" +
	"google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"

// OutputSampleRate is the one rate the Live API speaks at.
const OutputSampleRate = 24_000

// apiKeyEnvVar holds the credentials when Options does not.
const apiKeyEnvVar = "GOOGLE_API_KEY"

// audioMimeType describes what ProcessAudio sends. The rate is part of the type rather
// than a field of its own.
var audioMimeType = fmt.Sprintf("audio/pcm;rate=%d", sts.InputSampleRate)

// maxDuration is how long Google lets an audio-only session run. The connection itself is
// cut roughly every ten minutes and resumed with a handle, which the provider does itself.
const maxDuration = 15 * time.Minute

// resumeAttempts is how many times a dropped connection is dialled again before the
// conversation is given up on.
const resumeAttempts = 3

// Options configures the provider. APIKey falls back to GOOGLE_API_KEY.
type Options struct {
	APIKey string
	Model  string
	URL    string
	// Voice is one of the prebuilt voices, such as Kore or Leda. Empty leaves the
	// model's own.
	Voice string
	// Language is the BCP-47 code the model speaks in, such as en-US. Empty leaves the
	// model to follow the caller.
	Language string
	// Instructions is the system prompt, which this API takes once, at setup.
	Instructions string
	// Tools are what the model may call, also taken once.
	Tools []llm.Tool
	// SilenceMs and PrefixPaddingMs tune the model's own silence timer.
	SilenceMs       *int
	PrefixPaddingMs *int
	// StartSensitivity and EndSensitivity are the detector's named sensitivities, such
	// as START_SENSITIVITY_HIGH, which is how this vendor spells a threshold.
	StartSensitivity string
	EndSensitivity   string
	// ThinkingLevel is how long a 3.1 model may think before speaking. Empty leaves it.
	ThinkingLevel string
	// ProactiveAudio and AffectiveDialog are the 2.5 models' extras.
	ProactiveAudio  *bool
	AffectiveDialog *bool
	// MediaResolution is how much of each frame the model looks at.
	MediaResolution string
	// InputTranscript and OutputTranscript ask the model to write down what it heard and
	// what it said.
	InputTranscript  bool
	OutputTranscript bool
	// HandshakeTimeout bounds the connect and the setup exchange.
	HandshakeTimeout time.Duration
	Logger           *slog.Logger
}

// clientMessage is a frame sent to the Live API. Exactly one field is ever set.
type clientMessage struct {
	Setup         *setup         `json:"setup,omitempty"`
	ClientContent *clientContent `json:"clientContent,omitempty"`
	RealtimeInput *realtimeInput `json:"realtimeInput,omitempty"`
	ToolResponse  *toolResponse  `json:"toolResponse,omitempty"`
}

// setup is the first frame, which configures the session.
type setup struct {
	Model                    string               `json:"model"`
	GenerationConfig         generationConfig     `json:"generationConfig"`
	SystemInstruction        *content             `json:"systemInstruction,omitempty"`
	Tools                    []toolDeclarations   `json:"tools,omitempty"`
	InputAudioTranscription  *struct{}            `json:"inputAudioTranscription,omitempty"`
	OutputAudioTranscription *struct{}            `json:"outputAudioTranscription,omitempty"`
	RealtimeInputConfig      *realtimeInputConfig `json:"realtimeInputConfig,omitempty"`
	SessionResumption        *sessionResumption   `json:"sessionResumption,omitempty"`
	Proactivity              *proactivity         `json:"proactivity,omitempty"`
	EnableAffectiveDialog    *bool                `json:"enableAffectiveDialog,omitempty"`
}

type generationConfig struct {
	ResponseModalities []string        `json:"responseModalities"`
	SpeechConfig       *speechConfig   `json:"speechConfig,omitempty"`
	ThinkingConfig     *thinkingConfig `json:"thinkingConfig,omitempty"`
	MediaResolution    string          `json:"mediaResolution,omitempty"`
}

type speechConfig struct {
	VoiceConfig  *voiceConfig `json:"voiceConfig,omitempty"`
	LanguageCode string       `json:"languageCode,omitempty"`
}

type voiceConfig struct {
	PrebuiltVoiceConfig prebuiltVoice `json:"prebuiltVoiceConfig"`
}

type prebuiltVoice struct {
	VoiceName string `json:"voiceName"`
}

type thinkingConfig struct {
	ThinkingLevel string `json:"thinkingLevel,omitempty"`
}

type proactivity struct {
	ProactiveAudio bool `json:"proactiveAudio"`
}

// content is a turn: whose it is and what it carries.
type content struct {
	Role  string `json:"role,omitempty"`
	Parts []part `json:"parts"`
}

type part struct {
	Text       string `json:"text,omitempty"`
	InlineData *blob  `json:"inlineData,omitempty"`
	Thought    bool   `json:"thought,omitempty"`
}

type blob struct {
	Data     string `json:"data"`
	MimeType string `json:"mimeType"`
}

type toolDeclarations struct {
	FunctionDeclarations []functionDeclaration `json:"functionDeclarations"`
}

type functionDeclaration struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

type realtimeInputConfig struct {
	AutomaticActivityDetection *activityDetection `json:"automaticActivityDetection,omitempty"`
}

type activityDetection struct {
	StartOfSpeechSensitivity string `json:"startOfSpeechSensitivity,omitempty"`
	EndOfSpeechSensitivity   string `json:"endOfSpeechSensitivity,omitempty"`
	PrefixPaddingMs          *int   `json:"prefixPaddingMs,omitempty"`
	SilenceDurationMs        *int   `json:"silenceDurationMs,omitempty"`
}

type sessionResumption struct {
	Handle string `json:"handle,omitempty"`
}

// clientContent is a turn typed rather than spoken.
type clientContent struct {
	Turns        []content `json:"turns"`
	TurnComplete bool      `json:"turnComplete"`
}

// realtimeInput carries audio, a frame, or the note that there will be no more audio.
type realtimeInput struct {
	Audio          *blob `json:"audio,omitempty"`
	Video          *blob `json:"video,omitempty"`
	AudioStreamEnd bool  `json:"audioStreamEnd,omitempty"`
}

type toolResponse struct {
	FunctionResponses []functionResponse `json:"functionResponses"`
}

type functionResponse struct {
	ID       string         `json:"id"`
	Name     string         `json:"name"`
	Response map[string]any `json:"response"`
}

// serverMessage is a frame sent by the Live API.
type serverMessage struct {
	SetupComplete           *json.RawMessage      `json:"setupComplete"`
	ServerContent           *serverContent        `json:"serverContent"`
	ToolCall                *toolCall             `json:"toolCall"`
	ToolCallCancellation    *toolCallCancellation `json:"toolCallCancellation"`
	UsageMetadata           *usageMetadata        `json:"usageMetadata"`
	GoAway                  *goAway               `json:"goAway"`
	SessionResumptionUpdate *resumptionUpdate     `json:"sessionResumptionUpdate"`
}

type serverContent struct {
	ModelTurn           *content       `json:"modelTurn"`
	InputTranscription  *transcription `json:"inputTranscription"`
	OutputTranscription *transcription `json:"outputTranscription"`
	TurnComplete        bool           `json:"turnComplete"`
	GenerationComplete  bool           `json:"generationComplete"`
	Interrupted         bool           `json:"interrupted"`
}

type transcription struct {
	Text string `json:"text"`
}

type toolCall struct {
	FunctionCalls []functionCall `json:"functionCalls"`
}

type functionCall struct {
	ID   string          `json:"id"`
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type toolCallCancellation struct {
	IDs []string `json:"ids"`
}

type usageMetadata struct {
	PromptTokenCount        int64            `json:"promptTokenCount"`
	CachedContentTokenCount int64            `json:"cachedContentTokenCount"`
	ResponseTokenCount      int64            `json:"responseTokenCount"`
	PromptTokensDetails     []modalityTokens `json:"promptTokensDetails"`
	ResponseTokensDetails   []modalityTokens `json:"responseTokensDetails"`
}

type modalityTokens struct {
	Modality   string `json:"modality"`
	TokenCount int64  `json:"tokenCount"`
}

type goAway struct {
	TimeLeft string `json:"timeLeft"`
}

type resumptionUpdate struct {
	NewHandle string `json:"newHandle"`
	Resumable bool   `json:"resumable"`
}

// STS is a Gemini Live conversation.
type STS struct {
	options Options
	logger  *slog.Logger
	emitter *sts.Emitter

	mu sync.Mutex
	// conn is the live connection. It changes when the session is resumed, so every use
	// reads it under the lock.
	conn *websocket.Conn
	// writeMu serialises writes: a websocket connection allows only one writer.
	writeMu sync.Mutex

	// participant is the speaker of the most recent audio.
	participant sts.Participant
	// hearing is what the model has written down of the caller's current turn, and
	// lastHeardAt when the last piece arrived, which stands in for the moment the caller
	// stopped since this API never says.
	hearing     strings.Builder
	listening   bool
	lastHeardAt time.Time
	heardAt     time.Time
	// turn is the reply in flight, generation counts them, and usage is what the server
	// has reported for the turn so far.
	turn       *sts.Turn
	generation int
	usage      sts.Usage
	// muted drops the rest of a reply the caller cut off from this side, since the API
	// has no way to tell the model to stop.
	muted bool
	// calls remembers the name each tool call was made under, which the answer has to
	// carry back.
	calls map[string]string
	// handle is what the session can be resumed with, once the server has issued one.
	handle  string
	started bool
	closed  bool
}

// New validates the options and returns an unstarted provider.
func New(settings Options) (*STS, error) {
	if settings.APIKey == "" {
		settings.APIKey = os.Getenv(apiKeyEnvVar)
	}
	if settings.APIKey == "" {
		return nil, fmt.Errorf("gemini: api key is required (set %s)", apiKeyEnvVar)
	}
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.URL == "" {
		settings.URL = DefaultURL
	}
	if !strings.HasPrefix(settings.URL, "ws://") && !strings.HasPrefix(settings.URL, "wss://") {
		return nil, fmt.Errorf("gemini: url must be ws:// or wss://, got %s", settings.URL)
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
		calls:   map[string]string{},
	}, nil
}

// CapabilitiesFor is what a Live model can be asked for. Every native audio model here does
// the same things; what they take at setup they cannot be given again.
func CapabilitiesFor(model string) sts.Capabilities {
	return sts.Capabilities{
		InputModalities:  []string{options.ModalityImage},
		Text:             true,
		Tools:            true,
		InputTranscript:  true,
		OutputTranscript: true,
		Endpointing:      true,
		Usage:            true,
		Resumable:        true,
		MaxDuration:      maxDuration,
	}
}

// Start dials the Live API and completes the setup exchange. It returns once the server
// reports the session is ready for audio.
func (s *STS) Start(ctx context.Context) error {
	s.mu.Lock()
	if s.started {
		s.mu.Unlock()
		return errors.New("gemini: already started")
	}
	s.started = true
	s.mu.Unlock()

	conn, err := s.connect(ctx, "")
	if err != nil {
		return err
	}
	s.mu.Lock()
	s.conn = conn
	s.mu.Unlock()

	s.emitter.Send(sts.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
	go s.readLoop(conn)
	return nil
}

// connect dials and configures one connection, resuming an earlier session when given its
// handle.
func (s *STS) connect(ctx context.Context, handle string) (*websocket.Conn, error) {
	dialer := &websocket.Dialer{HandshakeTimeout: s.options.HandshakeTimeout}
	conn, response, err := dialer.DialContext(ctx, s.endpoint(), nil)
	if err != nil {
		if response != nil {
			return nil, fmt.Errorf("gemini: dial: %w (http %d)", err, response.StatusCode)
		}
		return nil, fmt.Errorf("gemini: dial: %w", err)
	}
	if err := s.handshake(conn, handle); err != nil {
		conn.Close()
		return nil, err
	}
	return conn, nil
}

// ProcessAudio streams one chunk of the caller's speech.
func (s *STS) ProcessAudio(pcm sts.PcmData, participant sts.Participant) error {
	if err := pcm.Validate(sts.InputSampleRate); err != nil {
		return fmt.Errorf("gemini: %w", err)
	}

	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()

	return s.send(clientMessage{RealtimeInput: &realtimeInput{Audio: &blob{
		Data:     base64.StdEncoding.EncodeToString(pcm.Bytes()),
		MimeType: audioMimeType,
	}}})
}

// SendText injects a typed turn, which the model answers as it would a spoken one.
func (s *STS) SendText(text string, participant sts.Participant) error {
	s.mu.Lock()
	s.participant = participant
	s.mu.Unlock()
	return s.turnOf(text)
}

// SendFrame offers the model a still image of what the caller is looking at.
func (s *STS) SendFrame(frame llm.ImagePart) error {
	if len(frame.Data) == 0 {
		return errors.New("gemini: a frame has to carry its bytes")
	}
	return s.send(clientMessage{RealtimeInput: &realtimeInput{Video: &blob{
		Data:     base64.StdEncoding.EncodeToString(frame.Data),
		MimeType: frame.MIME,
	}}})
}

// SetInstructions is refused: the Live API takes its system instruction at setup and
// nowhere else, and pretending otherwise would leave the caller believing the model had
// been told something it had not.
func (s *STS) SetInstructions(string) error { return sts.ErrInstructionsFixed }

// SetTools is refused for the same reason as SetInstructions.
func (s *STS) SetTools([]llm.Tool) error { return sts.ErrToolsFixed }

// Answer returns a tool's result. The model carries on by itself once it has it.
func (s *STS) Answer(callID string, output string, err error) error {
	s.mu.Lock()
	name := s.calls[callID]
	delete(s.calls, callID)
	s.mu.Unlock()
	if name == "" {
		return fmt.Errorf("gemini: no tool call %q is waiting for an answer", callID)
	}

	response := map[string]any{"output": output}
	if err != nil {
		response = map[string]any{"error": err.Error()}
	}
	return s.send(clientMessage{ToolResponse: &toolResponse{FunctionResponses: []functionResponse{{
		ID:       callID,
		Name:     name,
		Response: response,
	}}}})
}

// Prompt asks the model to speak, guided by the text.
func (s *STS) Prompt(text string) error { return s.turnOf(text) }

// Interrupt stops forwarding the reply in flight and settles it as cut off. The Live API
// has no word for a client-side interrupt, so the model is not told: it stops of its own
// accord when it hears the caller, which is what a barge-in sounds like to it anyway.
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

// SampleRate is the rate the model speaks at.
func (s *STS) SampleRate() int { return OutputSampleRate }

// Capabilities implements sts.STS.
func (s *STS) Capabilities() sts.Capabilities { return CapabilitiesFor(s.options.Model) }

// endpoint is the socket with the key attached, which is how this API authenticates.
func (s *STS) endpoint() string {
	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + "key=" + url.QueryEscape(s.options.APIKey)
}

// handshake configures the session and waits for the server to accept it.
func (s *STS) handshake(conn *websocket.Conn, handle string) error {
	payload, err := json.Marshal(clientMessage{Setup: s.setup(handle)})
	if err != nil {
		return err
	}
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("gemini: send setup: %w", err)
	}

	if err := conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return fmt.Errorf("gemini: read setup: %w", err)
	}
	_, raw, err := conn.ReadMessage()
	if err != nil {
		return fmt.Errorf("gemini: read setup: %w", err)
	}
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
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

// setup is the configuration frame: audio back, in the voice and language asked for, with
// the transcripts, tools and detector settings the session wants.
func (s *STS) setup(handle string) *setup {
	configured := &setup{
		Model:            "models/" + s.options.Model,
		GenerationConfig: generationConfig{ResponseModalities: []string{"AUDIO"}, MediaResolution: s.options.MediaResolution},
		// Always asked for, even without a handle: a session that never asked cannot be
		// resumed when its connection is cut.
		SessionResumption: &sessionResumption{Handle: handle},
	}
	if s.options.Voice != "" || s.options.Language != "" {
		configured.GenerationConfig.SpeechConfig = &speechConfig{LanguageCode: s.options.Language}
		if s.options.Voice != "" {
			configured.GenerationConfig.SpeechConfig.VoiceConfig = &voiceConfig{
				PrebuiltVoiceConfig: prebuiltVoice{VoiceName: s.options.Voice},
			}
		}
	}
	if s.options.ThinkingLevel != "" {
		configured.GenerationConfig.ThinkingConfig = &thinkingConfig{ThinkingLevel: s.options.ThinkingLevel}
	}
	if s.options.Instructions != "" {
		configured.SystemInstruction = &content{Parts: []part{{Text: s.options.Instructions}}}
	}
	if len(s.options.Tools) > 0 {
		declarations := make([]functionDeclaration, 0, len(s.options.Tools))
		for _, offered := range s.options.Tools {
			declarations = append(declarations, functionDeclaration{
				Name:        offered.Name,
				Description: offered.Description,
				Parameters:  offered.Parameters,
			})
		}
		configured.Tools = []toolDeclarations{{FunctionDeclarations: declarations}}
	}
	if s.options.InputTranscript {
		configured.InputAudioTranscription = &struct{}{}
	}
	if s.options.OutputTranscript {
		configured.OutputAudioTranscription = &struct{}{}
	}
	if s.options.SilenceMs != nil || s.options.PrefixPaddingMs != nil || s.options.StartSensitivity != "" || s.options.EndSensitivity != "" {
		configured.RealtimeInputConfig = &realtimeInputConfig{AutomaticActivityDetection: &activityDetection{
			StartOfSpeechSensitivity: s.options.StartSensitivity,
			EndOfSpeechSensitivity:   s.options.EndSensitivity,
			PrefixPaddingMs:          s.options.PrefixPaddingMs,
			SilenceDurationMs:        s.options.SilenceMs,
		}}
	}
	if s.options.ProactiveAudio != nil {
		configured.Proactivity = &proactivity{ProactiveAudio: *s.options.ProactiveAudio}
	}
	configured.EnableAffectiveDialog = s.options.AffectiveDialog
	return configured
}

// turnOf sends a typed turn and tells the model it is complete, which is what makes it
// answer.
func (s *STS) turnOf(text string) error {
	return s.send(clientMessage{ClientContent: &clientContent{
		Turns:        []content{{Role: "user", Parts: []part{{Text: text}}}},
		TurnComplete: true,
	}})
}

func (s *STS) send(frame clientMessage) error {
	s.mu.Lock()
	conn, started, closed := s.conn, s.started, s.closed
	s.mu.Unlock()
	if closed {
		return errors.New("gemini: session closed")
	}
	if !started || conn == nil {
		return errors.New("gemini: not started")
	}

	payload, err := json.Marshal(frame)
	if err != nil {
		return err
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return fmt.Errorf("gemini: write: %w", err)
	}
	return nil
}

// readLoop translates server frames into events until the connection ends, then resumes
// the session if it can.
func (s *STS) readLoop(conn *websocket.Conn) {
	for {
		_, raw, err := conn.ReadMessage()
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

// handleReadError resumes the session where the server has given it a handle, and reports
// the loss where it has not. A reply cut off by a dropped connection is settled either way,
// because the audio the model was about to send is not coming.
func (s *STS) handleReadError(err error) {
	s.mu.Lock()
	closed, handle := s.closed, s.handle
	s.mu.Unlock()
	if closed {
		return
	}

	s.settleTurn(true)
	if handle != "" && s.resume(handle) {
		return
	}
	if websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
		s.emitter.Send(sts.Disconnected{Provider: ProviderName, Model: s.options.Model, Clean: true, At: time.Now()})
		return
	}
	s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: err, Context: "read", Fatal: true})
}

// resume dials the session again with its handle, so the conversation carries on where it
// was. It reports whether it managed to.
func (s *STS) resume(handle string) bool {
	s.emitter.Send(sts.Disconnected{Provider: ProviderName, Model: s.options.Model, Reason: "resuming", Clean: true, At: time.Now()})

	var err error
	for attempt := 0; attempt < resumeAttempts; attempt++ {
		ctx, cancel := context.WithTimeout(context.Background(), s.options.HandshakeTimeout)
		var conn *websocket.Conn
		conn, err = s.connect(ctx, handle)
		cancel()
		if err != nil {
			s.logger.Warn("could not resume the session", "attempt", attempt+1, "error", err)
			time.Sleep(time.Duration(attempt+1) * 500 * time.Millisecond)
			continue
		}

		s.mu.Lock()
		if s.closed {
			s.mu.Unlock()
			conn.Close()
			return true
		}
		s.conn = conn
		s.mu.Unlock()

		s.emitter.Send(sts.Connected{Provider: ProviderName, Model: s.options.Model, At: time.Now()})
		go s.readLoop(conn)
		return true
	}

	s.emitter.Send(sts.Error{Provider: ProviderName, Model: s.options.Model, Err: err, Context: "resume", Fatal: true})
	return false
}

func (s *STS) handleMessage(message serverMessage) {
	if message.SessionResumptionUpdate != nil && message.SessionResumptionUpdate.Resumable {
		s.mu.Lock()
		s.handle = message.SessionResumptionUpdate.NewHandle
		s.mu.Unlock()
	}
	if message.GoAway != nil {
		if left, err := time.ParseDuration(message.GoAway.TimeLeft); err == nil {
			s.emitter.Send(sts.SessionExpiring{TimeLeft: left})
		} else {
			s.emitter.Send(sts.SessionExpiring{})
		}
	}
	if message.UsageMetadata != nil {
		s.mu.Lock()
		s.usage = usageOf(*message.UsageMetadata)
		s.mu.Unlock()
	}
	if message.ToolCallCancellation != nil {
		s.mu.Lock()
		for _, id := range message.ToolCallCancellation.IDs {
			delete(s.calls, id)
		}
		s.mu.Unlock()
		s.emitter.Send(sts.ToolCancel{CallIDs: message.ToolCallCancellation.IDs})
	}
	if message.ToolCall != nil {
		s.asked(*message.ToolCall)
	}
	if message.ServerContent != nil {
		s.handleContent(*message.ServerContent)
	}
}

// handleContent is where the conversation happens: what the caller said, what the model
// said back, and where each turn ended.
func (s *STS) handleContent(server serverContent) {
	if server.InputTranscription != nil && server.InputTranscription.Text != "" {
		s.heard(server.InputTranscription.Text)
	}
	if server.OutputTranscription != nil && server.OutputTranscription.Text != "" {
		turn := s.currentTurn()
		s.emitter.Send(sts.OutputTranscript{ResponseID: turn.ID, Mode: stt.ModeDelta, Text: server.OutputTranscription.Text})
	}
	if server.ModelTurn != nil {
		s.spoke(*server.ModelTurn)
	}
	// Either boundary is the model's turn ending, so whatever was being dropped after a
	// local interrupt has now all arrived, and the next turn starts unmuted.
	if server.Interrupted {
		s.settleTurn(true)
		s.unmute()
	}
	if server.TurnComplete {
		s.finishHearing()
		s.settleTurn(false)
		s.unmute()
	}
}

func (s *STS) unmute() {
	s.mu.Lock()
	s.muted = false
	s.mu.Unlock()
}

// heard is a piece of what the caller is saying. The first piece of a turn is also the
// only word this API gives that the caller started talking.
func (s *STS) heard(text string) {
	s.mu.Lock()
	first := !s.listening
	s.listening = true
	s.lastHeardAt = time.Now()
	s.hearing.WriteString(text)
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
	text := strings.TrimSpace(s.hearing.String())
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

// spoke forwards the audio in a model turn. Text parts are the model's thoughts and are
// not spoken, so they are not forwarded as anything.
func (s *STS) spoke(turn content) {
	s.finishHearing()

	for _, piece := range turn.Parts {
		if piece.InlineData == nil || !strings.HasPrefix(piece.InlineData.MimeType, "audio/") {
			continue
		}
		raw, err := base64.StdEncoding.DecodeString(piece.InlineData.Data)
		if err != nil {
			s.logger.Debug("undecodable audio", "error", err)
			continue
		}

		// The rest of a reply the caller cut off from this side is not a new reply, so
		// it is dropped before it can open one.
		s.mu.Lock()
		muted := s.muted
		s.mu.Unlock()
		if muted {
			continue
		}
		current := s.currentTurn()
		s.emitter.Send(current.Chunk(audio.FromBytes(raw, OutputSampleRate, 1)))
	}
}

// asked forwards the model's tool calls, remembering each one's name for its answer.
func (s *STS) asked(call toolCall) {
	s.finishHearing()
	turn := s.currentTurn()

	for _, function := range call.FunctionCalls {
		s.mu.Lock()
		s.calls[function.ID] = function.Name
		s.mu.Unlock()

		arguments := string(function.Args)
		if arguments == "" {
			arguments = "{}"
		}
		s.emitter.Send(sts.ToolCall{ResponseID: turn.ID, CallID: function.ID, Name: function.Name, Arguments: arguments})
	}
}

// currentTurn is the reply in flight, opened now if the model has begun one without any
// event saying so, which is how this API begins every reply.
func (s *STS) currentTurn() *sts.Turn {
	s.mu.Lock()
	if s.turn != nil {
		turn := s.turn
		s.mu.Unlock()
		return turn
	}
	s.generation++
	turn := sts.NewTurn("", s.generation, s.heardAt)
	s.heardAt = time.Time{}
	s.turn = turn
	s.mu.Unlock()

	s.emitter.Send(sts.ResponseStarted{ResponseID: turn.ID, Generation: turn.Generation, At: time.Now()})
	return turn
}

// settleTurn closes the reply in flight, if there is one, and reports it with what the
// server said it cost.
func (s *STS) settleTurn(interrupted bool) {
	s.mu.Lock()
	turn := s.turn
	usage := s.usage
	s.turn = nil
	s.usage = sts.Usage{}
	s.mu.Unlock()
	if turn == nil {
		return
	}
	s.emitter.Send(turn.Complete(ProviderName, s.options.Model, interrupted, usage))
}

// usageOf is what the server's count means in this package's words.
func usageOf(reported usageMetadata) sts.Usage {
	cost := sts.Usage{
		InputTokens:       reported.PromptTokenCount,
		CachedInputTokens: reported.CachedContentTokenCount,
		OutputTokens:      reported.ResponseTokenCount,
	}
	for _, detail := range reported.PromptTokensDetails {
		if detail.Modality == "AUDIO" {
			cost.InputAudioTokens += detail.TokenCount
		}
	}
	for _, detail := range reported.ResponseTokensDetails {
		if detail.Modality == "AUDIO" {
			cost.OutputAudioTokens += detail.TokenCount
		}
	}
	return cost
}
