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
// The Live API has no response ids. A reply is whatever the model says between one
// turnComplete and the next, assigned an ID here, except that a turnComplete sent while the
// interaction is still in progress does not end it. The caller is taken to have stopped where
// the server's own detector said their speech ended, and at the last piece of transcript
// before the model began when it said nothing.
package gemini

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/keypool"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// ProviderName is the stable name used in routing config and stats.
const ProviderName = "gemini"

// DefaultModel is Gemini 3.8 Live, the model Google now point a voice agent at. It took
// over from 3.1 Flash Live, a legacy preview Google may shut down from 17 November 2026.
const DefaultModel = "gemini-3.8-live"

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

// requiredThinking is the level a model that refuses a setup without one is given when
// nobody asked: its floor, since thinking spends the latency budget before the first word.
// 3.8 Live Extended Thinking takes LOW, MEDIUM or HIGH and closes the socket otherwise.
var requiredThinking = map[string]string{"gemini-3.8-live-extended-thinking": "LOW"}

// nonBlockingModels are the models whose NON_BLOCKING calls have been heard keeping the
// conversation going and saying an update's detail (probed 2026-10-24). 3.8 Live Extended
// Thinking times out answering a tool call either way, so it is left blocking.
var nonBlockingModels = map[string]bool{"gemini-3.1-flash-live-preview": true, "gemini-3.8-live": true}

// TakesNonBlocking reports whether a model's tools are best declared NON_BLOCKING.
func TakesNonBlocking(model string) bool { return nonBlockingModels[model] }

// activityStart and activityEnd are the detector's "Start of sentence" and "End of sentence"
// signals (https://ai.google.dev/api/live#VoiceActivity). The end arrives with the reply
// (probed 2026-10-20 on gemini-3.8-live and 3.1-flash-live-preview), so only its audioOffset
// says when the caller stopped.
const (
	activityStart = "ACTIVITY_START"
	activityEnd   = "ACTIVITY_END"
)

// interactionInProgress is what a model that reasons or runs tools in the background sends
// with a turnComplete that is not the end of its answer: it spoke a filler, and more of the
// same reply is coming.
const interactionInProgress = "IN_PROGRESS"

// schedulingWhenIdle has the model take a NON_BLOCKING call's response up once it is not
// speaking, rather than cutting itself off for it.
const schedulingWhenIdle = "WHEN_IDLE"

// nonBlockingNote is what a NON_BLOCKING tool's declaration adds, since the model keeps
// talking while the call runs.
const nonBlockingNote = "Its result comes back later: say you are on it, keep the conversation going, " +
	"and do not guess the result."

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
	// NonBlocking declares the tools NON_BLOCKING: the model keeps talking while a call runs,
	// can be told how it is going (UpdateCall), and hears its result when it is next idle.
	// Only for a model that takes such declarations.
	NonBlocking bool
	// SilenceMs and PrefixPaddingMs tune the model's own silence timer.
	SilenceMs       *int
	PrefixPaddingMs *int
	// StartSensitivity and EndSensitivity are the detector's named sensitivities, such
	// as START_SENSITIVITY_HIGH, which is how this vendor spells a threshold.
	StartSensitivity string
	EndSensitivity   string
	// ThinkingLevel is how long a 3.1 model may think before speaking. Empty leaves it,
	// or takes the floor for a model that will not open without one.
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
	Behavior    string         `json:"behavior,omitempty"`
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
	// WillContinue and Scheduling answer a NON_BLOCKING call: whether more responses follow,
	// and when the model is to take this one up.
	WillContinue *bool  `json:"willContinue,omitempty"`
	Scheduling   string `json:"scheduling,omitempty"`
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
	VoiceActivity           *voiceActivity        `json:"voiceActivity"`
}

// voiceActivity is the server's detector at work. audioOffset is "relative to the start of
// the audio stream" (https://ai.google.dev/api/live#VoiceActivity), a Duration such as
// "4.600s".
type voiceActivity struct {
	Type        string `json:"type"`
	AudioOffset string `json:"audioOffset"`
}

type serverContent struct {
	ModelTurn           *content       `json:"modelTurn"`
	InputTranscription  *transcription `json:"inputTranscription"`
	OutputTranscription *transcription `json:"outputTranscription"`
	TurnComplete        bool           `json:"turnComplete"`
	GenerationComplete  bool           `json:"generationComplete"`
	Interrupted         bool           `json:"interrupted"`
	InteractionStatus   string         `json:"interactionStatus"`
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
	ThoughtsTokenCount      int64            `json:"thoughtsTokenCount"`
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
	keys    *keypool.Pool
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
	// lastHeardAt when the last piece arrived. endedAt is when the server's detector says
	// the caller stopped, which lastHeardAt stands in for when it said nothing.
	hearing     strings.Builder
	listening   bool
	lastHeardAt time.Time
	endedAt     time.Time
	heardAt     time.Time
	// sentMs is how much audio this connection has carried, and sentAt when its last chunk
	// went, which together turn an audioOffset into the moment that audio was sent.
	sentMs float64
	sentAt time.Time
	// turn is the reply in flight, generation counts them, and usage is what the server
	// has reported for the turn so far. carried is what the earlier turns of a reply that
	// spans several cost, since the server reports each turn on its own.
	turn       *sts.Turn
	generation int
	usage      sts.Usage
	carried    sts.Usage
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
	keys := keypool.FromEnv(settings.APIKey, apiKeyEnvVar).Pin()
	if keys.Len() == 0 {
		return nil, stack.Wrap(fmt.Errorf("gemini: api key is required (set %s)", apiKeyEnvVar))
	}
	settings.APIKey = keys.Peek()
	if settings.Model == "" {
		settings.Model = DefaultModel
	}
	if settings.ThinkingLevel == "" {
		settings.ThinkingLevel = requiredThinking[settings.Model]
	}
	if settings.URL == "" {
		settings.URL = DefaultURL
	}
	if !strings.HasPrefix(settings.URL, "ws://") && !strings.HasPrefix(settings.URL, "wss://") {
		return nil, stack.Wrap(fmt.Errorf("gemini: url must be ws:// or wss://, got %s", settings.URL))
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
		return stack.Wrap(errors.New("gemini: already started"))
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
	conn, response, _, err := s.keys.Dial(ctx, dialer, func(key string) (string, http.Header) {
		return s.endpoint(key), nil
	})
	if err != nil {
		if response != nil {
			return nil, stack.Wrap(fmt.Errorf("gemini: dial: %w (http %d)", err, response.StatusCode))
		}
		return nil, stack.Wrap(fmt.Errorf("gemini: dial: %w", err))
	}
	stop := context.AfterFunc(ctx, func() { _ = conn.Close() })
	defer stop()
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

	if err := s.send(clientMessage{RealtimeInput: &realtimeInput{Audio: &blob{
		Data:     base64.StdEncoding.EncodeToString(pcm.Bytes()),
		MimeType: audioMimeType,
	}}}); err != nil {
		return err
	}
	s.mu.Lock()
	s.sentMs += pcm.DurationMs()
	s.sentAt = time.Now()
	s.mu.Unlock()
	return nil
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
func (s *STS) SetInstructions(string) error { return stack.Wrap(sts.ErrInstructionsFixed) }

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
	answer := functionResponse{ID: callID, Name: name, Response: response}
	if s.options.NonBlocking {
		closed := false
		answer.WillContinue, answer.Scheduling = &closed, schedulingWhenIdle
	}
	return s.send(clientMessage{ToolResponse: &toolResponse{FunctionResponses: []functionResponse{answer}}})
}

// UpdatesCalls reports whether the session's calls are NON_BLOCKING, so an open one can be
// told of.
func (s *STS) UpdatesCalls() bool { return s.options.NonBlocking && len(s.options.Tools) > 0 }

// UpdateCall tells the model how an open call is going with a partial response, which it
// takes up when it is next idle. The call stays open for its Answer.
func (s *STS) UpdateCall(update sts.CallUpdate) error {
	if !s.UpdatesCalls() {
		return stack.Wrap(sts.ErrNoCallUpdates)
	}
	s.mu.Lock()
	name := s.calls[update.CallID]
	s.mu.Unlock()
	if name == "" {
		return fmt.Errorf("gemini: no tool call %q is open", update.CallID)
	}
	open := true
	return s.send(clientMessage{ToolResponse: &toolResponse{FunctionResponses: []functionResponse{{
		ID: update.CallID, Name: name, WillContinue: &open, Scheduling: schedulingWhenIdle,
		Response: map[string]any{"status": update.Status, "instructions": update.Guidance},
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

// Voice is the prebuilt voice asked for, or empty when the vendor picks it.
func (s *STS) Voice() string { return s.options.Voice }

// SampleRate is the rate the model speaks at.
func (s *STS) SampleRate() int { return OutputSampleRate }

// Capabilities implements sts.STS.
func (s *STS) Capabilities() sts.Capabilities { return CapabilitiesFor(s.options.Model) }

// endpoint is the socket with key attached, which is how this API authenticates.
func (s *STS) endpoint(key string) string {
	separator := "?"
	if strings.Contains(s.options.URL, "?") {
		separator = "&"
	}
	return s.options.URL + separator + "key=" + url.QueryEscape(key)
}

// handshake configures the session and waits for the server to accept it.
func (s *STS) handshake(conn *websocket.Conn, handle string) error {
	payload, err := json.Marshal(clientMessage{Setup: s.setup(handle)})
	if err != nil {
		return stack.Wrap(err)
	}
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return stack.Wrap(fmt.Errorf("gemini: send setup: %w", err))
	}

	if err := conn.SetReadDeadline(time.Now().Add(s.options.HandshakeTimeout)); err != nil {
		return stack.Wrap(fmt.Errorf("gemini: read setup: %w", err))
	}
	_, raw, err := conn.ReadMessage()
	if err != nil {
		return stack.Wrap(fmt.Errorf("gemini: read setup: %w", err))
	}
	if err := conn.SetReadDeadline(time.Time{}); err != nil {
		return stack.Wrap(fmt.Errorf("gemini: read setup: %w", err))
	}

	var message serverMessage
	if err := json.Unmarshal(raw, &message); err != nil {
		return stack.Wrap(fmt.Errorf("gemini: decode setup: %w", err))
	}
	if message.SetupComplete == nil {
		return stack.Wrap(fmt.Errorf("gemini: setup rejected: %s", strings.TrimSpace(string(raw))))
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
			declared := functionDeclaration{
				Name:        offered.Name,
				Description: offered.Description,
				Parameters:  offered.Parameters,
			}
			if s.options.NonBlocking {
				declared.Behavior = "NON_BLOCKING"
				declared.Description = strings.TrimSpace(declared.Description + " " + nonBlockingNote)
			}
			declarations = append(declarations, declared)
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
		return stack.Wrap(errors.New("gemini: session closed"))
	}
	if !started || conn == nil {
		return stack.Wrap(errors.New("gemini: not started"))
	}

	payload, err := json.Marshal(frame)
	if err != nil {
		return stack.Wrap(err)
	}
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	if err := conn.WriteMessage(websocket.TextMessage, payload); err != nil {
		return stack.Wrap(fmt.Errorf("gemini: write: %w", err))
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
		// Whether a resumed connection restarts the offsets is not documented. Counting
		// afresh is right if it does, and if it does not, detected drops offsets past what
		// this connection sent.
		s.sentMs = 0
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
	// Before the content and tool calls, which settle the caller's turn: the end of speech
	// can come in the same frame as the reply.
	if message.VoiceActivity != nil {
		s.detected(*message.VoiceActivity)
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
		// The words can come before the audio, so they settle the caller's turn first,
		// as spoke and asked do: a reply opened while the caller is still heard would
		// begin before their wait did and be timed from the turn before. The rest of a
		// reply the caller cut off is dropped first, as in spoke: it is not theirs to end.
		s.mu.Lock()
		muted := s.muted
		s.mu.Unlock()
		if !muted {
			s.finishHearing()
			turn := s.currentTurn()
			s.emitter.Send(sts.OutputTranscript{ResponseID: turn.ID, Mode: stt.ModeDelta, Text: server.OutputTranscription.Text})
		}
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
	// A turn that ends with the interaction still in progress is a filler spoken while
	// the model reasons or waits on a tool, so the reply stays open and a local interrupt
	// in the pause still has something to cut off.
	if server.TurnComplete && server.InteractionStatus == interactionInProgress {
		s.finishHearing()
		s.carryUsage()
	} else if server.TurnComplete {
		s.finishHearing()
		s.settleTurn(false)
		s.unmute()
	}
}

// carryUsage keeps what the turn that just ended cost for the reply it belongs to.
func (s *STS) carryUsage() {
	s.mu.Lock()
	if s.turn != nil {
		s.carried = sumUsage(s.carried, s.usage)
	}
	s.usage = sts.Usage{}
	s.mu.Unlock()
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

// detected keeps the moment the server's detector says the caller stopped: the wall time the
// audio at its offset was sent. A start clears an end the turn before left unused.
func (s *STS) detected(activity voiceActivity) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if activity.Type == activityStart {
		s.endedAt = time.Time{}
		return
	}
	if activity.Type != activityEnd {
		return
	}
	offset, err := time.ParseDuration(activity.AudioOffset)
	behindMs := s.sentMs - float64(offset.Microseconds())/1000
	// An offset past what this connection sent counts from an earlier one, so it says
	// nothing about when this audio went.
	if err != nil || behindMs < 0 {
		s.logger.Debug("unplaceable end of speech", "offset", activity.AudioOffset, "sent_ms", s.sentMs)
		return
	}
	s.endedAt = s.sentAt.Add(-time.Duration(behindMs * float64(time.Millisecond)))
}

// finishHearing settles the caller's turn: the transcript so far becomes the final one, and
// the detector's end of speech, or else the last piece heard, is the moment they stopped.
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
	if !s.endedAt.IsZero() {
		stoppedAt = s.endedAt
		s.endedAt = time.Time{}
	}
	// A reply already open takes the stop: none that follows answers it.
	if s.turn != nil {
		s.turn.HeardAt(stoppedAt)
	} else {
		s.heardAt = stoppedAt
	}
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
	// The rest of a reply the caller cut off from this side is not a new reply, so it is
	// dropped before it can end the caller's turn or open one.
	s.mu.Lock()
	muted := s.muted
	s.mu.Unlock()
	if muted {
		return
	}
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
	usage := sumUsage(s.carried, s.usage)
	s.turn = nil
	s.usage = sts.Usage{}
	s.carried = sts.Usage{}
	s.mu.Unlock()
	if turn == nil {
		return
	}
	s.emitter.Send(turn.Complete(ProviderName, s.options.Model, interrupted, usage))
}

// usageOf is what the server's count means in this package's words. Google bills each
// modality at its own rate, on the details rather than promptTokenCount, which they fall
// short of. Thinking is not in responseTokenCount and is billed as text output, as is any
// part of the response the details leave out, such as a function call.
func usageOf(reported usageMetadata) sts.Usage {
	cost := sts.Usage{
		InputTokens:       reported.PromptTokenCount,
		CachedInputTokens: reported.CachedContentTokenCount,
		OutputTokens:      reported.ResponseTokenCount + reported.ThoughtsTokenCount,
		ReasoningTokens:   reported.ThoughtsTokenCount,
	}
	for _, detail := range reported.PromptTokensDetails {
		switch detail.Modality {
		case "TEXT":
			cost.InputTextTokens += detail.TokenCount
		case "AUDIO":
			cost.InputAudioTokens += detail.TokenCount
		// Google price video at the image rate.
		case "IMAGE", "VIDEO":
			cost.InputImageTokens += detail.TokenCount
		}
	}
	// A prompt with no details is priced at the audio rate, the dearest and what a prompt
	// with no split costs: summed with a turn that has them, it would otherwise go unbilled.
	if len(reported.PromptTokensDetails) == 0 {
		cost.InputAudioTokens = reported.PromptTokenCount
	}
	for _, detail := range reported.ResponseTokensDetails {
		switch detail.Modality {
		case "TEXT":
			cost.OutputTextTokens += detail.TokenCount
		case "AUDIO":
			cost.OutputAudioTokens += detail.TokenCount
		}
	}
	undescribed := max(reported.ResponseTokenCount-cost.OutputTextTokens-cost.OutputAudioTokens, 0)
	cost.OutputTextTokens += undescribed + reported.ThoughtsTokenCount
	return cost
}

// sumUsage adds two turns' usage, as a reply that spans several is billed for each.
func sumUsage(a, b sts.Usage) sts.Usage {
	return sts.Usage{
		InputTokens:            a.InputTokens + b.InputTokens,
		CachedInputTokens:      a.CachedInputTokens + b.CachedInputTokens,
		OutputTokens:           a.OutputTokens + b.OutputTokens,
		InputTextTokens:        a.InputTextTokens + b.InputTextTokens,
		InputAudioTokens:       a.InputAudioTokens + b.InputAudioTokens,
		InputImageTokens:       a.InputImageTokens + b.InputImageTokens,
		CachedInputTextTokens:  a.CachedInputTextTokens + b.CachedInputTextTokens,
		CachedInputAudioTokens: a.CachedInputAudioTokens + b.CachedInputAudioTokens,
		CachedInputImageTokens: a.CachedInputImageTokens + b.CachedInputImageTokens,
		OutputTextTokens:       a.OutputTextTokens + b.OutputTextTokens,
		OutputAudioTokens:      a.OutputAudioTokens + b.OutputAudioTokens,
		ReasoningTokens:        a.ReasoningTokens + b.ReasoningTokens,
	}
}
