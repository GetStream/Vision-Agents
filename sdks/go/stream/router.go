package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

// poll is how often a recording job is asked whether it is done. Transcription runs faster
// than real time, so asking every second costs nothing next to the work itself.
const poll = time.Second

// audio is how many frames of speech may be waiting to be read. A voice produces a few a
// second, so anybody keeping up never fills it.
const audio = 64

// Router is everything the acceleration backend routes, configured once.
//
// Each of the three streaming modalities has a Realtime session and a Recording job, and
// search has neither, because a question and its answer are one round trip. Everything the
// named config holds is a default that a per-call option overrides.
//
//	router := stream.Router{Config: "healthcare"}
//
//	transcript, err := router.STT().Recording(ctx, stream.Recorded{URL: "call.mp3"},
//	    &acceleration.SttOptions{Diarize: &yes})
type Router struct {
	// Config names a stored router config, by the name it was stored under or by its id.
	// Without one, every call says what it wants for itself.
	Config string
	// Tags are cost labels carried onto everything routed here, on top of the config's own.
	Tags map[string]string
	// Backend is where the router is and who is billed. Its zero value reads the
	// environment.
	Backend Backend
	// Logger is where a session reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// STT is transcription, live or from a recording.
func (r Router) STT() Transcribing { return Transcribing{router: r} }

// TTS is a voice, live or recorded.
func (r Router) TTS() Speaking { return Speaking{router: r} }

// LLM is the model that answers.
func (r Router) LLM() Answering { return Answering{router: r} }

// Search answers one question out of what is true now.
func (r Router) Search(
	ctx context.Context,
	query string,
	options *acceleration.SearchOptions,
) (*acceleration.SearchAnswer, error) {
	client, err := r.client()
	if err != nil {
		return nil, err
	}

	body := acceleration.SearchJSONRequestBody{Query: query, Options: options}
	r.name(&body.ConfigId)
	r.label(&body.Tags)

	found, err := client.SearchWithResponse(ctx, body)
	if err != nil {
		return nil, fmt.Errorf("stream: searching: %w", err)
	}
	if found.JSON200 == nil {
		return nil, refusal(found.Status(), found.JSON400, found.JSON401, found.JSON404)
	}
	return found.JSON200, nil
}

// Transcribing routes transcription.
type Transcribing struct{ router Router }

// Realtime opens a transcription socket, configured and ready for audio.
func (t Transcribing) Realtime(
	ctx context.Context,
	options *acceleration.SttOptions,
) (*Transcriber, error) {
	socket, err := t.router.open(ctx, "stt", Frame{"stt": block(options)})
	if err != nil {
		return nil, err
	}

	transcriber := &Transcriber{
		socket:      socket,
		transcripts: make(chan Transcript, audio),
	}
	go transcriber.read(t.router.logger())
	return transcriber, nil
}

// Recording transcribes a whole recording and returns the transcript.
//
// This is the non-realtime form, served by the batch half of a vendor rather than the
// streaming one, which is both cheaper and more accurate. It waits for the job unless
// Recorded.Callback is set, in which case it returns as soon as the job is accepted and the
// router calls back instead.
func (t Transcribing) Recording(
	ctx context.Context,
	recording Recorded,
	options *acceleration.SttOptions,
) (*acceleration.Transcription, error) {
	client, err := t.router.client()
	if err != nil {
		return nil, err
	}

	source, err := recording.source()
	if err != nil {
		return nil, err
	}

	body := acceleration.TranscribeRecordingJSONRequestBody{Source: source, Options: options}
	t.router.name(&body.ConfigId)
	t.router.label(&body.Tags)
	setString(&body.Callback, recording.Callback)

	accepted, err := client.TranscribeRecordingWithResponse(ctx, body)
	if err != nil {
		return nil, fmt.Errorf("stream: sending the recording: %w", err)
	}
	if accepted.JSON202 == nil {
		return nil, refusal(accepted.Status(), accepted.JSON400, accepted.JSON401, accepted.JSON404)
	}

	job := accepted.JSON202
	if recording.Callback != "" {
		return job, nil
	}

	for job.Status == acceleration.RecordingStatusQueued || job.Status == acceleration.RecordingStatusRunning {
		if err := wait(ctx); err != nil {
			return nil, err
		}

		asked, err := client.GetTranscriptionWithResponse(ctx, job.Id)
		if err != nil {
			return nil, fmt.Errorf("stream: asking about the recording: %w", err)
		}
		if asked.JSON200 == nil {
			return nil, refusal(asked.Status(), asked.JSON400, asked.JSON401, asked.JSON404)
		}
		job = asked.JSON200
	}
	if job.Status == acceleration.RecordingStatusFailed {
		return job, fmt.Errorf("stream: the recording failed: %s", value(job.Error))
	}
	return job, nil
}

// Speaking routes a voice.
type Speaking struct{ router Router }

// Realtime opens a speech socket, configured and ready for text.
func (s Speaking) Realtime(
	ctx context.Context,
	options *acceleration.TtsOptions,
) (*Voice, error) {
	socket, err := s.router.open(ctx, "tts", Frame{"tts": block(options)})
	if err != nil {
		return nil, err
	}

	voice := &Voice{socket: socket, audio: make(chan Audio, audio)}
	go voice.read(s.router.logger())
	return voice, nil
}

// Recording speaks a whole text into one file.
//
// Nothing is listening to an audiobook while it is being made, so this asks for the file
// rather than the stream, which is what lets a codec and a bitrate be chosen.
func (s Speaking) Recording(
	ctx context.Context,
	text string,
	options *acceleration.TtsOptions,
) (*acceleration.Speech, error) {
	client, err := s.router.client()
	if err != nil {
		return nil, err
	}

	body := acceleration.RecordSpeechJSONRequestBody{Text: text, Options: options}
	s.router.name(&body.ConfigId)
	s.router.label(&body.Tags)

	accepted, err := client.RecordSpeechWithResponse(ctx, body)
	if err != nil {
		return nil, fmt.Errorf("stream: sending the text: %w", err)
	}
	if accepted.JSON202 == nil {
		return nil, refusal(accepted.Status(), accepted.JSON400, accepted.JSON401, accepted.JSON404)
	}

	job := accepted.JSON202
	for job.Status == acceleration.RecordingStatusQueued || job.Status == acceleration.RecordingStatusRunning {
		if err := wait(ctx); err != nil {
			return nil, err
		}

		asked, err := client.GetSpeechWithResponse(ctx, job.Id)
		if err != nil {
			return nil, fmt.Errorf("stream: asking about the speech: %w", err)
		}
		if asked.JSON200 == nil {
			return nil, refusal(asked.Status(), asked.JSON400, asked.JSON401, asked.JSON404)
		}
		job = asked.JSON200
	}
	if job.Status == acceleration.RecordingStatusFailed {
		return job, fmt.Errorf("stream: the speech failed: %s", value(job.Error))
	}
	return job, nil
}

// Answering routes completions.
type Answering struct{ router Router }

// Realtime opens a completions socket, configured and ready for a question.
func (a Answering) Realtime(
	ctx context.Context,
	options *acceleration.LlmOptions,
) (*Model, error) {
	socket, err := a.router.open(ctx, "llm", Frame{"llm": block(options)})
	if err != nil {
		return nil, err
	}

	model := &Model{socket: socket, answers: make(chan Answer, events)}
	go model.read(a.router.logger())
	return model, nil
}

// Recorded is a whole recording to transcribe.
type Recorded struct {
	// URL is a fetchable audio or video file, which is what anything longer than a clip
	// should be: the provider fetches it itself.
	URL string
	// Audio is the file, for a caller with a clip and nowhere to host it.
	Audio []byte
	// Callback is a URL the finished job is POSTed to. Set it and Recording returns as soon
	// as the job is accepted rather than waiting for it.
	Callback string
}

// source is the recording as the router takes it: a URL, or the bytes in base64.
func (r Recorded) source() (acceleration.RecordingSource, error) {
	switch {
	case r.URL != "" && len(r.Audio) > 0:
		return acceleration.RecordingSource{}, errors.New("stream: a recording is either a url or the audio itself, not both")
	case r.URL != "":
		return acceleration.RecordingSource{Url: &r.URL}, nil
	case len(r.Audio) > 0:
		// The wire form is base64, which encoding/json does for a []byte on its own.
		return acceleration.RecordingSource{Audio: &r.Audio}, nil
	default:
		return acceleration.RecordingSource{}, errors.New("stream: a recording needs a url or the audio itself")
	}
}

// File reads a local recording, for a clip small enough to send inline.
func File(path string) (Recorded, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return Recorded{}, fmt.Errorf("stream: reading %s: %w", path, err)
	}
	return Recorded{Audio: contents}, nil
}

// Transcript is one thing the transcriber heard.
type Transcript struct {
	Text  string
	Final bool
	// Speaker is who said it, when diarization was asked for.
	Speaker  string
	Language string
	Error    string
	Frame    Frame
}

// Transcriber is one open transcription socket.
type Transcriber struct {
	socket      *Socket
	transcripts chan Transcript
}

// Send hands over 16 kHz mono PCM to be transcribed.
func (t *Transcriber) Send(pcm []byte) error { return t.socket.SendAudio(pcm) }

// Transcripts yields what was heard until the socket closes, when the channel closes.
func (t *Transcriber) Transcripts() <-chan Transcript { return t.transcripts }

// Close shuts the socket. Safe to call twice.
func (t *Transcriber) Close() error { return t.socket.Close() }

func (t *Transcriber) read(logger *slog.Logger) {
	defer close(t.transcripts)

	for {
		frame, _, err := t.socket.Read()
		if err != nil {
			ended(logger, "transcription", err)
			return
		}
		if frame == nil || frame.Type() == "" {
			continue
		}
		t.transcripts <- Transcript{
			Text:     frame.String("text"),
			Final:    frame.Bool("final"),
			Speaker:  frame.String("speaker"),
			Language: frame.String("language"),
			Error:    frame.String("error"),
			Frame:    frame,
		}
	}
}

// Audio is one piece of speech, as the provider produced it.
type Audio struct {
	// Samples are signed 16-bit little-endian PCM, preceded on the wire by a header saying
	// how to play them.
	Samples    []byte
	SampleRate int
	Channels   int
	// Done says the utterance is finished, and carries no samples.
	Done bool
	// Error is why nothing was spoken, when that is what happened.
	Error string
}

// Voice is one open speech socket.
type Voice struct {
	socket *Socket
	audio  chan Audio
}

// Speak says text, whose audio arrives on Audio. One utterance at a time: the frames come
// back bare, so two overlapping ones would be indistinguishable.
func (v *Voice) Speak(text string) error {
	return v.socket.Send(Frame{"type": "speak", "text": text, "final": true})
}

// Audio yields speech until the socket closes, when the channel closes.
func (v *Voice) Audio() <-chan Audio { return v.audio }

// Interrupt abandons what is being spoken.
func (v *Voice) Interrupt() error { return v.socket.Send(Frame{"type": "interrupt"}) }

// Close shuts the socket. Safe to call twice.
func (v *Voice) Close() error { return v.socket.Close() }

func (v *Voice) read(logger *slog.Logger) {
	defer close(v.audio)

	for {
		frame, payload, err := v.socket.Read()
		if err != nil {
			ended(logger, "speech", err)
			return
		}
		if payload != nil {
			v.audio <- pcm(payload)
			continue
		}
		switch frame.Type() {
		case "synthesis_complete":
			v.audio <- Audio{Done: true}
		case "error":
			v.audio <- Audio{Error: frame.String("error")}
		}
	}
}

// Question is what to answer and how, which is the response parameters the router speaks
// rather than a second vocabulary for the same things. What the config holds fills in
// whatever is left empty here.
type Question struct {
	// Instructions is what the model answers under.
	Instructions string
	// Messages is the conversation so far, oldest first.
	Messages []Said
	// Tools are what the model may ask to have run.
	Tools []acceleration.SessionTool
	// ToolChoice is auto, none, required, or the name of a tool it must call.
	ToolChoice string
	// MaxOutputTokens caps the reply. Zero leaves the model's own limit.
	MaxOutputTokens int
	// Temperature and ReasoningEffort are how varied and how considered the answer is.
	Temperature     *float64
	ReasoningEffort string
	// Format is text or json_object, and Verbosity how much of it there is.
	Format    string
	Verbosity string
	// PreviousResponseID continues from an earlier answer the provider still holds.
	PreviousResponseID string
	// PromptCacheKey is what a cached prompt prefix is keyed by.
	PromptCacheKey string
	// Metadata is passed to the provider untouched.
	Metadata map[string]string
}

// Said is one turn of the conversation.
type Said struct {
	Role    string
	Content string
}

// Answer is the reply arriving as it is written.
type Answer struct {
	// Delta is the next piece of text, on a delta.
	Delta string
	// Text is the whole answer, on the frame that finishes it.
	Text string
	Done bool
	// Error is why there is no answer, when that is what happened.
	Error string
	Frame Frame
}

// Model is one open completions socket.
type Model struct {
	socket  *Socket
	answers chan Answer
}

// Ask sends a question, whose answer arrives on Answers.
func (m *Model) Ask(question Question) error {
	said := make([]Frame, 0, len(question.Messages))
	for _, message := range question.Messages {
		said = append(said, Frame{"role": message.Role, "content": message.Content})
	}

	frame := Frame{"type": "respond", "messages": said}
	if question.Instructions != "" {
		frame["instructions"] = question.Instructions
	}
	if len(question.Tools) > 0 {
		frame["tools"] = question.Tools
	}
	if question.ToolChoice != "" {
		frame["tool_choice"] = question.ToolChoice
	}
	if question.MaxOutputTokens > 0 {
		frame["max_output_tokens"] = question.MaxOutputTokens
	}
	if question.Temperature != nil {
		frame["temperature"] = *question.Temperature
	}
	if question.ReasoningEffort != "" {
		frame["reasoning_effort"] = question.ReasoningEffort
	}
	if question.Format != "" {
		frame["format"] = question.Format
	}
	if question.Verbosity != "" {
		frame["verbosity"] = question.Verbosity
	}
	if question.PreviousResponseID != "" {
		frame["previous_response_id"] = question.PreviousResponseID
	}
	if question.PromptCacheKey != "" {
		frame["prompt_cache_key"] = question.PromptCacheKey
	}
	if len(question.Metadata) > 0 {
		frame["metadata"] = question.Metadata
	}
	return m.socket.Send(frame)
}

// Answers yields the reply until the socket closes, when the channel closes.
func (m *Model) Answers() <-chan Answer { return m.answers }

// Interrupt abandons the answer in flight.
func (m *Model) Interrupt() error { return m.socket.Send(Frame{"type": "interrupt"}) }

// Close shuts the socket. Safe to call twice.
func (m *Model) Close() error { return m.socket.Close() }

func (m *Model) read(logger *slog.Logger) {
	defer close(m.answers)

	for {
		frame, _, err := m.socket.Read()
		if err != nil {
			ended(logger, "completions", err)
			return
		}
		if frame == nil {
			continue
		}
		switch frame.Type() {
		case "delta":
			m.answers <- Answer{Delta: frame.String("text"), Frame: frame}
		case "complete":
			m.answers <- Answer{Text: frame.String("text"), Done: true, Frame: frame}
		case "error":
			m.answers <- Answer{Error: frame.String("error"), Frame: frame}
		}
	}
}

// open dials one modality socket and sends the start frame that says what it is for.
func (r Router) open(ctx context.Context, modality string, options Frame) (*Socket, error) {
	backend, err := r.Backend.Resolve()
	if err != nil {
		return nil, err
	}
	if r.Config == "" && !named(options) {
		return nil, errors.New("stream: routing needs a target, either in the options or in a config")
	}

	socket := NewSocket(
		backend.SocketURL("/v1/"+modality+"/stream"),
		backend.Headers(),
		backend.HTTPClient,
		r.logger(),
	)
	if err := socket.Open(ctx); err != nil {
		return nil, err
	}

	opening := Frame{"type": "start", "config_id": r.Config}
	if len(r.Tags) > 0 {
		opening["tags"] = r.Tags
	}
	for key, value := range options {
		opening[key] = value
	}
	if err := socket.Send(opening); err != nil {
		_ = socket.Close()
		return nil, fmt.Errorf("stream: opening the %s socket: %w", modality, err)
	}
	return socket, nil
}

// client is an HTTP client for the router this is configured against.
func (r Router) client() (*acceleration.ClientWithResponses, error) {
	return r.Backend.Client()
}

func (r Router) logger() *slog.Logger {
	if r.Logger == nil {
		return slog.Default()
	}
	return r.Logger
}

// name fills in the config a request is made under, when there is one.
func (r Router) name(field **string) { setString(field, r.Config) }

// label fills in the cost labels a request carries, when there are any.
func (r Router) label(field **map[string]string) {
	if len(r.Tags) > 0 {
		tags := r.Tags
		*field = &tags
	}
}

// block is one modality's options as the start frame carries them. A nil block sends an
// empty one, which is a socket that takes everything from its config.
func block[Options any](options *Options) any {
	if options == nil {
		return Frame{}
	}
	return options
}

// named reports whether the options say what to route to, which is what makes a socket
// without a config routable.
func named(options Frame) bool {
	for _, held := range options {
		switch typed := held.(type) {
		case *acceleration.SttOptions:
			return typed != nil && typed.Target != nil && *typed.Target != ""
		case *acceleration.TtsOptions:
			return typed != nil && typed.Target != nil && *typed.Target != ""
		case *acceleration.LlmOptions:
			return typed != nil && typed.Target != nil && *typed.Target != ""
		}
	}
	return false
}

// pcm reads one audio frame, whose header says how to play what follows.
func pcm(payload []byte) Audio {
	if len(payload) < 8 {
		return Audio{}
	}
	rate := int(payload[0]) | int(payload[1])<<8 | int(payload[2])<<16 | int(payload[3])<<24
	channels := int(payload[4]) | int(payload[5])<<8
	return Audio{Samples: payload[8:], SampleRate: rate, Channels: channels}
}

// wait sleeps between asking about a job, or stops if the caller has stopped waiting.
func wait(ctx context.Context) error {
	select {
	case <-time.After(poll):
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ended logs a socket that stopped, unless it stopped because it was closed.
func ended(logger *slog.Logger, modality string, err error) {
	if !errors.Is(err, ErrSocketClosed) {
		logger.Debug("the "+modality+" socket ended", "error", err)
	}
}

// refusal is what the router said went wrong, whichever field it said it in.
func refusal(status string, failures ...*acceleration.Error) error {
	for _, failure := range failures {
		if failure != nil {
			return fmt.Errorf("stream: %s", failure.Error)
		}
	}
	return fmt.Errorf("stream: the router answered %s", strings.TrimSpace(status))
}

// value reads a field that may not have been set.
func value(held *string) string {
	if held == nil {
		return ""
	}
	return *held
}
