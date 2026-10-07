//go:build integration

package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"image"
	"image/png"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagegen"
	"github.com/GetStream/Vision-Agents/acceleration/internal/imagerouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/lcmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// The providers the suite routes to. Every one of them answers in process: what a vendor
// makes of real audio, a real picture or a real question is that provider package's own
// suite, and what is under test here is the HTTP surface in front of them.

// routableConfig is one stub provider, reachable by the shortcuts a session asks for.
//
// It declares every term, because what is under test here is the HTTP surface: a config
// that asks to be diarized has to reach a provider that can be, and which providers can
// is the routing package's own suite.
func routableConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider:  "stub",
			Model:     "stub-model",
			Languages: []string{"en"},
			Realtime:  true,
			Terms:     everyTerm,
		}},
		Aliases: map[string]routing.Alias{
			"llm-flow":       {Languages: []string{"en"}, RequireRealtime: true},
			"en-low-latency": {Languages: []string{"en"}, RequireRealtime: true},
			// Where a batch job with no target of its own goes.
			"en-recorded":           {Languages: []string{"en"}},
			"multilingual-recorded": {},
			// Where a question with no target of its own goes.
			"search-fast": {Languages: []string{"en"}},
		},
	}
}

// everyTerm is everything a caller may ask a provider for, apart from the smart mode that
// cannot be combined with word timings.
var everyTerm = []options.Term{
	options.DetectLanguage, options.Interim, options.Endpointing, options.Diarize,
	options.MaxSpeakers, options.Keyterms, options.Format, options.Redact, options.Events,
	options.Channels, options.Words, options.Summary, options.Entities, options.Verbatim,
	options.ProfanityFilter, options.Speed, options.Volume, options.Emotion,
	options.Stability, options.Pronunciations, options.ChunkSchedule, options.Domains,
	options.Category, options.Recency, options.Location, options.Contents,
	options.OutputSchema, options.SemanticTurns, options.ManualTurns, options.Tools,
	options.TextInput, options.InputTranscript, options.OutputTranscript,
}

// reasoningConfig adds the model that can see, which is what a subagent handed a picture
// has to be routed to.
func reasoningConfig() routing.ModalityConfig {
	config := routableConfig()
	config.Providers = append(config.Providers, routing.ProviderConfig{
		Provider: "vision", Model: "vision-model",
		Languages: []string{"en"}, InputModalities: []string{"image"},
	})
	config.Aliases["vlm"] = routing.Alias{RequireInputModalities: []string{"image"}}
	// Reached by name only: no shortcut asks for Latin.
	config.Providers = append(config.Providers,
		routing.ProviderConfig{Provider: "echo", Model: "echo-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "noted", Model: "noted-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "tooling", Model: "tool-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "connecting", Model: "connector-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "slow", Model: "slow-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "recites", Model: "recites-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "counted", Model: "counted-model", Languages: []string{"la"}},
		routing.ProviderConfig{Provider: "summarising", Model: "summarising-model", Languages: []string{"la"}},
	)
	// Where a socket that names no target goes.
	config.Aliases["llm-fast"] = routing.Alias{Languages: []string{"en"}}
	return config
}

// silentEdge is a call with no network in it.
type silentEdge struct{ inbound chan agent.InboundAudio }

func (e *silentEdge) Join(context.Context) error       { return nil }
func (e *silentEdge) Audio() <-chan agent.InboundAudio { return e.inbound }
func (e *silentEdge) PublishAudio(audio.PcmData) error { return nil }

func (e *silentEdge) Leave() error {
	close(e.inbound)
	return nil
}

// scriptedLLM answers with a fixed reply and, on the first turn, whatever tool the test
// wants the model to reach for.
type scriptedLLM struct {
	mu    sync.Mutex
	turns int
	reply string
	calls []llm.ToolCall
	asked []llm.ResponseParams
	sees  bool
	// echoes answers with the instructions the model was given instead of reply, which is
	// how a test reads back what a session knew before anybody spoke.
	echoes bool
	// recites answers with every message it was handed, one per line in the order it was
	// handed them, which is how a test reads back the history a session was opened with.
	recites bool
	// held, when set, makes each reply wait for the test to let it through, which is what
	// stopping a command mid-answer needs.
	held chan struct{}
	// takes, when set, is how long each reply is in the writing. A command is only
	// stoppable while it is still being answered.
	takes time.Duration
	// usage is what each reply reports the model read and wrote.
	usage llm.Usage
	// summarises answers a request for JSON as a reviewer would, with a summary that is the
	// whole of what it was given to read. Any other request is answered with reply.
	summarises bool
}

func (s *scriptedLLM) Start(context.Context) error { return nil }

func (s *scriptedLLM) Create(ctx context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	s.mu.Lock()
	s.turns++
	s.asked = append(s.asked, params)
	first := s.turns == 1
	reply := s.reply
	if s.echoes {
		reply = params.Instructions
	}
	if s.recites {
		handed := make([]string, 0, len(params.Input))
		for _, message := range params.Input {
			handed = append(handed, string(message.Role)+": "+message.Content)
		}
		reply = strings.Join(handed, "\n")
	}
	if s.summarises && params.Text.Format == llm.FormatJSONObject && len(params.Input) > 0 {
		summary, _ := json.Marshal(map[string]string{"summary": params.Input[0].Content})
		reply = string(summary)
	}
	held := s.held
	var calls []llm.ToolCall
	if first {
		calls = append([]llm.ToolCall(nil), s.calls...)
	}
	s.mu.Unlock()

	if held != nil {
		select {
		case <-held:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if s.takes > 0 {
		select {
		case <-time.After(s.takes):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}

	script := llmtest.New(llm.StreamOptions{
		ResponseID: params.ID,
		Provider:   s.Provider(),
		Model:      s.Model(),
	})
	script.OutputText(reply)
	if s.usage != (llm.Usage{}) {
		script.Usage(s.usage)
	}
	if len(calls) > 0 {
		script.ToolCalls(calls...)
	}
	script.Done()
	return script.Stream(), nil
}

func (s *scriptedLLM) Provider() string { return "stub" }
func (s *scriptedLLM) Model() string    { return "stub-llm" }
func (s *scriptedLLM) Capabilities() llm.Capabilities {
	if s.sees {
		return llm.Capabilities{InputModalities: []string{llm.ModalityImage}}
	}
	return llm.Capabilities{}
}
func (s *scriptedLLM) Close() error { return nil }

// requests is every set of parameters the router has asked this model for.
func (s *scriptedLLM) requests() []llm.ResponseParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]llm.ResponseParams(nil), s.asked...)
}

// answers lets through the replies a held model is sitting on.
func (s *scriptedLLM) answers() {
	s.mu.Lock()
	held := s.held
	s.held = nil
	s.mu.Unlock()
	if held != nil {
		close(held)
	}
}

// holds makes the model wait for answers before it replies.
func (s *scriptedLLM) holds() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.held = make(chan struct{})
}

// quietSTT hears nothing, since these tests drive a session over HTTP rather than through
// a microphone.
type quietSTT struct{ emitter *stt.Emitter }

func (s *quietSTT) Start(context.Context) error                     { return nil }
func (s *quietSTT) ProcessAudio(stt.PcmData, stt.Participant) error { return nil }
func (s *quietSTT) Events() <-chan stt.Event                        { return s.emitter.Events() }
func (s *quietSTT) Provider() string                                { return "stub" }
func (s *quietSTT) Model() string                                   { return "stub-stt" }

func (s *quietSTT) Close() error {
	s.emitter.Close()
	return nil
}

// recordingTTS keeps what it was asked to say.
type recordingTTS struct {
	emitter *tts.Emitter

	mu   sync.Mutex
	said []string
}

func (s *recordingTTS) Start(context.Context) error { return nil }

func (s *recordingTTS) Synthesize(request tts.Request) error {
	s.mu.Lock()
	s.said = append(s.said, request.Text)
	s.mu.Unlock()

	if request.Final {
		s.emitter.Send(tts.SynthesisComplete{SynthesisID: request.ID})
	}
	return nil
}

func (s *recordingTTS) Interrupt() error         { return nil }
func (s *recordingTTS) Events() <-chan tts.Event { return s.emitter.Events() }
func (s *recordingTTS) Provider() string         { return "stub" }
func (s *recordingTTS) Model() string            { return "stub-tts" }
func (s *recordingTTS) Streaming() bool          { return false }
func (s *recordingTTS) Performs() bool           { return false }
func (s *recordingTTS) Prompt() string           { return "" }

func (s *recordingTTS) Close() error {
	s.emitter.Close()
	return nil
}

// spoken is everything this voice has been told to say.
func (s *recordingTTS) spoken() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]string(nil), s.said...)
}

// recordedTranscriber transcribes whatever it is handed into the same transcript, so a job
// can be followed from the row it creates to the result a caller reads back.
type recordedTranscriber struct{ model string }

func transcriberRegistry() *sttrouter.Transcribers {
	registry := sttrouter.NewTranscriberRegistry()
	registry.Register("stub", func(spec routing.Spec) (stt.Transcriber, error) {
		return &recordedTranscriber{model: spec.Model}, nil
	})
	return registry
}

func (t *recordedTranscriber) Transcribe(_ context.Context, recording stt.Recording) (stt.Transcription, error) {
	transcription := stt.Transcription{
		Text:            "a call costs a penny",
		Language:        "en",
		AudioDurationMs: 4000,
	}
	if recording.Words {
		transcription.Words = []stt.Word{
			{Text: "a", StartMs: 0, EndMs: 200, Confidence: 0.9},
			{Text: "call", StartMs: 200, EndMs: 600, Confidence: 0.9},
		}
	}
	if recording.Diarize {
		transcription.Speakers = []string{"speaker_0"}
	}
	return transcription, nil
}

func (t *recordedTranscriber) Start(context.Context) error { return nil }
func (t *recordedTranscriber) Close() error                { return nil }
func (t *recordedTranscriber) Provider() string            { return "stub" }
func (t *recordedTranscriber) Model() string               { return t.model }

// recordedVoice speaks whatever it is handed into the same audio.
type recordedVoice struct{ model string }

func recorderRegistry() *ttsrouter.Recorders {
	registry := ttsrouter.NewRecorderRegistry()
	registry.Register("stub", func(spec routing.Spec) (tts.Recorder, error) {
		return &recordedVoice{model: spec.Model}, nil
	})
	return registry
}

func (v *recordedVoice) Record(_ context.Context, recording tts.Recording) (tts.Recorded, error) {
	format := recording.Format
	if format == "" {
		format = "mp3_44100_128"
	}
	return tts.Recorded{
		Audio:           []byte{0xff, 0xfb, 0x90},
		Format:          format,
		AudioDurationMs: 1200,
		Characters:      int64(len(recording.Text)),
	}, nil
}

func (v *recordedVoice) Start(context.Context) error { return nil }
func (v *recordedVoice) Close() error                { return nil }
func (v *recordedVoice) Provider() string            { return "stub" }
func (v *recordedVoice) Model() string               { return v.model }

// answerer answers every question with the same result.
type answerer struct{}

func searchRegistry() *searchrouter.Registry {
	registry := searchrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (search.Provider, error) { return answerer{}, nil })
	return registry
}

func (answerer) Search(_ context.Context, query search.Query) (search.Result, error) {
	return search.Result{Answer: "A call costs a penny.", Documents: []search.Document{
		{URL: "https://example.test/pricing", Title: "Pricing", Text: query.Text, Score: 0.9},
	}}, nil
}

func (answerer) Start(context.Context) error { return nil }
func (answerer) Close() error                { return nil }
func (answerer) Provider() string            { return "stub" }
func (answerer) Model() string               { return "stub-search" }

// judge stands in for a classifier: it answers what it is told to.
type judge struct {
	name     string
	answered lcm.Result

	mu    sync.Mutex
	asked []lcm.Request
}

// classifiers are the two the suite routes to, so a test can tell a target apart from the
// default one.
var classifiers = struct {
	quick, careful *judge
}{
	quick:   &judge{name: "quick", answered: judged},
	careful: &judge{name: "careful", answered: judged},
}

// judged is the ruling both classifiers give, one answer of each type.
var judged = lcm.Result{
	Model: "judge-2026-09",
	Answers: map[string]lcm.Answer{
		"refund": {Type: lcm.TypeNoul, Yes: 0.91},
		"topic": {
			Type: lcm.TypeChoice, Chosen: "billing", Confidence: 0.8,
			Probabilities: map[string]float64{"billing": 0.9, "other": 0.1},
		},
		"urgency": {
			Type: lcm.TypeScore, Level: 1.4, Confidence: 0.6,
			Legend:        map[string]string{"0": "can wait", "1": "this week", "2": "today"},
			Probabilities: map[string]float64{"0": 0.1, "1": 0.4, "2": 0.5},
		},
	},
	Usage: lcm.Usage{InputTokens: 406, OutputTokens: 69},
}

// refusals are the states a test asks about to be refused instead of ruled on, since what
// a vendor will not answer is a property of the question rather than of the classifier.
var refusals = map[string]error{
	"unanswerable": errors.New(`typesafe: "refund" was not answered`),
	"rate limited": fmt.Errorf("typesafe: the API returned 429: %w", lcm.ErrRateLimited),
	"unavailable":  fmt.Errorf("typesafe: the API returned 529: %w", lcm.ErrUnavailable),
}

func (j *judge) Classify(_ context.Context, request lcm.Request) (lcm.Result, error) {
	j.mu.Lock()
	j.asked = append(j.asked, request)
	j.mu.Unlock()

	state, _ := request.State.(string)
	for asked, failure := range refusals {
		if strings.Contains(state, asked) {
			return lcm.Result{}, failure
		}
	}
	return j.answered, nil
}

func (j *judge) Start(context.Context) error { return nil }
func (j *judge) Close() error                { return nil }
func (j *judge) Provider() string            { return j.name }
func (j *judge) Model() string               { return "stub" }

// questions is every request this classifier has been asked to rule on.
func (j *judge) questions() []lcm.Request {
	j.mu.Lock()
	defer j.mu.Unlock()
	return append([]lcm.Request(nil), j.asked...)
}

// ruledOn is what this classifier was asked about state, and whether it was asked at all.
// A test names its own state, since the stub is shared.
func (j *judge) ruledOn(state string) (lcm.Request, bool) {
	for _, asked := range j.questions() {
		if named, ok := asked.State.(string); ok && named == state {
			return asked, true
		}
	}
	return lcm.Request{}, false
}

func classifyConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "quick", Model: "judge", Languages: []string{"en"},
				Realtime: true, Tier: routing.LowLatency,
			},
			{
				Provider: "careful", Model: "judge", Languages: []string{"en"},
				Tier: routing.HighQuality,
			},
		},
		Aliases: map[string]routing.Alias{
			"classify-fast": {RequireRealtime: true, Tier: routing.LowLatency},
		},
	}
}

func classifierRegistry() *lcmrouter.Registry {
	registry := lcmrouter.NewRegistry()
	registry.Register("quick", func(routing.Spec) (lcm.Provider, error) { return classifiers.quick, nil })
	registry.Register("careful", func(routing.Spec) (lcm.Provider, error) { return classifiers.careful, nil })
	return registry
}

// painter stands in for an image provider: it draws the same picture every time.
type painter struct {
	name string

	mu    sync.Mutex
	asked []imagegen.Request
}

var painters = struct {
	quick, lush *painter
}{
	quick: &painter{name: "quick"},
	lush:  &painter{name: "lush"},
}

// drawn is a 32 by 32 png, which is a picture as far as the endpoint is concerned.
var drawn = func() imagegen.Result {
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, image.NewRGBA(image.Rect(0, 0, 32, 32))); err != nil {
		panic(err)
	}
	seed := int64(7)
	return imagegen.Result{Images: []imagegen.Image{{
		Data: encoded.Bytes(), MediaType: "image/png", Width: 1024, Height: 1024, Seed: &seed,
	}}}
}()

// unsafely is the prompt a test asks for a picture it cannot have with, since what a
// safety checker refuses is a property of the prompt rather than of the provider.
const unsafely = "unsafe"

// roses is a picture to send a model, as the data URI a caller writes one as.
var roses = llm.ImagePart{MIME: "image/png", Data: drawn.Images[0].Data}.DataURI()

func (p *painter) Generate(_ context.Context, request imagegen.Request) (imagegen.Result, error) {
	p.mu.Lock()
	p.asked = append(p.asked, request)
	p.mu.Unlock()
	if strings.Contains(request.Prompt, unsafely) {
		return imagegen.Result{}, imagegen.Fail(imagegen.ContentFiltered, true,
			errors.New("the safety checker flagged the picture"))
	}
	return drawn, nil
}

func (p *painter) Start(context.Context) error { return nil }
func (p *painter) Close() error                { return nil }
func (p *painter) Provider() string            { return p.name }
func (p *painter) Model() string               { return "stub" }

// prompts is every request this provider has been asked to draw.
func (p *painter) prompts() []imagegen.Request {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]imagegen.Request(nil), p.asked...)
}

// drew is what this provider was asked for the picture named by prompt, and whether it was
// asked at all. A test names its own prompt, since the stub is shared.
func (p *painter) drew(prompt string) (imagegen.Request, bool) {
	for _, asked := range p.prompts() {
		if asked.Prompt == prompt {
			return asked, true
		}
	}
	return imagegen.Request{}, false
}

func imageConfig() routing.ModalityConfig {
	return routing.ModalityConfig{
		Providers: []routing.ProviderConfig{
			{
				Provider: "quick", Model: "fast", Languages: []string{"en"}, Tier: routing.LowLatency,
				Terms: []options.Term{
					options.Size, options.AspectRatio, options.Seed,
					options.NegativePrompt, options.Format,
				},
				Price: routing.Price{PerImage: 0.04},
			},
			{
				Provider: "lush", Model: "best", Languages: []string{"en"}, Tier: routing.HighQuality,
				Terms: []options.Term{options.AspectRatio},
				Price: routing.Price{PerImage: 0.067},
			},
		},
		Aliases: map[string]routing.Alias{
			"image-fast":    {Title: "Fast images", Tier: routing.LowLatency},
			"image-quality": {Title: "Best images", Tier: routing.HighQuality},
		},
	}
}

func painterRegistry() *imagerouter.Registry {
	registry := imagerouter.NewRegistry()
	registry.Register("quick", func(routing.Spec) (imagegen.Provider, error) { return painters.quick, nil })
	registry.Register("lush", func(routing.Spec) (imagegen.Provider, error) { return painters.lush, nil })
	return registry
}

// stubConversation stands in for a speech-to-speech model, so the socket can be driven
// without a vendor. Whatever the socket sends it is recorded; whatever the test emits on it
// comes back down the socket.
type stubConversation struct {
	emitter      *sts.Emitter
	capabilities sts.Capabilities
	heard        chan sts.PcmData
	typed        chan string
	answers      chan string
	interrupts   chan int
	closed       chan struct{}
}

// conversing is every stub the registry has made, so a test can reach the one its socket
// landed on.
var conversing = make(chan *stubConversation, 16)

func stsrouterStub() (*stsrouter.Router, error) {
	config := routing.ModalityConfig{
		Providers: []routing.ProviderConfig{{
			Provider: "stub", Model: "stub-sts", Languages: []string{"en"}, Realtime: true,
		}},
		Aliases: map[string]routing.Alias{"sts-fast": {RequireRealtime: true}},
	}
	registry := stsrouter.NewRegistry()
	registry.Register("stub", func(routing.Spec) (sts.STS, error) {
		stub := newStubConversation(sts.Capabilities{})
		select {
		case conversing <- stub:
		default:
		}
		return stub, nil
	})
	return stsrouter.New(stsrouter.Options{Config: config, Registry: registry})
}

func newStubConversation(capabilities sts.Capabilities) *stubConversation {
	return &stubConversation{
		emitter:      sts.NewEmitter(sts.EmitterBuffer),
		capabilities: capabilities,
		heard:        make(chan sts.PcmData, 16),
		typed:        make(chan string, 16),
		answers:      make(chan string, 16),
		interrupts:   make(chan int, 16),
		closed:       make(chan struct{}),
	}
}

func (s *stubConversation) Start(context.Context) error { return nil }
func (s *stubConversation) ProcessAudio(pcm sts.PcmData, _ sts.Participant) error {
	s.heard <- pcm
	return nil
}
func (s *stubConversation) SendText(text string, _ sts.Participant) error {
	s.typed <- text
	return nil
}
func (s *stubConversation) SendFrame(llm.ImagePart) error { return sts.ErrNoImages }
func (s *stubConversation) SetInstructions(string) error  { return nil }
func (s *stubConversation) SetTools([]llm.Tool) error     { return nil }
func (s *stubConversation) Answer(callID, output string, _ error) error {
	s.answers <- callID + "=" + output
	return nil
}
func (s *stubConversation) Prompt(string) error          { return nil }
func (s *stubConversation) Interrupt(playedMs int) error { s.interrupts <- playedMs; return nil }
func (s *stubConversation) Events() <-chan sts.Event     { return s.emitter.Events() }

func (s *stubConversation) Close() error {
	select {
	case <-s.closed:
	default:
		close(s.closed)
	}
	s.emitter.Close()
	return nil
}

func (s *stubConversation) Provider() string               { return "stub" }
func (s *stubConversation) Model() string                  { return "stub-sts" }
func (s *stubConversation) SampleRate() int                { return 24_000 }
func (s *stubConversation) Capabilities() sts.Capabilities { return s.capabilities }

// knowledgeBase is a knowledge base kept in memory, keyed the way a real one is. Passages
// replace whatever is already stored under their id, which is the property the endpoint
// depends on for posting a document twice to be an edit rather than a second copy.
type knowledgeBase struct {
	mu        sync.Mutex
	namespace string
	passages  map[string]knowledge.Document
}

func newKnowledgeBase() *knowledgeBase {
	return &knowledgeBase{passages: map[string]knowledge.Document{}}
}

func (b *knowledgeBase) Upsert(_ context.Context, namespace string, documents []knowledge.Document) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.namespace = namespace
	for _, document := range documents {
		b.passages[document.ID] = document
	}
	return nil
}

func (b *knowledgeBase) Delete(_ context.Context, _ string, ids []string) error {
	b.mu.Lock()
	defer b.mu.Unlock()

	for _, id := range ids {
		delete(b.passages, id)
	}
	return nil
}

func (b *knowledgeBase) Fetch(_ context.Context, _ string, ids []string) ([]knowledge.Document, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	var found []knowledge.Document
	for _, id := range ids {
		if document, ok := b.passages[id]; ok {
			found = append(found, document)
		}
	}
	return found, nil
}

// stored is the namespace last written to and the passages it holds.
func (b *knowledgeBase) stored() (string, map[string]knowledge.Document) {
	b.mu.Lock()
	defer b.mu.Unlock()

	copied := make(map[string]knowledge.Document, len(b.passages))
	for id, document := range b.passages {
		copied[id] = document
	}
	return b.namespace, copied
}

// pageReader answers every url with the same page, which is enough for the endpoints to be
// exercised: what a real crawler makes of a real page is the provider's own suite.
type pageReader struct{}

func (pageReader) Read(_ context.Context, address string) (search.Page, error) {
	return search.Page{
		URL:   address,
		Title: "Pricing",
		Text:  "# Pricing\n\nA call costs a penny.\n",
	}, nil
}

// keptMemories holds memories in a slice, and forgets them the way a real store would.
type keptMemories struct {
	mu   sync.Mutex
	kept []memory.Scope
}

func (m *keptMemories) Recall(context.Context, memory.Query) ([]memory.Memory, error) {
	return nil, nil
}

func (m *keptMemories) Remember(_ context.Context, scope memory.Scope, _ []llm.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.kept = append(m.kept, scope)
	return nil
}

func (m *keptMemories) Truncate(_ context.Context, appID, userID string) error {
	m.forget(func(kept memory.Scope) bool { return kept.AppID == appID && kept.UserID == userID })
	return nil
}

func (m *keptMemories) ForgetRun(_ context.Context, appID, runID string) error {
	m.forget(func(kept memory.Scope) bool { return kept.AppID == appID && kept.RunID == runID })
	return nil
}

func (m *keptMemories) Provider() string { return "kept" }
func (m *keptMemories) Close() error     { return nil }

func (m *keptMemories) forget(matches func(memory.Scope) bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	left := m.kept[:0]
	for _, kept := range m.kept {
		if !matches(kept) {
			left = append(left, kept)
		}
	}
	m.kept = left
}

// remaining is every memory still held.
func (m *keptMemories) remaining() []memory.Scope {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]memory.Scope(nil), m.kept...)
}

// namedScheme is a scheme registered by its name alone, which is all a connector definition
// reads of one: whether a custom definition may name it. Nothing in these suites connects
// through it, so it acquires, issues and applies nothing.
type namedScheme string

func (n namedScheme) Name() string { return string(n) }

func (namedScheme) Begin(context.Context, core.BeginInput) (core.BeginOutput, error) {
	return core.BeginOutput{}, errors.New("a named scheme does not connect")
}

func (namedScheme) Complete(context.Context, core.CompleteInput) (core.StoredCredentials, core.AccountInfo, error) {
	return core.StoredCredentials{}, core.AccountInfo{}, errors.New("a named scheme does not connect")
}

func (namedScheme) Retrieve(context.Context, core.StoredCredentials, core.ResolvedManifest, core.RetrieveOptions) (core.AccessCredential, core.StoredCredentials, error) {
	return core.AccessCredential{}, core.StoredCredentials{}, errors.New("a named scheme does not connect")
}

func (namedScheme) Wrap(base http.RoundTripper, _ core.AccessCredential) http.RoundTripper {
	return base
}

func (namedScheme) Classify(*http.Response, []byte, error) core.Outcome { return core.Outcome{} }

func (namedScheme) Revoke(context.Context, core.StoredCredentials, core.ResolvedManifest) error {
	return nil
}
