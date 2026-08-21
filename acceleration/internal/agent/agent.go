// Package agent runs a voice conversation over the three routed modalities.
//
// It is the Go counterpart of the Python Agent in agents-core: audio from the edge is
// transcribed, settled turns are answered by a model, and the reply is spoken back. What
// makes it worth having in this service is that it is built from the routers rather than
// from provider instances, so every turn is routed, failed over and billed by the same
// machinery as a direct API call.
package agent

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// eventBuffer is how many events may queue before a slow consumer applies backpressure.
const eventBuffer = 64

const presenceTick = 500 * time.Millisecond

// sentenceSuffix separates a turn id from the sequence number of a sentence within it, for
// providers that need one synthesis per sentence.
const sentenceSuffix = "#"

// Options configures an Agent.
//
// The three modalities arrive as routers plus targets rather than as providers, because the
// routers are what own failover and billing, and because speech-to-text is opened per
// participant rather than once for the call.
type Options struct {
	Edge Edge
	// Text holds the conversation in writing rather than on a call. Nothing is
	// transcribed, nothing is spoken and no call is joined, so the edge and the two
	// speech routers are not needed and are not used. Everything between hearing and
	// answering is the same, which is the point: the harness, the skills and the tools
	// are what a text agent is being asked for.
	Text bool
	// Instructions is the system prompt, sent with every turn.
	Instructions string
	// CustomerID owns every request the agent makes. It is what the usage is billed to.
	CustomerID string
	// AgentID identifies this agent across calls. Transcripts are stored under it and
	// every request the agent makes is recorded against it.
	AgentID string
	// CallID is the call being served, recorded alongside each request.
	CallID string
	// Tags are the customer's own cost labels, carried onto every request the agent
	// makes so a conversation's whole spend can be attributed at once.
	Tags routing.Tags

	LLM       *llmrouter.Router
	LLMTarget string
	STT       *sttrouter.Router
	STTTarget string
	TTS       *ttsrouter.Router
	TTSTarget string

	// SubagentTarget routes the slower, more capable model that runs the work the voice
	// model hands over. It is a target on the same router as LLMTarget, because the
	// difference between them is which model, not which service. Empty means the agent
	// answers everything itself.
	SubagentTarget string
	// Skills are what the voice model may hand over. They mean nothing without a
	// subagent to run them.
	Skills harness.Skills
	// Telephony is what the agent may do to the call itself. Without it the agent can
	// only talk, which is what a call that is not on a phone network can do.
	Telephony Telephony
	// ToolRunner carries out the tools that are not the two acting on the phone call,
	// which is how a caller outside this process owns its own tools.
	ToolRunner ToolRunner
	// Tools are what the voice model may do rather than say. Each is only offered when
	// something on this call can run it: the telephony pair needs Telephony, and every
	// other tool needs a ToolRunner.
	Tools harness.Tools
	// Sandbox is where the subagent runs code it writes. It is never offered to the model
	// holding the conversation: running code takes seconds, and a conversation cannot
	// spare them.
	Sandbox sandbox.Sandbox
	// Tasks caps how much delegated work may run at once. Zero leaves the harness's own
	// default in place.
	Tasks int
	// Duplex lets the agent listen and talk at the same time rather than strictly taking
	// turns. Both halves of it are off by default.
	Duplex DuplexOptions

	// Voice selects the speaker. Its meaning is the text-to-speech provider's.
	Voice string
	// LanguageHints narrow the candidates in every modality.
	LanguageHints []string
	// MaxTokens caps each reply. Zero leaves the model's own default in place.
	MaxTokens int
	// Memory carries what earlier conversations established into this one. Without it
	// the agent starts every call knowing nothing but its instructions.
	Memory memory.Store
	// AppID scopes memories to the application using this service, so two deployments
	// sharing one memory account do not read each other's.
	AppID string
	// MemoryUserID is who the memories are about. Empty means the customer, which is
	// what a caller with no user of its own to scope by gets.
	MemoryUserID string
	// MemoryFilter narrows recall further with the caller's own labels, such as the
	// company the user belongs to.
	MemoryFilter map[string]string
	// RecallLimit caps how many memories are recalled on joining. Zero leaves the
	// store's own default.
	RecallLimit int
	// Knowledge is what the agent may look up mid-conversation. Without it, or without a
	// namespace to read, the lookup tool is not offered: a model told it can search and
	// then refused would promise the caller an answer it cannot get.
	Knowledge knowledge.Store
	// KnowledgeNamespace is which body of knowledge this agent reads.
	KnowledgeNamespace string
	// KnowledgeLimit caps how many passages one lookup returns. Zero leaves the store's
	// own default.
	KnowledgeLimit int
	// Store records what each turn cost the participant in waiting. Without it the
	// timings are still emitted as Turn events, they are just not persisted.
	Store  *store.Store
	Live   *live.Client
	Logger *slog.Logger
}

// Agent is a voice agent in one call.
type Agent struct {
	options Options
	logger  *slog.Logger
	emitter *Emitter

	llm *llmrouter.Session
	tts *ttsrouter.Session
	// harness stands between what a participant said and the model that answers them. It
	// decides what the model is asked, and takes the model's requests for help back out
	// of the reply before any of it reaches the voice.
	harness *harness.Harness

	// turns measures each exchange end to end and, when a store is configured, records it.
	turns     *turnTracker
	turnStore *turnRecorder
	// memory carries what earlier conversations established into this one.
	memory *memoryWriter
	// knowledge answers what the business already wrote down, when the agent has a
	// namespace to read.
	knowledge *knowledgeReader
	// recalled is what the agent already knew on joining, rendered as a system message
	// and prepended to the instructions on every turn.
	recalled string

	// ctx is the call's lifetime. Every session the agent opens derives from it.
	ctx    context.Context
	cancel context.CancelFunc

	mu sync.Mutex
	// prompt is what the agent was told to be. It starts as the configured instructions
	// and lives here rather than in the options because a caller may change what the
	// agent is part way through a call.
	prompt string
	// history is the conversation so far. It lives here rather than in a provider so a
	// failover between providers mid-conversation loses nothing.
	history []llm.Message
	// listeners holds one transcription session per participant, because a speech-to-text
	// stream is bound to a single speaker.
	listeners map[string]*sttrouter.Session
	// speakingTurn is the reply currently allowed to produce audio. Audio belonging to any
	// other turn is dropped rather than published, which is what makes barge-in immediate
	// even while a provider is still sending.
	speakingTurn string
	// utterances counts syntheses that have not settled, so Finish knows when the agent
	// has stopped talking.
	utterances int
	// generating is true while the voice model is still writing the current reply.
	generating bool
	joined     bool
	closed     bool

	// lastParticipant is who the agent was last talking to, so a reply prompted by
	// delegated work coming back is attributed to the person who is waiting for it.
	lastParticipant stt.Participant
	lastHeardAt     time.Time
	lastSpokeAt     time.Time
	// lastCandidateTurn owns delegated work from the last relevant caller turn.
	lastCandidateTurn string
	// cadence turns evolving transcript revisions into stable candidates without relying
	// on provider turn boundaries.
	cadence *cadence
	// candidates are waiting on the fast flow controller.
	candidates map[string]candidate
	// queued is a relevant turn heard while the agent chose to finish speaking.
	queued *queuedCandidate

	// duplex tracks listening acknowledgements and transcript confidence.
	duplex *duplex
	// harnessDrained closes once every harness event has been reported, which is what
	// lets shutdown report the work it abandoned on the way out. It is nil until the
	// consumer that closes it is running.
	harnessDrained chan struct{}

	// chunk assembles model deltas into sentences. Only the model consumer touches it.
	chunk chunker
	// spoken is the reply as the caller will hear it, which is what goes into the
	// history: a request for help was written to the harness, not said, so remembering
	// it would have the model reading its own instructions back on the next turn.
	spoken strings.Builder
	// replying is the turn whose text the chunker and the harness are holding, so a
	// completion arriving late for an abandoned turn cannot clear the one after it.
	replying string
	// sentences counts the sentences sent in the current turn, for the synthesis id.
	sentences int
	// openTurn is the turn a streaming voice has an utterance open for, so a turn's
	// sentences are one billed utterance rather than one each.
	openTurn string

	// following serialises the turns the agent takes of its own accord, so two answers
	// landing together cannot both decide the agent is free to speak.
	following sync.Mutex

	running   sync.WaitGroup
	closeOnce sync.Once
}

type queuedCandidate struct {
	candidate candidate
	clarify   bool
}

// New validates the options and returns an Agent. It opens nothing; Join does that.
func New(options Options) (*Agent, error) {
	if options.LLM == nil {
		return nil, errors.New("agent: an llm router is required")
	}
	// A conversation in writing has nowhere to listen and nothing to speak with, so the
	// three that carry a voice are only required when there is one.
	if !options.Text {
		if options.Edge == nil {
			return nil, errors.New("agent: an edge is required")
		}
		if options.STT == nil {
			return nil, errors.New("agent: an stt router is required")
		}
		if options.TTS == nil {
			return nil, errors.New("agent: a tts router is required")
		}
	}
	if options.CustomerID == "" {
		return nil, errors.New("agent: a customer id is required")
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if err := options.Tags.Validate(); err != nil {
		return nil, err
	}

	logger := options.Logger.With("customer", options.CustomerID)
	agent := &Agent{
		options:    options,
		logger:     logger,
		emitter:    NewEmitter(eventBuffer),
		prompt:     options.Instructions,
		listeners:  map[string]*sttrouter.Session{},
		cadence:    newCadence(0, 0, 0, logger),
		candidates: map[string]candidate{},
		duplex:     newDuplex(options.Duplex),
	}

	owner := routing.Owner{
		CustomerID: options.CustomerID,
		AgentID:    options.AgentID,
		CallID:     options.CallID,
		Tags:       options.Tags,
	}

	// Turns are keyed by agent id, so an agent without one is measured but not stored.
	if options.Store != nil && options.AgentID != "" {
		agent.turnStore = newTurnRecorder(options.Store, owner, logger)
	}
	agent.turns = newTurnTracker(agent.finishTurn)

	if options.Memory != nil {
		// Memories belong to the customer unless the caller named someone more specific,
		// and are recorded as a modality of their own so what remembering costs is
		// reported alongside what the models cost.
		scope := memory.Scope{
			AppID:  options.AppID,
			UserID: options.CustomerID,
			Extra:  options.MemoryFilter,
		}
		if options.MemoryUserID != "" {
			scope.UserID = options.MemoryUserID
		}
		agent.memory = newMemoryWriter(
			options.Memory,
			scope,
			owner,
			routing.NewRecorder(routing.Memory, options.Store, options.Live, logger),
			logger,
		)
	}

	// A knowledge store without a namespace reads nothing, so the agent is treated as
	// having none rather than being offered a search that would always fail.
	if options.Knowledge != nil && options.KnowledgeNamespace != "" {
		agent.knowledge = newKnowledgeReader(
			options.Knowledge,
			options.KnowledgeNamespace,
			options.KnowledgeLimit,
			owner,
			routing.NewRecorder(routing.Knowledge, options.Store, options.Live, logger),
			logger,
		)
	}

	return agent, nil
}

// finishTurn reports a measured exchange and records it.
func (a *Agent) finishTurn(turn Turn) {
	a.emitter.Send(turn)
	if a.turnStore != nil {
		a.turnStore.Record(turn)
	}
}

// Join opens the model and voice sessions, then joins the call and starts listening.
//
// The order matters: the sessions come first so the agent can answer the moment someone
// speaks, rather than losing the first thing it hears while it connects.
func (a *Agent) Join(ctx context.Context) error {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return errors.New("agent: already closed")
	}
	if a.joined {
		a.mu.Unlock()
		return errors.New("agent: already joined")
	}
	a.ctx, a.cancel = context.WithCancel(ctx)
	a.joined = true
	a.mu.Unlock()

	model, err := a.options.LLM.Start(a.ctx, llmrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        a.options.LLMTarget,
		LanguageHints: a.options.LanguageHints,
	})
	if err != nil {
		return fmt.Errorf("agent: start llm: %w", err)
	}
	a.llm = model

	// Flow decisions use their own fast-model session so deciding whether speech is
	// complete never competes with the reply being streamed to the voice.
	controller, err := a.options.LLM.Start(a.ctx, llmrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        a.options.LLMTarget,
		LanguageHints: a.options.LanguageHints,
	})
	if err != nil {
		return fmt.Errorf("agent: start flow controller: %w", err)
	}

	// The subagent is routed like anything else, so the work it does is failed over and
	// billed the same way a turn is.
	var subagent *llmrouter.Session
	if a.options.SubagentTarget != "" {
		subagent, err = a.options.LLM.Start(a.ctx, llmrouter.Request{
			CustomerID:    a.options.CustomerID,
			AgentID:       a.options.AgentID,
			CallID:        a.options.CallID,
			Tags:          a.options.Tags,
			Target:        a.options.SubagentTarget,
			LanguageHints: a.options.LanguageHints,
		})
		if err != nil {
			return fmt.Errorf("agent: start subagent: %w", err)
		}
	}

	a.harness, err = harness.New(harness.Options{
		Model:      model,
		Controller: controller,
		Subagent:   subagent,
		Skills:     a.options.Skills,
		Tools:      a.availableTools(),
		Sandbox:    a.options.Sandbox,
		Tasks:      a.options.Tasks,
		MaxTokens:  a.options.MaxTokens,
		Logger:     a.logger,
	})
	if err != nil {
		return err
	}

	if !a.options.Text {
		voice, err := a.options.TTS.Start(a.ctx, ttsrouter.Request{
			CustomerID:    a.options.CustomerID,
			AgentID:       a.options.AgentID,
			CallID:        a.options.CallID,
			Tags:          a.options.Tags,
			Target:        a.options.TTSTarget,
			LanguageHints: a.options.LanguageHints,
			Voice:         a.options.Voice,
		})
		if err != nil {
			return fmt.Errorf("agent: start tts: %w", err)
		}
		a.tts = voice
	}

	// What earlier conversations established is fetched before the call starts, so the
	// first turn is already answered in the light of it rather than the second.
	if a.memory != nil {
		a.recalled = memory.Prompt(a.memory.Recall(a.ctx, a.options.RecallLimit))
	}

	if !a.options.Text {
		if err := a.options.Edge.Join(a.ctx); err != nil {
			return fmt.Errorf("agent: join edge: %w", err)
		}
	}

	a.mu.Lock()
	a.harnessDrained = make(chan struct{})
	a.lastSpokeAt = time.Now()
	a.mu.Unlock()

	a.running.Add(2)
	go a.consumeLLM()
	go a.consumeHarness()
	// The other four all begin at a microphone or end at a speaker, so a conversation
	// held in writing runs none of them.
	if !a.options.Text {
		a.running.Add(4)
		go a.consumeTTS()
		go a.consumeEdge()
		go a.consumeCadence()
		go a.consumePresence()
	}

	if a.options.Text {
		a.logger.Info("joined", "llm", a.llm.Provider()+"/"+a.llm.Model())
	} else {
		a.logger.Info("joined",
			"llm", a.llm.Provider()+"/"+a.llm.Model(),
			"tts", a.tts.Provider()+"/"+a.tts.Model())
	}
	a.emitter.Send(Joined{At: time.Now()})
	return nil
}

// SimpleResponse answers a piece of text through the model, as though a participant had
// said it. It returns once the request is on its way: the reply arrives on Events and is
// spoken as it streams.
func (a *Agent) SimpleResponse(ctx context.Context, text string) error {
	return a.respond(stt.Participant{ID: "caller"}, text, heard{at: time.Now()})
}

// Say speaks a piece of text without asking the model. A greeting is exactly this: the
// agent already knows what it wants to say, so a model would only add latency and cost.
func (a *Agent) Say(ctx context.Context, text string) error {
	turnID := fmt.Sprintf("say-%d", time.Now().UnixNano())

	// A conversation in writing has no voice to say it with, so it is reported as said
	// instead: what a caller would have heard is what a reader reads.
	if a.options.Text {
		a.emitter.Send(Responded{TurnID: turnID, Text: text})
		return nil
	}

	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	a.speakingTurn = turnID
	a.mu.Unlock()

	return a.speakWhole(turnID, text)
}

// Interrupt abandons the reply being spoken, the way a participant talking over the agent
// would. It is what a caller outside the call has instead of a voice.
func (a *Agent) Interrupt() {
	a.mu.Lock()
	participant := a.lastParticipant
	a.mu.Unlock()
	a.interrupt(participant)
}

// SetInstructions changes what the agent is told to be from the next turn on. The reply
// being spoken keeps the prompt it was started with, because rewriting it mid-sentence
// would have the agent change character in the middle of a thought.
func (a *Agent) SetInstructions(text string) {
	a.mu.Lock()
	a.prompt = text
	a.mu.Unlock()
}

// Events carries what happened in the conversation. It is closed by Close.
func (a *Agent) Events() <-chan Event { return a.emitter.Events() }

// LLM exposes the model session, so a caller can reach the provider's own features or the
// price the conversation is billed at.
func (a *Agent) LLM() *llmrouter.Session { return a.llm }

// TTS exposes the voice session.
func (a *Agent) TTS() *ttsrouter.Session { return a.tts }

// History returns the conversation so far.
func (a *Agent) History() []llm.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append([]llm.Message(nil), a.history...)
}

// Finish waits for the agent to stop talking, so a caller can hang up without cutting off
// the last sentence. Work still running counts as talking: the caller was told an answer
// was coming. It returns the context's error if the wait outlasts it.
func (a *Agent) Finish(ctx context.Context) error {
	ticker := time.NewTicker(20 * time.Millisecond)
	defer ticker.Stop()

	for {
		a.mu.Lock()
		quiet := a.utterances == 0
		a.mu.Unlock()
		if quiet && !a.delegating() {
			return nil
		}

		select {
		case <-ticker.C:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// Close leaves the call and releases every session. It is safe to call more than once.
func (a *Agent) Close() error {
	var err error
	a.closeOnce.Do(func() { err = a.close() })
	return err
}

func (a *Agent) close() error {
	a.mu.Lock()
	a.closed = true
	cancel := a.cancel
	drained := a.harnessDrained
	listeners := make([]*sttrouter.Session, 0, len(a.listeners))
	for _, listener := range a.listeners {
		listeners = append(listeners, listener)
	}
	a.listeners = map[string]*sttrouter.Session{}
	a.mu.Unlock()

	a.cadence.Close()
	if cancel != nil {
		cancel()
	}

	// The edge leaves first: it is the source of the audio that keeps the rest busy.
	var failures []error
	if a.options.Edge != nil {
		if err := a.options.Edge.Leave(); err != nil {
			failures = append(failures, fmt.Errorf("leave edge: %w", err))
		}
	}
	for _, listener := range listeners {
		if err := listener.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close stt: %w", err))
		}
	}
	// The harness goes before the model: it owns the subagent, and abandoning work
	// nobody will hear is the last useful thing either of them does. Its consumer is
	// waited on here rather than at the end, so what it abandoned is still reported: the
	// events channel below is about to close.
	if a.harness != nil {
		if err := a.harness.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close harness: %w", err))
		}
		if drained != nil {
			<-drained
		}
	}
	if a.llm != nil {
		if err := a.llm.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close llm: %w", err))
		}
	}
	if a.tts != nil {
		if err := a.tts.Close(); err != nil {
			failures = append(failures, fmt.Errorf("close tts: %w", err))
		}
	}

	// Left is sent before the emitter closes, and the emitter closes before the consumers
	// are waited on: a consumer blocked emitting to a caller that has stopped reading has
	// to be let go of, or shutdown would depend on someone draining the channel.
	a.emitter.Send(Left{At: time.Now()})
	a.emitter.Close()
	a.running.Wait()

	// The writers are drained after the consumers have stopped, so a turn that finished
	// on the way out is still recorded and still remembered.
	if a.turnStore != nil {
		a.turnStore.Close()
	}
	if a.memory != nil {
		a.memory.Close()
	}
	if a.knowledge != nil {
		a.knowledge.Close()
	}

	return errors.Join(failures...)
}

// consumeEdge feeds each participant's audio to their own transcription session.
func (a *Agent) consumeEdge() {
	defer a.running.Done()

	for inbound := range a.options.Edge.Audio() {
		listener, err := a.listen(inbound.Participant)
		if err != nil {
			a.fail(err, "stt")
			continue
		}
		if err := listener.ProcessAudio(inbound.Audio, inbound.Participant); err != nil {
			a.fail(err, "stt")
		}
	}
}

// listen returns a participant's transcription session, opening one on first hearing them.
func (a *Agent) listen(participant stt.Participant) (*sttrouter.Session, error) {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil, errors.New("agent: closed")
	}
	if existing, ok := a.listeners[participant.ID]; ok {
		a.mu.Unlock()
		return existing, nil
	}
	ctx := a.ctx
	a.mu.Unlock()

	session, err := a.options.STT.Start(ctx, sttrouter.Request{
		CustomerID:    a.options.CustomerID,
		AgentID:       a.options.AgentID,
		CallID:        a.options.CallID,
		Tags:          a.options.Tags,
		Target:        a.options.STTTarget,
		LanguageHints: a.options.LanguageHints,
	})
	if err != nil {
		return nil, fmt.Errorf("agent: start stt for %s: %w", participant.ID, err)
	}

	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		_ = session.Close()
		return nil, errors.New("agent: closed")
	}
	a.listeners[participant.ID] = session
	a.mu.Unlock()

	a.running.Add(1)
	go a.consumeSTT(session)
	return session, nil
}

// consumeSTT feeds transcript revisions to the cadence controller.
func (a *Agent) consumeSTT(session *sttrouter.Session) {
	defer a.running.Done()

	for event := range session.Events() {
		switch typed := event.(type) {
		case stt.Transcript:
			if strings.TrimSpace(typed.Text) == "" {
				a.logger.Debug("the transcriber sent nothing but silence",
					"provider", typed.Provider, "participant", typed.Participant.ID, "mode", typed.Mode)
				continue
			}
			a.logger.Debug("transcribed",
				"provider", typed.Provider, "model", typed.Model,
				"participant", typed.Participant.ID, "mode", typed.Mode, "text", typed.Text,
				"confidence", typed.Confidence, "latency_ms", typed.ProcessingTimeMs)
			a.mu.Lock()
			a.lastHeardAt = time.Now()
			a.lastParticipant = typed.Participant
			a.mu.Unlock()
			if phrase := a.duplex.Heard(typed.Participant, typed.Text, a.quiet()); phrase != "" {
				a.backchannel(typed.Participant, phrase)
			}
			if superseded := a.cadence.Observe(typed); superseded != "" {
				a.logger.Debug("the caller said more, abandoning the decision in flight",
					"candidate", superseded)
				a.mu.Lock()
				delete(a.candidates, superseded)
				a.mu.Unlock()
				if err := a.harness.CancelDecision(superseded); err != nil {
					a.fail(err, "flow")
				}
			}

		case stt.Connected:
			a.logger.Info("listening", "provider", typed.Provider, "model", typed.Model)

		case stt.Disconnected:
			// An unclean disconnect is the transcriber going away mid-call, which reads
			// to everyone else as the caller having gone quiet.
			if typed.Clean {
				a.logger.Debug("the transcriber closed",
					"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)
				continue
			}
			a.logger.Warn("the transcriber dropped, nothing more will be heard from it",
				"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)

		case stt.Error:
			a.fail(typed.Err, "stt")
		}
	}
}

// consumeCadence asks the fast controller what to do once a transcript revision has held
// still for long enough.
func (a *Agent) consumeCadence() {
	defer a.running.Done()

	for {
		select {
		case ready := <-a.cadence.Ready():
			a.mu.Lock()
			if a.closed || a.harness == nil {
				a.mu.Unlock()
				continue
			}
			a.candidates[ready.ID] = ready
			history := append([]llm.Message(nil), a.history...)
			instructions := a.instructions()
			speaking := a.generating || a.utterances > 0
			current := a.harness
			a.mu.Unlock()

			a.logger.Debug("asking the flow controller what to do",
				"candidate", ready.ID, "text", ready.Text, "speaking", speaking,
				"confidence", ready.Confidence)

			if err := current.Decide(harness.FlowTurn{
				ID:           ready.ID,
				Instructions: instructions,
				History:      history,
				Participant:  participantName(ready.Participant),
				Text:         ready.Text,
				Speaking:     speaking,
			}); err != nil {
				a.mu.Lock()
				delete(a.candidates, ready.ID)
				a.mu.Unlock()
				a.cadence.Resolve(ready.ID, true)
				a.fail(err, "flow")
			}
		case <-a.ctx.Done():
			return
		}
	}
}

// consumePresence keeps long listening or thinking gaps from sounding like a dead call.
func (a *Agent) consumePresence() {
	defer a.running.Done()

	ticker := time.NewTicker(presenceTick)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			participant, hearing := a.cadence.Active()
			a.mu.Lock()
			current := a.harness
			lastParticipant := a.lastParticipant
			lastSpokeAt := a.lastSpokeAt
			a.mu.Unlock()
			if !hearing {
				participant = lastParticipant
			}
			if !hearing && (current == nil || !current.Delegating()) {
				continue
			}
			if phrase := a.duplex.Presence(participant, lastSpokeAt, a.quiet()); phrase != "" {
				a.backchannel(participant, phrase)
			}
		case <-a.ctx.Done():
			return
		}
	}
}

func (a *Agent) applyDecision(decision harness.Decided) {
	a.mu.Lock()
	ready, ok := a.candidates[decision.CandidateID]
	delete(a.candidates, decision.CandidateID)
	a.mu.Unlock()
	if !ok {
		a.logger.Debug("a decision arrived for a turn that has moved on",
			"candidate", decision.CandidateID)
		return
	}
	a.logger.Debug("the flow controller decided",
		"candidate", decision.CandidateID, "disposition", decision.Disposition,
		"floor", decision.Floor, "text", ready.Text)

	if decision.Err != nil || !decision.Valid() {
		a.cadence.Resolve(decision.CandidateID, true)
		a.fail(decision.Error(), "flow")
		return
	}
	if decision.Disposition == harness.Wait {
		a.cadence.Resolve(decision.CandidateID, true)
		return
	}
	if !a.cadence.Resolve(decision.CandidateID, false) {
		a.logger.Debug("not answering, the words changed while the controller was deciding",
			"candidate", decision.CandidateID)
		return
	}
	if decision.Disposition == harness.Ignore {
		a.logger.Debug("not answering, the controller read this as not meant for the agent",
			"candidate", decision.CandidateID, "text", ready.Text)
		return
	}

	a.emitter.Send(Heard{
		Participant: ready.Participant,
		Text:        ready.Text,
		Language:    ready.Language,
	})
	a.mu.Lock()
	previousCandidate := a.lastCandidateTurn
	a.lastCandidateTurn = ready.ID
	currentHarness := a.harness
	a.mu.Unlock()
	if previousCandidate != "" && previousCandidate != ready.ID && currentHarness != nil {
		currentHarness.CancelTurn(previousCandidate, harness.ReasonSuperseded)
	}

	if !a.quiet() {
		a.mu.Lock()
		overlapped := a.speakingTurn
		utterances, generating := a.utterances, a.generating
		a.mu.Unlock()
		a.logger.Debug("heard over the agent's own speech",
			"candidate", decision.CandidateID, "floor", decision.Floor,
			"overlapped", overlapped, "utterances", utterances, "generating", generating)
		a.emitter.Send(OverlapDecided{
			TurnID:      overlapped,
			Participant: ready.Participant,
			Action:      string(decision.Floor),
		})
		if decision.Floor == harness.Stop || decision.Floor == harness.Shorten {
			a.harness.CancelTurn(overlapped, harness.ReasonSuperseded)
		}
		switch decision.Floor {
		case harness.Stop:
			a.interrupt(ready.Participant)
		case harness.Shorten:
			a.queue(ready, decision.Disposition == harness.Clarify)
			a.shorten()
			return
		case harness.Continue:
			a.queue(ready, decision.Disposition == harness.Clarify)
			return
		}
	}
	a.logger.Debug("answering", "candidate", decision.CandidateID, "text", ready.Text)
	if err := a.respondCandidate(ready, decision.Disposition == harness.Clarify); err != nil {
		a.fail(err, "llm")
	}
}

// heard is when a turn became answerable, what the transcriber spent settling it, and how
// sure it was of the words.
type heard struct {
	at           time.Time
	sttLatencyMs float64
	confidence   float64
}

func participantName(participant stt.Participant) string {
	if participant.Name != "" {
		return participant.Name
	}
	if participant.UserID != "" {
		return participant.UserID
	}
	return participant.ID
}

// turnStamp names a turn. The clock is enough: a conversation cannot produce two turns
// in the same nanosecond.
func turnStamp() string { return strconv.FormatInt(time.Now().UnixNano(), 10) }

// respond asks the harness to reply to a turn.
func (a *Agent) respond(participant stt.Participant, text string, listened heard) error {
	return a.respondTurn(replyPrefix+turnStamp(), participant, text, listened, "")
}

func (a *Agent) respondCandidate(ready candidate, clarify bool) error {
	note := ""
	if clarify {
		note = "The caller addressed you, but their meaning is ambiguous. Ask one short clarifying question."
	}
	return a.respondTurn(ready.ID, ready.Participant, ready.Text, heard{
		at:           ready.ReadyAt,
		sttLatencyMs: ready.STTLatencyMs,
		confidence:   ready.Confidence,
	}, note)
}

// respondAfterTool asks for a reply to what a tool returned.
//
// There is nothing to add to the history, unlike respondTurn: it already ends with the
// tool's result, because the caller said nothing and the tool did. Without this the outcome
// waits there unspoken until the caller happens to talk again, which after a transfer that
// did not go through is somebody waiting to be handed somewhere they are not going.
func (a *Agent) respondAfterTool(turnID string) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	history := append([]llm.Message(nil), a.history...)
	participant := a.lastParticipant
	a.speakingTurn = turnID
	a.generating = true
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.begin(turnID, participant, time.Now(), 0)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant})

	return a.harness.Respond(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
	})
}

func (a *Agent) respondTurn(
	turnID string,
	participant stt.Participant,
	text string,
	listened heard,
	note string,
) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	a.history = append(a.history, llm.Message{Role: llm.User, Content: text})
	history := append([]llm.Message(nil), a.history...)

	a.speakingTurn = turnID
	a.generating = true
	a.lastParticipant = participant
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.begin(turnID, participant, listened.at, listened.sttLatencyMs)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Prompt: text})

	return a.harness.Respond(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		Note:         joinNotes(note, a.duplex.Note(listened.confidence)),
	})
}

func joinNotes(notes ...string) string {
	var kept []string
	for _, note := range notes {
		if strings.TrimSpace(note) != "" {
			kept = append(kept, note)
		}
	}
	return strings.Join(kept, "\n\n")
}

// backchannel makes a short listening noise while someone else is talking. It never goes
// near the model: a murmur is not a turn, and treating it as one would mean paying for a
// completion to say "mhm".
func (a *Agent) backchannel(participant stt.Participant, phrase string) {
	turnID := backchannelPrefix + turnStamp()

	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return
	}
	a.speakingTurn = turnID
	a.mu.Unlock()

	a.logger.Debug("murmuring while the caller talks",
		"turn", turnID, "participant", participant.ID, "phrase", phrase)
	if err := a.speakWhole(turnID, phrase); err != nil {
		a.fail(err, "tts")
		return
	}
	a.emitter.Send(Backchannel{Participant: participant, Text: phrase})
}

// quiet reports whether the agent has stopped talking.
func (a *Agent) quiet() bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.utterances == 0 && !a.generating
}

// instructions is the system prompt for a turn: what the agent was told to be, ahead of
// it whatever it already knew about the person it is talking to.
func (a *Agent) instructions() string {
	if a.recalled == "" {
		return a.prompt
	}
	if a.prompt == "" {
		return a.recalled
	}
	return a.recalled + "\n\n" + a.prompt
}

// consumeLLM turns the model's deltas into sentences and sends them to be spoken.
//
// It is the only goroutine that speaks.
func (a *Agent) consumeLLM() {
	defer a.running.Done()

	for event := range a.llm.Events() {
		a.handle(event)
	}
}

// handle deals with one event from the model.
func (a *Agent) handle(event llm.Event) {
	switch typed := event.(type) {
	case llm.TextDelta:
		if !a.speaking(typed.CompletionID) {
			// The turn was interrupted, so the rest of the reply is not spoken.
			return
		}
		a.say(typed.CompletionID, typed.Text)

	case llm.Error:
		a.fail(typed.Err, "llm")

	case llm.CompletionComplete:
		if !a.speaking(typed.CompletionID) {
			// The reply the chunker is holding is only cleared by the turn it belongs
			// to, so a completion arriving late for an abandoned one cannot cut the
			// turn after it short.
			if a.replying == typed.CompletionID {
				a.resetTurn()
			}
			return
		}
		a.finish(typed)
	}
}

// say sends one delta of a reply on its way to the voice.
func (a *Agent) say(turnID, delta string) {
	// What the model wrote is not all meant for the caller: a request for help is
	// addressed to the harness, and is taken out here rather than spoken.
	speech := a.harness.Filter(turnID, delta)
	if speech == "" {
		return
	}
	a.replying = turnID
	a.spoken.WriteString(speech)
	a.emitter.Send(ResponseDelta{TurnID: turnID, Text: speech})
	// The delta is the whole of the reply when there is no voice: a reader has already
	// been given it, and cutting it into sentences would only be for the speaking.
	if a.options.Text {
		return
	}

	for _, sentence := range a.chunk.Add(speech) {
		if err := a.speakSentence(turnID, sentence); err != nil {
			a.fail(err, "tts")
		}
	}
}

// finish closes out a reply the caller heard.
func (a *Agent) finish(typed llm.CompletionComplete) {
	// Text the harness was holding on the chance it began a request for help was only
	// ever text, so it is spoken.
	tail := a.harness.Flush()
	a.spoken.WriteString(tail)
	if a.options.Text {
		// There is no voice to release it to, so the held text is reported as the last
		// of the reply. Without this a reader would be missing whatever the harness was
		// still deciding about when the model stopped.
		if tail != "" {
			a.emitter.Send(ResponseDelta{TurnID: typed.CompletionID, Text: tail})
		}
	} else {
		for _, sentence := range a.chunk.Add(tail) {
			if err := a.speakSentence(typed.CompletionID, sentence); err != nil {
				a.fail(err, "tts")
			}
		}
		// Whatever did not end in punctuation is still worth saying.
		if remainder := a.chunk.Flush(); remainder != "" {
			if err := a.speakSentence(typed.CompletionID, remainder); err != nil {
				a.fail(err, "tts")
			}
		}
		// A model that reaches for a tool without a word leaves the caller listening to
		// nothing until it comes back, which on a phone is indistinguishable from having
		// been cut off. Prompting for it is not enough: the models that do it reliably
		// are not the ones fast enough to hold a conversation.
		if fillsPause(typed) && strings.TrimSpace(a.spoken.String()) == "" {
			filler := a.duplex.Working()
			a.spoken.WriteString(filler)
			if err := a.speakSentence(typed.CompletionID, filler); err != nil {
				a.fail(err, "tts")
			}
		}
		if err := a.closeUtterance(typed.CompletionID); err != nil {
			a.fail(err, "tts")
		}
	}
	// How many syntheses the turn produces is only settled once the reply is, and it is
	// what tells the tracker when the turn has finished being spoken.
	a.turns.completed(typed.CompletionID, typed.TimeToFirstTokenMs, a.expectedSyntheses(typed.CompletionID))
	said := strings.TrimSpace(a.spoken.String())
	a.resetTurn()

	a.mu.Lock()
	a.generating = false
	// A reply that only called a tool is still a turn the model took, and it has to be
	// recorded with the calls on it: the result sent back answers one of them, and a
	// provider refuses a conversation where it answers nothing.
	if said != "" || len(typed.ToolCalls) > 0 {
		a.history = append(a.history, llm.Message{
			Role:      llm.Assistant,
			Content:   said,
			ToolCalls: typed.ToolCalls,
		})
	}
	exchange := lastExchange(a.history)
	history := append([]llm.Message(nil), a.history...)
	currentHarness := a.harness
	a.mu.Unlock()

	// Remembering happens off the turn path: extraction takes longer than a turn and the
	// next thing the participant says must not wait for it.
	if a.memory != nil {
		a.memory.Remember(exchange)
	}
	if currentHarness != nil {
		if err := currentHarness.MaybeCompact(history, typed.InputTokens, typed.CachedInputTokens); err != nil {
			a.fail(err, "compaction")
		}
	}

	a.emitter.Send(Responded{
		TurnID:             typed.CompletionID,
		Text:               said,
		TimeToFirstTokenMs: typed.TimeToFirstTokenMs,
	})
	// Tools are handed over rather than run here, because this is the goroutine that
	// speaks and a transfer is several seconds of network the caller would hear as silence.
	if currentHarness != nil && len(typed.ToolCalls) > 0 {
		currentHarness.Requested(typed.CompletionID, typed.ToolCalls)
	}
	a.respondQueued()
}

// fillsPause reports whether a turn that said nothing should say something before the
// tools it asked for are run.
func fillsPause(typed llm.CompletionComplete) bool {
	// A turn that is itself a tool's answer gets no follow-up, so filling the pause on
	// one would leave the caller with a promise to check as the last thing they heard.
	if len(typed.ToolCalls) == 0 || strings.HasPrefix(typed.CompletionID, toolPrefix) {
		return false
	}
	// Pressing a menu option is meant to be silent. The menu answers next, and talking
	// over it is talking to nobody.
	return slices.ContainsFunc(typed.ToolCalls, func(call llm.ToolCall) bool {
		return call.Name != toolPress
	})
}

// consumeTTS publishes the agent's speech to the edge as it is synthesised.
func (a *Agent) consumeTTS() {
	defer a.running.Done()

	// Abandoned audio arrives a frame at a time, so it is reported once per utterance
	// rather than once per frame.
	dropping := ""

	for event := range a.tts.Events() {
		switch typed := event.(type) {
		case tts.AudioChunk:
			// Audio from an abandoned turn is dropped here rather than published, so
			// barge-in silences the agent even while the provider is still sending.
			if !a.speaking(turnOf(typed.SynthesisID)) {
				if dropping != typed.SynthesisID {
					dropping = typed.SynthesisID
					a.logger.Debug("dropping audio for a turn the agent has left behind",
						"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID))
				}
				continue
			}
			if err := a.options.Edge.PublishAudio(typed.Audio); err != nil {
				a.fail(err, "edge")
			}
			a.mu.Lock()
			a.lastSpokeAt = time.Now()
			a.mu.Unlock()
			// The wait ends when the participant can hear something, so this is timed
			// after the publish rather than before it.
			a.turns.firstAudio(turnOf(typed.SynthesisID), time.Now())

		case tts.SynthesisStarted:
			a.logger.Debug("the voice took an utterance",
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"provider", typed.Provider, "voice", typed.Voice)

		case tts.SynthesisComplete:
			a.settle()
			a.logger.Debug("finished speaking",
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"audio_ms", typed.AudioDurationMs, "ttfb_ms", typed.TimeToFirstByteMs,
				"interrupted", typed.Interrupted)
			a.respondQueued()
			a.turns.spoke(turnOf(typed.SynthesisID), typed.TimeToFirstByteMs, typed.AudioDurationMs)
			a.emitter.Send(Spoke{
				TurnID:            turnOf(typed.SynthesisID),
				AudioDurationMs:   typed.AudioDurationMs,
				TimeToFirstByteMs: typed.TimeToFirstByteMs,
			})
			// An answer that came back while the agent was talking waited for this.
			a.followUp()

		case tts.Connected:
			a.logger.Info("ready to speak", "provider", typed.Provider, "model", typed.Model)

		case tts.Disconnected:
			// Losing the voice mid-call is silence the caller hears as a dead line.
			if typed.Clean {
				a.logger.Debug("the voice closed",
					"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)
				continue
			}
			a.logger.Warn("the voice dropped, the agent has lost its speech",
				"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)

		case tts.Error:
			// A failure naming an utterance still settles it, since the router turns that
			// utterance's completion into the failed row.
			if typed.SynthesisID == "" {
				a.settle()
			}
			a.fail(typed.Err, "tts")
		}
	}
}

// consumeHarness reports what the harness decided, and speaks whatever the subagent came
// back with.
func (a *Agent) consumeHarness() {
	defer a.running.Done()
	defer close(a.harnessDrained)

	for event := range a.harness.Events() {
		switch typed := event.(type) {
		case harness.Decided:
			a.applyDecision(typed)

		case harness.Compacted:
			a.applyCompaction(typed)

		case harness.Delegated:
			a.emitter.Send(Delegated{
				TaskID: typed.TaskID,
				Skill:  typed.Skill,
				Prompt: typed.Prompt,
				TurnID: typed.TurnID,
			})

		case harness.ToolRequested:
			a.runTool(typed)

		case harness.Settled:
			if typed.State == harness.Cancelled {
				a.emitter.Send(TaskCancelled{
					TaskID: typed.TaskID,
					Skill:  typed.Skill,
					Reason: typed.Reason,
				})
				continue
			}
			a.emitter.Send(TaskSettled{
				TaskID:    typed.TaskID,
				Skill:     typed.Skill,
				Text:      typed.Text,
				Question:  typed.Question,
				ElapsedMs: typed.ElapsedMs,
				Err:       typed.Err,
			})
			if typed.Actionable() {
				a.followUp()
			}
		}
	}
}

func (a *Agent) applyCompaction(compacted harness.Compacted) {
	a.mu.Lock()
	if len(a.history) < len(compacted.Prefix) ||
		!sameMessages(a.history[:len(compacted.Prefix)], compacted.Prefix) {
		a.mu.Unlock()
		return
	}
	before := len(a.history)
	tail := append([]llm.Message(nil), a.history[len(compacted.Prefix):]...)
	a.history = append([]llm.Message{{
		Role:    llm.System,
		Content: "Earlier conversation summary:\n" + compacted.Summary,
	}}, tail...)
	after := len(a.history)
	a.mu.Unlock()

	a.emitter.Send(ConversationCompacted{
		Before:  before,
		After:   after,
		Summary: compacted.Summary,
	})
}

// sameMessages reports whether two stretches of history are the same turns.
//
// The comparison is field by field rather than whole-struct, because a message carries the
// tool calls it made and a slice cannot be compared with ==.
func sameMessages(first, second []llm.Message) bool {
	return slices.EqualFunc(first, second, func(left, right llm.Message) bool {
		return left.Role == right.Role &&
			left.Content == right.Content &&
			left.ToolCallID == right.ToolCallID &&
			slices.Equal(left.ToolCalls, right.ToolCalls)
	})
}

// follow starts a turn nobody asked for, because work the caller was told was coming has
// come back and they are owed the answer.
//
// An agent still speaking is left alone: taking the turn from itself would cut its own
// sentence off. What came back stays pending in the harness, and the last synthesis to
// settle tries again.
//
// The lock is held across the whole turn because deciding to speak and taking what there
// is to say must not be separable: two answers landing together would otherwise give one
// of them a turn with nothing in it.
func (a *Agent) follow() error {
	a.following.Lock()
	defer a.following.Unlock()

	a.mu.Lock()
	if a.harness == nil || a.utterances > 0 || !a.harness.Pending() {
		a.mu.Unlock()
		return nil
	}
	history := append([]llm.Message(nil), a.history...)
	turnID := replyPrefix + turnStamp()
	a.speakingTurn = turnID
	a.generating = true
	participant := a.lastParticipant
	instructions := a.instructions()
	a.mu.Unlock()

	// This turn is deliberately not measured. A Turn reports the wait between someone
	// finishing a sentence and hearing the answer start, and nobody said anything here.
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant})

	return a.harness.Respond(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
	})
}

// followUp speaks whatever the caller is owed, reporting a failure rather than returning
// it because nothing that calls it has anyone to return it to.
func (a *Agent) followUp() {
	if err := a.follow(); err != nil {
		a.fail(err, "llm")
	}
}

// delegating reports whether the subagent is still working on something.
func (a *Agent) delegating() bool {
	a.mu.Lock()
	current := a.harness
	a.mu.Unlock()
	return current != nil && current.Delegating()
}

// speakSentence sends one sentence of a reply to the voice.
//
// A streaming provider takes a turn's sentences as deltas of one utterance, which keeps a
// turn to a single billed synthesis. A provider that cannot take deltas gets each sentence
// as its own final request instead.
func (a *Agent) speakSentence(turnID, text string) error {
	voice := a.voice()
	if voice == nil {
		return errors.New("agent: not joined")
	}

	if !voice.Streaming() {
		id := fmt.Sprintf("%s%s%d", turnID, sentenceSuffix, a.sentences)
		a.sentences++
		a.begin()
		return voice.Synthesize(tts.Request{ID: id, Text: text, Final: true})
	}

	if a.openTurn != turnID {
		a.openTurn = turnID
		a.begin()
	}
	return voice.Synthesize(tts.Request{ID: turnID, Text: text})
}

// speakWhole says a piece of text that is already complete, as one utterance.
func (a *Agent) speakWhole(turnID, text string) error {
	voice := a.voice()
	if voice == nil {
		return errors.New("agent: not joined")
	}
	a.begin()
	return voice.Synthesize(tts.Request{ID: turnID, Text: text, Final: true})
}

// closeUtterance ends a streaming voice's utterance for a turn. A non-streaming one
// finished each sentence as it went, so there is nothing left to close.
func (a *Agent) closeUtterance(turnID string) error {
	voice := a.voice()
	if voice == nil || !voice.Streaming() || a.openTurn != turnID {
		return nil
	}
	return voice.Synthesize(tts.Request{ID: turnID, Final: true})
}

// expectedSyntheses is how many syntheses a finished reply will produce in total. A
// streaming voice takes the whole turn as one utterance; a voice that cannot take deltas
// got one request per sentence. Only the model consumer calls this, which is the same
// goroutine that maintains both counts.
func (a *Agent) expectedSyntheses(turnID string) int {
	if voice := a.voice(); voice != nil && voice.Streaming() {
		if a.openTurn == turnID {
			return 1
		}
		return 0
	}
	return a.sentences
}

// resetTurn forgets the text and the utterance of a turn that has ended, whether it
// finished or was interrupted.
func (a *Agent) resetTurn() {
	a.chunk.Reset()
	a.harness.Reset()
	a.spoken.Reset()
	a.sentences = 0
	a.openTurn = ""
}

// interrupt abandons the reply being spoken because a participant started talking.
func (a *Agent) interrupt(participant stt.Participant) {
	a.mu.Lock()
	turnID := a.speakingTurn
	if turnID == "" {
		a.mu.Unlock()
		return
	}
	// A murmur is meant to overlap with what someone is saying, so hearing them carry on
	// is not an interruption: there is no reply to abandon and nothing was cut short.
	if strings.HasPrefix(turnID, backchannelPrefix) {
		a.mu.Unlock()
		return
	}
	a.speakingTurn = ""
	a.generating = false
	model, voice := a.llm, a.tts
	a.mu.Unlock()

	a.logger.Debug("stopping mid-reply, the caller took the floor",
		"turn", turnID, "participant", participant.ID)

	if voice != nil {
		if err := voice.Interrupt(); err != nil {
			a.fail(err, "tts")
		}
	}
	if model != nil {
		if err := model.Interrupt(turnID); err != nil {
			a.fail(err, "llm")
		}
	}

	a.turns.interrupt(turnID)
	a.emitter.Send(Interrupted{TurnID: turnID, Participant: participant})
}

// shorten stops the model from adding more while allowing speech already sent to the
// voice to finish.
func (a *Agent) shorten() {
	a.mu.Lock()
	turnID := a.speakingTurn
	model := a.llm
	a.mu.Unlock()
	if turnID == "" || strings.HasPrefix(turnID, backchannelPrefix) || model == nil {
		return
	}
	a.logger.Debug("cutting the reply short, letting the audio already sent finish", "turn", turnID)
	if err := model.Interrupt(turnID); err != nil {
		a.fail(err, "llm")
	}
}

func (a *Agent) queue(ready candidate, clarify bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.queued != nil {
		a.logger.Debug("dropping a turn that never got answered",
			"candidate", a.queued.candidate.ID, "text", a.queued.candidate.Text)
	}
	a.logger.Debug("holding a turn until the agent stops talking",
		"candidate", ready.ID, "text", ready.Text, "clarify", clarify)
	a.queued = &queuedCandidate{candidate: ready, clarify: clarify}
}

func (a *Agent) respondQueued() {
	a.mu.Lock()
	if a.queued == nil || a.generating || a.utterances > 0 {
		a.mu.Unlock()
		return
	}
	queued := *a.queued
	a.queued = nil
	a.mu.Unlock()

	a.logger.Debug("answering the turn that was waiting",
		"candidate", queued.candidate.ID, "text", queued.candidate.Text)
	if err := a.respondCandidate(queued.candidate, queued.clarify); err != nil {
		a.fail(err, "llm")
	}
}

// speaking reports whether a turn is still the one allowed to produce audio.
func (a *Agent) speaking(turnID string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.speakingTurn != "" && a.speakingTurn == turnID
}

// voice returns the voice session, or nil when the agent has not joined.
func (a *Agent) voice() *ttsrouter.Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.tts
}

// begin counts an utterance as in flight, so Finish waits for it.
func (a *Agent) begin() {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.utterances++
}

// settle counts an utterance as finished.
func (a *Agent) settle() {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.utterances > 0 {
		a.utterances--
	}
}

// fail reports a failure without ending the conversation. One bad turn is a lost reply,
// not a lost call.
func (a *Agent) fail(err error, context string) {
	if err == nil {
		return
	}
	a.logger.Error("agent failure", "context", context, "error", err)
	a.emitter.Send(Error{Err: err, Context: context})
}

// lastExchange is the turn just finished: what was asked and what was answered. It is the
// smallest thing worth remembering, and a history not ending in an answered question has
// no finished exchange to offer.
func lastExchange(history []llm.Message) []llm.Message {
	if len(history) < 2 {
		return nil
	}
	pair := history[len(history)-2:]
	if pair[0].Role != llm.User || pair[1].Role != llm.Assistant {
		return nil
	}
	return append([]llm.Message(nil), pair...)
}

// turnOf strips the per-sentence suffix from a synthesis id, so audio can be matched to
// the turn it belongs to however the voice was fed.
func turnOf(synthesisID string) string {
	if index := strings.Index(synthesisID, sentenceSuffix); index >= 0 {
		return synthesisID[:index]
	}
	return synthesisID
}
