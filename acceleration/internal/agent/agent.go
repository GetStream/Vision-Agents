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
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// eventBuffer is how many events may queue before a slow consumer applies backpressure.
const eventBuffer = 64

// sentenceSuffix separates a turn id from the sequence number of a sentence within it, for
// providers that need one synthesis per sentence.
const sentenceSuffix = "#"

// promotionBuffer is how many guesses may be waiting to be spoken. More than one means
// two people finished a sentence at the same moment, which is rare and brief.
const promotionBuffer = 4

// Options configures an Agent.
//
// The three modalities arrive as routers plus targets rather than as providers, because the
// routers are what own failover and billing, and because speech-to-text is opened per
// participant rather than once for the call.
type Options struct {
	Edge Edge
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
	// RecallLimit caps how many memories are recalled on joining. Zero leaves the
	// store's own default.
	RecallLimit int
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
	// recalled is what the agent already knew on joining, rendered as a system message
	// and prepended to the instructions on every turn.
	recalled string

	// ctx is the call's lifetime. Every session the agent opens derives from it.
	ctx    context.Context
	cancel context.CancelFunc

	mu sync.Mutex
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
	joined     bool
	closed     bool

	// lastParticipant is who the agent was last talking to, so a reply prompted by
	// delegated work coming back is attributed to the person who is waiting for it.
	lastParticipant stt.Participant
	// guesses are replies begun on transcripts that had not settled. They are held here
	// rather than spoken, because the words they answer may yet be revoked.
	guesses map[string]*speculation

	// duplex tracks what each participant is in the middle of saying.
	duplex *duplex
	// promotions carries a guess that turned out to be right over to the goroutine that
	// owns the reply pipeline, since that is the only one allowed to speak.
	promotions chan promotion
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

// New validates the options and returns an Agent. It opens nothing; Join does that.
func New(options Options) (*Agent, error) {
	if options.Edge == nil {
		return nil, errors.New("agent: an edge is required")
	}
	if options.LLM == nil {
		return nil, errors.New("agent: an llm router is required")
	}
	if options.STT == nil {
		return nil, errors.New("agent: an stt router is required")
	}
	if options.TTS == nil {
		return nil, errors.New("agent: a tts router is required")
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
		listeners:  map[string]*sttrouter.Session{},
		guesses:    map[string]*speculation{},
		duplex:     newDuplex(options.Duplex),
		promotions: make(chan promotion, promotionBuffer),
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
		// Memories belong to the customer, and are recorded as a modality of their own so
		// what remembering costs is reported alongside what the models cost.
		agent.memory = newMemoryWriter(
			options.Memory,
			memory.Scope{AppID: options.AppID, UserID: options.CustomerID},
			owner,
			routing.NewRecorder(routing.Memory, options.Store, options.Live, logger),
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
		Model:     model,
		Subagent:  subagent,
		Skills:    a.options.Skills,
		Tasks:     a.options.Tasks,
		MaxTokens: a.options.MaxTokens,
		Logger:    a.logger,
	})
	if err != nil {
		return err
	}

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

	// What earlier conversations established is fetched before the call starts, so the
	// first turn is already answered in the light of it rather than the second.
	if a.memory != nil {
		a.recalled = memory.Prompt(a.memory.Recall(a.ctx, a.options.RecallLimit))
	}

	if err := a.options.Edge.Join(a.ctx); err != nil {
		return fmt.Errorf("agent: join edge: %w", err)
	}

	a.mu.Lock()
	a.harnessDrained = make(chan struct{})
	a.mu.Unlock()

	a.running.Add(4)
	go a.consumeLLM()
	go a.consumeTTS()
	go a.consumeEdge()
	go a.consumeHarness()

	a.logger.Info("joined",
		"llm", a.llm.Provider()+"/"+a.llm.Model(),
		"tts", a.tts.Provider()+"/"+a.tts.Model())
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
	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	turnID := fmt.Sprintf("say-%d", time.Now().UnixNano())
	a.speakingTurn = turnID
	a.mu.Unlock()

	return a.speakWhole(turnID, text)
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

	if cancel != nil {
		cancel()
	}

	// The edge leaves first: it is the source of the audio that keeps the rest busy.
	var failures []error
	if err := a.options.Edge.Leave(); err != nil {
		failures = append(failures, fmt.Errorf("leave edge: %w", err))
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
		// Guessing at a reply needs the transcriber to say when it thinks a turn has
		// ended, before it is sure.
		EagerTurns: a.options.Duplex.Speculate,
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

// consumeSTT turns one participant's speech into turns the agent answers.
func (a *Agent) consumeSTT(session *sttrouter.Session) {
	defer a.running.Done()

	for event := range session.Events() {
		switch typed := event.(type) {
		case stt.TurnStarted:
			// Someone talking again means whatever was guessed at was only half of what
			// they had to say.
			if abandoned := a.duplex.Began(typed.Participant); abandoned != "" {
				a.abandon(abandoned)
			}
			// Barge-in. The provider's own turn detection decides this, which is why there
			// is no separate voice-activity detector here.
			a.interrupt(typed.Participant)

		case stt.TurnEnded:
			// A provisional end of turn is the transcriber saying it thinks they have
			// finished, while reserving the right to change its mind.
			if !typed.Eager {
				continue
			}
			// The event itself carries no words, so what the reply answers is the last
			// revision of the transcript, which arrived just before it.
			guessing := a.duplex.Interim(typed.Participant)
			turnID, abandoned, ok := a.duplex.Eager(typed.Participant, guessing)
			if abandoned != "" {
				a.abandon(abandoned)
			}
			if !ok {
				continue
			}
			if err := a.speculate(turnID, typed.Participant, guessing); err != nil {
				a.fail(err, "llm")
			}

		case stt.Transcript:
			// Only settled turns are answered. An interim transcript is a revision of a
			// turn that has not finished, so replying to one means replying to half a
			// sentence — but it is worth letting the person know they are being heard.
			if !typed.Final() {
				if phrase := a.duplex.Heard(typed.Participant, typed.Text, a.quiet()); phrase != "" {
					a.backchannel(typed.Participant, phrase)
				}
				continue
			}
			if strings.TrimSpace(typed.Text) == "" {
				continue
			}
			a.emitter.Send(Heard{
				Participant: typed.Participant,
				Text:        typed.Text,
				Language:    typed.Language,
			})
			// The wait the participant feels starts here, with the transcript that
			// settled their turn, not with the request the agent then makes.
			heard := heard{
				at:           time.Now(),
				sttLatencyMs: typed.ProcessingTimeMs,
				confidence:   typed.Confidence,
			}

			promoted, abandoned := a.duplex.Settled(typed.Participant, typed.Text)
			if abandoned != "" {
				a.abandon(abandoned)
			}
			if promoted != "" {
				// The answer is already half written, so it goes to the goroutine that
				// is allowed to speak it rather than being started again here.
				a.promotions <- promotion{
					turnID:      promoted,
					participant: typed.Participant,
					text:        typed.Text,
					listened:    heard,
				}
				continue
			}
			if err := a.respond(typed.Participant, typed.Text, heard); err != nil {
				a.fail(err, "llm")
			}

		case stt.Error:
			a.fail(typed.Err, "stt")
		}
	}
}

// heard is when a turn became answerable, what the transcriber spent settling it, and how
// sure it was of the words.
type heard struct {
	at           time.Time
	sttLatencyMs float64
	confidence   float64
}

// turnStamp names a turn. The clock is enough: a conversation cannot produce two turns
// in the same nanosecond.
func turnStamp() string { return strconv.FormatInt(time.Now().UnixNano(), 10) }

// respond asks the harness to reply to a turn.
func (a *Agent) respond(participant stt.Participant, text string, listened heard) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	a.history = append(a.history, llm.Message{Role: llm.User, Content: text})
	history := append([]llm.Message(nil), a.history...)

	turnID := replyPrefix + turnStamp()
	a.speakingTurn = turnID
	a.lastParticipant = participant
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.begin(turnID, participant, listened.at, listened.sttLatencyMs)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Prompt: text})

	return a.harness.Respond(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		Note:         a.duplex.Note(listened.confidence),
	})
}

// speculation is a reply begun on a transcript the transcriber had not settled. It is
// kept rather than spoken until the turn really does end on the same words.
type speculation struct {
	participant stt.Participant
	// guessed is the provisional transcript the reply answers.
	guessed string
	// text is the reply as it arrives, unfiltered: a guess never delegates, because the
	// sentence it was made on may not have been said.
	text strings.Builder
	// complete is set once the model has finished, so a guess promoted after the fact is
	// still closed out properly.
	complete *llm.CompletionComplete
}

// promotion is a guess that turned out to be right, on its way to the goroutine allowed
// to speak it.
type promotion struct {
	turnID      string
	participant stt.Participant
	// text is the settled transcript, which is what the conversation remembers.
	text     string
	listened heard
}

// speculate starts answering a turn the transcriber has provisionally ended.
//
// The reply is not spoken and the conversation is not told about it: what this buys is
// the model's time to first token, which is most of the wait, at the price of a
// completion that is sometimes thrown away.
func (a *Agent) speculate(turnID string, participant stt.Participant, text string) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return errors.New("agent: not joined")
	}
	// Guessing while something is waiting to be said would take what there is to say into
	// a reply that may never be heard.
	if a.harness.Pending() {
		a.mu.Unlock()
		return nil
	}
	history := append(append([]llm.Message(nil), a.history...), llm.Message{Role: llm.User, Content: text})
	a.guesses[turnID] = &speculation{participant: participant, guessed: text}
	instructions := a.instructions()
	a.mu.Unlock()

	return a.harness.Respond(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
	})
}

// hold keeps a delta of a speculative reply, reporting whether it belonged to one.
func (a *Agent) hold(turnID, delta string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	guess, ok := a.guesses[turnID]
	if !ok {
		return false
	}
	guess.text.WriteString(delta)
	return true
}

// held records that a speculative reply has finished, reporting whether it was one. The
// guess stays: it is still worth having if the turn settles on the same words.
func (a *Agent) held(complete llm.CompletionComplete) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	guess, ok := a.guesses[complete.CompletionID]
	if !ok {
		return false
	}
	guess.complete = &complete
	return true
}

// abandon throws away a reply guessed at on words that were not said after all.
func (a *Agent) abandon(turnID string) {
	a.mu.Lock()
	guess, ok := a.guesses[turnID]
	delete(a.guesses, turnID)
	a.mu.Unlock()

	if !ok {
		return
	}
	if guess.complete == nil {
		if err := a.llm.Interrupt(turnID); err != nil {
			a.fail(err, "llm")
		}
	}
	a.emitter.Send(Speculated{
		TurnID:      turnID,
		Participant: guess.participant,
		Text:        guess.guessed,
		Promoted:    false,
	})
}

// promote turns a reply that was guessed at into the one being spoken, because the turn
// settled on the words it was guessed from. It runs on the goroutine that owns the reply
// pipeline, which is the only one allowed to speak.
func (a *Agent) promote(promoted promotion) {
	a.mu.Lock()
	guess, ok := a.guesses[promoted.turnID]
	if !ok {
		// The caller carried on talking between the guess being found right and it being
		// promoted, so it was abandoned in the meantime.
		a.mu.Unlock()
		return
	}
	delete(a.guesses, promoted.turnID)
	a.history = append(a.history, llm.Message{Role: llm.User, Content: promoted.text})
	a.speakingTurn = promoted.turnID
	a.lastParticipant = promoted.participant
	a.mu.Unlock()

	a.turns.begin(promoted.turnID, promoted.participant, promoted.listened.at, promoted.listened.sttLatencyMs)
	a.emitter.Send(Responding{
		TurnID:      promoted.turnID,
		Participant: promoted.participant,
		Prompt:      promoted.text,
	})
	a.emitter.Send(Speculated{
		TurnID:      promoted.turnID,
		Participant: promoted.participant,
		Text:        guess.guessed,
		Promoted:    true,
	})

	// Everything the model had already written is spoken at once, which is the whole
	// point: the answer was being generated while the caller was still finishing.
	a.say(promoted.turnID, guess.text.String())
	if guess.complete != nil {
		a.finish(*guess.complete)
	}
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
	return a.utterances == 0
}

// instructions is the system prompt for a turn: what the agent was told to be, ahead of
// it whatever it already knew about the person it is talking to.
func (a *Agent) instructions() string {
	if a.recalled == "" {
		return a.options.Instructions
	}
	if a.options.Instructions == "" {
		return a.recalled
	}
	return a.recalled + "\n\n" + a.options.Instructions
}

// consumeLLM turns the model's deltas into sentences and sends them to be spoken.
//
// It is the only goroutine that speaks, which is why a guess that turned out to be right
// is promoted here rather than where it was found to be right.
func (a *Agent) consumeLLM() {
	defer a.running.Done()

	events := a.llm.Events()
	for {
		select {
		case event, open := <-events:
			if !open {
				return
			}
			a.handle(event)
		case promoted := <-a.promotions:
			a.promote(promoted)
		}
	}
}

// handle deals with one event from the model.
func (a *Agent) handle(event llm.Event) {
	switch typed := event.(type) {
	case llm.TextDelta:
		// A reply begun on a transcript that had not settled is kept rather than spoken,
		// because the words it answers may yet be revoked.
		if a.hold(typed.CompletionID, typed.Text) {
			return
		}
		if !a.speaking(typed.CompletionID) {
			// The turn was interrupted, so the rest of the reply is not spoken.
			return
		}
		a.say(typed.CompletionID, typed.Text)

	case llm.Error:
		a.fail(typed.Err, "llm")

	case llm.CompletionComplete:
		if a.held(typed) {
			return
		}
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
	if err := a.closeUtterance(typed.CompletionID); err != nil {
		a.fail(err, "tts")
	}
	// How many syntheses the turn produces is only settled once the reply is, and it is
	// what tells the tracker when the turn has finished being spoken.
	a.turns.completed(typed.CompletionID, typed.TimeToFirstTokenMs, a.expectedSyntheses(typed.CompletionID))
	said := strings.TrimSpace(a.spoken.String())
	a.resetTurn()

	a.mu.Lock()
	if said != "" {
		a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: said})
	}
	exchange := lastExchange(a.history)
	a.mu.Unlock()

	// Remembering happens off the turn path: extraction takes longer than a turn and the
	// next thing the participant says must not wait for it.
	if a.memory != nil {
		a.memory.Remember(exchange)
	}

	a.emitter.Send(Responded{
		TurnID:             typed.CompletionID,
		Text:               said,
		TimeToFirstTokenMs: typed.TimeToFirstTokenMs,
	})
}

// consumeTTS publishes the agent's speech to the edge as it is synthesised.
func (a *Agent) consumeTTS() {
	defer a.running.Done()

	for event := range a.tts.Events() {
		switch typed := event.(type) {
		case tts.AudioChunk:
			// Audio from an abandoned turn is dropped here rather than published, so
			// barge-in silences the agent even while the provider is still sending.
			if !a.speaking(turnOf(typed.SynthesisID)) {
				continue
			}
			if err := a.options.Edge.PublishAudio(typed.Audio); err != nil {
				a.fail(err, "edge")
			}
			// The wait ends when the participant can hear something, so this is timed
			// after the publish rather than before it.
			a.turns.firstAudio(turnOf(typed.SynthesisID), time.Now())

		case tts.SynthesisComplete:
			a.settle()
			a.turns.spoke(turnOf(typed.SynthesisID), typed.TimeToFirstByteMs, typed.AudioDurationMs)
			a.emitter.Send(Spoke{
				TurnID:            turnOf(typed.SynthesisID),
				AudioDurationMs:   typed.AudioDurationMs,
				TimeToFirstByteMs: typed.TimeToFirstByteMs,
			})
			// An answer that came back while the agent was talking waited for this.
			a.followUp()

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
		case harness.Delegated:
			a.emitter.Send(Delegated{
				TaskID: typed.TaskID,
				Skill:  typed.Skill,
				Prompt: typed.Prompt,
				TurnID: typed.TurnID,
			})

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
	model, voice := a.llm, a.tts
	a.mu.Unlock()

	if voice != nil {
		if err := voice.Interrupt(); err != nil {
			a.fail(err, "tts")
		}
	}
	if model != nil {
		if err := model.Interrupt(); err != nil {
			a.fail(err, "llm")
		}
	}

	a.turns.interrupt(turnID)
	a.emitter.Send(Interrupted{TurnID: turnID, Participant: participant})
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
