// Package agent runs a voice conversation over the three routed modalities.
//
// It is the Go counterpart of the Python Agent in sdks/python: audio from the edge is
// transcribed, settled turns are answered by a model, and the reply is spoken back. What
// makes it worth having in this service is that it is built from the routers rather than
// from provider instances, so every turn is routed, failed over and billed by the same
// machinery as a direct API call.
package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"math"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/guardrail"
	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/live"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
	"github.com/GetStream/Vision-Agents/acceleration/internal/searchrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stsrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// eventBuffer is how many events may queue before a slow consumer applies backpressure.
const eventBuffer = 64

// appLabel is the memory label a caller's own app id is kept under.
const appLabel = "app_id"

// replyBuffer is how many deltas may queue across every reply being generated before the
// goroutine draining one waits on the goroutine that speaks.
const replyBuffer = 64

const (
	presenceTick       = 500 * time.Millisecond
	playoutWaitCeiling = 2 * time.Second
)

// sentenceSuffix separates a turn id from the sequence number of a sentence within it, for
// providers that need one synthesis per sentence.
const sentenceSuffix = "#"

// interruptedReplyNote applies to the next caller response after we preserve generated
// text from an interrupted cascade reply. It keeps that context from sounding like a
// promise to continue unless the caller asks.
const interruptedReplyNote = "The previous reply was interrupted; generated assistant text in the conversation history may not have been heard in full. Answer the latest caller turn, and resume the previous reply only if the caller asks."

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
	// Caller is the end user who asked for this conversation, which is who its daily
	// limits are counted against. Empty for one a customer's own backend started, which is
	// not limited.
	Caller routing.Caller
	// AgentID identifies this agent across calls. Transcripts are stored under it and
	// every request the agent makes is recorded against it.
	AgentID string
	// ConfigID names the agent config this call was created from. It is the prompt cache
	// key, because a config is exactly the set of calls whose instructions are identical:
	// the first turn of the first call writes them to the provider's cache and every turn
	// of every call after that reads them back. Empty leaves prompt caching implicit.
	ConfigID string
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
	// STS routes one native audio model that hears the caller and speaks back, in place
	// of the three above. When STSTarget is set the agent is native: it opens that model
	// and no transcriber, conversation model or voice, so LLM, STT and TTS may all be nil
	// unless the session may later be moved back onto a cascade. The
	// model owns endpointing, transcription, synthesis and barge-in, and nothing here acts
	// as any of them. LLM is still used for the subagent.
	STS       *stsrouter.Router
	STSTarget string

	// SubagentTarget routes the slower, more capable model that runs the work the voice
	// model hands over. It is a target on the same router as LLMTarget, because the
	// difference between them is which model, not which service. Empty means the agent
	// answers everything itself.
	SubagentTarget string
	// ControllerTarget routes the flow controller, a fast non-thinking classifier that
	// only ever returns one small JSON object about who holds the floor. It is a target on
	// the same router as LLMTarget. Empty falls back to LLMTarget, so a caller who names no
	// controller shares the conversation's model.
	ControllerTarget string
	// Skills are what the voice model may hand over. They mean nothing without a
	// subagent to run them.
	Skills harness.Skills
	// Telephony is what the agent may do to the call itself. Without it the agent can
	// only talk, which is what a call that is not on a phone network can do.
	Telephony Telephony
	// ToolRunner carries out the tools that are not the two acting on the phone call,
	// which is how a caller outside this process owns its own tools.
	ToolRunner    ToolRunner
	OnToolStarted func(ToolStarted)
	// Tools are what the voice model may do rather than say. Each is only offered when
	// something on this call can run it: the telephony pair needs Telephony, and every
	// other tool needs a ToolRunner.
	Tools harness.Tools
	// Sandbox is where the subagent runs code it writes. It is never offered to the model
	// holding the conversation: running code takes seconds, and a conversation cannot
	// spare them.
	Sandbox sandbox.Sandbox
	// Publish puts the files the subagent's code hands back where the caller can see them,
	// which is the conversation's channel when there is one. Nil means nowhere.
	Publish sandbox.Publisher
	// Tasks caps how much delegated work may run at once. Zero leaves the harness's own
	// default in place.
	Tasks int
	// Duplex lets the agent listen and talk at the same time rather than strictly taking
	// turns. Both halves of it are off by default.
	Duplex         DuplexOptions
	VideoSource    string
	VideoMaxFrames int
	// EOT is an optional raw acoustic endpoint score for settled voice candidates.
	EOT *EOTClient
	// EOTMode selects whether EOT gates or directly resolves eligible quiet-floor turns.
	EOTMode      EOTMode
	EOTThreshold float64

	// SpeculativeReplies starts the reply to a settled turn while the flow controller is
	// still deciding whether it was meant for the agent, and holds it until the ruling says
	// to answer. It takes the ruling's round trip off every answered turn, and costs the
	// tokens of the replies a ruling throws away. Nil leaves it on, and a pointer to false
	// asks for the reply only once the ruling is in.
	SpeculativeReplies *bool
	// ReplySilence is how long a caller must have been quiet before the first audio of the
	// reply to them is let out, measured on their audio rather than on their words. A reply
	// that is ready sooner waits for it, and a voice in the meantime only restarts the count:
	// the wait delays the reply and never drops it. Nil leaves it at 700ms, and a pointer to
	// zero lets a reply start as soon as it is ready. It does not apply to a greeting, a
	// murmur, or a turn the agent takes without having been spoken to.
	ReplySilence *time.Duration
	// ReplySilenceMax is the longest the first audio of a reply is held for ReplySilence once
	// it is ready. A caller whose line never goes quiet, because of a conversation in the room
	// or a steady babble, does not confirm the silence, and the reply is let out when this has
	// passed. Nil leaves it at one second, and it must be longer than zero while ReplySilence
	// is on.
	ReplySilenceMax *time.Duration
	// ReplySilenceConfident is how long a caller must have been quiet, in place of ReplySilence,
	// when their turn was decided by a successful acoustic end-of-turn score of at least
	// ReplyConfidentScore: the silence is there for endings that are in doubt, and a score that
	// high says this one is not. The normal silence applies to every other turn, including one the
	// flow controller decided. Nil leaves it at 300ms, and a pointer to zero lets such a reply
	// out as soon as it is ready.
	ReplySilenceConfident *time.Duration
	// ReplyConfidentScore is the acoustic end-of-turn score from which the turn is taken to have
	// ended for sure. It must be between 0 and 1. Nil leaves it at 0.9, and a pointer to zero
	// turns the shorter silence off.
	ReplyConfidentScore *float64
	// PreviewDebounce is how long a caller's words have to hold still before the reply to them
	// is started, ahead of the wait that decides whether they have finished, so the model has
	// been working for part of that wait. Words that change again restart it, and words that
	// end on a comma, a joining word or a hesitation are not previewed at all. It applies
	// wherever SpeculativeReplies does. Nil leaves it at 60ms, and a pointer to zero starts
	// the reply when the wait is over, as it did before.
	PreviewDebounce *time.Duration
	// PreviewQuiet is how long a caller's audio has to have been quiet, as well as their words
	// having held still for PreviewDebounce, before the reply to them is started ahead of that
	// wait. It is looked at again when the debounce runs out, and at most three replies are
	// started this way for one run of the caller's words: after that the reply is started when
	// the wait is over. Nil leaves it at 120ms, and a pointer to zero looks at the words alone.
	PreviewQuiet *time.Duration

	// Voice selects the speaker. Its meaning is the text-to-speech provider's.
	Voice string
	// Speed is the voice's rate of delivery, 1 being its own. Zero leaves it there, and a
	// voice that cannot be sped up is not routed to when it is set.
	Speed float64
	// LanguageHints narrow the candidates in every modality.
	LanguageHints []string
	// Keyterms are the business-specific words the transcriber should expect. A provider
	// that cannot be told about vocabulary ignores them.
	Keyterms []string
	// MaxTokens caps each reply. Zero leaves the model's own default in place.
	MaxTokens int
	// Overwrites is what whoever opened the session asked to change about how the model
	// answers, written over every turn. Only the safe knobs are here -- effort, length,
	// randomness, verbosity -- because instructions and tools belong to the agent and a
	// caller able to rewrite those could make a session impersonate a different agent.
	Overwrites options.LLM
	// Memory carries what earlier conversations established into this one. Without it
	// the agent starts every call knowing nothing but its instructions.
	Memory memory.Store
	// AppID narrows memories within the customer, so two deployments of one customer do
	// not read each other's. It is kept as a label rather than as the app id, which is
	// always the customer, so it can never reach another customer's.
	AppID string
	// SessionID is the session memories are learned in.
	SessionID string
	// Incognito recalls memories but writes none: what is said here is not kept anywhere.
	Incognito bool
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
	// Search is what the agent may find out that neither it nor the handbook knows,
	// because it depends on today. Without it, or without a target, the search tool is not
	// offered, for the same reason the lookup is not: an agent that promises to check and
	// then cannot is worse than one that never offered.
	Search *searchrouter.Router
	// SearchTarget routes the search, and is a target on that router the way LLMTarget is
	// on its own.
	SearchTarget string
	// Guardrail screens what a caller asks before the agent answers it. Nil is an agent
	// that answers everything, which is every agent that declared no policy.
	Guardrail guardrail.Guardrail
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

	// pipe is the pipeline hearing and answering on the call. A swap replaces it.
	pipe *pipeline
	// nativeMode is whether pipe is a speech-to-speech model, read on every audio chunk.
	nativeMode atomic.Bool
	// switching holds the floor while a swap changes the pipeline: no turn starts and no
	// audio is taken in.
	switching atomic.Bool
	// swapping serialises swaps.
	swapping sync.Mutex

	llm *llmrouter.Session
	// replies is where every reply in flight is fanned in to, so the one goroutine that
	// speaks stays one goroutine however many turns are being generated at once. Each
	// cascade pipeline has its own.
	replies chan llm.Event
	// streams are the replies still being generated, by turn. Closing one is barge-in.
	streams        map[string]*llm.Stream
	previews       map[string]*replyPreview
	modelCallTimes map[string]float64
	// kept are the previews held across a Wait, by participant: the reply started for words
	// the flow controller asked to wait on, which the next check of the same words takes
	// over instead of asking the model for it again.
	kept map[string]*keptPreview
	// previewTurns says which candidate took over the preview started under another one's
	// id, so model calls reported under the old id count towards the turn that is using them.
	previewTurns map[string]string
	// voiced follows when each participant's audio last carried a voice. It is nil when a reply
	// is not held to the caller's silence.
	voiced *voiceActivity
	// replySilence is how long a caller must have been quiet before the first audio of the
	// reply to them is let out.
	replySilence time.Duration
	// replySilenceMax is the longest that first audio is held for it once it is ready, so a
	// line that never goes quiet cannot stop a reply from being heard.
	replySilenceMax time.Duration
	// replySilenceConfident is the silence a reply is held for instead when the acoustic
	// end-of-turn score that decided its turn was at least replyConfidentScore.
	replySilenceConfident time.Duration
	replyConfidentScore   float64
	// confident are the candidates whose turn such a score decided, until the ruling has been
	// carried out.
	confident map[string]struct{}
	// generatingCancel abandons a conversation Create that has not returned a stream yet.
	// Interrupt used to Close only an existing stream, so a reply waiting on headers kept
	// the event loop and the floor until Cerebras answered.
	generatingCancel map[string]context.CancelFunc
	toolCancels      map[string]context.CancelFunc
	// pumps are the goroutines draining those streams into replies.
	pumps sync.WaitGroup

	tts *ttsrouter.Session
	// sts is the native audio session, and is what a native agent has instead of llm,
	// tts and listeners. The harness still runs delegated work.
	sts *stsrouter.Session
	// harness stands between what a participant said and the model that answers them. It
	// decides what the model is asked, and takes the model's requests for help back out
	// of the reply before any of it reaches the voice.
	harness *harness.Harness

	// turns measures each exchange end to end and, when a store is configured, records it.
	turns     *turnTracker
	turnStore *turnRecorder
	// decisionStore keeps what the conversation decided, so a call can be read back after
	// the process that held it is gone.
	decisionStore *decisionRecorder
	// memory carries what earlier conversations established into this one.
	memory *memoryWriter
	// knowledge answers what the business already wrote down, when the agent has a
	// namespace to read.
	knowledge *knowledgeReader
	// searcher answers what is true now, when the deployment has a search provider. It is
	// nil until Start has routed one.
	searcher *searchrouter.Session
	// recalled is what the agent already knew on joining, rendered as a system message
	// and prepended to the instructions on every turn.
	recalled string
	// voicePrompt is what the voice asked to have said about it, appended to the
	// instructions so the model writes lines the voice can actually perform.
	voicePrompt string
	// performs reports whether the voice acts bracketed directions. When it does not,
	// they are taken out before it is asked to speak rather than read out as words.
	performs bool

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
	// audioHistory retains one bounded PCM window per active participant for optional EOT.
	audioHistory       map[string]*pcm16leRing
	eotGates           map[string]*eotGate
	eotSnapshotOrdinal uint64
	// voices is the diarised label of the first voice heard on each participant's track,
	// which is taken to be the caller's. A later turn in a different voice is somebody
	// else at the same microphone: the track says who joined the call, and it is the
	// wrong answer for everybody else in the room with them. Empty for the transcribers
	// that cannot tell one voice from another, which leaves it saying nothing rather
	// than guessing.
	voices map[string]string
	// speakingTurn is the turn the agent is currently on. It says which reply an
	// interruption would abandon and which one the floor belongs to; it does not decide
	// what may be heard, because the agent starts a turn for itself while the turn before
	// it is still being spoken.
	speakingTurn string
	// gated is the reply to a caller's words that has not let out any of its audio yet, which
	// makes it one the caller has heard nothing of.
	gated heldReply
	// saying is the filtered generated text available for the current reply. It may be ahead
	// of playout, so it gives overlap judgments and interruption context without claiming
	// that every generated word reached the caller.
	saying string
	// interruptedReplyPending means the previous cascade reply was interrupted before its
	// audio was known to be fully heard. The next real caller reply gets a private note;
	// previews can use it but do not consume it.
	interruptedReplyPending bool
	// playoutCtx is shared by every synthesis sent since the last interruption. Cancelling
	// it stops a writer already in the edge; synthesisContexts keeps late chunks attached
	// to the epoch that owned their request instead of a newer one.
	playoutCtx    context.Context
	cancelPlayout context.CancelFunc
	synthesisCtx  map[string]context.Context
	interruptDone chan struct{}
	// abandoned is every turn an interruption gave up on. Audio belonging to one of them
	// is dropped rather than published, which is what makes barge-in immediate even while
	// a provider is still sending. It holds one entry per interruption and lives only as
	// long as the call.
	abandoned map[string]struct{}
	// utterances counts syntheses that have not settled, so Finish knows when the agent
	// has stopped talking.
	utterances int
	// generating is true while the voice model is still writing the current reply.
	generating bool
	// toolReply is set when a tool returned and the caller has not been told yet. A
	// second tool in the same turn must not start a competing generate: it would steal
	// speakingTurn and drop the first result unspoken.
	toolReply bool
	// pendingTools is how many tool calls from the current turn have not come back yet.
	// The spoken follow-up waits until this is zero so two results share one generate.
	pendingTools int
	// owedTurn is the last turn that ended with tools or delegated work outstanding, which
	// the reply delivering that work continues.
	owedTurn string
	joined   bool
	closed   bool

	// lastParticipant is who the agent was last talking to, so a reply prompted by
	// delegated work coming back is attributed to the person who is waiting for it.
	lastParticipant stt.Participant
	lastHeardAt     time.Time
	lastSpokeAt     time.Time
	// The native agent's own state. A speech-to-speech model takes one stream, so bound
	// is the participant whose audio it hears and arrivals is everyone whose audio has
	// come in, heard or not, which is what a warm transfer counts. waitingSince is when the
	// caller stopped and has not been answered, which a native reply is timed from; zero
	// when nothing is owed. hearing is what the model has written down of the turn so far,
	// and heardText the last turn it settled, which is what the reply is a reply to.
	bound           stt.Participant
	arrivals        map[string]struct{}
	waitingSince    time.Time
	hearing         strings.Builder
	heardText       string
	nativeListening bool
	nativeAwaiting  bool
	nativeCalls     map[string]struct{}
	// cadence turns evolving transcript revisions into stable candidates without relying
	// on provider turn boundaries.
	cadence *cadence
	// converse makes every judgement about how to handle the call, and reports each one
	// so the reasoning behind a conversation can be read back rather than inferred.
	converse *converse

	// duplex tracks listening acknowledgements and transcript confidence.
	duplex *duplex
	// harnessDrained closes once every harness event has been reported, which is what
	// lets shutdown report the work it abandoned on the way out. It is nil until the
	// consumer that closes it is running.
	harnessDrained chan struct{}

	// chunk assembles model deltas into sentences. Only the model consumer touches it.
	chunk chunker
	// directions takes the voice's stage directions out of the reply. They are meant for
	// the provider that can act them, so they never reach a reader or the history.
	directions tts.Directions
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
	if options.EOTMode == "" {
		options.EOTMode = EOTModeGate
	}
	if !options.EOTMode.valid() {
		return nil, errors.New("agent: EOT mode must be gate or primary")
	}
	native := options.STSTarget != ""
	if native && options.STS == nil {
		return nil, stack.Wrap(errors.New("agent: an sts router is required"))
	}
	if !native && options.LLM == nil {
		return nil, stack.Wrap(errors.New("agent: an llm router is required"))
	}
	if options.Text && native {
		return nil, stack.Wrap(errors.New("agent: a text agent has no voice, so it cannot run a speech-to-speech model"))
	}
	if options.LLM == nil && options.SubagentTarget != "" {
		return nil, stack.Wrap(errors.New("agent: a subagent requires an llm router"))
	}
	// A conversation in writing has nowhere to listen and nothing to speak with, so the
	// three that carry a voice are only required when there is one. A native agent's one
	// model is its transcriber and its voice both, so it needs neither router.
	if !options.Text {
		if options.Edge == nil {
			return nil, stack.Wrap(errors.New("agent: an edge is required"))
		}
		if !native {
			if options.STT == nil {
				return nil, stack.Wrap(errors.New("agent: an stt router is required"))
			}
			if options.TTS == nil {
				return nil, stack.Wrap(errors.New("agent: a tts router is required"))
			}
		}
	}
	if options.CustomerID == "" {
		return nil, stack.Wrap(errors.New("agent: a customer id is required"))
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	if err := options.Tags.Validate(); err != nil {
		return nil, err
	}
	replySilence := defaultReplySilence
	if options.ReplySilence != nil {
		replySilence = *options.ReplySilence
	}
	if replySilence < 0 {
		return nil, stack.Wrap(errors.New("agent: the reply silence cannot be negative"))
	}
	replySilenceMax := defaultReplySilenceMax
	if options.ReplySilenceMax != nil {
		replySilenceMax = *options.ReplySilenceMax
	}
	if replySilenceMax < 0 {
		return nil, stack.Wrap(errors.New("agent: the longest hold of a reply cannot be negative"))
	}
	// Without a limit a line that never goes quiet holds a reply for as long as it lasts.
	if replySilence > 0 && replySilenceMax == 0 {
		return nil, stack.Wrap(errors.New("agent: the longest hold of a reply must be longer than zero while the reply silence is on"))
	}
	replySilenceConfident := defaultReplySilenceConfident
	if options.ReplySilenceConfident != nil {
		replySilenceConfident = *options.ReplySilenceConfident
	}
	if replySilenceConfident < 0 {
		return nil, stack.Wrap(errors.New("agent: the reply silence for a confident ending cannot be negative"))
	}
	replyConfidentScore := defaultReplyConfidentScore
	if options.ReplyConfidentScore != nil {
		replyConfidentScore = *options.ReplyConfidentScore
	}
	if math.IsNaN(replyConfidentScore) || replyConfidentScore < 0 || replyConfidentScore > 1 {
		return nil, stack.Wrap(errors.New("agent: the confident score must be between 0 and 1"))
	}
	previewDebounce := defaultPreviewDebounce
	if options.PreviewDebounce != nil {
		previewDebounce = *options.PreviewDebounce
	}
	if previewDebounce < 0 {
		return nil, stack.Wrap(errors.New("agent: the preview debounce cannot be negative"))
	}
	previewQuiet := defaultPreviewQuiet
	if options.PreviewQuiet != nil {
		previewQuiet = *options.PreviewQuiet
	}
	if previewQuiet < 0 {
		return nil, stack.Wrap(errors.New("agent: the preview quiet cannot be negative"))
	}
	// Memories belong to the customer unless the caller named someone more specific, and
	// are always kept under the customer as the app id, so no caller can reach another
	// customer's and a customer's can be deleted without knowing how its callers labelled
	// them. A caller's own app id narrows like any other label.
	scope := memory.Scope{
		AppID:   options.CustomerID,
		UserID:  options.CustomerID,
		AgentID: options.ConfigID,
		RunID:   options.SessionID,
		Extra:   options.MemoryFilter,
	}
	if options.AppID != "" {
		scope.Extra = maps.Clone(scope.Extra)
		if scope.Extra == nil {
			scope.Extra = map[string]string{}
		}
		scope.Extra[appLabel] = options.AppID
	}
	if options.MemoryUserID != "" {
		scope.UserID = options.MemoryUserID
	}
	// A session spelled out rather than started from a config is its own agent.
	if scope.AgentID == "" {
		scope.AgentID = options.AgentID
	}
	if options.Memory != nil {
		if err := scope.Validate(); err != nil {
			return nil, err
		}
	}

	logger := options.Logger.With("customer", options.CustomerID)
	owner := routing.Owner{
		CustomerID: options.CustomerID,
		Caller:     options.Caller,
		AgentID:    options.AgentID,
		CallID:     options.CallID,
		Tags:       options.Tags,
	}

	settling := newCadence(0, 0, 0, logger)
	settling.preview = previewDebounce
	settling.previewQuiet = previewQuiet
	listening := newDuplex(options.Duplex)
	emitter := NewEmitter(eventBuffer)
	agent := &Agent{
		options:          options,
		logger:           logger,
		emitter:          emitter,
		prompt:           options.Instructions,
		listeners:        map[string]*sttrouter.Session{},
		audioHistory:     map[string]*pcm16leRing{},
		eotGates:         map[string]*eotGate{},
		voices:           map[string]string{},
		abandoned:        map[string]struct{}{},
		streams:          map[string]*llm.Stream{},
		previews:         map[string]*replyPreview{},
		kept:             map[string]*keptPreview{},
		previewTurns:     map[string]string{},
		modelCallTimes:   map[string]float64{},
		generatingCancel: map[string]context.CancelFunc{},
		synthesisCtx:     map[string]context.Context{},
		cadence:          settling,
		duplex:           listening,
		replySilence:     replySilence,
		replySilenceMax:  replySilenceMax,

		replySilenceConfident: replySilenceConfident,
		replyConfidentScore:   replyConfidentScore,
		confident:             map[string]struct{}{},
	}
	settling.previewing = agent.previewsEarly
	// Only a call has a caller's audio to tell silence from, and it is listened to for the hold a
	// reply is put on and for the quiet a reply is started on.
	if !options.Text && (replySilence > 0 || (previewQuiet > 0 && previewDebounce > 0 && agent.previewsReplies())) {
		agent.voiced = newVoiceActivity()
		settling.quietFor = func(participantID string) time.Duration {
			return agent.voiced.quietFor(participantID, time.Now())
		}
	}

	// Turns are keyed by agent id and decisions by call id, so an agent missing either is
	// still measured and still reports itself, it is just not kept.
	if options.Store != nil && options.AgentID != "" {
		agent.turnStore = newTurnRecorder(options.Store, owner, logger)
	}
	var record func(Decided)
	if options.Store != nil && options.CallID != "" {
		agent.decisionStore = newDecisionRecorder(options.Store, owner, logger)
		record = agent.decisionStore.Record
	}
	agent.converse = newConverse(settling, listening, emitter, record, 0, logger)
	agent.turns = newTurnTracker(agent.finishTurn)
	agent.nativeMode.Store(native)

	if options.Memory != nil {
		// Memory is recorded as a modality of its own so what remembering costs is
		// reported alongside what the models cost.
		agent.memory = newMemoryWriter(
			options.Memory,
			scope,
			options.Incognito,
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
			knowledge.Scoped(options.CustomerID, options.KnowledgeNamespace),
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
	a.mu.Lock()
	delete(a.modelCallTimes, turn.TurnID)
	a.mu.Unlock()
	a.logger.Info("voice turn timing", "turn", turn.TurnID,
		"stt_ms", turn.STTLatencyMs, "cadence_ms", turn.CadenceMs,
		"decision_ms", turn.DecisionMs, "model_to_first_text_ms", turn.ModelToFirstTextMs,
		"text_to_tts_ms", turn.TextToTTSMs, "tts_to_audio_ms", turn.TTSToAudioMs,
		"transcript_to_audio_ms", turn.RoundtripMs,
		"speech_end_to_audio_ms", turn.SpeechEndToAudioMs,
		"first_frame_queued_ms", turn.FirstFrameQueuedMs,
		"first_audible_frame_ms", turn.FirstAudibleFrameMs,
		"speech_end_to_audible_ms", turn.SpeechEndToAudibleMs,
		"interrupted", turn.Interrupted)
	a.emitter.Send(turn)
	if a.turnStore != nil {
		a.turnStore.Record(turn)
	}
}

func (a *Agent) recordModelCall(timing llm.CallTiming) {
	if timing.Purpose == "reply" && timing.Success {
		// A preview kept across a Wait was asked for under the earlier candidate's id, and
		// counts towards the turn that took it over. The call itself is still reported as the
		// request it was.
		a.mu.Lock()
		turnID := timing.TurnID
		if taken, ok := a.previewTurns[turnID]; ok {
			turnID = taken
		}
		a.mu.Unlock()
		if !a.turns.modelTiming(turnID, timing.TTFTMs) {
			a.mu.Lock()
			if a.previews[turnID] != nil {
				a.modelCallTimes[turnID] = timing.TTFTMs
			}
			a.mu.Unlock()
		}
	}
	a.emitter.Send(ModelCall{CallTiming: timing})
}

// Join opens the model and voice sessions, then joins the call and starts listening.
//
// The order matters: the sessions come first so the agent can answer the moment someone
// speaks, rather than losing the first thing it hears while it connects.
func (a *Agent) Join(ctx context.Context) error {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: already closed"))
	}
	if a.joined {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: already joined"))
	}
	a.ctx, a.cancel = context.WithCancel(ctx)
	a.joined = true
	a.mu.Unlock()

	// Searching is routed before the tools are worked out, because whether the model is
	// offered one depends on whether a provider answered.
	a.startSearching(a.ctx)

	// What earlier conversations established is fetched before the call starts, so the
	// first turn is already answered in the light of it rather than the second. A native
	// model takes it in the instructions its session opens with.
	if a.memory != nil {
		a.recalled = memory.Prompt(a.memory.Recall(a.ctx, a.options.RecallLimit))
	}

	a.mu.Lock()
	settings := a.settingsLocked()
	a.mu.Unlock()
	if a.native() {
		prep, err := a.openNative(settings)
		if err != nil {
			return err
		}
		a.startNative(prep)
	} else {
		prep, err := a.openCascade(settings)
		if err != nil {
			return err
		}
		a.startCascade(prep)
	}

	a.mu.Lock()
	a.lastSpokeAt = time.Now()
	a.mu.Unlock()

	if !a.options.Text {
		if err := a.options.Edge.Join(a.ctx); err != nil {
			return stack.Wrap(fmt.Errorf("agent: join edge: %w", err))
		}
		a.running.Add(1)
		go a.consumeEdge()

		// Only a transport with other people in it can answer this, so it is asked for
		// rather than required. Without it nothing is reported and the agent behaves as
		// it did before, which is right for a loopback with nobody else on it.
		if roster, ok := a.options.Edge.(Roster); ok {
			a.running.Add(1)
			go a.consumeRoster(roster)
		}
	}

	changed := a.modelsChanged()
	a.logger.Info("joined", "llm", changed.LLM, "tts", changed.TTS, "sts", changed.STS)
	a.emitter.Send(Joined{At: time.Now()})
	return nil
}

// SimpleResponse answers a piece of text through the model, as though a participant had
// said it. It returns once the request is on its way: the reply arrives on Events and is
// spoken as it streams.
func (a *Agent) SimpleResponse(ctx context.Context, text string) error {
	_, err := a.RespondTo(ctx, text, nil)
	return err
}

// RespondTo answers a piece of text through the model, attaching images to that turn.
//
// It returns the turn the answer is being given as, which is what lets a caller follow one
// particular reply: every event of it carries the id, and it is what a session records the
// turn under. A native agent returns an empty one, because a speech-to-speech model decides
// for itself what counts as a turn and there is nothing here to name.
func (a *Agent) RespondTo(ctx context.Context, text string, images []llm.ImagePart) (string, error) {
	if a.native() {
		return "", a.respondNative(text, images)
	}
	if len(images) > 0 {
		a.mu.Lock()
		current := a.harness
		a.mu.Unlock()
		if current == nil {
			return "", stack.Wrap(errors.New("agent: not joined"))
		}
		id := replyPrefix + turnStamp()
		parts := llm.TextParts(text)
		for index, image := range images {
			if err := image.Validate(); err != nil {
				return "", err
			}
			image.Data = append([]byte(nil), image.Data...)
			described := map[string]any{"source": "attachment", "frame_id": fmt.Sprintf("%s-image-%d", id, index+1), "received_at_ms": time.Now().UnixMilli()}
			if image.Caption != "" {
				described["caption"] = image.Caption
			}
			metadata, _ := json.Marshal(described)
			parts = append(parts, llm.ContentPart{Text: string(metadata)}, llm.ContentPart{Image: &image})
		}
		if _, err := current.Delegate("vision", text, id, parts, nil); err != nil {
			return "", err
		}
		return id, a.respondTurn(id, stt.Participant{ID: "caller"}, text, heard{at: time.Now()}, "Visual analysis has been requested. Wait for its findings before answering the visual question.", nil)
	}
	id := replyPrefix + turnStamp()
	return id, a.respondTurn(id, stt.Participant{ID: "caller"}, text, heard{at: time.Now()}, "", nil)
}

// VideoFramesTool is the caller's tool the agent reads frames of the user's video through.
const VideoFramesTool = "get_video_frames"

func (a *Agent) captureVideo(ctx context.Context, request harness.CaptureRequest) ([]llm.ContentPart, error) {
	if a.options.ToolRunner == nil {
		return nil, errors.New("agent: no video source is connected")
	}
	source, frames := request.Source, request.Frames
	if source == "" {
		source = a.options.VideoSource
	}
	if frames == 0 {
		frames = a.options.VideoMaxFrames
	}
	if frames == 0 {
		frames = 1
	}
	arguments, err := json.Marshal(map[string]any{"source": source, "at_ms": request.At.UnixMilli(), "limit": frames})
	if err != nil {
		return nil, err
	}
	return a.options.ToolRunner.Run(ctx, llm.ToolCall{ID: request.TaskID + "-capture", Name: VideoFramesTool, Arguments: string(arguments)})
}

// Ask answers a piece of text in writing and says none of it.
//
// It is how a message written to the agent is answered while a call is going on. Speaking
// the answer would interrupt whoever is on the phone with a reply to something they never
// said, and the person who wrote it is not listening to the call anyway.
//
// The reply is drained here rather than pumped into the speaking goroutine, which is the
// whole of what keeps it quiet: only a reply the agent pumps reaches the voice. Nothing
// about the turn in progress is touched, so a caller mid-sentence is not interrupted and an
// interruption has nothing new to abandon.
//
// The exchange is kept, so what was asked in writing can be referred to out loud. No tools
// are offered: a written aside must not press a keypad or transfer a call that the person
// writing cannot see.
func (a *Agent) Ask(ctx context.Context, text string) (string, error) {
	if strings.TrimSpace(text) == "" {
		return "", errors.New("agent: there is nothing to answer")
	}
	// A native agent has no text model to ask on the side: everything it says, it says
	// out loud, on the call.
	if a.native() {
		return "", errors.New("agent: a speech-to-speech agent answers only on the call")
	}

	a.mu.Lock()
	if a.closed || a.llm == nil {
		a.mu.Unlock()
		return "", errors.New("agent: not joined")
	}
	a.history = append(a.history, llm.Message{Role: llm.User, Content: text})
	history := a.replayLocked()
	instructions := a.instructions()
	model, overwrites := a.llm, a.options.Overwrites
	a.mu.Unlock()

	turnID := writtenPrefix + turnStamp()
	screening := a.screening(ctx, turnID, text)
	if screening != nil && screening.blocking {
		if verdict := screening.verdict(); !verdict.Allowed {
			return a.refuseWritten(turnID, verdict), nil
		}
		screening = nil
	}

	stream, err := model.Create(ctx, llm.ResponseParams{
		ID:              turnID,
		Instructions:    instructions,
		Input:           history,
		MaxOutputTokens: a.options.MaxTokens,
		PromptCacheKey:  a.options.ConfigID,
	}.Overwrite(overwrites))
	if err != nil {
		return "", err
	}
	response, err := llm.Collect(stream)
	if err != nil {
		return "", err
	}

	// The reply is complete and has gone nowhere: this is a function that returns a
	// string, so holding it until the verdict is in costs the reader nothing but the wait,
	// and what was written is discarded unread if the policy refuses it.
	if screening != nil {
		waited := time.Now()
		if verdict := screening.verdict(); !verdict.Allowed {
			return a.refuseWritten(turnID, verdict), nil
		}
		a.logger.Debug("a written answer waited on the guardrail",
			"turn", turnID, "held_ms", routing.MsSince(waited))
	}

	a.mu.Lock()
	a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: response.OutputText})
	a.mu.Unlock()
	return response.OutputText, nil
}

// refuseWritten is refuse for an aside in writing, which has no turn on the call to put a
// reply through and hands the refusal back to whoever asked instead.
func (a *Agent) refuseWritten(turnID string, verdict guardrail.Verdict) string {
	refusal := verdict.Refusal
	if refusal == "" {
		refusal = a.options.Guardrail.Policy().Refusal
	}

	a.mu.Lock()
	a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: refusal})
	a.mu.Unlock()

	a.logger.Info("a written turn was refused by the guardrail",
		"turn", turnID, "reason", verdict.Reason, "probability", verdict.Probability)
	a.emitter.Send(Blocked{
		TurnID:      turnID,
		Reason:      verdict.Reason,
		Probability: verdict.Probability,
	})
	return refusal
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
	// A native model has no way to say exact words: it says what it makes of a prompt.
	// Pretending otherwise would promise a caller a script and deliver a paraphrase.
	if a.native() {
		return stack.Wrap(errors.New("agent: a speech-to-speech agent cannot say exact words; use Prompt"))
	}

	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: not joined"))
	}
	a.speakingTurn = turnID
	a.saying = text
	a.mu.Unlock()

	return a.speakWhole(turnID, text)
}

// Interrupt abandons the reply being spoken, the way a participant talking over the agent
// would. It is what a caller outside the call has instead of a voice.
func (a *Agent) Interrupt() {
	a.mu.Lock()
	participant := a.lastParticipant
	turnID := a.speakingTurn
	if turnID == "" && a.interruptDone == nil {
		// Text-only work has no speaking turn to name, but an explicit Interrupt still
		// means the caller is giving up on any tool work and speech left in the edge.
		a.cancelPlayoutLocked()
		a.utterances = 0
		a.saying = ""
		a.toolReply = false
		a.dropSpeech()
		for _, cancel := range a.toolCancels {
			cancel()
		}
	}
	a.mu.Unlock()
	if turnID == "" {
		return
	}
	if stopped, ok := a.stopPlayback(participant, turnID, time.Time{}, "manual", "api"); ok {
		a.abandon(turnID)
		a.finishInterrupt(stopped)
	}
}

// SetInstructions changes what the agent is told to be from the next turn on. The reply
// being spoken keeps the prompt it was started with, because rewriting it mid-sentence
// would have the agent change character in the middle of a thought.
func (a *Agent) SetInstructions(text string) {
	a.mu.Lock()
	a.prompt = text
	instructions := a.instructions()
	if a.native() {
		instructions = a.nativeInstructions(a.harness != nil)
	}
	model := a.sts
	a.mu.Unlock()

	// A native model holds the prompt itself, so it is told. One that took its
	// instructions only when the session opened refuses, and the refusal is reported
	// rather than swallowed: a caller who changed the prompt and heard nothing of it would
	// believe the agent had changed.
	if model != nil {
		if err := model.SetInstructions(instructions); err != nil {
			a.fail(err, "sts")
		}
	}
}

// Events carries what happened in the conversation. It is closed by Close.
func (a *Agent) Events() <-chan Event { return a.emitter.Events() }

// LLM exposes the model session, so a caller can reach the provider's own features or the
// price the conversation is billed at.
func (a *Agent) LLM() *llmrouter.Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.llm
}

// TTS exposes the voice session.
func (a *Agent) TTS() *ttsrouter.Session { return a.voice() }

// STS exposes the speech-to-speech session, which a native agent has in place of the
// three above. Nil on a cascade.
func (a *Agent) STS() *stsrouter.Session { return a.speech() }

// STT is one of the live transcriptions, if anybody has been heard yet.
func (a *Agent) STT() *sttrouter.Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	for _, session := range a.listeners {
		return session
	}
	return nil
}

// Subagent is the slower model delegated work runs on.
func (a *Agent) Subagent() *llmrouter.Session {
	a.mu.Lock()
	current := a.harness
	a.mu.Unlock()
	if current == nil {
		return nil
	}
	return current.Subagent()
}

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
		if !a.Busy() && !a.speechPending() {
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
	a.cancelPlayoutAndForgetLocked()
	a.dropSpeech()
	if a.interruptDone != nil {
		close(a.interruptDone)
		a.interruptDone = nil
	}
	a.mu.Unlock()
	// A swap waiting for its turn boundary gives up once closed is set, and one already
	// changing the pipeline finishes first, so the pipeline released below is whole.
	a.swapping.Lock()
	defer a.swapping.Unlock()

	a.cadence.Close()
	if cancel != nil {
		cancel()
	}
	a.cancelPreviews()

	// The edge leaves first: it is the source of the audio that keeps the rest busy.
	var failures []error
	if a.options.Edge != nil {
		if err := a.options.Edge.Leave(); err != nil {
			failures = append(failures, fmt.Errorf("leave edge: %w", err))
		}
	}
	p, released := a.releasePipeline(false)
	failures = append(failures, released...)

	// Left is sent before the emitter closes, and the emitter closes before the consumers
	// are waited on: a consumer blocked emitting to a caller that has stopped reading has
	// to be let go of, or shutdown would depend on someone draining the channel.
	a.emitter.Send(Left{At: time.Now()})
	a.emitter.Close()
	if p != nil {
		p.running.Wait()
	}
	a.running.Wait()

	// The writers are drained after the consumers have stopped, so a turn that finished
	// on the way out is still recorded and still remembered.
	if a.turnStore != nil {
		a.turnStore.Close()
	}
	if a.decisionStore != nil {
		a.decisionStore.Close()
	}
	if a.memory != nil {
		a.memory.Close()
	}
	if a.knowledge != nil {
		a.knowledge.Close()
	}
	if a.searcher != nil {
		failures = append(failures, a.searcher.Close())
	}

	return errors.Join(failures...)
}

// consumeEdge feeds each participant's audio to their own transcription session.
func (a *Agent) consumeEdge() {
	defer a.running.Done()

	for inbound := range a.options.Edge.Audio() {
		if a.switching.Load() {
			continue
		}
		// A native model hears one stream rather than one per participant.
		if a.native() {
			a.hear(inbound)
			continue
		}
		a.retainEOTAudioTimed(inbound.Participant.ID, inbound.Audio.SampleRate, inbound.Audio.Channels,
			inbound.Audio.Samples, inbound.Timing)
		if a.voiced != nil {
			a.voiced.observe(inbound.Participant.ID, inbound.Audio, time.Now())
		}
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

// consumeRoster turns who comes and goes into events a watcher can wait on.
func (a *Agent) consumeRoster(roster Roster) {
	defer a.running.Done()

	for attendance := range roster.Attendance() {
		if attendance.Joined {
			a.logger.Debug("a participant joined",
				"participant", attendance.Participant.UserID)
			a.emitter.Send(ParticipantJoined{
				Participant: attendance.Participant,
				At:          time.Now(),
			})
			continue
		}
		a.unbind(attendance.Participant)
		a.logger.Debug("a participant left", "participant", attendance.Participant.UserID)
		a.emitter.Send(ParticipantLeft{
			Participant: attendance.Participant,
			At:          time.Now(),
		})
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
	target := a.options.STTTarget
	a.mu.Unlock()

	session, err := a.startListener(target)
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
	go a.consumeSTT(participant.ID, session)
	return session, nil
}

// dropListener retires a transcription session that cannot carry on, so the next audio
// from that participant opens a new one.
//
// A transcriber does not only stop when the call does: a provider will cut a session that
// has been idle, and the socket is not usable afterwards. Left in the map, that session is
// handed every later chunk and the agent is deaf to the participant for the rest of the
// call rather than for a moment.
func (a *Agent) dropListener(participantID string, session *sttrouter.Session) {
	a.mu.Lock()
	// Only when it is still the one in use: a replacement may already have been opened,
	// and retiring that would put us back where we started.
	if a.listeners[participantID] == session {
		delete(a.listeners, participantID)
	}
	// The voice, though, goes either way: a diarised label belongs to the session that
	// made it up, so whichever session serves this participant next has to be listened to
	// afresh rather than held to names it never chose.
	delete(a.voices, participantID)
	a.mu.Unlock()

	// Closing releases the provider's socket and ends the stream of events this was
	// reading. Doing it twice is safe, which is what lets Close race with this.
	if err := session.Close(); err != nil {
		a.logger.Debug("closing a dropped transcriber failed", "error", err)
	}
}

// consumeSTT feeds transcript revisions to the cadence controller.
func (a *Agent) consumeSTT(participantID string, session *sttrouter.Session) {
	defer a.running.Done()
	defer a.dropListener(participantID, session)

	for event := range session.Events() {
		switch typed := event.(type) {
		case stt.Transcript:
			receivedAt := time.Now()
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
			superseded, saying := a.cadence.Observe(typed)
			if saying != "" {
				// Different words are not the ones a preview kept for a Wait was started on.
				a.dropKeptPreview(typed.Participant.ID)
			}
			state := a.floor()
			var stopped interruption
			stopNow := false
			fastTranscript := typed
			fastTranscript.Text = saying
			if saying != "" && a.primaryPartialInterrupt(fastTranscript, state) {
				if current, ok := a.stopPlayback(typed.Participant, state.Speaking, receivedAt,
					"primary_partial", "transcript"); ok {
					stopped, stopNow = current, true
					state = a.floor()
					// Cadence already owns this accepted revision before provider cleanup
					// can release queued work or another turn.
				}
			}
			actions := a.converse.observeRevision(typed, state, superseded, saying)
			if stopNow {
				a.finishInterruptedTurn(stopped)
			}
			a.act(actions)
			a.expeditePrimaryEOTFinal(typed)

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
			a.logger.Warn("the transcriber dropped, a new one opens on the next audio",
				"provider", typed.Provider, "model", typed.Model, "reason", typed.Reason)
			return

		case stt.Error:
			a.fail(typed.Err, "stt")
			// A fatal error has already ended the session upstream. Reading on would be
			// waiting for words from a socket that is gone.
			if typed.Fatal {
				return
			}
		}
	}
}

func (a *Agent) expeditePrimaryEOTFinal(transcript stt.Transcript) {
	if !transcript.Final() || a.options.EOT == nil || a.options.EOTMode != EOTModePrimary ||
		a.options.Text || (a.options.Guardrail != nil &&
		a.options.Guardrail.Policy().Mode == guardrail.ModeBlocking) ||
		!a.hasEOTAudio(transcript.Participant.ID) {
		return
	}

	current, ok := a.cadence.currentCandidate(transcript.Participant.ID)
	if !ok || !sameWords(current.Text, transcript.Text) {
		return
	}
	quiet := func() bool {
		a.mu.Lock()
		defer a.mu.Unlock()
		if a.closed || a.switching.Load() || a.harness == nil || a.pipe == nil || a.pipe.native ||
			a.generating || a.utterances > 0 || a.pendingTools > 0 {
			return false
		}
		return !a.anotherVoiceLocked(current)
	}
	if !quiet() || a.speechPending() || !quiet() {
		return
	}
	a.cadence.ExpediteFinal(transcript)
}

// consumeCadence puts a turn to the conversation once its words have held still.
func (a *Agent) consumeCadence(p *pipeline) {
	defer p.running.Done()

	for {
		select {
		case ready := <-a.cadence.Ready():
			a.mu.Lock()
			gone, switching := a.closed || a.harness == nil, a.switching.Load()
			a.mu.Unlock()
			if gone {
				continue
			}
			if switching {
				// Model swaps cancel a decision that already left the cadence timer.
				// Put its words back on the normal retry timer instead of asking the old
				// controller while the pipeline is being replaced.
				a.converse.Unasked(ready.ID)
				continue
			}
			a.act([]Action{a.converse.Settled(ready, a.floor())})
		case early := <-a.cadence.Previews():
			a.previewEarly(early)
		case <-p.ctx.Done():
			return
		}
	}
}

// consumePresence keeps long listening or thinking gaps from sounding like a dead call.
func (a *Agent) consumePresence(p *pipeline) {
	defer p.running.Done()

	ticker := time.NewTicker(presenceTick)
	defer ticker.Stop()
	for {
		select {
		case <-ticker.C:
			a.act(a.converse.Tick(a.floor()))
			// Speech still draining out of the edge has no event when it finishes, so a
			// note that waited for it is retried here rather than never spoken. A turn
			// queued behind a hung create is the same: interrupt made the floor quiet
			// with no TTS complete to pick it up.
			a.respondQueued()
			a.followUp()
		case <-p.ctx.Done():
			return
		}
	}
}

// floor is what the agent is doing right now, which is what the judgements that depend on
// whether it is mid-sentence are made against.
func (a *Agent) floor() floor {
	pending := a.speechPending()
	a.mu.Lock()
	active := a.utterances > 0 || a.generating || a.pendingTools > 0 || pending || a.interruptDone != nil
	state := floor{
		Quiet:           !active,
		Speaking:        a.speakingTurn,
		Reply:           "",
		LastSpokeAt:     a.lastSpokeAt,
		LastHeardAt:     a.lastHeardAt,
		LastParticipant: a.lastParticipant,
	}
	if active {
		state.Reply = a.saying
		if state.Reply == "" {
			state.Reply = lastAssistantSaid(a.history)
		}
	}
	current := a.harness
	a.mu.Unlock()

	state.Delegating = current != nil && current.Delegating()
	return state
}

// act carries out what the conversation decided, in the order it decided it.
func (a *Agent) act(actions []Action) {
	for _, action := range actions {
		a.perform(action)
	}
}

// rule carries out what the conversation makes of a ruling, then lets go of the reply
// previewed for its words if nothing took it. An answer adopts the preview, and a Wait
// keeps it for the next check of the same words, so one still held afterwards belongs to
// words that were ignored, held back, found stale or otherwise not answered.
func (a *Agent) rule(ruling harness.Decided) {
	a.act(a.converse.Ruled(ruling, a.floor()))
	a.releasePreview(ruling.CandidateID)
	a.mu.Lock()
	delete(a.confident, ruling.CandidateID)
	a.mu.Unlock()
}

// rulePrimaryEOTLow is rule for the Wait an acoustic score below its threshold makes, which
// the same words are put to again sooner than they are after the flow controller's.
func (a *Agent) rulePrimaryEOTLow(ruling harness.Decided) {
	a.act(a.converse.ruledPrimaryEOTLow(ruling, a.floor()))
	a.releasePreview(ruling.CandidateID)
}

// perform carries out one decision. Every branch here is mechanical: which provider to
// touch and in what order. Why any of it is happening was settled in converse.
func (a *Agent) perform(action Action) {
	switch {
	case action.Kind == ActSupersede:
		a.cancelPreview(action.TurnID)
		a.cancelEOTGate(action.TurnID)
	case action.Kind == ActWait:
		a.keepPreview(action.Candidate)
	case action.Kind != ActAsk && action.Kind != ActAnswer:
		a.cancelPreview(action.Candidate.ID)
		a.dropKeptPreview(action.Candidate.Participant.ID)
	}
	switch action.Kind {
	case ActBackchannel:
		a.backchannel(action.Participant, action.Text)

	case ActCheckIn:
		a.checkIn(action.Participant, action.Text)

	case ActSupersede:
		if err := a.harness.CancelDecision(action.TurnID); err != nil {
			a.fail(err, "flow")
		}

	case ActAsk:
		a.ask(action.Candidate)

	case ActInterrupt:
		a.dropKeptPreviews()
		if stopped, ok := a.stopPlayback(action.Participant, action.TurnID, time.Time{},
			"semantic", "transcript"); ok {
			a.finishInterruptedTurn(stopped)
		}

	case ActShorten:
		a.abandon(action.TurnID)
		a.shorten()

	case ActQueue:
		a.abandon(action.Supersede)

	case ActAnswer:
		a.abandon(action.Supersede)
		if err := a.respondCandidate(action.Candidate, action.Clarify); err != nil {
			a.fail(err, "llm")
		}

	case ActFail:
		a.fail(action.Err, "flow")
	}
}

// ask puts a settled turn to the flow controller, on its own model session so deciding
// never competes with the reply being streamed to the voice.
func (a *Agent) ask(ready candidate) {
	a.mu.Lock()
	current := a.harness
	p := a.pipe
	if a.closed || current == nil {
		a.mu.Unlock()
		return
	}
	history := llm.OmitImages(append([]llm.Message(nil), a.history...))
	instructions := a.instructions()
	// Pending tools still own the turn even after its spoken acknowledgement ends.
	// The flow controller must be able to stop that work on a caller's correction.
	speaking := a.generating || a.utterances > 0 || a.pendingTools > 0
	reply := a.saying
	if reply == "" {
		reply = lastAssistantSaid(history)
	}
	anotherVoice := a.anotherVoiceLocked(ready)
	a.mu.Unlock()

	// Speech the voice has finished sending is still on its way out of the edge, so the
	// agent counts as speaking until it has drained.
	speaking = speaking || a.speechPending()
	eligible := a.previewEligible(ready, speaking, anotherVoice)
	// The acoustic score ruled these words unfinished and is being asked again. Nothing heard
	// since means it would score the same window, so it is not asked, and nothing is copied or
	// previewed for it. The preview of the first check is the one the answer will use.
	retrying := false
	if eligible && a.options.EOT != nil && a.options.EOTMode == EOTModePrimary {
		if deadline, again := a.converse.primaryRetry(ready); again {
			if a.eotAudioUnchanged(ready.Participant.ID) {
				a.waitForFreshAudio(ready, deadline)
				return
			}
			retrying = true
		}
	}
	previewing := eligible && a.previewsReplies()
	// The reply started for these same words before a Wait is taken over rather than asked
	// for again, so there is never more than one preview for a participant.
	if !a.takeKeptPreview(ready, previewing) && previewing && !retrying {
		a.preview(ready, current, instructions)
	}

	turn := harness.FlowTurn{
		ID:           ready.ID,
		Instructions: instructions,
		History:      history,
		Participant:  participantName(ready.Participant),
		Text:         ready.Text,
		Speaking:     speaking,
		Reply:        reply,
		Unfinished:   ready.Unfinished,
		AnotherVoice: anotherVoice,
	}
	var snapshot eotScoringSnapshot
	if eligible && a.options.EOT != nil {
		snapshot, _ = a.eotScoringSnapshot(ready.Participant.ID)
	}
	a.decideWithEOT(p, current, ready, turn, snapshot)
}

// previewEligible reports whether a reply may be started for a candidate's words before they
// are ruled on: the agent is not speaking, the words are a settled turn rather than a
// provisional one, they are the caller's own, there is a voice to speak the reply, and no
// policy has to clear them first.
func (a *Agent) previewEligible(ready candidate, speaking, anotherVoice bool) bool {
	return !speaking && !ready.Unfinished && !anotherVoice && !a.options.Text &&
		(a.options.Guardrail == nil || a.options.Guardrail.Policy().Mode != guardrail.ModeBlocking)
}

type previewResult struct {
	stream *llm.Stream
	err    error
}

type replyPreview struct {
	model     *llmrouter.Session
	turn      harness.Turn
	ready     chan previewResult
	events    chan llm.Event
	cancel    context.CancelFunc
	startedAt time.Time
	// kept says a Wait is holding the preview for the next check of the same words. It is
	// read and set under the agent's lock.
	kept bool
}

// previewsReplies reports whether a reply is started before the flow controller has ruled on
// its words. It is on unless the options turn it off.
func (a *Agent) previewsReplies() bool {
	return a.options.SpeculativeReplies == nil || *a.options.SpeculativeReplies
}

// previewsEarly reports whether a reply can be started for a caller's words ahead of the wait
// that decides whether they have finished: replies are previewed, there is a voice to speak
// them and a model that writes them, and no policy has to clear the words first. The cadence
// asks before it arms the debounce for them.
func (a *Agent) previewsEarly() bool {
	return a.previewsReplies() && !a.options.Text && !a.native() &&
		(a.options.Guardrail == nil || a.options.Guardrail.Policy().Mode != guardrail.ModeBlocking)
}

func (a *Agent) preview(ready candidate, current *harness.Harness, instructions string) {
	model := current.PreviewModel()
	if model == nil {
		return
	}
	a.mu.Lock()
	if a.closed || a.harness != current || a.switching.Load() {
		a.mu.Unlock()
		return
	}
	history := append(a.replayLocked(), a.userTurnLocked(ready.Text, nil))
	ctx, cancel := context.WithCancel(a.ctx)
	turn := harness.Turn{ID: ready.ID, Instructions: instructions, History: history,
		Note: joinNotes(a.pendingInterruptionNoteLocked(), a.duplex.Note(ready.Confidence))}
	p := &replyPreview{model: model, turn: turn, ready: make(chan previewResult, 1),
		events: make(chan llm.Event, replyBuffer), cancel: cancel,
		startedAt: time.Now()}
	a.previews[ready.ID] = p
	a.running.Add(1)
	a.mu.Unlock()

	go func() {
		defer a.running.Done()
		defer close(p.events)
		stream, err := current.Preview(ctx, turn)
		p.ready <- previewResult{stream: stream, err: err}
		if err != nil || stream == nil {
			return
		}
		stop := context.AfterFunc(ctx, func() { _ = stream.Close() })
		defer stop()
		defer stream.Close()
		for stream.Next() {
			select {
			case p.events <- stream.Current():
			case <-ctx.Done():
				return
			}
		}
	}()
}

func (a *Agent) cancelPreview(turnID string) {
	if turnID == "" {
		return
	}
	a.mu.Lock()
	p := a.previews[turnID]
	delete(a.previews, turnID)
	delete(a.modelCallTimes, turnID)
	if p != nil {
		a.forgetPreviewLocked(turnID, p)
	}
	a.mu.Unlock()
	if p == nil {
		return
	}
	p.cancel()
}

// forgetPreviewLocked drops what else is held about a preview that has left the map: its
// place among the kept ones and the turn it was taken over by. The caller holds the lock.
func (a *Agent) forgetPreviewLocked(turnID string, p *replyPreview) {
	if p.kept {
		for participantID, kept := range a.kept {
			if kept.key == turnID {
				kept.expires.Stop()
				delete(a.kept, participantID)
			}
		}
	}
	delete(a.previewTurns, p.turn.ID)
}

// releasePreview lets go of the preview for a candidate nothing adopted, unless a Wait is
// keeping it for the next check of the same words.
func (a *Agent) releasePreview(candidateID string) {
	a.mu.Lock()
	p := a.previews[candidateID]
	kept := p != nil && p.kept
	a.mu.Unlock()
	if !kept {
		a.cancelPreview(candidateID)
	}
}

// keptPreview is a reply preview held across a Wait.
type keptPreview struct {
	// key is the candidate the preview was started for, which it is held under.
	key string
	// revision is the words it was started on, which is what the next check has to carry for
	// the preview to be taken over: candidate ids change when the same words are put again.
	revision uint64
	// expires lets go of it when the patience for those words runs out.
	expires *time.Timer
}

// keepPreview holds the reply previewed for a candidate the flow controller asked to wait
// on, so that the next check of the same words takes it over instead of asking the model
// for the same reply again. It is let go if the words change, or the floor does, and when
// the patience for them runs out, which is when the same words are answered with a question
// the preview was not written for.
func (a *Agent) keepPreview(ready candidate) {
	a.mu.Lock()
	p := a.previews[ready.ID]
	if p == nil {
		a.mu.Unlock()
		return
	}
	// The words may have moved on since the ruling that asked to wait. Both are read under
	// the lock a revision takes to let go of a kept preview, so one that lands after the
	// check still finds this preview to drop.
	until, waiting := a.converse.patienceEnds(ready.Participant.ID)
	current, heard := a.cadence.currentCandidate(ready.Participant.ID)
	if !waiting || !heard || ready.Revision == 0 || current.Revision != ready.Revision ||
		a.closed || a.switching.Load() {
		a.mu.Unlock()
		a.cancelPreview(ready.ID)
		return
	}
	previous := a.holdPreviewLocked(p, ready, until)
	a.mu.Unlock()

	if previous != nil && previous.key != ready.ID {
		a.cancelPreview(previous.key)
	}
}

// holdPreviewLocked keeps a preview for the next check of the same words until the given time,
// and returns the preview it replaced, which the caller lets go of once it has released the
// lock. The caller holds the lock.
func (a *Agent) holdPreviewLocked(p *replyPreview, ready candidate, until time.Time) *keptPreview {
	kept := &keptPreview{key: ready.ID, revision: ready.Revision}
	kept.expires = time.AfterFunc(time.Until(until), func() { a.expireKeptPreview(ready.Participant.ID, kept) })
	previous := a.kept[ready.Participant.ID]
	if previous != nil {
		previous.expires.Stop()
	}
	a.kept[ready.Participant.ID] = kept
	p.kept = true
	return previous
}

// previewEarly starts the reply to words that have held still for the preview debounce, ahead
// of the wait that decides whether the caller has finished, so that the model has been working
// for part of that wait when the candidate for the same words arrives. It is kept the way a
// preview is kept across a Wait: the candidate takes it over through the same adoption, which
// still requires the conversation to be identical, and whatever lets go of a kept preview lets
// go of this one. There is never more than one for a participant, because new words let go of
// the one for the old.
func (a *Agent) previewEarly(early candidate) {
	if !a.previewsReplies() || !a.cadence.previewable(early) {
		return
	}
	a.mu.Lock()
	current := a.harness
	if a.closed || current == nil || a.pipe == nil || a.pipe.native || a.switching.Load() {
		a.mu.Unlock()
		return
	}
	speaking := a.generating || a.utterances > 0 || a.pendingTools > 0
	anotherVoice := a.anotherVoiceLocked(early)
	instructions := a.instructions()
	a.mu.Unlock()

	if !a.previewEligible(early, speaking || a.speechPending(), anotherVoice) {
		return
	}
	a.preview(early, current, instructions)
	a.keepEarlyPreview(early)
}

// keepEarlyPreview holds the reply just started for words, unless they changed while it was
// being started, for the candidate that puts them. If no candidate does, because nothing came
// of the words, it is let go after as long as the patience for words that are waited on.
func (a *Agent) keepEarlyPreview(early candidate) {
	a.mu.Lock()
	p := a.previews[early.ID]
	if p == nil {
		a.mu.Unlock()
		return
	}
	// Read under the lock a revision takes to let go of a kept preview, so a revision that
	// lands after this check still finds the preview to drop.
	current, heard := a.cadence.currentCandidate(early.Participant.ID)
	if !heard || current.Revision != early.Revision || a.closed || a.switching.Load() {
		a.mu.Unlock()
		a.cancelPreview(early.ID)
		return
	}
	previous := a.holdPreviewLocked(p, early, time.Now().Add(a.converse.patienceSpan()))
	a.mu.Unlock()

	if previous != nil && previous.key != early.ID {
		a.cancelPreview(previous.key)
	}
}

// takeKeptPreview hands the preview kept for a participant's words to the candidate that
// puts them again, reporting whether it did. A preview the candidate cannot use, because
// the words are not the same or the floor is no longer the agent's to take, is let go.
func (a *Agent) takeKeptPreview(ready candidate, usable bool) bool {
	a.mu.Lock()
	kept := a.kept[ready.Participant.ID]
	if kept == nil {
		a.mu.Unlock()
		return false
	}
	p := a.previews[kept.key]
	if !usable || p == nil || ready.Revision == 0 || kept.revision != ready.Revision || a.closed {
		a.mu.Unlock()
		a.cancelPreview(kept.key)
		return false
	}

	kept.expires.Stop()
	delete(a.kept, ready.Participant.ID)
	p.kept = false
	delete(a.previews, kept.key)
	a.previews[ready.ID] = p
	if ttft, ok := a.modelCallTimes[kept.key]; ok {
		delete(a.modelCallTimes, kept.key)
		a.modelCallTimes[ready.ID] = ttft
	}
	// A reply taken over by a candidate of the same id has nothing to be told apart from.
	if p.turn.ID != ready.ID {
		a.previewTurns[p.turn.ID] = ready.ID
	}
	a.mu.Unlock()
	return true
}

// expireKeptPreview lets go of a kept preview once the patience for its words has run out,
// unless it was taken over or let go of already.
func (a *Agent) expireKeptPreview(participantID string, kept *keptPreview) {
	a.mu.Lock()
	held := a.kept[participantID] == kept
	a.mu.Unlock()
	if held {
		a.cancelPreview(kept.key)
	}
}

// dropKeptPreview lets go of the preview kept for a participant's words, which are no longer
// the words being waited on, or are not going to be answered.
func (a *Agent) dropKeptPreview(participantID string) {
	if participantID == "" {
		return
	}
	a.mu.Lock()
	kept := a.kept[participantID]
	a.mu.Unlock()
	if kept != nil {
		a.cancelPreview(kept.key)
	}
}

// dropKeptPreviews lets go of every kept preview, for a floor that has changed under all of
// them.
func (a *Agent) dropKeptPreviews() {
	a.mu.Lock()
	keys := make([]string, 0, len(a.kept))
	for _, kept := range a.kept {
		keys = append(keys, kept.key)
	}
	a.mu.Unlock()
	for _, key := range keys {
		a.cancelPreview(key)
	}
}

// cancelPreviews cancels every reply preview still waiting on a ruling. A preview that is
// not adopted belongs to words nobody is answering, so whatever releases the pipeline or
// ends the call lets go of them.
func (a *Agent) cancelPreviews() {
	a.mu.Lock()
	previews := make([]string, 0, len(a.previews))
	for turnID := range a.previews {
		previews = append(previews, turnID)
	}
	a.mu.Unlock()
	for _, turnID := range previews {
		a.cancelPreview(turnID)
	}
}

// anotherVoiceLocked reports whether a turn came from somebody other than the person whose
// track it arrived on.
//
// The first voice heard on a track is taken to be the caller's, since they are the one who
// joined. A turn in a different voice is somebody in the room with them, whom the flow
// controller has no reason to answer. Transcribers that cannot tell voices apart say
// nothing here, and a turn nobody was named in is the caller's as far as anyone can tell.
//
// The caller holds the lock.
func (a *Agent) anotherVoiceLocked(ready candidate) bool {
	if ready.Speaker == "" {
		return false
	}
	known, heard := a.voices[ready.Participant.ID]
	if !heard {
		a.voices[ready.Participant.ID] = ready.Speaker
		return false
	}
	return known != ready.Speaker
}

// abandon drops the delegated work a turn owns, because whatever asked for it is no
// longer what the caller is waiting on. An empty id is nothing to abandon.
func (a *Agent) abandon(turnID string) {
	if turnID == "" {
		return
	}
	a.mu.Lock()
	current := a.harness
	a.mu.Unlock()
	if current != nil {
		current.CancelTurn(turnID, harness.ReasonSuperseded)
	}
}

// heard is when a turn became answerable, what the transcriber spent settling it, and how
// sure it was of the words.
type heard struct {
	at           time.Time
	revisedAt    time.Time
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

func lastAssistantSaid(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == llm.Assistant && strings.TrimSpace(history[i].Content) != "" {
			return history[i].Content
		}
	}
	return ""
}

// turnStamp names a turn. The clock is enough: a conversation cannot produce two turns
// in the same nanosecond.
func turnStamp() string { return strconv.FormatInt(time.Now().UnixNano(), 10) }

// respond asks the harness to reply to a turn.
func (a *Agent) respond(participant stt.Participant, text string, listened heard, images []llm.ImagePart) error {
	return a.respondTurn(replyPrefix+turnStamp(), participant, text, listened, "", images)
}

// respondCandidate answers a settled turn. The note is what the conversation decided the
// model should know beyond the words, and is empty on a turn that is simply answered.
func (a *Agent) respondCandidate(ready candidate, note string) error {
	if note != "" {
		a.cancelPreview(ready.ID)
	}
	if a.voiced != nil && a.replySilence > 0 {
		// What this reply says is for somebody who has just finished talking, so its first
		// audio waits for them to have been quiet long enough to have meant it.
		a.mu.Lock()
		_, confident := a.confident[ready.ID]
		a.gated = heldReply{turn: ready.ID, participant: ready.Participant, confident: confident}
		a.mu.Unlock()
	}
	return a.respondTurn(ready.ID, ready.Participant, ready.Text, heard{
		at:           ready.ReadyAt,
		revisedAt:    ready.RevisedAt,
		sttLatencyMs: ready.STTLatencyMs,
		confidence:   ready.Confidence,
	}, note, nil)
}

// noteToolDone records that one of the tools the current turn asked for has returned.
func (a *Agent) noteToolDone() {
	a.mu.Lock()
	if a.pendingTools > 0 {
		a.pendingTools--
	}
	a.mu.Unlock()
}

// queueToolReply starts a turn that says what the tools came back with, or marks one as
// owed if a generate is already in flight or more tools from this turn are still running.
// Two tools in one reply used to each start a generate, and the second stole speakingTurn
// so the first result was never said.
func (a *Agent) queueToolReply() {
	a.mu.Lock()
	a.toolReply = true
	wait := a.pendingTools > 0 || a.generating
	a.mu.Unlock()
	// A caller who is talking holds the floor, so the tool result waits for follow to pick
	// it up rather than being said over them.
	if wait || a.converse.Listening() {
		return
	}
	if err := a.respondAfterTool(toolPrefix + turnStamp()); err != nil {
		a.fail(err, "llm")
	}
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
	history := a.replayLocked()
	participant := a.lastParticipant
	a.speakingTurn = turnID
	a.generating = true
	a.toolReply = false
	continues := a.owedTurn
	a.owedTurn = ""
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.begin(turnID, participant, time.Now(), time.Time{}, 0)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Continues: continues})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		AfterTool:    true,
	}, "")
}

func (a *Agent) respondTurn(
	turnID string,
	participant stt.Participant,
	text string,
	listened heard,
	note string,
	images []llm.ImagePart,
) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: not joined"))
	}
	// Only an actual caller response consumes this note. A speculative preview sees the
	// same context, but cannot spend it before the settled turn is accepted.
	interruptionNote := a.pendingInterruptionNoteLocked()
	if interruptionNote != "" {
		a.interruptedReplyPending = false
	}
	a.history = append(a.history, a.userTurnLocked(text, images))
	history := a.replayLocked()

	a.speakingTurn = turnID
	a.generating = true
	a.lastParticipant = participant
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.begin(turnID, participant, listened.at, listened.revisedAt, listened.sttLatencyMs)
	a.turns.decided(turnID, time.Now())
	a.mu.Lock()
	if ttft := a.modelCallTimes[turnID]; ttft > 0 {
		a.turns.modelTiming(turnID, ttft)
	}
	if preview := a.previews[turnID]; preview != nil {
		a.turns.modelStarted(turnID, preview.startedAt)
	}
	a.mu.Unlock()
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Prompt: text})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		Note:         joinNotes(note, interruptionNote, a.duplex.Note(listened.confidence)),
	}, text)
}

// pendingInterruptionNoteLocked returns the private context note for the next caller
// response. The caller must hold a.mu.
func (a *Agent) pendingInterruptionNoteLocked() string {
	if a.interruptedReplyPending {
		return interruptedReplyNote
	}
	return ""
}

func (a *Agent) userTurnLocked(text string, images []llm.ImagePart) llm.Message {
	return llm.Message{Role: llm.User, Content: text}
}

func (a *Agent) replayLocked() []llm.Message {
	return append([]llm.Message(nil), a.history...)
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
	a.saying = phrase
	a.mu.Unlock()

	a.logger.Debug("murmuring while the caller talks",
		"turn", turnID, "participant", participant.ID, "phrase", phrase)
	if err := a.speakWhole(turnID, phrase); err != nil {
		a.fail(err, "tts")
		return
	}
	a.emitter.Send(Backchannel{Participant: participant, Text: phrase})
}

// checkIn asks a caller who has gone quiet whether there is anything else.
//
// Unlike a murmur it is a turn the agent took, so it goes into the history and is
// reported as speech: without that, the "no, that was everything" that comes back
// answers a question the model cannot see it asked.
func (a *Agent) checkIn(participant stt.Participant, phrase string) {
	turnID := backchannelPrefix + turnStamp()

	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return
	}
	a.speakingTurn = turnID
	a.saying = phrase
	a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: phrase})
	a.mu.Unlock()

	a.logger.Debug("asking whether anything else is needed",
		"turn", turnID, "participant", participant.ID, "phrase", phrase)
	if err := a.speakWhole(turnID, phrase); err != nil {
		a.fail(err, "tts")
		return
	}
	a.emitter.Send(Responded{TurnID: turnID, Text: phrase})
}

// instructions is the system prompt for a turn: what the agent was told to be, ahead of
// it whatever it already knew about the person it is talking to.
func (a *Agent) instructions() string {
	parts := make([]string, 0, 3)
	if a.recalled != "" {
		parts = append(parts, a.recalled)
	}
	if a.prompt != "" {
		parts = append(parts, a.prompt)
	}
	// How to write for the voice comes last, because it is about the delivery rather than
	// about who the agent is.
	if a.voicePrompt != "" {
		parts = append(parts, a.voicePrompt)
	}
	return strings.Join(parts, "\n\n")
}

// generate asks the harness for a reply and starts draining it.
//
// Every reply is pulled by a goroutine of its own and fanned into one channel, because a
// turn is answered on its own stream now but only one goroutine may speak: two turns
// writing to the voice at once is two voices. Create itself is on that goroutine too:
// waiting here for headers used to stall STT and flow rulings until Cerebras answered,
// which is how a follow-up sat unanswered behind the turn it interrupted.
// screen is what the guardrail should be asked about, which is only ever something a
// participant just said. A turn that continues one - a tool's result coming back, a
// delegated finding arriving - carries nothing new, and re-screening the words that
// started it would charge for a judgement already made.
func (a *Agent) generate(turn harness.Turn, screen string) error {
	a.mu.Lock()
	if a.closed || a.harness == nil || a.replies == nil {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: not joined"))
	}
	if a.switching.Load() {
		a.mu.Unlock()
		a.logger.Debug("a turn arrived while the models were changing, so it was dropped", "turn", turn.ID)
		return nil
	}
	current := a.harness
	preview := a.previews[turn.ID]
	delete(a.previews, turn.ID)
	stale := make([]string, 0, len(a.kept))
	for _, kept := range a.kept {
		stale = append(stale, kept.key)
	}
	ctx, cancel := context.WithCancel(a.ctx)
	a.generatingCancel[turn.ID] = cancel
	a.pumps.Add(1)
	a.mu.Unlock()
	// Whatever else was previewed was written for a conversation this reply is about to
	// change, so it could not be adopted whatever the words say.
	for _, key := range stale {
		a.cancelPreview(key)
	}

	go a.startReply(current, turn, ctx, screen, preview)
	return nil
}

// startReply opens the model stream and drains it. It is a goroutine of its own because
// Respond waits for response headers, and the event loop that called generate cannot sit
// in that: an overlap ruling that arrives while it does is the one that should cancel it.
//
// It is also where a guardrail is enforced, and it is enforced here rather than before the
// turn was started because of what a voice conversation costs in waiting. The check runs
// beside the model and the reply is held at the last moment before it would be spoken, so
// screening a turn the policy permits - which is nearly all of them - adds only whatever
// the check had still not finished by the time the model began answering. A turn the
// policy refuses has cost some tokens nobody will read, which is the trade.
func (a *Agent) startReply(
	current *harness.Harness, turn harness.Turn, ctx context.Context, screen string,
	preview *replyPreview,
) {
	defer a.pumps.Done()
	defer a.finishGenerate(turn.ID)

	screening := a.screening(ctx, turn.ID, screen)

	// In blocking mode the model is not asked at all until the verdict is in, so a refused
	// turn spends nothing on a reply nobody hears - and every caller waits for the check,
	// including all the ones who asked something perfectly ordinary.
	if screening != nil && screening.blocking {
		if verdict := screening.verdict(); !verdict.Allowed {
			a.refuse(turn.ID, verdict, 0)
			return
		}
		screening = nil
	}

	var stream *llm.Stream
	var err error
	usingPreview := false
	if preview != nil {
		if preview.turn.Instructions == turn.Instructions && preview.turn.Note == turn.Note &&
			slices.EqualFunc(preview.turn.History, turn.History, llm.SameMessage) &&
			current.AdoptPreview(turn, preview.model) {
			defer preview.cancel()
			select {
			case result := <-preview.ready:
				stream, err = result.stream, result.err
				usingPreview = stream != nil && err == nil
			case <-ctx.Done():
				preview.cancel()
				return
			}
		} else {
			preview.cancel()
		}
	}
	if preview != nil && stream == nil && ctx.Err() == nil {
		// The preview may have failed or lost a race to new context. The accepted
		// turn still deserves the ordinary reply path.
		err = nil
	}
	if stream == nil && err == nil {
		a.turns.modelStarted(turn.ID, time.Now())
		stream, err = current.Respond(ctx, turn)
	}
	if err != nil {
		if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
			return
		}
		// Ended the way a reply failing mid-stream ends, so the turn is over: left
		// generating, the agent never looks idle and a model switch waits on it forever.
		a.mu.Lock()
		replies := a.replies
		a.mu.Unlock()
		replies <- llm.ResponseFailed{ResponseID: turn.ID, Err: err, Context: "llm"}
		replies <- llm.ResponseCompleted{Response: llm.Response{ID: turn.ID, Status: llm.StatusFailed}}
		return
	}

	a.mu.Lock()
	a.streams[turn.ID] = stream
	_, abandoned := a.abandoned[turn.ID]
	closed := a.closed
	a.mu.Unlock()
	if abandoned || closed {
		// The caller took the floor while the request was still going out.
		stream.Close()
	}

	// The gate, and the reason it is exactly here: pump is what puts a reply on its way to
	// the voice, so a reply held on this side of it has not been heard, written down or
	// spoken. Nothing the model wrote escapes a refusal.
	if screening != nil {
		waited := time.Now()
		verdict := screening.verdict()
		if !verdict.Allowed {
			stream.Close()
			a.mu.Lock()
			delete(a.streams, turn.ID)
			a.mu.Unlock()
			a.refuse(turn.ID, verdict, routing.MsSince(waited))
			return
		}
	}

	if usingPreview {
		a.pumpPreview(turn.ID, preview)
		return
	}
	a.pump(turn.ID, stream)
}

// retarget gives an event of a preview the id of the turn that took it over. A preview kept
// across a Wait was started under the earlier candidate's id, which is not what the reply is
// known by once another candidate puts the same words.
func retarget(event llm.Event, from, to string) llm.Event {
	if from == to {
		return event
	}
	switch typed := event.(type) {
	case llm.ResponseCreated:
		typed.ResponseID = to
		return typed
	case llm.OutputTextDelta:
		typed.ResponseID = to
		return typed
	case llm.ReasoningTextDelta:
		typed.ResponseID = to
		return typed
	case llm.FunctionCallArgumentsDelta:
		typed.ResponseID = to
		return typed
	case llm.ResponseFailed:
		typed.ResponseID = to
		return typed
	case llm.ResponseCompleted:
		typed.Response.ID = to
		return typed
	}
	return event
}

func (a *Agent) pumpPreview(turnID string, preview *replyPreview) {
	a.mu.Lock()
	replies := a.replies
	a.mu.Unlock()
	for event := range preview.events {
		replies <- retarget(event, preview.turn.ID, turnID)
	}
	a.mu.Lock()
	delete(a.streams, turnID)
	a.mu.Unlock()
}

// screening is a guardrail check running beside the model.
type screening struct {
	// blocking says the model must not be asked until this has answered.
	blocking bool
	decided  chan guardrail.Verdict
}

// verdict waits for the check to answer.
func (s *screening) verdict() guardrail.Verdict { return <-s.decided }

// screening starts the check for a turn, or returns nil where there is nothing to screen:
// no policy, or a turn that carries no new words of the caller's.
func (a *Agent) screening(ctx context.Context, turnID, text string) *screening {
	if a.options.Guardrail == nil || strings.TrimSpace(text) == "" {
		return nil
	}

	policy := a.options.Guardrail.Policy()
	started := &screening{
		blocking: policy.Mode == guardrail.ModeBlocking,
		decided:  make(chan guardrail.Verdict, 1),
	}

	go func() {
		decided, err := a.options.Guardrail.Check(ctx, turnID, text)
		if err != nil {
			// A check that could not be made allows the turn. This is the one place that
			// decision is taken, so it is worth stating plainly: a classifier having an
			// outage, or a customer's own webhook being down, should not leave an agent
			// mute on every turn. Making it fail closed instead is this branch.
			a.logger.Error("the guardrail could not screen a turn, so it was answered",
				"turn", turnID, "error", err)
			decided = guardrail.Verdict{Allowed: true}
		}
		started.decided <- decided
	}()

	return started
}

// refuse answers a turn with the policy's refusal instead of the model's reply.
//
// The refusal is put through the same channel a model's reply travels, as a delta and a
// completion, rather than spoken from here. That is not indirection for its own sake: one
// goroutine owns the chunker, the voice and the turn timings, and a refusal written from
// this one would race it. Going the long way round also means a refused turn is recorded,
// reported and measured by exactly the code that does it for every other turn, so what a
// transcript, a stat row and a client see is a turn the agent answered briefly.
func (a *Agent) refuse(turnID string, verdict guardrail.Verdict, heldMs float64) {
	refusal := verdict.Refusal
	if refusal == "" {
		refusal = a.options.Guardrail.Policy().Refusal
	}

	a.logger.Info("a turn was refused by the guardrail",
		"turn", turnID, "reason", verdict.Reason, "probability", verdict.Probability,
		"held_ms", heldMs)
	a.emitter.Send(Blocked{
		TurnID:      turnID,
		Reason:      verdict.Reason,
		Probability: verdict.Probability,
		HeldMs:      heldMs,
	})

	a.mu.Lock()
	replies := a.replies
	a.mu.Unlock()
	replies <- llm.OutputTextDelta{ResponseID: turnID, Delta: refusal}
	replies <- llm.ResponseCompleted{Response: llm.Response{ID: turnID, OutputText: refusal}}
}

// finishGenerate drops the cancel for a turn whose Create has settled or been abandoned.
func (a *Agent) finishGenerate(turnID string) {
	a.mu.Lock()
	cancel := a.generatingCancel[turnID]
	delete(a.generatingCancel, turnID)
	for from, to := range a.previewTurns {
		if to == turnID {
			delete(a.previewTurns, from)
		}
	}
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// pump drains one reply into the channel the speaking goroutine reads.
func (a *Agent) pump(turnID string, stream *llm.Stream) {
	defer stream.Close()

	a.mu.Lock()
	replies := a.replies
	a.mu.Unlock()
	for stream.Next() {
		replies <- stream.Current()
	}

	a.mu.Lock()
	delete(a.streams, turnID)
	a.mu.Unlock()
}

// consumeLLM turns the model's deltas into sentences and sends them to be spoken.
//
// It is the only goroutine that speaks.
func (a *Agent) consumeLLM(p *pipeline, replies <-chan llm.Event) {
	defer p.running.Done()

	for event := range replies {
		a.handle(event)
	}
}

// handle deals with one event from the model.
func (a *Agent) handle(event llm.Event) {
	switch typed := event.(type) {
	case llm.OutputTextDelta:
		if !a.speaking(typed.ResponseID) {
			// The turn was interrupted, so the rest of the reply is not spoken.
			return
		}
		a.say(typed.ResponseID, typed.Delta)

	case llm.ReasoningTextDelta:
		// Thinking is only ever read, never spoken, so a voice has no use for it.
		if a.options.Text && typed.Delta != "" && a.speaking(typed.ResponseID) {
			a.emitter.Send(ReasoningDelta{TurnID: typed.ResponseID, Text: typed.Delta})
		}

	case llm.ResponseFailed:
		a.fail(typed.Err, "llm")

	case llm.ResponseCompleted:
		if !a.speaking(typed.Response.ID) {
			// The reply the chunker is holding is only cleared by the turn it belongs
			// to, so a response arriving late for an abandoned one cannot cut the turn
			// after it short.
			if a.replying == typed.Response.ID {
				a.resetTurn()
			}
			return
		}
		a.finish(typed.Response)
	}
}

// say sends one delta of a reply on its way to the voice.
func (a *Agent) say(turnID, delta string) {
	if !a.claimModelBuffers(turnID) {
		return
	}
	// What the model wrote is not all meant for the caller: a request for help is
	// addressed to the harness, and is taken out here rather than spoken.
	speech := a.harness.Filter(turnID, delta)
	if speech == "" {
		return
	}
	if !a.speaking(turnID) {
		return
	}
	a.turns.firstText(turnID, time.Now())

	// A stage direction is addressed to the voice, not to the caller: it is taken out of
	// what is read and remembered, and left in only for a voice that can act it. A
	// conversation in writing has no voice to address, so a bracket there is only ever
	// text -- a markdown link, an array literal -- and taking it out corrupts the reply.
	plain := speech
	if !a.options.Text {
		plain = a.directions.Add(speech)
	}
	a.mu.Lock()
	if a.speakingTurn != turnID {
		a.mu.Unlock()
		return
	}
	if plain != "" {
		a.spoken.WriteString(plain)
		a.saying = a.spoken.String()
	}
	a.mu.Unlock()
	if plain != "" {
		a.emitter.Send(ResponseDelta{TurnID: turnID, Text: plain})
	}
	// The delta is the whole of the reply when there is no voice: a reader has already
	// been given it, and cutting it into sentences would only be for the speaking.
	if a.options.Text {
		return
	}
	if !a.performs {
		speech = plain
	}
	if speech == "" {
		return
	}

	for _, sentence := range a.chunk.Add(speech) {
		if err := a.speakSentence(turnID, sentence); err != nil {
			a.fail(err, "tts")
		}
	}
}

// finish closes out a completed model reply.
func (a *Agent) finish(response llm.Response) {
	if !a.claimModelBuffers(response.ID) {
		return
	}

	// Text the harness was holding on the chance it began a request for help was only
	// ever text, so it is spoken.
	tail := a.harness.Flush()
	// A direction the stripper was still holding is released the same way: unfinished, it
	// was only ever text. In writing the stripper never ran, so there is nothing held.
	plain := tail
	if !a.options.Text {
		plain = a.directions.Add(tail) + a.directions.Flush()
	}
	a.spoken.WriteString(plain)
	if !a.performs {
		tail = plain
	}
	// Taken before resetTurn forgets them: a skill tag that named a tool is a call this
	// turn made, and has to be on the history the result answers.
	asked := a.harness.TakeAsked()
	calls := append(append([]llm.ToolCall(nil), response.ToolCalls...), asked...)
	said := strings.TrimSpace(a.spoken.String())
	a.mu.Lock()
	_, abandoned := a.abandoned[response.ID]
	active := a.speakingTurn == response.ID && !abandoned
	if active {
		// The ResponseCompleted tail is generated text too. Publish it as current context
		// before output work so a concurrent interruption can preserve the whole prefix.
		a.saying = said
	}
	a.mu.Unlock()
	if !active {
		a.resetTurn()
		return
	}

	if a.options.Text {
		// There is no voice to release it to, so the held text is reported as the last
		// of the reply. Without this a reader would be missing whatever the harness was
		// still deciding about when the model stopped.
		if plain != "" {
			a.emitter.Send(ResponseDelta{TurnID: response.ID, Text: plain})
		}
	} else {
		for _, sentence := range a.chunk.Add(tail) {
			if err := a.speakSentence(response.ID, sentence); err != nil {
				a.fail(err, "tts")
			}
		}
		// Whatever did not end in punctuation is still worth saying.
		if remainder := a.chunk.Flush(); remainder != "" {
			if err := a.speakSentence(response.ID, remainder); err != nil {
				a.fail(err, "tts")
			}
		}
		// A model that reaches for a tool without a word leaves the caller listening to
		// nothing until it comes back, which on a phone is indistinguishable from having
		// been cut off. Prompting for it is not enough: the models that do it reliably
		// are not the ones fast enough to hold a conversation.
		if fillsPause(response.ID, calls) && strings.TrimSpace(a.spoken.String()) == "" {
			filler := a.duplex.Working()
			a.spoken.WriteString(filler)
			a.mu.Lock()
			if a.speakingTurn == response.ID {
				a.saying = strings.TrimSpace(a.spoken.String())
			}
			a.mu.Unlock()
			if err := a.speakSentence(response.ID, filler); err != nil {
				a.fail(err, "tts")
			}
		}
		if err := a.closeUtterance(response.ID); err != nil {
			a.fail(err, "tts")
		}
	}
	said = strings.TrimSpace(a.spoken.String())
	expected := a.expectedSyntheses(response.ID)
	// Reset model-consumer buffers before releasing ownership, but leave a.saying intact
	// until history and generating state are committed under Agent.mu below.
	a.resetTurn()

	a.mu.Lock()
	_, abandoned = a.abandoned[response.ID]
	if a.speakingTurn != response.ID || abandoned {
		a.mu.Unlock()
		return
	}
	a.generating = false
	a.saying = ""
	// A reply that only called a tool is still a turn the model took, and it has to be
	// recorded with the calls on it: the result sent back answers one of them, and a
	// provider refuses a conversation where it answers nothing.
	if said != "" || len(calls) > 0 {
		a.history = append(a.history, llm.Message{
			Role:      llm.Assistant,
			Content:   said,
			ToolCalls: calls,
		})
		// A reply that is still held has nothing of it heard, so if it is abandoned in that
		// state its entry is taken back rather than left as something the caller was told.
		if a.gated.turn == response.ID {
			a.gated.committed = len(a.history)
		}
	}
	// History commit and generating=false are one ownership transition. stopPlayback uses
	// the same lock to decide whether this turn still needs a partial assistant entry.
	exchange := lastExchange(a.history)
	history := append([]llm.Message(nil), a.history...)
	currentHarness := a.harness
	a.mu.Unlock()

	// A model-complete turn is counted only if it won the race with interruption.
	a.turns.completed(response.ID, response.TimeToFirstTokenMs, expected)

	// A provider that kept this reply can be asked to carry on from it next turn rather
	// than read the conversation again.
	if currentHarness != nil {
		currentHarness.Remember(response)
	}

	// Remembering happens off the turn path: extraction takes longer than a turn and the
	// next thing the participant says must not wait for it.
	if a.memory != nil {
		a.memory.Remember(exchange)
	}
	if err := a.converse.Compact(currentHarness, history, response.Usage.InputTokens, response.Usage.InputTokensDetails.CachedTokens); err != nil {
		a.fail(err, "compaction")
	}

	pendingWork := len(calls) > 0 || a.Busy()
	if pendingWork {
		a.mu.Lock()
		a.owedTurn = response.ID
		a.mu.Unlock()
	}
	a.emitter.Send(Responded{
		PendingWork:        pendingWork,
		TurnID:             response.ID,
		Text:               said,
		TimeToFirstTokenMs: response.TimeToFirstTokenMs,
	})
	// Tools are handed over rather than run here, because this is the goroutine that
	// speaks and a transfer is several seconds of network the caller would hear as silence.
	if currentHarness != nil && len(calls) > 0 {
		pending := 0
		tools := a.availableTools()
		for _, call := range calls {
			if _, known := tools.Lookup(call.Name); known {
				pending++
			}
		}
		a.mu.Lock()
		a.pendingTools += pending
		a.mu.Unlock()
		currentHarness.Requested(response.ID, calls)
	}
	a.respondQueued()
	// A note that landed while this reply was being written waited for it to finish.
	a.followUp()
}

// claimModelBuffers assigns the harness, direction stripper and chunker to one active
// model response. After interruption, the next response must clear any unfinished buffers
// before it filters new text; a late completion for the abandoned response then cannot
// clear those new buffers because replying already names their owner.
func (a *Agent) claimModelBuffers(turnID string) bool {
	a.mu.Lock()
	if a.speakingTurn != turnID {
		a.mu.Unlock()
		return false
	}
	reset := a.replying != turnID
	a.replying = turnID
	a.mu.Unlock()
	if reset {
		a.resetTurn()
	}
	return true
}

// fillsPause reports whether a turn that said nothing should say something before the
// tools it asked for are run.
func fillsPause(completionID string, calls []llm.ToolCall) bool {
	// A turn that is itself a tool's answer gets no follow-up, so filling the pause on
	// one would leave the caller with a promise to check as the last thing they heard.
	if len(calls) == 0 || strings.HasPrefix(completionID, toolPrefix) {
		return false
	}
	// Pressing a menu option is meant to be silent. The menu answers next, and talking
	// over it is talking to nobody.
	return slices.ContainsFunc(calls, func(call llm.ToolCall) bool {
		return call.Name != toolPress
	})
}

// consumeTTS publishes the agent's speech to the edge as it is synthesised.
func (a *Agent) consumeTTS(p *pipeline, voice *ttsrouter.Session) {
	defer p.running.Done()

	// Abandoned audio arrives a frame at a time, so it is reported once per utterance
	// rather than once per frame.
	dropping := ""
	// released is the turn whose first frame has been let out, so none of its later frames is
	// held to the caller's silence again.
	released := ""

	for event := range voice.Events() {
		switch typed := event.(type) {
		case tts.AudioChunk:
			// Every active synthesis carries the publication epoch it began in. Normal
			// turn rollover leaves that epoch alone, while interruption clears it for all
			// queued tails as well as the current sentence.
			publishCtx, active := a.synthesisContext(typed.SynthesisID)
			if !active {
				a.turns.dropped(turnOf(typed.SynthesisID), typed.Audio.DurationMs())
				if dropping != typed.SynthesisID {
					dropping = typed.SynthesisID
					a.logger.Debug("dropping audio for a turn the agent has left behind",
						"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID))
				}
				continue
			}
			if turnID := turnOf(typed.SynthesisID); turnID != released {
				if !a.admitFirstFrame(p, publishCtx, turnID) {
					a.turns.dropped(turnID, typed.Audio.DurationMs())
					continue
				}
				released = turnID
			}
			var err error
			if marked, ok := a.options.Edge.(MarkedPlayout); ok {
				// Publishing returns once the chunk is queued, which for a long one is well
				// after the participants could have heard it begin, so the edge says when.
				err = marked.PublishAudioMarked(publishCtx, typed.Audio,
					a.turns.marksFor(turnOf(typed.SynthesisID)))
			} else if playout, ok := a.options.Edge.(ContextPlayout); ok {
				err = playout.PublishAudioContext(publishCtx, typed.Audio)
			} else {
				// An edge without cancellable playout gets the legacy call while Agent.mu
				// is held, so its bounded chunk write cannot race past DropSpeech.
				a.mu.Lock()
				if current := a.synthesisCtx[typed.SynthesisID]; current != publishCtx ||
					publishCtx.Err() != nil {
					a.mu.Unlock()
					a.turns.dropped(turnOf(typed.SynthesisID), typed.Audio.DurationMs())
					continue
				}
				err = a.options.Edge.PublishAudio(typed.Audio)
				a.mu.Unlock()
			}
			if errors.Is(err, context.Canceled) || publishCtx.Err() != nil {
				a.turns.dropped(turnOf(typed.SynthesisID), typed.Audio.DurationMs())
				continue
			}
			if err != nil {
				a.fail(err, "edge")
				continue
			}
			a.mu.Lock()
			stillActive := a.synthesisCtx[typed.SynthesisID] == publishCtx && publishCtx.Err() == nil
			if stillActive {
				a.lastSpokeAt = time.Now()
				// This records local edge admission after its backpressure wait, not when
				// the participant hears the chunk on the wire.
				a.turns.firstAudio(turnOf(typed.SynthesisID), time.Now())
				// Some of the reply is out, so it is no longer one nobody has heard.
				if a.gated.turn == turnOf(typed.SynthesisID) {
					a.gated = heldReply{}
				}
			}
			a.mu.Unlock()
			if !stillActive {
				a.turns.dropped(turnOf(typed.SynthesisID), typed.Audio.DurationMs())
				continue
			}

		case tts.SynthesisStarted:
			a.logger.Debug("the voice took an utterance",
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"provider", typed.Provider, "voice", typed.Voice)

		case tts.SynthesisComplete:
			active := a.finishSynthesis(typed.SynthesisID)
			a.logger.Debug("finished speaking",
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"audio_ms", typed.AudioDurationMs, "ttfb_ms", typed.TimeToFirstByteMs,
				"interrupted", typed.Interrupted)
			a.turns.spoke(turnOf(typed.SynthesisID), typed.TimeToFirstByteMs, typed.AudioDurationMs)
			if active && !typed.Interrupted {
				// An interrupted utterance was not heard in full, so it must not be
				// recorded as a finished spoken reply.
				a.emitter.Send(Spoke{
					TurnID:            turnOf(typed.SynthesisID),
					AudioDurationMs:   typed.AudioDurationMs,
					TimeToFirstByteMs: typed.TimeToFirstByteMs,
				})
			}
			if active {
				a.respondQueued()
				// An answer that came back while the agent was talking waited for this.
				a.followUp()
			}

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
			active := a.finishSynthesis(typed.SynthesisID)
			// A failure naming an utterance still settles it, since the router turns that
			// utterance's completion into the failed row.
			if typed.SynthesisID == "" {
				a.settle()
			}
			a.fail(typed.Err, "tts")
			if active {
				a.respondQueued()
				a.followUp()
			}
		}
	}
}

// consumeHarness reports what the harness decided, and speaks whatever the subagent came
// back with.
func (a *Agent) consumeHarness(p *pipeline, current *harness.Harness, drained chan struct{}) {
	defer a.running.Done()
	defer close(drained)

	events := current.Events()
	results := p.eotResults
	pipelineDone := p.ctx.Done()
	for events != nil || results != nil {
		var event harness.Event
		select {
		case typedEvent, ok := <-events:
			if !ok {
				events = nil
				continue
			}
			event = typedEvent
		case result := <-results:
			a.consumeEOTResult(result, current, p)
			continue
		case <-pipelineDone:
			results = nil
			pipelineDone = nil
			continue
		}
		switch typed := event.(type) {
		case harness.Decided:
			a.decideFromHarness(p, current, typed)

		case harness.Compacted:
			a.applyCompaction(typed)

		case harness.Delegated:
			a.converse.Delegating(typed.TaskID, typed.Skill, typed.Prompt, typed.TurnID)
			a.emitter.Send(Delegated{
				StartedAt: typed.StartedAt,
				TaskID:    typed.TaskID,
				Skill:     typed.Skill,
				Prompt:    typed.Prompt,
				TurnID:    typed.TurnID,
			})

		case harness.ToolRequested:
			// Flow decisions arrive on this same event stream. A tool waiting for
			// an external result must not prevent the caller from interrupting it.
			a.mu.Lock()
			if a.closed {
				a.mu.Unlock()
				continue
			}
			a.running.Add(1)
			a.mu.Unlock()
			ctx, cancel := a.prepareTool(typed)
			go func() {
				defer a.running.Done()
				a.executeTool(ctx, cancel, typed)
			}()

		case harness.Settled:
			a.converse.Delegated(typed.Result)
			if typed.State == harness.Cancelled {
				a.emitter.Send(TaskCancelled{
					TaskID: typed.TaskID,
					Skill:  typed.Skill,
					Reason: typed.Reason,
				})
			} else {
				a.emitter.Send(TaskSettled{
					Evidence:  typed.Evidence,
					TaskID:    typed.TaskID,
					Skill:     typed.Skill,
					Text:      typed.Text,
					Question:  typed.Question,
					ElapsedMs: typed.ElapsedMs,
					Err:       typed.Err,
					Files:     typed.Files,
				})
			}
			// Asked after the report rather than instead of it: work that ran out of time
			// is cancelled and still owes the caller a word.
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
	return slices.EqualFunc(first, second, llm.SameMessage)
}

// follow starts a turn nobody asked for, because work the caller was told was coming has
// come back and they are owed the answer.
//
// An agent still writing or speaking is left alone: taking the turn from itself would
// cut its own sentence off. Speech that has been published but not yet heard is the
// same: starting now would talk over the tail. What came back stays pending in the
// harness, and a later chance — a synthesis settling, the reply finishing, a presence
// tick — tries again.
//
// The lock is held across the whole turn because deciding to speak and taking what there
// is to say must not be separable: two answers landing together would otherwise give one
// of them a turn with nothing in it.
func (a *Agent) follow() error {
	if a.native() {
		return a.followNative()
	}
	a.following.Lock()
	defer a.following.Unlock()

	if a.waitingForPlayout() {
		return nil
	}

	a.mu.Lock()
	if a.harness == nil || a.generating || a.utterances > 0 {
		a.mu.Unlock()
		return nil
	}
	if !a.harness.Pending() && !a.toolReply {
		a.mu.Unlock()
		return nil
	}
	if a.pendingTools > 0 {
		a.mu.Unlock()
		return nil
	}
	// A turn nobody asked for must not take the floor from a caller who is still talking.
	if a.converse.Listening() {
		a.mu.Unlock()
		return nil
	}
	a.toolReply = false
	history := a.replayLocked()
	turnID := replyPrefix + turnStamp()
	a.speakingTurn = turnID
	a.generating = true
	participant := a.lastParticipant
	continues := a.owedTurn
	a.owedTurn = ""
	instructions := a.instructions()
	a.mu.Unlock()

	// This turn is deliberately not measured. A Turn reports the wait between someone
	// finishing a sentence and hearing the answer start, and nobody said anything here.
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Continues: continues})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
	}, "")
}

// waitingForPlayout reports whether already-published speech should still be left to drain
// before a follow-up turn begins.
//
// The edge is the one place that knows whether a tail is still buffered, so its answer is
// taken first. When that answer sticks long after the last chunk was published, though, the
// caller is better served by the follow-up starting than by silence that lasts until the
// next unrelated turn.
func (a *Agent) waitingForPlayout() bool {
	if !a.speechPending() {
		return false
	}
	a.mu.Lock()
	lastSpokeAt := a.lastSpokeAt
	a.mu.Unlock()
	return time.Since(lastSpokeAt) < playoutWaitCeiling
}

// followUp speaks whatever the caller is owed, reporting a failure rather than returning
// it because nothing that calls it has anyone to return it to.
func (a *Agent) followUp() {
	if err := a.follow(); err != nil {
		a.fail(err, "llm")
	}
}

// Busy reports whether the agent still has something to finish: a reply it is writing,
// speech it has not finished saying, work handed to the subagent, or an answer that has
// come back and still owes the caller a turn.
//
// It exists because one thing said to the agent can produce several replies -- the turn
// that called a tool, the turn that read the tool's answer, the turn a subagent's finding
// earned -- so a caller who waited only for the first would talk over the rest.
func (a *Agent) Busy() bool {
	a.mu.Lock()
	working := a.generating || a.utterances > 0
	current := a.harness
	a.mu.Unlock()

	if working {
		return true
	}
	if current == nil {
		return false
	}
	a.mu.Lock()
	owed := a.toolReply
	a.mu.Unlock()
	return current.Delegating() || current.Pending() || owed
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
	a.turns.ttsStarted(turnID, time.Now())

	if !voice.Streaming() {
		id := fmt.Sprintf("%s%s%d", turnID, sentenceSuffix, a.sentences)
		a.sentences++
		if !a.begin(turnID, id) {
			return nil
		}
		if err := voice.Synthesize(tts.Request{ID: id, Text: text, Final: true}); err != nil {
			a.finishSynthesis(id)
			return err
		}
		return nil
	}

	if a.openTurn != turnID {
		a.openTurn = turnID
		if !a.begin(turnID, turnID) {
			return nil
		}
	} else if !a.registerSynthesis(turnID, turnID, false) {
		return nil
	}
	if err := voice.Synthesize(tts.Request{ID: turnID, Text: text}); err != nil {
		a.finishSynthesis(turnID)
		return err
	}
	return nil
}

// speakWhole says a piece of text that is already complete, as one utterance.
func (a *Agent) speakWhole(turnID, text string) error {
	voice := a.voice()
	if voice == nil {
		return stack.Wrap(errors.New("agent: not joined"))
	}
	a.turns.ttsStarted(turnID, time.Now())
	if !a.begin(turnID, turnID) {
		return nil
	}
	if err := voice.Synthesize(tts.Request{ID: turnID, Text: text, Final: true}); err != nil {
		a.finishSynthesis(turnID)
		return err
	}
	return nil
}

// closeUtterance ends a streaming voice's utterance for a turn. A non-streaming one
// finished each sentence as it went, so there is nothing left to close.
func (a *Agent) closeUtterance(turnID string) error {
	voice := a.voice()
	if voice == nil || !voice.Streaming() || a.openTurn != turnID {
		return nil
	}
	if !a.registerSynthesis(turnID, turnID, false) {
		return nil
	}
	if err := voice.Synthesize(tts.Request{ID: turnID, Final: true}); err != nil {
		a.finishSynthesis(turnID)
		return err
	}
	return nil
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

// resetTurn forgets the model-consumer buffers of a turn. saying has its own Agent.mu
// ownership transition: finish clears it when history commits, and interruption clears
// it when local playout stops.
func (a *Agent) resetTurn() {
	a.chunk.Reset()
	if a.harness != nil {
		a.harness.Reset()
	}
	a.directions.Reset()
	a.spoken.Reset()
	a.sentences = 0
	a.openTurn = ""
}

// primaryPartialInterrupt recognizes a substantive transcript revision before it waits for
// the flow controller. The same revision still enters cadence, and only settled words can
// be answered.
func (a *Agent) primaryPartialInterrupt(transcript stt.Transcript, state floor) bool {
	if a.options.EOT == nil || a.options.EOTMode != EOTModePrimary || a.options.Text ||
		transcript.Final() || state.Quiet || state.Speaking == "" ||
		strings.HasPrefix(state.Speaking, backchannelPrefix) ||
		overlapNoise(transcript.Text) || !substantiveBargeIn(transcript.Text, state.Reply) {
		return false
	}

	a.mu.Lock()
	defer a.mu.Unlock()
	if a.closed || a.switching.Load() || a.speakingTurn != state.Speaking {
		return false
	}
	if transcript.Speaker != "" {
		known := a.voices[transcript.Participant.ID]
		if known == "" {
			a.voices[transcript.Participant.ID] = transcript.Speaker
		} else if known != transcript.Speaker {
			return false
		}
	}
	return true
}

type interruption struct {
	turnID      string
	participant stt.Participant
	reply       *llm.Stream
	voice       *ttsrouter.Session
	model       *stsrouter.Session
	done        chan struct{}
}

// stopPlayback abandons the current turn only if the action still names it. Cancelling
// the publication epoch and dropping the edge queue happen under the same lock that
// admits new synthesis, before any provider call can hold up local silence.
func (a *Agent) stopPlayback(
	participant stt.Participant,
	expectedTurn string,
	receivedAt time.Time,
	path string,
	source string,
) (interruption, bool) {
	a.mu.Lock()
	turnID := a.speakingTurn
	if turnID == "" || expectedTurn == "" || turnID != expectedTurn || a.interruptDone != nil {
		a.mu.Unlock()
		return interruption{}, false
	}
	// A murmur is meant to overlap with what someone is saying, so hearing them carry on
	// is not an interruption: there is no reply to abandon and nothing was cut short.
	if strings.HasPrefix(turnID, backchannelPrefix) {
		a.mu.Unlock()
		return interruption{}, false
	}
	wasGenerating := a.generating
	partial := a.saying // Keep only the string header on the local-stop path.
	// A reply still held to the caller's silence has let none of itself out, so the caller was
	// told nothing of it and it goes into the history as nothing heard.
	unheard := a.gated.turn == turnID
	committed := 0
	if unheard {
		committed = a.gated.committed
		a.gated = heldReply{}
	}
	a.abandoned[turnID] = struct{}{}
	a.speakingTurn = ""
	a.generating = false
	a.saying = ""
	a.toolReply = false
	stopped := interruption{
		turnID:      turnID,
		participant: participant,
		reply:       a.streams[turnID],
		voice:       a.tts,
		model:       a.sts,
		done:        make(chan struct{}),
	}
	a.interruptDone = stopped.done
	a.cancelPlayoutLocked()
	// Every synthesis in the canceled epoch has ended from the listener's point of view.
	// Keep its canceled context by ID until its terminal event, so a late completion can
	// neither claim a new epoch nor settle a newer utterance.
	a.utterances = 0
	a.dropSpeech()
	localStopAt := time.Now()
	// Native replies already record their partial response in replyComplete. For a cascade,
	// save an unfinished generated prefix once; a normal completed history entry is already
	// present when the model won the race, but its audio may still have been interrupted.
	if !a.nativeMode.Load() {
		switch {
		case unheard:
			// What was generated was never said, so none of it is kept as said. The entry a
			// finished reply added is taken back if nothing has been added since; one that
			// asked for tools stays, because their results answer it, and the next turn is
			// told it may not have been heard.
			if last := len(a.history) - 1; committed > 0 && last == committed-1 &&
				a.history[last].Role == llm.Assistant && len(a.history[last].ToolCalls) == 0 {
				a.history = a.history[:last]
			} else if committed > 0 {
				a.interruptedReplyPending = true
			}
		default:
			if wasGenerating {
				partial = strings.TrimSpace(partial)
				if partial != "" {
					a.history = append(a.history, llm.Message{Role: llm.Assistant, Content: partial})
				}
			}
			a.interruptedReplyPending = true
		}
	}
	for _, cancel := range a.toolCancels {
		cancel()
	}
	a.mu.Unlock()

	if !receivedAt.IsZero() {
		localStopMs := float64(localStopAt.Sub(receivedAt).Microseconds()) / 1000
		a.logger.Debug("stopped local playback",
			"path", path, "source", source, "turn", turnID, "local_stop_ms", localStopMs)
	}
	return stopped, true
}

func (a *Agent) finishInterruptedTurn(stopped interruption) {
	a.abandon(stopped.turnID)
	a.finishInterrupt(stopped)
}

// finishInterrupt cancels provider work after local playback has stopped.
func (a *Agent) finishInterrupt(stopped interruption) {
	a.finishGenerate(stopped.turnID)
	a.logger.Debug("stopping mid-reply, the caller took the floor",
		"turn", stopped.turnID, "participant", stopped.participant.ID)

	if stopped.voice != nil {
		if err := stopped.voice.Interrupt(); err != nil {
			a.fail(err, "tts")
		}
	}
	// A native model is told to stop the reply it is speaking. It has no word for how
	// much the caller heard beyond what it sent, so the router's own count stands in.
	if stopped.model != nil {
		if err := stopped.model.Interrupt(0); err != nil {
			a.fail(err, "sts")
		}
	}
	// The reply still settles after this, and is still billed: what it generated before
	// being cut off was generated all the same.
	if stopped.reply != nil {
		if err := stopped.reply.Close(); err != nil {
			a.fail(err, "llm")
		}
	}

	a.turns.interrupt(stopped.turnID)
	a.emitter.Send(Interrupted{TurnID: stopped.turnID, Participant: stopped.participant})
	a.mu.Lock()
	if a.interruptDone == stopped.done {
		a.interruptDone = nil
		close(stopped.done)
	}
	a.mu.Unlock()
	a.respondQueued()
}

// shorten stops the model from adding more while allowing speech already sent to the
// voice to finish.
func (a *Agent) shorten() {
	a.mu.Lock()
	turnID := a.speakingTurn
	reply := a.streams[turnID]
	a.mu.Unlock()
	if turnID == "" || strings.HasPrefix(turnID, backchannelPrefix) || reply == nil {
		return
	}
	a.logger.Debug("cutting the reply short, letting the audio already sent finish", "turn", turnID)
	if err := reply.Close(); err != nil {
		a.fail(err, "llm")
	}
}

// respondQueued answers the turn held back while the agent was talking, if it has
// stopped and there is one.
func (a *Agent) respondQueued() {
	if action, waiting := a.converse.Waiting(a.floor()); waiting {
		a.perform(action)
	}
}

// speechPending reports whether speech the agent has already published is still waiting to
// be heard. A voice reporting an utterance finished only means it sent the last of it, so
// the tail is still on its way out of the edge. An edge that cannot say has nothing to wait
// for, which is what an in-process loopback is.
func (a *Agent) speechPending() bool {
	playout, ok := a.options.Edge.(Playout)
	return ok && playout.SpeechPending()
}

// dropSpeech throws away speech the agent has published but the caller has not heard yet.
func (a *Agent) dropSpeech() {
	if playout, ok := a.options.Edge.(Playout); ok {
		playout.DropSpeech()
	}
}

// speaking reports whether a turn is still the one the agent is on.
func (a *Agent) speaking(turnID string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.speakingTurn != "" && a.speakingTurn == turnID
}

// abandonedTurn reports whether an interruption gave up on a turn, which is what stops
// its audio being heard.
func (a *Agent) abandonedTurn(turnID string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	_, gone := a.abandoned[turnID]
	return gone
}

// voice returns the voice session, or nil when the agent has not joined.
func (a *Agent) voice() *ttsrouter.Session {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.tts
}

// begin counts an utterance as in flight, so Finish waits for it.
func (a *Agent) begin(turnID, synthesisID string) bool {
	return a.registerSynthesis(turnID, synthesisID, true)
}

func (a *Agent) registerSynthesis(turnID, synthesisID string, count bool) bool {
	for {
		a.mu.Lock()
		if done := a.interruptDone; done != nil {
			a.mu.Unlock()
			<-done
			continue
		}
		if a.closed || turnID == "" || synthesisID == "" || a.speakingTurn != turnID {
			a.mu.Unlock()
			return false
		}
		if _, abandoned := a.abandoned[turnID]; abandoned {
			a.mu.Unlock()
			return false
		}
		if a.synthesisCtx == nil {
			a.synthesisCtx = map[string]context.Context{}
		}
		publishCtx, exists := a.synthesisCtx[synthesisID]
		if !exists {
			if !count {
				a.mu.Unlock()
				return false
			}
			if a.playoutCtx == nil {
				parent := a.ctx
				if parent == nil {
					parent = context.Background()
				}
				a.playoutCtx, a.cancelPlayout = context.WithCancel(parent)
			}
			publishCtx = a.playoutCtx
			a.synthesisCtx[synthesisID] = publishCtx
		} else if publishCtx.Err() != nil {
			a.mu.Unlock()
			return false
		}
		if count {
			a.utterances++
		}
		a.mu.Unlock()
		return true
	}
}

func (a *Agent) synthesisContext(synthesisID string) (context.Context, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	publishCtx := a.synthesisCtx[synthesisID]
	return publishCtx, publishCtx != nil && publishCtx.Err() == nil
}

// finishSynthesis forgets one exact synthesis ID and reports whether its epoch was still
// active. A canceled completion is retained long enough to be recognized, but never
// settles a new reply that began in a later epoch.
func (a *Agent) finishSynthesis(synthesisID string) bool {
	if synthesisID == "" {
		return false
	}
	a.mu.Lock()
	publishCtx := a.synthesisCtx[synthesisID]
	delete(a.synthesisCtx, synthesisID)
	active := publishCtx != nil && publishCtx.Err() == nil
	if active && a.utterances > 0 {
		// Deleting the ID and settling its count are one operation. Otherwise an
		// interruption could reset the count and a new reply could begin between them.
		a.utterances--
	}
	a.mu.Unlock()
	return active
}

// cancelPlayoutLocked invalidates every synthesis begun in the current publication epoch.
// A future synthesis creates one fresh context lazily; audio chunks never allocate one.
func (a *Agent) cancelPlayoutLocked() {
	if a.cancelPlayout != nil {
		a.cancelPlayout()
	}
	a.playoutCtx = nil
	a.cancelPlayout = nil
}

// cancelPlayoutAndForgetLocked is for pipeline shutdown/replacement, where the old voice
// will no longer produce useful terminal events. Interruptions use cancelPlayoutLocked
// alone so each late ID remains bound to its canceled epoch until it completes.
func (a *Agent) cancelPlayoutAndForgetLocked() {
	a.cancelPlayoutLocked()
	clear(a.synthesisCtx)
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

// RestoreHistory seeds a new text session with previously completed conversation turns.
func (a *Agent) RestoreHistory(history []llm.Message) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.history = append([]llm.Message(nil), history...)
}

// openSubagent starts the subagent shared by cascade and native conversations, or is nil
// when there is none. It is routed like anything else, so the work it does is failed over
// and billed the same way a turn is. A skill that captures video needs it to see.
func (a *Agent) openSubagent(target string) func(context.Context) (*llmrouter.Session, error) {
	if target == "" {
		return nil
	}
	var modalities []string
	for _, skill := range a.options.Skills.Skills {
		if skill.CaptureVideo {
			modalities = []string{llm.ModalityImage}
		}
	}
	return func(ctx context.Context) (*llmrouter.Session, error) {
		return a.options.LLM.Start(ctx, llmrouter.Request{
			CustomerID: a.options.CustomerID, Caller: a.options.Caller,
			AgentID: a.options.AgentID, CallID: a.options.CallID,
			Tags: a.options.Tags, Target: target,
			LanguageHints: a.options.LanguageHints, InputModalities: modalities,
		})
	}
}
