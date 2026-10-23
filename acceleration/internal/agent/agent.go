// Package agent runs a voice conversation over the three routed modalities.
//
// It is the Go counterpart of the Python Agent in sdks/python: audio from the edge is
// transcribed, settled turns are answered by a model, and the reply is spoken back. What
// makes it worth having in this service is that it is built from the routers rather than
// from provider instances, so every turn is routed, failed over and billed by the same
// machinery as a direct API call.
package agent

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/GetStream/Vision-Agents/acceleration/internal/audio"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionrouter"
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
	// Judge routes a decision model the flow controller asks in place of ControllerTarget,
	// when JudgeTarget names one. Nil or an empty JudgeTarget keeps the language model.
	Judge       *decisionrouter.Router
	JudgeTarget string
	// Skills are what the voice model may hand over to the subagent. A written
	// conversation with no subagent loads them itself, with load_skill.
	Skills harness.Skills
	// Telephony is what the agent may do to the call itself. Without it the agent can
	// only talk, which is what a call that is not on a phone network can do.
	Telephony Telephony
	// ToolRunner carries out the tools that are not the two acting on the phone call,
	// which is how a caller outside this process owns its own tools.
	ToolRunner    ToolRunner
	OnToolStarted func(ToolStarted)
	// ToolPolicy is what the tool named asks of the agent (its connector binding's policy).
	// Nil, or the zero ToolPolicy for a tool, is how every tool has always run.
	ToolPolicy func(tool string) ToolPolicy
	// Tools are what the voice model may do rather than say. Each is only offered when
	// something on this call can run it: the telephony pair needs Telephony, and every
	// other tool needs a ToolRunner.
	Tools harness.Tools
	// Sandbox is where code the agent writes is run. A voice model is never offered it,
	// since running code takes seconds a call cannot spare, so there the subagent runs it.
	// A written conversation with no subagent runs it itself, with run_code.
	Sandbox sandbox.Sandbox
	// Publish puts the files code hands back where the caller can see them, which is the
	// conversation's channel when there is one. Nil means nowhere.
	Publish sandbox.Publisher
	// Tasks caps how much delegated work may run at once. Zero leaves the harness's own
	// default in place.
	Tasks int
	// Duplex lets the agent listen and talk at the same time rather than strictly taking
	// turns. Both halves of it are off by default.
	Duplex         DuplexOptions
	VideoSource    string
	VideoMaxFrames int

	// SpeculativeReplies starts the reply to a settled turn while the flow controller is
	// still deciding whether it was meant for the agent, and holds it until the ruling says
	// to answer. It takes the ruling's round trip off every answered turn, and costs the
	// tokens of the replies a ruling throws away. Off by default.
	SpeculativeReplies bool
	// UnifiedReplies has the reply to a settled turn on a quiet floor open with a tag
	// ruling on it, in place of the flow controller.
	UnifiedReplies bool

	// Voice selects the speaker. Its meaning is the text-to-speech provider's.
	Voice string
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
	// Incognito recalls memories but writes none, and records no turn timings or
	// decisions: what is said here is not kept anywhere.
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
	streams map[string]*llm.Stream
	// generatingCancel abandons a conversation Create that has not returned a stream yet.
	// Interrupt used to Close only an existing stream, so a reply waiting on headers kept
	// the event loop and the floor until Cerebras answered.
	generatingCancel map[string]context.CancelFunc
	toolCancels      map[string]context.CancelFunc
	// toolsRunning names each tool still running, by call, so a caller left waiting can
	// be told what on.
	toolsRunning map[string]string
	// toolHolds are the calls still running that an interruption answers with
	// stillRunning rather than cancels, by call.
	toolHolds map[string]*toolHold
	// skillFiles are the files of each skill a written conversation loaded, by skill, and
	// writtenFiles the ones already put in the sandbox, so run_code writes a skill's
	// scripts once and again only when the skill has changed.
	skillFiles   map[string]map[string]string
	writtenFiles map[string]map[string]string
	// speculations are replies started before the flow controller ruled on their words,
	// held until it does, by candidate.
	speculations map[string]*speculation
	// pumps are the goroutines draining those streams into replies.
	pumps sync.WaitGroup

	tts *ttsrouter.Session
	// recoveringVoice holds off new replies while a lost voice is replaced. It is
	// protected by mu and remains set until the old session is closed.
	recoveringVoice *ttsrouter.Session
	// lostVoices are the voices this call has lost, as "provider/model", so the one that
	// replaces a lost voice is a different one.
	lostVoices []string
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
	// decisionStore keeps what the conversation decided for an hour, so a call can be read
	// back after the process that held it is gone.
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
	// compactedAway is what compaction took out of history, oldest first. The model reads
	// a summary of it instead, but what the caller said and the tools returned are still
	// facts of the call, and the checks on a tool call's arguments are made against them.
	compactedAway []llm.Message
	// opened is the line the agent opened the conversation with before anybody spoke, told
	// to the model in its instructions rather than kept as a turn (Say).
	opened string
	// lateResults are lateResult messages, and lines said while a call ran, held back while
	// the history ends in a call not yet answered (callsOpen), in the order they came.
	lateResults []llm.Message
	// listeners holds one transcription session per participant, because a speech-to-text
	// stream is bound to a single speaker.
	listeners map[string]*sttrouter.Session
	// departed are the participants who have left and not come back. Their audio and
	// their departure arrive on separate channels, so a frame can come after the leave,
	// and a session opened for it would stay open until the agent closes. Each holds when
	// the first such frame was dropped, zero until one is (see departedGrace).
	departed map[string]time.Time
	// now is the clock departedGrace is measured on.
	now func() time.Time
	// hearings tally what each participant has been heard to say, kept for the whole call
	// so a line at its end covers those who left.
	hearings map[string]*hearing
	// learnedTerms are names the caller said and the agent has said back, which the
	// transcriber is told to expect for the rest of the call (keyterms.go).
	learnedTerms []string
	// voices is the diarised label of the first voice heard on each participant's track,
	// which is taken to be the caller's. A later turn in a different voice is somebody
	// else at the same microphone: the track says who joined the call, and it is the
	// wrong answer for everybody else in the room with them. Empty for the transcribers
	// that cannot tell one voice from another, which leaves it saying nothing rather
	// than guessing.
	voices map[string]string
	// speakers are the people, by personOf, who have had a turn answered, each with the
	// label their words go under. Once there are two, every new turn names who said it,
	// because the model cannot tell them apart otherwise. Turns typed with RespondTo are
	// the caller's and are never counted.
	speakers map[string]nameTag
	// speakingTurn is the turn the agent is currently on. It says which reply an
	// interruption would abandon and which one the floor belongs to; it does not decide
	// what may be heard, because the agent starts a turn for itself while the turn before
	// it is still being spoken.
	speakingTurn string
	// saying is the reply as the caller has heard it so far, so a controller ruling on
	// words that overlap it can tell a correction from the line echoing back. It is the
	// spoken text as it streams, or the whole phrase for a greeting or a murmur.
	saying string
	// abandoned is every turn an interruption gave up on. Audio belonging to one of them
	// is dropped rather than published, which is what makes barge-in immediate even while
	// a provider is still sending. It holds one entry per interruption and lives only as
	// long as the call.
	abandoned map[string]struct{}
	// unheard is the caller turn being answered while none of its reply has gone out, which
	// the same caller saying more makes stale.
	unheard unheardReply
	// missed is what a caller said that a dropped reply was for, owed with their next answer.
	missed unheardReply
	// voiced is the turn whose audio last went out, and voicedAt when its first did.
	voiced   string
	voicedAt time.Time
	// utterances counts syntheses that have not settled, so Finish knows when the agent
	// has stopped talking.
	utterances int
	// heard is each synthesis that has not settled, with what the voice was sent for it,
	// for reporting when it starts being heard.
	heard map[string]*utterance
	// finished is the syntheses that settled lately, oldest first in finishedOrder, so a
	// delta sent for one after it settled does not open it again.
	finished      map[string]struct{}
	finishedOrder []string
	// interrupting holds off new replies until the voice, edge and old model stream have
	// all finished the interruption cleanup.
	interrupting bool
	// generating is true while the voice model is still writing the current reply.
	generating bool
	// composing is true from asking for a line the agent chose to say (compose) until it
	// is said or dropped.
	composing bool
	// toolReply is set when a tool returned and the caller has not been told yet. A
	// second tool in the same turn must not start a competing generate: it would steal
	// speakingTurn and drop the first result unspoken.
	toolReply bool
	// pendingTools is how many tool calls from the current turn have not come back yet.
	// The spoken follow-up waits until this is zero so two results share one generate.
	pendingTools int
	// toolRounds is how many replies to tool results have been started since the caller
	// last spoke (maxToolRounds).
	toolRounds int
	// heardCaller is set once a turn has come from somebody speaking rather than a typed
	// instruction; until then replies are offered no tools.
	heardCaller bool
	// promiseChased is set once a reply that promised to check and called nothing has been
	// asked for the call (keepPromise), since the caller last spoke.
	promiseChased bool
	// callerTurns counts the turns the caller has started, so a chase put off for a moment
	// is dropped once they have said something new.
	callerTurns int
	// chaseForced is set once a chase that still called nothing has been chased again with
	// the call required, since the caller last spoke.
	chaseForced bool
	// narrationAsked is set once a reply that only talked about the caller has been asked
	// again, since the caller last spoke.
	narrationAsked bool
	// refusalChased is set once a refusal nothing was logged for has been chased, since the
	// caller last spoke.
	refusalChased bool
	// refusedWithCalls is set when a reply refused the caller and called other tools in the
	// same breath, since the caller last spoke, so the refusal is chased once those are done.
	refusedWithCalls bool
	// stepChased is the tool whose described step was last chased up since the caller
	// last spoke (describedStep).
	stepChased string
	// forcedChase is the chase that was made to call a tool, and forcedNote what it was
	// told, so a forced chase that produced nothing can be asked again without forcing.
	forcedChase string
	forcedNote  string
	// owedTurn is the last turn that ended with tools or delegated work outstanding, which
	// the reply delivering that work continues.
	owedTurn string
	// asked is what the caller asked to change about the reply to the last thing it said
	// (RespondTo), kept for the replies that deliver that turn's tools and delegated work.
	asked options.LLM
	// sender is who sent that last thing with RespondTo, kept the same way, so a turn a
	// user typed into a session a backend opened for them is counted on their day. It is
	// zero for what was heard, which is counted on whoever opened the session.
	sender routing.Caller
	// scheduled says that last thing was a turn a scheduled task fired, kept the same way,
	// so neither it nor the replies that finish it are counted on an end user's day.
	scheduled bool

	joined bool
	closed bool

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
	nativeCalls     map[string]llm.ToolCall
	// nativeRetries counts replacements across the call; swapping protects it.
	nativeRetries int
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
	// performed is what the chunker has been given of the reply for a voice that acts
	// directions, which the reply's message has none of: where a sentence is in the
	// message is found from it.
	performed []rune
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
	// withheld says a sentence of this reply claimed something was done with no tool
	// behind it, so it and everything after it went unsaid. Only the model consumer
	// touches it.
	withheld bool
	// withheldFrom is the sentence the withholding began at, for cutting the history at
	// what the caller heard.
	withheldFrom string
	// narrated says a sentence of this reply spoke of the caller rather than to them, so it
	// and everything after it went unsaid without the chase a withheld claim gets.
	narrated bool
	// promised says a sentence of this reply, said before any tool answered, told the caller
	// a step was under way. A direction the model writes after it ([short pause], [checks
	// outage map]) is the wait acted out, and what follows it a guess at the step's result.
	// Only the model consumer touches it.
	promised bool

	// following serialises the turns the agent takes of its own accord, so two answers
	// landing together cannot both decide the agent is free to speak.
	following sync.Mutex

	running   sync.WaitGroup
	closeOnce sync.Once
}

// New validates the options and returns an Agent. It opens nothing; Join does that.
func New(options Options) (*Agent, error) {
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

	logger := options.Logger.With("customer", options.CustomerID, "session", options.SessionID, "call", options.CallID)
	owner := routing.Owner{
		CustomerID: options.CustomerID,
		Caller:     options.Caller,
		AgentID:    options.AgentID,
		CallID:     options.CallID,
		Tags:       options.Tags,
	}

	settling := newCadence(cadenceGapFromEnv(logger), 0, 0, logger)
	listening := newDuplex(options.Duplex)
	emitter := NewEmitter(eventBuffer)
	agent := &Agent{
		options:          options,
		logger:           logger,
		emitter:          emitter,
		prompt:           options.Instructions,
		listeners:        map[string]*sttrouter.Session{},
		departed:         map[string]time.Time{},
		now:              time.Now,
		hearings:         map[string]*hearing{},
		voices:           map[string]string{},
		speakers:         map[string]nameTag{},
		abandoned:        map[string]struct{}{},
		heard:            map[string]*utterance{},
		finished:         map[string]struct{}{},
		streams:          map[string]*llm.Stream{},
		generatingCancel: map[string]context.CancelFunc{},
		speculations:     map[string]*speculation{},
		cadence:          settling,
		duplex:           listening,
	}

	// Turns are keyed by agent id and decisions by call id, so an agent missing either is
	// still measured and still reports itself, it is just not kept. Neither is kept for an
	// incognito session: a decision holds what was heard, and a turn names the agent. Both
	// are kept in Redis; a turn's counts also go to Postgres for the stats.
	if (options.Store != nil || options.Live != nil) && options.AgentID != "" && !options.Incognito {
		agent.turnStore = newTurnRecorder(options.Store, options.Live, owner, logger)
	}
	var record func(Decided)
	if options.Live != nil && options.CallID != "" && !options.Incognito {
		agent.decisionStore = newDecisionRecorder(options.Live, owner, logger)
		record = agent.decisionStore.Record
	}
	agent.converse = newConverse(settling, listening, emitter, record, 0, logger)
	agent.turns = newTurnTracker(owner, agent.finishTurn)
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
	started := a.turns.originOf(turn.TurnID)
	a.logger.Info("voice turn timing", "turn", turn.TurnID, "kind", started.kind, "parent", started.parent,
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
			// Those already in the call were queued while the edge joined. Reporting them
			// here, before Joined, is what lets a reader take a call that is empty at
			// Joined to be empty.
			for range len(roster.Attendance()) {
				a.attend(<-roster.Attendance())
			}
			a.running.Add(1)
			go a.consumeRoster(roster)
		}
	}

	changed := a.modelsChanged()
	a.logger.Info("joined", "llm", changed.LLM, "tts", changed.TTS, "sts", changed.STS)
	var sfu SFU
	if reporter, ok := a.options.Edge.(SFUReporter); ok {
		sfu = reporter.SFU()
	}
	a.emitter.Send(Joined{At: time.Now(), SFU: sfu})
	return nil
}

// Attends reports whether the agent is on a call whose edge says who comes and goes
// (Roster), so that a call with nobody reported by Joined is known to be empty.
func (a *Agent) Attends() bool {
	_, ok := a.options.Edge.(Roster)
	return !a.options.Text && ok
}

// SimpleResponse answers a piece of text through the model, as though a participant had
// said it. It returns once the request is on its way: the reply arrives on Events and is
// spoken as it streams.
func (a *Agent) SimpleResponse(ctx context.Context, text string) error {
	_, err := a.RespondTo(ctx, text, nil, options.LLM{})
	return err
}

// RespondTo answers a piece of text through the model, attaching images to that turn.
//
// It returns the turn the answer is being given as, which is what lets a caller follow one
// particular reply: every event of it carries the id, and it is what a session records the
// turn under. A native agent returns an empty one, because a speech-to-speech model decides
// for itself what counts as a turn and there is nothing here to name.
//
// Images go to the vision skill when the agent has one, and to the conversation model
// itself when it has none but the model can see. With neither the turn is refused with
// ErrCannotSeeImages rather than answered blind.
//
// asked is written over the session's own overwrites for this reply and the replies that
// finish it once its tools come back. CheckAsked says whether the model takes it.
func (a *Agent) RespondTo(ctx context.Context, text string, images []llm.ImagePart, asked options.LLM) (string, error) {
	if err := a.CheckAsked(asked); err != nil {
		return "", err
	}
	if a.native() {
		return "", a.respondNative(text, images)
	}
	caller := stt.Participant{ID: typedCaller}
	sender, _ := routing.CallerFrom(ctx)
	scheduled := routing.ScheduledTurn(ctx)
	// Typed into a call, the words are what the caller said as much as anything transcribed,
	// and the transcript, the review and the logs all take a caller's line from Heard. A text
	// session reads its caller's lines from Responding instead.
	typed := func() {
		if !a.options.Text {
			a.emitter.Send(Heard{Participant: caller, Text: text})
		}
	}
	if len(images) > 0 {
		a.mu.Lock()
		current, model := a.harness, a.llm
		a.mu.Unlock()
		if current == nil {
			return "", stack.Wrap(errors.New("agent: not joined"))
		}
		id := replyPrefix + turnStamp()
		attached := make([]llm.ImagePart, 0, len(images))
		for _, image := range images {
			if err := image.Validate(); err != nil {
				return "", err
			}
			image.Data = append([]byte(nil), image.Data...)
			attached = append(attached, image)
		}
		switch {
		case current.Offers(visionSkill):
			parts := llm.TextParts(text)
			for index, image := range attached {
				described := map[string]any{"source": "attachment", "frame_id": fmt.Sprintf("%s-image-%d", id, index+1), "received_at_ms": time.Now().UnixMilli()}
				if image.Caption != "" {
					described["caption"] = image.Caption
				}
				metadata, _ := json.Marshal(described)
				parts = append(parts, llm.ContentPart{Text: string(metadata)}, llm.ContentPart{Image: &image})
			}
			if _, err := current.Delegate(visionSkill, text, id, parts, nil); err != nil {
				return "", err
			}
			typed()
			return id, a.respondTurn(id, caller, text, heard{at: time.Now(), typed: true}, "Visual analysis has been requested. Wait for its findings before answering the visual question.", nil, asked, sender, scheduled)
		case model != nil && model.Capabilities().Accepts(llm.ModalityImage):
			typed()
			return id, a.respondTurn(id, caller, text, heard{at: time.Now(), typed: true}, "", attached, asked, sender, scheduled)
		default:
			return "", stack.Wrap(ErrCannotSeeImages)
		}
	}
	id := replyPrefix + turnStamp()
	typed()
	return id, a.respondTurn(id, caller, text, heard{at: time.Now(), typed: true}, "", nil, asked, sender, scheduled)
}

// CheckAsked reports whether the conversation model takes what a caller asked to change
// about one reply, so a reasoning effort it does not answer to is refused before the turn
// starts rather than failing it.
func (a *Agent) CheckAsked(asked options.LLM) error {
	if asked.ReasoningEffort == "" && asked.MaxOutputTokens == nil {
		return nil
	}
	if a.native() {
		return errors.New("agent: a speech-to-speech agent takes no reasoning effort or output limit")
	}
	a.mu.Lock()
	model := a.llm
	a.mu.Unlock()
	if model == nil {
		return stack.Wrap(errors.New("agent: not joined"))
	}
	return model.Capabilities().Validate(llm.ResponseParams{}.Overwrite(asked))
}

// VideoFramesTool is the caller's tool the agent reads frames of the user's video through.
const VideoFramesTool = "get_video_frames"

// visionSkill is the skill images are handed to when the agent has one.
const visionSkill = "vision"

// ErrCannotSeeImages is a turn carrying images for an agent with nothing that can look at
// them: no vision skill, and a conversation model that takes text alone.
var ErrCannotSeeImages = errors.New("agent: images need a vision skill or a conversation model that accepts images, and this agent has neither")

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
	a.history = append(toolPicturesSeen(a.history), llm.Message{Role: llm.User, Content: text})
	history := make([]llm.Message, len(a.history))
	for i, message := range a.history {
		history[i] = named(message)
	}
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
	return a.sayFor(text, "")
}

// sayFor says words as they are, for purpose: PurposeGreeting, or empty for any other line.
func (a *Agent) sayFor(text, purpose string) error {
	turnID := fmt.Sprintf("%s%d", sayPrefix, time.Now().UnixNano())

	// A conversation in writing has no voice to say it with, so it is reported as said
	// instead: what a caller would have heard is what a reader reads.
	if a.options.Text {
		a.emitter.Send(Responded{TurnID: turnID, Text: text, Purpose: purpose})
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
	// The model reads what the agent said from the history, so a line said without it has
	// to be kept too, or the caller's answer to a greeting's question reaches it with no
	// question. A line that opens the conversation is told to it instead: chat templates
	// that need a user turn first refuse a conversation that opens on the assistant's.
	said := llm.Message{Role: llm.Assistant, Content: text}
	opening := len(a.history) == 0
	switch {
	case opening:
		a.opened = text
	case callsOpen(a.history):
		a.lateResults = append(a.lateResults, said)
	default:
		a.history = append(a.history, said)
	}
	a.mu.Unlock()
	a.turns.started(turnID, kindSay)

	// Said first, so a watcher has the line before any of its sound. A voice that will not
	// take it leaves it unsaid, and it is closed as cut off rather than left coming.
	a.emitter.Send(Said{TurnID: turnID, Text: text, Purpose: purpose})
	if err := a.speakWhole(turnID, text); err != nil {
		// Never heard, so the model must not be told it was said, and the transcript must not
		// keep it as said.
		a.mu.Lock()
		a.forgetSaidLocked(opening, said)
		if a.speakingTurn == turnID {
			a.abandoned[turnID] = struct{}{}
			a.speakingTurn = ""
			a.saying = ""
		}
		a.mu.Unlock()
		a.turns.interrupt(turnID)
		a.emitter.Send(Interrupted{TurnID: turnID})
		return err
	}
	return nil
}

// sayPrefix starts the turn id of a line said as written, without the model.
const sayPrefix = "say-"

// Scripted reports whether a turn is a line the agent said as written rather than a reply
// the model wrote, such as the greeting.
func Scripted(turnID string) bool { return strings.HasPrefix(turnID, sayPrefix) }

// forgetSaidLocked takes back a line Say kept for the model, when the voice never said it.
// The caller holds a.mu.
func (a *Agent) forgetSaidLocked(opening bool, said llm.Message) {
	if opening {
		if a.opened == said.Content {
			a.opened = ""
		}
		return
	}
	for _, kept := range []*[]llm.Message{&a.lateResults, &a.history} {
		if last := len(*kept) - 1; last >= 0 && (*kept)[last].Role == said.Role && (*kept)[last].Content == said.Content {
			*kept = (*kept)[:last]
			return
		}
	}
}

// Greet opens the call with a greeting. Unvaried, it is said word for word without asking
// the model. Varied, the model writes its own variation of it first, and the words are said
// as they are when it cannot. A native model cannot say exact words, so it is prompted with
// the greeting either way and says its own rendering of it.
func (a *Agent) Greet(ctx context.Context, greeting string, vary bool) error {
	if a.native() {
		if vary {
			return a.Prompt(ctx, "Open the call by greeting the caller with your own variation of this greeting, keeping what it says: "+greeting)
		}
		return a.Prompt(ctx, "Open the call by greeting the caller. Say this, in these words or close to them: "+greeting)
	}
	if vary {
		line, err := a.compose(ctx, fmt.Sprintf(greetPurpose, greeting))
		if err != nil {
			a.logger.Warn("could not vary the greeting, saying it as written", "error", err)
		} else if line != "" {
			greeting = line
		}
	}
	return a.sayFor(greeting, PurposeGreeting)
}

// Interrupt abandons the reply being spoken, the way a participant talking over the agent
// would. It is what a caller outside the call has instead of a voice.
func (a *Agent) Interrupt() {
	a.mu.Lock()
	participant := a.lastParticipant
	turnID := a.speakingTurn
	for _, cancel := range a.toolCancels {
		cancel()
	}
	a.toolReply = false
	a.mu.Unlock()
	a.abandon(turnID)
	a.interrupt(participant)
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

// record returns the whole call: what compaction took out of the history as well as the
// history itself. Whether a caller said a value or a tool returned it does not stop being
// true when the model is handed a summary instead. Before this, an ID a tool returned was
// refused as made up once the result had been summarised away.
func (a *Agent) record() []llm.Message {
	a.mu.Lock()
	defer a.mu.Unlock()
	return append(append([]llm.Message(nil), a.compactedAway...), a.history...)
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
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
	// A swap waiting for its turn boundary gives up once closed is set, and one already
	// changing the pipeline finishes first, so the pipeline released below is whole.
	a.swapping.Lock()
	defer a.swapping.Unlock()

	a.cadence.Close()

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
	a.mu.Lock()
	a.forgetUtterancesLocked()
	a.mu.Unlock()
	a.logHeardFromEveryone("end")

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

	// Kept here as well as on the agent so a frame does not wait on the agent's lock.
	tallies := map[string]*hearing{}
	for inbound := range a.options.Edge.Audio() {
		tally, ok := tallies[inbound.Participant.ID]
		if !ok {
			tally = a.heardFrom(inbound.Participant.ID)
			tallies[inbound.Participant.ID] = tally
		}
		tally.heardAudio(inbound.Audio)
		if a.switching.Load() {
			continue
		}
		// A native model hears one stream rather than one per participant.
		if a.native() {
			a.hear(inbound)
			continue
		}
		listener, err := a.listen(inbound.Participant)
		if errors.Is(err, errDeparted) {
			continue
		}
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
		a.attend(attendance)
	}
}

// attend reports one arrival or departure.
func (a *Agent) attend(attendance Attendance) {
	if attendance.Joined {
		a.mu.Lock()
		delete(a.departed, attendance.Participant.ID)
		a.mu.Unlock()
		a.logger.Debug("a participant joined",
			"participant", attendance.Participant.UserID)
		a.emitter.Send(ParticipantJoined{
			Participant: attendance.Participant,
			At:          time.Now(),
		})
		return
	}
	a.unbind(attendance.Participant)
	a.mu.Lock()
	a.departed[attendance.Participant.ID] = time.Time{}
	listener := a.listeners[attendance.Participant.ID]
	a.mu.Unlock()
	if listener != nil {
		// A provider that fills silence never cuts an idle session, so a departed
		// participant's would stay open until the agent closes.
		// Closing waits on the final transcript, so it must not hold up the next roster event.
		a.forgetListener(attendance.Participant.ID, listener)
		a.running.Add(1)
		go func() {
			defer a.running.Done()
			a.closeListener(listener)
		}()
	}
	a.logger.Debug("a participant left", "participant", attendance.Participant.UserID)
	a.emitter.Send(ParticipantLeft{
		Participant: attendance.Participant,
		At:          time.Now(),
	})
}

// errDeparted is listen refusing a participant who has left the call.
var errDeparted = errors.New("agent: participant left")

// departedGrace is how long audio from a participant who left is dropped, counted from the
// first frame dropped rather than from the leave, so a consumer held up opening someone
// else's session (a dial, up to a 15 s handshake: stt/muse/muse.go:246) does not use it up.
// The frames that trail a leave were queued before it: at most audioBuffer (10 chunks of
// 20 ms, streamedge/streamedge.go:41-43) on the channel, one held in each track's Send
// (streamedge/streamedge.go:429) and what the track buffered behind it. consumeEdge hands
// them over back to back, since a dropped frame opens nothing, so they normally come within
// milliseconds of the first. A stall after the first dropped frame (another participant's
// first dial, or a slow ProcessAudio) can outlast the grace: a trailing frame after it
// reopens a session, which then lives until the agent closes, as on base. A second covers
// the normal case with room to spare, and is all a
// participant still in the call loses when the SFU reports a leave that did not happen
// (a migration keeps the session id: getstream-go-webrtc call.go:558-563).
const departedGrace = time.Second

// dropsDeparted reports whether listen must drop a participant's audio because they left, and
// forgets the leave once departedGrace has passed: audio still coming then is somebody
// who is still in the call. Called with a.mu held.
func (a *Agent) dropsDeparted(participantID string) bool {
	dropped, gone := a.departed[participantID]
	if !gone {
		return false
	}
	now := a.now()
	if dropped.IsZero() {
		a.departed[participantID] = now
		return true
	}
	if now.Sub(dropped) < departedGrace {
		return true
	}
	delete(a.departed, participantID)
	return false
}

// listen returns a participant's transcription session, opening one on first hearing them.
func (a *Agent) listen(participant stt.Participant) (*sttrouter.Session, error) {
	a.mu.Lock()
	if a.closed {
		a.mu.Unlock()
		return nil, errors.New("agent: closed")
	}
	if a.dropsDeparted(participant.ID) {
		a.mu.Unlock()
		return nil, errDeparted
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
	// The participant may have left while the session was opening. Closing waits on the
	// final transcript, so it must not hold up everyone else's audio.
	if _, gone := a.departed[participant.ID]; gone {
		a.running.Add(1)
		a.mu.Unlock()
		go func() {
			defer a.running.Done()
			a.closeListener(session)
		}()
		return nil, errDeparted
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
	a.forgetListener(participantID, session)
	a.closeListener(session)
}

// forgetListener takes a session out of use without closing it.
func (a *Agent) forgetListener(participantID string, session *sttrouter.Session) {
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
}

// closeListener releases the provider's socket and ends the stream of events this was
// reading. Doing it twice is safe, which is what lets Close race with this.
func (a *Agent) closeListener(session *sttrouter.Session) {
	if err := session.Close(); err != nil {
		a.logger.Debug("closing a dropped transcriber failed", "error", err)
	}
}

// consumeSTT feeds transcript revisions to the cadence controller.
func (a *Agent) consumeSTT(participantID string, session *sttrouter.Session) {
	defer a.running.Done()
	defer a.dropListener(participantID, session)

	tally := a.heardFrom(participantID)
	partials := sttPartials.WithLabelValues(session.Provider())
	finals := sttFinals.WithLabelValues(session.Provider())
	for event := range session.Events() {
		switch typed := event.(type) {
		case stt.Transcript:
			if strings.TrimSpace(typed.Text) == "" {
				a.logger.Debug("the transcriber sent nothing but silence",
					"provider", typed.Provider, "participant", typed.Participant.ID, "mode", typed.Mode)
				continue
			}
			tally.heardWords(typed, partials, finals)
			a.logger.Debug("transcribed",
				"provider", typed.Provider, "model", typed.Model,
				"participant", typed.Participant.ID, "mode", typed.Mode, "text", typed.Text,
				"confidence", typed.Confidence, "latency_ms", typed.ProcessingTimeMs)
			a.mu.Lock()
			a.lastHeardAt = time.Now()
			a.lastParticipant = typed.Participant
			a.mu.Unlock()
			a.act(a.converse.Observe(typed, a.floor()))

		case stt.Connected:
			a.logger.Info("listening", "provider", typed.Provider, "model", typed.Model)

		case stt.Failover:
			// What the participant said as the vendor refused is lost; what they say next
			// is heard by the next model.
			a.logger.Warn("the transcriber refused the session, the next model took it over",
				"from", typed.From, "provider", typed.Provider, "model", typed.Model, "error", typed.Err)
			partials = sttPartials.WithLabelValues(typed.Provider)
			finals = sttFinals.WithLabelValues(typed.Provider)

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

// consumeCadence puts a turn to the conversation once its words have held still.
func (a *Agent) consumeCadence(p *pipeline) {
	defer p.running.Done()

	for {
		select {
		case ready := <-a.cadence.Ready():
			a.mu.Lock()
			gone := a.closed || a.harness == nil
			a.mu.Unlock()
			if gone {
				continue
			}
			a.logHeard("turn", ready.Participant.ID, "turn", ready.ID)
			a.act([]Action{a.converse.Settled(ready, a.floor())})
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
	a.mu.Lock()
	state := floor{
		Quiet:           !a.interrupting && a.recoveringVoice == nil && a.utterances == 0 && !a.generating && a.pendingTools == 0,
		Talking:         a.interrupting || a.recoveringVoice != nil || a.utterances > 0 || a.generating,
		Recovering:      a.recoveringVoice != nil,
		Owed:            a.toolReply,
		Composing:       a.composing,
		Speaking:        a.speakingTurn,
		LastSpokeAt:     a.lastSpokeAt,
		LastHeardAt:     a.lastHeardAt,
		LastParticipant: a.lastParticipant,
		Asked:           owesAnAnswer(lastAssistantSaid(a.history)),
	}
	// A reply cut off, or followed by a murmur or a tool's reply, is no reply to drop, and
	// nor is one that ended with nothing said: the agent is quiet again.
	if !state.Quiet && a.unheard.turnID == a.speakingTurn {
		state.Unheard = a.unheard
	}
	tools := slices.Sorted(maps.Values(a.toolsRunning))
	current := a.harness
	a.mu.Unlock()

	var skills []string
	if current != nil {
		state.Owed = state.Owed || current.Pending()
		skills = current.Working()
	}
	state.Working = workingOn(skills, tools)
	return state
}

// workingOn names what the caller is waiting on, for telling them it is still going: the
// first skill at work, else the first tool, empty when neither is running.
func workingOn(skills, tools []string) string {
	if len(skills) > 0 {
		return "the " + strings.ReplaceAll(skills[0], "_", " ")
	}
	if len(tools) == 0 {
		return ""
	}
	name := tools[0]
	if _, after, found := strings.Cut(name, "__"); found {
		name = after
	}
	return "the " + strings.ReplaceAll(name, "_", " ") + " lookup"
}

// act carries out what the conversation decided, in the order it decided it.
func (a *Agent) act(actions []Action) {
	for _, action := range actions {
		a.perform(action)
	}
}

// perform carries out one decision. Every branch here is mechanical: which provider to
// touch and in what order. Why any of it is happening was settled in converse.
func (a *Agent) perform(action Action) {
	switch action.Kind {
	case ActBackchannel:
		a.backchannel(action.Participant, action.Text)

	case ActCheckIn:
		a.checkIn(action.Participant, action.Compose)

	case ActSupersede:
		a.dropSpeculation(action.TurnID)
		if err := a.harness.CancelDecision(action.TurnID); err != nil {
			a.fail(err, "flow")
		}

	case ActAsk:
		a.ask(action.Candidate)

	case ActInterrupt:
		if action.Unheard {
			a.dropUnheard(action.Participant, action.TurnID)
			return
		}
		// Work the interrupted reply handed over goes on: talking over the agent is not
		// taking back what was asked, and the model drops it when the caller did.
		a.dropSpeculations()
		a.stopTools()
		a.interrupt(action.Participant)

	case ActShorten:
		a.shorten()

	case ActAnswer:
		a.mu.Lock()
		missed := strings.TrimSpace(a.missedLocked(action.Candidate.Participant) + " " + action.Unanswered)
		a.unheard = unheardReply{turnID: action.Candidate.ID, participant: action.Candidate.Participant.ID,
			decidedAt: time.Now(), said: newestWords(strings.TrimSpace(missed+" "+action.Candidate.Said()), missedWords)}
		a.mu.Unlock()
		note := joinNotes(action.Clarify, missedNote(missed))
		if a.adoptSpeculation(action.Candidate, note) {
			return
		}
		if err := a.respondCandidate(action.Candidate, note); err != nil {
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
	participant := ready.Participant.Name
	if participant == "" {
		participant = a.speakerLocked(ready.Participant).label
	}
	a.mu.Unlock()

	// Speech the voice has finished sending is still on its way out of the edge, so the
	// agent counts as speaking until it has drained.
	speaking = speaking || a.speechPending()

	if a.speculate(current, ready, speaking, anotherVoice) {
		return
	}
	if err := current.Decide(harness.FlowTurn{
		ID:           ready.ID,
		Instructions: instructions,
		History:      history,
		Participant:  participant,
		Text:         ready.Said(),
		Speaking:     speaking,
		Reply:        reply,
		Unfinished:   ready.Unfinished,
		AnotherVoice: anotherVoice,
		Waited:       a.converse.waitedOn(ready),
		Ended:        ready.Ended,
		InParts:      a.converse.inParts(ready.Participant.ID),
	}); err != nil {
		a.dropSpeculation(ready.ID)
		a.converse.Unasked(ready.ID)
		a.fail(err, "flow")
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
	// typed is a turn written rather than spoken, such as the instruction opening a call.
	typed bool
}

func lastAssistantSaid(history []llm.Message) string {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == llm.Assistant && strings.TrimSpace(history[i].Content) != "" {
			return history[i].Content
		}
	}
	return ""
}

// owesAnAnswer reports whether the agent's last words asked the caller something they owe an
// answer to. "Anything else?" is an invitation, not a question left open: a caller who says
// nothing to it is done, and checking on them seconds later sounds like the agent is
// rushing them.
func owesAnAnswer(said string) bool {
	said = strings.ToLower(strings.TrimSpace(said))
	if strings.Contains(said, "anything else") {
		return false
	}
	if strings.HasSuffix(said, "?") {
		return true
	}
	last := said
	if cut := strings.LastIndexAny(strings.TrimRight(said, ".!"), ".!?"); cut >= 0 {
		last = said[cut+1:]
	}
	for _, request := range requests {
		if strings.Contains(last, request) {
			return true
		}
	}
	return false
}

// requests are how a reply that ends on a full stop still asks for something: "To get
// started, I'll need your date of birth."
var requests = []string{
	"i'll need", "i will need", "i need your", "i just need", "please tell me", "please provide",
	"please give me", "please share", "please confirm", "please spell", "please read",
	"could you", "can you", "may i have", "go ahead with your",
}

// turnStamp names a turn. The clock is enough: a conversation cannot produce two turns
// in the same nanosecond.
func turnStamp() string { return strconv.FormatInt(time.Now().UnixNano(), 10) }

// respond asks the harness to reply to a turn.
func (a *Agent) respond(participant stt.Participant, text string, listened heard, images []llm.ImagePart) error {
	return a.respondTurn(replyPrefix+turnStamp(), participant, text, listened, "", images, options.LLM{}, routing.Caller{}, false)
}

// respondCandidate answers a settled turn. The note is what the conversation decided the
// model should know beyond the words, and is empty on a turn that is simply answered.
func (a *Agent) respondCandidate(ready candidate, note string) error {
	return a.respondTurn(ready.ID, ready.Participant, ready.Said(), heard{
		at:           ready.ReadyAt,
		revisedAt:    ready.RevisedAt,
		sttLatencyMs: ready.STTLatencyMs,
		confidence:   ready.Confidence,
	}, note, nil, options.LLM{}, routing.Caller{}, false)
}

// noteToolDone records that one of the tools the current turn asked for has returned. A
// call already answered with stillRunning stopped counting when it was.
func (a *Agent) noteToolDone(hold *toolHold) {
	a.mu.Lock()
	if a.pendingTools > 0 && (hold == nil || !hold.answered) {
		a.pendingTools--
	}
	a.mu.Unlock()
}

// queueToolReply starts a turn that says what the tools came back with, or leaves it
// pending if a generate is already in flight or more tools are still running.
// Two tools in one reply used to each start a generate, and the second stole speakingTurn
// so the first result was never said.
func (a *Agent) queueToolReply() {
	a.mu.Lock()
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
	// follow can take the result between queueToolReply letting go of the lock and this
	// taking it, and a second turn on it is the caller answered twice.
	if !a.toolReply || a.generating || a.interrupting || a.recoveringVoice != nil {
		a.mu.Unlock()
		return nil
	}
	history := a.replayLocked()
	participant := a.lastParticipant
	a.speakingTurn = turnID
	a.generating = true
	a.toolReply = false
	a.toolRounds++
	answers := !a.options.Text && a.toolRounds >= maxToolRounds
	continues := a.owedTurn
	a.owedTurn = ""
	instructions := a.instructions()
	asked := a.asked
	a.mu.Unlock()

	a.turns.started(turnID, kindTool)
	a.turns.begin(turnID, participant, time.Now(), time.Time{}, 0)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Continues: continues})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		AfterTool:    true,
		Answers:      answers,
		Asked:        asked,
	}, "")
}

func (a *Agent) respondTurn(
	turnID string,
	participant stt.Participant,
	text string,
	listened heard,
	note string,
	images []llm.ImagePart,
	asked options.LLM,
	sender routing.Caller,
	scheduled bool,
) error {
	a.mu.Lock()
	if a.harness == nil {
		a.mu.Unlock()
		return stack.Wrap(errors.New("agent: not joined"))
	}
	a.keepTurnLocked(participant, text)
	history := a.replayLocked()

	a.speakingTurn = turnID
	a.generating = true
	a.toolRounds = 0
	a.promiseChased = false
	a.chaseForced = false
	a.narrationAsked = false
	a.refusalChased = false
	a.refusedWithCalls = false
	a.callerTurns++
	a.stepChased = ""
	a.toolReply = false
	a.asked = asked
	a.sender = sender
	a.scheduled = scheduled
	a.lastParticipant = participant
	if !listened.typed {
		a.heardCaller = true
	}
	// A call opened with a typed instruction ("greet the caller") has nothing to act on
	// yet, and a model offered tools then logs a summary of the empty call.
	noTools := !a.options.Text && !a.heardCaller
	instructions := a.instructions()
	a.mu.Unlock()

	a.turns.started(turnID, kindCaller)
	a.turns.begin(turnID, participant, listened.at, listened.revisedAt, listened.sttLatencyMs)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Prompt: text})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		Note:         joinNotes(note, a.duplex.Note(listened.confidence)),
		Images:       images,
		Asked:        asked,
		NoTools:      noTools,
	}, text)
}

// typedCaller is the participant RespondTo answers as. Typed words are the caller's own,
// so they never count as a second person in the call.
const typedCaller = "caller"

// severalSpeakers is what the model is told once a second person has spoken. Their names
// come ahead of their words, and a reply is spoken, so one that opened on a name the way
// their turns do would be heard saying it.
const severalSpeakers = "Several people are talking in this call. Each of their messages starts with " +
	"the name of who said it, then a colon. Never start your reply with such a label."

// userTurnLocked is a participant's words as a turn of the conversation. It names who
// said them once somebody else in the call has had a turn answered too.
func (a *Agent) userTurnLocked(participant stt.Participant, text string) llm.Message {
	turn := llm.Message{Role: llm.User, Content: text}
	_, known := a.speakers[personOf(participant)]
	if participant.ID != typedCaller && len(a.speakers) > 0 && !(known && len(a.speakers) == 1) {
		turn.Speaker = a.speakerLocked(participant).label
	}
	return turn
}

// keepTurnLocked adds a turn being answered to the conversation and counts who said it.
// What tools showed the model before it stand as a note from here on (toolPicturesSeen).
func (a *Agent) keepTurnLocked(participant stt.Participant, text string) {
	a.history = append(toolPicturesSeen(a.history), a.userTurnLocked(participant, text))
	if a.missed.participant == participant.ID {
		a.missed = unheardReply{}
	}
	if participant.ID != typedCaller {
		a.speakers[personOf(participant)] = a.speakerLocked(participant)
	}
}

// nameTag is how a person counted in speakers is labelled.
type nameTag struct {
	// label goes ahead of their words.
	label string
	// fallback is the label they were given for having no display name at their first
	// answered turn. Empty for a person who had one.
	fallback string
}

// speakerLocked is the label a person's words go under. It is fixed by their first
// answered turn, unless that turn had no display name: such people go under "Speaker 1",
// "Speaker 2" and on, in the order they first spoke, until their name is known and
// replaces it. A person whose name another already goes under is numbered, "Alex (2)", in
// the order they first spoke, so the model tells them apart without being told their user
// or participant ID, which are the app's and mean nothing to it.
func (a *Agent) speakerLocked(participant stt.Participant) nameTag {
	person := personOf(participant)
	name := speakerLabel(participant)
	current, ok := a.speakers[person]
	if ok && (name == "" || current.label != current.fallback) {
		return current
	}
	var taken []string
	fallbacks := 0
	for other, otherSpeaker := range a.speakers {
		if other != person {
			taken = append(taken, otherSpeaker.label)
		}
		if otherSpeaker.fallback != "" {
			fallbacks++
		}
	}
	nameless := name == ""
	if nameless {
		// "Speaker N" is the wording Kanat chose on AI-1458 (2026-10-19).
		name = fmt.Sprintf("Speaker %d", fallbacks+1)
	}
	label := name
	for n := 2; slices.Contains(taken, label); n++ {
		label = fmt.Sprintf("%s (%d)", name, n)
	}
	if nameless {
		return nameTag{label: label, fallback: label}
	}
	return nameTag{label: label, fallback: current.fallback}
}

// personOf is who a participant is. A person who rejoins gets a new ID but keeps their
// user, so the user, when there is one, is what is counted.
func personOf(participant stt.Participant) string {
	if participant.UserID != "" {
		return participant.UserID
	}
	return participant.ID
}

// speakerLabel is a participant's display name as it goes ahead of their words, empty
// when they have none. A line break or a colon in a display name could pass for the start
// of another person's line, so they go.
func speakerLabel(participant stt.Participant) string {
	return strings.Join(strings.Fields(strings.ReplaceAll(participant.Name, ":", "")), " ")
}

func (a *Agent) replayLocked() []llm.Message {
	replay := append([]llm.Message(nil), a.history...)
	for i, message := range replay {
		replay[i] = named(message)
	}
	if a.options.Text {
		return replay
	}
	return spokenResults(replay)
}

// named is a turn as the model reads it: who said it ahead of the words, when it says.
func named(message llm.Message) llm.Message {
	if message.Speaker != "" {
		message.Content = message.Speaker + ": " + message.Content
	}
	return message
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
	a.turns.started(turnID, kindBackchannel)

	a.logger.Debug("murmuring while the caller talks",
		"turn", turnID, "participant", participant.ID, "phrase", phrase)
	if err := a.speakWhole(turnID, phrase); err != nil {
		a.fail(err, "tts")
		return
	}
	a.emitter.Send(Backchannel{Participant: participant, Text: phrase})
}

// checkIn says a line the model writes for purpose to a caller who has heard nothing for
// a while: whether there is anything else, or that what they asked for is still coming.
//
// The model takes a moment to write it, so it is written off the loop that decided it.
// A line that cannot be written is not worth failing the turn over: the caller hears
// nothing, which is what they would have heard anyway.
func (a *Agent) checkIn(participant stt.Participant, purpose string) {
	kind := composedKind(purpose)
	if participant.ID == "" {
		// Nobody has been heard in words yet, so the line is for everyone who was heard.
		a.logHeardFromEveryone("checkin", "kind", kind)
	} else {
		a.logHeard("checkin", participant.ID, "kind", kind)
	}
	a.mu.Lock()
	if a.composing || a.tts == nil {
		a.mu.Unlock()
		return
	}
	a.composing = true
	ctx := a.ctx
	a.mu.Unlock()

	asked := time.Now()
	go func() {
		defer func() {
			a.mu.Lock()
			a.composing = false
			a.mu.Unlock()
		}()
		line, err := a.compose(ctx, purpose)
		if err != nil {
			a.logger.Warn("could not write a line to say", "kind", kind, "error", err)
			return
		}
		a.sayComposed(participant, line, purpose, asked)
	}()
}

// sayComposed says a line compose wrote for purpose, asked for at asked, unless the moment
// it was for has passed: anyone spoke since, an answer is owed or on its way, or the work
// an update is about ended, or the silence a check-in is about turned into work.
//
// Unlike a murmur it is a turn the agent took, so it goes into the history and is
// reported as speech: without that, the "no, that was everything" that comes back
// answers a question the model cannot see it asked. Said between a call and its result,
// it joins the history after the result, which a provider insists comes first.
func (a *Agent) sayComposed(participant stt.Participant, line, purpose string, asked time.Time) {
	if line == "" {
		return
	}
	state := a.floor()
	passed := state.Talking || state.Owed || state.LastHeardAt.After(asked) ||
		state.LastSpokeAt.After(asked) || a.converse.Listening()
	switch purpose {
	case updatePurpose:
		passed = passed || state.Working == ""
	case idlePurpose:
		passed = passed || state.Working != ""
	}
	if passed {
		a.logger.Debug("not saying a line written for a moment that passed", "line", line)
		return
	}

	turnID := backchannelPrefix + turnStamp()
	a.mu.Lock()
	if a.tts == nil {
		a.mu.Unlock()
		return
	}
	a.speakingTurn = turnID
	a.saying = line
	said := llm.Message{Role: llm.Assistant, Content: line}
	if callsOpen(a.history) {
		a.lateResults = append(a.lateResults, said)
	} else {
		a.history = append(a.history, said)
	}
	a.mu.Unlock()

	kind := composedKind(purpose)
	a.turns.started(turnID, kind)
	a.logger.Info("saying a line the agent chose", "turn", turnID, "kind", kind, "participant", participant.ID)
	if err := a.speakWhole(turnID, line); err != nil {
		a.fail(err, "tts")
		return
	}
	a.emitter.Send(Responded{TurnID: turnID, Text: line, Purpose: composedFor(purpose)})
}

// composedFor names what a composed line is for, as a watcher is told it.
func composedFor(purpose string) string {
	switch purpose {
	case idlePurpose:
		return PurposeCheckIn
	case updatePurpose:
		return PurposeUpdate
	}
	return ""
}

// composedKind is the kind of turn a line written for purpose is.
func composedKind(purpose string) string {
	switch purpose {
	case holdPurpose:
		return kindHold
	case updatePurpose:
		return kindUpdate
	}
	return kindCheckIn
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
	if a.opened != "" {
		parts = append(parts, fmt.Sprintf("You opened the call by saying: %q", a.opened))
	}
	// How to write for the voice comes last, because it is about the delivery rather than
	// about who the agent is.
	if a.voicePrompt != "" {
		parts = append(parts, a.voicePrompt)
	}
	if len(a.speakers) > 1 {
		parts = append(parts, severalSpeakers)
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
	ctx := a.ctx
	if !a.sender.IsZero() {
		ctx = routing.WithCaller(ctx, a.sender)
	}
	if a.scheduled {
		ctx = routing.WithScheduledTurn(ctx)
	}
	ctx, cancel := context.WithCancel(ctx)
	a.generatingCancel[turn.ID] = cancel
	a.pumps.Add(1)
	a.mu.Unlock()

	go a.startReply(current, turn, ctx, screen)
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

	a.turns.modelStarted(turn.ID, time.Now())
	stream, err := current.Respond(ctx, turn)
	if err != nil {
		if ctx.Err() != nil {
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

	a.pump(turn.ID, stream)
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
	a.mu.Unlock()
	if cancel != nil {
		cancel()
	}
}

// pump drains one reply into the channel the speaking goroutine reads.
func (a *Agent) pump(turnID string, stream *llm.Stream) {
	defer stream.Close()

	if a.options.UnifiedReplies {
		_, from, to := harness.ParseUnified(stream.Lead(harness.UnifiedLead))
		stream.Cut(from, to)
	}
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
	a.claimReply(turnID)
	// What the model wrote is not all meant for the caller: a request for help is
	// addressed to the harness, and is taken out here rather than spoken.
	speech := a.harness.Filter(turnID, delta)
	if speech == "" {
		return
	}
	a.turns.firstText(turnID, time.Now())

	// A stage direction is addressed to the voice, not to the caller: it is taken out of
	// what is read and remembered, and left in only for a voice that can act it. A
	// conversation in writing has no voice to address, so a bracket there is only ever
	// text -- a markdown link, an array literal -- and taking it out corrupts the reply.
	plain := speech
	if !a.options.Text {
		plain = a.messageText(a.directions.Add(speech))
	}
	if plain != "" {
		a.spoken.WriteString(plain)
		a.mu.Lock()
		a.saying = a.spoken.String()
		a.mu.Unlock()
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

	for _, sentence := range a.sentencesOf(speech) {
		if a.withhold(sentence.text) {
			continue
		}
		if err := a.speakSentence(turnID, sentence.text, a.inMessage(sentence)); err != nil {
			a.fail(err, "tts")
		}
	}
}

// withhold reports whether a sentence is kept from the caller because it, or one before it
// in this reply, claims something was done that no tool has done since the caller spoke.
// Gemma wrote "One moment, checking... I've checked the outage map and there are no
// outages" in one breath; chased afterwards, the caller had already heard the invented
// result. Unsaid, the chase that follows makes the call and the result answers instead.
//
// Qwen wrote "Let me look up your appointments. [short pause] I see you have one on
// Thursday": once a step is promised, a direction and everything after it is held back
// the same way, and the chase makes the call the promise was for.
func (a *Agent) withhold(sentence string) bool {
	if a.withheld || a.narrated {
		return true
	}
	if narrates(sentence) {
		a.narrated = true
		a.withheldFrom = sentence
		a.logger.Debug("holding back a reply that talks about the caller", "sentence", sentence)
		return true
	}
	if len(a.availableTools().Tools) == 0 || a.delegating() {
		return false
	}
	pause := a.promised && directionTags.MatchString(sentence)
	if !pause && !claimsDone(sentence) && !promisesStep(sentence) {
		return false
	}
	a.mu.Lock()
	backed := backsClaim(a.history, sentence, toolNames(a.availableTools()))
	ran := toolRanSinceCaller(a.history)
	a.mu.Unlock()
	if claimsDone(sentence) && backed || !claimsDone(sentence) && ran {
		return false
	}
	if pause || claimsDone(sentence) {
		a.withheld = true
		a.withheldFrom = sentence
		a.logger.Debug("holding back what no tool has made true", "sentence", sentence)
		return true
	}
	a.promised = true
	return false
}

// narrates reports whether a sentence speaks of the caller in the third person, which a
// reply to them never does. Nemotron wrote its reasoning into the reply ("The caller said
// yes after I offered two times. This is ambiguous.") and once read out the system's note
// to it ("The caller is listening, not reading: …").
func narrates(sentence string) bool {
	lower := strings.TrimSpace(spokenLower(directionTags.ReplaceAllString(sentence, "")))
	return slices.ContainsFunc(narrations, func(opening string) bool { return strings.HasPrefix(lower, opening) })
}

var narrations = []string{"the caller ", "the caller's ", "the user ", "the user's ", "caller said", "user said"}

// spokenLower lowercases a reply for matching with straight apostrophes: Qwen writes
// "I’ll reschedule".
func spokenLower(said string) string {
	return strings.ReplaceAll(strings.ToLower(said), "’", "'")
}

// promisesStep reports whether a sentence tells the caller a lookup is under way.
func promisesStep(sentence string) bool {
	lower := spokenLower(sentence)
	return slices.ContainsFunc(holdWords, func(hold string) bool { return strings.Contains(lower, hold) }) ||
		slices.ContainsFunc(actionPromises, func(promise string) bool { return strings.Contains(lower, promise) }) ||
		slices.ContainsFunc(stepWords, func(step string) bool { return strings.Contains(lower, step) })
}

// stepWords are what a sentence starting a step says beyond asking the caller to wait.
var stepWords = []string{
	"let me verify", "let me confirm your", "let me pull", "let me find", "i'll check",
	"i will check", "i'll look", "i'll verify", "i'll pull",
}

// heardOf cuts a reply's text where withholding began, so the history holds what the
// caller heard.
func heardOf(said, from string) string {
	if from != "" {
		plain := strings.TrimSpace(directionTags.ReplaceAllString(from, ""))
		if i := strings.Index(said, plain); plain != "" && i >= 0 {
			return strings.TrimSpace(said[:i])
		}
	}
	return unsaid(said)
}

var directionTags = regexp.MustCompile(`\[[^\]]*\]`)

// unsaid cuts a reply's text where a withheld claim begins, so the history holds what the
// caller heard.
func unsaid(said string) string {
	lower := spokenLower(said)
	cut := len(said)
	for _, claim := range doneClaims {
		if i := strings.Index(lower, claim); i >= 0 && i < cut {
			cut = i
		}
	}
	if at := doneClaim.FindStringIndex(lower); at != nil && at[0] < cut {
		cut = at[0]
	}
	if cut == len(said) {
		return said
	}
	start := strings.LastIndexAny(said[:cut], ".!?…")
	if start < 0 {
		return ""
	}
	return strings.TrimSpace(said[:start+1])
}

// finish closes out a reply the caller heard.
func (a *Agent) finish(response llm.Response) {
	a.claimReply(response.ID)
	// Text the harness was holding on the chance it began a request for help was only
	// ever text, so it is spoken.
	tail := a.harness.Flush()
	// A direction the stripper was still holding is released the same way: unfinished, it
	// was only ever text. In writing the stripper never ran, so there is nothing held.
	plain := tail
	if !a.options.Text {
		plain = a.messageText(a.directions.Add(tail) + a.directions.Flush())
	}
	a.spoken.WriteString(plain)
	if !a.performs {
		tail = plain
	}
	// Taken before resetTurn forgets them: a skill tag that named a tool is a call this
	// turn made, and has to be on the history the result answers.
	asked := a.harness.TakeAsked()
	calls := withSpokenNumbers(append(append([]llm.ToolCall(nil), response.ToolCalls...), asked...))
	hold := false
	if a.options.Text {
		// There is no voice to release it to, so the held text is reported as the last
		// of the reply. Without this a reader would be missing whatever the harness was
		// still deciding about when the model stopped.
		if plain != "" {
			a.emitter.Send(ResponseDelta{TurnID: response.ID, Text: plain})
		}
	} else {
		for _, sentence := range a.sentencesOf(tail) {
			if a.withhold(sentence.text) {
				continue
			}
			if err := a.speakSentence(response.ID, sentence.text, a.inMessage(sentence)); err != nil {
				a.fail(err, "tts")
			}
		}
		// Whatever did not end in punctuation is still worth saying.
		if remainder := a.chunk.Flush(); remainder.text != "" && !a.withhold(remainder.text) {
			if err := a.speakSentence(response.ID, remainder.text, a.inMessage(remainder)); err != nil {
				a.fail(err, "tts")
			}
		}
		// A model that reaches for a tool without a word leaves the caller listening to
		// nothing until it comes back, which on a phone is indistinguishable from having
		// been cut off. Prompting for it is not enough: the models that do it reliably
		// are not the ones fast enough to hold a conversation. What the operator asked to
		// be said is said at once; anything else is written for what the caller asked,
		// once the work is handed over.
		if a.fillsPause(response.ID, calls) && strings.TrimSpace(a.spoken.String()) == "" {
			if filler := a.preSpeech(calls); filler != "" {
				if err := a.speakSentence(response.ID, filler, a.appendSpoken(filler)); err != nil {
					a.fail(err, "tts")
				}
			} else {
				hold = true
			}
		}
		// A reply that failed before a word of it was said leaves the caller waiting for
		// one that is not coming, and after "one moment" that silence is the answer lost.
		if response.Status == llm.StatusFailed && strings.TrimSpace(a.spoken.String()) == "" {
			if err := a.speakSentence(response.ID, lostReply, a.appendSpoken(lostReply)); err != nil {
				a.fail(err, "tts")
			}
		}
		if err := a.closeUtterance(response.ID); err != nil {
			a.fail(err, "tts")
		}
	}
	// How many syntheses the turn produces is only settled once the reply is, and it is
	// what tells the tracker when the turn has finished being spoken.
	a.turns.completed(response.ID, response.TimeToFirstTokenMs, a.expectedSyntheses(response.ID))
	said := strings.TrimSpace(a.spoken.String())
	withheld, narrated := a.withheld, a.narrated
	if withheld || narrated {
		said = heardOf(said, a.withheldFrom)
	}
	a.resetTurn()

	a.mu.Lock()
	a.generating = false
	// A reply that only called a tool is still a turn the model took, and it has to be
	// recorded with the calls on it: the result sent back answers one of them, and a
	// provider refuses a conversation where it answers nothing.
	if said != "" || len(calls) > 0 {
		a.history = append(a.history, llm.Message{
			Role:      llm.Assistant,
			Content:   said,
			ToolCalls: calls,
			Reasoning: response.ReasoningText,
		})
	}
	exchange := lastExchange(a.history)
	history := append([]llm.Message(nil), a.history...)
	currentHarness := a.harness
	a.mu.Unlock()

	// A provider that kept this reply can be asked to carry on from it next turn rather
	// than read the conversation again.
	if currentHarness != nil {
		currentHarness.Remember(response)
	}
	a.learnTerms(said, history)

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
	if hold {
		a.mu.Lock()
		participant := a.lastParticipant
		a.mu.Unlock()
		a.checkIn(participant, holdPurpose)
	}
	a.mu.Lock()
	forced, forcedNote := a.forcedChase == response.ID, a.forcedNote
	if forced {
		a.forcedChase = ""
	}
	a.mu.Unlock()
	if forced && len(calls) == 0 && said == "" {
		// Constrained to a call, Gemma on vLLM sometimes writes only whitespace; asked again
		// with the note alone it at least answers.
		a.mu.Lock()
		a.promiseChased = false
		a.mu.Unlock()
		a.keepPromise(response.ID, forcedNote, false)
	}
	if narrated && said == "" && len(calls) == 0 && !forced {
		// Nemotron met "Yes, please go ahead" with its reasoning and the caller was left
		// with nothing. A reply nobody heard is asked again whatever was chased before it,
		// once per caller turn.
		a.mu.Lock()
		again := !a.narrationAsked
		if again {
			a.narrationAsked, a.promiseChased = true, false
		}
		a.mu.Unlock()
		if again {
			a.keepPromise(response.ID, speakToTheCaller, false)
		}
	}
	logger := loggingTool(a.availableTools())
	if logger != "" && len(calls) > 0 && refuses(said) && !slices.ContainsFunc(calls, func(call llm.ToolCall) bool {
		return call.Name == logger
	}) {
		// MiniMax refused a bill credit while opening a ticket, and the hand-off it owed
		// was never chased.
		a.mu.Lock()
		a.refusedWithCalls = true
		a.mu.Unlock()
	}
	a.mu.Lock()
	refusedEarlier := a.refusedWithCalls
	a.mu.Unlock()
	if logger != "" && len(calls) == 0 && !forced && (said != "" && refuses(said) || refusedEarlier) &&
		!calledInCall(history, logger) {
		// Told to refuse an oxycodone refill, offer a callback and log an escalation, Nemotron
		// Bolt and PhoneLLM Mini refused and offered in every run, and never logged.
		a.mu.Lock()
		again := !a.refusalChased
		a.refusalChased = true
		a.mu.Unlock()
		if again {
			a.keepPromise(response.ID, fmt.Sprintf(logTheRefusal, logger), true)
		}
	}
	// Work handed to the subagent is the promise being kept, and asking again supersedes it.
	if !a.options.Text && len(calls) == 0 && !a.delegating() && response.Status == llm.StatusCompleted &&
		len(a.availableTools().Tools) > 0 {
		backed := backsClaim(history, said, toolNames(a.availableTools()))
		step := stepNamed(history, a.availableTools().Tools)
		owes := withheld || owesACall(said, backed)
		if owes && strings.HasPrefix(response.ID, toolPrefix) {
			// Qwen met the chase with "Let me check for any outages. One moment, please."
			// and no call, twice over: a chase that still promises is asked once more, with
			// the call required.
			a.mu.Lock()
			again := a.promiseChased && !a.chaseForced && !changedSinceCaller(history, anyTool)
			if again {
				a.promiseChased, a.chaseForced = false, true
			}
			a.mu.Unlock()
			withheld = withheld || again
		}
		switch {
		case withheld:
			a.keepPromise(response.ID, keepPromise, true)
		case step != "":
			// Made to call after dispatch_tech failed for want of a reboot, Qwen called
			// dispatch_tech again, so the chase offers the named step alone.
			a.chase(response.ID, fmt.Sprintf(doTheStep, step), true, step)
		case owesACall(said, backed):
			// A hold phrase can come before the caller has given what the call needs, and
			// a forced call is then made with blanks; a claim that it is done cannot.
			a.keepPromise(response.ID, keepPromise, !backed && claimsDone(said))
		case !changedSinceCaller(history, anyTool) && wentAhead(history) && !doneBeforeGoAhead(history):
			// A go-ahead to "you're booked" has nothing left to do: forced to call, Qwen
			// ordered from the menu.
			a.keepPromise(response.ID, actOnGoAhead, true)
		default:
			// A step is chased even after another promise was: in Voicebench the chase that
			// made the outage check used that up, and the reboot it led to went uncalled.
			// Each step is chased once, so a model that still describes it is left be.
			step := describedStep(said, a.availableTools(), history)
			a.mu.Lock()
			fresh := step != "" && a.stepChased != step
			if fresh {
				a.stepChased = step
				a.promiseChased = false
			}
			a.mu.Unlock()
			if fresh {
				a.keepPromise(response.ID, fmt.Sprintf(callTheStep, step), true)
			}
		}
	}
	a.respondQueued()
	// A note that landed while this reply was being written waited for it to finish.
	a.followUp()
}

// keepPromise asks the model once more, with its tools, after a reply that told the caller
// it was checking, or that something was done, and called nothing: Gemma did both in
// Voicebench, and the caller waited on a lookup, or left with a booking, that was never
// made. It is asked once per thing the caller said, so a model that still calls nothing is
// left to the caller's next words.
func (a *Agent) keepPromise(promised, note string, mustCall bool) {
	a.chase(promised, note, mustCall, "")
}

// chase asks again for the reply to promised, held to the one tool only names if any.
func (a *Agent) chase(promised, note string, mustCall bool, only string) {
	a.chaseAfter(promised, note, mustCall, only, 0)
}

// chaseAfter is chase on its waited-th try. A caller still heard puts the chase off rather
// than dropping it: on a quiet line the energy gate kept hearing the hiss, every chase
// was dropped, and the caller who said "please go ahead" waited on a reschedule that
// never came. A reply begun meanwhile supersedes it; a turn ignored as the TV does not.
func (a *Agent) chaseAfter(promised, note string, mustCall bool, only string, waited int) {
	a.mu.Lock()
	if a.promiseChased || a.harness == nil || a.closed || a.generating || a.interrupting ||
		a.recoveringVoice != nil || a.toolRounds >= maxToolRounds {
		a.logger.Debug("a chase was dropped", "turn", promised, "only", only, "chased", a.promiseChased,
			"generating", a.generating, "interrupting", a.interrupting, "recovering", a.recoveringVoice != nil,
			"rounds", a.toolRounds, "waited", waited)
		a.mu.Unlock()
		return
	}
	if waited < chaseRetries && a.converse.Listening() {
		replying := a.speakingTurn
		a.mu.Unlock()
		time.AfterFunc(chaseRetry, func() {
			a.mu.Lock()
			moved := a.speakingTurn != replying
			a.mu.Unlock()
			if !moved {
				a.chaseAfter(promised, note, mustCall, only, waited+1)
				return
			}
			a.logger.Debug("a chase gave way to a reply", "turn", promised, "only", only)
		})
		return
	}
	a.promiseChased = true
	turnID := toolPrefix + turnStamp()
	if mustCall {
		a.forcedChase, a.forcedNote = turnID, note
	}
	history := a.replayLocked()
	participant := a.lastParticipant
	a.speakingTurn = turnID
	a.generating = true
	instructions := a.instructions()
	asked := a.asked
	a.mu.Unlock()

	a.logger.Info("the reply owed a call and made none, asking for it",
		"turn", turnID, "chases", promised, "must_call", mustCall, "only", only)
	a.turns.started(turnID, kindChase)
	a.turns.begin(turnID, participant, time.Now(), time.Time{}, 0)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Continues: promised})
	if err := a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		AfterTool:    true,
		Note:         note,
		// Told in a note alone, Gemma met it with "Is there anything else?" and the
		// booking it had announced was never made.
		MustCall: mustCall,
		OnlyTool: only,
		Asked:    asked,
	}, ""); err != nil {
		a.fail(err, "llm")
	}
}

// chaseRetry and chaseRetries bound how long a chase waits on a caller still heard.
const (
	chaseRetry   = 500 * time.Millisecond
	chaseRetries = 6
)

// keepPromise is the note a reply chasing up an unkept promise is given.
const keepPromise = "You just told the caller you were checking, or that something was done, " +
	"but called no tool. Call the tool that does it now, without saying anything first, " +
	"filling every argument from what the caller has already said."

// actOnGoAhead is the note a reply that met a go-ahead with no call is chased up with.
const actOnGoAhead = "The caller just told you to go ahead. Call the tool that does what they " +
	"agreed to now, with what you have, without asking for anything optional first."

// logTheRefusal is the note a refusal is chased with when nothing has been logged in the call.
const logTheRefusal = "You just refused the caller. If your instructions say to log, escalate " +
	"or hand off what you refused, call %s now with the reason, then tell the caller. If they " +
	"do not, answer the caller's last words again."

// refuses reports whether said turns the caller down.
func refuses(said string) bool {
	lower := spokenLower(said)
	return refusedAction.MatchString(lower) ||
		slices.ContainsFunc(refusals, func(refusal string) bool { return strings.Contains(lower, refusal) })
}

// refusedAction is "can't" or "cannot" before something the caller asked to have done:
// PhoneLLM Max said "I can't credit a bill under a threat".
var refusedAction = regexp.MustCompile(`\b(can't|cannot|can not|won't be able to)\s+(credit|refill|change|apply|skip|send|process|waive|override|approve|authorize|refund|prescribe|give you|do that)\b`)

var refusals = []string{
	"i can't refill", "i cannot refill", "i'm not able to", "i am not able to", "i'm unable to",
	"i am unable to", "i can't skip", "i cannot skip", "i'm not allowed", "i can't do that",
	"i cannot do that", "i can't send", "i cannot send", "i can't process", "i cannot process",
	"can't change your", "cannot change your", "can't apply", "cannot apply", "isn't eligible",
	"is not eligible", "aren't eligible", "are not eligible",
}

// loggingTool is the tool that logs, escalates or hands the call off, when there is one:
// log_escalation, or create_transfer_summary, which no model called for a caller who
// threatened to cancel after being refused a credit.
func loggingTool(tools harness.Tools) string {
	for _, tool := range tools.Tools {
		name := strings.ToLower(tool.Name)
		if strings.HasPrefix(name, "log_") || slices.ContainsFunc(handOffs, func(word string) bool { return strings.Contains(name, word) }) {
			return tool.Name
		}
	}
	return ""
}

var handOffs = []string{"escalat", "transfer_summary", "handoff", "hand_off"}

// calledInCall reports whether tool has answered without failing anywhere in the call.
func calledInCall(history []llm.Message, tool string) bool {
	names := map[string]string{}
	for _, message := range history {
		for _, call := range message.ToolCalls {
			names[call.ID] = call.Name
		}
	}
	return slices.ContainsFunc(history, func(message llm.Message) bool {
		return message.Role == llm.ToolResult && names[message.ToolCallID] == tool && !failedResult(message.Content)
	})
}

// speakToTheCaller is the note a reply that only talked about the caller is asked again with.
const speakToTheCaller = "Your last reply described the caller instead of speaking to them, and " +
	"they heard none of it. Answer them now, speaking to them directly."

// callTheStep is the note a reply that described a tool's step instead of calling it is
// chased up with.
const callTheStep = "You just described the step %s does but did not call it. Call %[1]s " +
	"now, without saying anything first, even if the caller says they already tried it."

// describedStep is the tool a reply described the step of instead of calling it, or empty.
// In Voicebench Gemma said "Let's try walking through a reboot of your gateway" with
// walk_reboot offered and never called, so the reboot the ticket depended on was never
// recorded. A tool counts as described when every word of its name of four letters or more
// opens a word of the reply, and there are at least two. A reply that asks the caller for
// something is not one: what the call needs may still be missing. "I'll need to walk you
// through a reboot" asks for nothing, so owesAnAnswer's "i'll need" is too broad here.
func describedStep(said string, tools harness.Tools, history []llm.Message) string {
	lower := strings.ToLower(strings.TrimSpace(said))
	if strings.HasSuffix(lower, "?") || slices.ContainsFunc(asks, func(ask string) bool { return strings.Contains(lower, ask) }) {
		return ""
	}
	called := map[string]bool{}
	for _, message := range history {
		for _, call := range message.ToolCalls {
			called[call.Name] = true
		}
	}
	words := strings.FieldsFunc(strings.ToLower(said), func(r rune) bool {
		return !unicode.IsLetter(r)
	})
	for _, tool := range tools.Tools {
		if called[tool.Name] {
			continue
		}
		var parts []string
		for _, part := range strings.Split(strings.ToLower(tool.Name), "_") {
			if len(part) >= 4 {
				parts = append(parts, part)
			}
		}
		if len(parts) < 2 {
			continue
		}
		if !slices.ContainsFunc(parts, func(part string) bool {
			return !slices.ContainsFunc(words, func(word string) bool { return strings.HasPrefix(word, part) })
		}) {
			return tool.Name
		}
	}
	return ""
}

// asks are what a reply asking the caller for a value says.
var asks = []string{
	"need your", "could you", "can you", "may i have", "please tell", "please give",
	"please provide", "please confirm", "please spell", "please read",
}

// doTheStep is the note a reply that met a failed call naming another tool's step with no
// call is chased up with.
const doTheStep = "The tool you called failed and said what has to happen first, which your " +
	"%s tool does. Call it now, without saying anything first."

// stepNamed returns the tool whose step the last tool result, a failure, says has to come
// first ("reboot the gateway before dispatching a truck" names walk_reboot), or empty. Told
// to call that tool and try again, Gemma and Qwen both asked the caller about it instead.
func stepNamed(history []llm.Message, tools []harness.Tool) string {
	if len(history) == 0 || history[len(history)-1].Role != llm.ToolResult {
		return ""
	}
	result := history[len(history)-1]
	lower := strings.ToLower(result.Content)
	if !strings.Contains(lower, "error") && !strings.Contains(lower, "did not work") {
		return ""
	}
	// The guidance the result is replayed with names tools of its own.
	if end := strings.LastIndex(lower, "}"); end >= 0 {
		lower = lower[:end+1]
	}
	failed := ""
	if call := lastCall(history); call != nil {
		failed = call.Name
	}
	for _, tool := range tools {
		if tool.Name != failed && strings.Contains(lower, strings.ToLower(tool.Name)) {
			return tool.Name
		}
	}
	for _, tool := range tools {
		if tool.Name == failed {
			continue
		}
		for word := range strings.SplitSeq(strings.ToLower(tool.Name), "_") {
			// "identity not verified" names verify_account by its stem.
			if len(word) >= 5 && strings.Contains(lower, word[:5]) {
				return tool.Name
			}
		}
	}
	return ""
}

// lastCall is the most recent call the agent made in history, or nil.
func lastCall(history []llm.Message) *llm.ToolCall {
	for i := len(history) - 1; i >= 0; i-- {
		if calls := history[i].ToolCalls; len(calls) > 0 {
			return &calls[len(calls)-1]
		}
	}
	return nil
}

// doneBeforeGoAhead reports whether the agent's last words before the caller's told them
// the work was done.
func doneBeforeGoAhead(history []llm.Message) bool {
	caller := -1
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role == llm.User {
			caller = i
			break
		}
	}
	for i := caller - 1; i >= 0; i-- {
		if history[i].Role == llm.User {
			return false
		}
		if history[i].Role == llm.Assistant && strings.TrimSpace(history[i].Content) != "" {
			return claimsDone(history[i].Content)
		}
	}
	return false
}

// wentAhead reports whether the caller's last words told the agent to go ahead.
func wentAhead(history []llm.Message) bool {
	for i := len(history) - 1; i >= 0; i-- {
		if history[i].Role != llm.User {
			continue
		}
		said := strings.ToLower(history[i].Content)
		return slices.ContainsFunc(goAheads, func(phrase string) bool { return strings.Contains(said, phrase) })
	}
	return false
}

// goAheads are what a caller says to have the agent act.
var goAheads = []string{"go ahead", "please do", "book it", "do it", "proceed", "yes please", "yes, please"}

// owesACall reports whether a reply that called nothing said what only a tool could make
// true: a last sentence that asks the caller to wait on a lookup, or, when no tool has run
// since the caller last spoke (backed), that something has been done.
func owesACall(said string, backed bool) bool {
	if !backed && claimsDone(said) {
		return true
	}
	last := strings.TrimRight(spokenLower(said), " .…!\n")
	if last == "" {
		return false
	}
	// "One moment, checking. … Is that correct?" still owes the check it put the caller on
	// hold for.
	for _, hold := range holdOpenings {
		if strings.HasPrefix(last, hold) {
			return true
		}
	}
	if strings.HasSuffix(last, "?") {
		return false
	}
	for _, end := range []string{". ", "! ", "? ", "… "} {
		if i := strings.LastIndex(last, end); i >= 0 {
			last = last[i+len(end):]
		}
	}
	for _, hold := range holdWords {
		if strings.Contains(last, hold) {
			return true
		}
	}
	return slices.ContainsFunc(actionPromises, func(promise string) bool { return strings.Contains(last, promise) }) ||
		slices.ContainsFunc(fetchPromises, func(promise string) bool { return strings.Contains(last, promise) })
}

// fetchPromises are what a sentence telling the caller a record is about to be pulled says.
// Nemotron ended "You're verified, Maya. Let me pull up your appointments now." and made no
// call. "Let me verify" is not one: after a read-back it can come before the caller agrees.
var fetchPromises = []string{"let me pull", "let me find", "i'll pull", "i will pull", "i'll look", "i will look"}

// actionPromises are what a sentence telling the caller a change is about to be made says.
// Qwen ended "I will reschedule your appointment to Tuesday morning." and made no call.
var actionPromises = []string{
	"i'll book", "i will book", "i'll reschedule", "i will reschedule", "i'll move", "i will move",
	"i'll update", "i will update", "i'll change", "i will change", "i'll cancel", "i will cancel",
	"i'll go ahead", "i will go ahead", "i'll open", "i will open", "i'll create", "i will create",
	"i'll dispatch", "i will dispatch", "i'll schedule", "i will schedule", "i'll reserve",
	"let me book", "let me reschedule", "let me move", "let me update", "let me change",
	"let me cancel", "let me open", "let me create", "let me schedule", "let me reserve",
}

// claimsDone reports whether a reply told the caller something has been done.
func claimsDone(said string) bool {
	lower := spokenLower(said)
	return slices.ContainsFunc(doneClaims, func(claim string) bool { return strings.Contains(lower, claim) }) ||
		doneClaim.MatchString(lower)
}

// doneClaim is a reply saying something has been, or is being, done: "Your appointment has
// been successfully rescheduled" with no reschedule called.
var doneClaim = regexp.MustCompile(`\b(has|have) been (successfully )?(re)?(booked|scheduled|moved|updated|cancell?ed|dispatched|created|opened|placed|verified|logged|submitted|processed|changed)\b|\b(is|are) being (re)?(booked|scheduled|moved|updated|cancell?ed|dispatched|created|opened|placed|processed)\b`)

// doneClaims are what a reply telling the caller something has been done says.
var doneClaims = []string{
	"you're booked", "you are booked", "you're all set", "you are all set", "has been booked",
	"i've booked", "i have booked", "i've checked", "i have checked", "i've updated",
	"i've cancelled", "i've canceled", "i've scheduled", "i've rescheduled", "i've opened",
	"i've verified", "i've placed", "i've sent", "is confirmed", "has been scheduled",
	"has been cancelled", "has been dispatched", "is on the way",
	"i'm logging", "i am logging", "i will log", "i'll log", "i'm opening", "i am opening",
	"i'm booking", "i am booking", "i'm dispatching", "i am dispatching",
	"got you down for", "got you booked", "i've moved", "i've changed", "i've reserved",
	"has been rescheduled", "has been moved", "is booked", "is all set",
	"i opened", "i booked", "i scheduled", "i rescheduled", "i cancelled", "i canceled",
	"i dispatched", "i created", "i logged", "is dispatched", "are dispatched", "has been opened",
	"has been created", "has been logged",
}

// backsClaim reports whether what a tool has answered since the caller last spoke makes
// said's claim true: "I've checked" takes any tool, while "you're booked" or "has been
// rescheduled" takes one that changes something. Qwen said "your appointment has been
// rescheduled" on the strength of the lookup that found it.
//
// A change claim that names what one of the tools makes ("I opened a ticket" against
// create_ticket) takes that tool: PhoneLLM Mini told the caller "I opened a ticket for
// you... Ticket number 12345" with only walk_reboot behind it.
func backsClaim(history []llm.Message, said string, tools []string) bool {
	if !claimsChange(said) {
		return toolRanSinceCaller(history)
	}
	named := claimedObjects(said, tools)
	return changedSinceCaller(history, func(tool string) bool {
		return len(named) == 0 || slices.ContainsFunc(named, func(object string) bool {
			return slices.Contains(strings.Split(strings.ToLower(tool), "_")[1:], object)
		})
	})
}

// claimedObjects are the objects of tools ("ticket" in create_ticket) that said names, a
// word they open counting ("technician" for dispatch_tech).
func claimedObjects(said string, tools []string) []string {
	words := strings.FieldsFunc(spokenLower(said), func(r rune) bool { return !unicode.IsLetter(r) })
	var named []string
	for _, tool := range tools {
		parts := strings.Split(strings.ToLower(tool), "_")
		for _, object := range parts[1:] {
			said := slices.ContainsFunc(words, func(word string) bool { return strings.HasPrefix(word, object) })
			if len(object) >= 4 && said && !slices.Contains(named, object) {
				named = append(named, object)
			}
		}
	}
	return named
}

// claimsChange reports whether said claims more than a lookup.
func claimsChange(said string) bool {
	lower := spokenLower(said)
	return slices.ContainsFunc(doneClaims, func(claim string) bool {
		return strings.Contains(lower, claim) && !slices.Contains(lookedClaims, claim)
	})
}

// lookedClaims are the done claims a lookup is enough for.
var lookedClaims = []string{"i've checked", "i have checked", "i've verified"}

// changedSinceCaller reports whether a tool that changes something, rather than one that
// only reads (lookup_, check_, verify_ and the like), has answered without failing since
// the caller last spoke, and is one that counts.
func changedSinceCaller(history []llm.Message, counts func(tool string) bool) bool {
	names := map[string]string{}
	for _, message := range history {
		for _, call := range message.ToolCalls {
			names[call.ID] = call.Name
		}
	}
	for i := len(history) - 1; i >= 0; i-- {
		switch history[i].Role {
		case llm.ToolResult:
			name := names[history[i].ToolCallID]
			if !failedResult(history[i].Content) && !readsOnly(name) && counts(name) {
				return true
			}
		case llm.User:
			return false
		}
	}
	return false
}

func anyTool(string) bool { return true }

// readVerbs open the names of tools that only read.
var readVerbs = []string{"lookup", "look_up", "check", "get", "find", "search", "verify", "list", "read", "fetch", "query"}

// readsOnly reports whether a tool's name says it only reads. A name that cannot be found
// counts as a change, so a claim it backs is let through as before.
func readsOnly(name string) bool {
	name = strings.ToLower(name)
	return name != "" && slices.ContainsFunc(readVerbs, func(verb string) bool { return strings.HasPrefix(name, verb) })
}

// toolRanSinceCaller reports whether a tool has answered since the caller last spoke, and
// answered with something other than its failure: "I've got you down for 8:15" after the
// booking came back "no table at 8:15" is as unbacked as one with no call at all.
func toolRanSinceCaller(history []llm.Message) bool {
	for i := len(history) - 1; i >= 0; i-- {
		switch history[i].Role {
		case llm.ToolResult:
			if !failedResult(history[i].Content) {
				return true
			}
		case llm.User:
			return false
		}
	}
	return false
}

// failedResult reports whether a tool result is the tool failing rather than answering.
func failedResult(content string) bool {
	if strings.HasPrefix(content, didNotWork) {
		return true
	}
	var body struct {
		Error any   `json:"error"`
		OK    *bool `json:"ok"`
	}
	if json.Unmarshal([]byte(content), &body) != nil {
		return false
	}
	return body.Error != nil && body.Error != "" || body.OK != nil && !*body.OK
}

// holdOpenings open a reply that puts the caller on hold.
var holdOpenings = []string{"one moment", "a moment", "one sec", "hold on", "bear with me", "please hold"}

// holdWords are what a sentence asking the caller to wait on a lookup says.
var holdWords = []string{
	"one moment", "a moment", "one sec", "a sec", "hold on", "bear with me", "checking",
	"let me check", "let me look", "let me see", "looking that up", "looking into",
	"pulling that up", "pulling up",
}

// preSpeech is what the first of calls that names one asks to be said while it runs
// (ToolPolicy.PreSpeech), or empty for none.
func (a *Agent) preSpeech(calls []llm.ToolCall) string {
	if a.options.ToolPolicy == nil {
		return ""
	}
	for _, call := range calls {
		if phrase := a.options.ToolPolicy(call.Name).PreSpeech; phrase != "" {
			return phrase
		}
	}
	return ""
}

// fillerQuiet is how long the line has to have been quiet before a step taken on a result
// is announced. Audio is published as it plays, so a shorter gap is the last reply still
// being heard.
const fillerQuiet = 300 * time.Millisecond

// fillsPause reports whether a turn that said nothing should say something before the
// tools it asked for are run. Only a result that earns a reply of its own is worth
// promising: anything else leaves a promise to check as the last thing the caller heard.
// A turn reading a result has already been promised, and a filler there only queues behind
// the last one, unless that has finished playing: the next step (the outage check after
// the account is verified) is otherwise run in silence.
func (a *Agent) fillsPause(completionID string, calls []llm.ToolCall) bool {
	if strings.HasPrefix(completionID, toolPrefix) {
		a.mu.Lock()
		playing := a.utterances > 0 || time.Since(a.lastSpokeAt) < fillerQuiet
		a.mu.Unlock()
		if playing {
			return false
		}
	}
	return slices.ContainsFunc(calls, func(call llm.ToolCall) bool {
		return a.followsTool(harness.ToolRequested{TurnID: completionID, Call: call})
	})
}

// consumeTTS publishes the agent's speech to the edge as it is synthesised.
func (a *Agent) consumeTTS(p *pipeline, voice *ttsrouter.Session) {
	defer p.running.Done()

	// Abandoned audio arrives a frame at a time, so it is reported once per utterance
	// rather than once per frame.
	dropping := ""
	// lost is set once the voice has gone, which a provider may say twice: as a fatal error
	// and as the socket dropping.
	lost := false
	// An expressive voice can read a phrase 20 to 30 dB below the one before it, which a
	// caller hears as the agent trailing off and their phone hears as it having finished.
	leveller := audio.NewLeveller()
	overlapping := newOverlap(voice.Provider(), a.abandonedTurn, a.logger)
	queue := newPlayout(a.options.Edge, voice.Provider())
	marked, marks := a.options.Edge.(MarkedPlayout)
	// The utterance the last chunk was of. A synthesis arrives as a run of chunks, so it is
	// looked up once per run rather than once per chunk.
	var heard *utterance
	heardID := ""

	for event := range voice.Events() {
		switch typed := event.(type) {
		case tts.AudioChunk:
			// Audio from an abandoned turn is dropped here rather than published, so
			// barge-in silences the agent even while the provider is still sending.
			//
			// Only an interruption abandons a turn. A turn the agent started for itself,
			// to say what a tool returned or what delegated work came back with, does not:
			// treating it as one would throw away the tail of the sentence being spoken
			// and cut the agent off mid-word.
			lookup := typed.SynthesisID != heardID
			publish, first, found := a.publishing(turnOf(typed.SynthesisID), typed.SynthesisID, lookup)
			if lookup {
				heard, heardID = found, typed.SynthesisID
			}
			if !publish {
				a.turns.dropped(turnOf(typed.SynthesisID), typed.Audio.DurationMs())
				if dropping != typed.SynthesisID {
					dropping = typed.SynthesisID
					a.logger.Debug("dropping audio for a turn the agent has left behind",
						"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID))
				}
				continue
			}
			speech := leveller.Level(typed.Audio)
			var err error
			if marks {
				// Publishing returns once the chunk is queued, which for a long one is well
				// after the participants could have heard it begin, so the edge says when.
				err = marked.PublishAudioMarked(speech, withUtterance(a.turns.marksFor(turnOf(typed.SynthesisID)), heard))
			} else {
				err = a.options.Edge.PublishAudio(speech)
			}
			if errors.Is(err, ErrCallLeft) {
				if dropping != typed.SynthesisID {
					dropping = typed.SynthesisID
					a.logger.Debug("dropping audio for a call that has been left", "synthesis", typed.SynthesisID)
				}
				continue
			}
			if err != nil {
				a.fail(err, "edge")
			}
			// An edge that cannot say when its track took the audio is taken at its word,
			// like firstAudio below.
			if heard.sent(typed.Audio.DurationMs()) && !marks {
				heard.FirstAudiblePulled(time.Now())
			}
			overlapping.published(typed.SynthesisID, typed.Audio.DurationMs())
			a.mu.Lock()
			a.lastSpokeAt = time.Now()
			a.mu.Unlock()
			// The wait ends when the participant can hear something, so this is timed
			// after the publish rather than before it.
			a.turns.firstAudio(turnOf(typed.SynthesisID), time.Now())
			if first && a.converse.Listening() {
				talkedOver.WithLabelValues(a.turns.originOf(turnOf(typed.SynthesisID)).kind).Inc()
			}

		case tts.SynthesisStarted:
			queue.started(typed.SynthesisID)
			a.logger.Info("the voice took an utterance",
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"kind", a.turns.originOf(turnOf(typed.SynthesisID)).kind,
				"provider", typed.Provider, "voice", typed.Voice,
				"caller_talking", a.converse.Listening())

		case tts.SynthesisComplete:
			done := a.takeUtterance(typed.SynthesisID)
			if heardID == typed.SynthesisID {
				heard, heardID = nil, ""
			}
			overlapping.finished(typed.SynthesisID)
			a.settleVoice(voice)
			a.logger.Info("finished speaking", append([]any{
				"synthesis", typed.SynthesisID, "turn", turnOf(typed.SynthesisID),
				"kind", a.turns.originOf(turnOf(typed.SynthesisID)).kind,
				"audio_ms", typed.AudioDurationMs, "ttfb_ms", typed.TimeToFirstByteMs,
				"interrupted", typed.Interrupted,
			}, queue.finished(typed.SynthesisID)...)...)
			a.respondQueued()
			a.turns.spoke(turnOf(typed.SynthesisID), typed.TimeToFirstByteMs, typed.AudioDurationMs)
			if !typed.Interrupted {
				// An interrupted utterance was not heard in full, so it must not be
				// recorded as a finished spoken reply.
				// All of it is queued at the edge, behind whatever is still to play.
				a.heardBefore(done, func() float64 { return queue.ahead(typed.AudioDurationMs) })
				a.emitter.Send(Spoke{
					TurnID:            turnOf(typed.SynthesisID),
					SynthesisID:       typed.SynthesisID,
					Characters:        typed.Characters,
					Message:           done.span(),
					AudioDurationMs:   typed.AudioDurationMs,
					TimeToFirstByteMs: typed.TimeToFirstByteMs,
				})
			}
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
			if !lost {
				lost = true
				a.loseVoice(p, voice)
			}

		case tts.Error:
			if typed.Fatal {
				if !lost {
					lost = true
					a.loseVoice(p, voice)
				}
			} else if typed.SynthesisID == "" {
				// A failure naming an utterance still settles it, since the router turns
				// that utterance's completion into the failed row. A nameless failure
				// settles only while this session still owns the voice.
				a.settleVoice(voice)
			}
			a.fail(typed.Err, "tts")
		}
	}
}

// consumeHarness reports what the harness decided, and speaks whatever the subagent came
// back with.
func (a *Agent) consumeHarness(current *harness.Harness, drained chan struct{}) {
	defer a.running.Done()
	defer close(drained)

	for event := range current.Events() {
		switch typed := event.(type) {
		case harness.Decided:
			ruler, disposition := cmp.Or(string(typed.Ruler), "none"), string(typed.Disposition)
			if !typed.Valid() {
				disposition = "failed"
			}
			flowDecisions.WithLabelValues(disposition, ruler).Inc()
			flowDecisionSeconds.WithLabelValues(ruler).Observe(typed.TookMs / 1000)
			a.act(a.converse.Ruled(typed, a.floor()))
			// An answer took the reply started for these words; any other ruling leaves it
			// unwanted.
			a.dropSpeculation(typed.CandidateID)

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

		case harness.ToolDropped:
			a.dropTool(typed.Name)

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
	a.compactedAway = append(a.compactedAway, a.history[:len(compacted.Prefix)]...)
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
	if a.harness == nil || a.generating || a.utterances > 0 || a.interrupting || a.recoveringVoice != nil {
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
	if a.toolReply {
		a.mu.Unlock()
		return a.respondAfterTool(toolPrefix + turnStamp())
	}
	history := a.replayLocked()
	turnID := replyPrefix + turnStamp()
	a.speakingTurn = turnID
	a.generating = true
	participant := a.lastParticipant
	continues := a.owedTurn
	a.owedTurn = ""
	instructions := a.instructions()
	asked := a.asked
	a.mu.Unlock()

	// This turn is deliberately not measured. A Turn reports the wait between someone
	// finishing a sentence and hearing the answer start, and nobody said anything here.
	a.turns.started(turnID, kindFollowUp)
	a.emitter.Send(Responding{TurnID: turnID, Participant: participant, Continues: continues})

	return a.generate(harness.Turn{
		ID:           turnID,
		Instructions: instructions,
		History:      history,
		Asked:        asked,
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
// speech it has not finished saying, a tool still running, work handed to the subagent,
// or an answer that has come back and still owes the caller a turn.
//
// It exists because one thing said to the agent can produce several replies -- the turn
// that called a tool, the turn that read the tool's answer, the turn a subagent's finding
// earned -- so a caller who waited only for the first would talk over the rest.
func (a *Agent) Busy() bool {
	a.mu.Lock()
	// A transfer finishes the call from inside itself, so counting it would have it wait on
	// its own end.
	working := a.interrupting || a.recoveringVoice != nil || a.generating || a.utterances > 0 ||
		slices.ContainsFunc(slices.Collect(maps.Values(a.toolsRunning)), func(name string) bool { return !telephonyTool(name) })
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

// countedDigits is "last four 9821" in a reply. Read straight through it is heard as "last
// 49821", and a comma's pause in the voice is too short to stop that, so pauseCounted says
// what the count is of.
var countedDigits = regexp.MustCompile(`(?i)\b(last|first|final)\s+(two|three|four|five|six|[2-6])\s+(\d)`)

func pauseCounted(text string) string {
	return countedDigits.ReplaceAllString(text, "$1 $2 digits, $3")
}

// speakSentence sends one sentence of a reply to the voice, from where it is in the reply's
// message.
//
// A streaming provider takes a turn's sentences as deltas of one utterance, which keeps a
// turn to a single billed synthesis. A provider that cannot take deltas gets each sentence
// as its own final request instead.
func (a *Agent) speakSentence(turnID, text string, message MessageSpan) error {
	voice := a.voice()
	if voice == nil {
		return errors.New("agent: not joined")
	}
	// A direction with no words to act it on is refused as empty text by voices that
	// strip their tags first, and refusing it costs the call its voice.
	if strings.TrimSpace(tts.Unperformed(text)) == "" {
		return nil
	}
	text = pauseCounted(text)
	a.turns.ttsStarted(turnID, time.Now())

	if !voice.Streaming() {
		id := fmt.Sprintf("%s%s%d", turnID, sentenceSuffix, a.sentences)
		a.sentences++
		a.begin()
		return a.synthesize(voice, turnID, tts.Request{ID: id, Text: text, Final: true}, message)
	}

	if a.openTurn != turnID {
		a.openTurn = turnID
		a.begin()
	}
	return a.synthesize(voice, turnID, tts.Request{ID: turnID, Text: text}, message)
}

// speakWhole says a piece of text that is already complete, as one utterance. Its message,
// if it has one, is the text.
func (a *Agent) speakWhole(turnID, text string) error {
	voice := a.voice()
	if voice == nil {
		return stack.Wrap(errors.New("agent: not joined"))
	}
	a.turns.ttsStarted(turnID, time.Now())
	a.begin()
	whole := MessageSpan{End: utf8.RuneCountInString(text)}
	if err := a.synthesize(voice, turnID, tts.Request{ID: turnID, Text: text, Final: true}, whole); err != nil {
		// Refused, it is never said, so it is not counted as being said either.
		a.settle()
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

// claimReply gives the model consumer's buffers to this reply before any text is
// filtered or flushed. An abandoned stream may deliver its completion later.
func (a *Agent) claimReply(turnID string) {
	if a.replying != turnID {
		a.resetTurn()
		a.replying = turnID
	}
}

// resetTurn forgets the text and the utterance of a turn that has ended, whether it
// finished or was interrupted.
func (a *Agent) resetTurn() {
	a.chunk.Reset()
	a.performed = a.performed[:0]
	a.harness.Reset()
	a.directions.Reset()
	a.spoken.Reset()
	a.sentences = 0
	a.openTurn = ""
	a.withheld = false
	a.withheldFrom = ""
	a.narrated = false
	a.promised = false
	a.mu.Lock()
	a.saying = ""
	a.mu.Unlock()
}

// unheardReply is the reply to a caller's turn before any of it has gone out.
type unheardReply struct {
	turnID      string
	participant string
	decidedAt   time.Time
	// said is what the reply was to answer, with whatever an earlier dropped reply was for.
	said string
}

// missedLocked is what the participant said that a dropped reply was for, empty when none was.
func (a *Agent) missedLocked(participant stt.Participant) string {
	if a.missed.participant != participant.ID {
		return ""
	}
	return a.missed.said
}

// missedWords is how many of a caller's unanswered words a dropped reply's note carries.
// About forty seconds of speech at a conversational 150 words a minute: a caller who goes on
// longer over every reply is answered on what they are still on, and the note is not resent
// ever larger on each turn.
const missedWords = 100

// newestWords is the last limit words of text.
func newestWords(text string, limit int) string {
	fields := strings.Fields(text)
	if len(fields) <= limit {
		return text
	}
	return strings.Join(fields[len(fields)-limit:], " ")
}

// missedNote tells the model what a dropped reply was for, so the caller's turn after it
// answers both rather than only the newest words.
func missedNote(said string) string {
	if said == "" {
		return ""
	}
	return fmt.Sprintf("The caller said more before you could answer them, and you have not yet "+
		"answered what they said earlier: %q. Answer everything they said since your last reply "+
		"together. Treat their later words as a correction only where they clearly correct the earlier ones.", said)
}

// dropUnheard abandons the reply to a caller's turn the same caller said more over, while
// none of it has gone out: after that it is them talking over the agent, the controller's call.
func (a *Agent) dropUnheard(participant stt.Participant, turnID string) {
	sent := a.turns.sentToVoice(turnID)
	var stale unheardReply
	if !a.abandonReply(participant, func(speaking string) bool {
		if speaking != turnID || a.unheard.turnID != turnID {
			return false
		}
		stale, a.unheard = a.unheard, unheardReply{}
		a.missed = stale
		return true
	}) {
		return
	}
	// The caller took the floor as surely as by talking over the reply, so its tools stop the same way.
	a.stopTools()
	staleReplies.Inc()
	a.logger.Info("dropped a reply the caller said more over before hearing any of it",
		"turn", turnID, "participant", participant.ID,
		"gap_ms", time.Since(stale.decidedAt).Milliseconds(), "sent_to_voice", sent)
}

// interrupt abandons the reply being spoken because a participant started talking.
func (a *Agent) interrupt(participant stt.Participant) {
	a.abandonReply(participant, nil)
}

// abandonReply abandons the reply being spoken, and reports whether there was one. still,
// when set, is asked under the lock whether it is still the reply to abandon.
func (a *Agent) abandonReply(participant stt.Participant, still func(turnID string) bool) bool {
	a.mu.Lock()
	turnID := a.speakingTurn
	if turnID == "" {
		a.mu.Unlock()
		return false
	}
	// A murmur is meant to overlap with what someone is saying, so hearing them carry on
	// is not an interruption: there is no reply to abandon and nothing was cut short.
	if strings.HasPrefix(turnID, backchannelPrefix) || still != nil && !still(turnID) {
		a.mu.Unlock()
		return false
	}
	heard, played := a.voiced == turnID, time.Since(a.voicedAt)
	cut := a.cutShortLocked(turnID, time.Now())
	a.abandoned[turnID] = struct{}{}
	a.interrupting = true
	a.speakingTurn = ""
	a.generating = false
	a.saying = ""
	reply, voice, model, current := a.streams[turnID], a.tts, a.sts, a.harness
	a.mu.Unlock()
	if heard {
		playedBeforeInterrupt.WithLabelValues(a.turns.originOf(turnID).kind).Observe(played.Seconds())
	}
	// Clear audio already queued at the edge before provider or model interruption can block.
	// Keep the final drop below as well to catch audio published by an in-flight send.
	a.dropSpeech()

	a.finishGenerate(turnID)
	if current != nil {
		current.Release(turnID)
	}
	a.logger.Debug("stopping mid-reply, the caller took the floor",
		"turn", turnID, "participant", participant.ID)

	if voice != nil {
		if err := voice.Interrupt(); err != nil {
			a.fail(err, "tts")
		}
	}
	// A native model is told to stop the reply it is speaking. It has no word for how
	// much the caller heard beyond what it sent, so the router's own count stands in.
	if model != nil {
		if err := model.Interrupt(0); err != nil {
			a.fail(err, "sts")
		}
	}
	// Cancelling the voice only stops what it has not synthesised yet. What it already sent
	// is queued at the edge, and leaving it there is the caller being talked over for as
	// long as that queue is deep.
	a.dropSpeech()
	// The reply still settles after this, and is still billed: what it generated before
	// being cut off was generated all the same.
	if reply != nil {
		if err := reply.Close(); err != nil {
			a.fail(err, "llm")
		}
	}

	a.turns.interrupt(turnID)
	a.emitter.Send(cut.cut(Interrupted{TurnID: turnID, Participant: participant}))
	a.mu.Lock()
	a.interrupting = false
	a.mu.Unlock()
	a.respondQueued()
	return true
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

// publishing reports whether audio of a turn goes out (not once it was abandoned), and whether
// it is the turn's first, which makes a caller's reply heard. Interleaved turns count a first
// at each switch, which is the agent talking over itself (speechOverlaps).
//
// With lookup it also returns the utterance the synthesis is, under the same lock, so that
// finding it costs the voice's goroutine nothing more than the check it makes anyway.
func (a *Agent) publishing(turnID, synthesisID string, lookup bool) (publish, first bool, heard *utterance) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if lookup {
		heard = a.utteranceLocked(synthesisID)
	}
	if _, gone := a.abandoned[turnID]; gone {
		return false, false, heard
	}
	if a.unheard.turnID == turnID {
		a.unheard = unheardReply{}
	}
	if a.voiced == turnID {
		return true, false, heard
	}
	a.voiced, a.voicedAt = turnID, time.Now()
	return true, true, heard
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

// settleVoice counts a synthesis only while the session that emitted it still owns the
// voice. Events already buffered by a retired session must not settle its replacement.
func (a *Agent) settleVoice(voice *ttsrouter.Session) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.tts == voice && a.utterances > 0 {
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

// Continue picks up the conversation another agent held: what was said and who said it,
// so the people already heard keep their labels when a session switches between voice and
// writing.
func (a *Agent) Continue(previous *Agent) {
	previous.mu.Lock()
	history := append([]llm.Message(nil), previous.history...)
	compactedAway := append([]llm.Message(nil), previous.compactedAway...)
	// A copy: the old agent can still keep a turn between this and its Close, and would
	// write to a map the new agent reads.
	speakers := maps.Clone(previous.speakers)
	opened := previous.opened
	previous.mu.Unlock()
	a.mu.Lock()
	defer a.mu.Unlock()
	a.history = history
	a.compactedAway = compactedAway
	a.speakers = speakers
	a.opened = opened
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
