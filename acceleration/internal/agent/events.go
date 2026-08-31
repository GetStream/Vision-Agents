package agent

import (
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
)

// Event is emitted on the channel returned by Agent.Events.
//
// These describe the conversation rather than the providers. The three modality event
// streams stay available through the sessions for anyone who needs that detail.
type Event interface {
	isAgentEvent()
}

// Joined means the agent is in the call and listening.
type Joined struct {
	At time.Time
}

func (Joined) isAgentEvent() {}

// ParticipantJoined means somebody else is now in the call.
//
// Reported because an agent in a call it did not start has no other way of knowing anybody
// is there: an agent answering a phone has to know the caller arrived before it says hello,
// and until they speak there is no audio to infer it from.
type ParticipantJoined struct {
	Participant stt.Participant
	At          time.Time
}

func (ParticipantJoined) isAgentEvent() {}

// ParticipantLeft means somebody else has gone. On a phone call there is only one of them,
// so this is the caller hanging up.
type ParticipantLeft struct {
	Participant stt.Participant
	At          time.Time
}

func (ParticipantLeft) isAgentEvent() {}

// Hearing is a transcript revision as it arrives, before anything has been decided about
// it. It is what the agent is hearing rather than what it has heard: the words are still
// changing, and most revisions are replaced by the next one.
//
// Nothing in the conversation acts on this. It is here so a person watching a call can
// see the words appear as the agent does, which is the difference between watching an
// agent think and waiting to find out what it concluded.
type Hearing struct {
	Participant stt.Participant
	Text        string
	Language    string
}

func (Hearing) isAgentEvent() {}

// Heard is a settled turn from a participant, once the conversation has decided it was
// meant for the agent. A revision of a turn that has not finished is a Hearing instead:
// acting on one would mean answering half a sentence.
type Heard struct {
	Participant stt.Participant
	Text        string
	Language    string
}

func (Heard) isAgentEvent() {}

// Decided is one judgement the conversation made about how to handle the call: whether
// the caller had finished, whether what they said was meant for the agent, who keeps the
// floor when both are talking, and when a silence needs filling.
//
// It carries the reason as well as the choice. A latency figure explains a slow call on
// its own, but only the reasoning explains a call that went somewhere nobody expected.
type Decided struct {
	At   time.Time
	Kind string
	// Reason is why, in words.
	Reason string
	// TurnID is the turn the judgement was about.
	TurnID string
	// Participant is who it concerned.
	Participant stt.Participant
	// Text is what was heard, or what the agent decided to say.
	Text string
	// LatencyMs is what the flow controller took to rule. Zero where nothing was asked.
	LatencyMs float64
}

func (Decided) isAgentEvent() {}

// Responding means the agent has asked the model for a reply.
type Responding struct {
	TurnID string
	// Participant is who the agent is replying to.
	Participant stt.Participant
	// Prompt is what the agent is replying to.
	Prompt string
}

func (Responding) isAgentEvent() {}

// ResponseDelta is a piece of the reply as it streams back.
type ResponseDelta struct {
	TurnID string
	Text   string
}

func (ResponseDelta) isAgentEvent() {}

// Responded means the model finished. The reply may still be being spoken.
type Responded struct {
	TurnID string
	Text   string
	// TimeToFirstTokenMs is how long the participant waited for the model to start.
	TimeToFirstTokenMs float64
}

func (Responded) isAgentEvent() {}

// Spoke means a piece of the reply finished being synthesised and published.
type Spoke struct {
	TurnID string
	// AudioDurationMs is how much speech was published.
	AudioDurationMs float64
	// TimeToFirstByteMs is how long the participant waited to hear anything.
	TimeToFirstByteMs float64
}

func (Spoke) isAgentEvent() {}

// Turn is a finished exchange, measured the way the participant experienced it. The
// legs are reported alongside the whole so a slow conversation can be attributed to
// transcription, the model or the voice rather than only observed.
//
// A leg that did not happen is zero: an interrupted turn may never have reached the
// voice, and a pipeline where one model hears and speaks for itself has only the
// roundtrip to report.
type Turn struct {
	TurnID      string
	Participant stt.Participant
	// StartedAt is when the settled transcript arrived, which is when the wait begins.
	StartedAt time.Time
	// STTLatencyMs is what the transcriber spent settling the turn.
	STTLatencyMs float64
	// LLMTTFTMs is the wait between asking the model and its first token.
	LLMTTFTMs float64
	// TTSTTFBMs is the wait between sending the first sentence and the first audio.
	TTSTTFBMs float64
	// RoundtripMs is the whole delay: settled transcript to first audio published.
	RoundtripMs float64
	// SpeechEndToAudioMs is voice in to voice out: the roundtrip plus the time the
	// transcriber spent deciding the participant had stopped.
	SpeechEndToAudioMs float64
	// AudioOutMs is how much speech the agent published for the turn.
	AudioOutMs  float64
	Interrupted bool
}

func (Turn) isAgentEvent() {}

// Delegated means the model handed a piece of work to the subagent and carried on
// talking. The caller hears the filler around it, never the request itself.
type Delegated struct {
	TaskID string
	Skill  string
	// Prompt is what was handed over.
	Prompt string
	// TurnID is the reply the request was made in.
	TurnID string
}

func (Delegated) isAgentEvent() {}

// TaskSettled means delegated work finished and, if it produced anything the caller is
// owed, the agent has started a turn to say so.
type TaskSettled struct {
	TaskID string
	Skill  string
	// Text is the answer, when there is one.
	Text string
	// Question is what the subagent needs asked before it can go further.
	Question string
	// ElapsedMs is how long the caller was kept company for.
	ElapsedMs float64
	Err       error
}

func (TaskSettled) isAgentEvent() {}

// TaskCancelled means delegated work was abandoned because its premise was gone: the
// caller moved on, the model dropped it, it ran out of time, or the call ended.
type TaskCancelled struct {
	TaskID string
	Skill  string
	Reason string
}

func (TaskCancelled) isAgentEvent() {}

// ToolRan means the model asked for a tool and it has been carried out, or tried. It is
// reported whether or not it worked, because a tool that failed still changed what the
// agent goes on to say.
type ToolRan struct {
	TurnID string
	Tool   string
	// Arguments is the JSON the model filled in, as it wrote it.
	Arguments string
	// Result is what the model was told about the outcome.
	Result string
	Err    error
}

func (ToolRan) isAgentEvent() {}

// Transferred means a human has been dialled onto the call. The agent leaves once the
// handover is done, which is immediately for a cold transfer and after the summary has been
// spoken for a warm one.
type Transferred struct {
	TurnID string
	To     string
	// Summary is what the human is told about the caller. Empty means a cold transfer.
	Summary string
}

func (Transferred) isAgentEvent() {}

// Pressed means the agent pressed digits at a menu on a call it had placed.
type Pressed struct {
	TurnID string
	Digits string
}

func (Pressed) isAgentEvent() {}

// LookedUp means the agent read the knowledge base to answer something. It carries how
// many passages came back rather than the passages themselves: a watcher wants to know the
// agent went looking and whether it found anything, not to re-read the handbook.
type LookedUp struct {
	TurnID string
	Query  string
	// Documents is how many passages bore on the question. Zero means the knowledge base
	// had nothing, which is why the agent said it did not know.
	Documents int
}

func (LookedUp) isAgentEvent() {}

// Searched means the agent found out something that is true now to answer a question. Like
// a lookup it carries how many sources came back rather than the sources themselves.
type Searched struct {
	TurnID string
	Query  string
	// Results is how many sources bore on the question. Zero means the search found
	// nothing, which is why the agent said it could not find out.
	Results int
}

func (Searched) isAgentEvent() {}

// Backchannel means the agent made a listening noise while a participant was still
// talking, which is not a turn and does not go near the model.
type Backchannel struct {
	Participant stt.Participant
	Text        string
}

func (Backchannel) isAgentEvent() {}

// Interrupted means a participant started talking over the agent, so the reply being
// spoken was abandoned.
type Interrupted struct {
	TurnID      string
	Participant stt.Participant
}

func (Interrupted) isAgentEvent() {}

// OverlapDecided says how the agent handled speech arriving while it was talking.
type OverlapDecided struct {
	TurnID      string
	Participant stt.Participant
	Action      string
}

func (OverlapDecided) isAgentEvent() {}

// ConversationCompacted means an old history prefix was replaced with a summary.
type ConversationCompacted struct {
	Before  int
	After   int
	Summary string
}

func (ConversationCompacted) isAgentEvent() {}

// Error reports a failure from one of the three modalities or from the edge. The agent
// keeps going: a failed turn is one lost reply, not a lost conversation.
type Error struct {
	Err error
	// Context says which part failed, e.g. "stt", "llm", "tts" or "edge".
	Context string
}

func (e Error) Error() string { return e.Err.Error() }

func (e Error) Unwrap() error { return e.Err }

func (Error) isAgentEvent() {}

// Left means the agent is out of the call.
type Left struct {
	At time.Time
}

func (Left) isAgentEvent() {}

// Emitter fans agent events out to a single consumer channel.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
