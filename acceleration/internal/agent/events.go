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

// Heard is a settled turn from a participant. Interim transcripts are not surfaced: they
// are revisions of a turn that has not finished, and acting on them would mean answering
// half a sentence.
type Heard struct {
	Participant stt.Participant
	Text        string
	Language    string
}

func (Heard) isAgentEvent() {}

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

// Backchannel means the agent made a listening noise while a participant was still
// talking, which is not a turn and does not go near the model.
type Backchannel struct {
	Participant stt.Participant
	Text        string
}

func (Backchannel) isAgentEvent() {}

// Speculated means the agent started answering a turn the transcriber had provisionally
// ended. Promoted reports whether the guess held: a promoted reply had its answer ready
// before the turn even settled, and an unpromoted one was thrown away unheard.
type Speculated struct {
	TurnID      string
	Participant stt.Participant
	// Text is the provisional transcript the guess was made on.
	Text     string
	Promoted bool
}

func (Speculated) isAgentEvent() {}

// Interrupted means a participant started talking over the agent, so the reply being
// spoken was abandoned.
type Interrupted struct {
	TurnID      string
	Participant stt.Participant
}

func (Interrupted) isAgentEvent() {}

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
