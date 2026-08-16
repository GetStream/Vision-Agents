package harness

import (
	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
)

// Event is emitted on the channel returned by Harness.Events.
//
// These describe what the harness decided, not what the models did. The model sessions
// stay available for anyone who needs that detail.
type Event interface {
	isHarnessEvent()
}

// Delegated means the fast model handed a piece of work to the subagent, and is free to
// keep talking to the caller while it runs.
type Delegated struct {
	TaskID string
	Skill  string
	// Prompt is what was handed over. It was never spoken.
	Prompt string
	// TurnID is the reply the request was made in.
	TurnID string
}

func (Delegated) isHarnessEvent() {}

// Settled means a task finished, one way or another. A settled task with an answer is
// worth telling the caller about, which is why it is reported rather than merely folded
// into the next prompt.
type Settled struct {
	Result
}

func (Settled) isHarnessEvent() {}

// Emitter fans harness events out to a single consumer channel.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
