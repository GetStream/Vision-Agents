package harness

import (
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/emit"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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

// ToolRequested means the fast model asked for a tool to be run. Unlike a delegated skill
// nothing is running yet: the harness cannot act on the call, so it says what was asked and
// leaves the doing to whoever is on the call.
type ToolRequested struct {
	// TurnID is the reply the call was made in.
	TurnID string
	Call   llm.ToolCall
}

func (ToolRequested) isHarnessEvent() {}

// Settled means a task finished, one way or another. A settled task with an answer is
// worth telling the caller about, which is why it is reported rather than merely folded
// into the next prompt.
type Settled struct {
	Result
}

func (Settled) isHarnessEvent() {}

// Decided is what the fast flow controller chose for one stable transcript revision.
type Decided struct {
	CandidateID string
	Disposition Disposition
	Floor       Floor
	Err         error
}

func (d Decided) Valid() bool {
	return d.Err == nil && d.CandidateID != "" && d.Disposition.Valid() && d.Floor.Valid()
}

func (d Decided) Error() error {
	if d.Err != nil {
		return d.Err
	}
	if d.CandidateID == "" {
		return errors.New("harness: flow decision has no candidate")
	}
	if !d.Disposition.Valid() {
		return errors.New("harness: invalid flow disposition")
	}
	if !d.Floor.Valid() {
		return errors.New("harness: invalid floor decision")
	}
	return nil
}

func (Decided) isHarnessEvent() {}

// Compacted replaces an old history prefix with a summary.
type Compacted struct {
	TaskID  string
	Prefix  []llm.Message
	Summary string
}

func (Compacted) isHarnessEvent() {}

// Emitter fans harness events out to a single consumer channel.
type Emitter = emit.Emitter[Event]

// NewEmitter returns an Emitter with the given channel buffer.
func NewEmitter(buffer int) *Emitter { return emit.New[Event](buffer) }
