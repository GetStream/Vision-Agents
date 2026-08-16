package harness

import (
	"strings"
)

// needPrefix is how a skill says it cannot finish without something only the caller can
// tell it. Every skill's instructions ask for it, because a subagent that guesses at a
// missing detail is worse than one that has the agent ask.
const needPrefix = "NEED:"

// State is how a task ended.
type State string

const (
	// Done means the subagent answered.
	Done State = "done"
	// Cancelled means the answer stopped being worth having before it arrived.
	Cancelled State = "cancelled"
	// Failed means the subagent could not answer.
	Failed State = "failed"
)

// Reasons a task was cancelled.
const (
	// ReasonSuperseded means a newer request for the same skill replaced it. The premise
	// of the old one is what changed, which is exactly the case worth abandoning.
	ReasonSuperseded = "superseded"
	// ReasonDropped means the model said the answer no longer matters.
	ReasonDropped = "dropped"
	// ReasonDeadline means the answer stopped being worth having.
	ReasonDeadline = "deadline"
	// ReasonClosed means the conversation ended.
	ReasonClosed = "closed"
)

// Result is a finished task.
type Result struct {
	TaskID string
	Skill  string
	State  State
	// Text is the answer, when there is one.
	Text string
	// Question is what the agent must ask the caller before the work can go any further.
	// It is set instead of Text.
	Question string
	// Reason says why a cancelled task was abandoned.
	Reason string
	Err    error
	// ElapsedMs is how long the caller was kept company for.
	ElapsedMs float64
}

// Answered reports whether the task produced something worth telling the caller.
func (r Result) Answered() bool { return r.State == Done && r.Text != "" }

// Actionable reports whether the caller is owed something because of this result: an
// answer, a question, or the news that the answer is not coming. A cancelled task is
// not, because its premise is gone and nobody was still waiting on it.
func (r Result) Actionable() bool {
	switch r.State {
	case Done:
		return r.Text != "" || r.Question != ""
	case Failed:
		return true
	}
	return false
}

// answer splits a subagent's reply into an answer and a question it needs asked first.
func answer(text string) (string, string) {
	trimmed := strings.TrimSpace(text)
	if !strings.HasPrefix(trimmed, needPrefix) {
		return trimmed, ""
	}
	return "", strings.TrimSpace(strings.TrimPrefix(trimmed, needPrefix))
}
