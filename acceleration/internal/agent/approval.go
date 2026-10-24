package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// confirmTool is what a voice model calls with the caller's answer to a tool's approval
// question. It is offered only on a voice call with a tool that asks one.
const confirmTool = "confirm_tool_call"

// approvalWait is how long a voice caller has to answer an approval question, from when it
// finished playing.
const approvalWait = 15 * time.Second

// approvalAnswer is what a call that needs a yes is answered with at once, on a cascade that
// has said its question as written: the call is closed, so every model can go on.
const approvalAnswer = "Not run yet: %s runs only once the caller says yes, and they have just been asked: %q. " +
	"When they answer, call " + confirmTool + " with tool_call_id %q and allowed true for a yes, or false for " +
	"anything else. Do not call %s again for this and do not say it was done."

// nativeApprovalAnswer is approvalAnswer for a speech-to-speech model, which says the
// question itself.
const nativeApprovalAnswer = "Not run yet: %s runs only once the caller says yes. Ask them now, in these words: %q. " +
	"When they answer, call " + confirmTool + " with tool_call_id %q and allowed true for a yes, or false for " +
	"anything else. Do not call %s again for this and do not say it was done."

// Outcomes of confirm_tool_call that ran nothing.
const (
	approvalDeclined = "The caller declined, so %s was not run. Tell them so in a few words."
	approvalLapsed   = "The caller did not say yes in time, so %s was not run and that question is closed. " +
		"If they want it, call %s again to ask them again."
	approvalUnknown = "There is no question waiting for the caller's answer under that id, so nothing was run. " +
		"If they want something that needs their yes, call its tool to ask them."
)

// approvalSilence is the note a cascade's history is given when the caller let the wait pass.
const approvalSilence = "(The caller did not say yes to %q within %d seconds, so %s was not run. " +
	"If they want it, call %s again to ask them again.)"

// approval is one call waiting for the caller's yes. Its fields are guarded by Agent.mu.
type approval struct {
	call     llm.ToolCall
	question string
	// asked is set once the question was handed to whatever says it.
	asked bool
	// expires is when silence declines the call, set once the question finished playing.
	expires time.Time
}

// approvedRun is a call the caller said yes to, running under the confirmation that ran it.
type approvedRun struct {
	confirmation string
	tool         string
}

type approvedKey struct{}

// WithApproval is ctx for a call the caller said yes to.
func WithApproval(ctx context.Context) context.Context {
	return context.WithValue(ctx, approvedKey{}, true)
}

// Approved reports whether a call's context is that of a call the caller said yes to, so
// whoever runs it does not ask them again.
func Approved(ctx context.Context) bool {
	approved, _ := ctx.Value(approvedKey{}).(bool)
	return approved
}

// placeholder is one {name} in an approval question.
var placeholder = regexp.MustCompile(`\{([A-Za-z_][A-Za-z0-9_]*)\}`)

// fillQuestion is template with each {name} replaced by that argument of the call. A name
// the arguments do not have is left as written.
func fillQuestion(template, arguments string) string {
	var values map[string]any
	if json.Unmarshal([]byte(arguments), &values) != nil {
		return template
	}
	return placeholder.ReplaceAllStringFunc(template, func(found string) string {
		value, ok := values[found[1:len(found)-1]]
		if !ok || value == nil {
			return found
		}
		if text, isText := value.(string); isText {
			return text
		}
		encoded, err := json.Marshal(value)
		if err != nil {
			return found
		}
		return string(encoded)
	})
}

// approvalQuestion is the question a call to tool asks before it runs, when it asks one: only
// on a call, since in writing the app shows the call and asks.
func (a *Agent) approvalQuestion(tool harness.Tool) string {
	if a.options.Text || tool.Approval == nil {
		return ""
	}
	return strings.TrimSpace(tool.Approval.Title)
}

// asksApproval reports whether any tool on offer asks before it runs.
func (a *Agent) asksApproval() bool {
	for _, tool := range a.options.Tools.Tools {
		if a.approvalQuestion(tool) != "" {
			return true
		}
	}
	return false
}

// offersConfirm reports whether the model is offered confirm_tool_call: a tool on offer
// asks, and no tool of the caller's has the name.
func (a *Agent) offersConfirm() bool {
	if a.options.ToolRunner == nil || !a.asksApproval() {
		return false
	}
	_, taken := a.options.Tools.Lookup(confirmTool)
	return !taken
}

// confirmToolDefinition is confirm_tool_call as the model is offered it.
func confirmToolDefinition() harness.Tool {
	return harness.Tool{
		Name: confirmTool,
		Description: "Give the caller's answer to a question a tool asked before running. Call it only once " +
			"they have answered, with the tool_call_id the question came with.",
		Parameters: map[string]any{
			"type":     "object",
			"required": []string{"tool_call_id", "allowed"},
			"properties": map[string]any{
				"tool_call_id": map[string]any{"type": "string", "description": "The id the question came with."},
				"allowed":      map[string]any{"type": "boolean", "description": "True only for a yes."},
			},
		},
	}
}

// askApproval holds call for the caller's yes and returns what the model is answered with
// at once. A cascade owes the caller the question as written (respondAfterTool); a native
// model is told to ask it.
func (a *Agent) askApproval(call llm.ToolCall, question string) []llm.ContentPart {
	question = fillQuestion(question, call.Arguments)
	answer := approvalAnswer
	native := a.native()
	if native {
		answer = nativeApprovalAnswer
	}
	a.mu.Lock()
	if a.approvals == nil {
		a.approvals = map[string]*approval{}
	}
	a.approvals[call.ID] = &approval{call: call, question: question}
	if !native {
		a.approvalQuestions = append(a.approvalQuestions, question)
	}
	a.mu.Unlock()
	a.logger.Info("a call waits for the caller's yes", "tool", call.Name, "tool_call", call.ID)
	return llm.TextParts(fmt.Sprintf(answer, call.Name, question, call.ID, call.Name))
}

// askedApproval marks the questions a native model was answered with as asked.
func (a *Agent) askedApproval(callID string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	if waiting := a.approvals[callID]; waiting != nil {
		waiting.asked = true
	}
}

// takeApprovalQuestionsLocked returns the questions a cascade owes the caller. The caller
// holds the lock.
func (a *Agent) takeApprovalQuestionsLocked() string {
	if len(a.approvalQuestions) == 0 {
		return ""
	}
	line := strings.Join(a.approvalQuestions, " ")
	a.approvalQuestions = nil
	return line
}

// askLocked says an approval question as written, in place of the reply to the call that
// asked it, and unlocks. It goes into the history as the agent's turn, so the yes that comes
// back answers a question the model can see.
func (a *Agent) askLocked(turnID, question string) error {
	a.toolReply = false
	a.speakingTurn = turnID
	a.saying = question
	said := llm.Message{Role: llm.Assistant, Content: question}
	if callsOpen(a.history) {
		a.lateResults = append(a.lateResults, said)
	} else {
		a.history = append(a.history, said)
	}
	a.mu.Unlock()

	a.turns.started(turnID, kindTool)
	err := a.speakWhole(turnID, question)
	// Marked only once the voice has the question, so its wait cannot start before it plays.
	a.mu.Lock()
	for _, waiting := range a.approvals {
		waiting.asked = true
	}
	a.mu.Unlock()
	if err != nil {
		return err
	}
	a.emitter.Send(Responded{TurnID: turnID, Text: question})
	return nil
}

// confirm runs the call the caller answered, when they said yes in time, and returns what
// the model is told. The confirmation's own context is the call's: its timeout starts now.
func (a *Agent) confirm(ctx context.Context, confirmation llm.ToolCall) ([]llm.ContentPart, error) {
	var answer struct {
		ToolCallID string `json:"tool_call_id"`
		Allowed    bool   `json:"allowed"`
	}
	if err := json.Unmarshal([]byte(confirmation.Arguments), &answer); err != nil {
		return nil, fmt.Errorf("agent: %s's arguments: %w", confirmTool, err)
	}
	a.mu.Lock()
	waiting, lapsed := a.approvalLocked(answer.ToolCallID)
	if waiting != nil {
		delete(a.approvals, waiting.call.ID)
	}
	a.mu.Unlock()
	switch {
	case lapsed != "":
		return llm.TextParts(fmt.Sprintf(approvalLapsed, lapsed, lapsed)), nil
	case waiting == nil:
		return llm.TextParts(approvalUnknown), nil
	}
	call := waiting.call
	a.emitter.Send(ToolApprovalDecided{ID: call.ID, TurnID: call.TurnID, Allowed: answer.Allowed})
	a.logger.Info("the caller answered a call's question", "tool", call.Name, "tool_call", call.ID, "allowed", answer.Allowed)
	if !answer.Allowed {
		return llm.TextParts(fmt.Sprintf(approvalDeclined, call.Name)), nil
	}
	if a.options.ToolRunner == nil {
		return nil, fmt.Errorf("agent: %s is not a tool this agent can run", call.Name)
	}
	a.mu.Lock()
	if a.approvedRuns == nil {
		a.approvedRuns = map[string]approvedRun{}
	}
	a.approvedRuns[call.ID] = approvedRun{confirmation: confirmation.ID, tool: call.Name}
	a.mu.Unlock()
	defer func() {
		a.mu.Lock()
		delete(a.approvedRuns, call.ID)
		a.mu.Unlock()
	}()
	return a.options.ToolRunner.Run(WithApproval(ctx), call)
}

// approvalLocked finds the call a confirmation answers: the one it names, or the only one
// waiting when it names none the agent holds, since a model can garble an id it read once.
// A call whose wait had passed is named in lapsed. The caller holds the lock.
func (a *Agent) approvalLocked(id string) (waiting *approval, lapsed string) {
	if found, ok := a.approvals[id]; ok {
		waiting = found
	} else if len(a.approvals) == 1 {
		for _, only := range a.approvals {
			waiting = only
		}
	}
	if waiting == nil {
		if tool, ok := a.lapsedApprovals[id]; ok {
			return nil, tool
		}
		return nil, ""
	}
	return waiting, ""
}

// approvedName is the tool a confirmation would run, for the policy it runs under, or the
// confirmation's own name when it names nothing waiting.
func (a *Agent) approvedName(call llm.ToolCall) string {
	if call.Name != confirmTool {
		return call.Name
	}
	var answer struct {
		ToolCallID string `json:"tool_call_id"`
	}
	_ = json.Unmarshal([]byte(call.Arguments), &answer)
	a.mu.Lock()
	defer a.mu.Unlock()
	if waiting, _ := a.approvalLocked(answer.ToolCallID); waiting != nil {
		return waiting.call.Name
	}
	return call.Name
}

// runningAs is the call a running call's id is held under: an approved call runs under the
// confirmation that ran it, which is the call the model has open. The caller holds the lock.
func (a *Agent) runningAsLocked(id string) string {
	if run, ok := a.approvedRuns[id]; ok {
		return run.confirmation
	}
	return id
}

// tickApprovals starts each asked question's wait once nothing is being said, and declines
// a call whose wait has passed.
func (a *Agent) tickApprovals() {
	quiet := !a.waitingForPlayout()
	now := time.Now()
	wait := a.options.ApprovalWait
	if wait <= 0 {
		wait = approvalWait
	}
	a.mu.Lock()
	quiet = quiet && !a.generating && a.utterances == 0 && !a.nativeAwaiting
	var lapsed []*approval
	for id, waiting := range a.approvals {
		switch {
		case !waiting.asked:
		case waiting.expires.IsZero():
			if quiet {
				waiting.expires = now.Add(wait)
			}
		case now.After(waiting.expires):
			delete(a.approvals, id)
			lapsed = append(lapsed, waiting)
			if a.lapsedApprovals == nil {
				a.lapsedApprovals = map[string]string{}
			}
			a.lapsedApprovals[id] = waiting.call.Name
			if a.native() {
				continue
			}
			note := llm.Message{Role: llm.User, Content: fmt.Sprintf(approvalSilence,
				waiting.question, int(wait.Round(time.Second)/time.Second), waiting.call.Name, waiting.call.Name)}
			if callsOpen(a.history) {
				a.lateResults = append(a.lateResults, note)
			} else {
				a.history = append(a.history, note)
			}
		}
	}
	a.mu.Unlock()
	for _, waiting := range lapsed {
		a.logger.Info("nobody said yes to a call in time", "tool", waiting.call.Name, "tool_call", waiting.call.ID)
		a.emitter.Send(ToolApprovalDecided{ID: waiting.call.ID, TurnID: waiting.call.TurnID, Summary: "Not confirmed in time"})
	}
}
