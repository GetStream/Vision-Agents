package session

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// approvalWait is how long a call held for approval waits for its person: ten minutes, as long
// as the consent a call waiting for a login waits on lives (Consents, api.attemptLifetime).
const approvalWait = 10 * time.Minute

// approvals are a dispatcher's calls held for a person to allow them, by call id, and the
// conversation that asks.
type approvals struct {
	// await shows the question and the call's arguments on the step of a call of the reply being
	// written, for owner, and returns the command the reply answers (conversation.AwaitApproval).
	// waits reports whether the step still waits (conversation.AwaitsApproval), and decide
	// settles it if it still does, and reports whether it did (conversation.DecideApproval).
	// Without await, as in a session with no conversation, nobody is asked.
	await  func(owner, callID, arguments string, approval persistent.ToolApproval) (string, bool)
	waits  func(callID string) bool
	decide func(agent.ToolApprovalDecided) bool
	// wait is how long a call waits, approvalWait when zero.
	wait time.Duration

	mu   sync.Mutex
	held map[string]heldCall
}

// heldCall is a call waiting for its person, who answers it on the command its reply answers.
type heldCall struct {
	commandID string
	answer    chan approvalAnswer
}

type approvalAnswer struct {
	allowed bool
	summary string
}

// asksIn has the approvals ask in conv, the session's own conversation.
func (a *approvals) asksIn(conv *persistent.Conversation) {
	a.await, a.waits, a.decide = conv.AwaitApproval, conv.AwaitsApproval, conv.DecideApproval
}

// holdForApproval holds a call until the session's user allows it on its step, and reports
// whether they did. Otherwise nothing is sent, and it returns what the model reads instead: the
// person declined, nobody answered in approvalWait, or the reply ended first. A session with
// nobody to ask refuses the call at once: one with no conversation, as an incognito run has, a
// shared one, or one opened as the app, as an MCP event's is (conversation.CanAskToConnect).
//
// The turn ending stops the wait whatever the binding's policy says, as for a call still
// waiting for a slot: it was never sent.
func (d *dispatcher) holdForApproval(ctx context.Context, r route, call llm.ToolCall) (bool, []llm.ContentPart, failure, error) {
	a := &d.approvals
	answer := make(chan approvalAnswer, 1)
	// The call is held before its question is shown, under the lock an answer takes, so an
	// answer that comes at once finds it (answerApproval).
	a.mu.Lock()
	commandID, asked := "", false
	if a.await != nil {
		commandID, asked = a.await(d.spec.Caller.UserID, call.ID, call.Arguments, persistent.ToolApproval{
			Title: fmt.Sprintf("Allow %s to run %s?", r.binding.Name, r.tool)})
	}
	if asked {
		if a.held == nil {
			a.held = map[string]heldCall{}
		}
		a.held[call.ID] = heldCall{commandID: commandID, answer: answer}
	}
	a.mu.Unlock()
	if !asked {
		return false, approvalText("unavailable", fmt.Sprintf("%s runs only once the user allows it, and nobody can be "+
				"asked here, so it was not run. Tell the user it needs their approval in a chat of their own.", call.Name)),
			failure{kind: store.InvocationDenied, message: "declined: nobody can approve it here"}, nil
	}
	defer func() {
		a.mu.Lock()
		delete(a.held, call.ID)
		a.mu.Unlock()
	}()
	expiry := time.NewTimer(cmp.Or(a.wait, approvalWait))
	defer expiry.Stop()
	// The reply can end and leave its calls running, as an interruption that lets tools finish
	// does (agent.stopTools); such a call stops waiting rather than run for nobody.
	poll := time.NewTicker(loginPoll)
	defer poll.Stop()
	stopped := func() (bool, []llm.ContentPart, failure, error) {
		return false, approvalText("stopped", fmt.Sprintf("The reply was stopped before the user allowed %s, so it was "+
			"not run.", call.Name)), failure{kind: store.InvocationDenied, message: "declined: the reply ended before the person answered"}, nil
	}
	for {
		select {
		case <-ctx.Done():
			return false, nil, failure{kind: store.InvocationClientTimeout},
				stack.Wrap(fmt.Errorf("session: %s stopped waiting for its approval: %w", call.Name, ctx.Err()))
		case got := <-answer:
			if !a.decide(agent.ToolApprovalDecided{ID: call.ID, TurnID: call.TurnID, Allowed: got.allowed, Summary: got.summary}) {
				return stopped()
			}
			if got.allowed {
				return true, nil, failure{}, nil
			}
			return false, approvalText("declined", fmt.Sprintf("The user declined %s, so it was not "+
					"run. Do not call it again unless they ask.", call.Name)),
				failure{kind: store.InvocationDenied, message: "declined by the person"}, nil
		case <-expiry.C:
			a.decide(agent.ToolApprovalDecided{ID: call.ID, TurnID: call.TurnID, Summary: "Not allowed in time"})
			return false, approvalText("declined", fmt.Sprintf("Nobody allowed %s in time, so it "+
					"was declined and not run. Tell the user, and that they can ask again.", call.Name)),
				failure{kind: store.InvocationDenied, message: "declined by timeout"}, nil
		case <-poll.C:
			if !a.waits(call.ID) {
				return stopped()
			}
		}
	}
}

// answerApproval hands a person's answer to the call id held for them, when it names the
// command the call's reply answers. It reports whether it did. The call is taken as it is
// answered, so it is answered once.
func (d *dispatcher) answerApproval(callID, commandID string, allowed bool, summary string) bool {
	d.approvals.mu.Lock()
	held, ok := d.approvals.held[callID]
	ok = ok && held.commandID == commandID
	if ok {
		delete(d.approvals.held, callID)
	}
	d.approvals.mu.Unlock()
	if ok {
		held.answer <- approvalAnswer{allowed: allowed, summary: summary}
	}
	return ok
}

// approvalText is what the model reads of a call that waited for approval and was not run.
func approvalText(status, message string) []llm.ContentPart {
	raw, _ := json.Marshal(struct {
		Status  string `json:"status"`
		Message string `json:"message"`
	}{status, message})
	return llm.TextParts(string(raw))
}

// AnswerConnectorApproval hands a person's answer to a connector call the router holds for
// their approval, by the call's id and the command its reply answers. Allowed, the call is
// sent; declined, it is not, and the model is told so. Only the session's own user or its
// customer's backend may answer. It reports whether a call was held for the answer.
func (s *Session) AnswerConnectorApproval(id, commandID string, allowed bool, summary string, by Owner) bool {
	mayAnswer := by.CustomerID == s.spec.CustomerID &&
		(by.Kind == auth.KindServer || by.UserID == s.spec.Caller.UserID)
	return s.connectors != nil && mayAnswer && s.connectors.answerApproval(id, commandID, allowed, summary)
}
