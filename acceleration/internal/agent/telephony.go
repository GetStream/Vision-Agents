package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// The tools the agent knows how to run. A tool the model asks for by any other name is
// reported and refused, which is what tells the model to say something instead.
const (
	toolTransfer = "transfer"
	toolPress    = "press"
	toolLookup   = "lookup"
)

// arrivalTimeout bounds how long a warm transfer waits for the human to pick up. Past it
// the summary is worth nothing: either nobody answered, or they have been listening to the
// caller for long enough to have been introduced already.
const arrivalTimeout = 45 * time.Second

// arrivalPoll is how often the wait checks whether a new voice is on the call.
const arrivalPoll = 250 * time.Millisecond

// NavigatingInstructions is what to tell an agent that is calling somebody rather than
// being called, so it can get through whatever answers.
//
// It is a preset rather than the default because the two situations want opposite things.
// An agent taking calls should speak up: silence on a support line is a caller wondering
// whether anyone is there. An agent placing them should keep quiet until it is asked
// something, because most of what it hears at the start is a recording that cannot hear it
// back, and talking over a menu only makes the menu start again.
const NavigatingInstructions = `You have called this number, so you are the one who wants
something. Whoever answers may be a person or a recording.

A recording is not listening to you. Let it finish, then do what it asked: press the option
it named if it offered a keypad, or say the one word it asked for if it did not. Never
explain yourself to a recording and never press at a person, who cannot hear the tones.

When a person answers, say who you are and what you want in one sentence, and then have the
conversation.`

// Telephony is what the agent can do to the call it is on, as opposed to what it can say
// on it.
//
// It exists for the same reason Edge does. The agent decides that a caller should be handed
// to a person, and something else knows what a trunk is; keeping the two apart is what lets
// the decision be tested without a phone network.
type Telephony interface {
	// Transfer brings a human onto the call. It returns once they have been dialled,
	// which is before they answer: there is no ringing to wait through at this end.
	Transfer(ctx context.Context, to string) error
	// SendDigits presses digits on the keypad of the call the agent placed, which is how
	// it gets past a menu.
	SendDigits(ctx context.Context, digits string) error
}

// ToolRunner carries out the tools this package does not know about, which is every tool
// but the two that act on the phone call.
//
// It exists so a caller who is not in this process can own its own tools: the Python SDK
// registers functions against a session, the model asks for one by name, and the runner is
// what takes the request out to whoever can answer it. The result is words for the model,
// so a runner that failed says so in a sentence rather than returning an error the caller
// would never hear about.
type ToolRunner interface {
	Run(ctx context.Context, call llm.ToolCall) (string, error)
}

// telephonyTool reports whether a tool is one of the two this package runs itself.
func telephonyTool(name string) bool {
	return name == toolTransfer || name == toolPress
}

// transferArguments is what the model fills in to hand the caller over. A summary makes it
// a warm transfer and its absence makes it a cold one, because the difference between the
// two is only whether the human is told anything before the agent goes.
type transferArguments struct {
	To      string `json:"to"`
	Summary string `json:"summary"`
}

// pressArguments is what the model fills in to press a menu option.
type pressArguments struct {
	Digits string `json:"digits"`
}

// runTool carries out one call the model asked for and tells it what happened.
//
// Whatever the tool did, the model is told in words: a tool that failed is a fact about the
// conversation, and a model that is told the transfer did not go through can apologise for
// it instead of waiting for a caller who is no longer being handed anywhere.
func (a *Agent) runTool(requested harness.ToolRequested) {
	result, left, err := a.callTool(requested.Call)
	if err != nil {
		result = fmt.Sprintf("That did not work: %s. Tell the caller, in your own words.", err)
	}
	a.resolveTool(requested.Call, result)

	a.emitter.Send(ToolRan{
		TurnID:    requested.TurnID,
		Tool:      requested.Call.Name,
		Arguments: requested.Call.Arguments,
		Result:    result,
		Err:       err,
	})
	if !left {
		// Only a failure is worth breaking silence for. A tool that worked has already
		// said what it needs to in its result, and pressing at a menu asks for quiet
		// until the menu answers.
		//
		// A turn that is itself the answer to a failed tool does not get another, or a
		// model that responds to a broken trunk by trying it again would keep the call
		// and the bill going without the caller hearing a word.
		if err != nil && !strings.HasPrefix(requested.TurnID, toolPrefix) {
			if err := a.respondAfterTool(toolPrefix + turnStamp()); err != nil {
				a.fail(err, "llm")
			}
		}
		return
	}
	// Closing waits for the goroutine this is running on, so the agent cannot end its own
	// call from inside the loop handling the request to end it.
	go func() {
		if err := a.Close(); err != nil {
			a.logger.Error("could not leave after transferring", "error", err)
		}
	}()
}

// callTool runs one tool, reporting what to tell the model and whether the agent has left
// the call.
func (a *Agent) callTool(call llm.ToolCall) (string, bool, error) {
	a.mu.Lock()
	ctx := a.ctx
	a.mu.Unlock()
	if ctx == nil {
		return "", false, errors.New("agent: not joined")
	}

	if telephonyTool(call.Name) {
		if a.options.Telephony == nil {
			return "", false, errors.New("agent: this call has no telephony")
		}
		if call.Name == toolTransfer {
			return a.transfer(ctx, call)
		}
		return a.press(ctx, call)
	}

	if call.Name == toolLookup {
		if a.knowledge == nil {
			return "", false, errors.New("agent: this agent has no knowledge base")
		}
		return a.lookup(ctx, call)
	}

	if a.options.ToolRunner == nil {
		return "", false, fmt.Errorf("agent: %s is not a tool this agent can run", call.Name)
	}
	// A remote tool cannot end the call: leaving is the transfer's business, and a runner
	// that hung up would strand the conversation the agent is still holding history for.
	result, err := a.options.ToolRunner.Run(ctx, call)
	return result, false, err
}

// availableTools is what the voice model is offered, which is the configured set minus
// whatever nothing on this call can carry out. Offering a transfer the agent cannot make
// would have it promise the caller a person and then sit there.
func (a *Agent) availableTools() harness.Tools {
	var available harness.Tools
	for _, tool := range a.options.Tools.Tools {
		runnable := a.options.ToolRunner != nil
		switch {
		case telephonyTool(tool.Name):
			runnable = a.options.Telephony != nil
		case tool.Name == toolLookup:
			runnable = a.knowledge != nil
		}
		if runnable {
			available.Tools = append(available.Tools, tool)
		}
	}
	return available
}

// transfer hands the caller to a human and leaves.
//
// A warm transfer waits for the human to be on the call before saying anything, because a
// summary spoken to an empty seat introduces nobody. Everyone in the call hears it, the
// caller included, which is why the model is told to write it as though they were listening.
func (a *Agent) transfer(ctx context.Context, call llm.ToolCall) (string, bool, error) {
	var arguments transferArguments
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", false, fmt.Errorf("agent: could not read the transfer arguments: %w", err)
	}
	if strings.TrimSpace(arguments.To) == "" {
		return "", false, errors.New("agent: a transfer needs a number to transfer to")
	}

	known := a.heardSoFar()
	if err := a.options.Telephony.Transfer(ctx, arguments.To); err != nil {
		return "", false, fmt.Errorf("agent: transfer to %s: %w", arguments.To, err)
	}
	a.emitter.Send(Transferred{
		TurnID:  call.ID,
		To:      arguments.To,
		Summary: arguments.Summary,
	})

	summary := strings.TrimSpace(arguments.Summary)
	if summary == "" {
		return "The caller has been transferred and you have left the call.", true, nil
	}
	if !a.awaitArrival(ctx, known) {
		a.logger.Info("nobody answered the transfer in time, so the summary was not spoken",
			"to", arguments.To)
		return "The caller has been transferred and you have left the call.", true, nil
	}

	turnID := handoffPrefix + turnStamp()
	a.mu.Lock()
	a.speakingTurn = turnID
	a.mu.Unlock()

	if err := a.speakWhole(turnID, summary); err != nil {
		return "", false, fmt.Errorf("agent: speak the handover summary: %w", err)
	}
	if err := a.Finish(ctx); err != nil {
		return "", false, fmt.Errorf("agent: finish the handover summary: %w", err)
	}
	return "The caller has been introduced and transferred, and you have left the call.", true, nil
}

// press presses digits at a menu.
func (a *Agent) press(ctx context.Context, call llm.ToolCall) (string, bool, error) {
	var arguments pressArguments
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", false, fmt.Errorf("agent: could not read the press arguments: %w", err)
	}
	digits := strings.TrimSpace(arguments.Digits)
	if digits == "" {
		return "", false, errors.New("agent: pressing needs digits to press")
	}

	if err := a.options.Telephony.SendDigits(ctx, digits); err != nil {
		return "", false, fmt.Errorf("agent: press %s: %w", digits, err)
	}
	a.emitter.Send(Pressed{TurnID: call.ID, Digits: digits})
	return fmt.Sprintf("You pressed %s. Wait for the menu to answer before saying anything.",
		digits), false, nil
}

// resolveTool records what a tool returned, so the next turn carries the outcome.
//
// The result is a message rather than a note because the model asked for it by name: a
// provider matches every call against a result, and a conversation that replays the call
// without one is refused.
func (a *Agent) resolveTool(call llm.ToolCall, result string) {
	a.mu.Lock()
	a.history = append(a.history, llm.Message{
		Role:       llm.ToolResult,
		Content:    result,
		ToolCallID: call.ID,
	})
	a.mu.Unlock()
}

// heardSoFar is who the agent has opened a transcription session for, which is everyone it
// has heard speak.
func (a *Agent) heardSoFar() map[string]struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()

	known := make(map[string]struct{}, len(a.listeners))
	for id := range a.listeners {
		known[id] = struct{}{}
	}
	return known
}

// awaitArrival waits for a voice the agent had not heard before, reporting whether one
// arrived.
//
// Someone joining is not announced to the agent, but the first thing a person does on
// answering a call is say hello, and a participant the agent has never heard is a
// participant who has just started talking. That is close enough to an arrival to introduce
// the caller to.
func (a *Agent) awaitArrival(ctx context.Context, known map[string]struct{}) bool {
	deadline, cancel := context.WithTimeout(ctx, arrivalTimeout)
	defer cancel()

	ticker := time.NewTicker(arrivalPoll)
	defer ticker.Stop()

	for {
		select {
		case <-deadline.Done():
			return false
		case <-ticker.C:
			if a.arrived(known) {
				return true
			}
		}
	}
}

// arrived reports whether anyone is being listened to who was not before.
func (a *Agent) arrived(known map[string]struct{}) bool {
	a.mu.Lock()
	defer a.mu.Unlock()

	for id := range a.listeners {
		if _, heard := known[id]; !heard {
			return true
		}
	}
	return false
}
