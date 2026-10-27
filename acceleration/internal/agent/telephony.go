package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/harness"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sandbox"
)

// The tools the agent knows how to run. A tool the model asks for by any other name is
// reported and refused, which is what tells the model to say something instead.
const (
	toolTransfer = "transfer"
	toolPress    = "press"
	toolLookup   = "lookup"
	toolSearch   = "search"
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
	Run(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, error)
}

// ToolPolicy is what one tool asks of the agent (Options.ToolPolicy). The zero value asks
// for nothing.
type ToolPolicy struct {
	// PreSpeech is what to say while the tool runs, in place of a line the model writes
	// for the moment (holdPurpose), and what ToolStarted carries. Empty says the model's.
	PreSpeech string
	// Waits is set for a tool its runner lets finish after its context ends. The call is
	// answered then with stillRunning, and its result, when it comes, is added as a
	// message of its own (lateResult).
	Waits bool
	// Cancels is set for a tool whose call is to be cancelled when the caller interrupts,
	// where any other but a telephony tool outlives the interruption.
	Cancels bool
}

// stillRunning answers a call that goes on after an interruption. A provider refuses a
// conversation that replays a call without its result right after it, and the next turn
// must not wait for the call.
const stillRunning = "This is still running. Its result will follow in a later message."

// stillRunningAnswer is stillRunning for one call, naming the tool the model would drop.
func stillRunningAnswer(tool string) string {
	return fmt.Sprintf("%s Only if the caller no longer wants it, write <drop skill=%q/>.", stillRunning, tool)
}

// spokenResult ends what a tool returned when a voice model reads it (spokenResults).
const spokenResult = "The caller is listening, not reading: tell them the part that answers " +
	"them in one to three short sentences, with no links, lists, markup or record ids, " +
	"and offer the rest."

// spokenResults is history as a voice model is given it: every tool result but a telephony
// tool's ends with spokenResult. The fast models read a page of results out whole,
// whatever the instructions said, unless the result itself says not to. It is the same on
// every turn, so a provider that keeps its replies still sees a conversation that only grew.
func spokenResults(history []llm.Message) []llm.Message {
	names := map[string]string{}
	for i, message := range history {
		for _, call := range message.ToolCalls {
			names[call.ID] = call.Name
		}
		if message.Role != llm.ToolResult || telephonyTool(names[message.ToolCallID]) {
			continue
		}
		if len(message.Parts) > 0 {
			message.Parts = append(slices.Clone(message.Parts), llm.ContentPart{Text: spokenResult})
		} else {
			message.Content += "\n" + spokenResult
		}
		history[i] = message
	}
	return history
}

// maxToolRounds is the reply to a tool result, counted since the caller last spoke, that a
// voice call makes answer rather than reach for another tool. Progressive tools spend a
// round describing each tool before running it, so it leaves room for three of them and a
// query corrected once.
const maxToolRounds = 8

// lateResult introduces the result of a call answered with stillRunning. It comes after
// other turns, where a tool result answering an earlier call is refused, so it is the
// caller's message, which every provider takes anywhere.
const lateResult = "The %s call (%s) that was still running has finished. Its result:"

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
	ctx, cancel := a.prepareTool(requested)
	a.executeTool(ctx, cancel, requested)
}

// alreadyCalled opens the result a repeated call is given instead of being run again.
const alreadyCalled = "You already made this exact call and it returned the following. " +
	"Answer the caller from it and do not call it again: "

// didNotWork opens the result of a call that failed.
const didNotWork = "That did not work: "

// answeredSinceCaller returns the result of an identical call, the same tool with the same
// arguments, already answered since the caller last spoke. After a result Gemma often asks
// for the same call again, and run twice a dispatch or a booking is made twice. A call that
// failed may be tried again, since what failed may not fail twice, and so may one whose result
// says nothing was done (undoneResult), and code whose result tells the model to run it again
// now that the user connected what it needed (harness.RunAgain, connected being the bindings
// the user has connected): answered from that result, the repeat after the user connected got
// the 401 from before.
//
// A call that changes something is matched across the whole call: Nemotron booked a table,
// heard "Yes, please go ahead" and booked it again, and the second booking failed against the
// first.
func answeredSinceCaller(history []llm.Message, call llm.ToolCall, connected []string) (string, bool) {
	same := func(other llm.ToolCall) bool {
		return other.Name == call.Name && fillsIn(other.Arguments, call.Arguments)
	}
	changes := !readsOnly(call.Name)
	for i := len(history) - 1; i >= 0 && (changes || history[i].Role != llm.User); i-- {
		if history[i].Role != llm.ToolResult || failedResult(history[i].Content) ||
			undoneResult(history[i].Content) || strings.Contains(history[i].Content, notRun) ||
			call.Name == sandbox.ToolName && harness.RunAgain(history[i].Content, connected) {
			continue
		}
		for j := i - 1; j >= 0 && (changes || history[j].Role != llm.User); j-- {
			if slices.ContainsFunc(history[j].ToolCalls, func(asked llm.ToolCall) bool {
				return asked.ID == history[i].ToolCallID && same(asked)
			}) {
				return history[i].Content, true
			}
		}
	}
	return "", false
}

// routerStatuses are the statuses only the router answers a connector call it held back with
// (internal/session: connector_login.go, step_up.go, connector_tools.go; internal/plugins:
// user.go) until the person signs in or grants a scope. A tool does not use them for its own
// results, so they count on the status alone.
var routerStatuses = []string{"authorization_required", "scope_required", "credential_rejected"}

// sharedStatuses are statuses the router also answers with when a sign-in or an approval ended
// without the person acting (internal/session: connector_login.go, connector_approval.go;
// internal/plugins/user.go; internal/conversation/parts.go) but a third-party tool uses too:
// {"status":"stopped","recording_id":"r1"} is a write that was done. They count only in the
// router's own shape, exactly {status, message}.
var sharedStatuses = []string{"declined", "expired", "failed", "stopped", "unavailable"}

// undoneResult reports whether a tool result is one the router held back (routerStatuses,
// sharedStatuses): nothing was done. Once the person has acted, the same call must run: the
// step-up's own result tells the model to call it again.
func undoneResult(content string) bool {
	var body map[string]json.RawMessage
	if json.Unmarshal([]byte(content), &body) != nil {
		return false
	}
	var status string
	if json.Unmarshal(body["status"], &status) != nil {
		return false
	}
	if slices.Contains(routerStatuses, status) {
		return true
	}
	_, hasMessage := body["message"]
	return slices.Contains(sharedStatuses, status) && hasMessage && len(body) == 2
}

// refusedBefore returns the answer of an identical call the tool already refused since the
// caller last spoke ({"ok": false}, {"error": ...}). Sent again with nothing new from the
// caller, the same values get the same answer. Once the caller has spoken they may have
// confirmed the values or put right what the tool needed, so the call is theirs to make again.
// A call that could not be made, or that failed for a reason of the moment (transientResult),
// was not refused at all.
func refusedBefore(history []llm.Message, call llm.ToolCall) (string, bool) {
	for i := len(history) - 1; i >= 0 && history[i].Role != llm.User; i-- {
		result := history[i]
		if result.Role != llm.ToolResult || strings.HasPrefix(result.Content, didNotWork) ||
			!failedResult(result.Content) || transientResult(result.Content) {
			continue
		}
		for j := i - 1; j >= 0 && history[j].Role != llm.User; j-- {
			if slices.ContainsFunc(history[j].ToolCalls, func(asked llm.ToolCall) bool {
				return asked.ID == result.ToolCallID && asked.Name == call.Name && sameArguments(asked.Arguments, call.Arguments)
			}) {
				return result.Content, true
			}
		}
	}
	return "", false
}

// transientWords are what a tool's error says when it failed for a reason of the moment
// rather than because of the values it was sent.
var transientWords = []string{"timeout", "timed out", "rate limit", "too many requests",
	"service unavailable", "temporarily unavailable"}

// transientResult reports whether a failed tool result says the same call may succeed if made
// again: an HTTP status that says so (llm.TransientStatus) under "status", "status_code" or
// "code", at the top or inside "error", or an error message in transientWords.
func transientResult(content string) bool {
	var body map[string]any
	if json.Unmarshal([]byte(content), &body) != nil {
		return false
	}
	fields := []map[string]any{body}
	var messages []string
	switch failure := body["error"].(type) {
	case string:
		messages = append(messages, failure)
	case map[string]any:
		fields = append(fields, failure)
		if message, ok := failure["message"].(string); ok {
			messages = append(messages, message)
		}
	}
	if message, ok := body["message"].(string); ok {
		messages = append(messages, message)
	}
	for _, field := range fields {
		for _, key := range []string{"status", "status_code", "code"} {
			if status, ok := field[key].(float64); ok && status == float64(int(status)) && llm.TransientStatus(int(status)) {
				return true
			}
		}
	}
	return slices.ContainsFunc(messages, func(message string) bool {
		message = strings.ToLower(message)
		return slices.ContainsFunc(transientWords, func(word string) bool { return strings.Contains(message, word) })
	})
}

// alreadyRefused opens the answer to a call repeated after the tool refused it.
// It opens with didNotWork so that it backs no claim.
const alreadyRefused = didNotWork + "you already made this exact call with these values and it was refused: "

// tryOtherValues closes it.
const tryOtherValues = ". Do not call it again with the same values before the caller says more."

// fillsIn reports whether a later call's arguments are an earlier call's with at most the
// text arguments it left empty or out filled in, and the others it left out given their
// zero value. create_reservation went out without the phone, then again with it and
// "high_chair": false, and the second booking failed against the first.
func fillsIn(earlier, later string) bool {
	if sameArguments(earlier, later) {
		return true
	}
	var before, after map[string]any
	if json.Unmarshal([]byte(earlier), &before) != nil || json.Unmarshal([]byte(later), &after) != nil {
		return false
	}
	for key, value := range before {
		if text, ok := value.(string); ok && strings.TrimSpace(text) == "" {
			continue
		}
		if !reflect.DeepEqual(value, after[key]) {
			return false
		}
	}
	for key, value := range after {
		if _, ok := before[key]; !ok {
			if _, text := value.(string); !text && !reflect.ValueOf(value).IsZero() {
				return false
			}
		}
	}
	return true
}

// sameArguments compares two calls' arguments as JSON, so spacing and key order do not count.
func sameArguments(a, b string) bool {
	var left, right any
	if json.Unmarshal([]byte(a), &left) != nil || json.Unmarshal([]byte(b), &right) != nil {
		return strings.TrimSpace(a) == strings.TrimSpace(b)
	}
	return reflect.DeepEqual(left, right)
}

// notRun marks a result that was never run, so it is not taken for an answer.
const notRun = " was not run: "

// leftBlank is the result of a call that was not run because it left required arguments
// empty.
const leftBlank = "%s was not run: %s left empty. Take them from what the caller has said and " +
	"call it again, or ask the caller for them."

// blankRequired returns the arguments a tool's schema requires that a call left out or empty.
// Made to call, Qwen fills the schema with whatever it has, and verify_account went out
// with the PIN alone although the caller had just given the last four and the address.
func blankRequired(tool harness.Tool, arguments string) []string {
	required, _ := tool.Parameters["required"].([]any)
	if len(required) == 0 {
		return nil
	}
	var given map[string]any
	if json.Unmarshal([]byte(arguments), &given) != nil {
		return nil
	}
	var blank []string
	for _, name := range required {
		key, _ := name.(string)
		value, present := given[key]
		if text, isText := value.(string); !present || value == nil || isText && strings.TrimSpace(text) == "" {
			blank = append(blank, key)
		}
	}
	return blank
}

// prepareTool registers cancellation before native execution starts in a goroutine.
func (a *Agent) prepareTool(requested harness.ToolRequested) (context.Context, context.CancelFunc) {
	cancels := telephonyTool(requested.Call.Name) ||
		a.options.ToolPolicy != nil && a.options.ToolPolicy(a.approvedName(requested.Call)).Cancels
	a.mu.Lock()
	parent := a.ctx
	if parent == nil {
		parent = context.Background()
	}
	if requester := a.requesterLocked(); requester != (Requester{}) {
		parent = WithRequester(parent, requester)
	}
	ctx, cancel := context.WithCancel(parent)
	if a.toolCancels == nil {
		a.toolCancels = map[string]context.CancelFunc{}
	}
	a.toolCancels[requested.Call.ID] = cancel
	if a.toolContexts == nil {
		a.toolContexts = map[string]context.Context{}
	}
	a.toolContexts[requested.Call.ID] = ctx
	if a.toolTurns == nil {
		a.toolTurns = map[string]string{}
	}
	a.toolTurns[requested.Call.ID] = requested.TurnID
	if a.toolsRunning == nil {
		a.toolsRunning = map[string]string{}
	}
	a.toolsRunning[requested.Call.ID] = requested.Call.Name
	// Every tool but the two that act on the phone call, and one whose binding asks for it to
	// be cancelled, outlives an interruption: what it was asked is still what the caller
	// wants, and it is answered when it comes back.
	if !cancels {
		if a.toolHolds == nil {
			a.toolHolds = map[string]*toolHold{}
		}
		a.toolHolds[requested.Call.ID] = &toolHold{}
	} else if _, abandoned := a.abandoned[requested.TurnID]; abandoned {
		cancel()
	}
	a.mu.Unlock()
	return ctx, cancel
}

func (a *Agent) executeTool(ctx context.Context, cancel context.CancelFunc, requested harness.ToolRequested) {
	defer func() {
		cancel()
		a.mu.Lock()
		delete(a.toolCancels, requested.Call.ID)
		delete(a.toolContexts, requested.Call.ID)
		delete(a.toolTurns, requested.Call.ID)
		delete(a.toolHolds, requested.Call.ID)
		a.mu.Unlock()
	}()
	began := time.Now()
	started := ToolStarted{ID: requested.Call.ID, TurnID: requested.TurnID, Tool: requested.Call.Name, Arguments: requested.Call.Arguments, StartedAt: time.Now().UTC()}
	started.Product, started.SDK = toolScope(requested.Call)
	started.PreSpeech = a.preSpeech([]llm.ToolCall{requested.Call})
	if a.options.OnToolStarted != nil {
		a.options.OnToolStarted(started)
	}
	a.emitter.Send(started)
	requested.Call.TurnID = requested.TurnID
	hold := a.holdOnInterrupt(ctx, requested.Call)
	var parts []llm.ContentPart
	var files []sandbox.Attachment
	var left bool
	var err error
	tool, _ := a.availableTools().Lookup(requested.Call.Name)
	// ran is whether the tool was called: the router refuses some calls itself, before it.
	var ran bool
	// What the caller said and the tools returned is checked against the whole call, not the
	// history the model reads: compaction summarises that, and a value it summarised away
	// was still said or returned.
	record := a.record()
	if requested.Call.Name == confirmTool && a.offersConfirm() {
		ran = true
		parts, err = a.confirm(ctx, requested.Call)
	} else if earlier, repeated := answeredSinceCaller(a.History(), requested.Call, a.connected()); repeated {
		parts = llm.TextParts(alreadyCalled + earlier)
	} else if refused, again := refusedBefore(a.History(), requested.Call); again {
		parts = llm.TextParts(alreadyRefused + refused + tryOtherValues)
	} else if key, value, known, unfound := unfoundID(record, requested.Call, toolNames(a.availableTools())); unfound {
		parts = llm.TextParts(unfoundIDResult(key, value, known))
	} else if key, known, missing := missingID(record, requested.Call, a.availableTools()); missing {
		parts = llm.TextParts(unfoundIDResult(key, "", known))
	} else if blank := blankRequired(tool, requested.Call.Arguments); len(blank) > 0 && !telephonyTool(tool.Name) {
		parts = llm.TextParts(fmt.Sprintf(leftBlank, requested.Call.Name, strings.Join(blank, ", ")))
	} else if key, empty := emptyNamed(requested.Call, a.availableTools()); empty {
		parts = llm.TextParts(emptyResult(requested.Call.Name, key))
	} else if key, value, heard, unheard := unheardNumber(record, requested.Call); unheard {
		parts = llm.TextParts(unheardResult(key, value, heard))
	} else if name, spelled, misspelled := misspelledName(record, requested.Call); misspelled {
		parts = llm.TextParts(misspelledResult(name, spelled))
	} else if name, stand := placeholderName(record, requested.Call); stand {
		parts = llm.TextParts(placeholderResult(name))
	} else if requested.Call.Name == sandbox.ToolName && a.runsCode() {
		var said string
		ran = true
		said, files, err = a.runCode(ctx, requested.Call)
		parts = llm.TextParts(said)
	} else if question := a.approvalQuestion(tool); question != "" {
		parts = a.askApproval(requested.Call, question)
	} else {
		ran = true
		parts, left, err = a.callTool(ctx, requested.Call)
	}
	a.mu.Lock()
	delete(a.toolsRunning, requested.Call.ID)
	delete(a.toolUpdates, requested.Call.ID)
	charged := a.chargedLocked(ctx)
	a.mu.Unlock()
	// Visual tool results go to the vision worker in both native and cascade calls. The
	// call is still open in the history it is handed, which a provider refuses to replay.
	// With no worker, as in a chat, a model that sees images is shown them itself.
	if err == nil && a.harness != nil && llm.HasImage([]llm.Message{{Parts: parts}}) && !a.looksItself() {
		_, err = a.harness.Delegate(charged, visionSkill, "Analyze this tool result for the caller's current question.", requested.TurnID, parts, withOpenCallsAnswered(a.History()))
		if err == nil {
			parts = llm.TextParts(llm.TextOf(parts) + "\nVisual analysis requested; wait for the vision findings before interpreting the images.")
		}
	}
	if err != nil {
		parts = llm.TextParts(fmt.Sprintf(didNotWork+"%s. If it names a step one of your "+
			"tools does, call that tool and try again; otherwise tell the caller, in your own words.", err))
	}
	result := llm.TextOf(parts)
	a.mu.Lock()
	if hold == nil {
		hold = a.toolHolds[requested.Call.ID]
	}
	a.resolveToolLocked(requested.Call, parts, hold, ctx.Err() != nil)
	if !left && !a.native() && ctx.Err() == nil && a.followsTool(requested) {
		a.toolReply = true
	}
	outcome := toolOutcome(ctx.Err() != nil, err != nil, !ran, hold != nil && hold.answered)
	a.mu.Unlock()
	took := time.Since(began)
	toolCallSeconds.WithLabelValues(outcome).Observe(took.Seconds())
	a.logger.Info("a tool call came back", "tool", requested.Call.Name, "tool_call", requested.Call.ID,
		"turn", requested.TurnID, "duration_ms", took.Milliseconds(), "outcome", outcome)

	a.emitter.Send(ToolRan{
		ID:        requested.Call.ID,
		TurnID:    requested.TurnID,
		Tool:      requested.Call.Name,
		Arguments: requested.Call.Arguments,
		Result:    result,
		Files:     files,
		Err:       err,
	})
	// A native model is answered on its own session and carries on by itself; there is no
	// follow-up turn to queue because the model speaks the moment it has the result.
	if model := a.speech(); model != nil {
		defer a.noteToolDone(hold)
		if ctx.Err() == nil {
			a.mu.Lock()
			a.nativeAwaiting = true
			a.mu.Unlock()
			err := model.Answer(requested.Call.ID, result, nil)
			if err == nil {
				a.askedApproval(requested.Call.ID)
			} else {
				a.mu.Lock()
				a.nativeAwaiting = false
				a.mu.Unlock()
				capabilities := model.Capabilities()
				if a.switching.Load() {
					a.logger.Warn("could not deliver a tool result while the speech session was being replaced", "error", err)
				} else if !capabilities.Resumable && capabilities.Tools {
					a.loseNativeWrite(model, err)
				} else {
					a.fail(err, "sts")
				}
			}
		}
		if left {
			go func() {
				if err := a.Close(); err != nil {
					a.logger.Error("could not leave after transferring", "error", err)
				}
			}()
		}
		return
	}
	a.noteToolDone(hold)
	if ctx.Err() != nil {
		return
	}
	if !left {
		// A tool result is not something the caller can hear. Whether it worked or not,
		// somebody asked a question and is waiting on the answer, so the agent says it
		// rather than sitting on it until they happen to speak again.
		if a.followsTool(requested) {
			a.queueToolReply()
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

// toolOutcome names how a tool call ended: cancelled, failed, refused by the router before
// the tool was called, come back after it was answered with stillRunning, or ok.
func toolOutcome(cancelled, failed, refused, answered bool) string {
	switch {
	case cancelled:
		return "cancelled"
	case failed:
		return "error"
	case refused:
		return "refused"
	case answered:
		return "still-running"
	}
	return "ok"
}

// followsTool reports whether what a tool returned earns a reply of its own.
//
// Pressing at a menu does not: the digits are the whole point of the tool and the menu is
// what answers next, so talking over it would be talking to nobody. A transfer asked for
// in a reply to a tool does not either, so a broken one cannot keep the line open by
// retrying itself. Other tools chain, a lookup leading to the lookup it needed, and are
// always followed: a result nobody speaks is the answer lost. A voice call that has
// chained maxToolRounds times is made to answer instead (respondAfterTool).
func (a *Agent) followsTool(requested harness.ToolRequested) bool {
	switch {
	case requested.Call.Name == toolPress:
		return false
	case a.options.Text:
		return true
	case telephonyTool(requested.Call.Name):
		return !strings.HasPrefix(requested.TurnID, toolPrefix)
	}
	return true
}

// callTool runs one tool, reporting what to tell the model and whether the agent has left
// the call.
func (a *Agent) callTool(ctx context.Context, call llm.ToolCall) ([]llm.ContentPart, bool, error) {
	if err := ctx.Err(); err != nil {
		return nil, false, err
	}
	if a.native() && (call.Name == delegateSkill || call.Name == cancelSkill) {
		return a.nativeDelegate(call)
	}

	if telephonyTool(call.Name) {
		if a.options.Telephony == nil {
			return nil, false, errors.New("agent: this call has no telephony")
		}
		var result string
		var left bool
		var err error
		if call.Name == toolTransfer {
			result, left, err = a.transfer(ctx, call)
		} else {
			result, left, err = a.press(ctx, call)
		}
		return llm.TextParts(result), left, err
	}

	if call.Name == toolLookup {
		if a.knowledge == nil {
			return nil, false, errors.New("agent: this agent has no knowledge base")
		}
		result, left, err := a.lookup(ctx, call)
		return llm.TextParts(result), left, err
	}

	if call.Name == toolSearch {
		if a.searcher == nil {
			return nil, false, errors.New("agent: this agent cannot search")
		}
		result, left, err := a.search(ctx, call)
		return llm.TextParts(result), left, err
	}

	if call.Name == harness.LoadSkillTool && a.loadsSkills() {
		result, err := a.loadSkill(ctx, call)
		return llm.TextParts(result), false, err
	}

	if a.options.ToolRunner == nil {
		return nil, false, fmt.Errorf("agent: %s is not a tool this agent can run", call.Name)
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
		case tool.Name == toolSearch:
			runnable = a.searcher != nil
		}
		if runnable {
			available.Tools = append(available.Tools, tool)
		}
	}
	if a.offersConfirm() {
		available.Tools = append(available.Tools, confirmToolDefinition())
	}
	if a.loadsSkills() {
		available.Tools = append(available.Tools, a.options.Skills.LoadTool())
	}
	if a.runsCode() {
		available.Tools = append(available.Tools, harness.SandboxTool(a.options.Sandbox))
	}
	return available
}

// loadsSkills reports whether this agent's skills are read by the model holding the
// conversation rather than handed to a subagent: a written conversation with skills and no
// subagent. A tool of the caller's own by the same name keeps the name.
func (a *Agent) loadsSkills() bool {
	if !a.options.Text || a.options.SubagentTarget != "" || len(a.options.Skills.Skills) == 0 {
		return false
	}
	_, declared := a.options.Tools.Lookup(harness.LoadSkillTool)
	return !declared
}

// loadSkill hands the model a skill's instructions to follow itself, or with a path one of
// the files the skill brought, so it reads only the ones it needs.
func (a *Agent) loadSkill(ctx context.Context, call llm.ToolCall) (string, error) {
	var arguments struct {
		Name string `json:"name"`
		Path string `json:"path"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", fmt.Errorf("agent: could not read the load_skill arguments: %w", err)
	}
	name := strings.TrimSpace(arguments.Name)
	if file := strings.TrimSpace(arguments.Path); file != "" {
		content, err := a.options.Skills.File(ctx, name, file)
		if err != nil {
			return "", err
		}
		result, err := json.Marshal(map[string]string{"name": name, "path": file, "content": content})
		return string(result), err
	}
	loaded, err := a.options.Skills.Read(ctx, name)
	if err != nil {
		return "", err
	}
	instructions := loaded.Instructions
	if len(loaded.Files) > 0 {
		instructions += "\n\n" + harness.Listing(name, slices.Sorted(maps.Keys(loaded.Files)), a.runsCode())
		a.mu.Lock()
		if a.skillFiles == nil {
			a.skillFiles = map[string]map[string]string{}
		}
		a.skillFiles[name] = loaded.Files
		a.mu.Unlock()
	}
	result, err := json.Marshal(map[string]string{"name": name, "instructions": instructions})
	return string(result), err
}

// runsCode reports whether the model holding a written conversation runs code itself: it
// has a sandbox and no subagent to hand code to. A tool of the caller's own by the same
// name keeps the name.
func (a *Agent) runsCode() bool {
	if !a.options.Text || a.options.SubagentTarget != "" || a.options.Sandbox == nil {
		return false
	}
	_, declared := a.options.Tools.Lookup(sandbox.ToolName)
	return !declared
}

// connected are the sandbox's late grants' bindings the user has connected by now
// (sandbox.Connecting).
func (a *Agent) connected() []string {
	box, ok := a.options.Sandbox.(sandbox.Connecting)
	if !ok {
		return nil
	}
	return box.Connected()
}

// runCode runs what the model wrote in the sandbox, with the scripts of every skill it has
// loaded put there first, and says what happened. The files the code handed back are
// published and returned for the reply to carry.
//
// Code that ran and failed is described rather than returned as an error: the model can
// only do something sensible about it if it is told how it failed.
func (a *Agent) runCode(ctx context.Context, call llm.ToolCall) (string, []sandbox.Attachment, error) {
	var arguments struct {
		Code  string   `json:"code"`
		Files []string `json:"files"`
	}
	if err := json.Unmarshal([]byte(call.Arguments), &arguments); err != nil {
		return "", nil, fmt.Errorf("agent: could not read the run_code arguments: %w", err)
	}
	if err := a.writeSkillFiles(ctx); err != nil {
		return "", nil, fmt.Errorf("agent: could not put the skills' files in the sandbox: %w", err)
	}
	result, err := a.options.Sandbox.Run(ctx, arguments.Code, arguments.Files)
	if err != nil {
		return "", nil, fmt.Errorf("agent: could not run the code: %w", err)
	}
	said, files := harness.HandBack(ctx, a.options.Publish, result, a.logger)
	return said, files, nil
}

// writeSkillFiles puts the files of each skill loaded so far that brought scripts into the
// sandbox under skills/<name>/, unless they are already there as they now are. A sandbox made
// again, after its provider deleted it, gets all of them again.
func (a *Agent) writeSkillFiles(ctx context.Context) error {
	a.mu.Lock()
	scripted := slices.ContainsFunc(slices.Collect(maps.Values(a.skillFiles)), func(files map[string]string) bool {
		return len(harness.Scripts(slices.Collect(maps.Keys(files)))) > 0
	})
	a.mu.Unlock()
	if !scripted {
		return nil
	}
	generation, err := a.options.Sandbox.Ready(ctx)
	if err != nil {
		return err
	}
	a.mu.Lock()
	if generation != a.sandboxGeneration {
		a.writtenFiles, a.sandboxGeneration = nil, generation
	}
	stale := map[string]map[string]string{}
	for name, files := range a.skillFiles {
		if len(harness.Scripts(slices.Collect(maps.Keys(files)))) > 0 && !maps.Equal(a.writtenFiles[name], files) {
			stale[name] = files
		}
	}
	a.mu.Unlock()
	for _, name := range slices.Sorted(maps.Keys(stale)) {
		files := stale[name]
		for _, file := range slices.Sorted(maps.Keys(files)) {
			if err := a.options.Sandbox.Write(ctx, harness.BundlePath(name, file), []byte(files[file])); err != nil {
				return err
			}
		}
		a.mu.Lock()
		if a.writtenFiles == nil {
			a.writtenFiles = map[string]map[string]string{}
		}
		a.writtenFiles[name] = files
		a.mu.Unlock()
	}
	return nil
}

// Tools names what the agent may do rather than say on this call.
func (a *Agent) Tools() []string {
	var names []string
	for _, tool := range a.availableTools().Tools {
		names = append(names, tool.Name)
	}
	return names
}

// ToolDefinitions is what the model is offered on this call, as it is sent.
func (a *Agent) ToolDefinitions() []llm.Tool { return a.availableTools().Requests() }

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
	a.turns.started(turnID, kindHandoff)

	// A native model has no way to read the summary out, so it is asked to introduce the
	// caller from it instead.
	if model := a.speech(); model != nil {
		if err := model.Prompt("A colleague has just joined the call to take over. Introduce the caller to them from this summary, then stop: " + summary); err != nil {
			return "", false, fmt.Errorf("agent: speak the handover summary: %w", err)
		}
	} else if err := a.speakWhole(turnID, summary); err != nil {
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

// resolveToolLocked records what a tool returned, so the next turn carries the outcome.
// The caller holds the lock while recording both the result and whether it earns a reply.
//
// The result is a message rather than a note because the model asked for it by name: a
// provider matches every call against a result, and a conversation that replays the call
// without one is refused.
//
// A call answered with stillRunning at an interruption (hold) has its result added as the
// caller's message instead (lateResult). While the history ends in calls not all answered
// (callsOpen), that message is held back, so it never comes between a call and its
// result, and added once they are.
//
// A call answered with stillRunning and then cancelled was dropped, and what it returned
// is nothing anybody is waiting for.
func (a *Agent) resolveToolLocked(call llm.ToolCall, parts []llm.ContentPart, hold *toolHold, cancelled bool) {
	message := llm.Message{
		Role:       llm.ToolResult,
		ToolCallID: call.ID,
	}
	late := false
	if hold != nil {
		hold.resolved = true
		if hold.answered && cancelled {
			return
		}
		if hold.answered {
			message = llm.Message{Role: llm.User}
			parts = append(llm.TextParts(fmt.Sprintf(lateResult, call.Name, call.ID)), parts...)
			late = true
		}
	}
	if llm.HasImage([]llm.Message{{Parts: parts}}) {
		message.Parts = parts
	} else {
		message.Content = llm.TextOf(parts)
	}
	if late {
		a.lateResults = append(a.lateResults, message)
	} else {
		a.history = append(a.history, message)
	}
	if !callsOpen(a.history) {
		a.history = append(a.history, a.lateResults...)
		a.lateResults = nil
	}
}

// looksItself reports whether a tool's pictures are shown to the conversation model rather
// than handed to a vision worker: there is no worker to hand them to, as in a chat, and the
// model takes images.
func (a *Agent) looksItself() bool {
	a.mu.Lock()
	current, model := a.harness, a.llm
	a.mu.Unlock()
	return current != nil && !current.Offers(visionSkill) && model != nil && model.Capabilities().Accepts(llm.ModalityImage)
}

// toolPicturesSeen is history with the pictures tools showed the model replaced by a note,
// for a new turn: like a person's photos, a tool's pictures are seen with the reply they
// came back for alone, rather than paid for again on every turn after it.
func toolPicturesSeen(history []llm.Message) []llm.Message {
	if !llm.HasImage(history) {
		return history
	}
	out := append([]llm.Message(nil), history...)
	for i, message := range out {
		if message.Role != llm.ToolResult || !message.HasImage() {
			continue
		}
		var words []llm.ContentPart
		for _, part := range message.Parts {
			if part.Image == nil {
				words = append(words, part)
			}
		}
		words = append(words, llm.ContentPart{Text: pictureSeen})
		out[i].Parts, out[i].Content = nil, llm.TextOf(words)
	}
	return out
}

// pictureSeen stands for a tool's picture once the reply it came back for is over.
const pictureSeen = "[the picture this returned was seen with an earlier reply]"

// callsOpen reports whether history ends in an assistant message with a tool call that no
// result after it answers yet.
func callsOpen(history []llm.Message) bool {
	return len(openCalls(history)) > 0
}

// openCalls returns the tool calls of the assistant message history ends on that no
// result after it answers yet.
func openCalls(history []llm.Message) []llm.ToolCall {
	answered := map[string]bool{}
	i := len(history) - 1
	for ; i >= 0 && history[i].Role == llm.ToolResult; i-- {
		answered[history[i].ToolCallID] = true
	}
	if i < 0 || history[i].Role != llm.Assistant {
		return nil
	}
	var open []llm.ToolCall
	for _, call := range history[i].ToolCalls {
		if !answered[call.ID] {
			open = append(open, call)
		}
	}
	return open
}

// toolHold is one call whose tool goes on after an interruption. Its fields are guarded by
// Agent.mu.
type toolHold struct {
	// answered is set once the call was answered with stillRunning.
	answered bool
	// resolved is set once its result went into the history.
	resolved bool
}

// holdOnInterrupt returns the hold prepareTool registered for call, and answers the call
// with stillRunning the moment ctx ends when its runner lets it finish anyway
// (ToolPolicy.Waits). Nil for a telephony tool, which an interruption cancels.
func (a *Agent) holdOnInterrupt(ctx context.Context, call llm.ToolCall) *toolHold {
	a.mu.Lock()
	hold := a.toolHolds[call.ID]
	a.mu.Unlock()
	if hold != nil && a.options.ToolPolicy != nil && a.options.ToolPolicy(call.Name).Waits {
		context.AfterFunc(ctx, func() {
			a.mu.Lock()
			defer a.mu.Unlock()
			a.answerLocked(call.ID, hold)
		})
	}
	return hold
}

// answerLocked answers a call whose tool is still running with stillRunning, so the next
// turn replays it answered and is not held up by it. The caller holds the lock.
func (a *Agent) answerLocked(callID string, hold *toolHold) {
	if hold.resolved || hold.answered {
		return
	}
	hold.answered = true
	if a.pendingTools > 0 {
		a.pendingTools--
	}
	a.history = append(a.history, llm.Message{
		Role: llm.ToolResult, ToolCallID: callID, Content: stillRunningAnswer(a.toolsRunning[callID]),
	})
	if !callsOpen(a.history) {
		a.history = append(a.history, a.lateResults...)
		a.lateResults = nil
	}
}

// stopTools ends the tools still running when the caller takes the floor: a telephony
// tool is cancelled, and any other is answered with stillRunning and left to finish.
func (a *Agent) stopTools() {
	a.mu.Lock()
	defer a.mu.Unlock()
	for id, cancel := range a.toolCancels {
		if hold := a.toolHolds[id]; hold != nil {
			a.answerLocked(id, hold)
			continue
		}
		cancel()
	}
	a.toolReply = false
}

// HoldTool makes a running call that an interruption would cancel outlive one from now on,
// as a call to a tool that never asked to be cancelled does: answered stillRunning, and its
// result told when it comes. It reports false when the call is not running, which for a
// call that was is an interruption that came first.
func (a *Agent) HoldTool(id string) bool {
	a.mu.Lock()
	defer a.mu.Unlock()
	id = a.runningAsLocked(id)
	if ctx, running := a.toolContexts[id]; !running || ctx.Err() != nil {
		return false
	}
	if a.toolHolds == nil {
		a.toolHolds = map[string]*toolHold{}
	}
	if a.toolHolds[id] == nil {
		a.toolHolds[id] = &toolHold{}
	}
	return true
}

// dropTool cancels the calls to a tool still running, because the caller no longer wants
// what they would return.
func (a *Agent) dropTool(name string) {
	a.mu.Lock()
	defer a.mu.Unlock()
	for id, running := range a.toolsRunning {
		if running != name {
			continue
		}
		if hold := a.toolHolds[id]; hold != nil {
			a.answerLocked(id, hold)
		}
		if cancel := a.toolCancels[id]; cancel != nil {
			cancel()
		}
	}
}

// heardSoFar is who the agent has opened a transcription session for, which is everyone it
// has heard speak.
func (a *Agent) heardSoFar() map[string]struct{} {
	a.mu.Lock()
	defer a.mu.Unlock()

	known := make(map[string]struct{}, len(a.listeners)+len(a.arrivals))
	for id := range a.listeners {
		known[id] = struct{}{}
	}
	// A native agent opens no listeners, so who it has heard is who has sent it audio.
	for id := range a.arrivals {
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
	for id := range a.arrivals {
		if _, heard := known[id]; !heard {
			return true
		}
	}
	return false
}

// toolScope deliberately extracts only recognized routing labels, never raw arguments.
func toolScope(call llm.ToolCall) (string, string) {
	var scope struct {
		Product string
		SDK     string
	}
	if json.Unmarshal([]byte(call.Arguments), &scope) != nil {
		return "", ""
	}
	switch scope.Product {
	case "chat", "video", "moderation", "feeds":
	default:
		return "", ""
	}
	switch scope.SDK {
	case "react", "ios-swiftui", "ios", "react-native", "android", "flutter", "javascript", "unity", "unreal", "go", "python":
	default:
		return scope.Product, ""
	}
	return scope.Product, scope.SDK
}
