package tools

import (
	"context"
	"errors"
	"sync"
)

// Call is the call a function is running for, as the router described it.
type Call struct {
	// ID is the id the model gave the call. The router sends the same id when it sends a
	// call again, so it is the key to make a side effect happen once.
	ID string
	// SessionID is the session that asked.
	SessionID string
	// TurnID is the turn the call belongs to.
	TurnID string
	// User is the person whose turn asked for it. Empty when the router did not say.
	User User
	// Conversation is where the call was made. Empty when the router did not say.
	Conversation Conversation
	// Approved says the caller already said yes to this call, when the function asked
	// with NeedsApproval and the router asked them on a voice call.
	Approved bool
}

// Line is how a running call reaches the session it runs for. The socket that delivered the
// call puts it in the call's context (WithLine); without one, as outside a call, Update, Say
// and DisallowInterruptions do nothing.
type Line interface {
	// Update says how the call is going, or with verbatim, words to say as written.
	Update(ctx context.Context, id, text string, verbatim bool) error
	// Hold asks for the call to keep going when the caller talks over the agent, and
	// reports whether it was still running to be held.
	Hold(ctx context.Context, id string) (bool, error)
}

// ErrNotHeld is why the context DisallowInterruptions returns is cancelled when the call
// had already been stopped.
var ErrNotHeld = errors.New("tools: the call was stopped before it could be held")

// Update says how the call is going. The agent tells a voice caller in its own words once
// nobody is talking, and a conversation shows it on the call's step until the call ends. A
// newer update replaces one not yet said, and the result replaces both.
func (c *Call) Update(ctx context.Context, text string) error {
	line := lineFrom(ctx)
	if line == nil {
		return nil
	}
	return line.Update(ctx, c.ID, text, false)
}

// Say has the agent say text as written while the call runs, or post it in a text
// conversation. A speech-to-speech model is asked for these words and may paraphrase them.
func (c *Call) Say(ctx context.Context, text string) error {
	line := lineFrom(ctx)
	if line == nil {
		return nil
	}
	return line.Update(ctx, c.ID, text, true)
}

// DisallowInterruptions keeps the call going from now on when the caller talks over the
// agent, which is what a write that must not be left halfway wants. It returns the context
// to do that write on: only the call's timeout and its socket ending cancel it, and not the
// router's tool_cancel. When the call was stopped before it could be held, the context
// returned is already cancelled, so the write fails before it starts.
func (c *Call) DisallowInterruptions(ctx context.Context) context.Context {
	run, _ := ctx.Value(runKey{}).(*callRun)
	line := lineFrom(ctx)
	if line == nil || run == nil {
		return ctx
	}
	held, err := line.Hold(ctx, c.ID)
	if err == nil && !held {
		err = ErrNotHeld
	}
	if err != nil {
		stopped, stop := context.WithCancelCause(ctx)
		stop(err)
		return stopped
	}
	writing, stop := context.WithCancel(context.WithoutCancel(ctx))
	if deadline, ok := ctx.Deadline(); ok {
		var stopAtDeadline context.CancelFunc
		writing, stopAtDeadline = context.WithDeadline(writing, deadline)
		cancelBoth := stop
		stop = func() { stopAtDeadline(); cancelBoth() }
	}
	unwatch := context.AfterFunc(run.outer, stop)
	run.mu.Lock()
	run.releases = append(run.releases, func() { unwatch(); stop() })
	run.mu.Unlock()
	return writing
}

type (
	runKey  struct{}
	lineKey struct{}
)

// WithLine returns a context whose calls reach their session through line. The socket that
// delivers a call sets it; a test of a function's progress can too.
func WithLine(ctx context.Context, line Line) context.Context {
	return context.WithValue(ctx, lineKey{}, line)
}

func lineFrom(ctx context.Context) Line {
	line, _ := ctx.Value(lineKey{}).(Line)
	return line
}

// callRun is one run of a call: the context it was delivered on, which ends with its socket,
// and what DisallowInterruptions set up, released once the function returns.
type callRun struct {
	outer    context.Context
	mu       sync.Mutex
	releases []func()
}

// startRun returns ctx carrying a run of a call delivered on it, and what ends the run.
func startRun(ctx context.Context) (context.Context, func()) {
	run := &callRun{outer: ctx}
	return context.WithValue(ctx, runKey{}, run), func() {
		run.mu.Lock()
		releases := run.releases
		run.releases = nil
		run.mu.Unlock()
		for _, release := range releases {
			release()
		}
	}
}

// User is the person whose turn asked for a call.
type User struct {
	ID     string
	Name   string
	Custom map[string]any
}

// Conversation is the Chat channel a call was made in, and the session's own labels.
type Conversation struct {
	CID    string
	Custom map[string]any
}

type callKey struct{}

// ToolCallContextFrom returns the call a function is running for. Outside a call, as in a
// unit test of the function itself, it returns an empty one rather than nil.
func ToolCallContextFrom(ctx context.Context) *Call {
	if toolCallContext, ok := ctx.Value(callKey{}).(*Call); ok {
		return toolCallContext
	}
	return &Call{}
}

// WithToolCallContext returns a context carrying toolCallContext, which ToolCallContextFrom
// reads back. Run sets it; a test running a function directly can too.
func WithToolCallContext(ctx context.Context, toolCallContext *Call) context.Context {
	return context.WithValue(ctx, callKey{}, toolCallContext)
}
