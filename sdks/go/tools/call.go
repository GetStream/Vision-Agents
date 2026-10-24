package tools

import "context"

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

// CallFrom returns the call a function is running for. Outside a call, as in a unit test
// of the function itself, it returns an empty one rather than nil.
func CallFrom(ctx context.Context) *Call {
	if call, ok := ctx.Value(callKey{}).(*Call); ok {
		return call
	}
	return &Call{}
}

// WithCall returns a context carrying call, which CallFrom reads back. Run sets it; a test
// running a function directly can too.
func WithCall(ctx context.Context, call *Call) context.Context {
	return context.WithValue(ctx, callKey{}, call)
}
