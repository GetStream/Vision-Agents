package tui

import (
	"context"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// Session is the conversation the terminal UI is attached to. Agent adapts an
// *agents.Session to it, and anything else that can answer questions and say what it is
// doing while it answers them can satisfy it too.
type Session interface {
	// ID identifies the live session, which is what a browser opens to watch it and what
	// resumes the conversation.
	ID() string
	// ConversationID is the channel the conversation is saved under.
	ConversationID() string
	// ContextTruncated says whether the session was given less than the whole history.
	ContextTruncated() bool
	// Events carries what the agent is thinking, running and saying. It closes when the
	// session ends.
	Events() <-chan stream.Event
	// Respond asks a question. The answer arrives on Events.
	Respond(text string) error
	// Interrupt stops the answer being worked on.
	Interrupt() error
	// Close ends the session.
	Close(ctx context.Context) error
}

// Agent adapts a Go SDK session, which asks through its Responses, to a Session.
func Agent(session *agents.Session) Session { return agentSession{session} }

type agentSession struct{ *agents.Session }

func (s agentSession) Respond(text string) error {
	_, err := s.Responses.Create(context.Background(), text)
	return err
}

// Opener resumes the session of an id. An empty id starts a new conversation; the id the
// session settles on is read back from it. The terminal UI closes whatever it
// opened, including the session an earlier call returned.
type Opener func(ctx context.Context, sessionID string) (Session, error)

// History reads a page of saved messages older than a cursor. An empty cursor reads the
// newest page, and the cursor for the page before it comes back in the page.
type History func(ctx context.Context, conversationID, before string) (stream.ConversationPage, error)

// BackendHistory reads saved history from a router, which is what an agent whose
// conversations are persisted has. agentUserID is the user the agent writes as.
func BackendHistory(backend stream.Backend, agentUserID string) History {
	return func(ctx context.Context, conversationID, before string) (stream.ConversationPage, error) {
		return backend.ConversationHistory(ctx, conversationID, agentUserID, before)
	}
}

// State is what the terminal UI knows about the conversation, offered to a caller
// deciding what else the header should say.
type State struct {
	// ConversationID is empty until the first session is open.
	ConversationID string
	// SessionID identifies the live session behind the conversation.
	SessionID string
	// Scope is the product and SDK the most recent tool worked in, as "chat / react".
	// Empty until a tool has reported one.
	Scope string
	// Connecting is set while a session is being opened, Busy while an answer is being
	// worked on.
	Connecting, Busy bool
	// Truncated says the session was given less than the whole history.
	Truncated bool
}
