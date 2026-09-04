// Package agents is the Go SDK for agents that run in the acceleration backend.
//
// An agent here is configuration and function calling. The conversation itself — joining
// the call, hearing the caller, answering and speaking — happens in the backend, and what
// arrives here are the events saying so.
package agents

import (
	"context"
	"errors"
	"log/slog"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/edge"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// UserKey is the memory filter key naming who the memories are about. Everything else in
// the filter narrows recall; this one is what recall is keyed by.
const UserKey = "user_id"

// Options is everything an agent is.
type Options struct {
	// Name is what the agent is called. It names the stored config the folder syncs to and
	// is who the agent appears as in a call.
	Name string
	// Dir is an agent directory holding instructions.md, skills/ and knowledge/. What it
	// says fills in whatever is left empty here.
	Dir string
	// Instructions is the system prompt.
	Instructions string
	// LLM is the pipeline running in the backend.
	LLM *stream.Pipeline
	// Harness is what stands between what a caller said and the model that answers them.
	Harness *Harness
	// CostTracking labels every request the session makes, so spend can be attributed to
	// whatever the labels mean to you rather than only to a model.
	CostTracking map[string]string
	// MemoryFilter is who the memories are about, under "user_id", and what narrows recall.
	MemoryFilter map[string]string
	// Edge creates the Stream calls the backend joins. Nil builds one from the environment
	// the first time a call is needed.
	Edge *edge.Edge
	// UserID is who the agent joins a call as. Empty is derived from the name.
	UserID string
	// Logger is where the agent reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Agent is a configured agent, before and between the calls it holds.
type Agent struct {
	options Options
	folder  *Folder
	logger  *slog.Logger
}

// New validates an agent's configuration and reads its directory, if it has one.
func New(options Options) (*Agent, error) {
	if options.LLM == nil {
		return nil, errors.New("agents: an agent needs an llm")
	}

	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	agent := &Agent{options: options, logger: logger}
	if options.Dir != "" {
		folder, err := Load(options.Dir)
		if err != nil {
			return nil, err
		}
		agent.folder = folder
		folder.fill(&agent.options)
	}

	if agent.options.Name == "" {
		return nil, errors.New("agents: an agent needs a name")
	}
	// Validated after the directory has been read, since what it holds is part of what is
	// being validated.
	if err := agent.options.Harness.Validate(); err != nil {
		return nil, err
	}
	if agent.options.UserID == "" {
		agent.options.UserID = userIDOf(agent.options.Name)
	}
	return agent, nil
}

// Name is what the agent is called.
func (a *Agent) Name() string { return a.options.Name }

// Instructions is the system prompt the agent joins with.
func (a *Agent) Instructions() string { return a.options.Instructions }

// LLM is the pipeline running in the backend.
func (a *Agent) LLM() *stream.Pipeline { return a.options.LLM }

// Functions are the agent's own, which the model is offered and this process runs.
func (a *Agent) Functions() *tools.Registry { return a.options.LLM.Functions() }

// Folder is the agent directory the agent was read from, or nil if it has none.
func (a *Agent) Folder() *Folder { return a.folder }

// Join has the backend join a call and hold a conversation on it.
//
// An empty call id creates one named after a random string, which is what a one-off
// conversation wants. It returns once the backend is in the call, so an agent that has
// joined is one that is already listening.
func (a *Agent) Join(ctx context.Context, call edge.Call) (*Session, error) {
	transport, err := a.edge()
	if err != nil {
		return nil, err
	}
	created, err := transport.CreateCall(ctx, call, edge.User{ID: a.options.UserID, Name: a.options.Name})
	if err != nil {
		return nil, err
	}
	return a.join(ctx, created, nil, false)
}

// Chat holds the conversation in writing rather than on a call.
//
// No call is joined, nothing is transcribed and nothing is spoken. Everything between
// hearing a question and answering it is unchanged: the same instructions, the same skills
// handed to the same slower model and the same knowledge base a call would have had.
func (a *Agent) Chat(ctx context.Context) (*Session, error) {
	return a.join(ctx, edge.Call{}, nil, false)
}

// join renders the agent's configuration into a session and opens it.
func (a *Agent) join(ctx context.Context, call edge.Call, phone *acceleration.SessionPhone, navigating bool) (*Session, error) {
	remote := stream.Call{
		ID:           call.ID,
		Type:         call.Type,
		UserID:       a.options.UserID,
		UserName:     a.options.Name,
		AgentID:      a.options.UserID,
		Instructions: a.options.Instructions,
		Tags:         a.options.CostTracking,
		Memory:       memoryOf(a.options.MemoryFilter),
		Phone:        phone,
		Navigating:   navigating,
	}
	a.options.Harness.apply(&remote)

	session, err := a.options.LLM.Join(ctx, remote)
	if err != nil {
		return nil, err
	}
	return &Session{agent: a, pipeline: a.options.LLM, call: call, session: session}, nil
}

// Client is the acceleration router this agent talks to.
func (a *Agent) Client() (*acceleration.ClientWithResponses, error) {
	backend, err := a.options.LLM.Backend()
	if err != nil {
		return nil, err
	}
	return backend.Client()
}

// edge builds the transport lazily, so an agent that only ever chats needs no Stream
// credentials.
func (a *Agent) edge() (*edge.Edge, error) {
	if a.options.Edge != nil {
		return a.options.Edge, nil
	}
	transport, err := edge.New(edge.Options{})
	if err != nil {
		return nil, err
	}
	a.options.Edge = transport
	return transport, nil
}

// Session is one conversation the agent is holding.
type Session struct {
	agent    *Agent
	pipeline *stream.Pipeline
	call     edge.Call
	session  *acceleration.Session
}

// ID is the backend's id for the session.
func (s *Session) ID() string { return s.session.Id }

// Call is the Stream call the conversation is on. Its id is empty for a chat.
func (s *Session) Call() edge.Call { return s.call }

// Session is what the router said when it created this.
func (s *Session) Session() *acceleration.Session { return s.session }

// Events yields what the backend did until the conversation ends, when the channel closes.
func (s *Session) Events() <-chan stream.Event { return s.pipeline.Events() }

// Say speaks text without going through the model, for when you already know what should be
// said.
func (s *Session) Say(text string) error { return s.pipeline.Say(text, false) }

// Respond answers text through the model, as though it had been said on the call.
func (s *Session) Respond(text string) error { return s.pipeline.Respond(text, true) }

// Interrupt abandons the reply being spoken.
func (s *Session) Interrupt() error { return s.pipeline.Interrupt() }

// SetInstructions changes what the agent is told to be, from the next turn.
func (s *Session) SetInstructions(instructions string) error {
	return s.pipeline.SetInstructions(instructions)
}

// Wait blocks until the conversation ends or the context does.
func (s *Session) Wait(ctx context.Context) error {
	events := s.Events()
	for {
		select {
		case _, open := <-events:
			if !open {
				return nil
			}
		case <-ctx.Done():
			return ctx.Err()
		}
	}
}

// MonitorURL is a link a person can open to join this call from a browser and hear the
// agent.
func (s *Session) MonitorURL() (string, error) {
	if s.call.ID == "" {
		return "", errors.New("agents: a conversation held in writing has no call to watch")
	}
	transport, err := s.agent.edge()
	if err != nil {
		return "", err
	}
	return transport.MonitorURL(s.call, edge.User{ID: "monitor-" + s.session.Id, Name: "Monitor"})
}

// Close ends the conversation. Safe to call after it has already ended.
func (s *Session) Close(ctx context.Context) error { return s.pipeline.Leave(ctx) }

// memoryOf splits the filter into who the memories are about and what narrows them.
func memoryOf(filter map[string]string) *acceleration.SessionMemory {
	if len(filter) == 0 {
		return nil
	}

	memory := &acceleration.SessionMemory{}
	narrowing := map[string]string{}
	for key, value := range filter {
		if key == UserKey {
			user := value
			memory.UserId = &user
			continue
		}
		narrowing[key] = value
	}
	if len(narrowing) > 0 {
		memory.Filter = &narrowing
	}
	return memory
}

// userIDOf turns a name into something a call can be joined under.
func userIDOf(name string) string {
	id := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		default:
			return '-'
		}
	}, name)
	id = strings.Trim(id, "-")
	if id == "" {
		return "vision-agent"
	}
	return id
}
