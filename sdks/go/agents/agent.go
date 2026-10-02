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
	"maps"
	"os"
	"path/filepath"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/client"
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
	// Dir is an agent directory holding agent.yaml, instructions.md, skills/ and knowledge/.
	// What it says fills in whatever is left empty here. Empty reads agents/<Name> when
	// there is one, which is where `agents create` puts it.
	Dir string
	// Tools are the functions the model is offered and this process runs.
	Tools []tools.Tool
	// Instructions is the system prompt.
	Instructions string
	// Guardrail is a guardrail.md: frontmatter saying how a turn is screened, then the
	// policy in prose. Empty means every turn is answered. It is enforced in the backend,
	// not here, so a turn the policy refuses never reaches the model.
	Guardrail string
	// LLM is the pipeline running in the backend. Nil runs the stored config named Name.
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
	// Sessions opens the agent's conversations.
	Sessions *Sessions

	options Options
	folder  *Folder
	logger  *slog.Logger
}

// New validates an agent's configuration and reads its directory, if it has one.
func New(options Options) (*Agent, error) {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}

	agent := &Agent{options: options, logger: logger}
	agent.Sessions = &Sessions{agent: agent}
	dir := options.Dir
	if dir == "" && options.Name != "" {
		if _, err := os.Stat(filepath.Join("agents", options.Name, AgentFile)); err == nil {
			dir = filepath.Join("agents", options.Name)
		}
	}
	if dir != "" {
		folder, err := Load(dir)
		if err != nil {
			return nil, err
		}
		agent.folder = folder
		folder.fill(&agent.options)
	}

	if agent.options.Name == "" {
		return nil, errors.New("agents: an agent needs a name")
	}
	if agent.options.LLM == nil {
		agent.options.LLM = stream.Accelerated(stream.Config{Agent: agent.options.Name})
	}
	for _, tool := range agent.options.Tools {
		if err := agent.options.LLM.Functions().Add(tool); err != nil {
			return nil, err
		}
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

// Tools are the agent's own functions, which the model is offered and this process runs.
//
// They are declared to the backend when a session is created, so a tool added to a session
// already running is offered from the next one.
//
//	agent.Tools().Add(LookupOrder{orders: orders})
func (a *Agent) Tools() *tools.Registry { return a.options.LLM.Functions() }

// Folder is the agent directory the agent was read from, or nil if it has none.
func (a *Agent) Folder() *Folder { return a.folder }

// Sessions opens an agent's conversations.
type Sessions struct {
	agent *Agent
}

// SessionOptions is what one conversation changes about the agent holding it. The zero
// value holds the conversation in writing, as the agent is configured.
type SessionOptions struct {
	// Call is the Stream call to join. Nil holds the conversation in writing: nothing is
	// transcribed and nothing is spoken. A call with an empty id creates one named after a
	// random string, which is what a one-off conversation wants.
	Call *edge.Call

	ConversationID string
	// AgentID is the conversation being answered, which names the channel replies are
	// written into and is what the backend finds a running session by when somebody
	// writes to it again. Empty answers in one of the agent's own.
	//
	// A worker answering several conversations has to set it. Left empty they all join
	// under the agent's user id, and a message arriving for one of them reaches whichever
	// started last.
	AgentID string

	// Instructions replace the agent's own system prompt for this conversation.
	Instructions string
	// CostTracking labels are added to the agent's, and win where both name a key.
	CostTracking map[string]string
	// MemoryFilter replaces the agent's, for a conversation about somebody else.
	MemoryFilter map[string]string

	// Title and Description are what a person finds this conversation by afterwards. Both
	// are searched, so the opening question makes a reasonable title.
	Title       string
	Description string
	// ProjectID groups conversations, and is carried as a cost label too.
	ProjectID string
	// Custom is anything of the caller's own worth remembering about the conversation, which
	// a later query can match on.
	Custom map[string]any
	// Incognito holds the conversation and keeps nothing: no session row, no turns and no
	// transcript. It cannot be found afterwards, which is the point.
	Incognito bool
	// ModelOverwrites changes the models for this conversation alone, over whatever the
	// agent's own configuration decided.
	ModelOverwrites *acceleration.ModelOverwrites
}

// Create opens a conversation. It returns once the backend is holding it, so an agent on a
// call is already listening.
func (s *Sessions) Create(ctx context.Context, options SessionOptions) (*Session, error) {
	a := s.agent
	var call edge.Call
	if options.Call != nil {
		transport, err := a.edge()
		if err != nil {
			return nil, err
		}
		call, err = transport.CreateCall(ctx, *options.Call, edge.User{ID: a.options.UserID, Name: a.options.Name})
		if err != nil {
			return nil, err
		}
	}
	return a.join(ctx, call, nil, false, options)
}

// Join has the backend join a call and hold a conversation on it.
func (a *Agent) Join(ctx context.Context, call edge.Call) (*Session, error) {
	return a.Sessions.Create(ctx, SessionOptions{Call: &call})
}

// Chat holds the conversation in writing rather than on a call.
func (a *Agent) Chat(ctx context.Context, options ...SessionOptions) (*Session, error) {
	var chosen SessionOptions
	if len(options) > 0 {
		chosen = options[0]
	}
	chosen.Call = nil
	return a.Sessions.Create(ctx, chosen)
}

// join renders the agent's configuration into a session and opens it.
func (a *Agent) join(ctx context.Context, call edge.Call, phone *acceleration.SessionPhone, navigating bool, options ...SessionOptions) (*Session, error) {
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
	if len(options) > 0 {
		chosen := options[0]
		remote.ConversationID = chosen.ConversationID
		remote.Title = chosen.Title
		remote.Description = chosen.Description
		remote.ProjectID = chosen.ProjectID
		remote.Custom = chosen.Custom
		remote.Incognito = chosen.Incognito
		remote.ModelOverwrites = chosen.ModelOverwrites
		if chosen.AgentID != "" {
			remote.AgentID = chosen.AgentID
		}
		if chosen.Instructions != "" {
			remote.Instructions = chosen.Instructions
		}
		if len(chosen.CostTracking) > 0 {
			tags := maps.Clone(a.options.CostTracking)
			if tags == nil {
				tags = map[string]string{}
			}
			maps.Copy(tags, chosen.CostTracking)
			remote.Tags = tags
		}
		if chosen.MemoryFilter != nil {
			remote.Memory = memoryOf(chosen.MemoryFilter)
		}
	}
	backend, err := a.options.LLM.Backend()
	if err != nil {
		return nil, err
	}
	resources, err := client.New(backend)
	if err != nil {
		return nil, err
	}
	if _, err := a.options.LLM.Join(ctx, remote); err != nil {
		return nil, err
	}
	held, err := resources.Hold(a.options.Name, a.options.LLM)
	if err != nil {
		return nil, err
	}
	return &Session{Session: held, agent: a, call: call}, nil
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
//
// It is a client.Session, so it reads the same as one opened by name: session.Responses,
// Fork, Say and Events are all there. What it adds is the call it is on.
type Session struct {
	*client.Session

	agent *Agent
	call  edge.Call
}

// Call is the Stream call the conversation is on. Its id is empty for a chat.
func (s *Session) Call() edge.Call { return s.call }

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
	return transport.MonitorURL(s.call, edge.User{ID: "monitor-" + s.ID(), Name: "Monitor"})
}

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
