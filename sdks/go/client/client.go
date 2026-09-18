// Package client is the resource half of the Go SDK: agents addressed by name, the
// conversations they hold, and the turns those conversations are made of.
//
// It is the same surface the JavaScript and Python clients present, spelled the way Go spells
// things. Where stream.Pipeline is one conversation configured in full and driven from here,
// this is a handle on an agent that was configured once in the backend:
//
//	api, err := client.New(stream.Backend{})
//	agent := api.Agent("docs")
//	session, err := agent.Sessions.Create(ctx, client.SessionOptions{Title: "Sendbird"})
//	answer, err := session.Responses.Create(ctx, "Is Stream better than Sendbird?")
//
//	items := answer.Items.Unwind(ctx, 0)
//	for item := range items.Items() {
//	    fmt.Println(item.Kind, item.Text)
//	}
//
// Nothing here does inference or touches media. The conversation is held in the backend and
// what arrives here are the events saying so.
package client

import (
	"context"
	"errors"
	"fmt"
	"os"
	"sync"

	getstream "github.com/GetStream/getstream-go/v5"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// Client is the backend held once: where the router is, and who is calling it.
type Client struct {
	backend stream.Backend

	// mu guards the Stream client, which is built on the first conversation that asks for
	// chat or video and shared by every one after it.
	mu       sync.Mutex
	streamed *getstream.Stream
}

// New resolves the backend and refuses one that cannot be reached or billed.
//
// The zero Backend reads the environment, which is what a program deployed next to a router
// wants. Resolved here rather than at the first call, so a missing credential is reported
// where the backend was configured and not in the middle of a conversation.
func New(backend stream.Backend) (*Client, error) {
	resolved, err := backend.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{backend: resolved}, nil
}

// SetUser says who this client is acting for, and hands over the token that proves it.
//
// It returns a client rather than mutating this one, because a process usually holds both: its
// own backend credential for the things only a backend may do, and a client per user for the
// conversations that belong to them. Sharing one and switching the user on it would make which
// user a request was for depend on when it happened to run.
func (c *Client) SetUser(userID, token string) (*Client, error) {
	if userID == "" {
		return nil, errors.New("client: a user needs an id")
	}
	if token == "" {
		return nil, fmt.Errorf("client: there is no token for %s to hold", userID)
	}

	backend := c.backend
	backend.UserID = userID
	backend.Token = token
	resolved, err := backend.Resolve()
	if err != nil {
		return nil, err
	}
	return &Client{backend: resolved}, nil
}

// ServerSide says whether this client speaks for the app itself rather than for one user.
//
// What the guest claim and the agent configs are refused to anybody else for: a token naming
// a user is a device's credential, and a device may hold a conversation but not rewrite the
// agent holding it or reassign somebody else's conversations.
func (c *Client) ServerSide() bool { return c.backend.UserID == "" }

// Agent is an agent addressed by the name it is configured under.
//
// No request is made: this is the name in a wrapper, and a name that matches nothing
// configured is refused when a conversation is opened rather than here.
func (c *Client) Agent(name string) *Agent {
	agent := &Agent{client: c, name: name, functions: tools.NewRegistry()}
	agent.Sessions = &Sessions{client: c, agent: agent}
	return agent
}

// Backend is where this client points, for the parts of the SDK that take one directly.
func (c *Client) Backend() stream.Backend { return c.backend }

// api is the generated client, carrying this client's credentials.
func (c *Client) api() (*acceleration.ClientWithResponses, error) {
	return c.backend.Client()
}

// stream is the Stream SDK, for the chat a conversation is written into and the call it is
// held on.
//
// A server credential rather than this client's token, because getstream-go signs its own
// requests and has no user-token mode. That is the right shape for Go, which is a backend: it
// is the browser SDKs that connect as a user, with the token SetUser was given.
func (c *Client) stream() (*getstream.Stream, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.streamed != nil {
		return c.streamed, nil
	}

	key, secret := c.backend.APIKey, c.backend.APISecret
	if key == "" {
		key = os.Getenv(stream.APIKeyEnv)
	}
	if secret == "" {
		secret = os.Getenv(stream.APISecretEnv)
	}
	if key == "" || secret == "" {
		return nil, fmt.Errorf("client: chat and video connect to Stream rather than to this "+
			"router, so they need %s and %s", stream.APIKeyEnv, stream.APISecretEnv)
	}

	connected, err := getstream.NewClient(key, secret)
	if err != nil {
		return nil, fmt.Errorf("client: connecting to Stream as %s: %w", key, err)
	}
	c.streamed = connected
	return connected, nil
}

// Agent is one configured agent, and its conversations.
type Agent struct {
	// Sessions is this agent's conversations: opening one, and reading the old ones back.
	Sessions *Sessions

	client *Client
	name   string
	// functions are the caller's own, offered by every conversation this agent opens. Held
	// on the agent rather than per session because a function registered once should not
	// have to be registered again for the next conversation.
	functions *tools.Registry
}

// Name is what the agent is called, which is what a caller knows it as.
func (a *Agent) Name() string { return a.name }

// Functions are the ones this agent's conversations offer the model, to register into.
//
// It satisfies the target agents.Register takes, so the same registration works here:
//
//	agents.Register(agent, "get_weather", "Get current weather", func(...) {...})
func (a *Agent) Functions() *tools.Registry { return a.functions }

// Config is how the agent is configured, as the backend has it, or nil for a name nothing is
// stored under.
//
// Server side only: how an agent is configured is not a device's to read.
func (a *Agent) Config(ctx context.Context) (*acceleration.AgentConfig, error) {
	api, err := a.client.api()
	if err != nil {
		return nil, err
	}

	listed, err := api.ListAgentConfigsWithResponse(ctx,
		&acceleration.ListAgentConfigsParams{Name: &a.name})
	if err != nil {
		return nil, fmt.Errorf("client: looking up the agent %s: %w", a.name, err)
	}
	if listed.JSON200 == nil {
		return nil, failure("looking up the agent "+a.name, listed.Status(),
			listed.JSON400, listed.JSON401)
	}
	for _, config := range *listed.JSON200 {
		if config.Name == a.name {
			return &config, nil
		}
	}
	return nil, nil
}

// failure turns whichever error body arrived into one error, or reports the status when none
// did. Every refusal in the spec is the same shape, so this is the whole of it.
func failure(what, status string, bodies ...*acceleration.Error) error {
	for _, body := range bodies {
		if body != nil {
			return fmt.Errorf("client: %s: %s", what, body.Error)
		}
	}
	return fmt.Errorf("client: %s: %s", what, status)
}

// pointer is a value the generated types want as an optional, for the ones a caller set. A
// zero is left off rather than sent: the wire's own default is what an unset field means, and
// sending a zero would overwrite a configured value with nothing.
func pointer[T comparable](value T) *T {
	var zero T
	if value == zero {
		return nil
	}
	return &value
}
