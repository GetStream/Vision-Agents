package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// router is everything the tool asks of the router.
type router interface {
	Healthy(ctx context.Context) error
	Numbers(ctx context.Context) ([]acceleration.PhoneNumber, error)
	LogsAvailable(ctx context.Context) error
	LogRows(ctx context.Context, sessionID string) ([]acceleration.AgentLog, error)
	LogRow(ctx context.Context, id string) (acceleration.AgentLog, error)
	StartCall(ctx context.Context, from, to string) (agentSession, error)
	// Listen returns once the worker is registered.
	Listen(ctx context.Context, number string) (listener, error)
}

// agentSession is the agent's side of one call.
type agentSession interface {
	ID() string
	Wait(ctx context.Context) error
	Close(ctx context.Context) error
}

const agentName = "phonecheck"

// sdkRouter is the router through the Go SDK, with one agent config saved under agentName.
type sdkRouter struct {
	agent   *agents.Agent
	client  *acceleration.ClientWithResponses
	backend stream.Backend
	logger  *slog.Logger
	saved   bool
}

// newSDKRouter only builds the client; nothing reaches the router yet. An empty customer is
// left to the SDK, which reads STREAM_ACCELERATION_CUSTOMER_ID.
func newSDKRouter(address, customer string, logger *slog.Logger) (*sdkRouter, error) {
	agent, err := agents.New(agents.Options{
		Name:         agentName,
		Instructions: "You are on a short test phone call. Answer in one sentence.",
		LLM: stream.Accelerated(stream.Config{
			Agent:   agentName,
			Backend: stream.Backend{URL: address, CustomerID: customer},
		}),
		Logger: logger,
	})
	if err != nil {
		return nil, err
	}
	backend, err := agent.LLM().Backend()
	if err != nil {
		return nil, err
	}
	client, err := backend.Client()
	if err != nil {
		return nil, err
	}
	return &sdkRouter{agent: agent, client: client, backend: backend, logger: logger}, nil
}

func (r *sdkRouter) Healthy(ctx context.Context) error {
	resp, err := r.client.GetHealthWithResponse(ctx)
	if err != nil {
		return err
	}
	return statusError(resp.HTTPResponse, resp.Body)
}

func (r *sdkRouter) Numbers(ctx context.Context) ([]acceleration.PhoneNumber, error) {
	phone, err := r.agent.Phone()
	if err != nil {
		return nil, err
	}
	return phone.Numbers(ctx)
}

func (r *sdkRouter) LogsAvailable(ctx context.Context) error {
	one := 1
	resp, err := r.client.ListAgentLogsWithResponse(ctx, &acceleration.ListAgentLogsParams{Limit: &one})
	if err != nil {
		return err
	}
	if resp.StatusCode() == http.StatusServiceUnavailable {
		return fmt.Errorf("%w: %w", errNoLogStore, statusError(resp.HTTPResponse, resp.Body))
	}
	return statusError(resp.HTTPResponse, resp.Body)
}

// LogRows reads every page of the session's rows; a long call has more than one. A page with
// no new cursor ends the read, so a router that repeats itself cannot loop it forever.
func (r *sdkRouter) LogRows(ctx context.Context, sessionID string) ([]acceleration.AgentLog, error) {
	var rows []acceleration.AgentLog
	limit := 250
	params := &acceleration.ListAgentLogsParams{SessionId: &sessionID, Limit: &limit}
	for {
		resp, err := r.client.ListAgentLogsWithResponse(ctx, params)
		if err != nil {
			return nil, err
		}
		if resp.JSON200 == nil {
			return nil, statusError(resp.HTTPResponse, resp.Body)
		}
		rows = append(rows, resp.JSON200.Items...)
		if !resp.JSON200.HasMore {
			return rows, nil
		}
		next := resp.JSON200.NextCursor
		if next == "" || (params.Cursor != nil && next == *params.Cursor) {
			return rows, nil
		}
		params.Cursor = &next
	}
}

func (r *sdkRouter) LogRow(ctx context.Context, id string) (acceleration.AgentLog, error) {
	resp, err := r.client.GetAgentLogWithResponse(ctx, id)
	if err != nil {
		return acceleration.AgentLog{}, err
	}
	if resp.JSON200 == nil {
		return acceleration.AgentLog{}, statusError(resp.HTTPResponse, resp.Body)
	}
	return *resp.JSON200, nil
}

// saveConfig saves the agent's config once, on the first call, so a run whose checks fail
// writes nothing. The config is left on the router.
func (r *sdkRouter) saveConfig(ctx context.Context) error {
	if r.saved {
		return nil
	}
	if _, err := r.agent.Sync(ctx); err != nil {
		return fmt.Errorf("save the %s agent config: %w", agentName, err)
	}
	r.saved = true
	return nil
}

func (r *sdkRouter) StartCall(ctx context.Context, from, to string) (agentSession, error) {
	if err := r.saveConfig(ctx); err != nil {
		return nil, err
	}
	session, err := r.agent.StartCall(ctx, from, to)
	if err != nil {
		return nil, err // not session: a nil *agents.Session in agentSession is not nil
	}
	return session, nil
}

// statusError is nil for a 2xx answer and names the status otherwise.
func statusError(resp *http.Response, body []byte) error {
	if resp == nil {
		return errors.New("the router sent no answer")
	}
	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}
	return fmt.Errorf("the router answered %s: %s", resp.Status, strings.TrimSpace(string(body)))
}
