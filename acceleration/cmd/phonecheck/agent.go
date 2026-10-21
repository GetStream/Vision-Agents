package main

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"time"

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

// listener is an inbound worker waiting for the call to the agent number.
type listener interface {
	Answer(ctx context.Context) (agentSession, error)
	Stop()
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
}

// newSDKRouter saves the agent's config on the router once and leaves it there. An empty
// customer is left to the SDK, which reads STREAM_ACCELERATION_CUSTOMER_ID.
func newSDKRouter(ctx context.Context, address, customer string, logger *slog.Logger) (*sdkRouter, error) {
	r, err := sdkRouterAt(address, customer, logger)
	if err != nil {
		return nil, err
	}
	if _, err := r.agent.Sync(ctx); err != nil {
		return nil, fmt.Errorf("save the %s agent config: %w", agentName, err)
	}
	return r, nil
}

func sdkRouterAt(address, customer string, logger *slog.Logger) (*sdkRouter, error) {
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

// LogRows reads every page of the session's rows; a long call has more than one.
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

func (r *sdkRouter) StartCall(ctx context.Context, from, to string) (agentSession, error) {
	session, err := r.agent.StartCall(ctx, from, to)
	if err != nil {
		return nil, err
	}
	return session, nil
}

// Listen registers a worker for one call to number. It is not agents.Agent.WaitForCall,
// which attaches the number first and so could rewire it.
func (r *sdkRouter) Listen(ctx context.Context, number string) (listener, error) {
	dispatch, err := stream.NewDispatch(stream.DispatchOptions{Backend: r.backend, Capacity: 1, Logger: r.logger})
	if err != nil {
		return nil, err
	}
	running, stop := context.WithCancel(ctx)
	l := &sdkListener{
		agent: r.agent, dispatch: dispatch, stop: stop,
		arrived: make(chan stream.InboundCall, 1), ended: make(chan struct{}),
	}
	dispatch.OnCall(l.take(number))
	go func() {
		l.runErr = dispatch.Run(running)
		close(l.ended)
	}()
	if err := l.registered(ctx); err != nil {
		l.Stop()
		return nil, err
	}
	return l, nil
}

// sdkListener is one dispatch worker, taking the first call to its number.
type sdkListener struct {
	agent    *agents.Agent
	dispatch *stream.Dispatch
	stop     context.CancelFunc
	arrived  chan stream.InboundCall
	ended    chan struct{} // closed when Run returns; runErr is set by then
	runErr   error
}

// take only accepts a call to number, once, as WaitForCall does.
func (l *sdkListener) take(number string) stream.CallHandler {
	return func(_ context.Context, call stream.InboundCall) error {
		if call.CalledNumber != number {
			return fmt.Errorf("this worker is waiting on %s, not %s", number, call.CalledNumber)
		}
		select {
		case l.arrived <- call:
			return nil
		default:
			return errors.New("this worker is already answering a call")
		}
	}
}

// registered returns once the router has named the worker, so a call dialed after it has
// somewhere to go.
func (l *sdkListener) registered(ctx context.Context) error {
	tick := time.NewTicker(100 * time.Millisecond)
	defer tick.Stop()
	for l.dispatch.WorkerID() == "" {
		select {
		case <-l.ended:
			return l.stoppedEarly()
		case <-ctx.Done():
			return ctx.Err()
		case <-tick.C:
		}
	}
	return nil
}

func (l *sdkListener) Answer(ctx context.Context) (agentSession, error) {
	var call stream.InboundCall
	select {
	case call = <-l.arrived:
	case <-l.ended:
		return nil, l.stoppedEarly()
	case <-ctx.Done():
		return nil, ctx.Err()
	}
	l.Stop()
	session, err := l.agent.Answer(ctx, call)
	if err != nil {
		return nil, err
	}
	return session, nil
}

func (l *sdkListener) stoppedEarly() error {
	if l.runErr != nil {
		return fmt.Errorf("the worker stopped: %w", l.runErr)
	}
	return errors.New("the worker stopped before a call arrived")
}

// Stop ends the worker and waits for it to leave the router. Safe to call again.
func (l *sdkListener) Stop() {
	l.stop()
	<-l.ended
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
