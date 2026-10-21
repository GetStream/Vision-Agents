package main

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/agents"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// errNoCallArrived is Answer's error when no call reached the worker in time.
var errNoCallArrived = errors.New("no call reached the worker")

// listener is an inbound worker waiting for the call to the agent number.
type listener interface {
	// Answer waits up to wait for the call to arrive, then joins it under ctx.
	Answer(ctx context.Context, wait time.Duration) (agentSession, error)
	// Stop ends the worker and waits, up to ctx, for it to leave the router.
	Stop(ctx context.Context) error
}

// Listen registers a worker for one call to number. It is not agents.Agent.WaitForCall,
// which attaches the number first and so could rewire it.
func (r *sdkRouter) Listen(ctx context.Context, number string) (listener, error) {
	if err := r.saveConfig(ctx); err != nil {
		return nil, err
	}
	dispatch, err := stream.NewDispatch(stream.DispatchOptions{Backend: r.backend, Capacity: 1, Logger: r.logger})
	if err != nil {
		return nil, err
	}
	running, cancel := context.WithCancel(ctx)
	l := &sdkListener{
		agent: r.agent, dispatch: dispatch, cancel: cancel,
		arrived: make(chan stream.InboundCall, 1), ended: make(chan struct{}),
	}
	dispatch.OnCall(l.take(number))
	go func() {
		l.runErr = dispatch.Run(running)
		close(l.ended)
	}()
	if err := l.registered(ctx); err != nil {
		l.cancel()
		return nil, err
	}
	return l, nil
}

// sdkListener is one dispatch worker, taking the first call to its number.
type sdkListener struct {
	agent    *agents.Agent
	dispatch *stream.Dispatch
	cancel   context.CancelFunc // ends the worker
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

// Answer stops the worker taking more calls once one arrives; teardown waits for it to leave.
func (l *sdkListener) Answer(ctx context.Context, wait time.Duration) (agentSession, error) {
	call, err := l.arrival(ctx, wait)
	if err != nil {
		return nil, err
	}
	l.cancel()
	session, err := l.agent.Answer(ctx, call)
	if err != nil {
		return nil, err // not session: a nil *agents.Session in agentSession is not nil
	}
	return session, nil
}

func (l *sdkListener) arrival(ctx context.Context, wait time.Duration) (stream.InboundCall, error) {
	timer := time.NewTimer(wait)
	defer timer.Stop()
	select {
	case call := <-l.arrived:
		return call, nil
	case <-l.ended:
		return stream.InboundCall{}, l.stoppedEarly()
	case <-timer.C:
		return stream.InboundCall{}, errNoCallArrived
	case <-ctx.Done():
		return stream.InboundCall{}, ctx.Err()
	}
}

func (l *sdkListener) stoppedEarly() error {
	if l.runErr != nil {
		return fmt.Errorf("the worker stopped: %w", l.runErr)
	}
	return errors.New("the worker stopped before a call arrived")
}

// Stop is safe to call again.
func (l *sdkListener) Stop(ctx context.Context) error {
	l.cancel()
	select {
	case <-l.ended:
		return nil
	case <-ctx.Done():
		return fmt.Errorf("the worker did not stop: %w", ctx.Err())
	}
}
