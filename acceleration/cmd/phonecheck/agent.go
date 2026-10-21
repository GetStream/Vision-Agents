package main

import (
	"context"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
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
