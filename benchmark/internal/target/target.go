package target

import "context"

// Target is a system Voicebench can start for one benchmark call.
type Target interface {
	Prepare(ctx context.Context) (func(), error)
	StartCall(ctx context.Context, callID string, callType string) (func(), error)
}

// Noop joins no agent. It is used when --call-id points at an agent already in the call.
type Noop struct{}

func (Noop) Prepare(context.Context) (func(), error) { return func() {}, nil }

func (Noop) StartCall(context.Context, string, string) (func(), error) { return func() {}, nil }
