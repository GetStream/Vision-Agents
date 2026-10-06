package sipbridge

import (
	"context"
	"sync"
)

// fakeLeg records what was asked of it and answers from its fields.
type fakeLeg struct {
	inviteAnswer   []byte
	inviteErr      error
	inviteBlocks   bool // Invite waits for ctx to end, like a phone nobody picks up
	reinviteAnswer []byte
	reinviteErr    error
	reinviteErrs   []error // consumed one per Reinvite call; nil entry or past end means reinviteErr
	ackErr         error
	byeErr         error
	byeBlocks      bool // Bye waits for ctx to end, like a trunk that does not answer
	onBye          func(ctx context.Context)

	mu         sync.Mutex
	methods    []string
	bodies     [][]byte
	reinvites  int
	byeCtxErrs []error // ctx.Err() seen by each Bye call
}

func (f *fakeLeg) record(method string, body []byte) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.methods = append(f.methods, method)
	f.bodies = append(f.bodies, body)
}

func (f *fakeLeg) Invite(ctx context.Context, body []byte) ([]byte, error) {
	f.record("invite", body)
	if f.inviteBlocks {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	return f.inviteAnswer, f.inviteErr
}

func (f *fakeLeg) Ack(_ context.Context, body []byte) error {
	f.record("ack", body)
	return f.ackErr
}

func (f *fakeLeg) Reinvite(_ context.Context, body []byte) ([]byte, error) {
	f.record("reinvite", body)
	f.mu.Lock()
	defer f.mu.Unlock()
	err := f.reinviteErr
	if len(f.reinviteErrs) > f.reinvites && f.reinviteErrs[f.reinvites] != nil {
		err = f.reinviteErrs[f.reinvites]
	}
	f.reinvites++
	return f.reinviteAnswer, err
}

func (f *fakeLeg) Info(_ context.Context, contentType string, body []byte) error {
	f.record("info:"+contentType, body)
	return nil
}

func (f *fakeLeg) Bye(ctx context.Context) error {
	f.mu.Lock()
	f.methods = append(f.methods, "bye")
	f.bodies = append(f.bodies, nil)
	f.byeCtxErrs = append(f.byeCtxErrs, ctx.Err())
	f.mu.Unlock()
	if f.onBye != nil {
		f.onBye(ctx)
	}
	if f.byeBlocks {
		<-ctx.Done()
		return ctx.Err()
	}
	return f.byeErr
}

func (f *fakeLeg) byeContextErrs() []error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]error(nil), f.byeCtxErrs...)
}

func (f *fakeLeg) calls() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]string(nil), f.methods...)
}

func (f *fakeLeg) body(i int) []byte {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.bodies[i]
}

func (f *fakeLeg) last() string {
	calls := f.calls()
	if len(calls) == 0 {
		return ""
	}
	return calls[len(calls)-1]
}
