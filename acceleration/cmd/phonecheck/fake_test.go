package main

import (
	"context"
	"errors"
	"flag"
	"slices"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

const (
	agentNumber  = "+15550100001"
	personNumber = "+15550100002"
	otherNumber  = "+15550100003"
)

// steps is what the fakes were asked to do, in order. The person and the session wait side
// by side, so it is locked.
type steps struct {
	mu   sync.Mutex
	list []string
}

func (s *steps) add(step string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.list = append(s.list, step)
}

func (s *steps) all() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.list)
}

// calls leaves out the router reads.
func (s *steps) calls() []string {
	var out []string
	for _, step := range s.all() {
		switch step {
		case "ready", "healthy", "numbers", "logs_available", "log_rows", "log_row":
			continue
		}
		out = append(out, step)
	}
	return out
}

// placedACall reports whether anything but a read happened.
func (s *steps) placedACall() bool {
	return len(s.calls()) > 0
}

// fakeRouter answers from fields; a nil error is an answer.
type fakeRouter struct {
	steps      *steps
	healthErr  error
	checkDelay time.Duration // how long the health check takes
	numbers    []acceleration.PhoneNumber
	numbersErr error
	logsErr    error
	// rowLists is what each LogRows call returns in turn; the last list repeats.
	rowLists [][]acceleration.AgentLog
	listed   int
	rows     map[string]acceleration.AgentLog
	session  *fakeSession // what StartCall starts and the listener answers with
	startErr error
	listener *fakeListener
}

func (r *fakeRouter) Healthy(context.Context) error {
	r.steps.add("healthy")
	time.Sleep(r.checkDelay)
	return r.healthErr
}

func (r *fakeRouter) Numbers(context.Context) ([]acceleration.PhoneNumber, error) {
	r.steps.add("numbers")
	return r.numbers, r.numbersErr
}

func (r *fakeRouter) LogsAvailable(context.Context) error {
	r.steps.add("logs_available")
	return r.logsErr
}

func (r *fakeRouter) LogRows(context.Context, string) ([]acceleration.AgentLog, error) {
	r.steps.add("log_rows")
	if len(r.rowLists) == 0 {
		return nil, nil
	}
	list := r.rowLists[min(r.listed, len(r.rowLists)-1)]
	r.listed++
	return list, nil
}

func (r *fakeRouter) LogRow(_ context.Context, id string) (acceleration.AgentLog, error) {
	r.steps.add("log_row")
	row, ok := r.rows[id]
	if !ok {
		return acceleration.AgentLog{}, errors.New("no log row")
	}
	return row, nil
}

func (r *fakeRouter) StartCall(context.Context, string, string) (agentSession, error) {
	r.steps.add("start_call")
	if r.startErr != nil {
		return nil, r.startErr
	}
	return r.session, nil
}

func (r *fakeRouter) Listen(context.Context, string) (listener, error) {
	r.steps.add("listen")
	if r.listener == nil {
		return nil, errors.New("no listener")
	}
	return r.listener, nil
}

// fakeListener answers with its session. When neverArrives, no call comes within the wait;
// joinErr is what joining an arrived call fails with.
type fakeListener struct {
	steps        *steps
	session      *fakeSession
	neverArrives bool
	joinErr      error
}

func (l *fakeListener) Answer(ctx context.Context, wait time.Duration) (agentSession, error) {
	if l.neverArrives {
		select {
		case <-time.After(wait):
			return nil, errNoCallArrived
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	l.steps.add("agent_answer")
	if l.joinErr != nil {
		return nil, l.joinErr
	}
	return l.session, nil
}

func (l *fakeListener) Stop(context.Context) error {
	l.steps.add("stop")
	return nil
}

// fakeSession ends once the person's call has, as the router ends a phone session.
type fakeSession struct {
	steps       *steps
	id          string
	personEnded <-chan struct{}
	closeErr    error
}

func (s *fakeSession) ID() string { return s.id }

func (s *fakeSession) Wait(ctx context.Context) error {
	select {
	case <-s.personEnded:
		s.steps.add("session_wait")
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

func (s *fakeSession) Close(context.Context) error {
	s.steps.add("close")
	return s.closeErr
}

// fakePerson is a provider named "fake" with one flag of its own.
type fakePerson struct {
	steps     *steps
	region    string
	calls     []person.Call // what Dial and Wait were given
	ended     chan struct{} // closed when Wait returns; nil is fine
	waiting   chan struct{} // closed when Wait starts; nil is fine
	stayOn    bool          // Wait lasts until ctx ends
	waitErr   error         // what Wait returns once the call ended
	hangupErr error
	readyErr  error
}

func (p *fakePerson) Name() string { return "fake" }

func (p *fakePerson) Flags(fs *flag.FlagSet) {
	fs.StringVar(&p.region, "fake-region", "", "where the fake vendor places calls")
}

func (p *fakePerson) Ready(person.Direction) error {
	p.steps.add("ready")
	return p.readyErr
}

func (p *fakePerson) Dial(_ context.Context, call person.Call) error {
	p.steps.add("dial")
	p.calls = append(p.calls, call)
	return nil
}

func (p *fakePerson) Wait(ctx context.Context, call person.Call) error {
	p.calls = append(p.calls, call)
	if p.waiting != nil {
		close(p.waiting)
	}
	if p.stayOn {
		<-ctx.Done()
		return ctx.Err()
	}
	p.steps.add("person_wait")
	if p.ended != nil {
		close(p.ended)
	}
	return p.waitErr
}

func (p *fakePerson) Hangup(context.Context, person.Call) error {
	p.steps.add("hangup")
	return p.hangupErr
}

// attached is a number the router has wired to Stream.
func attached(e164, vendor string) acceleration.PhoneNumber {
	trunk := "stream-trunk"
	return acceleration.PhoneNumber{E164: e164, Vendor: vendor, StreamTrunkId: &trunk}
}

// unattached is a number the router has registered but not wired.
func unattached(e164, vendor string) acceleration.PhoneNumber {
	return acceleration.PhoneNumber{E164: e164, Vendor: vendor}
}
