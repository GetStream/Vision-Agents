package main

import (
	"context"
	"errors"
	"flag"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"

	"github.com/GetStream/Vision-Agents/acceleration/cmd/phonecheck/person"
)

const (
	agentNumber  = "+15550100001"
	personNumber = "+15550100002"
	otherNumber  = "+15550100003"
)

// steps is what the fakes were asked to do, in order.
type steps []string

// placedACall reports whether anything but a read happened.
func (s steps) placedACall() bool {
	for _, step := range s {
		switch step {
		case "start call", "listen", "dial", "wait":
			return true
		}
	}
	return false
}

// fakeRouter answers from fields; a nil error is an answer.
type fakeRouter struct {
	steps      *steps
	healthErr  error
	numbers    []acceleration.PhoneNumber
	numbersErr error
	logsErr    error
}

func (r *fakeRouter) Healthy(context.Context) error {
	*r.steps = append(*r.steps, "healthy")
	return r.healthErr
}

func (r *fakeRouter) Numbers(context.Context) ([]acceleration.PhoneNumber, error) {
	*r.steps = append(*r.steps, "numbers")
	return r.numbers, r.numbersErr
}

func (r *fakeRouter) LogsAvailable(context.Context) error {
	*r.steps = append(*r.steps, "logs available")
	return r.logsErr
}

func (r *fakeRouter) LogRows(context.Context, string) ([]acceleration.AgentLog, error) {
	*r.steps = append(*r.steps, "log rows")
	return nil, nil
}

func (r *fakeRouter) LogRow(context.Context, string) (acceleration.AgentLog, error) {
	*r.steps = append(*r.steps, "log row")
	return acceleration.AgentLog{}, errors.New("no log row")
}

func (r *fakeRouter) StartCall(context.Context, string, string) (agentSession, error) {
	*r.steps = append(*r.steps, "start call")
	return nil, errors.New("not in these tests")
}

func (r *fakeRouter) Listen(context.Context, string) (listener, error) {
	*r.steps = append(*r.steps, "listen")
	return nil, errors.New("not in these tests")
}

// fakePerson is a provider named "fake" with one flag of its own.
type fakePerson struct {
	steps  *steps
	region string
}

func (p *fakePerson) Name() string { return "fake" }

func (p *fakePerson) Flags(fs *flag.FlagSet) {
	fs.StringVar(&p.region, "fake-region", "", "where the fake vendor places calls")
}

func (p *fakePerson) Dial(context.Context, person.Call) error {
	*p.steps = append(*p.steps, "dial")
	return nil
}

func (p *fakePerson) Wait(context.Context, person.Call) error {
	*p.steps = append(*p.steps, "wait")
	return nil
}

func (p *fakePerson) Hangup(context.Context) error {
	*p.steps = append(*p.steps, "hangup")
	return nil
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
