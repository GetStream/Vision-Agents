package tools

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"
)

// RunSuite is how a registry runs calls: within limits, once per id, and stoppable.
type RunSuite struct {
	suite.Suite
	registry *Registry
}

func TestRunSuite(t *testing.T) {
	suite.Run(t, new(RunSuite))
}

func (s *RunSuite) SetupTest() {
	s.registry = NewRegistry()
}

// peak records the most calls seen running at once.
type peak struct{ running, most atomic.Int32 }

func (p *peak) enter() func() {
	now := p.running.Add(1)
	for {
		seen := p.most.Load()
		if now <= seen || p.most.CompareAndSwap(seen, now) {
			break
		}
	}
	return func() { p.running.Add(-1) }
}

// burst makes calls to name all at once and waits for every one.
func (s *RunSuite) burst(name string, calls int) {
	var all sync.WaitGroup
	for range calls {
		all.Go(func() {
			_, err := s.registry.Call(s.T().Context(), name, "{}")
			s.NoError(err)
		})
	}
	all.Wait()
}

// slowQuery registers a query that takes a moment, counted by seen.
func (s *RunSuite) slowQuery(seen *peak, options ...Option) {
	s.Require().NoError(Register(s.registry, "query", "Query the database", func(context.Context, struct{}) (any, error) {
		defer seen.enter()()
		time.Sleep(20 * time.Millisecond)
		return "ok", nil
	}, options...))
}

func (s *RunSuite) TestABurstOfCallsIsHeldToTheRegistrysCeiling() {
	var seen peak
	s.slowQuery(&seen)

	s.burst("query", 50)

	s.LessOrEqual(seen.most.Load(), int32(DefaultConcurrency))
}

func (s *RunSuite) TestARegistryCanBeGivenACeilingOfItsOwn() {
	var seen peak
	s.Require().NoError(s.registry.SetConcurrency(3))
	s.slowQuery(&seen)

	s.burst("query", 20)

	s.LessOrEqual(seen.most.Load(), int32(3))
}

func (s *RunSuite) TestAToolsOwnConcurrencyHoldsBelowTheRegistrys() {
	var seen peak
	s.slowQuery(&seen, Concurrency(4))

	s.burst("query", 50)

	s.LessOrEqual(seen.most.Load(), int32(4))
}

func (s *RunSuite) TestACallWaitingForAPlaceTimesOutOnItsOwnDeadline() {
	release := make(chan struct{})
	defer close(release)
	// The function ignores its context, so the first call holds the one place until released.
	s.Require().NoError(Register(s.registry, "query", "Query the database", func(context.Context, struct{}) (any, error) {
		<-release
		return "ok", nil
	}, Concurrency(1), Timeout(100*time.Millisecond)))
	go func() { _, _ = s.registry.Run(context.Background(), Call{ID: "first"}, "query", "{}") }()
	time.Sleep(20 * time.Millisecond)

	began := time.Now()
	_, err := s.registry.Run(s.T().Context(), Call{ID: "second"}, "query", "{}")

	s.ErrorIs(err, context.DeadlineExceeded)
	s.Less(time.Since(began), time.Second)
}

func (s *RunSuite) TestAFunctionReadsTheCallItIsRunningFor() {
	seen := make(chan Call, 1)
	s.Require().NoError(Register(s.registry, "lookup_order", "Look up an order", func(ctx context.Context, _ struct{}) (any, error) {
		seen <- *CallFrom(ctx)
		return "ok", nil
	}))
	want := Call{ID: "call-1", SessionID: "session-1", TurnID: "turn-1"}

	_, err := s.registry.Run(s.T().Context(), want, "lookup_order", "")

	s.Require().NoError(err)
	s.Equal(want, <-seen)
}

func (s *RunSuite) TestOutsideACallTheCallIsEmptyRatherThanNil() {
	call := CallFrom(context.Background())

	s.Require().NotNil(call)
	s.Empty(call.ID)
}

func (s *RunSuite) TestCancellingACallStopsItsFunction() {
	started := make(chan struct{})
	s.Require().NoError(Register(s.registry, "search", "Search", func(ctx context.Context, _ struct{}) (any, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	}))
	ended := make(chan error, 1)
	go func() {
		_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", SessionID: "s"}, "search", "{}")
		ended <- err
	}()
	<-started

	s.Require().True(s.registry.Cancel("s", "call-1"), "the running call was not found to cancel")

	select {
	case err := <-ended:
		s.ErrorIs(err, context.Canceled)
	case <-time.After(2 * time.Second):
		s.FailNow("the cancelled call is still running")
	}
	s.False(s.registry.Cancel("s", "call-1"), "a call that has ended was found to cancel")
}

func (s *RunSuite) TestACancelledCallLeavesNoReceipt() {
	var ran atomic.Int32
	s.Require().NoError(Register(s.registry, "search", "Search", func(ctx context.Context, _ struct{}) (any, error) {
		if ran.Add(1) == 1 {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return "found", nil
	}))
	ctx, cancel := context.WithCancel(s.T().Context())
	time.AfterFunc(20*time.Millisecond, cancel)
	_, _ = s.registry.Run(ctx, Call{ID: "call-1"}, "search", "{}")

	output, err := s.registry.Run(s.T().Context(), Call{ID: "call-1"}, "search", "{}")

	s.Require().NoError(err)
	s.Equal("found", output.Text, "a cancelled call is no answer to repeat")
}

func (s *RunSuite) TestACopyOfAFinishedCallGetsItsReceipt() {
	var ran atomic.Int32
	s.Require().NoError(Register(s.registry, "book", "Book", func(context.Context, struct{}) (any, error) {
		ran.Add(1)
		return "booked", nil
	}))
	call := Call{ID: "call-1", SessionID: "s"}
	_, err := s.registry.Run(s.T().Context(), call, "book", "{}")
	s.Require().NoError(err)

	output, err := s.registry.Run(s.T().Context(), call, "book", "{}")

	s.Require().NoError(err)
	s.Equal("booked", output.Text)
	s.EqualValues(1, ran.Load())
}

func (s *RunSuite) TestTheSameIdInAnotherSessionIsAnotherCall() {
	var ran atomic.Int32
	s.Require().NoError(Register(s.registry, "book", "Book", func(context.Context, struct{}) (any, error) {
		ran.Add(1)
		return "booked", nil
	}))

	for _, session := range []string{"a", "b"} {
		_, err := s.registry.Run(s.T().Context(), Call{ID: "call-1", SessionID: session}, "book", "{}")
		s.Require().NoError(err)
	}

	s.EqualValues(2, ran.Load())
}

// sharedReceipts is a store two registries share, as two processes would share Redis.
type sharedReceipts struct {
	mu   sync.Mutex
	kept map[string]Receipt
}

func (r *sharedReceipts) Load(_ context.Context, key string) (Receipt, bool, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	receipt, ok := r.kept[key]
	return receipt, ok, nil
}

func (r *sharedReceipts) Store(_ context.Context, key string, receipt Receipt) error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.kept[key] = receipt
	return nil
}

func (s *RunSuite) TestASharedStoreFencesACallAcrossProcesses() {
	receipts := &sharedReceipts{kept: map[string]Receipt{}}
	var ran atomic.Int32
	registries := make([]*Registry, 2)
	for i := range registries {
		registries[i] = NewRegistry()
		registries[i].SetReceipts(receipts)
		s.Require().NoError(Register(registries[i], "book", "Book", func(context.Context, struct{}) (any, error) {
			ran.Add(1)
			return "booked", nil
		}))
	}

	for _, registry := range registries {
		output, err := registry.Run(s.T().Context(), Call{ID: "call-1", SessionID: "s"}, "book", "{}")
		s.Require().NoError(err)
		s.Equal("booked", output.Text)
	}

	s.EqualValues(1, ran.Load(), "the booking ran in both processes")
}

func (s *RunSuite) TestABadOptionIsRefusedAtRegistration() {
	run := func(context.Context, struct{}) (any, error) { return nil, nil }

	s.Error(Register(s.registry, "a", "a", run, Timeout(0)))
	s.Error(Register(s.registry, "b", "b", run, Concurrency(0)))
	s.Error(Register(s.registry, "c", "c", run, NeedsApproval("  ")))
}

func (s *RunSuite) TestAnApprovalQuestionMayNameOnlyTheToolsArguments() {
	run := func(context.Context, struct {
		FlightID string `json:"flight_id"`
		Seats    int    `json:"seats"`
	}) (any, error) {
		return nil, nil
	}

	s.Require().NoError(Register(s.registry, "book", "Book a flight", run, NeedsApproval("Book {seats} seats on {flight_id}?")))
	err := Register(s.registry, "rebook", "Rebook a flight", run, NeedsApproval("Rebook {flight}?"))
	s.Require().Error(err)
	s.Contains(err.Error(), "{flight}")

	registered := s.registry.List()
	s.Require().Len(registered, 1)
	s.Equal("Book {seats} seats on {flight_id}?", registered[0].Approval)
}
