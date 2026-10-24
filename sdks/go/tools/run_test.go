package tools

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

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

func burst(t *testing.T, registry *Registry, name string, calls int) {
	t.Helper()
	var all sync.WaitGroup
	for range calls {
		all.Go(func() {
			if _, err := registry.Call(t.Context(), name, "{}"); err != nil {
				t.Error(err)
			}
		})
	}
	all.Wait()
}

func TestABurstOfCallsIsHeldToTheRegistrysCeiling(t *testing.T) {
	registry := NewRegistry()
	var seen peak
	err := Register(registry, "query", "Query the database", func(context.Context, struct{}) (any, error) {
		defer seen.enter()()
		time.Sleep(20 * time.Millisecond)
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}

	burst(t, registry, "query", 50)
	if got := seen.most.Load(); got > DefaultConcurrency {
		t.Errorf("%d calls ran at once, want at most %d", got, DefaultConcurrency)
	}
}

func TestAToolsOwnConcurrencyHoldsBelowTheRegistrys(t *testing.T) {
	registry := NewRegistry()
	var seen peak
	err := Register(registry, "query", "Query the database", func(context.Context, struct{}) (any, error) {
		defer seen.enter()()
		time.Sleep(20 * time.Millisecond)
		return "ok", nil
	}, Concurrency(4))
	if err != nil {
		t.Fatal(err)
	}

	burst(t, registry, "query", 50)
	if got := seen.most.Load(); got > 4 {
		t.Errorf("%d calls ran at once, want at most 4", got)
	}
}

func TestACallWaitingForAPlaceTimesOutOnItsOwnDeadline(t *testing.T) {
	registry := NewRegistry()
	release := make(chan struct{})
	// The function ignores its context, so the first call holds the one place until released.
	err := Register(registry, "query", "Query the database", func(context.Context, struct{}) (any, error) {
		<-release
		return "ok", nil
	}, Concurrency(1), Timeout(100*time.Millisecond))
	if err != nil {
		t.Fatal(err)
	}

	go func() { _, _ = registry.Run(t.Context(), Call{ID: "first"}, "query", "{}") }()
	time.Sleep(20 * time.Millisecond)
	began := time.Now()
	_, err = registry.Run(t.Context(), Call{ID: "second"}, "query", "{}")
	close(release)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("the waiting call ended with %v, want its deadline", err)
	}
	if waited := time.Since(began); waited > time.Second {
		t.Errorf("the waiting call took %s to give up", waited)
	}
}

func TestAFunctionReadsTheCallItIsRunningFor(t *testing.T) {
	registry := NewRegistry()
	seen := make(chan Call, 1)
	err := Register(registry, "lookup_order", "Look up an order", func(ctx context.Context, _ struct{}) (any, error) {
		seen <- *CallFrom(ctx)
		return "ok", nil
	})
	if err != nil {
		t.Fatal(err)
	}

	want := Call{ID: "call-1", SessionID: "session-1", TurnID: "turn-1"}
	if _, err := registry.Run(t.Context(), want, "lookup_order", ""); err != nil {
		t.Fatal(err)
	}
	if got := <-seen; got.ID != want.ID || got.SessionID != want.SessionID || got.TurnID != want.TurnID {
		t.Errorf("the function saw %+v, want %+v", got, want)
	}
}

func TestOutsideACallTheCallIsEmptyRatherThanNil(t *testing.T) {
	if call := CallFrom(context.Background()); call == nil || call.ID != "" {
		t.Errorf("CallFrom outside a call returned %+v", call)
	}
}

func TestCancellingACallStopsItsFunction(t *testing.T) {
	registry := NewRegistry()
	started := make(chan struct{})
	err := Register(registry, "search", "Search", func(ctx context.Context, _ struct{}) (any, error) {
		close(started)
		<-ctx.Done()
		return nil, ctx.Err()
	})
	if err != nil {
		t.Fatal(err)
	}

	ended := make(chan error, 1)
	go func() {
		_, err := registry.Run(t.Context(), Call{ID: "call-1", SessionID: "s"}, "search", "{}")
		ended <- err
	}()
	<-started
	if !registry.Cancel("s", "call-1") {
		t.Fatal("the running call was not found to cancel")
	}
	select {
	case err := <-ended:
		if !errors.Is(err, context.Canceled) {
			t.Errorf("the call ended with %v", err)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("the cancelled call is still running")
	}
	if registry.Cancel("s", "call-1") {
		t.Error("a call that has ended was found to cancel")
	}
}

func TestACancelledCallLeavesNoReceipt(t *testing.T) {
	registry := NewRegistry()
	var ran atomic.Int32
	err := Register(registry, "search", "Search", func(ctx context.Context, _ struct{}) (any, error) {
		if ran.Add(1) == 1 {
			<-ctx.Done()
			return nil, ctx.Err()
		}
		return "found", nil
	})
	if err != nil {
		t.Fatal(err)
	}

	ctx, cancel := context.WithCancel(t.Context())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	_, _ = registry.Run(ctx, Call{ID: "call-1"}, "search", "{}")
	output, err := registry.Run(t.Context(), Call{ID: "call-1"}, "search", "{}")
	if err != nil || output.Text != "found" {
		t.Errorf("the call sent again got %q, %v; a cancelled call is no answer to repeat", output.Text, err)
	}
}

// sharedReceipts is a store two registries share, as two processes would share Redis.
type sharedReceipts struct {
	mu   sync.Mutex
	kept map[string]Receipt
}

func (s *sharedReceipts) Load(_ context.Context, key string) (Receipt, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	receipt, ok := s.kept[key]
	return receipt, ok, nil
}

func (s *sharedReceipts) Store(_ context.Context, key string, receipt Receipt) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.kept[key] = receipt
	return nil
}

func TestASharedStoreFencesACallAcrossProcesses(t *testing.T) {
	receipts := &sharedReceipts{kept: map[string]Receipt{}}
	var ran atomic.Int32
	registries := make([]*Registry, 2)
	for i := range registries {
		registries[i] = NewRegistry()
		registries[i].SetReceipts(receipts)
		err := Register(registries[i], "book", "Book", func(context.Context, struct{}) (any, error) {
			ran.Add(1)
			return "booked", nil
		})
		if err != nil {
			t.Fatal(err)
		}
	}

	call := Call{ID: "call-1", SessionID: "s"}
	for _, registry := range registries {
		output, err := registry.Run(t.Context(), call, "book", "{}")
		if err != nil || output.Text != "booked" {
			t.Fatalf("got %q, %v", output.Text, err)
		}
	}
	if got := ran.Load(); got != 1 {
		t.Errorf("the booking ran %d times across two processes", got)
	}
}

func TestABadOptionIsRefusedAtRegistration(t *testing.T) {
	registry := NewRegistry()
	run := func(context.Context, struct{}) (any, error) { return nil, nil }
	if err := Register(registry, "a", "a", run, Timeout(0)); err == nil {
		t.Error("a zero timeout was accepted")
	}
	if err := Register(registry, "b", "b", run, Concurrency(0)); err == nil {
		t.Error("zero concurrency was accepted")
	}
}
