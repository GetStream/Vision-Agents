package tools

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"
)

// DefaultConcurrency is how many calls a registry runs at once unless told otherwise.
const DefaultConcurrency = 16

// receiptsKept is how long the in-memory receipts remember a finished call. The router
// sends a call again only while it is still waiting for it, which is the tool's timeout.
const receiptsKept = 10 * time.Minute

// Receipt is the outcome of a call that finished, kept so the same call is not run twice.
type Receipt struct {
	Output Output
	// Failure is the error the call failed with, as words. Empty on success.
	Failure string
}

// Receipts remember which calls have finished, and with what.
//
// The registry keeps them in memory, which fences a call sent again to the same process.
// A deployment that runs more than one process, or restarts mid-call, backs them with a
// store every process shares, such as Redis or Postgres.
type Receipts interface {
	// Load returns the receipt stored under key, if there is one.
	Load(ctx context.Context, key string) (Receipt, bool, error)
	// Store keeps a receipt under key.
	Store(ctx context.Context, key string, receipt Receipt) error
}

// running is one call in flight, which a copy of it waits on rather than running again.
type running struct {
	done    chan struct{}
	cancel  context.CancelFunc
	output  Output
	failure error
}

// Run runs one call, at most once for its id, and returns what it said.
//
// A copy of a call still running here waits for it and gets the same answer, and a copy of
// one that finished gets its receipt. The function runs with the call in its context
// (ToolCallContextFrom), within the function's Timeout and its share of the registry's concurrency.
func (r *Registry) Run(ctx context.Context, call Call, name string, arguments string) (Output, error) {
	r.mu.RLock()
	function := r.functions[name]
	r.mu.RUnlock()
	if function == nil {
		return Output{}, fmt.Errorf("tools: nothing is registered as %s", name)
	}
	if arguments == "" {
		arguments = "{}"
	}
	ctx, finished := startRun(ctx)
	defer finished()
	if call.ID == "" {
		return r.execute(ctx, function, &call, arguments)
	}

	key := call.SessionID + "/" + call.ID
	r.mu.Lock()
	if earlier, ok := r.running[key]; ok {
		r.mu.Unlock()
		select {
		case <-earlier.done:
			return earlier.output, earlier.failure
		case <-ctx.Done():
			return Output{}, ctx.Err()
		}
	}
	callCtx, cancel := context.WithCancel(ctx)
	current := &running{done: make(chan struct{}), cancel: cancel}
	r.running[key] = current
	receipts := r.receipts
	r.mu.Unlock()

	defer func() {
		cancel()
		r.mu.Lock()
		delete(r.running, key)
		r.mu.Unlock()
		close(current.done)
	}()

	receipt, found, err := receipts.Load(ctx, key)
	if err != nil {
		current.failure = fmt.Errorf("tools: could not check whether %s already ran: %w", name, err)
		return Output{}, current.failure
	}
	if found {
		current.output = receipt.Output
		if receipt.Failure != "" {
			current.failure = errors.New(receipt.Failure)
		}
		return current.output, current.failure
	}

	current.output, current.failure = r.execute(callCtx, function, &call, arguments)
	// A call stopped before it finished has no outcome to keep: the router gave up on it,
	// and nothing it did is an answer.
	if callCtx.Err() == nil || current.failure == nil {
		stored := Receipt{Output: current.output}
		if current.failure != nil {
			stored.Failure = current.failure.Error()
		}
		_ = receipts.Store(context.WithoutCancel(ctx), key, stored)
	}
	return current.output, current.failure
}

// Cancel stops the call with this id in this session, if one is running here, and reports
// whether one was.
func (r *Registry) Cancel(sessionID, id string) bool {
	r.mu.Lock()
	current, ok := r.running[sessionID+"/"+id]
	r.mu.Unlock()
	if ok {
		current.cancel()
	}
	return ok
}

// Holds reports whether a function is registered under name.
func (r *Registry) Holds(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.functions[name] != nil
}

// SetConcurrency caps how many calls the whole registry runs at once, in place of
// DefaultConcurrency. Call it before the first call.
func (r *Registry) SetConcurrency(calls int) error {
	if calls <= 0 {
		return fmt.Errorf("tools: concurrency must be at least one, not %d", calls)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.slots = make(chan struct{}, calls)
	return nil
}

// SetReceipts replaces the in-memory receipts with a store shared across processes.
func (r *Registry) SetReceipts(receipts Receipts) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.receipts = receipts
}

// execute runs a function within its timeout and the concurrency it is allowed.
func (r *Registry) execute(ctx context.Context, function *Function, call *Call, arguments string) (Output, error) {
	if function.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, function.Timeout)
		defer cancel()
	}
	r.mu.RLock()
	slots := r.slots
	r.mu.RUnlock()
	for _, held := range []chan struct{}{slots, function.slots} {
		if held == nil {
			continue
		}
		select {
		case held <- struct{}{}:
			defer func() { <-held }()
		case <-ctx.Done():
			return Output{}, fmt.Errorf("tools: %s waited for a place to run and gave up: %w", function.Name, ctx.Err())
		}
	}
	if function.schema != nil {
		if err := check(function.schema, []byte(arguments)); err != nil {
			return Output{}, fmt.Errorf("tools: %s was asked for with arguments it cannot take: %w", function.Name, err)
		}
	}
	return function.run(WithToolCallContext(ctx, call), []byte(arguments))
}

// memoryReceipts keep receipts in this process for receiptsKept.
type memoryReceipts struct {
	mu   sync.Mutex
	kept map[string]keptReceipt
}

type keptReceipt struct {
	receipt Receipt
	until   time.Time
}

func newMemoryReceipts() *memoryReceipts {
	return &memoryReceipts{kept: map[string]keptReceipt{}}
}

func (m *memoryReceipts) Load(_ context.Context, key string) (Receipt, bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	kept, ok := m.kept[key]
	if !ok || time.Now().After(kept.until) {
		return Receipt{}, false, nil
	}
	return kept.receipt, true, nil
}

func (m *memoryReceipts) Store(_ context.Context, key string, receipt Receipt) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	now := time.Now()
	for stored, kept := range m.kept {
		if now.After(kept.until) {
			delete(m.kept, stored)
		}
	}
	m.kept[key] = keptReceipt{receipt: receipt, until: now.Add(receiptsKept)}
	return nil
}
