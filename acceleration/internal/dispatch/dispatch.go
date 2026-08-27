// Package dispatch hands an arriving call to one of the workers waiting for one.
//
// A call arrives at this service and has to be answered somewhere else: the agent runs in a
// customer's own process, which this service cannot reach. So the workers connect here and
// wait, and an arriving call is pushed down one of those connections. That is the whole of
// the inversion: nothing here knows how to start an agent, only which worker to wake.
//
// Workers are kept per customer. Two customers' workers are two independent rotations, so a
// busy customer cannot push another customer's calls onto a worker that is not theirs.
package dispatch

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

// ErrNoWorkers means nobody is waiting for a call. It is a distinct error because it is the
// one failure a caller can do nothing about: the call arrived, and there is no agent to
// answer it.
var ErrNoWorkers = errors.New("dispatch: no worker is waiting for a call")

// Call is an arriving call, described in the terms a worker needs to join it.
type Call struct {
	// CallID and CallType name the Stream call the caller is already in. An agent that
	// joins anything else hears silence.
	CallID   string
	CallType string
	// CalledNumber is the number that was rung, which is how a worker serving several
	// numbers knows which line this is.
	CalledNumber string
	// CallerNumber is who is calling, taken from the SIP participant's id.
	CallerNumber string
	// Custom is whatever was put on the Stream call, carried through unread.
	Custom map[string]string
	// At is when the call started, so a worker can tell a call it has just been handed
	// from one that waited in a queue.
	At time.Time
}

// Load is what a worker last said about itself.
//
// Round robin does not read any of it. It is reported because a policy that does is the
// next thing this will need, and a policy cannot be written against numbers nobody is
// collecting yet.
type Load struct {
	// ActiveAgents is how many calls the worker is currently in.
	ActiveAgents int
	// CPUPercent and MemoryPercent are the worker host's, not the process's, because what
	// matters is whether the host can take another call.
	CPUPercent    float64
	MemoryPercent float64
	// LatencyMs is the round trip the worker measured to this service. The worker measures
	// it rather than this service, because the network the worker is on is the one that
	// will carry the audio.
	LatencyMs float64
	// At is when the worker said it.
	At time.Time
}

// Worker is one connected process waiting for calls.
type Worker struct {
	// ID identifies the worker for the length of its connection. It is not stable across
	// reconnects, because a worker that reconnected is not holding the calls it had.
	ID         string
	CustomerID string

	// calls is buffered to the worker's declared capacity, so a worker with nothing free
	// is passed over rather than blocking the call that arrived.
	calls chan Call

	mu   sync.Mutex
	load Load
}

// Calls is what the worker's connection reads from. It is closed when the worker is
// released, which is what tells the connection to stop.
func (w *Worker) Calls() <-chan Call { return w.calls }

// Load returns what the worker last reported.
func (w *Worker) Load() Load {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.load
}

// Report records what the worker says about itself.
func (w *Worker) Report(load Load) {
	if load.At.IsZero() {
		load.At = time.Now().UTC()
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.load = load
}

// Pool is the workers currently connected, and whose turn it is.
type Pool struct {
	mu sync.Mutex
	// workers and cursors are keyed by customer, so one customer's rotation is untouched
	// by another's.
	workers map[string][]*Worker
	cursors map[string]int
	next    int
}

// NewPool returns an empty pool.
func NewPool() *Pool {
	return &Pool{
		workers: make(map[string][]*Worker),
		cursors: make(map[string]int),
	}
}

// Register adds a worker and returns it with the function that removes it.
//
// Releasing is the caller's job rather than something inferred from the connection, because
// a worker still in the pool after its socket closed is a call sent into a closed channel.
func (p *Pool) Register(customerID string, capacity int) (*Worker, func()) {
	if capacity < 1 {
		capacity = 1
	}

	p.mu.Lock()
	p.next++
	worker := &Worker{
		ID:         fmt.Sprintf("worker-%d", p.next),
		CustomerID: customerID,
		calls:      make(chan Call, capacity),
	}
	p.workers[customerID] = append(p.workers[customerID], worker)
	p.mu.Unlock()

	return worker, func() { p.release(worker) }
}

// Workers returns the workers waiting for one customer's calls, in rotation order.
func (p *Pool) Workers(customerID string) []*Worker {
	p.mu.Lock()
	defer p.mu.Unlock()
	// A copy, because the caller reading this must not see the slice change underneath it.
	return append([]*Worker(nil), p.workers[customerID]...)
}

// Assign gives a call to the next worker whose turn it is.
//
// Round robin, with one exception: a worker whose channel is full is skipped rather than
// waited for. Waiting would hold the arriving call behind a worker that already has more
// than it can answer, and there is another worker right there.
//
// The worker it went to is returned, which is what a caller logs and what a test asserts on.
func (p *Pool) Assign(customerID string, call Call) (*Worker, error) {
	if call.CallID == "" {
		return nil, errors.New("dispatch: a call needs an id")
	}

	// The whole of this is under the lock, including the sends. They cannot block, because
	// the buffer is the worker's capacity and a full one is skipped, and holding the lock
	// is what stops a worker being released between being chosen and being sent to: a send
	// on the channel release closed would panic.
	p.mu.Lock()
	defer p.mu.Unlock()

	waiting := p.workers[customerID]
	if len(waiting) == 0 {
		return nil, ErrNoWorkers
	}
	start := p.cursors[customerID]
	p.cursors[customerID] = (start + 1) % len(waiting)

	for offset := range waiting {
		worker := waiting[(start+offset)%len(waiting)]
		select {
		case worker.calls <- call:
			return worker, nil
		default:
			// Full. Try the next one.
		}
	}
	return nil, fmt.Errorf("dispatch: every worker for %s is at capacity", customerID)
}

// release takes a worker out of the rotation and closes its channel.
func (p *Pool) release(worker *Worker) {
	p.mu.Lock()
	defer p.mu.Unlock()

	waiting := p.workers[worker.CustomerID]
	for index, held := range waiting {
		if held != worker {
			continue
		}
		p.workers[worker.CustomerID] = append(waiting[:index:index], waiting[index+1:]...)
		close(worker.calls)
		break
	}
	if len(p.workers[worker.CustomerID]) == 0 {
		delete(p.workers, worker.CustomerID)
		delete(p.cursors, worker.CustomerID)
		return
	}
	// The cursor indexes a slice that just got shorter, so it has to be brought back
	// inside it or the next assignment panics.
	p.cursors[worker.CustomerID] %= len(p.workers[worker.CustomerID])
}
