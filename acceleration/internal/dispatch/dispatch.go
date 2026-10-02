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
	"strconv"
	"sync"
	"time"
)

// ErrNoWorkers means nobody is waiting. It is a distinct error because it is the one
// failure a caller can do nothing about: the call arrived, and there is no agent to answer
// it.
var ErrNoWorkers = errors.New("dispatch: no worker is waiting for a call")

const (
	// loadFresh is how old a worker's report may be before it is ignored. A worker says how
	// it is doing every fifteen seconds, so this is three reports: long enough to ride out
	// one that was lost, short enough that nothing is judged on a minute-old figure.
	loadFresh = 45 * time.Second
	// hotPercent is the CPU or memory at which a worker is left alone if anybody else can
	// take the work. It is a backstop rather than the policy: what is being handled now is
	// counted exactly, and this only catches a host in trouble for some other reason.
	hotPercent = 90.0
)

// Kind is what a piece of work is. A worker says which kinds it handles, because one that
// only answers in writing has no reason to be handed a phone call: it would drop it, and
// the caller would listen to a phone nobody picks up.
type Kind string

const (
	Calls    Kind = "call"
	Messages Kind = "message"
)

// Call is an arriving call, described in the terms a worker needs to join it.
type Call struct {
	// WorkID is what the worker names this call when it reports it finished. It is given
	// here rather than taken from CallID because every kind of work is counted the same
	// way, and a worker holding a call and a message holds two of whatever it can hold.
	WorkID string
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

// Message is something written to an agent that no session is running for.
//
// It arrives the same way a call does and for the same reason: the channel is reachable from
// here and the agent is not. Unlike a call it names no call to join, because there is
// nothing to join — the conversation is the channel.
type Message struct {
	// WorkID is what the worker names this message when it reports it finished, the way a
	// call's does.
	WorkID string
	// ChannelType and ChannelID name where it was written. Answering anywhere else would
	// be a reply nobody asked for in a conversation nobody is reading.
	ChannelType string
	ChannelID   string
	// AgentID is the agent the channel belongs to, which is also what a session started to
	// answer this should be given so its own replies land back here.
	AgentID string
	// ConfigID names the agent config the last conversation here ran under, so a worker
	// knows which agent is being written to rather than having to guess from the channel.
	ConfigID string
	// SessionID is the running session the message was written to, set when its agent
	// leaves text to dispatch. The worker answers by creating a response on it.
	SessionID string
	// CommandID is the durable command the message was sent as, which the worker passes
	// back when it creates that response so the reply lands on the command it answers.
	CommandID string
	// Custom is whatever the channel was created with, carried through unread the way a
	// call's is. It is how a worker learns what the conversation is for — which customer's
	// organization it belongs to, which locale to answer in — without this service having
	// to know what any of those mean.
	//
	// Whoever created the channel decided what is in here, so it is a claim rather than a
	// fact. ConfigID is deliberately not among the things read from it; see ConfigField.
	Custom map[string]string
	// Text is what was written.
	Text string
	// MessageID is the message in the channel, so a worker can reply in its thread or
	// react to it rather than only after it.
	MessageID string
	// UserID and UserName are who wrote it.
	UserID   string
	UserName string
	// At is when it was written.
	At time.Time
}

// Load is what a worker last said about itself.
//
// How much work a worker is holding is counted here rather than read from this, because a
// figure up to a report old is no basis for a decision taken now. What is read from it is
// CPU and memory, which nothing here can see any other way, and the work an older worker
// reports, which is the only number it gives.
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

// Registration is what a worker says about itself when it connects.
type Registration struct {
	// Capacity is how much work it takes at once, of every kind together. A worker holding
	// a call and a message is holding two.
	Capacity int
	// Active is the work it is still running from before it reconnected. Those pieces were
	// handed to a worker that has gone, so the pool has no record of them and would
	// otherwise fill this one up on top of what it is already doing.
	Active int
	// Handles is the kinds of work it accepts. Nil is all of them, which is what a worker
	// built against a router that never asked means. Empty is none of them, which a process
	// that only hosts tools is.
	Handles []Kind
	// Tracking is whether the worker reports each piece of work finished. One that does is
	// held to its capacity; one that does not is passed over only when its queue backs up,
	// which is all an older worker gave anybody to go on.
	Tracking bool
}

// Worker is one connected process waiting for calls.
type Worker struct {
	// ID identifies the worker for the length of its connection. It is not stable across
	// reconnects, because a worker that reconnected is not holding the calls it had.
	ID         string
	CustomerID string

	// capacity is how much work the worker said it takes at once.
	capacity int
	// handles is the kinds of work it accepts, nil meaning all of them.
	handles []Kind
	// tracking is whether it reports work finished, and so whether what it is holding is
	// known here rather than guessed at.
	tracking bool

	// calls is buffered to the worker's declared capacity, so a worker whose connection
	// has stopped reading is passed over rather than blocking the call that arrived.
	calls chan Call
	// messages is a queue of its own, so one kind of work cannot be stuck behind the
	// other on the way to the socket. What a worker may hold at once is capacity, which
	// counts both.
	messages chan Message
	// toolCalls is the hosted calls to deliver, and results the calls waiting on an
	// answer, keyed by the id the model gave each.
	toolCalls chan ToolCall

	mu sync.Mutex
	// working is the work handed to this worker that it has not reported finished, and
	// carried the work it brought with it through a reconnect, which has no id here.
	working map[string]struct{}
	carried int
	load    Load
	results map[string]chan ToolResult
}

// Calls is what the worker's connection reads from. It is closed when the worker is
// released, which is what tells the connection to stop.
func (w *Worker) Calls() <-chan Call { return w.calls }

// Messages is the other thing the worker's connection reads from. It is closed with the
// calls, when the worker is released.
func (w *Worker) Messages() <-chan Message { return w.messages }

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

// Done records that the worker has finished one piece of work, freeing what it was holding.
//
// An id nothing was handed under is work the worker carried through a reconnect: the pool
// that handed it out has gone, so all that is left of it is the count this worker arrived
// with, and finishing one of those is what takes the count down.
func (w *Worker) Done(workID string) {
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, held := w.working[workID]; held {
		delete(w.working, workID)
		return
	}
	if w.carried > 0 {
		w.carried--
	}
}

// Working is how much work the worker is holding right now.
func (w *Worker) Working() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.working) + w.carried
}

// takes reports whether this worker accepts that kind of work.
func (w *Worker) takes(kind Kind) bool {
	if w.handles == nil {
		return true
	}
	for _, handled := range w.handles {
		if handled == kind {
			return true
		}
	}
	return false
}

// holding is how much work the worker has, as a share of what it said it can hold.
//
// A worker that does not report work finished is read from what it last said about itself,
// which is up to a report out of date but is the only number it gives. A report too old to
// trust reads as idle rather than as busy, because refusing an older worker work on the
// strength of a stale figure would leave it idle for good.
func (w *Worker) holding(fresh time.Time) float64 {
	w.mu.Lock()
	defer w.mu.Unlock()

	held := w.load.ActiveAgents
	if w.tracking {
		held = len(w.working) + w.carried
	} else if !w.load.At.After(fresh) {
		held = 0
	}
	return float64(held) / float64(w.capacity)
}

// full reports whether the worker is at the capacity it declared. Only a worker that reports
// work finished can be: what an older one is holding is not known here, so the only thing
// that passes it over is a queue that has backed up.
func (w *Worker) full() bool {
	if !w.tracking {
		return false
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.working)+w.carried >= w.capacity
}

// hot reports whether the worker's host is in trouble, by its own account and recently
// enough to believe.
func (w *Worker) hot(fresh time.Time) bool {
	w.mu.Lock()
	defer w.mu.Unlock()
	if !w.load.At.After(fresh) {
		return false
	}
	return w.load.CPUPercent >= hotPercent || w.load.MemoryPercent >= hotPercent
}

// take records that one piece of work has been handed to this worker. The caller holds the
// pool's lock.
func (w *Worker) take(workID string) {
	if !w.tracking {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	w.working[workID] = struct{}{}
}

// Pool is the workers currently connected, and whose turn it is.
type Pool struct {
	mu sync.Mutex
	// workers and cursors are keyed by customer, so one customer's rotation is untouched
	// by another's.
	workers map[string][]*Worker
	cursors map[string]int
	next    int
	// work numbers the pieces of work handed out, so a worker reporting one finished names
	// something this pool gave it rather than something it chose.
	work int
	// hosted is the tools workers run for other sessions, by customer and then agent
	// id, and toolCursors whose turn it is for each tool.
	hosted      map[string]map[string][]*hosting
	toolCursors map[string]int
}

// NewPool returns an empty pool.
func NewPool() *Pool {
	return &Pool{
		workers:     make(map[string][]*Worker),
		cursors:     make(map[string]int),
		hosted:      make(map[string]map[string][]*hosting),
		toolCursors: make(map[string]int),
	}
}

// Register adds a worker and returns it with the function that removes it.
//
// Releasing is the caller's job rather than something inferred from the connection, because
// a worker still in the pool after its socket closed is a call sent into a closed channel.
func (p *Pool) Register(customerID string, registration Registration) (*Worker, func()) {
	if registration.Capacity < 1 {
		registration.Capacity = 1
	}

	p.mu.Lock()
	p.next++
	worker := &Worker{
		ID:         fmt.Sprintf("worker-%d", p.next),
		CustomerID: customerID,
		capacity:   registration.Capacity,
		handles:    registration.Handles,
		tracking:   registration.Tracking,
		calls:      make(chan Call, registration.Capacity),
		messages:   make(chan Message, registration.Capacity),
		toolCalls:  make(chan ToolCall, toolQueue),
		working:    map[string]struct{}{},
		carried:    max(registration.Active, 0),
		results:    map[string]chan ToolResult{},
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

// Assign gives a call to the worker best placed to answer it.
//
// The worker it went to is returned, which is what a caller logs and what a test asserts on.
func (p *Pool) Assign(customerID string, call Call) (*Worker, error) {
	if call.CallID == "" {
		return nil, errors.New("dispatch: a call needs an id")
	}
	return assign(p, customerID, call, Calls,
		func(worker *Worker) chan Call { return worker.calls },
		func(call *Call, workID string) { call.WorkID = workID })
}

// AssignMessage gives a message to a worker the same way a call is given, and against the
// same capacity: what a worker can hold is what it can hold, whichever kind of work fills
// it.
func (p *Pool) AssignMessage(customerID string, message Message) (*Worker, error) {
	if message.ChannelID == "" && message.SessionID == "" {
		return nil, errors.New("dispatch: a message needs a channel or a session")
	}
	return assign(p, customerID, message, Messages,
		func(worker *Worker) chan Message { return worker.messages },
		func(message *Message, workID string) { message.WorkID = workID })
}

// assign hands one piece of work to the worker holding the least of what it said it can
// hold, among those that take that kind of work and have room for more.
//
// The share rather than the count, so a worker that promised to hold twenty is given more
// than one that promised four. Equal shares go to whoever is next in the rotation, which is
// what keeps idle workers taking turns rather than piling onto whichever was registered
// first. A worker whose host it has just said is in trouble is left alone unless nobody
// else can take the work, since the alternative to a struggling worker is nobody at all.
//
// The whole of this is under the lock, including the send. It cannot block, because the
// buffer is the worker's capacity and a worker with a full queue is skipped, and holding
// the lock is what stops a worker being released between being chosen and being sent to: a
// send on the channel release closed would panic.
func assign[T any](p *Pool, customerID string, work T, kind Kind, queue func(*Worker) chan T, stamp func(*T, string)) (*Worker, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	waiting := p.workers[customerID]
	if len(waiting) == 0 {
		return nil, ErrNoWorkers
	}
	start := p.cursors[customerID]
	p.cursors[customerID] = (start + 1) % len(waiting)

	fresh := time.Now().UTC().Add(-loadFresh)
	var (
		chosen *Worker
		least  float64
		wasHot bool
		takers bool
	)
	for offset := range waiting {
		worker := waiting[(start+offset)%len(waiting)]
		if !worker.takes(kind) {
			continue
		}
		takers = true
		if worker.full() || len(queue(worker)) == cap(queue(worker)) {
			continue
		}
		held, hot := worker.holding(fresh), worker.hot(fresh)
		// Strictly better, so the first worker reached from the cursor keeps a tie.
		if chosen == nil || (wasHot && !hot) || (wasHot == hot && held < least) {
			chosen, least, wasHot = worker, held, hot
		}
	}
	if chosen == nil {
		if !takers {
			return nil, fmt.Errorf("dispatch: no worker for %s handles %s work", customerID, kind)
		}
		return nil, fmt.Errorf("dispatch: every worker for %s is at capacity", customerID)
	}

	p.work++
	workID := "work-" + strconv.Itoa(p.work)
	stamp(&work, workID)
	chosen.take(workID)
	queue(chosen) <- work
	return chosen, nil
}

// release takes a worker out of the rotation and closes its channels.
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
		close(worker.messages)
		close(worker.toolCalls)
		p.unhost(worker)
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
