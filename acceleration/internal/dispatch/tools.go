package dispatch

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// defaultHostedTimeout is how long a hosted tool is given when its worker did not say.
const defaultHostedTimeout = 2 * time.Minute

// toolQueue is how many hosted calls may wait on one worker's socket. A call is a small
// frame and the worker runs each on its own, so this only fills when the socket has stopped
// being read.
const toolQueue = 64

// Tool is a function a worker runs for sessions it did not open.
//
// A session's own tools are answered by whoever opened it, which is the right owner for a
// function that lives beside the caller. Some do not: a browser opening a conversation
// cannot read a source tree or reach a sandbox. A worker that can says so once, for an
// agent id, and every session opened under that agent is offered the tool and has its
// calls sent here.
type Tool struct {
	Name        string
	Description string
	// Parameters is a JSON Schema object describing the arguments.
	Parameters map[string]any
}

// ToolCall is one call a worker is asked to run.
type ToolCall struct {
	// ID is the id the model gave the call, which the result is matched on.
	ID string
	// SessionID is the conversation the call was made in, for the worker's logs.
	SessionID string
	Name      string
	// Arguments is the JSON object the model wrote.
	Arguments string
}

// ToolResult is what the worker said happened.
type ToolResult struct {
	Output string
	// Failure is why the tool did not work, in words for the model. Empty on success.
	Failure string
}

// hosting is one worker's offer of tools for one agent id.
type hosting struct {
	worker  *Worker
	tools   []Tool
	timeout time.Duration
}

func (h *hosting) offers(name string) bool {
	for _, tool := range h.tools {
		if tool.Name == name {
			return true
		}
	}
	return false
}

// ToolCalls is the hosted calls the worker's connection has to deliver. It is closed with
// the rest of the worker's queues.
func (w *Worker) ToolCalls() <-chan ToolCall { return w.toolCalls }

// Resolve hands a worker's answer back to the call waiting for it, reporting whether one was.
//
// An answer nobody is waiting on is dropped: the commonest reason for one is a tool that
// finished after the conversation gave up on it.
func (w *Worker) Resolve(id string, result ToolResult) bool {
	w.mu.Lock()
	waiting, found := w.results[id]
	delete(w.results, id)
	w.mu.Unlock()
	if !found {
		return false
	}
	waiting <- result
	return true
}

// Host records that a worker runs these tools for sessions under an agent id, replacing
// whatever it offered for that agent before. The offer is scoped to the worker's own
// customer: this is the pool, not the gate.
func (p *Pool) Host(worker *Worker, agentID string, tools []Tool, timeout time.Duration) error {
	if agentID == "" {
		return errors.New("dispatch: hosted tools need the agent they are for")
	}
	if len(tools) == 0 {
		return errors.New("dispatch: hosting no tools is not hosting")
	}
	if timeout <= 0 {
		timeout = defaultHostedTimeout
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.registered(worker) {
		return errors.New("dispatch: that worker has already gone")
	}
	byAgent := p.hosted[worker.CustomerID]
	if byAgent == nil {
		byAgent = map[string][]*hosting{}
		p.hosted[worker.CustomerID] = byAgent
	}
	offers := byAgent[agentID][:0:0]
	for _, offer := range byAgent[agentID] {
		if offer.worker != worker {
			offers = append(offers, offer)
		}
	}
	byAgent[agentID] = append(offers, &hosting{worker: worker, tools: append([]Tool(nil), tools...), timeout: timeout})
	return nil
}

// HostedTools is what the workers connected now run for sessions under one agent id, each
// name once, and how long the slowest of them is given.
func (p *Pool) HostedTools(customerID, agentID string) ([]Tool, time.Duration) {
	p.mu.Lock()
	defer p.mu.Unlock()

	var (
		offered []Tool
		seen    = map[string]bool{}
		longest time.Duration
	)
	for _, offer := range p.hosted[customerID][agentID] {
		longest = max(longest, offer.timeout)
		for _, tool := range offer.tools {
			if !seen[tool.Name] {
				seen[tool.Name] = true
				offered = append(offered, tool)
			}
		}
	}
	return offered, longest
}

// RunHosted sends one call to a worker hosting that tool for the agent and waits for its
// answer, the worker's own timeout, or the caller giving up.
//
// Workers take turns the way they do for calls, and one whose queue is full is passed over.
func (p *Pool) RunHosted(ctx context.Context, customerID, agentID string, call ToolCall) (string, error) {
	if call.ID == "" {
		return "", errors.New("dispatch: a hosted call needs an id")
	}
	answer := make(chan ToolResult, 1)

	p.mu.Lock()
	offers := p.hosted[customerID][agentID]
	key := customerID + "\x00" + agentID + "\x00" + call.Name
	start := p.toolCursors[key]
	p.toolCursors[key] = start + 1
	var chosen *hosting
	for offset := range offers {
		offer := offers[(start+offset)%len(offers)]
		if !offer.offers(call.Name) {
			continue
		}
		offer.worker.mu.Lock()
		if _, duplicate := offer.worker.results[call.ID]; duplicate {
			offer.worker.mu.Unlock()
			p.mu.Unlock()
			return "", fmt.Errorf("dispatch: %s was already asked for", call.ID)
		}
		offer.worker.results[call.ID] = answer
		offer.worker.mu.Unlock()
		select {
		case offer.worker.toolCalls <- call:
			chosen = offer
		default:
			offer.worker.forget(call.ID)
			continue
		}
		break
	}
	p.mu.Unlock()
	if chosen == nil {
		return "", fmt.Errorf("dispatch: no worker is running %s now", call.Name)
	}

	deadline, cancel := context.WithTimeout(ctx, chosen.timeout)
	defer cancel()
	select {
	case result := <-answer:
		if result.Failure != "" {
			return "", errors.New(result.Failure)
		}
		return result.Output, nil
	case <-deadline.Done():
		chosen.worker.forget(call.ID)
		if err := ctx.Err(); err != nil {
			return "", fmt.Errorf("dispatch: %s stopped: %w", call.Name, err)
		}
		return "", fmt.Errorf("dispatch: %s did not answer within %s", call.Name, chosen.timeout)
	}
}

func (w *Worker) forget(id string) {
	w.mu.Lock()
	delete(w.results, id)
	w.mu.Unlock()
}

// registered reports whether a worker is still in the pool. The caller holds the lock.
func (p *Pool) registered(worker *Worker) bool {
	for _, held := range p.workers[worker.CustomerID] {
		if held == worker {
			return true
		}
	}
	return false
}

// unhost takes a released worker's offers out, and fails the calls it was running so a
// conversation is told the tool went away rather than waiting out its timeout. The caller
// holds the lock.
func (p *Pool) unhost(worker *Worker) {
	byAgent := p.hosted[worker.CustomerID]
	for agentID, offers := range byAgent {
		kept := offers[:0:0]
		for _, offer := range offers {
			if offer.worker != worker {
				kept = append(kept, offer)
			}
		}
		if len(kept) == 0 {
			delete(byAgent, agentID)
		} else {
			byAgent[agentID] = kept
		}
	}
	if len(byAgent) == 0 {
		delete(p.hosted, worker.CustomerID)
	}

	worker.mu.Lock()
	waiting := worker.results
	worker.results = map[string]chan ToolResult{}
	worker.mu.Unlock()
	for _, answer := range waiting {
		answer <- ToolResult{Failure: "the worker running this tool disconnected"}
	}
}
