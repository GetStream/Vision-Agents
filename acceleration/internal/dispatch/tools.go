package dispatch

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

// defaultHostedTimeout is how long a hosted tool is given when its worker did not say, which
// is what a session gives the tools its caller runs.
const defaultHostedTimeout = 30 * time.Second

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
	// Interruptible says the caller interrupting cancels a call in flight.
	Interruptible bool `json:",omitempty"`
	// Timeout is how long one call may take, in place of what the worker gave all its
	// tools. Zero takes that.
	Timeout time.Duration `json:",omitempty"`
}

// ToolCall is one call a worker is asked to run.
type ToolCall struct {
	// ID is the id the model gave the call, which the result is matched on.
	ID string
	// SessionID is the conversation the call was made in, for the worker's logs.
	SessionID string
	// TurnID, User and Conversation say what the call is part of: the turn that asked for
	// it, the person whose turn that was, and the conversation it is held in.
	TurnID       string        `json:",omitempty"`
	User         *User         `json:",omitempty"`
	Conversation *Conversation `json:",omitempty"`
	Name         string
	// Arguments is the JSON object the model wrote.
	Arguments string
	// Progress is the conversation the call was made in, which the worker may hold the call
	// on or tell how it is going while it runs. Nil when nobody is listening.
	Progress Progress `json:"-"`
	// Cancel says the call with this id is no longer wanted: its deadline passed or the
	// session gave up on it. It travels on the same queue as the call, so it cannot
	// overtake it.
	Cancel bool `json:",omitempty"`
}

// Progress is what a running hosted call may tell the conversation it was made in, as a
// session's own tool may over the session socket.
type Progress interface {
	// HoldTool keeps the call running when the caller talks over the agent, reporting
	// whether it still was.
	HoldTool(id string) bool
	// ToolUpdate says how the call is going, or with verbatim, words to say as written.
	ToolUpdate(id, text string, verbatim bool)
}

// User is who a hosted call was made for.
type User struct {
	ID   string
	Name string `json:",omitempty"`
}

// Conversation is the conversation a hosted call was made in: its id, when it is kept, and
// the session's custom.
type Conversation struct {
	CID    string         `json:",omitempty"`
	Custom map[string]any `json:",omitempty"`
}

// ToolResult is what the worker said happened.
type ToolResult struct {
	// Output is text and pictures, as a session's own tools answer.
	Output []llm.ContentPart
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

// timeoutOf is how long a call to name is given on this host.
func (h *hosting) timeoutOf(name string) time.Duration {
	for _, tool := range h.tools {
		if tool.Name == name && tool.Timeout > 0 {
			return tool.Timeout
		}
	}
	return h.timeout
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
	delete(w.progress, id)
	w.mu.Unlock()
	if !found {
		return false
	}
	waiting <- result
	return true
}

// Hold keeps a hosted call the worker is running going when the caller talks over the
// agent, reporting the session the call was made in and whether it was still running to be
// held.
func (w *Worker) Hold(id string) (string, bool) {
	call := w.progressOf(id)
	return call.sessionID, call.progress != nil && call.progress.HoldTool(id)
}

// Update tells the conversation a hosted call was made in how it is going. An update for
// a call nobody is waiting on is dropped.
func (w *Worker) Update(id, text string, verbatim bool) {
	if call := w.progressOf(id); call.progress != nil {
		call.progress.ToolUpdate(id, text, verbatim)
	}
}

// listening is a running hosted call's conversation, and the session it is.
type listening struct {
	sessionID string
	progress  Progress
}

func (w *Worker) progressOf(id string) listening {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.progress[id]
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
	for _, tool := range tools {
		if tool.Timeout < 0 {
			return fmt.Errorf("dispatch: %s cannot have a negative timeout", tool.Name)
		}
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
			longest = max(longest, tool.Timeout)
			if !seen[tool.Name] {
				seen[tool.Name] = true
				offered = append(offered, tool)
			}
		}
	}
	return offered, longest
}

// AgentsHostedBy is the agent ids a worker hosts tools for.
func (p *Pool) AgentsHostedBy(worker *Worker) []string {
	p.mu.Lock()
	defer p.mu.Unlock()

	var agents []string
	for agentID, offers := range p.hosted[worker.CustomerID] {
		for _, offer := range offers {
			if offer.worker == worker {
				agents = append(agents, agentID)
				break
			}
		}
	}
	return agents
}

// RunHosted sends one call to a worker hosting that tool for the agent and waits for its
// answer, the worker's own timeout, or the caller giving up.
//
// It goes to whichever host has the least still to answer, and hosts that are equally busy
// take turns. One whose queue is full is passed over.
func (p *Pool) RunHosted(ctx context.Context, customerID, agentID string, call ToolCall) ([]llm.ContentPart, error) {
	if call.ID == "" {
		return nil, errors.New("dispatch: a hosted call needs an id")
	}
	answer := make(chan ToolResult, 1)

	p.mu.Lock()
	offers := p.hosted[customerID][agentID]
	key := customerID + "\x00" + agentID + "\x00" + call.Name
	start := p.toolCursors[key]
	p.toolCursors[key] = start + 1
	var (
		chosen *hosting
		least  int
	)
	for offset := range offers {
		offer := offers[(start+offset)%len(offers)]
		if !offer.offers(call.Name) || len(offer.worker.toolCalls) == cap(offer.worker.toolCalls) {
			continue
		}
		// Strictly fewer, so the first host reached from the cursor keeps a tie.
		if running := offer.worker.answering(); chosen == nil || running < least {
			chosen, least = offer, running
		}
	}
	if chosen == nil {
		p.mu.Unlock()
		return nil, fmt.Errorf("dispatch: no worker is running %s now", call.Name)
	}
	chosen.worker.mu.Lock()
	if _, duplicate := chosen.worker.results[call.ID]; duplicate {
		chosen.worker.mu.Unlock()
		p.mu.Unlock()
		return nil, fmt.Errorf("dispatch: %s was already asked for", call.ID)
	}
	chosen.worker.results[call.ID] = answer
	if call.Progress != nil {
		chosen.worker.progress[call.ID] = listening{sessionID: call.SessionID, progress: call.Progress}
	}
	chosen.worker.mu.Unlock()
	// Cannot block: the queue was not full a moment ago and nothing else writes to it
	// without this lock.
	chosen.worker.toolCalls <- call
	p.mu.Unlock()

	timeout := chosen.timeoutOf(call.Name)
	deadline, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	select {
	case result := <-answer:
		if result.Failure != "" {
			return nil, errors.New(result.Failure)
		}
		return result.Output, nil
	case <-deadline.Done():
		chosen.worker.forget(call.ID)
		p.cancelHosted(chosen.worker, call)
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("dispatch: %s stopped: %w", call.Name, err)
		}
		return nil, fmt.Errorf("dispatch: %s did not answer within %s", call.Name, timeout)
	}
}

// cancelHosted tells the worker running a call to stop it. A worker that has gone has
// nothing to stop, and one whose queue is full is not reading it, so neither is waited on.
func (p *Pool) cancelHosted(worker *Worker, call ToolCall) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if !p.registered(worker) {
		return
	}
	select {
	case worker.toolCalls <- ToolCall{ID: call.ID, SessionID: call.SessionID, Name: call.Name, Cancel: true}:
	default:
	}
}

func (w *Worker) forget(id string) {
	w.mu.Lock()
	delete(w.results, id)
	delete(w.progress, id)
	w.mu.Unlock()
}

// answering is how many hosted calls this worker has been given and not yet answered.
func (w *Worker) answering() int {
	w.mu.Lock()
	defer w.mu.Unlock()
	return len(w.results)
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
	worker.progress = map[string]listening{}
	worker.mu.Unlock()
	for _, answer := range waiting {
		answer <- ToolResult{Failure: "the worker running this tool disconnected"}
	}
}
