package node

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"math/rand/v2"
	"slices"
	"sync"
	"time"

	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/dispatch"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/relay"
)

// lookupTimeout bounds reading what the other nodes' workers host, which a session does
// once as it opens.
const lookupTimeout = 2 * time.Second

// replyMargin is how much longer than the worker's own timeout a call carried to another
// node is waited on, so the node running it reports the timeout rather than this one
// guessing at it.
const replyMargin = 5 * time.Second

// assignTimeout bounds handing a message to a worker on another node, which is a send on
// a queue there and nothing slower.
const assignTimeout = 5 * time.Second

// Kinds of request one node carries to another's workers.
const (
	toolKind    = "tool"
	messageKind = "message"
	// cancelKind stops the request of the same id, which the asking node gave up on.
	cancelKind = "cancel"
)

// Workers are the dispatch workers of the whole deployment, as a session on any node sees
// them.
//
// A worker waits on the node its socket landed on, and a session needing it may be running
// on any other. So each node says in the Directory which customers it has workers for and
// what they host, and work for a worker elsewhere is carried to its node over Redis, the
// same way the relay carries a session's events: no node is asked to trust a peer it could
// not already reach through Redis.
type Workers struct {
	pool      *dispatch.Pool
	directory *Directory
	bus       *relay.Bus
	requests  string
	replies   string
	logger    *slog.Logger

	mu      sync.Mutex
	waiting map[string]chan workReply
	// running is the work carried here that is still being done, by request id, so a
	// peer giving up on it can stop it.
	running map[string]context.CancelFunc
}

// toolOffer is what one node's workers host for an agent.
type toolOffer struct {
	Tools   []dispatch.Tool `json:"tools"`
	Timeout time.Duration   `json:"timeout"`
}

// workRequest is work carried to the node whose worker takes it: a hosted call, or a
// message for a running session.
type workRequest struct {
	ID         string             `json:"id"`
	Kind       string             `json:"kind"`
	ReplyTo    string             `json:"reply_to"`
	CustomerID string             `json:"customer_id"`
	AgentID    string             `json:"agent_id,omitempty"`
	Call       *dispatch.ToolCall `json:"call,omitempty"`
	Message    *dispatch.Message  `json:"message,omitempty"`
}

// workReply is what came of it: a tool's output, or the worker a message went to.
type workReply struct {
	ID string `json:"id"`
	// Parts is the tool's output, pictures included. Output is its text alone, which a node
	// from before parts reads, and is what one sends.
	Parts    []llm.ContentPart `json:"parts,omitempty"`
	Output   string            `json:"output,omitempty"`
	WorkerID string `json:"worker_id,omitempty"`
	Failure  string `json:"failure,omitempty"`
}

// NewWorkers returns Workers over this node's pool, and starts taking the work its peers
// carry here until ctx is done.
func NewWorkers(ctx context.Context, pool *dispatch.Pool, directory *Directory, bus *relay.Bus, logger *slog.Logger) (*Workers, error) {
	if pool == nil || directory == nil || bus == nil {
		return nil, errors.New("node: the deployment's workers need the pool, the directory and the relay")
	}
	if logger == nil {
		logger = slog.Default()
	}
	workers := &Workers{
		pool:      pool,
		directory: directory,
		bus:       bus,
		requests:  directory.requestsFor(directory.address),
		replies:   directory.hosted + "replies:" + directory.address,
		logger:    logger,
		waiting:   map[string]chan workReply{},
		running:   map[string]context.CancelFunc{},
	}
	if err := bus.Subscribe(ctx, workers.requests, func(payload []byte) { go workers.take(ctx, payload) }); err != nil {
		return nil, err
	}
	if err := bus.Subscribe(ctx, workers.replies, workers.resolve); err != nil {
		return nil, err
	}
	return workers, nil
}

// Offer says what the workers here host for an agent now, which a worker hosting tools or
// going away changes.
func (w *Workers) Offer(ctx context.Context, customerID, agentID string) {
	tools, timeout := w.pool.HostedTools(customerID, agentID)
	if len(tools) == 0 {
		w.directory.Offer(ctx, customerID, agentID, nil)
		return
	}
	offer, err := json.Marshal(toolOffer{Tools: tools, Timeout: timeout})
	if err != nil {
		w.logger.Error("could not say what this node's workers host", "agent", agentID, "error", err)
		return
	}
	w.directory.Offer(ctx, customerID, agentID, offer)
}

// HostedTools is what the workers anywhere in the deployment run for sessions under an
// agent, each name once, and how long the slowest of them is given.
func (w *Workers) HostedTools(customerID, agentID string) ([]dispatch.Tool, time.Duration) {
	offered, longest := w.pool.HostedTools(customerID, agentID)
	seen := map[string]bool{}
	for _, tool := range offered {
		seen[tool.Name] = true
	}
	for _, offer := range w.elsewhere(customerID, agentID) {
		longest = max(longest, offer.Timeout)
		for _, tool := range offer.Tools {
			if !seen[tool.Name] {
				seen[tool.Name] = true
				offered = append(offered, tool)
			}
		}
	}
	return offered, longest
}

// RunHosted runs one call on a worker hosting the tool, here if one is, and otherwise on a
// node whose worker does.
func (w *Workers) RunHosted(ctx context.Context, customerID, agentID string, call dispatch.ToolCall) ([]llm.ContentPart, error) {
	local, _ := w.pool.HostedTools(customerID, agentID)
	if slices.ContainsFunc(local, func(tool dispatch.Tool) bool { return tool.Name == call.Name }) {
		return w.pool.RunHosted(ctx, customerID, agentID, call)
	}

	var hosts []string
	var timeout time.Duration
	for address, offer := range w.elsewhere(customerID, agentID) {
		if slices.ContainsFunc(offer.Tools, func(tool dispatch.Tool) bool { return tool.Name == call.Name }) {
			hosts = append(hosts, address)
			timeout = max(timeout, offer.Timeout)
		}
	}
	if len(hosts) == 0 {
		return nil, fmt.Errorf("dispatch: no worker is running %s now", call.Name)
	}

	reply, err := w.ask(ctx, hosts[rand.IntN(len(hosts))], workRequest{
		Kind: toolKind, CustomerID: customerID, AgentID: agentID, Call: &call,
	}, timeout+replyMargin)
	if err != nil {
		return nil, fmt.Errorf("dispatch: %s: %w", call.Name, err)
	}
	if len(reply.Parts) == 0 {
		return llm.TextParts(reply.Output), nil
	}
	return reply.Parts, nil
}

// AssignMessage gives a message to one of the customer's workers, here if any is connected
// and otherwise on a node where one is, and returns the worker it went to.
func (w *Workers) AssignMessage(ctx context.Context, customerID string, message dispatch.Message) (string, error) {
	if len(w.pool.Workers(customerID)) > 0 {
		worker, err := w.pool.AssignMessage(customerID, message)
		if err != nil {
			return "", err
		}
		return worker.ID, nil
	}

	addresses, err := w.directory.Waiting(ctx, customerID)
	if err != nil {
		return "", err
	}
	if len(addresses) == 0 {
		return "", dispatch.ErrNoWorkers
	}
	reply, err := w.ask(ctx, addresses[rand.IntN(len(addresses))], workRequest{
		Kind: messageKind, CustomerID: customerID, Message: &message,
	}, assignTimeout)
	if err != nil {
		return "", err
	}
	return reply.WorkerID, nil
}

// ask carries a request to another node and waits for what came of it.
func (w *Workers) ask(ctx context.Context, address string, request workRequest, timeout time.Duration) (workReply, error) {
	request.ID = uuid.NewString()
	request.ReplyTo = w.replies
	answer := make(chan workReply, 1)
	w.mu.Lock()
	w.waiting[request.ID] = answer
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		delete(w.waiting, request.ID)
		w.mu.Unlock()
	}()

	if err := w.bus.Publish(ctx, w.directory.requestsFor(address), request); err != nil {
		return workReply{}, fmt.Errorf("could not reach the node with the worker: %w", err)
	}
	deadline, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()
	select {
	case reply := <-answer:
		if reply.Failure != "" {
			return workReply{}, errors.New(reply.Failure)
		}
		return reply, nil
	case <-deadline.Done():
		if err := ctx.Err(); err != nil {
			stop := workRequest{ID: request.ID, Kind: cancelKind, CustomerID: request.CustomerID}
			if err := w.bus.Publish(context.WithoutCancel(ctx), w.directory.requestsFor(address), stop); err != nil {
				w.logger.Warn("could not tell the node with the worker to stop", "kind", request.Kind, "error", err)
			}
			return workReply{}, fmt.Errorf("stopped: %w", err)
		}
		return workReply{}, fmt.Errorf("the node with the worker did not answer within %s", timeout)
	}
}

// elsewhere is what the other nodes' workers host for an agent.
func (w *Workers) elsewhere(customerID, agentID string) map[string]toolOffer {
	ctx, cancel := context.WithTimeout(context.Background(), lookupTimeout)
	defer cancel()
	raw, err := w.directory.Offers(ctx, customerID, agentID)
	if err != nil {
		w.logger.Warn("could not read what other nodes' workers host", "agent", agentID, "error", err)
		return nil
	}
	offers := make(map[string]toolOffer, len(raw))
	for address, payload := range raw {
		var offer toolOffer
		if err := json.Unmarshal(payload, &offer); err != nil {
			w.logger.Warn("ignoring an offer of hosted tools that could not be read", "node", address, "error", err)
			continue
		}
		offers[address] = offer
	}
	return offers
}

// take does work a peer carried here on a worker connected here, and sends back what came
// of it.
func (w *Workers) take(ctx context.Context, payload []byte) {
	var request workRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		w.logger.Warn("ignoring work carried here that could not be read", "error", err)
		return
	}
	if request.Kind == cancelKind {
		w.mu.Lock()
		stop, found := w.running[request.ID]
		w.mu.Unlock()
		if found {
			stop()
		}
		return
	}

	ctx, stop := context.WithCancel(ctx)
	w.mu.Lock()
	w.running[request.ID] = stop
	w.mu.Unlock()
	defer func() {
		w.mu.Lock()
		delete(w.running, request.ID)
		w.mu.Unlock()
		stop()
	}()

	reply := workReply{ID: request.ID}
	var err error
	switch {
	case request.Kind == toolKind && request.Call != nil:
		reply.Parts, err = w.pool.RunHosted(ctx, request.CustomerID, request.AgentID, *request.Call)
		reply.Output = llm.TextOf(reply.Parts)
	case request.Kind == messageKind && request.Message != nil:
		var worker *dispatch.Worker
		if worker, err = w.pool.AssignMessage(request.CustomerID, *request.Message); err == nil {
			reply.WorkerID = worker.ID
		}
	default:
		err = fmt.Errorf("dispatch: no work of kind %q is taken here", request.Kind)
	}
	if err != nil {
		reply.Failure = err.Error()
	}
	if err := w.bus.Publish(context.WithoutCancel(ctx), request.ReplyTo, reply); err != nil {
		w.logger.Error("could not send back what work carried here came to", "kind", request.Kind, "error", err)
	}
}

// resolve hands a reply to the request waiting for it. One nobody waits on gave up already.
func (w *Workers) resolve(payload []byte) {
	var reply workReply
	if err := json.Unmarshal(payload, &reply); err != nil {
		w.logger.Warn("ignoring a reply that could not be read", "error", err)
		return
	}
	w.mu.Lock()
	answer, found := w.waiting[reply.ID]
	w.mu.Unlock()
	if !found {
		return
	}
	select {
	case answer <- reply:
	default:
	}
}
