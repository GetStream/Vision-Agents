package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net/url"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// DispatchPath is where a worker waits for work on the router.
const DispatchPath = "/v1/dispatch"

const (
	// defaultCapacity is how much work a worker that did not say is assumed to take at
	// once. It matches the router's own assumption, so a worker and the router that is
	// choosing one agree about what it can hold.
	defaultCapacity = 4
	// defaultReportEvery is how often a worker says how it is doing.
	defaultReportEvery = 15 * time.Second
	// pongWait is how long a round trip is given before the last measurement is kept
	// instead. A figure that is out of date is more use than a zero.
	pongWait = 5 * time.Second
	// firstRetry and lastRetry bound the wait between attempts to reach a router that
	// dropped this worker, doubling from one to the other. A router being redeployed is
	// back within seconds; one that is down for longer is not worth asking twice a second.
	firstRetry = time.Second
	lastRetry  = 30 * time.Second
	// steadyAfter is how long a connection has to have lasted for its loss to be a fresh
	// drop rather than another failed attempt, so the wait starts again from firstRetry.
	steadyAfter = time.Minute
)

// errRefused is the router declining the tools this worker hosts, which reconnecting
// would only be told again.
var errRefused = errors.New("stream: the router refused to host tools")

// InboundCall is a call the router could not answer itself.
//
// It arrived over SIP at a number the customer holds, and the agent that should answer it
// runs in this process, which the router cannot reach. So it is pushed down a connection
// this process opened.
type InboundCall struct {
	// CallID and CallType name the Stream call the caller is already in. An agent that
	// joins anything else hears silence.
	CallID   string
	CallType string
	// CalledNumber is the number that was rung, which is how a worker serving several
	// numbers knows which line this is.
	CalledNumber string
	// CallerNumber is who is calling.
	CallerNumber string
	// Custom is whatever was put on the Stream call.
	Custom map[string]string
	// At is when the call started, so a call just handed over can be told from one that
	// waited in a queue.
	At time.Time
}

// InboundMessage is something written to an agent that no session is running for, or to a
// running session whose agent leaves text to dispatch.
//
// Otherwise a message written to an agent that *is* running never arrives here. The router
// answers that one from the session itself, because that agent is the one that knows what
// has been said so far.
type InboundMessage struct {
	// ChannelType and ChannelID name where it was written. Answering anywhere else would
	// be a reply nobody asked for in a conversation nobody is reading.
	ChannelType string
	ChannelID   string
	// AgentID is the agent the channel belongs to, and what a session started to answer
	// this has to be given so its replies land back here. See SessionOptions.AgentID.
	AgentID string
	// ConfigID names the stored agent config the last conversation here ran under, so a
	// worker serving several agents knows which one is being written to.
	ConfigID string
	// SessionID is the running session the message was written to, set when its agent
	// leaves text to dispatch. Nothing has answered it: create a response on this session
	// to have the model do so.
	SessionID string
	// CommandID is the durable command the message was sent as. Pass it, with Text, when
	// creating that response so the reply lands on the command it answers.
	CommandID string
	// Custom is whatever the channel was created with, carried through unread the way a
	// call's is. It is where a worker finds what the conversation is for and the router
	// has no opinion about: the organization to scope memory to, the locale to answer in,
	// whoever opened it.
	//
	// Whoever created the channel decided what is in here, so read it as a claim rather
	// than a fact. Which agent answers is not taken from it; that is ConfigID.
	Custom map[string]string
	// Text is what was written.
	Text string
	// MessageID is the message in the channel, for replying in its thread.
	MessageID string
	// UserID and UserName are who wrote it.
	UserID   string
	UserName string
	// At is when it was written.
	At time.Time
}

// CallHandler answers one arriving call. What it returns is told to the router, because a
// call nobody answered should show up there rather than only in this process's log.
type CallHandler func(context.Context, InboundCall) error

// MessageHandler answers one arriving message.
type MessageHandler func(context.Context, InboundMessage) error

// DispatchOptions is how a worker waits.
type DispatchOptions struct {
	// Backend is which router to wait on and who is billed. Its zero value reads the
	// environment.
	Backend Backend
	// Capacity is how much work to hold at once. The router passes over a worker that is
	// full rather than queueing behind it, so this is a promise about what this process
	// can actually answer. Zero takes the router's own assumption.
	Capacity int
	// ReportEvery is how often to tell the router how this process is doing. Zero is
	// every fifteen seconds.
	ReportEvery time.Duration
	// Logger is where the worker reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Dispatch waits for the calls and messages a router cannot answer itself.
//
// Neither arrives here first: somebody rang a number, or wrote in a channel, and the router
// found out by webhook. The agent, though, runs in this process, along with the functions
// the model calls. So this connects out and waits, and work is pushed down the connection.
// Nothing here has to be publicly reachable.
//
// Several workers can wait at once, in which case the router shares the work between them.
//
// Most callers want agents.Dispatch, which is this with the agent per conversation kept for
// them. This is the socket on its own, for a worker that holds something other than an
// agent.
type Dispatch struct {
	backend     Backend
	capacity    int
	reportEvery time.Duration
	logger      *slog.Logger

	call    CallHandler
	message MessageHandler
	hosted  []hosting

	mu        sync.Mutex
	socket    *Socket
	workerID  string
	latencyMS float64

	// running is the work being handled, waited for on the way out so that stopping does
	// not hang up on whoever is talking.
	running sync.WaitGroup
	active  atomic.Int64
	// handling is the calls and messages alone, which is what the router counts against
	// this worker's capacity. A hosted tool call is not one of them: the router tracks
	// those by the answer it is waiting for.
	handling atomic.Int64

	pong chan struct{}

	firstRetry time.Duration
}

// NewDispatch describes a worker without connecting it.
func NewDispatch(options DispatchOptions) (*Dispatch, error) {
	if options.Capacity < 0 {
		return nil, errors.New("stream: a worker cannot hold a negative amount of work")
	}
	if options.Capacity == 0 {
		options.Capacity = defaultCapacity
	}
	if options.ReportEvery == 0 {
		options.ReportEvery = defaultReportEvery
	}
	if options.Logger == nil {
		options.Logger = slog.Default()
	}
	return &Dispatch{
		backend:     options.Backend,
		capacity:    options.Capacity,
		reportEvery: options.ReportEvery,
		logger:      options.Logger,
		pong:        make(chan struct{}, 1),
		firstRetry:  firstRetry,
	}, nil
}

// OnCall registers what to do with an arriving call. The handler runs as its own goroutine,
// so one long call does not stop the next from being answered.
func (d *Dispatch) OnCall(handler CallHandler) { d.call = handler }

// OnMessage registers what to do with a message written to an agent that is not running.
func (d *Dispatch) OnMessage(handler MessageHandler) { d.message = handler }

// hosting is one set of functions this worker runs for every session under an agent id.
type hosting struct {
	agentID     string
	functions   *tools.Registry
	toolTimeout time.Duration
}

// HostedAgent is an agent whose tools a worker can host: a client.Agent or an agents.Agent.
//
// An interface rather than either, because both of those are built on this package and it
// cannot name them.
type HostedAgent interface {
	Name() string
	Tools() *tools.Registry
}

// Host runs an agent's tools for every session opened under it, whoever opened it.
//
// A session's own functions run in the process that opened it, which is no use to a
// conversation opened from a browser that wants to read a source tree. Hosting is the other
// direction: the router offers the agent's tools to each session naming the agent and sends
// every call to one of them here. The router matches a session on its agent id or its
// agent's name, so the name is what the agent is hosted under. toolTimeout is how long the
// router waits for one tool call to be answered before telling the model it failed, not how
// long the worker runs; zero takes the router's default of two minutes. Call before Run.
func (d *Dispatch) Host(agent HostedAgent, toolTimeout time.Duration) {
	d.hosted = append(d.hosted, hosting{agentID: agent.Name(), functions: agent.Tools(), toolTimeout: toolTimeout})
}

// host tells the router what this worker runs, once it is listening.
func (d *Dispatch) host() {
	for _, offer := range d.hosted {
		declared := []Frame{}
		for _, function := range offer.functions.List() {
			declared = append(declared, Frame{"name": function.Name, "description": function.Description, "parameters": function.Parameters})
		}
		d.tell(Frame{"type": "host_tools", "agent_id": offer.agentID, "tools": declared, "timeout_ms": offer.toolTimeout.Milliseconds()})
	}
}

// runHosted answers one hosted call, on its own goroutine: an investigation takes a minute,
// and the socket it arrived on is also what delivers the next.
func (d *Dispatch) runHosted(ctx context.Context, frame Frame) {
	id, name := frame.String("id"), frame.String("name")
	var functions *tools.Registry
	for _, offer := range d.hosted {
		for _, function := range offer.functions.List() {
			if function.Name == name {
				functions = offer.functions
			}
		}
	}
	if functions == nil {
		d.tell(Frame{"type": "tool_result", "id": id, "error": "this worker does not run " + name})
		return
	}

	d.logger.Info("running a hosted tool", "tool", name, "session", frame.String("session_id"))
	d.running.Add(1)
	d.active.Add(1)
	go func() {
		defer d.running.Done()
		defer d.active.Add(-1)

		result := Frame{"type": "tool_result", "id": id}
		output, err := functions.Call(ctx, name, frame.String("arguments"))
		if err != nil {
			d.logger.Error("a hosted tool failed", "tool", name, "error", err)
			result["error"] = err.Error()
		} else {
			result["output"] = output
		}
		if !d.tell(result) {
			d.logger.Error("a hosted tool's result never reached the router", "tool", name)
		}
	}()
}

// WorkerID is what the router calls this connection, for matching a log line here against
// one there. Empty until the router has said.
func (d *Dispatch) WorkerID() string {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.workerID
}

// Active is how much work is being handled right now.
func (d *Dispatch) Active() int { return int(d.active.Load()) }

// LatencyMS is the last round trip measured to the router.
//
// Measured from this side rather than the router's, because this is the side a call's audio
// has to cross.
func (d *Dispatch) LatencyMS() float64 {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.latencyMS
}

// Run waits for work until the context is cancelled, the router closes the connection on
// purpose, or it refuses the tools this worker hosts.
//
// A connection that drops any other way -- a router being redeployed, a load balancer
// ending an idle socket -- is opened again, with a fresh token, and the router is told
// again what this worker hosts, so a worker left running stays reachable. Only the first
// connection failing is returned, since that is a worker that was never going to work:
// the wrong address or a credential the router does not accept.
//
// Work already being handled is waited for on the way out, because dropping a call would
// hang up on whoever is talking.
func (d *Dispatch) Run(ctx context.Context) error {
	if d.call == nil && d.message == nil && len(d.hosted) == 0 {
		return errors.New("stream: register a handler with OnCall or OnMessage, or Host functions, before waiting for work")
	}

	backend, err := d.backend.Resolve()
	if err != nil {
		return err
	}
	socket, err := d.connect(ctx, backend)
	if err != nil {
		return err
	}
	d.logger.Info("waiting for work", "router", backend.URL, "capacity", d.capacity)

	reporting, done := context.WithCancel(ctx)
	go d.report(reporting)
	defer func() {
		done()
		d.running.Wait()
	}()

	retry := d.firstRetry
	for {
		opened := time.Now()
		failure := d.serve(ctx, socket)
		if ctx.Err() != nil || failure == nil || errors.Is(failure, errRefused) {
			return failure
		}
		if time.Since(opened) >= steadyAfter {
			retry = d.firstRetry
		}

		d.logger.Warn("lost the router, reconnecting", "error", failure, "in", retry)
		for {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(retry):
			}
			retry = min(retry*2, lastRetry)
			socket, err = d.connect(ctx, backend)
			if err == nil {
				break
			}
			if ctx.Err() != nil {
				return ctx.Err()
			}
			d.logger.Warn("could not reach the router", "error", err, "retrying in", retry)
		}
	}
}

// connect opens one dispatch socket, with credentials minted for it: a token signed when
// the worker started would have expired by the time a long-running one reconnects.
func (d *Dispatch) connect(ctx context.Context, backend Backend) (*Socket, error) {
	credentials, err := backend.Credentials()
	if err != nil {
		return nil, err
	}
	socket := NewSocket(backend.SocketURL(DispatchPath)+"?"+d.waiting().Encode(),
		credentials, backend.HTTPClient, d.logger)
	if err := socket.Open(ctx); err != nil {
		return nil, err
	}
	return socket, nil
}

// waiting is what this worker says about itself on the way in: how much it can hold, how
// much it is already holding, and which kinds of work it answers.
//
// On the handshake rather than in a frame because the router may hand this worker something
// before it has read anything, and a worker that looks idle and takes nothing is worse than
// one that never connected.
func (d *Dispatch) waiting() url.Values {
	asked := url.Values{}
	asked.Set("capacity", strconv.Itoa(d.capacity))
	// What is still being handled from before a reconnect. The pool that handed it out has
	// gone, so without this the one taking over fills this worker up on top of it.
	asked.Set("active", strconv.FormatInt(d.handling.Load(), 10))

	kinds := []string{}
	if d.call != nil {
		kinds = append(kinds, "call")
	}
	if d.message != nil {
		kinds = append(kinds, "message")
	}
	// Always said, even when it is nothing: a worker that only hosts tools answers neither,
	// and one handed a call it has no handler for leaves a caller listening to a phone.
	asked.Set("handles", strings.Join(kinds, ","))
	return asked
}

// serve waits for work on one connection until it ends.
func (d *Dispatch) serve(ctx context.Context, socket *Socket) error {
	d.mu.Lock()
	d.socket = socket
	d.mu.Unlock()

	// A read blocks in the socket rather than on a channel, so cancellation has to reach
	// it by closing the connection underneath it.
	stopped := make(chan struct{})
	go func() {
		select {
		case <-ctx.Done():
			socket.Close()
		case <-stopped:
		}
	}()

	failure := d.read(ctx, socket)
	close(stopped)

	d.mu.Lock()
	d.socket = nil
	d.workerID = ""
	d.mu.Unlock()
	socket.Close()
	return failure
}

// read applies what the router sends until it stops.
func (d *Dispatch) read(ctx context.Context, socket *Socket) error {
	for {
		frame, _, err := socket.Read()
		if err != nil {
			if ctx.Err() != nil {
				return ctx.Err()
			}
			// Going away is a router shutting down, which is when a worker should find the
			// one replacing it rather than stop.
			if errors.Is(err, ErrSocketClosed) || websocket.IsCloseError(err, websocket.CloseNormalClosure) {
				return nil
			}
			return fmt.Errorf("stream: the dispatch socket ended: %w", err)
		}
		if frame == nil {
			continue
		}

		switch frame.Type() {
		case "call":
			d.answer(ctx, frame.String("work_id"), callOf(frame))
		case "message":
			d.reply(ctx, frame.String("work_id"), messageOf(frame))
		case "ready":
			d.mu.Lock()
			d.workerID = frame.String("worker_id")
			d.mu.Unlock()
			d.logger.Info("the router calls this worker", "worker", frame.String("worker_id"))
			d.host()
		case "tool_call":
			d.runHosted(ctx, frame)
		case "hosting":
			d.logger.Info("the router sends this worker's tools here", "agent", frame.String("agent_id"))
		case "hosting_refused":
			// Not worth waiting on: a worker whose tools were refused is one nobody will
			// call, and saying so is better than sitting connected looking healthy.
			return fmt.Errorf("%w for agent %s: %s", errRefused,
				frame.String("agent_id"), frame.String("reason"))
		case "pong":
			select {
			case d.pong <- struct{}{}:
			default:
			}
		default:
			d.logger.Debug("ignoring a dispatch frame", "type", frame.Type())
		}
	}
}

// answer starts handling one call.
//
// On its own goroutine rather than inline, because reading the socket is also what delivers
// the next call: answering one caller in line would leave the next listening to a ringing
// phone.
func (d *Dispatch) answer(ctx context.Context, workID string, call InboundCall) {
	if d.call == nil {
		d.logger.Debug("ignoring a call: no handler is registered for one", "call", call.CallID)
		d.finished(workID, errors.New("stream: this worker answers no calls"))
		return
	}

	d.logger.Info("answering a call", "from", call.CallerNumber, "on", call.CalledNumber)
	d.running.Add(1)
	d.active.Add(1)
	d.handling.Add(1)
	go func() {
		defer d.running.Done()
		defer d.active.Add(-1)
		defer d.handling.Add(-1)

		err := d.call(ctx, call)
		if err != nil {
			d.logger.Error("a call could not be answered", "call", call.CallID, "error", err)
		}
		d.finished(workID, err)
	}()
}

// reply starts handling one message, on its own goroutine for the same reason a call is.
func (d *Dispatch) reply(ctx context.Context, workID string, message InboundMessage) {
	if d.message == nil {
		d.logger.Debug("ignoring a message: no handler is registered for one", "channel", message.ChannelID)
		d.finished(workID, errors.New("stream: this worker answers no messages"))
		return
	}

	d.logger.Info("answering a message", "from", message.UserID, "channel", message.ChannelID)
	d.running.Add(1)
	d.active.Add(1)
	d.handling.Add(1)
	go func() {
		defer d.running.Done()
		defer d.active.Add(-1)
		defer d.handling.Add(-1)

		err := d.message(ctx, message)
		if err != nil {
			d.logger.Error("a message could not be answered", "channel", message.ChannelID, "error", err)
		}
		d.finished(workID, err)
	}()
}

// finished tells the router one piece of work is over, which is what gives this worker its
// room for the next back.
//
// What went wrong goes with it rather than staying here. The router is where somebody is
// looking when a caller says nobody picked up, and a traceback in this process's log is no
// use to them. It is said even for work this worker had no handler for, because the room it
// took is held until something says it is free.
func (d *Dispatch) finished(workID string, failure error) {
	done := Frame{"type": "done", "work_id": workID}
	if failure != nil {
		done["error"] = failure.Error()
	}
	d.tell(done)
}

// report tells the router how this process is doing, on a timer.
//
// None of it decides where work goes. What this worker is holding is counted at the router
// from what it has handed out and what has been reported done, which is exact and current;
// this is so an operator can see which worker is under load without logging into it.
//
// Host CPU and memory are not sent, which is why the router's backstop for a host in
// trouble never fires for a Go worker. There is no portable way to read either here, and a
// figure invented in this process would be read there as a real one.
func (d *Dispatch) report(ctx context.Context) {
	ticker := time.NewTicker(d.reportEvery)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.measure(ctx)
			d.tell(Frame{
				"type":          "load",
				"active_agents": d.Active(),
				"latency_ms":    d.LatencyMS(),
			})
		}
	}
}

// measure times a round trip to the router, keeping the last figure if none comes back.
func (d *Dispatch) measure(ctx context.Context) {
	select {
	case <-d.pong:
	default:
	}

	sent := time.Now()
	if !d.tell(Frame{"type": "ping", "at": float64(sent.UnixNano()) / float64(time.Second)}) {
		return
	}

	waited := time.NewTimer(pongWait)
	defer waited.Stop()
	select {
	case <-d.pong:
		d.mu.Lock()
		d.latencyMS = float64(time.Since(sent).Microseconds()) / 1000
		d.mu.Unlock()
	case <-waited.C:
		d.logger.Debug("the router did not answer a ping", "within", pongWait)
	case <-ctx.Done():
	}
}

// tell sends one frame, reporting whether it went.
//
// A closed socket is not an error here: every one of these is something the router would
// like to know rather than something a conversation depends on.
func (d *Dispatch) tell(frame Frame) bool {
	d.mu.Lock()
	socket := d.socket
	d.mu.Unlock()
	if socket == nil || !socket.IsOpen() {
		return false
	}
	if err := socket.Send(frame); err != nil {
		d.logger.Debug("could not reach the router", "error", err)
		return false
	}
	return true
}

// customOf narrows a frame's custom data to the strings a handler can read. The router
// sends strings, and a value that is not one is dropped rather than rendered, because a
// number that arrived as a JSON float should not reach a handler as "1.7e+01".
func customOf(frame Frame, key string) map[string]string {
	custom := map[string]string{}
	for name, value := range frame.Frame(key) {
		if text, ok := value.(string); ok {
			custom[name] = text
		}
	}
	return custom
}

// callOf reads a call frame off the wire.
func callOf(frame Frame) InboundCall {
	custom := customOf(frame, "custom")
	callType := frame.String("call_type")
	if callType == "" {
		callType = "default"
	}
	return InboundCall{
		CallID:       frame.String("call_id"),
		CallType:     callType,
		CalledNumber: frame.String("called_number"),
		CallerNumber: frame.String("caller_number"),
		Custom:       custom,
		At:           timeOf(frame.String("at")),
	}
}

// messageOf reads a message frame off the wire.
func messageOf(frame Frame) InboundMessage {
	channelType := frame.String("channel_type")
	if channelType == "" {
		channelType = "agent"
	}
	// A router that names no agent means the channel, which is what names it there too.
	agentID := frame.String("agent_id")
	if agentID == "" {
		agentID = frame.String("channel_id")
	}
	return InboundMessage{
		ChannelType: channelType,
		ChannelID:   frame.String("channel_id"),
		AgentID:     agentID,
		ConfigID:    frame.String("config_id"),
		SessionID:   frame.String("session_id"),
		CommandID:   frame.String("command_id"),
		Custom:      customOf(frame, "custom"),
		Text:        frame.String("text"),
		MessageID:   frame.String("message_id"),
		UserID:      frame.String("user_id"),
		UserName:    frame.String("user_name"),
		At:          timeOf(frame.String("at")),
	}
}

// timeOf reads an RFC 3339 timestamp, leaving the zero time for one that cannot be read:
// work is not worth refusing over when it arrived.
func timeOf(text string) time.Time {
	at, err := time.Parse(time.RFC3339, text)
	if err != nil {
		return time.Time{}
	}
	return at
}
