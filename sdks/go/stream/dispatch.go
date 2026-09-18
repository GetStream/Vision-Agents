package stream

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gorilla/websocket"
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
)

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

// InboundMessage is something written to an agent that no session is running for.
//
// A message written to an agent that *is* running never arrives here. The router answers
// that one from the session itself, because that agent is the one that knows what has been
// said so far.
type InboundMessage struct {
	// ChannelType and ChannelID name where it was written. Answering anywhere else would
	// be a reply nobody asked for in a conversation nobody is reading.
	ChannelType string
	ChannelID   string
	// AgentID is the agent the channel belongs to, and what a session started to answer
	// this has to be given so its replies land back here. See ChatOptions.AgentID.
	AgentID string
	// ConfigID names the stored agent config the last conversation here ran under, so a
	// worker serving several agents knows which one is being written to.
	ConfigID string
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

	mu        sync.Mutex
	socket    *Socket
	workerID  string
	latencyMS float64

	// running is the work being handled, waited for on the way out so that stopping does
	// not hang up on whoever is talking.
	running sync.WaitGroup
	active  atomic.Int64

	pong chan struct{}
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
	}, nil
}

// OnCall registers what to do with an arriving call. The handler runs as its own goroutine,
// so one long call does not stop the next from being answered.
func (d *Dispatch) OnCall(handler CallHandler) { d.call = handler }

// OnMessage registers what to do with a message written to an agent that is not running.
func (d *Dispatch) OnMessage(handler MessageHandler) { d.message = handler }

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

// Run waits for work until the context is cancelled or the router closes the connection.
//
// Work already being handled is waited for on the way out, because dropping a call would
// hang up on whoever is talking.
func (d *Dispatch) Run(ctx context.Context) error {
	if d.call == nil && d.message == nil {
		return errors.New("stream: register a handler with OnCall or OnMessage before waiting for work")
	}

	backend, err := d.backend.Resolve()
	if err != nil {
		return err
	}
	credentials, err := backend.Credentials()
	if err != nil {
		return err
	}

	address := backend.SocketURL(DispatchPath) + "?capacity=" + strconv.Itoa(d.capacity)
	socket := NewSocket(address, credentials, backend.HTTPClient, d.logger)
	if err := socket.Open(ctx); err != nil {
		return err
	}
	d.mu.Lock()
	d.socket = socket
	d.mu.Unlock()
	d.logger.Info("waiting for work", "router", backend.URL, "capacity", d.capacity)

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

	reporting, done := context.WithCancel(ctx)
	go d.report(reporting)

	failure := d.read(ctx, socket)

	close(stopped)
	done()
	d.running.Wait()

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
			if errors.Is(err, ErrSocketClosed) ||
				websocket.IsCloseError(err, websocket.CloseNormalClosure, websocket.CloseGoingAway) {
				return nil
			}
			return fmt.Errorf("stream: the dispatch socket ended: %w", err)
		}
		if frame == nil {
			continue
		}

		switch frame.Type() {
		case "call":
			d.answer(ctx, callOf(frame))
		case "message":
			d.reply(ctx, messageOf(frame))
		case "ready":
			d.mu.Lock()
			d.workerID = frame.String("worker_id")
			d.mu.Unlock()
			d.logger.Info("the router calls this worker", "worker", frame.String("worker_id"))
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
func (d *Dispatch) answer(ctx context.Context, call InboundCall) {
	if d.call == nil {
		d.logger.Debug("ignoring a call: no handler is registered for one", "call", call.CallID)
		return
	}

	d.logger.Info("answering a call", "from", call.CallerNumber, "on", call.CalledNumber)
	d.running.Add(1)
	d.active.Add(1)
	go func() {
		defer d.running.Done()
		defer d.active.Add(-1)

		if err := d.call(ctx, call); err != nil {
			d.logger.Error("a call could not be answered", "call", call.CallID, "error", err)
			d.tell(Frame{"type": "rejected", "call_id": call.CallID, "reason": err.Error()})
			return
		}
		d.tell(Frame{"type": "accepted", "call_id": call.CallID})
	}()
}

// reply starts handling one message, on its own goroutine for the same reason a call is.
//
// Nothing is reported back to the router. Accepting and rejecting are about a caller waiting
// on a line, and there is no line here: a message nobody answered is a log line, not a
// silence somebody is sitting in.
func (d *Dispatch) reply(ctx context.Context, message InboundMessage) {
	if d.message == nil {
		d.logger.Debug("ignoring a message: no handler is registered for one", "channel", message.ChannelID)
		return
	}

	d.logger.Info("answering a message", "from", message.UserID, "channel", message.ChannelID)
	d.running.Add(1)
	d.active.Add(1)
	go func() {
		defer d.running.Done()
		defer d.active.Add(-1)

		if err := d.message(ctx, message); err != nil {
			d.logger.Error("a message could not be answered", "channel", message.ChannelID, "error", err)
		}
	}()
}

// report tells the router how this process is doing, on a timer.
//
// The router does not use any of it to choose a worker yet. It is sent so that a policy
// which does has numbers to read, and so an operator can see which worker is under load
// without logging into it.
//
// Host CPU and memory are not sent. Go has no portable way to read either, and a figure
// invented here would be read there as a real one.
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
