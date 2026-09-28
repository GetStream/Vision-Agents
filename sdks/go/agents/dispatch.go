package agents

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/GetStream/Vision-Agents/sdks/go/stream"
)

// InboundCall is a call the router could not answer itself.
type InboundCall = stream.InboundCall

// InboundMessage is something written to an agent that no session is running for.
type InboundMessage = stream.InboundMessage

// AgentFactory builds the agent for a conversation nothing is answering yet. It is given
// the message so an agent can be configured from who wrote it and which config the channel
// was last answered under.
type AgentFactory func(context.Context, InboundMessage) (*Agent, error)

// queued is how many questions may be waiting on one conversation before the next is
// refused. A person writing faster than an agent can answer is a few messages ahead of it
// at most; a hundred is somebody's script.
const queued = 8

// defaultTurnTimeout is how long one answer is given before the conversation gives up on
// it and takes the next question.
//
// The backend ends a turn with responded or error, so this is only reached when it does
// neither. Without it a conversation that never settles is one nobody can write to again,
// which is worse than an answer that was abandoned.
//
// Five minutes suits an agent whose tools answer in seconds. One whose tools read a source
// tree or wait on a sandbox needs longer, and says so with DispatchOptions.TurnTimeout.
const defaultTurnTimeout = 5 * time.Minute

// DispatchOptions is how a worker waits. It is stream.DispatchOptions with the agent kept
// per conversation.
type DispatchOptions struct {
	// Backend is which router to wait on and who is billed. Its zero value reads the
	// environment.
	Backend stream.Backend
	// Capacity is how much work to hold at once. Zero takes the router's assumption.
	Capacity int
	// ReportEvery is how often to tell the router how this process is doing.
	ReportEvery time.Duration
	// TurnTimeout is how long one answer is given before the conversation abandons it and
	// takes the next question. Zero is five minutes.
	//
	// It has to outlast the slowest tool this worker registers, and by more than one of
	// them: a turn the model spends on three searches and a source investigation is one
	// turn, not four. Set below that and a long answer is abandoned while it is still
	// being written.
	TurnTimeout time.Duration
	// OnEvent is told everything the backend says about every conversation this worker
	// holds, which is the only way to see it: the conversation reads its own session, and
	// a second reader would take events from it.
	//
	// It is where tool timings, model failures and answers go when they are wanted for
	// something other than the channel they are being written into — a metric, a trace, a
	// benchmark. It is called from the goroutine driving that conversation, so it should
	// hand work off rather than do it. Nil watches nothing.
	OnEvent func(channelID string, event stream.Event)
	// Logger is where the worker reports what it could not do. Nil uses the default.
	Logger *slog.Logger
}

// Dispatch waits for messages and calls the router cannot answer itself, and keeps the
// agent answering each conversation.
//
// The agent runs here, which is the point: the model runs in the backend, and the functions
// it calls run in this process, next to whatever they need to reach. A worker in another
// language holds its own functions the same way, and the router does not have to know the
// difference.
//
// Example:
//
//	dispatch, err := agents.NewDispatch(agents.DispatchOptions{Capacity: 8})
//	if err != nil {
//		return err
//	}
//	dispatch.OnMessage(func(ctx context.Context, message agents.InboundMessage) error {
//		conversation, err := dispatch.Conversation(ctx, message, build)
//		if err != nil {
//			return err
//		}
//		return conversation.Respond(message.Text)
//	})
//	return dispatch.Run(ctx)
type Dispatch struct {
	dispatch *stream.Dispatch
	logger   *slog.Logger

	turnTimeout time.Duration
	onEvent     func(string, stream.Event)

	mu sync.Mutex
	// held is the agent answering each channel. A channel is one conversation, so the
	// agent that answered the last message on it is the one that knows what has been said
	// and should answer the next. They are kept until this worker stops waiting, so a
	// conversation is not restarted between messages.
	held map[string]*Conversation
}

// NewDispatch describes a worker without connecting it.
func NewDispatch(options DispatchOptions) (*Dispatch, error) {
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	if options.TurnTimeout < 0 {
		return nil, errors.New("agents: an answer cannot be given less than no time to finish")
	}
	if options.TurnTimeout == 0 {
		options.TurnTimeout = defaultTurnTimeout
	}
	socket, err := stream.NewDispatch(stream.DispatchOptions{
		Backend:     options.Backend,
		Capacity:    options.Capacity,
		ReportEvery: options.ReportEvery,
		Logger:      logger,
	})
	if err != nil {
		return nil, err
	}
	return &Dispatch{
		dispatch:    socket,
		logger:      logger,
		turnTimeout: options.TurnTimeout,
		onEvent:     options.OnEvent,
		held:        map[string]*Conversation{},
	}, nil
}

// OnMessage registers what to do with a message written to an agent that is not running.
func (d *Dispatch) OnMessage(handler func(context.Context, InboundMessage) error) {
	d.dispatch.OnMessage(handler)
}

// OnCall registers what to do with an arriving call.
func (d *Dispatch) OnCall(handler func(context.Context, InboundCall) error) {
	d.dispatch.OnCall(handler)
}

// Host runs these functions for every session opened under an agent id, whoever opened it,
// giving the router timeout for each call. See stream.Dispatch.Host.
func (d *Dispatch) Host(agentID string, functions Registrar, timeout time.Duration) {
	d.dispatch.Host(agentID, functions.Functions(), timeout)
}

// WorkerID is what the router calls this connection.
func (d *Dispatch) WorkerID() string { return d.dispatch.WorkerID() }

// Active is how much work is being handled right now.
func (d *Dispatch) Active() int { return d.dispatch.Active() }

// Conversation is the one answering this message's channel, started if none is.
//
// A channel is one conversation. The second message on it goes to the agent that answered
// the first, which is still open and knows what has been said; only a channel nothing is
// answering calls the factory. The session is opened on the message's own channel, so what
// it writes lands in the conversation the question was asked in.
func (d *Dispatch) Conversation(ctx context.Context, message InboundMessage, build AgentFactory) (*Conversation, error) {
	if message.ChannelID == "" {
		return nil, errors.New("agents: a message with no channel is one there is nowhere to answer")
	}

	d.mu.Lock()
	defer d.mu.Unlock()

	if answering, held := d.held[message.ChannelID]; held {
		if !answering.Ended() {
			return answering, nil
		}
		// The session behind it ended. Nothing is closed here, because whatever ended it
		// has already done that; what is left is the entry pointing at it.
		delete(d.held, message.ChannelID)
	}

	agent, err := build(ctx, message)
	if err != nil {
		return nil, err
	}
	session, err := agent.Chat(ctx, ChatOptions{Persist: true, AgentID: message.AgentID})
	if err != nil {
		return nil, err
	}

	started := &Conversation{
		agent:   agent,
		session: session,
		logger:  d.logger.With("channel", message.ChannelID),
		timeout: d.turnTimeout,
		turns:   make(chan string, queued),
		settled: make(chan struct{}),
	}
	if d.onEvent != nil {
		channel := message.ChannelID
		started.watch = func(event stream.Event) { d.onEvent(channel, event) }
	}
	go started.run(context.WithoutCancel(ctx))

	d.held[message.ChannelID] = started
	d.logger.Info("started an agent", "channel", message.ChannelID, "session", session.ID())
	return started, nil
}

// Run waits for work until the context is cancelled or the router closes the connection
// on purpose, reconnecting when it drops, then closes the conversations this worker was
// holding. See stream.Dispatch.Run.
func (d *Dispatch) Run(ctx context.Context) error {
	failure := d.dispatch.Run(ctx)

	d.mu.Lock()
	held := make([]*Conversation, 0, len(d.held))
	for _, conversation := range d.held {
		held = append(held, conversation)
	}
	clear(d.held)
	d.mu.Unlock()

	// Closing is given a context of its own: the one that stopped this worker is usually
	// already cancelled, and a session left open is a conversation the router still thinks
	// is being answered here.
	closing, done := context.WithTimeout(context.WithoutCancel(ctx), 10*time.Second)
	defer done()
	for _, conversation := range held {
		if err := conversation.Close(closing); err != nil {
			d.logger.Debug("a conversation did not close cleanly", "error", err)
		}
	}
	return failure
}

// Conversation is one channel's agent, and the questions waiting to be put to it.
type Conversation struct {
	agent   *Agent
	session *Session
	logger  *slog.Logger
	timeout time.Duration
	watch   func(stream.Event)

	turns   chan string
	settled chan struct{}
	once    sync.Once
}

// Agent is the agent answering here.
func (c *Conversation) Agent() *Agent { return c.agent }

// Session is the conversation the backend is holding.
func (c *Conversation) Session() *Session { return c.session }

// Ended reports whether this conversation has stopped answering.
func (c *Conversation) Ended() bool {
	select {
	case <-c.settled:
		return true
	default:
		return false
	}
}

// Respond puts a question to the agent and returns without waiting for the answer.
//
// There is nothing to wait for: the answer is written into the channel by the backend as it
// is generated, so the person who wrote is already reading it.
//
// Questions are answered one at a time. Session.Respond interrupts whatever is being said,
// which is right on a call and wrong here: two messages written in quick succession would
// throw the first answer away half-written. So the second waits for the first to settle
// rather than cutting it off.
func (c *Conversation) Respond(text string) error {
	if c.Ended() {
		return errors.New("agents: this conversation has ended")
	}
	select {
	case c.turns <- text:
		return nil
	default:
		return fmt.Errorf("agents: %d questions are already waiting to be answered here", queued)
	}
}

// Close ends the conversation.
func (c *Conversation) Close(ctx context.Context) error {
	c.end()
	return c.session.Close(ctx)
}

func (c *Conversation) end() { c.once.Do(func() { close(c.settled) }) }

// run puts the waiting questions to the agent, one at a time, and reads what the backend
// says about them.
//
// Reading is not optional. The session's events are a channel the pipeline's own socket
// reader writes into, and that reader is also what delivers the model's tool calls, so a
// conversation nobody reads stops running this process's functions once the buffer fills.
func (c *Conversation) run(ctx context.Context) {
	defer c.end()

	events := c.session.Events()
	for {
		select {
		case <-ctx.Done():
			return

		case text := <-c.turns:
			if err := c.session.Respond(text); err != nil {
				c.logger.Error("a question never reached the model", "error", err)
				return
			}
			if !c.await(ctx, events) {
				return
			}

		case event, open := <-events:
			if !open {
				return
			}
			c.seen(event)
		}
	}
}

// seen offers one event to whoever is watching this worker's conversations.
//
// Inline rather than on a goroutine of its own, so that what a watcher is told arrives in
// the order the backend said it. A watcher that cannot keep up is one holding up an
// answer, which is its own problem to solve and not one to hide by reordering.
func (c *Conversation) seen(event stream.Event) {
	if c.watch != nil {
		c.watch(event)
	}
}

// await reads until the turn is finished, reporting whether the conversation can take
// another.
func (c *Conversation) await(ctx context.Context, events <-chan stream.Event) bool {
	timeout := c.timeout
	if timeout <= 0 {
		timeout = defaultTurnTimeout
	}
	abandon := time.NewTimer(timeout)
	defer abandon.Stop()

	for {
		select {
		case <-ctx.Done():
			return false

		case <-abandon.C:
			c.logger.Error("an answer never finished", "after", timeout)
			return true

		case event, open := <-events:
			if !open {
				return false
			}
			c.seen(event)
			switch event.Kind {
			case "responded":
				// Work the model left running means the turn is not over: more of the
				// answer follows once it settles.
				if !event.PendingWork {
					return true
				}
			case "error":
				c.logger.Error("the conversation reported a failure", "error", event.Error)
				return true
			}
		}
	}
}
