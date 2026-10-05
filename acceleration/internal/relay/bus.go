// Package relay carries a session's events between the nodes of a deployment, so a
// socket may be held by a node other than the one running the conversation.
//
// A session lives in one process's memory. A browser reconnecting, or a load balancer
// with no reason to prefer one node, lands wherever it lands, and a watcher that lands
// elsewhere would be told the conversation does not exist. The node holding the session
// publishes its events here and the node holding the socket writes them out; what the
// watcher sends back goes the other way.
//
// Redis pub/sub reaches every node, which is what makes it the right shape for this --
// nothing has to know where a session is to find it -- and also what makes the Filter
// here necessary: a node is sent every session's events and has to drop what is not for
// it before it costs anything.
package relay

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"time"

	"github.com/google/uuid"
	"github.com/redis/rueidis"
)

// publishTimeout bounds one publish. Events are published from the goroutine draining a
// conversation, so a slow Redis should cost that goroutine a moment rather than the
// conversation.
const publishTimeout = 2 * time.Second

// resubscribeAfter is how long a broken subscription waits before trying again, which is
// long enough not to spin against a Redis that is down and short enough that a socket
// opened in the gap is only briefly unreachable.
const resubscribeAfter = time.Second

// Message types on the two channels. A command flows from the node holding the socket to
// the node holding the session, and an event the other way.
const (
	// Attach asks the node holding a session to watch it on this node's behalf.
	Attach = "attach"
	// Refresh says the socket an attach was for is still open.
	Refresh = "refresh"
	// Detach says it is not, and is sent when it closes.
	Detach = "detach"
	// Command carries what the watcher sent, for the holder to apply.
	Command = "command"

	// Attached answers an attach, and is what the socket waits for before it upgrades.
	Attached = "attached"
	// Frame is one of the session's events, already rendered.
	Frame = "frame"
	// Closed says the session ended or the proxy watcher was dropped.
	Closed = "closed"
)

// Owner is who a session belongs to, in the words the bus can carry: the session
// package's own Owner holds an auth.Kind, and a relay that imported either would be a
// transport that knows what it is carrying.
type Owner struct {
	CustomerID string `json:"customer_id"`
	UserID     string `json:"user_id"`
	Kind       string `json:"kind"`
}

// Key is what a node's filter holds. It is the session's owner rather than the watcher's
// own name, because a backend watching somebody else's conversation still has to be
// found by it.
func (o Owner) Key() string { return o.CustomerID + "\x00" + o.UserID }

// Request is one message from the node holding a socket to the node holding the session.
type Request struct {
	// Type is Attach, Refresh, Detach or Command.
	Type string `json:"type"`
	// Node is who sent it, so a node ignores its own.
	Node string `json:"node"`
	// Session is the conversation it is about, and Watcher the socket within it: one node
	// may hold several sockets onto the same session.
	Session string `json:"session"`
	Watcher string `json:"watcher"`
	// Owner is who the socket's caller is, which the holder checks the session against
	// before doing anything in their name. A node is not trusted to have checked.
	Owner Owner `json:"owner"`
	// ReplayPendingTools asks for the tool calls already in flight, which is what a tool
	// host reconnecting mid-call needs.
	ReplayPendingTools bool `json:"replay_pending_tools,omitempty"`
	// Payload is the frame the watcher sent, for a Command.
	Payload json.RawMessage `json:"payload,omitempty"`
}

// Response is one message from the node holding a session to the node holding the socket.
type Response struct {
	// Type is Attached, Frame or Closed.
	Type string `json:"type"`
	Node string `json:"node"`
	// Owner is the session's owner, which is the key the receiving node filters on before
	// it looks for the watcher.
	Owner   string `json:"owner"`
	Watcher string `json:"watcher"`
	// Payload is the rendered event, for a Frame.
	Payload json.RawMessage `json:"payload,omitempty"`
}

// Options configures a Bus.
type Options struct {
	// Redis is required, and is the same client the rest of the deployment uses.
	Redis rueidis.Client
	// Prefix names the two channels, so two deployments sharing one Redis do not read
	// each other's conversations. Pub/sub ignores the database number, so this is the
	// only thing that keeps them apart.
	Prefix string
	Logger *slog.Logger
}

// Bus publishes and subscribes to the two channels a relayed session needs.
type Bus struct {
	redis  rueidis.Client
	node   string
	events string
	// commands is a second channel rather than a second field on one message, so a node
	// subscribes only to the direction it is interested in and a deployment's traffic can
	// be told apart.
	commands string
	logger   *slog.Logger
}

// New returns a Bus with an identity of its own, which is how a node tells its own
// messages from its peers'.
func New(options Options) (*Bus, error) {
	if options.Redis == nil {
		return nil, errors.New("relay: a redis client is required")
	}
	prefix := options.Prefix
	if prefix == "" {
		prefix = "relay"
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Bus{
		redis:    options.Redis,
		node:     uuid.NewString(),
		events:   prefix + ":session-events",
		commands: prefix + ":session-commands",
		logger:   logger,
	}, nil
}

// Node is this process's identity on the bus.
func (b *Bus) Node() string { return b.node }

// Events is the channel a session's frames travel on.
func (b *Bus) Events() string { return b.events }

// Commands is the channel a watcher's requests travel on.
func (b *Bus) Commands() string { return b.commands }

// Publish sends one message, which every node subscribed to the channel receives.
func (b *Bus) Publish(ctx context.Context, channel string, message any) error {
	payload, err := json.Marshal(message)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(ctx, publishTimeout)
	defer cancel()

	return b.redis.Do(ctx, b.redis.B().Publish().Channel(channel).Message(string(payload)).Build()).Error()
}

// Subscribe starts delivering a channel's messages to deliver, and returns once the
// subscription is live so that whatever the caller publishes next is heard by its peers.
//
// It goes on delivering until the context is done, resubscribing if the connection
// breaks. Messages arrive on a goroutine of the client's, one at a time, so deliver has
// to hand anything slow to somebody else.
func (b *Bus) Subscribe(ctx context.Context, channel string, deliver func([]byte)) error {
	live := make(chan error, 1)
	go b.subscribe(ctx, channel, deliver, live)

	select {
	case err := <-live:
		return err
	case <-ctx.Done():
		return ctx.Err()
	}
}

// subscribe holds the subscription up, reporting only the first attempt: a caller waits
// to know whether the bus works at all, and a connection lost later is this loop's
// problem rather than theirs.
func (b *Bus) subscribe(ctx context.Context, channel string, deliver func([]byte), live chan<- error) {
	first := true
	for ctx.Err() == nil {
		err := b.redis.Dedicated(func(conn rueidis.DedicatedClient) error {
			broken := conn.SetPubSubHooks(rueidis.PubSubHooks{
				OnMessage: func(message rueidis.PubSubMessage) { deliver([]byte(message.Message)) },
			})
			// Do rather than Receive, because it returns once Redis has confirmed the
			// subscription rather than when it ends, which is what makes waiting for it
			// possible.
			if err := conn.Do(ctx, conn.B().Subscribe().Channel(channel).Build()).Error(); err != nil {
				return err
			}
			if first {
				live <- nil
				first = false
			}
			select {
			case err := <-broken:
				return err
			case <-ctx.Done():
				return ctx.Err()
			}
		})
		if first {
			live <- err
			first = false
		}
		if ctx.Err() != nil || errors.Is(err, rueidis.ErrClosing) {
			return
		}
		b.logger.Warn("the relay subscription broke", "channel", channel, "error", err)
		select {
		case <-time.After(resubscribeAfter):
		case <-ctx.Done():
			return
		}
	}
}
