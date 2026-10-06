// Package channelbridge moves messages between an external thread, such as a Slack thread,
// and the thread channel it is written into in Stream Chat (T57, AI-878; channels.md on
// connectors/planning, «Who moves messages: the channel bridge»).
//
// The inbound half takes the messages a verified provider event carries, links each external
// thread to a thread channel of its own, and writes the message there as the person who sent
// it, without a source, so Router's message hook wakes or starts the session as it does for
// any message written to an agent. The outbound half takes the agent's reply in a linked
// thread channel, which the message hook hands it, and sends it to the same external thread
// through the reply template of the connector's manifest, with the connection's credential.
package channelbridge

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"sync"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
)

// configField is the custom field on an agent channel naming the agent config that answers in
// it: api.ConfigField, which the message hook reads to find who answers a channel no session
// ran on. The value is the hook's; the bridge cannot import the api package, which serves it.
const configField = "agent_config_id"

// threadChannelPrefix starts every thread channel's id, so one reads as a thread channel in a
// list of agent channels. Never support-, which durable session commands own
// (conversation.SessionCommandChannel), and short enough that the id with a UUID after it is
// within Stream's «max length 64 characters» for a channel id
// (https://getstream.io/chat/docs/go-golang/creating_channels/).
const threadChannelPrefix = "thread-"

// writeTimeout bounds writing one inbound message into Stream Chat, off the request path. It
// is chatlog's writeTimeout for one write, 10 s, for three of them in a row (the author, the
// channel, the message).
const writeTimeout = 30 * time.Second

// sendTimeout bounds sending one reply to the provider: the 10 s the router gives one
// connector request (cmd/router connectorHTTPTimeout), twice, for the transport's one resend
// after a refused credential (core.Transports).
const sendTimeout = 20 * time.Second

// maxAnswerBytes caps what of a provider's answer to a reply is read, to check it was sent.
// chat.postMessage answers the posted message (https://docs.slack.dev/reference/methods/chat.postMessage),
// so 1 MiB is far over its size; the cap a 401's body is read with in core.Transports.
const maxAnswerBytes = 1 << 20

// Options configures a Bridge.
type Options struct {
	Store *store.Store
	// Stream is how the thread channels are written, in the Stream app each provider app is
	// pinned to.
	Stream *streamapp.Clients
	// Schemes are the registry's, by name: the one a connection names applies its credential
	// to a reply.
	Schemes map[string]core.Scheme
	// Transports builds each connection's outbound client, through which every reply leaves.
	Transports *core.Transports
	Logger     *slog.Logger
}

// Bridge is the channel bridge. Its writes into Stream Chat and its replies run off the
// request that brought them, so Close waits for the ones in flight.
type Bridge struct {
	store      *store.Store
	stream     *streamapp.Clients
	schemes    map[string]core.Scheme
	transports *core.Transports
	logger     *slog.Logger

	working sync.WaitGroup
	mu      sync.Mutex
	turns   map[string]*holder
}

// New validates the options and returns a Bridge.
func New(options Options) (*Bridge, error) {
	if options.Store == nil || options.Stream == nil || options.Transports == nil {
		return nil, stack.Wrap(errors.New("channelbridge: a store, Stream clients and transports are required"))
	}
	logger := options.Logger
	if logger == nil {
		logger = slog.Default()
	}
	return &Bridge{
		store:      options.Store,
		stream:     options.Stream,
		schemes:    options.Schemes,
		transports: options.Transports,
		logger:     logger,
		turns:      map[string]*holder{},
	}, nil
}

// Close waits for the writes and replies in flight.
func (b *Bridge) Close() {
	b.working.Wait()
}

// Deliver takes the messages one verified event of a provider app carried. For each it finds
// the app's connection of the provider unit and the agent that answers there, links the
// external thread to its thread channel, and claims the message by the provider's id for it,
// so a retried delivery is dropped. What it writes into Stream Chat it writes afterwards, off
// the request: a provider waits a few seconds at most (Slack: «respond ... within three
// seconds», https://docs.slack.dev/apis/events-api/).
//
// app is the provider app the event's route named, the zero record on a connector's own
// route, whose events name no customer. A message nobody can be found to answer is logged
// and dropped, since delivering it again finds nobody either. An error is the store failing,
// for the endpoint to answer 500, so the provider delivers again.
func (b *Bridge) Deliver(ctx context.Context, app store.ConnectorOAuthClient, messages []core.InboundMessage) error {
	for _, message := range messages {
		thread, config, fresh, err := b.take(ctx, app, message)
		if err != nil {
			return err
		}
		if !fresh {
			continue
		}
		b.working.Add(1)
		go func() {
			defer b.working.Done()
			b.write(thread, config, message)
		}()
	}
	return nil
}

// Reply sends text, the agent's reply in a linked thread channel, to its external thread, off
// the caller's request: the message hook that hands it over answers Stream at once. A reply
// that cannot be sent is logged.
func (b *Bridge) Reply(_ context.Context, thread store.ChannelThread, text string) {
	b.working.Add(1)
	go func() {
		defer b.working.Done()
		ctx, cancel := context.WithTimeout(context.Background(), sendTimeout)
		defer cancel()
		if err := b.send(ctx, thread, text); err != nil {
			b.logger.Error("could not send a reply to its external thread",
				"connector", thread.ConnectorID, "channel", thread.ChannelID, "error", err)
		}
	}()
}

// take finds who a message is for and claims it. fresh is false for a message nobody answers
// and for one already taken.
func (b *Bridge) take(ctx context.Context, app store.ConnectorOAuthClient, message core.InboundMessage) (store.ChannelThread, string, bool, error) {
	if app.CustomerID == "" || message.ProviderUnitID == "" {
		b.logger.Info("dropped an inbound message that names no provider app or no provider unit",
			"connector", message.ConnectorID)
		return store.ChannelThread{}, "", false, nil
	}
	connection, err := b.store.AppConnectionByAccount(ctx, app.CustomerID, message.ConnectorID, message.ProviderUnitID)
	if errors.Is(err, store.ErrNoConnectorConnection) {
		b.logger.Info("dropped an inbound message: the provider app has no connection of its provider unit",
			"connector", message.ConnectorID, "customer", app.CustomerID, "provider_unit", message.ProviderUnitID)
		return store.ChannelThread{}, "", false, nil
	}
	if err != nil {
		return store.ChannelThread{}, "", false, err
	}
	configs, err := b.store.AgentConfigsBindingConnection(ctx, app.CustomerID, connection.ID)
	if err != nil {
		return store.ChannelThread{}, "", false, err
	}
	if len(configs) != 1 {
		// Two agents answering one thread would talk over each other, and none answers a
		// thread nobody bound. Either is the customer's agent configs to fix.
		b.logger.Warn("dropped an inbound message: one agent config must bind the connection it came in on",
			"connector", message.ConnectorID, "customer", app.CustomerID, "connection", connection.ID, "configs", len(configs))
		return store.ChannelThread{}, "", false, nil
	}
	parts, err := b.threadParts(ctx, connection, message)
	if err != nil {
		return store.ChannelThread{}, "", false, err
	}
	thread := store.ChannelThread{
		ChannelID:      threadChannelPrefix + uuid.NewString(),
		CustomerID:     app.CustomerID,
		ConnectorID:    message.ConnectorID,
		ProviderUnitID: message.ProviderUnitID,
		ThreadKey:      message.ThreadKey,
		ConnectionID:   connection.ID,
		ThreadParts:    parts,
		StreamAppPK:    app.StreamAppPK,
	}
	if _, err := b.store.LinkChannelThread(ctx, &thread); err != nil {
		return store.ChannelThread{}, "", false, err
	}
	fresh, err := b.store.ClaimChannelThreadMessage(ctx, thread.ChannelID, message.ProviderMessageID)
	if err != nil {
		return store.ChannelThread{}, "", false, err
	}
	if !fresh {
		b.logger.Debug("dropped a retried inbound message", "connector", message.ConnectorID, "channel", thread.ChannelID)
	}
	return thread, configs[0].ID, fresh, nil
}

// threadParts are the named parts of a message's thread key, which its replies name. The
// verifier hands over the key alone, so they are read again from the raw body with the
// connection's manifest, by the provider's id for the message (core.ChannelMessage).
func (b *Bridge) threadParts(ctx context.Context, connection store.ConnectorConnection, message core.InboundMessage) (map[string]string, error) {
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return nil, err
	}
	if definition.Manifest.Channel == nil {
		return nil, stack.Wrap(fmt.Errorf("channelbridge: revision %d of %s has no channel block", connection.DefinitionRevision, connection.ConnectorID))
	}
	read, err := definition.Manifest.Channel.Read(message.ConnectorID, message.Raw)
	if err != nil {
		return nil, stack.Wrap(err)
	}
	for _, found := range read.Messages {
		if found.ProviderMessageID == message.ProviderMessageID && found.ThreadKey == message.ThreadKey {
			return found.ThreadParts, nil
		}
	}
	return nil, stack.Wrap(fmt.Errorf("channelbridge: revision %d of %s does not read message %s the way the event's did",
		connection.DefinitionRevision, connection.ConnectorID, message.ProviderMessageID))
}

// write writes one claimed message into its thread channel as its author, creating the
// channel the first time, with the agent config that answers it. One thread's messages are
// written one at a time, in the order they were taken as far as the lock keeps it.
func (b *Bridge) write(thread store.ChannelThread, configID string, message core.InboundMessage) {
	release := b.hold(thread.ChannelID)
	defer release()
	ctx, cancel := context.WithTimeout(context.Background(), writeTimeout)
	defer cancel()

	if err := b.writeInto(ctx, thread, configID, message); err != nil {
		b.logger.Error("could not write an inbound message into its thread channel",
			"connector", thread.ConnectorID, "channel", thread.ChannelID, "error", err)
	}
}

func (b *Bridge) writeInto(ctx context.Context, thread store.ChannelThread, configID string, message core.InboundMessage) error {
	bound, err := b.stream.ForApp(ctx, thread.CustomerID, thread.StreamAppPK)
	if err != nil {
		return err
	}
	author := authorUserID(thread.CustomerID, thread.ConnectorID, thread.ProviderUnitID, message.AuthorID)
	name := message.AuthorID
	if err := conversation.CreateMissingUsers(ctx, bound.Client, map[string]getstream.UserRequest{
		author: {ID: author, Name: &name},
	}); err != nil {
		return stack.Wrap(err)
	}
	// The data creates the channel on the thread's first message; the agent channel type
	// refuses a server-side create without a creator.
	_, err = bound.Client.Chat().GetOrCreateChannel(ctx, chatlog.ChannelType, thread.ChannelID, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{
			CreatedByID: &author,
			Custom: map[string]any{
				configField:                configID,
				conversation.CustomerField: thread.CustomerID,
			},
		},
	})
	if err != nil {
		return stack.Wrap(err)
	}
	// No source: the message hook answers only a message without one (api.addressed).
	_, err = bound.Client.Chat().SendMessage(ctx, chatlog.ChannelType, thread.ChannelID, &getstream.SendMessageRequest{
		Message: getstream.MessageRequest{Text: &message.Text, UserID: &author},
	})
	return stack.Wrap(err)
}

// send posts one reply through the connection's manifest reply template and transport.
func (b *Bridge) send(ctx context.Context, thread store.ChannelThread, text string) error {
	connection, err := b.store.ConnectorConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil {
		return err
	}
	scheme, found := b.schemes[connection.AuthScheme]
	if !found {
		return stack.Wrap(fmt.Errorf("%w: %q", store.ErrUnregisteredScheme, connection.AuthScheme))
	}
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	resolved, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		return stack.Wrap(err)
	}
	target, body, err := resolved.Reply(core.ReplyValues{
		Text:           text,
		ProviderUnitID: thread.ProviderUnitID,
		ThreadParts:    thread.ThreadParts,
		SinceInbound:   time.Since(thread.LastInboundAt),
	})
	if err != nil {
		return err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, target, bytes.NewReader(body))
	if err != nil {
		return stack.Wrap(err)
	}
	request.Header.Set("Content-Type", "application/json")
	ref := core.ConnectionRef{CustomerID: connection.CustomerID, ConnectionID: connection.ID}
	response, err := b.transports.Client(ref, scheme).Do(request)
	if err != nil {
		return stack.Wrap(err)
	}
	defer response.Body.Close() //nolint:errcheck // the answer has been read
	answer, err := io.ReadAll(io.LimitReader(response.Body, maxAnswerBytes))
	if err != nil {
		return stack.Wrap(err)
	}
	if response.StatusCode < 200 || response.StatusCode > 299 {
		return stack.Wrap(fmt.Errorf("channelbridge: %s answered a reply with %d", connection.ConnectorID, response.StatusCode))
	}
	sent, err := resolved.Channel.Reply.Accepts(answer)
	if err != nil {
		return stack.Wrap(fmt.Errorf("channelbridge: %s answered a reply with a body it does not read: %w", connection.ConnectorID, err))
	}
	if !sent {
		return stack.Wrap(fmt.Errorf("channelbridge: %s refused a reply in a %d answer", connection.ConnectorID, response.StatusCode))
	}
	return nil
}

// authorUserID is the Stream Chat user an external author writes as in a thread channel: one
// for each customer, connector, provider unit and author id. It is a digest so that any
// provider's ids fit the characters every Stream id here keeps to ([A-Za-z0-9_-]), and so
// that one person in two customers' workspaces is two users, never one that two customers
// write as. Who the person is across channels is the contact map's (T43), not this.
func authorUserID(customerID, connectorID, providerUnitID, authorID string) string {
	sum := sha256.Sum256([]byte(customerID + "\x00" + connectorID + "\x00" + providerUnitID + "\x00" + authorID))
	return connectorID + "-" + hex.EncodeToString(sum[:16])
}

// hold takes one thread channel's turn, so its messages are written one at a time.
func (b *Bridge) hold(key string) func() {
	b.mu.Lock()
	held, waiting := b.turns[key]
	if !waiting {
		held = &holder{}
		b.turns[key] = held
	}
	held.waiting++
	b.mu.Unlock()

	held.lock.Lock()
	return func() {
		held.lock.Unlock()
		b.mu.Lock()
		held.waiting--
		if held.waiting == 0 {
			delete(b.turns, key)
		}
		b.mu.Unlock()
	}
}

// holder is one thread channel's turn, counted so the last one out forgets it.
type holder struct {
	lock    sync.Mutex
	waiting int
}
