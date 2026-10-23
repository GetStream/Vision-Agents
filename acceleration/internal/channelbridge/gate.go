package channelbridge

import (
	"context"
	"errors"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/connectors/core"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stack"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// The reasons Reachable gives for a reply that would not go out.
var (
	// ErrOptedOut is a person who opted out of the thread's channel (keywords.go): nothing
	// reaches them until they opt back in.
	ErrOptedOut = errors.New("channelbridge: the person opted out of this channel")
	// ErrHeldBack is a reply the sandbox gate holds back, such as one past a sandboxed
	// customer's daily limit or to a number outside its recipients.
	ErrHeldBack = errors.New("channelbridge: the sandbox gate holds replies to this person back")
	// ErrOutsideWindow is a person who last wrote longer ago than the provider takes free text
	// (episodeSource.window), on a manifest with no after_window template.
	ErrOutsideWindow = errors.New("channelbridge: the person last wrote outside the reply window, and the manifest has no template for after it")
)

// Reachable says whether a reply on a thread would go out now, the words aside, so a reply
// nobody asked for, such as a scheduled task's, is refused before the turn that writes it is
// paid for. It asks what Reply asks (refusesReply), and what the provider would refuse: a
// connection that was removed (store.ErrNoConnectorConnection), and free text past the
// provider's window (ErrOutsideWindow, pastWindow).
func (b *Bridge) Reachable(ctx context.Context, thread store.ChannelThread) error {
	connection, err := b.store.ConnectorConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil {
		return err
	}
	if err := b.refusesReply(ctx, thread); err != nil {
		return err
	}
	definition, err := b.store.ConnectorDefinition(ctx, connection.CustomerID, connection.ConnectorID, connection.DefinitionRevision)
	if err != nil {
		return err
	}
	resolved, err := definition.Manifest.Resolve(connection.AuthScheme, connection.Inputs, connection.Metadata)
	if err != nil {
		return stack.Wrap(err)
	}
	if resolved.Channel != nil && pastWindow(thread.ConnectorID, resolved.Channel.Reply, time.Since(thread.LastInboundAt)) {
		return stack.Wrap(ErrOutsideWindow)
	}
	return nil
}

// pastWindow is whether free text to a person who last wrote since ago is past the
// connector's window (episodeSource.window), with no after_window template in the manifest's
// reply to send instead: a manifest sets after_window exactly when it sets window
// (checkChannel), and Reply sends that template past it.
func pastWindow(connectorID string, reply core.ReplyRule, since time.Duration) bool {
	window := episodeSources[connectorID].window
	return window > 0 && since >= window && reply.Window == 0
}

// allowed is whether dlc.Gate lets the bridge text to on a connector: the person has not
// opted out, and a sandboxed customer (no approved 10DLC use case) is within its recipients
// and its daily limit, as internal/channels asks before it hands a message to the agent and
// before each reply (T62a, AI-921). Only a texting connector, one whose episode source names
// an opt-out channel (SMS, WhatsApp, iMessage), passes the gate; Slack is no texting line and
// internal/channels has no Slack line. A refusal is false; an error is the store failing.
func (b *Bridge) allowed(ctx context.Context, customerID, connectorID, to string) (bool, error) {
	channel := episodeSources[connectorID].optOuts
	if channel == "" {
		return true, nil
	}
	err := b.gate.Allow(ctx, customerID, channel, to)
	if errors.Is(err, dlc.ErrRefused) {
		// The refusal names the number, which stays out of the logs.
		b.logger.Info("the sandbox gate held a channel message back", "connector", connectorID, "customer", customerID)
		return false, nil
	}
	return err == nil, err
}

// refusesReply is why a reply may not go to the person a thread's replies reach, or nil when
// it may: ErrOptedOut when they opted out of the connector's opt-out channel (keywords.go),
// ErrHeldBack when the gate refuses it, such as past a sandbox's daily limit.
func (b *Bridge) refusesReply(ctx context.Context, thread store.ChannelThread) error {
	source := episodeSources[thread.ConnectorID]
	if source.optOuts == "" {
		return nil
	}
	to, err := b.replyTo(ctx, thread)
	if err != nil {
		return err
	}
	optedOut, err := b.store.OptedOut(ctx, thread.CustomerID, to, source.optOuts)
	if err != nil {
		return err
	}
	if optedOut {
		return stack.Wrap(ErrOptedOut)
	}
	allowed, err := b.allowed(ctx, thread.CustomerID, thread.ConnectorID, to)
	if err != nil {
		return err
	}
	if !allowed {
		return stack.Wrap(ErrHeldBack)
	}
	return nil
}

// replyTo is the person a thread's replies reach, as its opt-outs and the gate name them. On
// SMS and WhatsApp it is the thread key: an SMS thread is the person's number (telnyx.yaml's
// one thread key part), a WhatsApp thread its digits (whatsapp.yaml's), which Read writes as
// they are, since a number has no % or : to encode. A Linq thread is a chat, whose id names
// nobody, so it is the person who started it: the contact map row of the thread's episode
// (store.ThreadContact), with the agent that answers on the thread's connection (take). A chat
// with no such row, such as one an email handle started, is "", which no opt-out names and
// no sandbox lists.
func (b *Bridge) replyTo(ctx context.Context, thread store.ChannelThread) (string, error) {
	source := episodeSources[thread.ConnectorID]
	if !source.chats {
		return source.recipientOf(thread.ThreadKey), nil
	}
	configs, err := b.store.AgentConfigsBindingConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil || len(configs) == 0 {
		return "", err
	}
	contact, err := b.store.ThreadContact(ctx, thread.CustomerID, configs[0].ID, chatlog.ChannelType+":"+thread.ChannelID)
	if errors.Is(err, store.ErrNoContact) {
		return "", nil
	}
	if err != nil {
		return "", err
	}
	return contact.Address, nil
}

// sent counts a reply the provider took against a sandboxed customer's daily limit (dlc.Gate
// Sent, which counts nothing for a customer the sandbox does not hold), on a texting
// connector, as internal/channels counts each reply.
func (b *Bridge) sent(thread store.ChannelThread) {
	if episodeSources[thread.ConnectorID].optOuts == "" {
		return
	}
	ctx, cancel := context.WithTimeout(context.Background(), sendTimeout)
	defer cancel()
	b.gate.Sent(ctx, thread.CustomerID)
}
