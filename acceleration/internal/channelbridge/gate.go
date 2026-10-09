package channelbridge

import (
	"context"
	"errors"

	"github.com/GetStream/Vision-Agents/acceleration/internal/chatlog"
	"github.com/GetStream/Vision-Agents/acceleration/internal/dlc"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

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

// mayReply is whether a reply may go to the person a thread's replies reach: not when they
// opted out of the connector's opt-out channel (keywords.go), nor when the gate refuses it,
// such as past a sandbox's daily limit.
func (b *Bridge) mayReply(ctx context.Context, thread store.ChannelThread) (bool, error) {
	source := episodeSources[thread.ConnectorID]
	if source.optOuts == "" {
		return true, nil
	}
	to, err := b.replyTo(ctx, thread)
	if err != nil {
		return false, err
	}
	optedOut, err := b.store.OptedOut(ctx, thread.CustomerID, to, source.optOuts)
	if err != nil || optedOut {
		return false, err
	}
	return b.allowed(ctx, thread.CustomerID, thread.ConnectorID, to)
}

// replyTo is the person a thread's replies reach, as its opt-outs and the gate name them. On
// SMS and WhatsApp it is the thread key: an SMS thread is the person's number (telnyx.yaml's
// one thread key part), a WhatsApp thread its digits (whatsapp.yaml's), which Read writes as
// they are, since a number has no % or : to encode. A Linq thread is a chat, whose id names
// nobody, so it is the person who started it: the contact map row of the thread's episode
// (store.ThreadContact), with the one agent that binds the thread's connection (take). A chat
// with no such row, such as one an email handle started, is "", which no opt-out names and
// no sandbox lists.
func (b *Bridge) replyTo(ctx context.Context, thread store.ChannelThread) (string, error) {
	source := episodeSources[thread.ConnectorID]
	if !source.chats {
		return source.recipientOf(thread.ThreadKey), nil
	}
	configs, err := b.store.AgentConfigsBindingConnection(ctx, thread.CustomerID, thread.ConnectionID)
	if err != nil || len(configs) != 1 {
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
