package store

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

// queueable is an inbound message with every field queueing it requires.
func queueable() ChannelInbound {
	return ChannelInbound{ChannelID: "thread-one", ProviderMessageID: "1759740000.000100", CustomerID: "acme-app", ConnectorID: "acme", ConfigID: "config-one"}
}

func TestQueueingAnInboundMessageWithoutAChannelIsRefused(t *testing.T) {
	message := queueable()
	message.ChannelID = ""
	_, err := (&Store{}).QueueChannelInbound(context.Background(), message)
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAnInboundMessageWithoutAProviderMessageIDIsRefused(t *testing.T) {
	message := queueable()
	message.ProviderMessageID = ""
	_, err := (&Store{}).QueueChannelInbound(context.Background(), message)
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAnInboundMessageWithoutACustomerIsRefused(t *testing.T) {
	message := queueable()
	message.CustomerID = ""
	_, err := (&Store{}).QueueChannelInbound(context.Background(), message)
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAnInboundMessageWithoutAConnectorIsRefused(t *testing.T) {
	message := queueable()
	message.ConnectorID = ""
	_, err := (&Store{}).QueueChannelInbound(context.Background(), message)
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAnInboundMessageWithoutAnAgentConfigIsRefused(t *testing.T) {
	message := queueable()
	message.ConfigID = ""
	_, err := (&Store{}).QueueChannelInbound(context.Background(), message)
	require.ErrorContains(t, err, "are required")
}
