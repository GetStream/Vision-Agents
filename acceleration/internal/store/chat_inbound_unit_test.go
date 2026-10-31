package store

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQueueingAChatInboundWithoutAMessageIDIsRefused(t *testing.T) {
	_, err := (&Store{}).QueueChatInbound(context.Background(), ChatInbound{StreamApp: 1257545, Body: []byte(`{}`)})
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAChatInboundWithoutABodyIsRefused(t *testing.T) {
	_, err := (&Store{}).QueueChatInbound(context.Background(), ChatInbound{StreamApp: 1257545, MessageID: "msg-one"})
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAChatInboundWithoutAStreamAppIsRefused(t *testing.T) {
	_, err := (&Store{}).QueueChatInbound(context.Background(), ChatInbound{MessageID: "msg-one", Body: []byte(`{}`)})
	require.ErrorContains(t, err, "Stream app is required")
}
