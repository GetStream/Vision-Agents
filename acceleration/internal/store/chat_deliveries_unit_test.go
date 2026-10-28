package store

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestQueueingAChatDeliveryWithoutAMessageIDIsRefused(t *testing.T) {
	_, err := (&Store{}).QueueChatDelivery(context.Background(), ChatDelivery{StreamApp: 1257545, Body: []byte(`{}`)})
	require.ErrorContains(t, err, "are required")
}

func TestQueueingAChatDeliveryWithoutABodyIsRefused(t *testing.T) {
	_, err := (&Store{}).QueueChatDelivery(context.Background(), ChatDelivery{StreamApp: 1257545, MessageID: "msg-one"})
	require.ErrorContains(t, err, "are required")
}
