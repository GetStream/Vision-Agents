//go:build integration

package api

import (
	"context"
	"os"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

// The opt-in agent names an existing local-development transcript the operator wants to read.
func TestLiveChatReaderJoinsExistingChannel(t *testing.T) {
	agentID := os.Getenv("VA_LIVE_CHAT_AGENT")
	if agentID == "" {
		t.Skip("unblock: set VA_LIVE_CHAT_AGENT and local Stream credentials to exercise an existing development transcript")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	ctx = context.WithValue(ctx, customerContextKey{}, "examples")
	server := &Server{streamKey: os.Getenv("STREAM_API_KEY"), streamSecret: os.Getenv("STREAM_API_SECRET")}
	result, err := server.CreateChatToken(ctx, CreateChatTokenRequestObject{Body: &ChatTokenRequest{AgentId: agentID}})
	require.NoError(t, err)
	token, ok := result.(CreateChatToken200JSONResponse)
	require.True(t, ok, "chat token was not issued")
	require.Equal(t, agentID, token.ChannelId)
	require.Equal(t, "reader-examples", token.UserId)
	client, err := getstream.NewClient(server.streamKey, server.streamSecret)
	require.NoError(t, err)
	channel, err := client.Chat().GetOrCreateChannel(ctx, "agent", agentID, &getstream.GetOrCreateChannelRequest{})
	require.NoError(t, err)
	found := false
	for _, member := range channel.Data.Members {
		if member.UserID != nil && *member.UserID == token.UserId {
			found = true
		}
	}
	require.True(t, found, "token reader must actually be a member of the existing channel")
	// Issuing another token keeps membership intact rather than recreating the channel.
	_, err = server.CreateChatToken(ctx, CreateChatTokenRequestObject{Body: &ChatTokenRequest{AgentId: agentID}})
	require.NoError(t, err)
}
