//go:build integration

package conversation

import (
	"context"
	"errors"
	"os"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// TestARewindAgainstRealChatHidesTheTailAndReadsAgree rewinds a real agent channel: the
// snapshot read names the newest message the rewind expects, a stale one is refused with
// the actual one, and after the rewind the read no longer returns the hidden messages.
func TestARewindAgainstRealChatHidesTheTailAndReadsAgree(t *testing.T) {
	if os.Getenv("STREAM_API_KEY") == "" || os.Getenv("STREAM_API_SECRET") == "" {
		t.Skip("STREAM_API_KEY and STREAM_API_SECRET not set")
	}
	ctx := context.Background()
	client, err := getstream.NewClient(os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET"))
	require.NoError(t, err)
	agent, person := "rewind-agent-"+uuid.NewString(), "rewind-person-"+uuid.NewString()
	require.NoError(t, CreateMissingUsers(ctx, client, map[string]getstream.UserRequest{agent: {ID: agent}, person: {ID: person}}))
	channel := "rewind-test-" + uuid.NewString()
	_, err = client.Chat().GetOrCreateChannel(ctx, "agent", channel, &getstream.GetOrCreateChannelRequest{
		Data: &getstream.ChannelInput{CreatedByID: &agent, Members: []getstream.ChannelMemberRequest{{UserID: agent}, {UserID: person}}}})
	require.NoError(t, err)
	t.Cleanup(func() {
		hard := true
		_, _ = client.Chat().DeleteChannel(context.Background(), "agent", channel, &getstream.DeleteChannelRequest{HardDelete: &hard})
	})
	var ids []string
	for i, text := range []string{"question", "answer", "follow-up", "second answer"} {
		user := person
		if i%2 == 1 {
			user = agent
		}
		id := uuid.NewString()
		_, err := client.Chat().SendMessage(ctx, "agent", channel, &getstream.SendMessageRequest{
			Message: getstream.MessageRequest{ID: &id, Text: &text, UserID: &user, Custom: map[string]any{"step": i}}})
		require.NoError(t, err)
		ids = append(ids, id)
		// Rewinds hide created_at ranges: two messages at one instant cannot be told apart.
		time.Sleep(5 * time.Millisecond)
	}
	rewinder := NewRewinder(client)

	tail, err := rewinder.ReadFrom(ctx, "agent", channel, ids[1], false)
	require.NoError(t, err)
	require.Len(t, tail.Messages, 2)
	require.Equal(t, ids[2], tail.Messages[0].ID)
	require.Equal(t, ids[3], tail.LastID)

	_, err = rewinder.Rewind(ctx, "agent", channel, ids[1], false, ids[2], agent)
	if errors.Is(err, ErrRewindUnavailable) {
		t.Skip("the app of STREAM_API_KEY is not granted chat.agent_rewind, or has it frozen")
	}
	conflict, ok := errors.AsType[*RewindConflictError](err)
	require.True(t, ok, "a stale expected last message is a conflict, got %v", err)
	require.Equal(t, ids[3], conflict.LastMessageID)

	got, err := rewinder.Rewind(ctx, "agent", channel, ids[1], false, tail.LastID, agent)
	require.NoError(t, err)
	require.Equal(t, ids[1], got.MessageID)
	require.Equal(t, 2, got.HiddenCountAtRewind)

	_, err = rewinder.Rewind(ctx, "agent", channel, ids[1], false, ids[1], agent)
	require.ErrorIs(t, err, ErrNothingToRewind)

	// Reads take the ranges from a replica, which can serve them from before the rewind
	// for a moment after it.
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		after, err := rewinder.ReadFrom(ctx, "agent", channel, ids[0], true)
		require.NoError(c, err)
		var read []string
		for _, m := range after.Messages {
			read = append(read, m.ID)
		}
		require.Equal(c, ids[:2], read)
		require.Equal(c, ids[1], after.LastID)
	}, 10*time.Second, 250*time.Millisecond)
}
