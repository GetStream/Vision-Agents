//go:build integration

package conversation

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/require"
)

// Recovery check for a deletion task that outlived the probe's wait. Retrying
// deletion requires a separate opt-in and is restricted to validated fixture IDs.
func TestLiveProbeCleanupStatus(t *testing.T) {
	taskID := os.Getenv("ATHENA_CHAT_CLEANUP_TASK")
	if taskID == "" {
		t.Skip("ATHENA_CHAT_CLEANUP_TASK required")
	}
	client, err := getstream.NewClient(os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET"))
	require.NoError(t, err)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()
	task, err := client.GetTask(ctx, taskID, &getstream.GetTaskRequest{})
	require.NoError(t, err)
	t.Logf("deletion task %s status=%s error=%v", taskID, task.Data.Status, task.Data.Error)
	t.Logf("deletion result: %v", task.Data.Result)
	ids := make([]string, 0, len(task.Data.Result))
	for id := range task.Data.Result {
		parts := strings.Split(strings.TrimPrefix(id, "athena-probe-"), "-")
		require.True(t, strings.HasPrefix(id, "athena-probe-"))
		require.Len(t, parts, 6)
		_, err := uuid.Parse(strings.Join(parts[:5], "-"))
		require.NoError(t, err)
		require.Contains(t, []string{"owner", "outsider", "bot"}, parts[5])
		ids = append(ids, id)
	}
	require.NotEmpty(t, ids)
	query := &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{
		FilterConditions: map[string]any{"id": map[string]any{"$in": ids}},
	}}
	users, err := client.QueryUsers(ctx, query)
	require.NoError(t, err)
	if len(users.Data.Users) > 0 && os.Getenv("ATHENA_CHAT_CLEANUP_RETRY") == "1" {
		remaining := make([]string, 0, len(users.Data.Users))
		for _, user := range users.Data.Users {
			require.Contains(t, ids, user.ID)
			remaining = append(remaining, user.ID)
		}
		hard := "hard"
		deleted, err := client.DeleteUsers(ctx, &getstream.DeleteUsersRequest{UserIds: remaining, User: &hard, Messages: &hard, Conversations: &hard})
		require.NoError(t, err)
		t.Logf("targeted cleanup task: %s", deleted.Data.TaskID)
		_, err = getstream.WaitForTask(ctx, client, deleted.Data.TaskID, getstream.WithWaitForTaskTimeout(2*time.Minute))
		require.NoError(t, err)
		users, err = client.QueryUsers(ctx, query)
		require.NoError(t, err)
	}
	require.Empty(t, users.Data.Users, "verify exact fixture IDs, even when task status is completed")
	cid := os.Getenv("ATHENA_CHAT_CLEANUP_CID")
	if cid != "" {
		channels, err := client.Chat().QueryChannels(ctx, &getstream.QueryChannelsRequest{FilterConditions: map[string]any{"cid": cid}})
		require.NoError(t, err)
		require.Empty(t, channels.Data.Channels)
	}
	require.Equal(t, "completed", task.Data.Status)
}
