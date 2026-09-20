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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// Explicit opt-in: creates three disposable identities and one conversation,
// verifies real client-token permissions, and removes only these fixtures.
// This is a permissions probe; it does not prove model/session delivery.
func TestLivePersonalConversationPermissions(t *testing.T) {
	if os.Getenv("ATHENA_CHAT_PROBE") != "1" {
		t.Skip("ATHENA_CHAT_PROBE=1 required")
	}
	key, secret := os.Getenv("STREAM_API_KEY"), os.Getenv("STREAM_API_SECRET")
	require.NotEmpty(t, key)
	require.NotEmpty(t, secret)
	client, err := getstream.NewClient(key, secret)
	require.NoError(t, err)
	prefix := "athena-probe-" + uuid.NewString()
	owner, outsider, bot := prefix+"-owner", prefix+"-outsider", prefix+"-bot"
	ids := []string{owner, outsider, bot}
	t.Logf("disposable identity prefix: %s", prefix)
	var cid string
	// Register cleanup before the first mutation, including partial setup failures.
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
		defer cancel()
		if cid != "" {
			hardDelete := true
			_, err := client.Chat().DeleteChannel(ctx, "agent", strings.TrimPrefix(cid, "agent:"), &getstream.DeleteChannelRequest{HardDelete: &hardDelete})
			assert.NoError(t, err, "remove fixture channel before deleting its members")
		}
		hard := "hard"
		deleted, err := client.DeleteUsers(ctx, &getstream.DeleteUsersRequest{
			UserIds: ids, User: &hard, Messages: &hard, Conversations: &hard,
		})
		if !assert.NoError(t, err, "remove disposable probe identities") {
			return
		}
		t.Logf("fixture deletion task: %s", deleted.Data.TaskID)
		_, err = getstream.WaitForTask(ctx, client, deleted.Data.TaskID, getstream.WithWaitForTaskTimeout(2*time.Minute))
		if !assert.NoError(t, err, "wait for fixture deletion") {
			return
		}
		users, err := client.QueryUsers(ctx, &getstream.QueryUsersRequest{Payload: &getstream.QueryUsersPayload{
			FilterConditions: map[string]any{"id": map[string]any{"$in": ids}},
		}})
		if assert.NoError(t, err) {
			require.Empty(t, users.Data.Users, "all fixture identities must be removed")
		}
		if cid != "" {
			channels, err := client.Chat().QueryChannels(ctx, &getstream.QueryChannelsRequest{FilterConditions: map[string]any{"cid": cid}})
			if assert.NoError(t, err) {
				require.Empty(t, channels.Data.Channels, "fixture channel must be removed")
			}
		}
		t.Log("disposable identities and conversation deletion verified")
	})
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()
	_, err = client.UpdateUsers(ctx, &getstream.UpdateUsersRequest{Users: map[string]getstream.UserRequest{
		owner: {ID: owner}, outsider: {ID: outsider}, bot: {ID: bot},
	}})
	require.NoError(t, err)
	service, err := New(t.TempDir())
	require.NoError(t, err)
	t.Cleanup(service.Close)
	conversation, _, _, err := service.OpenForCaller(ctx, "1257545", bot, "", owner)
	require.NoError(t, err)
	cid = conversation.CID()
	t.Logf("isolated conversation: %s", cid)
	_, err = service.HistoryForCaller(ctx, "1257545", bot, cid, "", outsider)
	require.ErrorContains(t, err, "another user")
	_, err = service.HistoryForCaller(ctx, "1257545", bot, cid, "", owner)
	require.NoError(t, err)

	userClient := func(id string) *getstream.Stream {
		t.Helper()
		token, err := client.CreateToken(id, getstream.WithExpiration(5*time.Minute))
		require.NoError(t, err)
		// The supplied token overrides SDK server signing. The dummy secret cannot
		// authorize a request, so a fallback to server credentials would fail.
		user, err := getstream.NewClient(key, "client-token-only", getstream.WithAuthToken(token))
		require.NoError(t, err)
		return user
	}
	state := true
	channelID := strings.TrimPrefix(cid, "agent:")
	member, err := userClient(owner).Chat().GetOrCreateChannel(ctx, "agent", channelID, &getstream.GetOrCreateChannelRequest{State: &state})
	require.NoError(t, err, "owner client must be able to read its channel")
	require.Equal(t, owner, member.Data.Channel.Custom["support_owner_id"])
	_, err = userClient(outsider).Chat().GetOrCreateChannel(ctx, "agent", channelID, &getstream.GetOrCreateChannelRequest{State: &state})
	var denied *getstream.StreamError
	require.ErrorAs(t, err, &denied, "nonmember client must not read private history")
	require.Equal(t, 403, denied.StatusCode)
	t.Log("owner client read succeeds; nonmember client read returns 403")
}
