package conversation

import (
	"crypto/rand"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

func TestOpeningAConversationLeavesAnExistingUserAlone(t *testing.T) {
	// The caller is a real person in the app, with a name and a picture the app gave them.
	// Opening a conversation with them must not write them back as a bare id.
	chat := chattest.NewServer(t)
	chat.PutUser(map[string]any{"id": "employee", "name": "Ada Lovelace", "image": "https://example.com/ada.png"})
	service := NewForChat(chat.Client)
	t.Cleanup(service.Close)

	// Fresh, so no earlier test can have it: CreateMissingUsers remembers users per URL and
	// key for the whole process, and a later chattest server can be given an earlier one's port.
	agent := "agent-" + rand.Text()
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", agent, "", "employee")
	require.NoError(t, err)
	c.Release()

	employee, ok := chat.User("employee")
	require.True(t, ok)
	require.Equal(t, "Ada Lovelace", employee["name"])
	require.Equal(t, "https://example.com/ada.png", employee["image"])
	_, created := chat.User(agent)
	require.True(t, created, "a user the app has never seen is still created")
}

func TestAChattestServerOnAReusedURLDoesNotInheritUsersKnownToAnEarlierOne(t *testing.T) {
	// An earlier server had this user created, then the new server got its port. The cache
	// outlives the earlier one, and the new server never saw the user.
	earlier, chat := chattest.NewServer(t), chattest.NewServer(t)
	id := "reused-" + rand.Text()
	remember(chat.URL+"\x00"+earlier.Key, []string{id})

	require.NoError(t, CreateMissingUsers(t.Context(), chat.Client, map[string]getstream.UserRequest{id: {ID: id}}))

	_, created := chat.User(id)
	require.True(t, created, "the user was taken for known from a server that is gone")
}
