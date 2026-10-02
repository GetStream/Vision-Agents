package conversation

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

func TestOpeningAConversationLeavesAnExistingUserAlone(t *testing.T) {
	// The caller is a real person in the app, with a name and a picture the app gave them.
	// Opening a conversation with them must not write them back as a bare id.
	chat := chattest.NewServer(t)
	chat.PutUser(map[string]any{"id": "employee", "name": "Ada Lovelace", "image": "https://example.com/ada.png"})
	service, err := NewForChat(t.TempDir(), chat.Client)
	require.NoError(t, err)
	t.Cleanup(service.Close)

	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	c.Release()

	employee, ok := chat.User("employee")
	require.True(t, ok)
	require.Equal(t, "Ada Lovelace", employee["name"])
	require.Equal(t, "https://example.com/ada.png", employee["image"])
	_, created := chat.User("agent")
	require.True(t, created, "a user the app has never seen is still created")
}
