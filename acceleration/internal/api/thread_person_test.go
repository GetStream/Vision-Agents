package api

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/acceleration/internal/channelbridge"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// A Linq thread key is the chat id, so a chat is one person's only through the person the
// bridge recorded for it, never through the user an empty handle would be.
func TestALinqChatIsOnePersonsOnlyWhileItHasAChatPerson(t *testing.T) {
	chat := store.ChannelThread{CustomerID: "acme", ConnectorID: "linq", ProviderUnitID: "+12025550100", ThreadKey: "chat-1"}
	person, personal := threadPerson(chat)
	require.False(t, personal)
	require.Empty(t, person)

	chat.ChatPerson = "+12025550199"
	person, personal = threadPerson(chat)
	require.True(t, personal)
	require.Equal(t, channelbridge.AuthorUserID("acme", "linq", "+12025550100", "+12025550199"), person,
		"the user the bridge writes the sender as, not the chat")
}
