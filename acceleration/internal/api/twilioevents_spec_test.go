package api

import (
	"os"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/require"
)

// TestTheTwilioEventsHookIsDeclared is the Twilio account-event hook: public, named in the
// spec with every answer it gives, and in the committed document.
func TestTheTwilioEventsHookIsDeclared(t *testing.T) {
	item := (&Server{}).newAPI(chi.NewRouter()).OpenAPI().Paths[twilioEventsHookPath]
	require.NotNil(t, item)
	require.NotNil(t, item.Post)
	require.Equal(t, "receiveTwilioEvent", item.Post.OperationID)
	require.NotNil(t, item.Post.Security)
	require.Empty(t, item.Post.Security)
	for _, code := range []string{"204", "400", "401", "404", "413"} {
		require.Contains(t, item.Post.Responses, code)
	}

	committed, err := os.ReadFile("../../api/openapi.yaml")
	require.NoError(t, err)
	require.Contains(t, string(committed), "/v1/phone/hooks/twilio/events:")
}
