package api

import (
	"net/http"
	"os"
	"testing"

	"github.com/go-chi/chi/v5"
	"github.com/stretchr/testify/require"
)

// TestTheTelnyxCallHookIsDeclared is the handwritten webhook: served on the mux, and
// named in the spec the same way the 10DLC report is, including the committed document.
func TestTheTelnyxCallHookIsDeclared(t *testing.T) {
	item := (&Server{}).newAPI(chi.NewRouter()).OpenAPI().Paths[telnyxCallHookPath]
	require.NotNil(t, item)
	require.NotNil(t, item.Post)
	require.Equal(t, "receiveTelnyxCall", item.Post.OperationID)
	require.Equal(t, http.MethodPost, item.Post.Method)
	require.NotNil(t, item.Post.Security)
	require.Empty(t, item.Post.Security)
	for _, code := range []string{"200", "401", "404", "413"} {
		require.Contains(t, item.Post.Responses, code)
	}
	require.Contains(t, item.Post.Responses["200"].Content, "application/xml")

	committed, err := os.ReadFile("../../api/openapi.yaml")
	require.NoError(t, err)
	require.Contains(t, string(committed), "/v1/phone/hooks/telnyx:")
}
