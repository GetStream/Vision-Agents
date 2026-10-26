package stream

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
)

func TestDetachStopsANumberTakingCalls(t *testing.T) {
	var path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodDelete {
			path = r.URL.EscapedPath()
		}
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)
	client, err := acceleration.NewClientWithResponses(server.URL)
	require.NoError(t, err)

	require.NoError(t, NewPhone(client).Detach(t.Context(), "+15125551234"))
	require.Equal(t, "/v1/phone/numbers/+15125551234/attach", path)
}

func TestDetachSaysWhyTheRouterRefused(t *testing.T) {
	server := refusing(t, http.StatusNotFound, "application/json", `{"error": {
		"message": "no phone number +15125551234 is held",
		"type": "not_found", "code": "phone_number_not_found",
		"doc_url": "https://getstream.io/agents/docs/api/errors/#phone_number_not_found"}}`)
	client, err := acceleration.NewClientWithResponses(server.URL)
	require.NoError(t, err)

	err = NewPhone(client).Detach(t.Context(), "+15125551234")
	require.ErrorContains(t, err, "no phone number +15125551234 is held")
}
