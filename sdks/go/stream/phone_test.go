package stream

import (
	"io"
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

func TestSetAgentPutsTheAgentOnTheNumber(t *testing.T) {
	var method, path, body string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.EscapedPath()
		sent, _ := io.ReadAll(r.Body)
		body = string(sent)
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"e164": "+15125551234", "agent_config_id": "cfg_1"}`))
	}))
	t.Cleanup(server.Close)
	client, err := acceleration.NewClientWithResponses(server.URL)
	require.NoError(t, err)

	number, err := NewPhone(client).SetAgent(t.Context(), "+15125551234", "support")
	require.NoError(t, err)
	require.Equal(t, http.MethodPut, method)
	require.Equal(t, "/v1/phone/numbers/+15125551234/agent", path)
	require.JSONEq(t, `{"agent":"support"}`, body)
	require.Equal(t, "cfg_1", *number.AgentConfigId)
}

func TestSetAgentSaysWhyTheRouterRefused(t *testing.T) {
	server := refusing(t, http.StatusConflict, "application/json", `{"error": {
		"message": "+15125551234 is attached to a worker",
		"type": "conflict", "code": "phone_number_conflict",
		"doc_url": "https://getstream.io/agents/docs/api/errors/#phone_number_conflict"}}`)
	client, err := acceleration.NewClientWithResponses(server.URL)
	require.NoError(t, err)

	_, err = NewPhone(client).SetAgent(t.Context(), "+15125551234", "support")
	require.ErrorContains(t, err, "+15125551234 is attached to a worker")
}

func TestRemoveAgentDeletesIt(t *testing.T) {
	var method, path string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		method, path = r.Method, r.URL.EscapedPath()
		w.WriteHeader(http.StatusNoContent)
	}))
	t.Cleanup(server.Close)
	client, err := acceleration.NewClientWithResponses(server.URL)
	require.NoError(t, err)

	require.NoError(t, NewPhone(client).RemoveAgent(t.Context(), "+15125551234"))
	require.Equal(t, http.MethodDelete, method)
	require.Equal(t, "/v1/phone/numbers/+15125551234/agent", path)
}
