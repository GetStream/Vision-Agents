package stream

import (
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

// AnswerApproval posts the answer to the conversation's tool_approvals, with the summary only
// when there is one, and a refusal comes back as the router's error.
func TestAnswerApprovalPostsToTheConversation(t *testing.T) {
	var path string
	var sent []map[string]any
	status := http.StatusNoContent
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.Method + " " + r.URL.Path
		var body map[string]any
		_ = json.NewDecoder(r.Body).Decode(&body)
		sent = append(sent, body)
		if status != http.StatusNoContent {
			w.Header().Set("Content-Type", "application/json")
		}
		w.WriteHeader(status)
		if status != http.StatusNoContent {
			_, _ = w.Write([]byte(`{"error": {"message": "this call waits for another person to allow it", "type": "permission_denied", "code": "permission_denied"}}`))
		}
	}))
	t.Cleanup(server.Close)
	backend := Backend{URL: server.URL, CustomerID: "acme"}

	if err := backend.AnswerApproval(t.Context(), "callouts:team-1", "call-1", "thread_1", true, ""); err != nil {
		t.Fatalf("allowed: %v", err)
	}
	if err := backend.AnswerApproval(t.Context(), "callouts:team-1", "call-2", "thread_1", false, "Not now"); err != nil {
		t.Fatalf("declined: %v", err)
	}
	status = http.StatusForbidden
	err := backend.AnswerApproval(t.Context(), "callouts:team-1", "call-3", "thread_1", true, "")

	if path != "POST /v1/agents/conversations/callouts:team-1/tool_approvals" {
		t.Errorf("posted to %q", path)
	}
	want := []map[string]any{
		{"tool_call_id": "call-1", "request_id": "thread_1", "allowed": true},
		{"tool_call_id": "call-2", "request_id": "thread_1", "allowed": false, "summary": "Not now"},
		{"tool_call_id": "call-3", "request_id": "thread_1", "allowed": true},
	}
	if !reflect.DeepEqual(sent, want) {
		t.Errorf("sent %v", sent)
	}
	var refused *RouterError
	if !errors.As(err, &refused) || refused.Status != http.StatusForbidden {
		t.Errorf("the refusal came back as %v", err)
	}
}
