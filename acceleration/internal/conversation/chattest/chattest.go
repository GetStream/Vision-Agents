// Package chattest is Stream Chat with nothing behind it. It is enough for a persistent
// conversation to open a channel, write to it and read its own history back, so a test
// about what a conversation does can be written without a Chat account.
package chattest

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	getstream "github.com/GetStream/getstream-go/v5"
)

type store struct {
	mu       sync.Mutex
	channels map[string]map[string]any
	messages map[string]map[string]any
	order    []string
}

// Client serves Chat from memory for the life of the test.
func Client(t *testing.T) *getstream.Stream {
	t.Helper()
	db := &store{channels: map[string]map[string]any{}, messages: map[string]map[string]any{}}
	server := httptest.NewServer(http.HandlerFunc(db.serve))
	t.Cleanup(server.Close)
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(server.URL))
	if err != nil {
		t.Fatalf("chattest: %v", err)
	}
	return client
}

func (db *store) serve(w http.ResponseWriter, r *http.Request) {
	db.mu.Lock()
	defer db.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	var body map[string]any
	if r.Body != nil {
		_ = json.NewDecoder(r.Body).Decode(&body)
	}
	parts := strings.Split(r.URL.Path, "/")
	result := map[string]any{}
	switch {
	case strings.HasSuffix(r.URL.Path, "/query"):
		id := parts[len(parts)-2]
		// A query carrying data creates the channel; one without it must not overwrite
		// the ownership already recorded on it.
		if data, ok := body["data"].(map[string]any); ok {
			db.channels[id] = data
		}
		result["channel"] = db.channels[id]
		result["members"] = db.channels[id]["members"]
		messages := []map[string]any{}
		for _, id := range db.order {
			if db.messages[id]["cid"] == "agent:"+parts[len(parts)-2] {
				messages = append(messages, db.messages[id])
			}
		}
		result["messages"] = messages
	case strings.HasSuffix(r.URL.Path, "/message"):
		message := body["message"].(map[string]any)
		id, _ := message["id"].(string)
		if id == "" {
			id = fmt.Sprintf("msg-%d", len(db.order)+1)
			message["id"] = id
		}
		if _, exists := db.messages[id]; !exists {
			db.order = append(db.order, id)
			message["cid"] = "agent:" + parts[len(parts)-2]
			if _, ok := message["custom"].(map[string]any); !ok {
				message["custom"] = map[string]any{}
			}
			db.messages[id] = message
		}
		result["message"] = db.messages[id]
	case strings.Contains(r.URL.Path, "/messages/"):
		id := parts[len(parts)-1]
		// An ephemeral patch is what a reply streaming into the channel looks like, and
		// is deliberately not stored: only the durable update is history.
		if id == "ephemeral" {
			id = parts[len(parts)-2]
		} else if r.Method == http.MethodPut {
			stored := db.messages[id]
			if stored["custom"] == nil {
				stored["custom"] = map[string]any{}
			}
			for key, value := range body["set"].(map[string]any) {
				if key == "text" || key == "attachments" {
					stored[key] = value
				} else {
					stored["custom"].(map[string]any)[key] = value
				}
			}
		}
		result["message"] = db.messages[id]
	}
	_ = json.NewEncoder(w).Encode(result)
}
