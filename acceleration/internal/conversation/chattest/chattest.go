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
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
)

type store struct {
	mu       sync.Mutex
	channels map[string]map[string]any
	messages map[string]map[string]any
	users    map[string]map[string]any
	order    []string
	now      func() time.Time
}

// Server is Chat in memory, for a test that needs to steer or look at what is stored.
type Server struct {
	// Client talks to it.
	Client *getstream.Stream
	// URL is where it is served, for a client of one's own to be pointed at.
	URL string
	db  *store
}

// NewServer serves Chat from memory for the life of the test.
func NewServer(t *testing.T) *Server {
	t.Helper()
	db := &store{
		channels: map[string]map[string]any{}, messages: map[string]map[string]any{},
		users: map[string]map[string]any{}, now: time.Now,
	}
	server := httptest.NewServer(http.HandlerFunc(db.serve))
	t.Cleanup(server.Close)
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(server.URL))
	if err != nil {
		t.Fatalf("chattest: %v", err)
	}
	return &Server{Client: client, URL: server.URL, db: db}
}

// Client serves Chat from memory for the life of the test.
func Client(t *testing.T) *getstream.Stream {
	t.Helper()
	return NewServer(t).Client
}

// At dates every message stored from now on, which Chat does by its own clock. A test
// places lines either side of something with it.
func (s *Server) At(at time.Time) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.now = func() time.Time { return at }
}

// Channel returns what an agent channel was created with, and whether it exists at all.
func (s *Server) Channel(id string) (map[string]any, bool) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	data, ok := s.db.channels[id]
	return data, ok
}

// User returns a user as Chat holds them, and whether Chat has them at all.
func (s *Server) User(id string) (map[string]any, bool) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	user, ok := s.db.users[id]
	return user, ok
}

// PutUser stores a user the app made itself, the way a real person is already there before
// the router writes anything near them.
func (s *Server) PutUser(user map[string]any) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.users[user["id"].(string)] = user
}

// refuse answers the way Chat does when it will not do what was asked.
func refuse(w http.ResponseWriter, message string) {
	w.WriteHeader(http.StatusBadRequest)
	_ = json.NewEncoder(w).Encode(map[string]any{"code": 4, "message": message, "StatusCode": http.StatusBadRequest})
}

// messagesIn returns a channel's messages in the order they were written.
func (db *store) messagesIn(id string) []map[string]any {
	messages := []map[string]any{}
	for _, mid := range db.order {
		if db.messages[mid]["cid"] == "agent:"+id {
			messages = append(messages, db.messages[mid])
		}
	}
	return messages
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
	case strings.HasSuffix(r.URL.Path, "/users") && r.Method == http.MethodGet:
		var payload struct {
			FilterConditions map[string]any `json:"filter_conditions"`
		}
		_ = json.Unmarshal([]byte(r.URL.Query().Get("payload")), &payload)
		users := []map[string]any{}
		if id, ok := payload.FilterConditions["id"].(map[string]any); ok {
			if in, ok := id["$in"].([]any); ok {
				for _, wanted := range in {
					if user, exists := db.users[fmt.Sprint(wanted)]; exists {
						users = append(users, user)
					}
				}
			}
		}
		result["users"] = users
	case strings.HasSuffix(r.URL.Path, "/users") && r.Method == http.MethodPost:
		// An upsert replaces the user whole, which is what Chat does.
		written, _ := body["users"].(map[string]any)
		for id, user := range written {
			if fields, ok := user.(map[string]any); ok {
				db.users[id] = fields
			}
		}
		result["users"] = written
	case r.Method == http.MethodPost && strings.HasSuffix(r.URL.Path, "/chat/channels"):
		// A query finds channels and never creates one.
		channels := []map[string]any{}
		filter, _ := body["filter_conditions"].(map[string]any)
		if cid, ok := filter["cid"].(string); ok {
			id := strings.TrimPrefix(cid, "agent:")
			if data, exists := db.channels[id]; exists && strings.HasPrefix(cid, "agent:") {
				channels = append(channels, map[string]any{
					"channel": data, "members": data["members"], "messages": db.messagesIn(id),
				})
			}
		}
		result["channels"] = channels
	case strings.HasSuffix(r.URL.Path, "/query"):
		id := parts[len(parts)-2]
		_, exists := db.channels[id]
		data, carries := body["data"].(map[string]any)
		// The agent channel type refuses a server-side create without a creator, so a
		// query for a channel that is not there creates nothing unless it names one.
		if !exists && (!carries || (data["created_by_id"] == nil && data["created_by"] == nil)) {
			refuse(w, "either data.created_by or data.created_by_id must be provided when using server side auth")
			return
		}
		// A query carrying data creates the channel; one without it must not overwrite
		// the ownership already recorded on it.
		if carries {
			db.channels[id] = data
		}
		result["channel"] = db.channels[id]
		result["members"] = db.channels[id]["members"]
		result["messages"] = db.messagesIn(id)
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
			message["created_at"] = db.now().UnixNano()
			// Chat answers with the author it resolved the id to, which is how a reader
			// learns who spoke: a stored message carries a user, not a user id. A user
			// nobody named comes back named after their own id.
			message["user"] = map[string]any{"id": message["user_id"], "name": message["user_id"]}
			if _, ok := message["custom"].(map[string]any); !ok {
				message["custom"] = map[string]any{}
			}
			db.messages[id] = message
		}
		result["message"] = db.messages[id]
	case strings.Contains(r.URL.Path, "/messages/") && r.Method == http.MethodDelete:
		id := parts[len(parts)-1]
		result["message"] = db.messages[id]
		if r.URL.Query().Get("hard") == "true" {
			delete(db.messages, id)
			for i, ordered := range db.order {
				if ordered == id {
					db.order = append(db.order[:i], db.order[i+1:]...)
					break
				}
			}
		} else if stored := db.messages[id]; stored != nil {
			stored["type"] = "deleted"
		}
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
