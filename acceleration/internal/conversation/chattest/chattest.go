// Package chattest is Stream Chat with nothing behind it. It is enough for a persistent
// conversation to open a channel, write to it and read its own history back, so a test
// about what a conversation does can be written without a Chat account.
package chattest

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"net/url"
	"slices"
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
	trunks   map[string]map[string]any
	// calls are the participant ids in each call's session, by "<type>:<id>".
	calls    map[string][]string
	rules    map[string]map[string]any
	order    []string
	now      func() time.Time
	app      App
	appReads int
	// keyed are apps answered for one api key, standing in for several apps at one URL.
	keyed map[string]App
	// hooks are the event hooks each api key's app holds, as an app update last set them,
	// each exactly as it was sent.
	hooks map[string][]json.RawMessage
	// updates are the bodies of the app updates made with each api key, as they were sent.
	updates map[string][]json.RawMessage
	// unresolvable are the hosts an app update refuses a webhook hook on; see Unresolvable.
	unresolvable map[string]bool
	// asked is every request served, with the key it was made with.
	asked []Request
	// held is the request Hold keeps waiting, under a lock of its own: mu is taken only once
	// a request is served, so a held request leaves every other one to be answered.
	holdMu sync.Mutex
	held   *held
}

// held is one request Hold keeps waiting: the path it ends in, a channel closed once it
// waits, and one closed to let it go on.
type held struct {
	suffix         string
	waiting, letGo chan struct{}
	once           sync.Once
}

// Request is one request the server was sent, and the api key it was made with.
type Request struct {
	Method, Path, APIKey string
}

// Writes reports whether a request changes anything in Stream, rather than reading it: a
// GET, a channel query and a user query read; everything else writes.
func (r Request) Writes() bool {
	return r.Method != http.MethodGet && !strings.HasSuffix(r.Path, "/chat/channels")
}

// App is what an app says of itself when asked: its id, and the channel and call types it
// holds.
type App struct {
	ID int64
	// ChannelTypes are the channel types it holds, each with its grants by role.
	ChannelTypes map[string]map[string][]string
	// CallTypes are the call types it holds.
	CallTypes []string
	// Suspended and DisableAuthChecks are what the app says of its standing.
	Suspended, DisableAuthChecks bool
	// Refuses answers 401 to being asked, as Stream does for a key it does not accept.
	Refuses bool
}

// safeGrants are the agent channel type as an app set up for the router holds it: members
// read and write, and nobody but the app's backend makes, changes or joins a channel.
var safeGrants = map[string][]string{
	"channel_member": {"read-channel", "read-channel-members", "create-message"},
	"admin":          {"create-channel", "update-channel", "delete-channel"},
}

// Server is Chat in memory, for a test that needs to steer or look at what is stored.
type Server struct {
	// Client talks to it.
	Client *getstream.Stream
	// URL is where it is served, for a client of one's own to be pointed at.
	URL string
	db  *store
	t   *testing.T
}

// NewServer serves Chat from memory for the life of the test.
func NewServer(t *testing.T) *Server {
	t.Helper()
	db := &store{
		channels: map[string]map[string]any{}, messages: map[string]map[string]any{},
		users: map[string]map[string]any{}, trunks: map[string]map[string]any{},
		rules: map[string]map[string]any{}, calls: map[string][]string{}, hooks: map[string][]json.RawMessage{},
		updates: map[string][]json.RawMessage{}, now: time.Now,
		app: App{ID: 1, ChannelTypes: map[string]map[string][]string{"agent": safeGrants}, CallTypes: []string{"agent"}},
	}
	server := httptest.NewServer(http.HandlerFunc(db.serve))
	t.Cleanup(server.Close)
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(server.URL))
	if err != nil {
		t.Fatalf("chattest: %v", err)
	}
	return &Server{Client: client, URL: server.URL, db: db, t: t}
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

// SetApp says what the app is from now on. Every server starts as app 1 holding the agent
// channel type, with safe grants, and the agent call type.
func (s *Server) SetApp(app App) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.app = app
}

// SetAppFor says what the app is when asked with one api key, so one server can stand in
// for the deployment's app and a customer's at once. Every other key gets SetApp's.
func (s *Server) SetAppFor(apiKey string, app App) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	if s.db.keyed == nil {
		s.db.keyed = map[string]App{}
	}
	s.db.keyed[apiKey] = app
}

// EventHooks are the event hooks the app of an api key holds, as an app update last set
// them: none until one does.
func (s *Server) EventHooks(apiKey string) []getstream.EventHook {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	raw, err := json.Marshal(s.db.hooks[apiKey])
	if err != nil {
		s.t.Fatalf("chattest: %v", err)
	}
	var hooks []getstream.EventHook
	if err := json.Unmarshal(raw, &hooks); err != nil {
		s.t.Fatalf("chattest: %v", err)
	}
	return hooks
}

// SetEventHooks gives the app of an api key these hooks, each exactly as written, the way
// Stream holds fields the SDK's EventHook does not model.
func (s *Server) SetEventHooks(apiKey string, hooks ...json.RawMessage) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.hooks[apiKey] = hooks
}

// EventHooksJSON are the event hooks the app of an api key holds, each exactly as the last
// app update sent it.
func (s *Server) EventHooksJSON(apiKey string) []json.RawMessage {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return slices.Clone(s.db.hooks[apiKey])
}

// AppUpdates are the bodies of the app updates made with an api key, oldest first, as they
// were sent.
func (s *Server) AppUpdates(apiKey string) []json.RawMessage {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return slices.Clone(s.db.updates[apiKey])
}

// Unresolvable makes every app update that holds a webhook hook on host fail from now on,
// unchanged hooks included, as Stream refuses an update holding a hook whose url does not
// resolve: `phone hooks -remove` of one tunnel's call hook was refused over that same
// tunnel's message hook, left in the update, with «webhook URL for hook <id> must be a
// publicly accessible HTTP/HTTPS URL: unable to resolve url …» (UpdateApp's
// validateHookConfigs, 2026-10-09T13:14Z, AI-990 F22). A host that resolves, such as a
// stopped ngrok tunnel's, was kept in a later update that Stream took.
func (s *Server) Unresolvable(host string) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	if s.db.unresolvable == nil {
		s.db.unresolvable = map[string]bool{}
	}
	s.db.unresolvable[host] = true
}

// Requests are the requests made with an api key, oldest first.
func (s *Server) Requests(apiKey string) []Request {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var made []Request
	for _, request := range s.db.asked {
		if request.APIKey == apiKey {
			made = append(made, request)
		}
	}
	return made
}

// AppReads is how many times the app was asked what it is.
func (s *Server) AppReads() int {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return s.db.appReads
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

// Members returns the user ids a channel holds as members.
func (s *Server) Members(id string) []string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var ids []string
	members, _ := s.db.channels[id]["members"].([]any)
	for _, member := range members {
		if named, ok := member.(map[string]any); ok {
			if userID, ok := named["user_id"].(string); ok {
				ids = append(ids, userID)
			}
		}
	}
	return ids
}

// PutCall puts a call in session with the participants named, for a reader of the call's
// session to find: a SIP caller is sip-<number>.
func (s *Server) PutCall(callType, id string, participants ...string) {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	s.db.calls[callType+":"+id] = participants
}

// Trunks are the ids of the SIP trunks the app holds now.
func (s *Server) Trunks() []string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return slices.Sorted(maps.Keys(s.db.trunks))
}

// Rules are the ids of the SIP routing rules the app holds now.
func (s *Server) Rules() []string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return slices.Sorted(maps.Keys(s.db.rules))
}

// Rule is the body a SIP routing rule was created with, nil once it is deleted.
func (s *Server) Rule(id string) map[string]any {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	return s.db.rules[id]
}

// Hold keeps the next request whose path ends in suffix waiting, before it is served, until
// release is called: a Stream Chat call that takes as long as the test needs. waiting is
// closed once that request waits. Every other request is answered as before. The request is
// let go at the end of the test at the latest, so the server can close.
func (s *Server) Hold(suffix string) (waiting <-chan struct{}, release func()) {
	h := &held{suffix: suffix, waiting: make(chan struct{}), letGo: make(chan struct{})}
	release = func() { h.once.Do(func() { close(h.letGo) }) }
	s.t.Cleanup(release)
	s.db.holdMu.Lock()
	defer s.db.holdMu.Unlock()
	s.db.held = h
	return h.waiting, release
}

// wait keeps the request Hold asked for waiting until it is let go.
func (db *store) wait(path string) {
	db.holdMu.Lock()
	h := db.held
	if h == nil || !strings.HasSuffix(path, h.suffix) {
		db.holdMu.Unlock()
		return
	}
	db.held = nil
	db.holdMu.Unlock()
	close(h.waiting)
	<-h.letGo
}

// unique is an id nothing else has.
func unique() string {
	raw := make([]byte, 8)
	_, _ = rand.Read(raw)
	return hex.EncodeToString(raw)
}

// Messages are the texts of a channel's messages, in the order they were written.
func (s *Server) Messages(id string) []string {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var texts []string
	for _, message := range s.db.messagesIn(id) {
		text, _ := message["text"].(string)
		texts = append(texts, text)
	}
	return texts
}

// Stored are a channel's messages as Chat holds them, in the order they were written: what a
// test needs to deliver the message.new Chat would send for one.
func (s *Server) Stored(id string) []map[string]any {
	s.db.mu.Lock()
	defer s.db.mu.Unlock()
	var stored []map[string]any
	for _, message := range s.db.messagesIn(id) {
		stored = append(stored, maps.Clone(message))
	}
	return stored
}

// Delivered are a channel's messages as Chat's message.new carries them, in the order they
// were written. A hook has a message's custom fields at the top level of the message, beside
// its own fields, not under "custom" as Stored and the v2 API have them (Stream Chat docs,
// «Webhook Events», message.new; AI-989).
func (s *Server) Delivered(id string) []map[string]any {
	var delivered []map[string]any
	for _, stored := range s.Stored(id) {
		message := map[string]any{"id": stored["id"], "text": stored["text"], "user": stored["user"]}
		custom, _ := stored["custom"].(map[string]any)
		for key, value := range custom {
			message[key] = value
		}
		delivered = append(delivered, message)
	}
	return delivered
}

// refuse answers the way Chat does when it will not do what was asked.
func refuse(w http.ResponseWriter, message string) {
	w.WriteHeader(http.StatusBadRequest)
	_ = json.NewEncoder(w).Encode(map[string]any{"code": 4, "message": message, "StatusCode": http.StatusBadRequest})
}

// appFor is the app a request's api key is answered as.
func (db *store) appFor(r *http.Request) App {
	if app, ok := db.keyed[r.URL.Query().Get("api_key")]; ok {
		return app
	}
	return db.app
}

// unresolvableHook is the url of the first webhook hook in an app update whose host does
// not resolve, or empty when every one does.
func (db *store) unresolvableHook(hooks []json.RawMessage) string {
	for _, hook := range hooks {
		var fields struct {
			Address string `json:"webhook_url"`
		}
		_ = json.Unmarshal(hook, &fields)
		address := fields.Address
		if parsed, err := url.Parse(address); err == nil && db.unresolvable[parsed.Hostname()] {
			return address
		}
	}
	return ""
}

// numberTimes is a hook with its created_at and updated_at as GET /api/v2/app writes a
// time, integer nanoseconds (GetStream/chat lib/core/api/encoding/json.go), every other
// field as it was sent.
func numberTimes(hook json.RawMessage) json.RawMessage {
	decoder := json.NewDecoder(strings.NewReader(string(hook)))
	if open, err := decoder.Token(); err != nil || open != json.Delim('{') {
		return hook
	}
	out := []byte{'{'}
	for decoder.More() {
		key, _ := decoder.Token()
		var value json.RawMessage
		if err := decoder.Decode(&value); err != nil {
			return hook
		}
		name, _ := key.(string)
		var at time.Time
		if (name == "created_at" || name == "updated_at") && json.Unmarshal(value, &at) == nil && !at.IsZero() {
			value = json.RawMessage(fmt.Sprint(at.UnixNano()))
		}
		if len(out) > 1 {
			out = append(out, ',')
		}
		encoded, _ := json.Marshal(name)
		out = append(append(append(out, encoded...), ':'), value...)
	}
	return append(out, '}')
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
	db.wait(r.URL.Path)
	db.mu.Lock()
	defer db.mu.Unlock()
	w.Header().Set("Content-Type", "application/json")
	db.asked = append(db.asked, Request{Method: r.Method, Path: r.URL.Path, APIKey: r.URL.Query().Get("api_key")})
	var raw []byte
	if r.Body != nil {
		raw, _ = io.ReadAll(r.Body)
	}
	var body map[string]any
	_ = json.Unmarshal(raw, &body)
	parts := strings.Split(r.URL.Path, "/")
	result := map[string]any{}
	switch {
	case r.Method == http.MethodGet && strings.HasSuffix(r.URL.Path, "/api/v2/app"):
		db.appReads++
		app := db.appFor(r)
		if app.Refuses {
			w.WriteHeader(http.StatusUnauthorized)
			_ = json.NewEncoder(w).Encode(map[string]any{"code": 5, "message": "api key not valid", "StatusCode": http.StatusUnauthorized})
			return
		}
		channels, calls := map[string]any{}, map[string]any{}
		for name := range app.ChannelTypes {
			channels[name] = map[string]any{"name": name}
		}
		for _, name := range app.CallTypes {
			calls[name] = map[string]any{"name": name}
		}
		result["app"] = map[string]any{
			"id": app.ID, "channel_configs": channels, "call_types": calls,
			"suspended": app.Suspended, "disable_auth_checks": app.DisableAuthChecks,
			"event_hooks": db.hooks[r.URL.Query().Get("api_key")],
		}
	case r.Method == http.MethodPatch && strings.HasSuffix(r.URL.Path, "/api/v2/app"):
		// An app update replaces the hooks whole with the ones it sends, as Stream's does.
		if db.appFor(r).Refuses {
			w.WriteHeader(http.StatusUnauthorized)
			_ = json.NewEncoder(w).Encode(map[string]any{"code": 5, "message": "api key not valid", "StatusCode": http.StatusUnauthorized})
			return
		}
		db.updates[r.URL.Query().Get("api_key")] = append(db.updates[r.URL.Query().Get("api_key")], raw)
		var update struct {
			EventHooks *[]json.RawMessage `json:"event_hooks"`
		}
		_ = json.Unmarshal(raw, &update)
		if update.EventHooks != nil {
			hooks := *update.EventHooks
			// Stream decodes each hook's times into a time.Time, which takes only text
			// (GetStream/chat monolith/types/event_hook.go; update_app.go): a number, which is
			// how GET /api/v2/app writes one, is refused as invalid input.
			for index, hook := range hooks {
				var times struct {
					CreatedAt time.Time `json:"created_at"`
					UpdatedAt time.Time `json:"updated_at"`
				}
				if err := json.Unmarshal(hook, &times); err != nil {
					refuse(w, "UpdateApp failed with error: "+err.Error())
					return
				}
				hooks[index] = numberTimes(hook)
			}
			if refused := db.unresolvableHook(hooks); refused != "" {
				// The status and code are unverified: the refusal was seen only as the Go
				// client's error text, which carries the message alone.
				w.WriteHeader(http.StatusBadRequest)
				_ = json.NewEncoder(w).Encode(map[string]any{"code": 4, "StatusCode": http.StatusBadRequest,
					"message": "webhook URL for hook must be a publicly accessible HTTP/HTTPS URL: unable to resolve url " + refused})
				return
			}
			db.hooks[r.URL.Query().Get("api_key")] = hooks
		}
	case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/channeltypes/"):
		name := parts[len(parts)-1]
		grants, ok := db.appFor(r).ChannelTypes[name]
		if !ok {
			w.WriteHeader(http.StatusNotFound)
			_ = json.NewEncoder(w).Encode(map[string]any{"code": 16, "message": "channel type " + name + " does not exist", "StatusCode": http.StatusNotFound})
			return
		}
		result["name"], result["grants"] = name, grants
	case r.Method == http.MethodGet && strings.Contains(r.URL.Path, "/video/call/"):
		// A call nobody put is answered as one with nobody in session, which is what a call
		// whose session has not started reads as.
		cid := parts[len(parts)-2] + ":" + parts[len(parts)-1]
		call := map[string]any{"cid": cid, "type": parts[len(parts)-2], "id": parts[len(parts)-1]}
		if ids, ok := db.calls[cid]; ok {
			participants := []map[string]any{}
			for _, id := range ids {
				participants = append(participants, map[string]any{"user": map[string]any{"id": id}, "role": "user"})
			}
			call["session"] = map[string]any{"id": "session-" + cid, "participants": participants}
		}
		result["call"] = call
	case strings.HasSuffix(r.URL.Path, "/sip/inbound_trunks") && r.Method == http.MethodPost:
		// Stream's ids are unique across every app, which is what lets a test tell one app's
		// trunk from another's.
		id := "trunk-" + unique()
		db.trunks[id] = body
		result["sip_trunk"] = map[string]any{"id": id, "uri": "sip:" + id + "@sip.example.test", "username": id, "password": "secret"}
	case strings.HasSuffix(r.URL.Path, "/sip/inbound_routing_rules") && r.Method == http.MethodPost:
		id := "rule-" + unique()
		db.rules[id] = body
		result["id"] = id
	case strings.Contains(r.URL.Path, "/sip/inbound_trunks/") && r.Method == http.MethodDelete:
		delete(db.trunks, parts[len(parts)-1])
	case strings.Contains(r.URL.Path, "/sip/inbound_routing_rules/") && r.Method == http.MethodDelete:
		delete(db.rules, parts[len(parts)-1])
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
	case r.Method == http.MethodPatch && len(parts) >= 2 && parts[len(parts)-2] == "agent":
		// A partial update sets the fields it names on the channel.
		id := parts[len(parts)-1]
		data, exists := db.channels[id]
		if !exists {
			refuse(w, "channel "+id+" does not exist")
			return
		}
		set, _ := body["set"].(map[string]any)
		maps.Copy(data, set)
		result["channel"] = data
	case r.Method == http.MethodPost && len(parts) >= 2 && parts[len(parts)-2] == "agent":
		// An update adds the members it names to the channel, which is how a reader comes
		// to be able to watch it.
		id := parts[len(parts)-1]
		data, exists := db.channels[id]
		if !exists {
			refuse(w, "channel "+id+" does not exist")
			return
		}
		added, _ := body["add_members"].([]any)
		members, _ := data["members"].([]any)
		data["members"] = append(members, added...)
		result["channel"] = data
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
		// Chat answers with the creator it resolved created_by_id to, as it does a message's
		// author.
		if creator, ok := db.channels[id]["created_by_id"].(string); ok {
			channel := maps.Clone(db.channels[id])
			channel["created_by"] = map[string]any{"id": creator}
			result["channel"] = channel
		}
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
