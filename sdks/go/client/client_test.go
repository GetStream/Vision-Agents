package client

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gorilla/websocket"

	"github.com/GetStream/Vision-Agents/sdks/go/acceleration"
	"github.com/GetStream/Vision-Agents/sdks/go/stream"
	"github.com/GetStream/Vision-Agents/sdks/go/tools"
)

// router is a stand-in for the acceleration backend: the resource endpoints this package
// talks to, and the socket a session is watched on. A real HTTP server with a real WebSocket
// upgrader, so what is under test is the exchange rather than a description of it.
type router struct {
	*httptest.Server

	mu sync.Mutex
	// asked is every request that arrived, as "METHOD /path?query", in order.
	asked []string
	// bodies are the decoded JSON bodies, keyed by "METHOD /path".
	bodies map[string]map[string]any

	// configs are the stored agent configs a name is resolved against.
	configs []acceleration.AgentConfig
	// sessions is what a list or a search answers with.
	sessions []acceleration.Session
	// items is every page of response items, each handed out for the cursor naming it.
	items [][]acceleration.AgentResponseItem
}

func newRouter(t *testing.T) *router {
	t.Helper()

	backend := &router{bodies: map[string]map[string]any{}}
	mux := http.NewServeMux()

	mux.HandleFunc("GET /v1/agents/configs", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		backend.mu.Lock()
		stored := backend.configs
		backend.mu.Unlock()
		if stored == nil {
			stored = []acceleration.AgentConfig{}
		}
		answer(w, http.StatusOK, stored)
	})

	mux.HandleFunc("PATCH /v1/agents/configs/{id}", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		backend.mu.Lock()
		defer backend.mu.Unlock()
		sent := backend.bodies[r.Method+" "+r.URL.Path]
		for i, config := range backend.configs {
			if config.Id != r.PathValue("id") {
				continue
			}
			if guardrail, ok := sent["guardrail"].(string); ok {
				backend.configs[i].Guardrail = &guardrail
			}
			answer(w, http.StatusOK, backend.configs[i])
			return
		}
		answer(w, http.StatusNotFound, acceleration.Error{Error: "no such agent config"})
	})

	mux.HandleFunc("POST /v1/agents/sessions", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusCreated, acceleration.Session{
			Id: "session-1", AgentId: "agent-1", UserId: "jean", State: "running",
			ConversationId: ptr("agent:session-1"), CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("POST /v1/agents/sessions/query", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusOK, acceleration.SessionPage{Items: backend.stored()})
	})

	mux.HandleFunc("POST /v1/agents/sessions/{id}/fork", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusCreated, acceleration.Session{
			Id: "session-2", AgentId: "agent-1", UserId: "jean", State: "running",
			ForkedFrom: ptr(r.PathValue("id")), CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("PATCH /v1/agents/sessions/{id}", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		// A target that does not route is refused, and the session carries on as it was.
		backend.mu.Lock()
		asked := backend.bodies[r.Method+" "+r.URL.Path]["llm"]
		backend.mu.Unlock()
		if asked == "nobody/nothing" {
			answer(w, http.StatusBadRequest, acceleration.Error{Error: "nothing routes to nobody/nothing"})
			return
		}
		answer(w, http.StatusOK, acceleration.Session{
			Id: r.PathValue("id"), AgentId: "agent-1", UserId: "jean", State: "running",
			CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("POST /v1/agents/sessions/{id}/responses", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusAccepted, acceleration.AgentResponse{
			Id: "response-1", SessionId: r.PathValue("id"), Status: "running",
			Said: ptr("Is Stream better?"), CreatedAt: time.Now(),
		})
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/responses", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusOK, acceleration.AgentResponsePage{Items: []acceleration.AgentResponse{{
			Id: "response-1", SessionId: r.PathValue("id"), Status: "completed",
			CreatedAt: time.Now(),
		}}})
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/responses/items", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusOK, backend.page(r.URL.Query().Get("cursor")))
	})

	mux.HandleFunc("POST /v1/agents/sessions/{id}/rewind", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		// A persistent conversation is what the real router refuses to rewind.
		if r.PathValue("id") == "persistent" {
			answer(w, http.StatusBadRequest, acceleration.Error{
				Error: "a persistent conversation keeps its transcript in Chat"})
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("DELETE /v1/agents/sessions/{id}", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		if r.PathValue("id") == "someone-elses" {
			answer(w, http.StatusNotFound, acceleration.Error{Error: "unknown session"})
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("DELETE /v1/agents/sessions/{id}/memories", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		if r.PathValue("id") == "someone-elses" {
			answer(w, http.StatusNotFound, acceleration.Error{Error: "unknown session"})
			return
		}
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("DELETE /v1/agents/users/{user_id}/memories", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		w.WriteHeader(http.StatusNoContent)
	})

	mux.HandleFunc("POST /v1/agents/guests", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusCreated, acceleration.GuestUser{
			Id: "guest-1", Token: "guest-token", Name: ptr("Guest"),
		})
	})

	mux.HandleFunc("POST /v1/agents/guests/claim", func(w http.ResponseWriter, r *http.Request) {
		backend.record(r)
		answer(w, http.StatusOK, acceleration.ClaimGuestResult{
			GuestId: "guest-1", UserId: "jean", SessionsMoved: 3,
		})
	})

	mux.HandleFunc("GET /v1/agents/sessions/{id}/events", func(w http.ResponseWriter, r *http.Request) {
		connection, err := (&websocket.Upgrader{}).Upgrade(w, r, nil)
		if err != nil {
			return
		}
		defer connection.Close()
		for {
			if _, _, err := connection.ReadMessage(); err != nil {
				return
			}
		}
	})

	backend.Server = httptest.NewServer(mux)
	t.Cleanup(backend.Close)
	return backend
}

// client is a client pointed at this router, acting for the app itself.
func (r *router) client(t *testing.T) *Client {
	t.Helper()
	api, err := New(stream.Backend{URL: r.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}
	return api
}

func (r *router) record(request *http.Request) {
	r.mu.Lock()
	defer r.mu.Unlock()

	line := request.Method + " " + request.URL.Path
	if query := request.URL.RawQuery; query != "" {
		r.asked = append(r.asked, line+"?"+query)
	} else {
		r.asked = append(r.asked, line)
	}

	if request.Body != nil {
		var body map[string]any
		if err := json.NewDecoder(request.Body).Decode(&body); err == nil {
			r.bodies[line] = body
		}
	}
}

func (r *router) stored() []acceleration.Session {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.sessions == nil {
		return []acceleration.Session{}
	}
	return r.sessions
}

// page hands out the prepared page a cursor names, with the cursor to the one after it.
// No cursor is the first page.
func (r *router) page(cursor string) acceleration.AgentResponseItemPage {
	r.mu.Lock()
	defer r.mu.Unlock()
	at := 0
	if cursor != "" {
		fmt.Sscanf(cursor, "page-%d", &at)
	}
	if at >= len(r.items) {
		return acceleration.AgentResponseItemPage{Items: []acceleration.AgentResponseItem{}}
	}
	page := acceleration.AgentResponseItemPage{Items: r.items[at]}
	if at+1 < len(r.items) {
		page.HasMore = true
		page.NextCursor = ptr(fmt.Sprintf("page-%d", at+1))
	}
	return page
}

// query is the query string of the one request to a path, failing when none arrived.
func (r *router) query(t *testing.T, method, path string) url.Values {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()

	for _, line := range r.asked {
		before, after, split := strings.Cut(line, "?")
		if before != method+" "+path {
			continue
		}
		if !split {
			return url.Values{}
		}
		values, err := url.ParseQuery(after)
		if err != nil {
			t.Fatal(err)
		}
		return values
	}
	t.Fatalf("nothing was asked of %s %s; got %v", method, path, r.asked)
	return nil
}

// body is the JSON body sent to a path.
func (r *router) body(t *testing.T, method, path string) map[string]any {
	t.Helper()
	r.mu.Lock()
	defer r.mu.Unlock()

	body, sent := r.bodies[method+" "+path]
	if !sent {
		t.Fatalf("nothing was posted to %s %s", method, path)
	}
	return body
}

// requests counts how many times a path was asked for, whatever the query.
func (r *router) requests(method, path string) int {
	r.mu.Lock()
	defer r.mu.Unlock()

	counted := 0
	for _, line := range r.asked {
		if before, _, _ := strings.Cut(line, "?"); before == method+" "+path {
			counted++
		}
	}
	return counted
}

func TestASessionIsOpenedAgainstTheAgentByName(t *testing.T) {
	backend := newRouter(t)
	agent := backend.client(t).Agent("docs")

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{
		Title:       "Is Stream better?",
		Description: "The comparison question, again",
		ProjectID:   "docs",
		Custom:      map[string]any{"ticket": "4721"},
		ModelOverwrites: &acceleration.ModelOverwrites{
			Thinking: thinking("high"),
		},
		ConnectorBindings: []acceleration.SessionConnectorBinding{{
			Name: "crm", ConnectionId: "conn_alice",
		}},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())

	body := backend.body(t, "POST", "/v1/agents/sessions")
	if body["agent"] != "docs" {
		t.Errorf("the session was opened against %v", body["agent"])
	}
	if body["title"] != "Is Stream better?" || body["project_id"] != "docs" {
		t.Errorf("the labels went over as %v", body)
	}
	if body["text"] != true {
		t.Error("a session with no call should be held in writing")
	}
	overwrites, _ := body["model_overwrites"].(map[string]any)
	if overwrites["thinking"] != "high" {
		t.Errorf("the model overwrites went over as %v", body["model_overwrites"])
	}
	if got := body["connector_bindings"]; !reflect.DeepEqual(got, []any{
		map[string]any{"name": "crm", "connection_id": "conn_alice"},
	}) {
		t.Errorf("the selected connector accounts went over as %v", got)
	}
	if session.ID() != "session-1" {
		t.Errorf("the session is %q", session.ID())
	}
}

func TestAnIncognitoSessionAsksToKeepNothing(t *testing.T) {
	backend := newRouter(t)
	agent := backend.client(t).Agent("docs")

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{Incognito: true})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())

	body := backend.body(t, "POST", "/v1/agents/sessions")
	if body["incognito"] != true {
		t.Error("incognito was not asked for")
	}
}

func TestQueryingNarrowsToTheAgentAndTheFiltersGiven(t *testing.T) {
	backend := newRouter(t)
	backend.sessions = []acceleration.Session{{Id: "session-1", State: "closed"}}

	listed, err := backend.client(t).Agent("docs").Sessions.Query(t.Context(), Query{
		ProjectID: "docs",
		UserID:    "jean",
		Modality:  "video",
		Limit:     50,
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(listed.Items) != 1 || listed.Items[0].Id != "session-1" {
		t.Fatalf("the list came back as %v", listed)
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/query")
	filter, _ := body["filter"].(map[string]any)
	for field, want := range map[string]string{
		"agent":      "docs",
		"project_id": "docs",
		"user_id":    "jean",
		"modality":   "video",
	} {
		if got := filter[field]; got != want {
			t.Errorf("%s went over as %v rather than %q", field, got, want)
		}
	}
	if _, sent := filter["text"]; sent {
		t.Error("a listing was sent as a search")
	}
	if body["limit"] != float64(50) {
		t.Errorf("the limit went over as %v", body["limit"])
	}
	// The first page is asked for without a cursor rather than with an empty one.
	if _, sent := body["cursor"]; sent {
		t.Error("a cursor nobody asked for was sent")
	}
}

func TestQueryingTheNextPageSendsTheCursor(t *testing.T) {
	backend := newRouter(t)

	if _, err := backend.client(t).Agent("docs").Sessions.Query(t.Context(), Query{
		Cursor: "next-page",
	}); err != nil {
		t.Fatal(err)
	}

	if got := backend.body(t, "POST", "/v1/agents/sessions/query")["cursor"]; got != "next-page" {
		t.Errorf("the cursor went over as %v", got)
	}
}

func TestSearchingCarriesThePhraseAlongsideTheFilters(t *testing.T) {
	backend := newRouter(t)
	backend.sessions = []acceleration.Session{{Id: "session-1", State: "closed"}}

	if _, err := backend.client(t).Agent("docs").Sessions.Search(t.Context(),
		"sendbird comparison", Query{UserID: "jean"}); err != nil {
		t.Fatal(err)
	}

	filter, _ := backend.body(t, "POST", "/v1/agents/sessions/query")["filter"].(map[string]any)
	text, _ := filter["text"].(map[string]any)
	if text["$q"] != "sendbird comparison" {
		t.Errorf("the phrase went over as %v", filter["text"])
	}
	if filter["agent"] != "docs" || filter["user_id"] != "jean" {
		t.Errorf("the filters went over as %v", filter)
	}
}

func TestAskingSomethingNamesTheTurnItIsAnsweredAs(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	answer, err := session.Responses.Create(t.Context(), "Is Stream better than Sendbird?")
	if err != nil {
		t.Fatal(err)
	}
	if answer.ID() != "response-1" {
		t.Errorf("the turn is %q", answer.ID())
	}
	if answer.Status() != "running" {
		t.Errorf("the turn is %q", answer.Status())
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/responses")
	if body["text"] != "Is Stream better than Sendbird?" {
		t.Errorf("the question went over as %v", body["text"])
	}
	if id, _ := body["command_id"].(string); id == "" {
		t.Error("a question in a stored conversation is a command the router answers at most once")
	}
}

func TestImagesAndClipsGoOverWithTheQuestion(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	if _, err := session.Responses.Create(t.Context(), "Is the item damaged?",
		Image{URL: "https://example.com/receipt.png", Detail: acceleration.ImageSourceDetailHigh},
		Clip{URL: "https://example.com/unboxing.mp4", MaxFrames: 12},
		Clip{URL: "https://example.com/return.mp4"},
	); err != nil {
		t.Fatal(err)
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/responses")
	images, _ := json.Marshal(body["images"])
	if string(images) != `[{"detail":"high","url":"https://example.com/receipt.png"}]` {
		t.Errorf("the images went over as %s", images)
	}
	videos, _ := json.Marshal(body["videos"])
	if string(videos) != `[{"max_frames":12,"url":"https://example.com/unboxing.mp4"},{"url":"https://example.com/return.mp4"}]` {
		t.Errorf("the videos went over as %s", videos)
	}
	if _, carried := body["command_id"]; carried {
		t.Error("a command carries text only")
	}
}

func TestAClipOnDiskIsSentInline(t *testing.T) {
	for name, kind := range map[string]string{"unboxing.webm": "video/webm", "unboxing": "video/mp4"} {
		path := filepath.Join(t.TempDir(), name)
		if err := os.WriteFile(path, []byte("clip"), 0o600); err != nil {
			t.Fatal(err)
		}

		clip, err := ClipFile(path)
		if err != nil {
			t.Fatal(err)
		}
		if want := "data:" + kind + ";base64,Y2xpcA=="; clip.URL != want {
			t.Errorf("%s went over as %q, want %q", name, clip.URL, want)
		}
	}
}

func TestATurnsItemsAreNarrowedToThatTurn(t *testing.T) {
	backend := newRouter(t)
	backend.items = [][]acceleration.AgentResponseItem{{
		{ResponseId: "response-1", Ordinal: 0, Kind: "said", Text: ptr("Is Stream better?")},
	}}
	session := open(t, backend)

	answer, err := session.Responses.Create(t.Context(), "Is Stream better?")
	if err != nil {
		t.Fatal(err)
	}
	items, err := answer.Items.All(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if len(items) != 1 || items[0].Kind != "said" {
		t.Fatalf("the items came back as %v", items)
	}

	query := backend.query(t, "GET", "/v1/agents/sessions/session-1/responses/items")
	if query.Get("response_id") != "response-1" {
		t.Errorf("the items were asked for with response_id %q", query.Get("response_id"))
	}
}

func TestTheSessionsItemsAreEveryTurnsRatherThanOnes(t *testing.T) {
	backend := newRouter(t)
	backend.items = [][]acceleration.AgentResponseItem{{{ResponseId: "response-1", Kind: "said"}}}
	session := open(t, backend)

	if _, err := session.Responses.Items.All(t.Context()); err != nil {
		t.Fatal(err)
	}

	query := backend.query(t, "GET", "/v1/agents/sessions/session-1/responses/items")
	if _, narrowed := query["response_id"]; narrowed {
		t.Error("the session's own items were narrowed to one turn")
	}
}

func TestUnwindingFollowsTheCursorUntilTheLastPage(t *testing.T) {
	backend := newRouter(t)
	// The last page says there is no more, so a fourth request would mean the stream is
	// asking for a page it has already been told does not exist.
	backend.items = [][]acceleration.AgentResponseItem{
		items("response-1", 0, 2), items("response-1", 2, 2), items("response-1", 4, 1),
	}
	session := open(t, backend)

	stream := session.Responses.Items.Unwind(t.Context(), 2)
	read := 0
	for item := range stream.Items() {
		if item.Ordinal != read {
			t.Errorf("item %d arrived at ordinal %d", read, item.Ordinal)
		}
		read++
	}
	if err := stream.Err(); err != nil {
		t.Fatal(err)
	}

	if read != 5 {
		t.Errorf("%d items were unwound", read)
	}
	if asked := backend.requests("GET", "/v1/agents/sessions/session-1/responses/items"); asked != 3 {
		t.Errorf("the items were asked for %d times", asked)
	}
}

func TestUnwindingReportsWhyItStoppedRatherThanLookingLikeTheEnd(t *testing.T) {
	// A router that refuses, so the stream closes without having run out. A closed channel
	// on its own cannot tell the two apart, which is what Err is for.
	refusing := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		answer(w, http.StatusNotFound, acceleration.Error{Error: "no such session"})
	}))
	t.Cleanup(refusing.Close)

	api, err := New(stream.Backend{URL: refusing.URL, CustomerID: "acme"})
	if err != nil {
		t.Fatal(err)
	}

	stream := api.Agent("docs").Sessions.Responses("session-1").Items.Unwind(t.Context(), 0)
	for range stream.Items() {
		t.Error("an item arrived from a router that refused")
	}
	if stream.Err() == nil {
		t.Fatal("the stream ended as though it had run out")
	}
}

func TestRewindingCarriesOnFromTheResponseGiven(t *testing.T) {
	// A call, because a text conversation is kept in Chat and the router refuses to rewind
	// it: forking it at the response is what a text conversation does instead.
	backend := newRouter(t)
	session := onCall(t, backend)

	if err := session.Responses.Rewind(t.Context(), "response-1"); err != nil {
		t.Fatal(err)
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/rewind")
	if body["response_id"] != "response-1" {
		t.Errorf("the rewind went over as %v", body)
	}
}

func TestARewindTheRouterRefusesSaysWhy(t *testing.T) {
	backend := newRouter(t)
	responses := backend.client(t).Agent("docs").Sessions.Responses("persistent")

	err := responses.Rewind(t.Context(), "response-1")
	if err == nil || !strings.Contains(err.Error(), "transcript in Chat") {
		t.Fatalf("the refusal came back as %v", err)
	}
}

func TestAResponseThatWasNeverRecordedCannotBeRewoundTo(t *testing.T) {
	backend := newRouter(t)
	session := onCall(t, backend)

	if err := session.Responses.Rewind(t.Context(), ""); err == nil {
		t.Fatal("a response with no id was rewound to")
	}
	if asked := backend.requests("POST", "/v1/agents/sessions/session-1/rewind"); asked != 0 {
		t.Errorf("the router was asked %d times", asked)
	}
}

func TestDeletingASessionsMemoriesAsksForThatSessionsAlone(t *testing.T) {
	backend := newRouter(t)
	session := onCall(t, backend)

	if err := session.DeleteMemories(t.Context()); err != nil {
		t.Fatal(err)
	}
	if asked := backend.requests("DELETE", "/v1/agents/sessions/session-1/memories"); asked != 1 {
		t.Errorf("the router was asked %d times", asked)
	}
	if asked := backend.requests("DELETE", "/v1/agents/sessions/session-1"); asked != 0 {
		t.Error("deleting what a session remembered must not end it")
	}
}

func TestDeletingASessionAsksTheRouterToDeleteIt(t *testing.T) {
	backend := newRouter(t)
	session := onCall(t, backend)

	if err := session.Delete(t.Context()); err != nil {
		t.Fatal(err)
	}
	if asked := backend.requests("DELETE", "/v1/agents/sessions/session-1"); asked != 1 {
		t.Errorf("the router was asked %d times", asked)
	}
}

func TestAnotherCustomersSessionIsNotDeleted(t *testing.T) {
	backend := newRouter(t)

	err := backend.client(t).Agent("docs").Sessions.Delete(t.Context(), "someone-elses")
	if err == nil || !strings.Contains(err.Error(), "unknown session") {
		t.Fatalf("the refusal came back as %v", err)
	}
}

func TestAnotherCustomersSessionsMemoriesAreRefused(t *testing.T) {
	backend := newRouter(t)

	err := backend.client(t).Agent("docs").Sessions.DeleteMemories(t.Context(), "someone-elses")
	if err == nil || !strings.Contains(err.Error(), "unknown session") {
		t.Fatalf("the refusal came back as %v", err)
	}
}

func TestTruncatingDeletesEverythingAboutOneUser(t *testing.T) {
	backend := newRouter(t)

	if err := backend.client(t).Memories().Truncate(t.Context(), "user 123"); err != nil {
		t.Fatal(err)
	}
	if asked := backend.requests("DELETE", "/v1/agents/users/user 123/memories"); asked != 1 {
		t.Errorf("the router was asked %v", backend.asked)
	}
}

func TestTruncatingNobodyIsRefusedBeforeTheRouter(t *testing.T) {
	backend := newRouter(t)

	if err := backend.client(t).Memories().Truncate(t.Context(), ""); err == nil {
		t.Fatal("an empty user id was truncated")
	}
	if len(backend.asked) != 0 {
		t.Errorf("the router was asked %v", backend.asked)
	}
}

func TestUpdatingASessionPatchesOnlyWhatWasNamed(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	thinking := acceleration.UpdateSessionRequestThinkingHigh
	updated, err := session.Update(t.Context(), SessionUpdate{
		Title:    ptr("Pricing"),
		Llm:      ptr("openai/gpt-5"),
		Voice:    ptr(""),
		Thinking: &thinking,
	})
	if err != nil {
		t.Fatal(err)
	}
	if updated.Id != "session-1" {
		t.Errorf("the router answered for %q", updated.Id)
	}

	if asked := backend.requests("PATCH", "/v1/agents/sessions/session-1"); asked != 1 {
		t.Fatalf("the session was patched %d times", asked)
	}
	body := backend.body(t, "PATCH", "/v1/agents/sessions/session-1")
	want := map[string]any{"title": "Pricing", "llm": "openai/gpt-5", "voice": "", "thinking": "high"}
	if len(body) != len(want) {
		t.Errorf("the update went over as %v, want %v", body, want)
	}
	for key, value := range want {
		if body[key] != value {
			t.Errorf("%s went over as %v, want %v", key, body[key], value)
		}
	}
}

func TestAConversationNotBeingHeldCanStillBeRenamed(t *testing.T) {
	backend := newRouter(t)

	_, err := backend.client(t).Agent("docs").Sessions.Update(t.Context(), "ended-1",
		SessionUpdate{Title: ptr("Pricing")})
	if err != nil {
		t.Fatal(err)
	}
	if body := backend.body(t, "PATCH", "/v1/agents/sessions/ended-1"); body["title"] != "Pricing" {
		t.Errorf("the rename went over as %v", body)
	}
}

func TestAnUpdateTheRouterRefusesSaysWhy(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	_, err := session.Update(t.Context(), SessionUpdate{Llm: ptr("nobody/nothing")})
	if err == nil || !strings.Contains(err.Error(), "nothing routes to nobody/nothing") {
		t.Fatalf("the refusal came back as %v", err)
	}
}

func TestForkingAtAResponseBranchesFromThere(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	forked, err := session.Fork(t.Context(), ForkOptions{ResponseID: "response-1"})
	if err != nil {
		t.Fatal(err)
	}
	defer forked.Close(t.Context())

	if body := backend.body(t, "POST", "/v1/agents/sessions/session-1/fork"); body["response_id"] != "response-1" {
		t.Errorf("response_id went over as %v", body["response_id"])
	}
}

func TestForkingContinuesTheConversationAsANewOne(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	forked, err := session.Fork(t.Context(), ForkOptions{
		Title:           "With a harder model",
		ModelOverwrites: &acceleration.ModelOverwrites{Llm: ptr("openai/gpt-5")},
	})
	if err != nil {
		t.Fatal(err)
	}
	defer forked.Close(t.Context())

	if forked.ID() != "session-2" {
		t.Errorf("the fork is %q", forked.ID())
	}
	if forked.Responses.Items == session.Responses.Items {
		t.Error("the fork is reading the parent's items")
	}

	body := backend.body(t, "POST", "/v1/agents/sessions/session-1/fork")
	if body["title"] != "With a harder model" {
		t.Errorf("the fork's title went over as %v", body["title"])
	}
	// The history comes across by default, so an absent field and a true one mean the same
	// thing and only the refusal is worth sending.
	if _, sent := body["messages"]; sent {
		t.Error("messages was sent for a fork that wanted the history")
	}
}

func TestForkingWithoutMessagesSaysSo(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	forked, err := session.Fork(t.Context(), ForkOptions{WithoutMessages: true})
	if err != nil {
		t.Fatal(err)
	}
	defer forked.Close(t.Context())

	if body := backend.body(t, "POST", "/v1/agents/sessions/session-1/fork"); body["messages"] != false {
		t.Errorf("messages went over as %v", body["messages"])
	}
}

func TestAForkInheritsTheParentsFunctions(t *testing.T) {
	backend := newRouter(t)
	agent := backend.client(t).Agent("docs")
	if err := register(agent.Tools(), "get_weather"); err != nil {
		t.Fatal(err)
	}

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())

	forked, err := session.Fork(t.Context(), ForkOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer forked.Close(t.Context())

	registered := forked.Tools().List()
	if len(registered) != 1 || registered[0].Name != "get_weather" {
		t.Errorf("the fork offers %v", registered)
	}
}

func TestTheModelIsOfferedTheAgentsFunctions(t *testing.T) {
	backend := newRouter(t)
	agent := backend.client(t).Agent("docs")
	if err := register(agent.Tools(), "get_weather"); err != nil {
		t.Fatal(err)
	}

	session, err := agent.Sessions.Create(t.Context(), SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(t.Context())

	declared, _ := backend.body(t, "POST", "/v1/agents/sessions")["tools"].([]any)
	if len(declared) != 1 {
		t.Fatalf("the session declared %v", declared)
	}
	if tool, _ := declared[0].(map[string]any); tool["name"] != "get_weather" {
		t.Errorf("the declared tool is %v", declared[0])
	}
}

func TestAnAgentIsLookedUpByTheNameItIsConfiguredUnder(t *testing.T) {
	backend := newRouter(t)
	backend.configs = []acceleration.AgentConfig{{Id: "config-1", Name: "docs"}}

	config, err := backend.client(t).Agent("docs").Config(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if config == nil || config.Id != "config-1" {
		t.Fatalf("the config came back as %v", config)
	}
	if name := backend.query(t, "GET", "/v1/agents/configs").Get("name"); name != "docs" {
		t.Errorf("the config was asked for by %q", name)
	}
}

func TestUpdatingAConfigSendsOnlyWhatChanged(t *testing.T) {
	backend := newRouter(t)
	instructions := "be brief"
	backend.configs = []acceleration.AgentConfig{{Id: "config-1", Name: "docs", Instructions: &instructions}}
	policy := "Only answer questions about Acme."

	config, err := backend.client(t).Agent("docs").UpdateConfig(t.Context(),
		acceleration.AgentConfigPatch{Guardrail: &policy})
	if err != nil {
		t.Fatal(err)
	}
	if config.Guardrail == nil || *config.Guardrail != policy {
		t.Errorf("the guardrail came back as %v", config.Guardrail)
	}
	if config.Instructions == nil || *config.Instructions != instructions {
		t.Errorf("the instructions came back as %v", config.Instructions)
	}
	sent := backend.body(t, "PATCH", "/v1/agents/configs/config-1")
	if len(sent) != 1 {
		t.Errorf("the patch sent %v rather than only the guardrail", sent)
	}
}

func TestUpdatingAnAgentNothingIsStoredUnderFails(t *testing.T) {
	backend := newRouter(t)
	policy := "Only answer questions about Acme."

	_, err := backend.client(t).Agent("nowhere").UpdateConfig(t.Context(),
		acceleration.AgentConfigPatch{Guardrail: &policy})
	if err == nil || !strings.Contains(err.Error(), "nowhere") {
		t.Errorf("updating an agent nobody stored answered %v", err)
	}
}

func TestANameNothingIsStoredUnderIsNoConfigRatherThanAFailure(t *testing.T) {
	backend := newRouter(t)

	config, err := backend.client(t).Agent("nowhere").Config(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	if config != nil {
		t.Errorf("a name nothing matches returned %v", config)
	}
}

func TestAGuestIsMintedWithATokenToHold(t *testing.T) {
	backend := newRouter(t)

	guest, err := backend.client(t).GuestUser(t.Context(), GuestOptions{Name: "Guest"})
	if err != nil {
		t.Fatal(err)
	}
	if guest.Id != "guest-1" || guest.Token != "guest-token" {
		t.Fatalf("the guest came back as %v", guest)
	}
	if name := backend.body(t, "POST", "/v1/agents/guests")["name"]; name != "Guest" {
		t.Errorf("the guest was minted as %v", name)
	}
}

func TestActingAsAGuestIsTheGuestsToken(t *testing.T) {
	backend := newRouter(t)
	// A guest holds a token rather than a customer id, so the backend it acts through has to
	// be one that sends tokens at all.
	api, err := New(stream.Backend{URL: backend.URL, Authenticate: true, APIKey: "key", APISecret: "secret"})
	if err != nil {
		t.Fatal(err)
	}

	as, err := api.AsGuest(&Guest{Id: "guest-1", Token: "guest-token"})
	if err != nil {
		t.Fatal(err)
	}
	if as.Backend().UserID != "guest-1" || as.Backend().Token != "guest-token" {
		t.Errorf("the guest client is acting as %v", as.Backend().UserID)
	}
	if as.ServerSide() {
		t.Error("a client acting for a guest is not the app itself")
	}
	if !api.ServerSide() {
		t.Error("the client the guest came from stopped being the app itself")
	}
}

func TestClaimingAGuestMovesTheirConversations(t *testing.T) {
	backend := newRouter(t)

	claimed, err := backend.client(t).ClaimGuestUser(t.Context(), "guest-1", "jean")
	if err != nil {
		t.Fatal(err)
	}
	if claimed.SessionsMoved != 3 {
		t.Errorf("%d conversations moved", claimed.SessionsMoved)
	}
	body := backend.body(t, "POST", "/v1/agents/guests/claim")
	if body["guest_id"] != "guest-1" || body["user_id"] != "jean" {
		t.Errorf("the claim went over as %v", body)
	}
}

func TestClaimingNeedsBothTheGuestAndTheAccount(t *testing.T) {
	backend := newRouter(t)

	if _, err := backend.client(t).ClaimGuestUser(t.Context(), "guest-1", ""); err == nil {
		t.Error("a claim with nobody to claim for was allowed")
	}
	if backend.requests("POST", "/v1/agents/guests/claim") != 0 {
		t.Error("the router was asked to claim a guest for nobody")
	}
}

func TestSetUserRefusesAUserWithNothingToProveIt(t *testing.T) {
	backend := newRouter(t)
	api := backend.client(t)

	if _, err := api.SetUser("", "token"); err == nil {
		t.Error("a user with no id was allowed")
	}
	if _, err := api.SetUser("jean", ""); err == nil {
		t.Error("a user with no token was allowed")
	}
}

func TestChatIsRefusedForAConversationThatKeepsNoTranscript(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)
	// What the router said it opened is what the session reports, so this stands in for an
	// incognito conversation: neither has a channel to read.
	session.created.ConversationId = nil

	if _, err := session.Chat(); err == nil {
		t.Error("a session with no transcript handed back a channel")
	}
}

func TestVideoIsRefusedForAConversationHeldInWriting(t *testing.T) {
	backend := newRouter(t)
	session := open(t, backend)

	if _, err := session.Video(); err == nil {
		t.Error("a written conversation handed back a call")
	}
}

// open is a session on the router, closed when the test ends.
func open(t *testing.T, backend *router) *Session {
	t.Helper()
	session, err := backend.client(t).Agent("docs").Sessions.Create(t.Context(), SessionOptions{})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(t.Context()) })
	return session
}

// onCall is a session opened on a call rather than held in writing.
func onCall(t *testing.T, backend *router) *Session {
	t.Helper()
	session, err := backend.client(t).Agent("docs").Sessions.Create(t.Context(), SessionOptions{CallID: "call-1"})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = session.Close(t.Context()) })
	return session
}

// items is a page of response items numbered from an ordinal.
func items(response string, from, count int) []acceleration.AgentResponseItem {
	page := make([]acceleration.AgentResponseItem, 0, count)
	for at := range count {
		page = append(page, acceleration.AgentResponseItem{
			ResponseId: response, Ordinal: from + at, Kind: "response_text",
			Text: ptr(fmt.Sprintf("part %d", from+at)),
		})
	}
	return page
}

func answer(w http.ResponseWriter, status int, body any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(body)
}

func ptr[T any](value T) *T { return &value }

func thinking(effort string) *acceleration.ModelOverwritesThinking {
	value := acceleration.ModelOverwritesThinking(effort)
	return &value
}

// register adds one function, so a test about what a conversation offers the model does not
// have to spell out a schema to make the point.
func register(registry *tools.Registry, name string) error {
	return tools.Register(registry, name, "What the weather is",
		func(_ context.Context, in struct {
			Location string `json:"location" schema:"the city"`
		}) (any, error) {
			return "raining in " + in.Location, nil
		})
}
