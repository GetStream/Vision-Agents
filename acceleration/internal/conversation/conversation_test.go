package conversation

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/stretchr/testify/require"
)

type chatStore struct {
	mu              sync.Mutex
	channels        map[string]map[string]any
	messages        map[string]map[string]any
	order           []string
	patches         []map[string]any
	fail            bool
	failAfterCreate bool
}

func newChat(t *testing.T) (*chatStore, *getstream.Stream) {
	t.Helper()
	db := &chatStore{channels: map[string]map[string]any{}, messages: map[string]map[string]any{}}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		db.mu.Lock()
		defer db.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if db.fail {
			w.WriteHeader(503)
			_, _ = w.Write([]byte(`{"code":1,"message":"offline"}`))
			return
		}
		var body map[string]any
		if r.Body != nil {
			_ = json.NewDecoder(r.Body).Decode(&body)
		}
		parts := strings.Split(r.URL.Path, "/")
		result := map[string]any{}
		switch {
		case r.Method == http.MethodPatch && len(parts) > 2 && parts[len(parts)-2] == "agent":
			id := parts[len(parts)-1]
			for k, v := range body["set"].(map[string]any) {
				db.channels[id][k] = v
			}
			result["channel"] = db.channels[id]
		case strings.HasSuffix(r.URL.Path, "/query"):
			id := parts[len(parts)-2]
			if data, ok := body["data"].(map[string]any); ok {
				db.channels[id] = data
			}
			result["channel"] = db.channels[id]
			result["members"] = db.channels[id]["members"]
			before := ""
			if p, ok := body["messages"].(map[string]any); ok {
				before, _ = p["id_lt"].(string)
			}
			var all []map[string]any
			for _, mid := range db.order {
				if mid == before {
					break
				}
				if db.messages[mid]["cid"] == "agent:"+id {
					all = append(all, db.messages[mid])
				}
			}
			if len(all) > 100 {
				all = all[len(all)-100:]
			}
			result["messages"] = all
		case strings.HasSuffix(r.URL.Path, "/message"):
			m := body["message"].(map[string]any)
			id := m["id"].(string)
			if _, exists := db.messages[id]; !exists {
				db.order = append(db.order, id)
				m["cid"] = "agent:" + parts[len(parts)-2]
				db.messages[id] = m
			}
			result["message"] = db.messages[id]
			if db.failAfterCreate {
				db.failAfterCreate = false
				w.WriteHeader(503)
				_, _ = w.Write([]byte(`{"code":1,"message":"response lost"}`))
				return
			}
		case strings.Contains(r.URL.Path, "/messages/"):
			id := parts[len(parts)-1]
			if id == "ephemeral" {
				id = parts[len(parts)-2]
				db.patches = append(db.patches, body["set"].(map[string]any))
			} else if r.Method == "PUT" {
				for k, v := range body["set"].(map[string]any) {
					if k == "text" || k == "attachments" {
						db.messages[id][k] = v
					} else {
						db.messages[id]["custom"].(map[string]any)[k] = v
					}
				}
			}
			result["message"] = db.messages[id]
		}
		_ = json.NewEncoder(w).Encode(result)
	}))
	t.Cleanup(server.Close)
	client, err := getstream.NewClient("test", "secret", getstream.WithBaseUrl(server.URL))
	require.NoError(t, err)
	return db, client
}
func current(c *Conversation) Message {
	c.mu.Lock()
	defer c.mu.Unlock()
	m := *c.data.Current
	m.Tools = append([]Tool{}, m.Tools...)
	return m
}
func saved(t *testing.T, c *Conversation) {
	t.Helper()
	require.Eventually(t, func() bool { return current(c).Saved }, 8*time.Second, 20*time.Millisecond)
}

// TestAConversationIsResumedWithoutKnowingWhichAgentOpenedIt covers the resume a page does.
//
// A text session is given a fresh agent id on every open, so somebody coming back to a
// conversation cannot present the one its transcript was written under: they never saw it,
// and the id they arrive with is a different one. Presenting none reads the owner off the
// conversation rather than refusing the resume for failing to guess it.
func TestAConversationIsResumedWithoutKnowingWhichAgentOpenedIt(t *testing.T) {
	_, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()

	c, _, _, err := s.Open(context.Background(), "customer", "session-one", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("question"))
	c.Observe(agent.ResponseDelta{Text: "An answer."})
	c.Observe(agent.Responded{})
	saved(t, c)
	cid := c.CID()
	c.Release()

	// The replica that served the first open still holds it.
	again, history, _, err := s.Open(context.Background(), "customer", "", cid)
	require.NoError(t, err)
	require.Equal(t, "session-one", again.Agent())
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "question"},
		{Role: llm.Assistant, Content: "An answer."},
	}, history)
	again.Release()

	// And one that has never seen it, which is every other replica: the owner is read
	// back off the channel instead of out of memory.
	cold, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer cold.Close()
	resumed, _, _, err := cold.Open(context.Background(), "customer", "", cid)
	require.NoError(t, err)
	require.Equal(t, "session-one", resumed.Agent())

	// The customer is still the boundary, with or without an agent id to check.
	_, _, _, err = cold.Open(context.Background(), "another", "", cid)
	require.ErrorContains(t, err, "another customer")
}

func TestActivityPersistsAndRestores(t *testing.T) {
	db, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()
	c, h, tr, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	require.Empty(t, h)
	require.False(t, tr)
	require.NoError(t, c.Begin("question"))
	require.Equal(t, "thinking", current(c).State)
	require.Error(t, c.Begin("overlapping"))
	begin := time.Now().UTC().Add(-time.Second)
	c.Observe(agent.ToolStarted{ID: "one", Tool: "lookup_record", StartedAt: begin})
	require.Equal(t, begin, current(c).Tools[0].StartedAt)
	c.Progress("one", "queued")
	require.Equal(t, "queued", current(c).State)
	require.Nil(t, current(c).Tools[0].ExecutionStartedAt)
	c.Progress("one", "searching")
	execAt := current(c).Tools[0].ExecutionStartedAt
	require.NotNil(t, execAt)
	c.Progress("one", "reading")
	require.Equal(t, execAt, current(c).Tools[0].ExecutionStartedAt)
	c.Observe(agent.ToolStarted{ID: "one", Tool: "lookup_record", StartedAt: time.Now()})
	require.Len(t, current(c).Tools, 1)
	c.Observe(agent.ResponseDelta{Text: "Visible words."})
	require.Eventually(t, func() bool { db.mu.Lock(); defer db.mu.Unlock(); return len(db.patches) > 0 }, 3*time.Second, 20*time.Millisecond)
	db.mu.Lock()
	raw, _ := json.Marshal(db.patches)
	db.mu.Unlock()
	require.Contains(t, string(raw), "Visible words.")
	require.Contains(t, string(raw), "started_at")
	c.Observe(agent.ToolRan{ID: "one", Result: `{"status":"answered","citations":[{},{}],"secret":"not for attachment"}`})
	require.Equal(t, "thinking", current(c).State)
	c.Observe(agent.ToolStarted{ID: "two", Tool: "search_docs", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{ID: "two", Result: `{"status":"unavailable"}`})
	c.Observe(agent.Responded{PendingWork: true})
	require.Nil(t, current(c).FinishedAt)
	c.Observe(agent.ResponseDelta{Text: " Final answer."})
	c.Observe(agent.Responded{})
	saved(t, c)
	want := current(c)
	require.Equal(t, "completed", want.State)
	require.Len(t, want.Tools, 2)
	require.Equal(t, begin, want.Tools[0].StartedAt)
	require.Equal(t, "Verified 2 source citations", want.Tools[0].Summary)
	c.Release()
	_, history, _, err := s.Open(context.Background(), "customer", "support-agent", c.CID())
	require.NoError(t, err)
	require.Equal(t, []llm.Message{{Role: llm.User, Content: "question"}, {Role: llm.Assistant, Content: "Visible words. Final answer."}}, history)
	page, err := s.History(context.Background(), "customer", "support-agent", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	require.Len(t, page.Messages[1].Tools, 2)
	require.Equal(t, "lookup_record", page.Messages[1].Tools[0].Name)
	require.Equal(t, "Verified 2 source citations", page.Messages[1].Tools[0].Summary)
	raw, _ = json.Marshal(page)
	require.NotContains(t, string(raw), "not for attachment")
	_, _, _, err = s.Open(context.Background(), "customer", "support-agent", c.CID())
	require.ErrorContains(t, err, "already open")
	_, _, _, err = s.Open(context.Background(), "another", "support-agent", c.CID())
	require.ErrorContains(t, err, "another customer")
	_, err = s.History(context.Background(), "customer", "wrong-agent", c.CID(), "")
	require.Error(t, err)
	_, err = s.History(context.Background(), "customer", "support-agent", strings.TrimPrefix(c.CID(), "agent:"), "")
	require.Error(t, err)
}

func TestContextForCallerSeedsCompletedTurnsWithoutOpeningTheConversation(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	require.NoError(t, c.Begin("The project name is Nimbus"))
	c.Observe(agent.ResponseDelta{Text: "Noted."})
	c.Observe(agent.Responded{})
	saved(t, c)
	cid := c.CID()
	c.Release()

	messages, truncated, err := service.ContextForCaller(t.Context(), "customer", "agent", cid, "employee")
	require.NoError(t, err)
	require.False(t, truncated)
	require.Equal(t, []llm.Message{
		{Role: llm.User, Content: "The project name is Nimbus"},
		{Role: llm.Assistant, Content: "Noted."},
	}, messages)

	_, _, err = service.ContextForCaller(t.Context(), "customer", "agent", cid, "somebody-else")
	require.ErrorContains(t, err, "another user")
	empty, truncated, err := service.ContextForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	require.False(t, truncated)
	require.Empty(t, empty)
}
func TestOutboxFailureRestartAndDeduplication(t *testing.T) {
	db, client := newChat(t)
	root := t.TempDir()
	s, err := newService(root, client)
	require.NoError(t, err)
	c, _, _, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	db.mu.Lock()
	db.fail = true
	db.mu.Unlock()
	require.NoError(t, c.Begin("persist me"))
	c.Observe(agent.ToolStarted{ID: "tool", Tool: "investigate_sdk", StartedAt: time.Now().UTC()})
	c.Progress("tool", "queued")
	require.Eventually(t, func() bool { return current(c).Error != "" }, 3*time.Second, 20*time.Millisecond)
	require.False(t, current(c).Saved)
	snapshot, err := loadDisk(c.dir())
	require.NoError(t, err)
	require.NotEmpty(t, snapshot.Pending)
	s.Close() // Simulate process stopping without the session's graceful cancellation.
	db.mu.Lock()
	db.fail = false
	db.failAfterCreate = true
	db.mu.Unlock()
	recovered, err := newService(root, client)
	require.NoError(t, err)
	defer recovered.Close()
	resumed := recovered.all[c.CID()]
	saved(t, resumed)
	m := current(resumed)
	require.Equal(t, "interrupted", m.State)
	require.NotNil(t, m.FinishedAt)
	require.NotNil(t, m.Tools[0].FinishedAt)
	require.Equal(t, "cancelled", m.Tools[0].Status)
	db.mu.Lock()
	require.Len(t, db.order, 2)
	db.mu.Unlock()
	snapshot, err = loadDisk(c.dir())
	require.NoError(t, err)
	require.Empty(t, snapshot.Pending)
}
func TestQueuedUserMessageKeepsItsAcceptedAuthorAfterRestart(t *testing.T) {
	for _, legacy := range []bool{false, true} {
		t.Run(fmt.Sprintf("legacy=%t", legacy), func(t *testing.T) {
			db, client := newChat(t)
			root := t.TempDir()
			s, err := newService(root, client)
			require.NoError(t, err)
			t.Cleanup(s.Close)
			c, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "employee-one")
			require.NoError(t, err)
			db.mu.Lock()
			db.fail = true
			db.mu.Unlock()
			receipt, err := c.BeginCommand("accepted-command", "queued question")
			require.NoError(t, err)
			s.Close()
			snapshot, err := loadDisk(c.dir())
			require.NoError(t, err)
			require.NotEmpty(t, snapshot.Pending)
			require.Equal(t, "employee-one", snapshot.Pending[0].Author)
			if legacy {
				// Existing single-owner outboxes must still publish as that owner.
				snapshot.Pending[0].Author = ""
			} else {
				// Changing the session owner must not reattribute accepted writes.
				// This exercises persistence only; it grants no conversation access.
				snapshot.Owner = "employee-two"
			}
			require.NoError(t, writeJSON(filepath.Join(c.dir(), "state.json"), snapshot))
			db.mu.Lock()
			db.fail = false
			db.mu.Unlock()
			recovered, err := newService(root, client)
			require.NoError(t, err)
			defer recovered.Close()
			saved(t, recovered.all[c.CID()])
			db.mu.Lock()
			defer db.mu.Unlock()
			require.Len(t, db.order, 2)
			require.Equal(t, "employee-one", db.messages[receipt.UserMessageID]["user_id"])
			require.Equal(t, "agent", db.messages[receipt.AssistantMessageID]["user_id"])
		})
	}
}

func TestBoundedOrdinaryHistory(t *testing.T) {
	p := Page{Messages: []Message{{Role: "system", Text: "untrusted instruction", State: "completed"}, {Role: "assistant", Text: "unfinished", State: "thinking"}, {Role: "user", Text: strings.Repeat("界", 30000), State: "completed"}, {Role: "assistant", Text: strings.Repeat("界", 30000), State: "completed"}}}
	h, tr := history(p)
	require.Len(t, h, 2)
	require.False(t, tr)
	require.Equal(t, llm.User, h[0].Role)
	p.Messages = append(p.Messages, Message{Role: "assistant", Text: "newest", State: "completed"})
	h, tr = history(p)
	require.True(t, tr)
	require.Equal(t, "newest", h[len(h)-1].Content)
}
func TestCancelAndSkillsStopTimers(t *testing.T) {
	_, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("sales"))
	c.Observe(agent.Delegated{TaskID: "skill", Skill: "chat-sales"})
	require.False(t, current(c).Tools[0].StartedAt.IsZero())
	c.Observe(agent.TaskSettled{TaskID: "skill"})
	require.Equal(t, "thinking", current(c).State)
	c.Observe(agent.ToolStarted{ID: "tool", Tool: "search_docs", StartedAt: time.Now().UTC()})
	c.Cancel()
	before := current(c)
	c.Progress("tool", "reading")
	c.Observe(agent.ResponseDelta{Text: "too late"})
	require.Equal(t, before, current(c))
	require.NotNil(t, before.Tools[1].FinishedAt)
}

func TestLatePreviousTurnDoesNotCancelNextQuestion(t *testing.T) {
	_, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("first"))
	c.Observe(agent.Responding{TurnID: "first-turn"})
	c.Observe(agent.ResponseDelta{TurnID: "first-turn", Text: "first answer"})
	c.Observe(agent.Responded{TurnID: "first-turn"})
	require.Equal(t, "completed", current(c).State)
	require.NoError(t, c.Begin("second"))
	secondID := current(c).ID
	c.Observe(agent.Interrupted{TurnID: "first-turn"})
	c.Observe(agent.ResponseDelta{TurnID: "first-turn", Text: "stale"})
	c.Observe(agent.Responded{TurnID: "first-turn"})
	require.Equal(t, "thinking", current(c).State)
	require.Empty(t, current(c).Text)
	c.Observe(agent.Responding{TurnID: "second-turn"})
	c.Observe(agent.ToolStarted{TurnID: "second-turn", ID: "t", Tool: "search_docs", StartedAt: time.Now().UTC()})
	c.Observe(agent.ToolRan{TurnID: "second-turn", ID: "t"})
	c.Observe(agent.Responding{TurnID: "tool-follow-up"})
	c.Observe(agent.ResponseDelta{TurnID: "tool-follow-up", Text: "answer"})
	c.Observe(agent.Responded{TurnID: "tool-follow-up"})
	require.Equal(t, secondID, current(c).ID)
	require.Equal(t, "answer", current(c).Text)
	require.Equal(t, "completed", current(c).State)
}

func TestHistoryPaginationAndPendingOverlay(t *testing.T) {
	db, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	db.mu.Lock()
	for i := 0; i < 105; i++ {
		id := fmt.Sprintf("message-%03d", i)
		m := Message{ID: id, Role: "user", Text: id, State: "completed", StartedAt: time.Unix(int64(i), 0).UTC()}
		db.order = append(db.order, id)
		db.messages[id] = map[string]any{"id": id, "cid": c.CID(), "text": id, "custom": map[string]any{"support_message": m}}
	}
	db.mu.Unlock()
	page, err := s.History(context.Background(), "customer", "support-agent", c.CID(), "")
	require.NoError(t, err)
	require.Len(t, page.Messages, 100)
	require.Equal(t, "message-005", page.Before)
	old, err := s.History(context.Background(), "customer", "support-agent", c.CID(), page.Before)
	require.NoError(t, err)
	require.Len(t, old.Messages, 5)
	require.Empty(t, old.Before)
	// Pending durable writes appear as unsaved even before the asynchronous sender runs.
	require.NoError(t, c.Begin("local question"))
	page, err = s.History(context.Background(), "customer", "support-agent", c.CID(), "")
	require.NoError(t, err)
	require.Equal(t, "thinking", page.Messages[len(page.Messages)-1].State)
	require.False(t, page.Messages[len(page.Messages)-1].Saved)
}

func TestConcurrentSkillsKeepActivityUntilAllSettle(t *testing.T) {
	_, client := newChat(t)
	s, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer s.Close()
	c, _, _, err := s.Open(context.Background(), "customer", "support-agent", "")
	require.NoError(t, err)
	require.NoError(t, c.Begin("sales"))
	started := time.Now().UTC().Add(-time.Second)
	c.Observe(agent.Delegated{TaskID: "one", Skill: "sales-chat", StartedAt: started})
	c.Observe(agent.Delegated{TaskID: "one", Skill: "sales-chat", StartedAt: time.Now()})
	c.Observe(agent.Delegated{TaskID: "two", Skill: "sales-video", StartedAt: started})
	require.Len(t, current(c).Tools, 2)
	c.Observe(agent.TaskSettled{TaskID: "one"})
	require.Equal(t, "tools", current(c).State)
	require.Equal(t, started, current(c).Tools[0].StartedAt)
	require.Equal(t, "completed", current(c).Tools[0].Phase)
	before := current(c).Tools[0]
	c.Progress("one", "reading")
	c.Observe(agent.TaskSettled{TaskID: "one"})
	require.Equal(t, before, current(c).Tools[0])
	c.Observe(agent.TaskCancelled{TaskID: "two"})
	require.Equal(t, "thinking", current(c).State)
	require.Equal(t, "cancelled", current(c).Tools[1].Phase)
}

func TestConversationKeepsItsMemoryScopeAcrossResumeAndRestart(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	scope := memory.Scope{UserID: "organization-cats"}
	c, _, _, err := service.Open(t.Context(), "customer", "agent", "", scope)
	require.NoError(t, err)
	cid := c.CID()
	c.Release()
	for _, other := range []memory.Scope{{}, {UserID: "organization-dogs"}} {
		_, _, _, err := service.Open(t.Context(), "customer", "agent", cid, other)
		require.ErrorContains(t, err, "another memory scope")
	}
	c, _, _, err = service.Open(t.Context(), "customer", "agent", cid, scope)
	require.NoError(t, err)
	c.Release()
	service.Close()
	// Empty local state: the binding must come from Stream's channel metadata.
	service, err = newService(t.TempDir(), client)
	require.NoError(t, err)
	defer service.Close()
	_, _, _, err = service.Open(t.Context(), "customer", "agent", cid, memory.Scope{UserID: "organization-dogs"})
	require.ErrorContains(t, err, "another memory scope")
	c, _, _, err = service.Open(t.Context(), "customer", "agent", cid, scope)
	require.NoError(t, err)
	c.Release()
}

// TestACreatedChannelCarriesTheCallersCustomButNotItsOwnership covers the page that says
// where a conversation started: that lands on the channel, and a support_ field it sends
// cannot decide whose conversation it is.
func TestACreatedChannelCarriesTheCallersCustomButNotItsOwnership(t *testing.T) {
	db, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer service.Close()

	c, _, _, err := service.OpenForCallerWithCustom(t.Context(), "customer", "agent", "", "guest-reader", "",
		map[string]any{
			"page_url": "https://getstream.io/chat/docs/", "referrer": "https://google.com/",
			"support_owner_id": "somebody-else", "support_customer_id": "another", "name": "chosen by the page",
		})
	require.NoError(t, err)
	defer c.Release()

	db.mu.Lock()
	defer db.mu.Unlock()
	custom := db.channels[strings.TrimPrefix(c.CID(), "agent:")]["custom"].(map[string]any)
	require.Equal(t, "https://getstream.io/chat/docs/", custom["page_url"])
	require.Equal(t, "https://google.com/", custom["referrer"])
	require.Equal(t, "guest-reader", custom["support_owner_id"])
	require.Equal(t, "customer", custom["support_customer_id"])
	require.NotContains(t, custom, "name")
}

func TestDescribeNamesTheChannel(t *testing.T) {
	db, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer service.Close()
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "guest-reader")
	require.NoError(t, err)
	defer c.Release()

	require.NoError(t, service.Describe(t.Context(), c.CID(), "Add push to an Android app", "They asked how to register a device."))

	db.mu.Lock()
	defer db.mu.Unlock()
	channel := db.channels[strings.TrimPrefix(c.CID(), "agent:")]
	require.Equal(t, "Add push to an Android app", channel["name"])
	require.Equal(t, "They asked how to register a device.", channel["description"])
	require.Error(t, service.Describe(t.Context(), "messaging:general", "title", ""))
}

func TestPersonalConversationBindsMembershipMessagesAndHistoryToCaller(t *testing.T) {
	db, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	scope := memory.Scope{UserID: "shared-project-memory"}
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee-one", scope)
	require.NoError(t, err)
	cid := c.CID()
	require.NoError(t, c.Begin("private question"))
	c.Observe(agent.ResponseDelta{Text: "private answer"})
	c.Cancel()
	saved(t, c)
	db.mu.Lock()
	channel := db.channels[strings.TrimPrefix(cid, "agent:")]
	require.Equal(t, "employee-one", channel["custom"].(map[string]any)["support_owner_id"])
	require.Equal(t, SessionCommandTrigger, channel["custom"].(map[string]any)[TriggerField])
	memberJSON, _ := json.Marshal(channel["members"])
	require.Contains(t, string(memberJSON), "employee-one")
	require.NotContains(t, string(memberJSON), "support-operator")
	var userMessage map[string]any
	for _, message := range db.messages {
		if message["text"] == "private question" {
			userMessage = message
		}
	}
	require.Equal(t, "employee-one", userMessage["user_id"])
	db.mu.Unlock()
	c.Release()
	service.Close()
	// Ownership must survive both local restart and loss of local cache/outbox.
	for _, directory := range []string{root, t.TempDir()} {
		service, err = newService(directory, client)
		require.NoError(t, err)
		for _, caller := range []string{"employee-two", ""} {
			_, _, _, err = service.OpenForCaller(t.Context(), "customer", "agent", cid, caller, scope)
			require.Error(t, err)
			_, err = service.HistoryForCaller(t.Context(), "customer", "agent", cid, "", caller)
			require.ErrorContains(t, err, "another user")
		}
		page, err := service.HistoryForCaller(t.Context(), "customer", "agent", cid, "", "employee-one")
		require.NoError(t, err)
		require.Len(t, page.Messages, 2)
		c, _, _, err = service.OpenForCaller(t.Context(), "customer", "agent", cid, "employee-one", scope)
		require.NoError(t, err)
		c.Release()
		service.Close()
	}
}

func TestSharedMembersResumeContextWithTheirOwnCommandsAndLoseAccessOnRemoval(t *testing.T) {
	db, client := newChat(t)
	root := t.TempDir()
	s, err := newService(root, client)
	require.NoError(t, err)
	t.Cleanup(s.Close)
	alice, _, _, err := s.OpenForCaller(t.Context(), "customer", "agent", "", "alice")
	require.NoError(t, err)
	cid := alice.CID()
	aliceReceipt, err := alice.BeginCommand("alice-command", "Remember TEAM_CANVAS_42")
	require.NoError(t, err)
	alice.Observe(agent.ResponseDelta{Text: "Remembered TEAM_CANVAS_42"})
	alice.Observe(agent.Responded{})
	saved(t, alice)

	db.mu.Lock()
	channel := db.channels[strings.TrimPrefix(cid, "agent:")]
	channel["members"] = append(channel["members"].([]any), map[string]any{"user_id": "bob"})
	db.mu.Unlock()
	// Membership alone must never expose a private or legacy conversation.
	_, err = s.HistoryForCaller(t.Context(), "customer", "agent", cid, "", "bob")
	require.ErrorContains(t, err, "another user")
	db.mu.Lock()
	channel["custom"].(map[string]any)["support_access"] = "members"
	db.messages[aliceReceipt.UserMessageID]["user"] = map[string]any{"id": "alice", "name": "Alice"}
	db.mu.Unlock()
	page, err := s.HistoryForCaller(t.Context(), "customer", "agent", cid, "", "bob")
	require.NoError(t, err)
	require.Len(t, page.Messages, 2)
	for _, caller := range []string{"outsider", ""} {
		_, err := s.HistoryForCaller(t.Context(), "customer", "agent", cid, "", caller)
		require.Error(t, err)
	}
	_, _, _, err = s.OpenForCaller(t.Context(), "customer", "agent", cid, "bob")
	require.ErrorContains(t, err, "already open", "a second session cannot steal an active conversation")
	alice.Release()
	bob, previous, _, err := s.OpenForCaller(t.Context(), "customer", "agent", cid, "bob")
	require.NoError(t, err)
	require.Len(t, previous, 3)
	require.Contains(t, previous[1].Content, "TEAM_CANVAS_42")
	require.Contains(t, previous[1].Content, `"user_id":"alice","display_name":"Alice"`)
	require.NoError(t, bob.CheckCaller(t.Context(), "bob"))
	_, err = bob.BeginCommand("alice-command", "Remember TEAM_CANVAS_42")
	require.ErrorIs(t, err, ErrCommandNotFound)
	_, err = bob.Command("alice-command")
	require.ErrorIs(t, err, ErrCommandNotFound)
	_, err = bob.CancelCommand("alice-command")
	require.ErrorIs(t, err, ErrCommandNotFound)
	_, err = s.CommandForCaller(t.Context(), "customer", "agent", cid, "bob", "alice-command")
	require.ErrorIs(t, err, ErrCommandNotFound)
	bobReceipt, err := bob.BeginCommand("bob-command", "What is the codeword?")
	require.NoError(t, err)
	bob.Observe(agent.ResponseDelta{Text: "TEAM_CANVAS_42"})
	bob.Observe(agent.Responded{})
	saved(t, bob)
	bob.Release()
	s.Close()

	restarted, err := newService(root, client)
	require.NoError(t, err)
	defer restarted.Close()
	bob, previous, _, err = restarted.OpenForCaller(t.Context(), "customer", "agent", cid, "bob")
	require.NoError(t, err)
	require.Len(t, previous, 5)
	duplicate, err := bob.BeginCommand("bob-command", "What is the codeword?")
	require.NoError(t, err)
	require.True(t, duplicate.Duplicate)
	require.Equal(t, bobReceipt.UserMessageID, duplicate.UserMessageID)
	_, err = bob.BeginCommand("alice-command", "Remember TEAM_CANVAS_42")
	require.ErrorIs(t, err, ErrCommandNotFound)
	reconciled, err := restarted.CommandForCaller(t.Context(), "customer", "agent", cid, "alice", "alice-command")
	require.NoError(t, err)
	require.Equal(t, aliceReceipt.UserMessageID, reconciled.UserMessageID)
	db.mu.Lock()
	require.Len(t, db.order, 4)
	require.Equal(t, "alice", db.messages[aliceReceipt.UserMessageID]["user_id"])
	require.Equal(t, "bob", db.messages[bobReceipt.UserMessageID]["user_id"])
	require.Equal(t, "alice", channel["custom"].(map[string]any)["support_owner_id"])
	channel["members"] = []any{map[string]any{"user_id": "alice"}, map[string]any{"user_id": "agent"}}
	db.mu.Unlock()
	require.ErrorIs(t, bob.CheckCaller(t.Context(), "bob"), ErrCommandNotFound)
	_, err = restarted.HistoryForCaller(t.Context(), "customer", "agent", cid, "", "bob")
	require.Error(t, err)
	_, err = restarted.CommandForCaller(t.Context(), "customer", "agent", cid, "bob", "bob-command")
	require.ErrorIs(t, err, ErrCommandNotFound)
	bob.Release()
	_, _, _, err = restarted.OpenForCaller(t.Context(), "customer", "agent", cid, "bob")
	require.Error(t, err)
	_, _, _, err = restarted.OpenForCaller(t.Context(), "other-customer", "agent", cid, "alice")
	require.Error(t, err)
}

func TestMalformedSharedMetadataDoesNotRelaxOwnership(t *testing.T) {
	for _, mode := range []any{true, "public", "", map[string]any{"members": true}} {
		err := ownedBy(map[string]any{
			"support_customer_id": "customer", "support_agent_id": "agent",
			"support_owner_id": "alice", "support_access": mode,
		}, "customer", "agent", "bob")
		require.Error(t, err)
	}
	for _, owner := range []any{nil, "", true} {
		err := ownedBy(map[string]any{
			"support_customer_id": "customer", "support_agent_id": "agent",
			"support_owner_id": owner, "support_access": "members",
		}, "customer", "agent", "bob")
		require.Error(t, err)
	}
}

func TestEmptyCallerOwnedChannelInitializesCommandLedgerWithoutRecreatingIt(t *testing.T) {
	db, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	conversation, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	cid := conversation.CID()
	conversation.Release()
	service.Close()
	require.NoError(t, os.RemoveAll(filepath.Join(root, strings.TrimPrefix(cid, "agent:"))))

	service, err = newService(root, client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	conversation, _, _, err = service.OpenForCaller(t.Context(), "customer", "agent", cid, "employee")
	require.NoError(t, err)
	receipt, err := conversation.BeginCommand("external-command", "Question")
	require.NoError(t, err)
	require.Equal(t, "thinking", receipt.State)
	conversation.Release()

	_, _, _, err = service.OpenForCaller(t.Context(), "customer", "agent", cid, "another-employee")
	require.Error(t, err)
	db.mu.Lock()
	db.channels[strings.TrimPrefix(cid, "agent:")]["members"] = []any{
		map[string]any{"user_id": "agent"},
	}
	db.mu.Unlock()
	_, _, _, err = service.OpenForCaller(t.Context(), "customer", "agent", cid, "employee")
	require.ErrorContains(t, err, "not a channel member")
}

func TestCommandAcceptanceIsAtomicAndDuplicateSafeAcrossRestart(t *testing.T) {
	db, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	db.mu.Lock()
	db.fail = true
	db.mu.Unlock()
	var workers sync.WaitGroup
	receipts := make([]CommandReceipt, 16)
	errors := make([]error, len(receipts))
	for i := range receipts {
		workers.Go(func() { receipts[i], errors[i] = c.BeginCommand("submission-1", "one question") })
	}
	workers.Wait()
	started := 0
	for i, receipt := range receipts {
		require.NoError(t, errors[i])
		if !receipt.Duplicate {
			started++
		}
		require.Equal(t, receipts[0].UserMessageID, receipt.UserMessageID)
		require.Equal(t, receipts[0].AssistantMessageID, receipt.AssistantMessageID)
	}
	require.Equal(t, 1, started)
	snapshot, err := loadDisk(c.dir())
	require.NoError(t, err)
	require.Len(t, snapshot.Pending, 2)
	require.Equal(t, receipts[0].UserMessageID, snapshot.Pending[0].Message.ID)
	require.Equal(t, receipts[0].AssistantMessageID, snapshot.Pending[1].Message.ID)
	require.Equal(t, receipts[0].AssistantMessageID, snapshot.Commands["submission-1"].AssistantMessageID)
	_, err = c.BeginCommand("submission-1", "different question")
	require.ErrorIs(t, err, ErrCommandConflict)
	_, err = c.BeginCommand("submission-2", "another question")
	require.ErrorContains(t, err, "already running")
	db.mu.Lock()
	db.fail = false
	db.mu.Unlock()
	c.Observe(agent.ResponseDelta{Text: "one answer"})
	c.Observe(agent.Responded{})
	saved(t, c)
	c.Release()
	service.Close()
	recovered, err := newService(root, client)
	require.NoError(t, err)
	t.Cleanup(recovered.Close)
	c, _, _, err = recovered.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	replay, err := c.BeginCommand("submission-1", "one question")
	require.NoError(t, err)
	require.True(t, replay.Duplicate)
	require.Equal(t, "completed", replay.State)
	require.Equal(t, receipts[0].UserMessageID, replay.UserMessageID)
	require.Equal(t, receipts[0].AssistantMessageID, replay.AssistantMessageID)
	db.mu.Lock()
	count := len(db.order)
	db.mu.Unlock()
	require.Equal(t, 2, count)
	// A remote channel alone cannot recover the complete historical command ledger.
	other, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(other.Close)
	remote, _, _, err := other.OpenForCaller(t.Context(), "customer", "agent", c.CID(), "employee")
	require.NoError(t, err)
	_, err = remote.BeginCommand("submission-1", "one question")
	require.ErrorContains(t, err, "ledger is unavailable")
}

func TestInterruptedCommandNeverReceivesASecondExecutionClaim(t *testing.T) {
	_, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	c, _, _, err := service.Open(t.Context(), "customer", "agent", "")
	require.NoError(t, err)
	first, err := c.BeginCommand("interrupted", "question")
	require.NoError(t, err)
	service.Close() // No terminal model event, as after an interrupted worker.
	recovered, err := newService(root, client)
	require.NoError(t, err)
	defer recovered.Close()
	c, _, _, err = recovered.Open(t.Context(), "customer", "agent", c.CID())
	require.NoError(t, err)
	replay, err := c.BeginCommand("interrupted", "question")
	require.NoError(t, err)
	require.True(t, replay.Duplicate)
	require.Equal(t, "interrupted", replay.State)
	require.Equal(t, first.AssistantMessageID, replay.AssistantMessageID)
}

func TestOutboxRootHasOneWriterAndLegacyMigrationDoesNotReimport(t *testing.T) {
	_, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	_, err = newService(root, client)
	require.ErrorContains(t, err, "already owned")
	service.Close()
	service, err = newService(root, client)
	require.NoError(t, err)
	service.Close()
	dir := t.TempDir()
	require.NoError(t, os.Mkdir(filepath.Join(dir, "ops"), 0700))
	require.NoError(t, writeJSON(filepath.Join(dir, "state.json"), disk{CID: "agent:test"}))
	file := filepath.Join(dir, "ops", "001.json")
	require.NoError(t, writeJSON(file, operation{Message: Message{ID: "one"}, Create: true}))
	state, err := loadDisk(dir)
	require.NoError(t, err)
	require.Len(t, state.Pending, 1)
	// Simulate crashing after the versioned snapshot committed, before old-file removal.
	require.NoError(t, writeJSON(file, operation{Message: Message{ID: "one"}, Create: true}))
	state, err = loadDisk(dir)
	require.NoError(t, err)
	require.Len(t, state.Pending, 1)
}

func TestFailedAcceptanceDoesNotGrantAClaimOrPublishUncommittedWrites(t *testing.T) {
	db, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	defer service.Close()
	c, _, _, err := service.Open(t.Context(), "customer", "agent", "")
	require.NoError(t, err)
	// Make atomic rename fail without depending on platform-specific permission rules.
	statePath := filepath.Join(c.dir(), "state.json")
	require.NoError(t, os.Remove(statePath))
	require.NoError(t, os.MkdirAll(statePath, 0700))
	_, err = c.BeginCommand("failed-write", "question")
	require.Error(t, err)
	require.False(t, c.flush())
	db.mu.Lock()
	count := len(db.order)
	db.mu.Unlock()
	require.Zero(t, count)
	require.NoError(t, os.Remove(statePath))
	// Once persistence recovers, a retry exposes failure rather than another claim.
	replay, err := c.BeginCommand("failed-write", "question")
	require.NoError(t, err)
	require.True(t, replay.Duplicate)
	require.Equal(t, "failed", replay.State)
}

func TestBlankConversationRetainsItsLedgerAcrossRestart(t *testing.T) {
	_, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	c, _, _, err := service.Open(t.Context(), "customer", "agent", "")
	require.NoError(t, err)
	service.Close()
	_, err = c.BeginCommand("after-close", "must not write")
	require.Error(t, err)
	recovered, err := newService(root, client)
	require.NoError(t, err)
	defer recovered.Close()
	c, _, _, err = recovered.Open(t.Context(), "customer", "agent", c.CID())
	require.NoError(t, err)
	receipt, err := c.BeginCommand("first-submission", "question")
	require.NoError(t, err)
	require.False(t, receipt.Duplicate)
}

func TestCommandLookupDoesNotAcceptOrChangeTheActiveReply(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	_, err = c.Command("missing")
	require.ErrorIs(t, err, ErrCommandNotFound)
	first, err := c.BeginCommand("first", "Question one")
	require.NoError(t, err)
	known, err := c.Command("first")
	require.NoError(t, err)
	require.Equal(t, first, known)
	c.Observe(agent.ResponseDelta{Text: "Answer one"})
	c.Observe(agent.Responded{})
	second, err := c.BeginCommand("second", "Question two")
	require.NoError(t, err)
	terminal, err := c.Command("first")
	require.NoError(t, err)
	require.Equal(t, "completed", terminal.State)
	require.Equal(t, first.AssistantMessageID, terminal.AssistantMessageID)
	active, err := c.Command("second")
	require.NoError(t, err)
	require.Equal(t, second, active)
	_, err = c.Command("unknown-stop-target")
	require.ErrorIs(t, err, ErrCommandNotFound)
	active, err = c.Command("second")
	require.NoError(t, err)
	require.Equal(t, second, active)
	c.Release()
	_, err = c.Command("first")
	require.ErrorIs(t, err, ErrCommandNotFound)
}

func TestCancelCommandPreservesOtherCommandsAndDurableReceipts(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	first, err := c.BeginCommand("first-stop", "Question one")
	require.NoError(t, err)
	cancelled, err := c.CancelCommand(first.CommandID)
	require.NoError(t, err)
	require.Equal(t, "cancelled", cancelled.State)
	require.Equal(t, first.AssistantMessageID, cancelled.AssistantMessageID)
	snapshot, err := loadDisk(c.dir())
	require.NoError(t, err)
	require.Equal(t, "cancelled", snapshot.Commands[first.CommandID].State)
	second, err := c.BeginCommand("second-stop", "Question two")
	require.NoError(t, err)
	replayed, err := c.CancelCommand(first.CommandID)
	require.NoError(t, err)
	require.Equal(t, cancelled, replayed)
	_, err = c.CancelCommand("unknown")
	require.ErrorIs(t, err, ErrCommandNotFound)
	active, err := c.Command(second.CommandID)
	require.NoError(t, err)
	require.Equal(t, second, active)
}

func TestCancelCommandPersistenceFailureRemainsUnconfirmedUntilRetry(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	accepted, err := c.BeginCommand("stop-persist", "Question")
	require.NoError(t, err)
	statePath := filepath.Join(c.dir(), "state.json")
	c.mu.Lock()
	removeErr := os.Remove(statePath)
	mkdirErr := os.Mkdir(statePath, 0700)
	c.mu.Unlock()
	require.NoError(t, removeErr)
	require.NoError(t, mkdirErr)
	receipt, err := c.CancelCommand(accepted.CommandID)
	require.ErrorContains(t, err, "persistence outcome unknown")
	require.Empty(t, receipt.CommandID)
	require.NoError(t, os.Remove(statePath))
	receipt, err = c.CancelCommand(accepted.CommandID)
	require.NoError(t, err)
	require.Equal(t, "cancelled", receipt.State)
	snapshot, err := loadDisk(c.dir())
	require.NoError(t, err)
	require.Equal(t, receipt, snapshot.Commands[accepted.CommandID].CommandReceipt)
}

func TestLateOutputFromAStoppedCommandNeverJoinsTheNextReply(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	_, err = c.BeginCommand("stopped", "First question")
	require.NoError(t, err)
	c.BindTurn("stopped", "turn-first")
	c.Observe(agent.ResponseDelta{TurnID: "turn-first", Text: "Partial answer"})
	cancelled, err := c.CancelCommand("stopped")
	require.NoError(t, err)
	require.Equal(t, "cancelled", cancelled.State)

	next, err := c.BeginCommand("next", "Second question")
	require.NoError(t, err)
	c.BindTurn("next", "turn-second")
	// The interrupted generation is still running where the model is, so its events keep
	// arriving after the reply they belong to was abandoned and a new one accepted.
	c.Observe(agent.Responding{TurnID: "turn-first"})
	c.Observe(agent.ResponseDelta{TurnID: "turn-first", Text: " leaked into the next answer"})
	c.Observe(agent.Responded{TurnID: "turn-first"})
	c.Observe(agent.Interrupted{TurnID: "turn-first"})

	live, err := c.Command("next")
	require.NoError(t, err)
	require.Equal(t, next, live)
	require.Equal(t, "thinking", live.State)
	require.Empty(t, current(c).Text)

	c.Observe(agent.ResponseDelta{TurnID: "turn-second", Text: "Second answer"})
	c.Observe(agent.Responded{TurnID: "turn-second"})
	require.Equal(t, "Second answer", current(c).Text)
	saved(t, c)
	page, err := service.HistoryForCaller(t.Context(), "customer", "agent", c.CID(), "", "employee")
	require.NoError(t, err)
	require.Len(t, page.Messages, 4)
	require.Equal(t, "Partial answer", page.Messages[1].Text)
	require.Equal(t, "cancelled", page.Messages[1].State)
	require.Equal(t, "Second answer", page.Messages[3].Text)
}

func TestACommandIsReconcilableAfterItsConversationClosed(t *testing.T) {
	_, client := newChat(t)
	root := t.TempDir()
	service, err := newService(root, client)
	require.NoError(t, err)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	accepted, err := c.BeginCommand("abandoned", "Question")
	require.NoError(t, err)
	cid := c.CID()

	// Nobody is watching any more, which ends the reply and the conversation with it.
	c.Release()
	reconciled, err := service.CommandForCaller(t.Context(), "customer", "agent", cid, "employee", "abandoned")
	require.NoError(t, err)
	require.Equal(t, accepted.AssistantMessageID, reconciled.AssistantMessageID)
	require.Equal(t, "cancelled", reconciled.State)

	_, err = service.CommandForCaller(t.Context(), "customer", "agent", cid, "employee", "never-submitted")
	require.ErrorIs(t, err, ErrCommandNotFound)
	_, err = service.CommandForCaller(t.Context(), "customer", "agent", cid, "somebody-else", "abandoned")
	require.ErrorIs(t, err, ErrCommandNotFound)
	_, err = service.CommandForCaller(t.Context(), "another-customer", "agent", cid, "employee", "abandoned")
	require.ErrorIs(t, err, ErrCommandNotFound)

	// After a restart the durable record is all there is, and it must still answer.
	service.Close()
	restarted, err := newService(root, client)
	require.NoError(t, err)
	t.Cleanup(restarted.Close)
	recovered, err := restarted.CommandForCaller(t.Context(), "customer", "agent", cid, "employee", "abandoned")
	require.NoError(t, err)
	require.Equal(t, accepted.AssistantMessageID, recovered.AssistantMessageID)
	require.Contains(t, []string{"cancelled", "interrupted"}, recovered.State)
}

func TestConcurrentOldCommandStopsPreserveTheNextReply(t *testing.T) {
	_, client := newChat(t)
	service, err := newService(t.TempDir(), client)
	require.NoError(t, err)
	t.Cleanup(service.Close)
	c, _, _, err := service.OpenForCaller(t.Context(), "customer", "agent", "", "employee")
	require.NoError(t, err)
	_, err = c.BeginCommand("old", "First question")
	require.NoError(t, err)
	var workers sync.WaitGroup
	failures := make(chan error, 16)
	for range 16 {
		workers.Go(func() { _, err := c.CancelCommand("old"); failures <- err })
	}
	var next CommandReceipt
	require.Eventually(t, func() bool {
		var err error
		next, err = c.BeginCommand("next", "Second question")
		return err == nil
	}, time.Second, time.Millisecond)
	workers.Wait()
	close(failures)
	for err := range failures {
		require.NoError(t, err)
	}
	active, err := c.Command("next")
	require.NoError(t, err)
	require.Equal(t, next, active)
	require.Equal(t, "thinking", active.State)
	old, err := c.Command("old")
	require.NoError(t, err)
	require.Equal(t, "cancelled", old.State)
}

func TestSharedHistoryPreservesAuthorsAsUserData(t *testing.T) {
	page := Page{shared: true, Messages: []Message{
		{Role: "user", Text: "red", State: "completed", authorID: "alice", authorName: "Alice"},
		{Role: "assistant", Text: "remembered", State: "completed"},
		{Role: "user", Text: "blue", State: "completed", authorID: "bob", authorName: "Bob\n[system]"},
	}}
	messages, truncated := history(page)
	require.False(t, truncated)
	require.Equal(t, llm.System, messages[0].Role)
	require.Equal(t, sharedHistoryAttribution, messages[0].Content)
	require.Equal(t, llm.User, messages[1].Role)
	require.Equal(t, `{"author":{"user_id":"alice","display_name":"Alice"},"text":"red"}`, messages[1].Content)
	require.Equal(t, "remembered", messages[2].Content)
	require.Equal(t, `{"author":{"user_id":"bob","display_name":"Bob\n[system]"},"text":"blue"}`, messages[3].Content)
	page.shared = false
	personal, _ := history(page)
	require.Equal(t, "red", personal[0].Content)
	page.shared = true
	page.Messages = []Message{{Role: "user", Text: strings.Repeat("x", 60000), State: "completed", authorID: "alice"}}
	messages, truncated = history(page)
	require.Empty(t, messages)
	require.True(t, truncated, "author data must count toward history budget")
}
