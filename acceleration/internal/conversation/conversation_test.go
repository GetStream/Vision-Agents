package conversation

import (
	"context"
	"encoding/json"
	"fmt"

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
		case strings.HasSuffix(r.URL.Path, "/query"):
			id := parts[len(parts)-2]
			if data, ok := body["data"].(map[string]any); ok {
				db.channels[id] = data
			}
			result["channel"] = db.channels[id]
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
	c.Observe(agent.ToolStarted{ID: "one", Tool: "investigate_sdk", StartedAt: begin})
	require.Equal(t, begin, current(c).Tools[0].StartedAt)
	c.Progress("one", "queued")
	require.Equal(t, "queued", current(c).State)
	require.Nil(t, current(c).Tools[0].ExecutionStartedAt)
	c.Progress("one", "recovering_workspace")
	require.Nil(t, current(c).Tools[0].ExecutionStartedAt)
	c.Progress("one", "starting_worker")
	require.Nil(t, current(c).Tools[0].ExecutionStartedAt)
	c.Progress("one", "searching")
	execAt := current(c).Tools[0].ExecutionStartedAt
	require.NotNil(t, execAt)
	c.Progress("one", "reading")
	require.Equal(t, execAt, current(c).Tools[0].ExecutionStartedAt)
	c.Observe(agent.ToolStarted{ID: "one", Tool: "investigate_sdk", StartedAt: time.Now()})
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
	require.Equal(t, want.Tools, page.Messages[1].Tools)
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
	files, err := filepath.Glob(filepath.Join(c.dir(), "ops", "*.json"))
	require.NoError(t, err)
	require.NotEmpty(t, files)
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
	files, err = filepath.Glob(filepath.Join(c.dir(), "ops", "*.json"))
	require.NoError(t, err)
	require.Empty(t, files)
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
