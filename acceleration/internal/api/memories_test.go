package api

import (
	"context"
	"net/http"
	"sync"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// keptMemories holds memories in a slice, and forgets them the way a real store would.
type keptMemories struct {
	mu   sync.Mutex
	kept []memory.Scope
}

func (m *keptMemories) Recall(context.Context, memory.Query) ([]memory.Memory, error) {
	return nil, nil
}

func (m *keptMemories) Remember(_ context.Context, scope memory.Scope, _ []llm.Message) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.kept = append(m.kept, scope)
	return nil
}

func (m *keptMemories) Truncate(_ context.Context, appID, userID string) error {
	m.forget(func(kept memory.Scope) bool { return kept.AppID == appID && kept.UserID == userID })
	return nil
}

func (m *keptMemories) ForgetRun(_ context.Context, appID, runID string) error {
	m.forget(func(kept memory.Scope) bool { return kept.AppID == appID && kept.RunID == runID })
	return nil
}

func (m *keptMemories) Provider() string { return "kept" }
func (m *keptMemories) Close() error     { return nil }

func (m *keptMemories) forget(matches func(memory.Scope) bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	left := m.kept[:0]
	for _, kept := range m.kept {
		if !matches(kept) {
			left = append(left, kept)
		}
	}
	m.kept = left
}

func (m *keptMemories) remaining() []memory.Scope {
	m.mu.Lock()
	defer m.mu.Unlock()
	return append([]memory.Scope(nil), m.kept...)
}

func (s *SessionAPISuite) TestTruncatingAUserDeletesOnlyThatUsersMemories() {
	s.memories.kept = []memory.Scope{
		{AppID: "acme", UserID: "222", RunID: "one"},
		{AppID: "acme", UserID: "222", RunID: "two"},
		{AppID: "acme", UserID: "333", RunID: "three"},
		{AppID: "other", UserID: "222", RunID: "four"},
	}

	response := s.send(http.MethodDelete, "/v1/agents/users/222/memories", "acme", nil)

	s.Equal(http.StatusNoContent, response.StatusCode)
	s.Equal([]memory.Scope{
		{AppID: "acme", UserID: "333", RunID: "three"},
		{AppID: "other", UserID: "222", RunID: "four"},
	}, s.memories.remaining(), "another user's, and another customer's same user, are kept")
}

func (s *SessionAPISuite) TestDeletingASessionsMemoriesKeepsTheUsersOthers() {
	created := s.writes(CreateSessionRequest{})
	s.memories.kept = []memory.Scope{
		{AppID: "acme", UserID: "222", RunID: created.Id},
		{AppID: "acme", UserID: "222", RunID: "an-earlier-session"},
	}

	response := s.send(http.MethodDelete, "/v1/agents/sessions/"+created.Id+"/memories", "acme", nil)

	s.Equal(http.StatusNoContent, response.StatusCode)
	s.Equal([]memory.Scope{{AppID: "acme", UserID: "222", RunID: "an-earlier-session"}},
		s.memories.remaining())
}

func (s *SessionAPISuite) TestAnotherCustomersSessionsMemoriesCannotBeDeleted() {
	created := s.writes(CreateSessionRequest{})
	s.memories.kept = []memory.Scope{{AppID: "acme", UserID: "222", RunID: created.Id}}

	response := s.send(http.MethodDelete, "/v1/agents/sessions/"+created.Id+"/memories", "other", nil)

	s.Equal(http.StatusNotFound, response.StatusCode)
	s.Len(s.memories.remaining(), 1, "a session id says nothing about whose it is")
}

func (s *SessionAPISuite) TestEndingASessionKeepsWhatItRemembered() {
	created := s.writes(CreateSessionRequest{})
	s.memories.kept = []memory.Scope{{AppID: "acme", UserID: "222", RunID: created.Id}}

	response := s.send(http.MethodDelete, "/v1/agents/sessions/"+created.Id, "acme", nil)

	s.Equal(http.StatusNoContent, response.StatusCode)
	s.Len(s.memories.remaining(), 1, "memory is what the next conversation starts from")
}
