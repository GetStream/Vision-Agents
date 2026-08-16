package mem0

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/memory"
)

// platform stands in for mem0's API so the wire contract can be tested without a key.
type platform struct {
	server *httptest.Server

	path    string
	auth    string
	body    map[string]any
	status  int
	respond string
}

func newPlatform() *platform {
	stub := &platform{status: http.StatusOK, respond: `{"results":[]}`}
	stub.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stub.path = r.URL.Path
		stub.auth = r.Header.Get("Authorization")

		raw, _ := io.ReadAll(r.Body)
		stub.body = map[string]any{}
		_ = json.Unmarshal(raw, &stub.body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(stub.status)
		_, _ = w.Write([]byte(stub.respond))
	}))
	return stub
}

type Mem0Suite struct {
	suite.Suite
	ctx      context.Context
	platform *platform
	store    *Store
	scope    memory.Scope
}

func TestMem0Suite(t *testing.T) {
	suite.Run(t, new(Mem0Suite))
}

func (s *Mem0Suite) SetupTest() {
	s.ctx = context.Background()
	s.platform = newPlatform()
	s.T().Cleanup(s.platform.server.Close)

	store, err := New(Options{
		APIKey:  "test-key",
		BaseURL: s.platform.server.URL,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.store = store
	s.scope = memory.Scope{AppID: "router", UserID: "acme"}
}

func (s *Mem0Suite) TestAKeyIsRequired() {
	s.T().Setenv("MEM0_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "MEM0_API_KEY")
}

func (s *Mem0Suite) TestTheKeyIsSentAsATokenHeader() {
	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "where do they live"})
	s.Require().NoError(err)

	s.Equal("Token test-key", s.platform.auth)
}

func (s *Mem0Suite) TestRecallScopesTheSearchToWhoTheMemoriesBelongTo() {
	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "where do they live", Limit: 3})
	s.Require().NoError(err)

	s.Equal("/v3/memories/search/", s.platform.path)
	s.Equal("where do they live", s.platform.body["query"])
	s.Equal(float64(3), s.platform.body["top_k"])
	s.Equal(map[string]any{"user_id": "acme", "app_id": "router"}, s.platform.body["filters"],
		"v3 rejects entity ids at the top level, they belong in filters")
}

func (s *Mem0Suite) TestRecallReturnsWhatIsKnownMostRelevantFirst() {
	s.platform.respond = `{"results":[
		{"id":"m1","memory":"Lives in Austin","score":0.9},
		{"id":"m2","memory":"Allergic to nuts","score":0.4}
	]}`

	recalled, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})
	s.Require().NoError(err)

	s.Require().Len(recalled, 2)
	s.Equal(memory.Memory{ID: "m1", Text: "Lives in Austin", Score: 0.9}, recalled[0])
	s.Equal("Allergic to nuts", recalled[1].Text)
}

func (s *Mem0Suite) TestRecallWithoutAUserIsRejectedBeforeTheNetwork() {
	_, err := s.store.Recall(s.ctx, memory.Query{Text: "anything"})

	s.ErrorContains(err, "user id")
	s.Empty(s.platform.path, "an unscoped recall would read somebody else's memories")
}

func (s *Mem0Suite) TestRememberHandsTheConversationOver() {
	err := s.store.Remember(s.ctx, s.scope, []llm.Message{
		{Role: llm.User, Content: "I moved to Austin"},
		{Role: llm.Assistant, Content: "Noted."},
	})
	s.Require().NoError(err)

	s.Equal("/v3/memories/add/", s.platform.path)
	s.Equal("acme", s.platform.body["user_id"])
	s.Equal("router", s.platform.body["app_id"])
	s.Equal([]any{
		map[string]any{"role": "user", "content": "I moved to Austin"},
		map[string]any{"role": "assistant", "content": "Noted."},
	}, s.platform.body["messages"])
}

func (s *Mem0Suite) TestRememberingNothingIsNotACall() {
	s.Require().NoError(s.store.Remember(s.ctx, s.scope, nil))
	s.Require().NoError(s.store.Remember(s.ctx, s.scope, []llm.Message{{Role: llm.User}}))

	s.Empty(s.platform.path, "an empty conversation has nothing to learn from")
}

func (s *Mem0Suite) TestAFailureCarriesWhatThePlatformSaid() {
	s.platform.status = http.StatusBadRequest
	s.platform.respond = `{"error":"400 Bad Request"}`

	_, err := s.store.Recall(s.ctx, memory.Query{Scope: s.scope, Text: "anything"})

	s.ErrorContains(err, "400")
	s.ErrorContains(err, "Bad Request")
}
