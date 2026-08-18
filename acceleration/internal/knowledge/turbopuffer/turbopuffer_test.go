package turbopuffer

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/knowledge"
)

// search stands in for turbopuffer's API so the wire contract can be tested without a key.
type search struct {
	server *httptest.Server

	path    string
	auth    string
	body    map[string]any
	status  int
	respond string
}

func newSearch() *search {
	stub := &search{status: http.StatusOK, respond: `{"rows":[]}`}
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

type TurbopufferSuite struct {
	suite.Suite
	ctx    context.Context
	search *search
	store  *Store
}

func TestTurbopufferSuite(t *testing.T) {
	suite.Run(t, new(TurbopufferSuite))
}

func (s *TurbopufferSuite) SetupTest() {
	s.ctx = context.Background()
	s.search = newSearch()
	s.T().Cleanup(s.search.server.Close)

	store, err := New(Options{
		APIKey:  "test-key",
		BaseURL: s.search.server.URL,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.store = store
}

func (s *TurbopufferSuite) TestAKeyIsRequired() {
	s.T().Setenv("TURBOPUFFER_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "TURBOPUFFER_API_KEY")
}

func (s *TurbopufferSuite) TestASearchRanksTheNamespaceItWasGiven() {
	_, err := s.store.Search(s.ctx, knowledge.Query{
		Namespace: "handbook", Text: "what does delivery cost", Limit: 3,
	})
	s.Require().NoError(err)

	s.Equal("Bearer test-key", s.search.auth)
	s.Equal("/v2/namespaces/handbook/query", s.search.path)
	s.Equal([]any{"text", "BM25", "what does delivery cost"}, s.search.body["rank_by"])
	s.Equal(float64(3), s.search.body["limit"])
}

func (s *TurbopufferSuite) TestSearchReturnsThePassagesAndWhereTheyCameFrom() {
	s.search.respond = `{"rows":[
		{"id":"shipping_0","$dist":4.2,"text":"Delivery is free over $50.","source":"shipping.md"},
		{"id":"shipping_1","$dist":1.1,"text":"Next-day costs $9.","source":"shipping.md"}
	]}`

	found, err := s.store.Search(s.ctx, knowledge.Query{Namespace: "handbook", Text: "delivery"})
	s.Require().NoError(err)

	s.Require().Len(found, 2)
	s.Equal(knowledge.Document{
		ID:     "shipping_0",
		Text:   "Delivery is free over $50.",
		Source: "shipping.md",
		Score:  4.2,
	}, found[0])
	s.Equal("Next-day costs $9.", found[1].Text)
}

func (s *TurbopufferSuite) TestANumericIdIsStillReadable() {
	// A namespace written from curl carries integer ids, and a document nobody can name
	// is no less an answer.
	s.search.respond = `{"rows":[{"id":7,"$dist":2.0,"text":"We open at nine."}]}`

	found, err := s.store.Search(s.ctx, knowledge.Query{Namespace: "handbook", Text: "hours"})
	s.Require().NoError(err)

	s.Require().Len(found, 1)
	s.Equal("7", found[0].ID)
	s.Empty(found[0].Source)
}

func (s *TurbopufferSuite) TestAnEmptyKnowledgeBaseIsNotAFailure() {
	// A namespace nobody has written to yet 404s. An agent configured against one that
	// is still being filled should say it does not know, not fail the turn.
	s.search.status = http.StatusNotFound
	s.search.respond = `{"status":"error","error":"namespace not found"}`

	found, err := s.store.Search(s.ctx, knowledge.Query{Namespace: "handbook", Text: "anything"})

	s.Require().NoError(err)
	s.Empty(found)
}

func (s *TurbopufferSuite) TestSearchingWithoutANamespaceIsRejectedBeforeTheNetwork() {
	_, err := s.store.Search(s.ctx, knowledge.Query{Text: "anything"})

	s.ErrorContains(err, "namespace")
	s.Empty(s.search.path, "an unscoped search would answer out of somebody else's handbook")
}

func (s *TurbopufferSuite) TestAFailureCarriesWhatTheServiceSaid() {
	s.search.status = http.StatusUnprocessableEntity
	s.search.respond = `{"error":"text is not indexed for full-text search"}`

	_, err := s.store.Search(s.ctx, knowledge.Query{Namespace: "handbook", Text: "anything"})

	s.ErrorContains(err, "422")
	s.ErrorContains(err, "full-text search")
}
