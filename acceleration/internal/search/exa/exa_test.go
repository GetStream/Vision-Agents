package exa

import (
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/search"
)

// web stands in for Exa's API so the wire contract can be tested without a key.
type web struct {
	server *httptest.Server

	path   string
	key    string
	body   map[string]any
	status int
	// respond is the JSON the stub answers with. No results is the default because a
	// search that found nothing is the case worth not crashing on.
	respond string
}

func newWeb() *web {
	stub := &web{status: http.StatusOK, respond: `{"results":[]}`}
	stub.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		stub.path = r.URL.Path
		stub.key = r.Header.Get("x-api-key")

		raw, _ := io.ReadAll(r.Body)
		stub.body = map[string]any{}
		_ = json.Unmarshal(raw, &stub.body)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(stub.status)
		_, _ = w.Write([]byte(stub.respond))
	}))
	return stub
}

type ExaSuite struct {
	suite.Suite
	ctx      context.Context
	web      *web
	provider *Provider
}

func TestExaSuite(t *testing.T) {
	suite.Run(t, new(ExaSuite))
}

func (s *ExaSuite) SetupTest() {
	s.ctx = context.Background()
	s.web = newWeb()
	s.T().Cleanup(s.web.server.Close)

	provider, err := New(Options{
		APIKey:  "test-key",
		BaseURL: s.web.server.URL,
		Logger:  slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *ExaSuite) TestAKeyIsRequired() {
	s.T().Setenv("EXA_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "EXA_API_KEY")
}

func (s *ExaSuite) TestAWayOfSearchingExaDoesNotHaveIsRefused() {
	_, err := New(Options{APIKey: "k", Model: "telepathic"})

	s.ErrorContains(err, "telepathic")
}

func (s *ExaSuite) TestASearchAsksForHighlightsRatherThanWholePages() {
	// Five whole articles handed to a voice model get read out. The highlights are asked
	// for against the question rather than the top of the page, so what comes back is the
	// part that bears on it.
	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70 in Colorado"})
	s.Require().NoError(err)

	s.Equal("test-key", s.web.key)
	s.Equal("/search", s.web.path)
	s.Equal("traffic on I-70 in Colorado", s.web.body["query"])
	s.Equal(ModelFast, s.web.body["type"])

	contents, ok := s.web.body["contents"].(map[string]any)
	s.Require().True(ok, "the search asks for contents")
	highlights, ok := contents["highlights"].(map[string]any)
	s.Require().True(ok)
	s.Equal("traffic on I-70 in Colorado", highlights["query"])
}

func (s *ExaSuite) TestSearchReturnsTheSourcesAndNoSummary() {
	// Exa writes no answer of its own, which is what makes it the quick option: nothing
	// here waits on a model. The prompt renders the sources alone.
	s.web.respond = `{"results":[
		{"title":"COtrip","url":"https://cotrip.org","highlights":["No closures reported."],"score":0.9},
		{"title":"CDOT","url":"https://codot.gov","text":"Chain law is off.","score":0.4}
	]}`

	found, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Empty(found.Answer)
	s.Require().Len(found.Documents, 2)
	s.Equal("COtrip", found.Documents[0].Title)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
	s.Equal("No closures reported.", found.Documents[0].Text)
	s.InDelta(0.9, found.Documents[0].Score, 0.001)
	s.Equal("Chain law is off.", found.Documents[1].Text,
		"a result with no highlights falls back to its text")
}

func (s *ExaSuite) TestAQuestionWithNothingInItIsNotSent() {
	_, err := s.provider.Search(s.ctx, search.Query{Text: "  "})

	s.ErrorContains(err, "nothing to look for")
	s.Empty(s.web.path, "the provider was asked nothing")
}

func (s *ExaSuite) TestARefusedSearchSaysWhatCameBack() {
	s.web.status = http.StatusUnauthorized
	s.web.respond = `{"error":"bad key"}`

	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic"})

	s.ErrorContains(err, "401")
	s.ErrorContains(err, "bad key")
}

func (s *ExaSuite) TestReadingAPageReturnsItAsMarkdown() {
	s.web.respond = `{
		"results":[{"url":"https://example.com/pricing","title":"Pricing","text":"# Pricing\n\nA call costs a penny."}],
		"statuses":[{"id":"https://example.com/pricing","status":"success"}]
	}`

	page, err := s.provider.Read(s.ctx, "https://example.com/pricing")
	s.Require().NoError(err)

	s.Equal("/contents", s.web.path)
	s.Equal([]any{"https://example.com/pricing"}, s.web.body["urls"])
	s.Equal(true, s.web.body["text"])
	s.Equal("Pricing", page.Title)
	s.Contains(page.Text, "A call costs a penny.")
}

func (s *ExaSuite) TestAPageThatCouldNotBeReadIsAnErrorEvenThoughTheCallSucceeded() {
	// This endpoint reports a failed url in the body rather than in the status, so a
	// caller trusting the 200 would store an empty page as though it had worked.
	s.web.respond = `{
		"results":[],
		"statuses":[{"id":"https://example.com/gone","status":"error",
			"error":{"tag":"CRAWL_NOT_FOUND","httpStatusCode":404}}]
	}`

	_, err := s.provider.Read(s.ctx, "https://example.com/gone")

	s.ErrorContains(err, "CRAWL_NOT_FOUND")
	s.ErrorContains(err, "https://example.com/gone")
}

func (s *ExaSuite) TestAPageWithNothingOnItIsNotWorthStoring() {
	s.web.respond = `{"results":[{"url":"https://example.com/blank","title":"Blank","text":"  "}]}`

	_, err := s.provider.Read(s.ctx, "https://example.com/blank")

	s.ErrorContains(err, "nothing to read")
}
