package tavily

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

// web stands in for Tavily's API so the wire contract can be tested without a key.
type web struct {
	server *httptest.Server

	path   string
	auth   string
	body   map[string]any
	status int
	// respond is the JSON the stub answers with. Empty results are the default because a
	// search that found nothing is the case worth not crashing on.
	respond string
}

func newWeb() *web {
	stub := &web{status: http.StatusOK, respond: `{"answer":"","results":[]}`}
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

type TavilySuite struct {
	suite.Suite
	ctx      context.Context
	web      *web
	provider *Provider
}

func TestTavilySuite(t *testing.T) {
	suite.Run(t, new(TavilySuite))
}

func (s *TavilySuite) SetupTest() {
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

func (s *TavilySuite) TestAKeyIsRequired() {
	s.T().Setenv("TAVILY_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "TAVILY_API_KEY")
}

func (s *TavilySuite) TestASearchAsksForASummaryAlongsideTheResults() {
	// The summary is the point of this provider: a caller on the phone wants the sentence
	// rather than the pages, and a second round trip to write it would cost more than the
	// search did.
	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70 in Colorado"})
	s.Require().NoError(err)

	s.Equal("Bearer test-key", s.web.auth)
	s.Equal("/search", s.web.path)
	s.Equal("traffic on I-70 in Colorado", s.web.body["query"])
	s.Equal(true, s.web.body["include_answer"])
}

func (s *TavilySuite) TestSearchReturnsTheSummaryAndWhereItCameFrom() {
	s.web.respond = `{
		"answer":"I-70 is clear through the Eisenhower Tunnel.",
		"results":[
			{"title":"COtrip","url":"https://cotrip.org","content":"No closures reported.","score":0.9},
			{"title":"CDOT","url":"https://codot.gov","content":"Chain law is off.","score":0.4}
		]
	}`

	found, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Equal("I-70 is clear through the Eisenhower Tunnel.", found.Answer)
	s.Require().Len(found.Documents, 2)
	s.Equal("COtrip", found.Documents[0].Title)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
	s.Equal("No closures reported.", found.Documents[0].Text)
	s.InDelta(0.9, found.Documents[0].Score, 0.001)
}

func (s *TavilySuite) TestAQuestionWithNothingInItIsNotSent() {
	_, err := s.provider.Search(s.ctx, search.Query{Text: "  "})

	s.ErrorContains(err, "nothing to look for")
	s.Empty(s.web.path, "the provider was asked nothing")
}

func (s *TavilySuite) TestARefusedSearchSaysWhatCameBack() {
	s.web.status = http.StatusUnauthorized
	s.web.respond = `{"detail":"bad key"}`

	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic"})

	s.ErrorContains(err, "401")
	s.ErrorContains(err, "bad key")
}

func (s *TavilySuite) TestTheDepthIsTheModelThatWasRouted() {
	// Advanced crawls the pages it finds rather than trusting the index snippet, which is
	// worth roughly double the latency to some callers and not to others. Routing is what
	// decides, so the model has to reach the wire.
	provider, err := New(Options{
		APIKey: "k", Model: ModelAdvanced, BaseURL: s.web.server.URL,
	})
	s.Require().NoError(err)

	_, err = provider.Search(s.ctx, search.Query{Text: "traffic"})
	s.Require().NoError(err)

	s.Equal(ModelAdvanced, provider.Model())
	s.Equal("advanced", s.web.body["search_depth"])
}

func (s *TavilySuite) TestADepthTavilyDoesNotSearchAtIsRefused() {
	_, err := New(Options{APIKey: "k", Model: "exhaustive"})

	s.ErrorContains(err, "exhaustive")
}

func (s *TavilySuite) TestTheLimitFallsBackToTheProvidersOwn() {
	provider, err := New(Options{APIKey: "k", BaseURL: s.web.server.URL, Limit: 2})
	s.Require().NoError(err)

	_, err = provider.Search(s.ctx, search.Query{Text: "traffic"})
	s.Require().NoError(err)

	s.Equal(float64(2), s.web.body["max_results"])
}
