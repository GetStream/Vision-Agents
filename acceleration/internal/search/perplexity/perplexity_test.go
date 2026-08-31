package perplexity

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

// web stands in for Perplexity's API so the wire contract can be tested without a key.
type web struct {
	server *httptest.Server

	path   string
	auth   string
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

type PerplexitySuite struct {
	suite.Suite
	ctx      context.Context
	web      *web
	provider *Provider
}

func TestPerplexitySuite(t *testing.T) {
	suite.Run(t, new(PerplexitySuite))
}

func (s *PerplexitySuite) SetupTest() {
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

// sonar returns a provider routed to a model that writes the answer rather than listing
// the pages.
func (s *PerplexitySuite) sonar() *Provider {
	provider, err := New(Options{
		APIKey: "test-key", Model: "sonar-pro", BaseURL: s.web.server.URL,
	})
	s.Require().NoError(err)
	return provider
}

func (s *PerplexitySuite) TestAKeyIsRequired() {
	s.T().Setenv("PERPLEXITY_API_KEY", "")

	_, err := New(Options{})

	s.ErrorContains(err, "PERPLEXITY_API_KEY")
}

func (s *PerplexitySuite) TestTheSearchModelAsksTheRankedIndex() {
	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70 in Colorado"})
	s.Require().NoError(err)

	s.Equal("Bearer test-key", s.web.auth)
	s.Equal("/search", s.web.path)
	s.Equal("traffic on I-70 in Colorado", s.web.body["query"])
}

func (s *PerplexitySuite) TestTheIndexReturnsPagesAndNoSummary() {
	s.web.respond = `{"results":[
		{"title":"COtrip","url":"https://cotrip.org","snippet":"No closures reported."},
		{"title":"CDOT","url":"https://codot.gov","snippet":"Chain law is off."}
	]}`

	found, err := s.provider.Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Empty(found.Answer)
	s.Require().Len(found.Documents, 2)
	s.Equal("COtrip", found.Documents[0].Title)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
	s.Equal("No closures reported.", found.Documents[0].Text)
}

func (s *PerplexitySuite) TestASonarModelAsksTheOneThatWritesTheAnswer() {
	// The model is what chooses the endpoint, which is how Sonar is an option a routing
	// target picks rather than a second provider to configure.
	s.web.respond = `{
		"choices":[{"message":{"content":"I-70 is clear through the Eisenhower Tunnel."}}],
		"search_results":[{"title":"COtrip","url":"https://cotrip.org","snippet":"No closures."}]
	}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Equal("/chat/completions", s.web.path)
	s.Equal("sonar-pro", s.web.body["model"])
	s.Equal("I-70 is clear through the Eisenhower Tunnel.", found.Answer)
	s.Require().Len(found.Documents, 1)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
}

func (s *PerplexitySuite) TestAnAnswerWithOnlyCitationsStillSaysWhereItCameFrom() {
	// An answer nobody can check is worse than no answer, so the older citations shape is
	// read when the newer one is absent.
	s.web.respond = `{
		"choices":[{"message":{"content":"The tunnel is open."}}],
		"citations":["https://cotrip.org"]
	}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Require().Len(found.Documents, 1)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
}

func (s *PerplexitySuite) TestAQuestionWithNothingInItIsNotSent() {
	_, err := s.provider.Search(s.ctx, search.Query{Text: "  "})

	s.ErrorContains(err, "nothing to look for")
	s.Empty(s.web.path, "the provider was asked nothing")
}

func (s *PerplexitySuite) TestARefusedSearchSaysWhatCameBack() {
	s.web.status = http.StatusUnauthorized
	s.web.respond = `{"error":{"message":"bad key"}}`

	_, err := s.provider.Search(s.ctx, search.Query{Text: "traffic"})

	s.ErrorContains(err, "401")
	s.ErrorContains(err, "bad key")
}

func (s *PerplexitySuite) TestTheLimitFallsBackToTheProvidersOwn() {
	provider, err := New(Options{APIKey: "k", BaseURL: s.web.server.URL, Limit: 2})
	s.Require().NoError(err)

	_, err = provider.Search(s.ctx, search.Query{Text: "traffic"})
	s.Require().NoError(err)

	s.Equal(float64(2), s.web.body["max_results"])
}
