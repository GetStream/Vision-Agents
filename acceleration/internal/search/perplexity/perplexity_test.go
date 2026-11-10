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

	"github.com/GetStream/Vision-Agents/acceleration/internal/pricing"
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
		APIKey: "test-key", Model: "sonar", BaseURL: s.web.server.URL,
	})
	s.Require().NoError(err)
	return provider
}

func (s *PerplexitySuite) TestSonarProIsRefusedSinceTheAgentAPIHasNoSuchModel() {
	_, err := New(Options{APIKey: "test-key", Model: "sonar-pro", BaseURL: s.web.server.URL})

	s.ErrorContains(err, `"sonar-pro" is neither the search index nor sonar`)
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
	// Probed on 3b7a53f97: /chat/completions with "sonar-pro", the Sonar Chat Completions
	// Perplexity stopped supporting on 2026-09-27.
	s.web.respond = `{
		"output":[
			{"type":"search_results","queries":["I-70 traffic"],
			 "results":[{"id":1,"title":"COtrip","url":"https://cotrip.org","snippet":"No closures.","source":"web"}]},
			{"type":"message","role":"assistant","status":"completed",
			 "content":[{"type":"output_text","text":"I-70 is clear through the Eisenhower Tunnel.","annotations":[]}]}
		]
	}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "traffic on I-70"})
	s.Require().NoError(err)

	s.Equal("/v1/agent", s.web.path)
	s.Equal("perplexity/sonar", s.web.body["model"])
	s.Equal("traffic on I-70", s.web.body["input"])
	s.Equal([]any{map[string]any{"type": "web_search", "max_results": float64(5)}}, s.web.body["tools"])
	s.Equal(map[string]any{"type": "web_search"}, s.web.body["tool_choice"], "it searches rather than answer from memory")
	s.Equal("I-70 is clear through the Eisenhower Tunnel.", found.Answer)
	s.Require().Len(found.Documents, 1)
	s.Equal("https://cotrip.org", found.Documents[0].URL)
	s.Equal("No closures.", found.Documents[0].Text)
	s.Zero(found.VendorCostMicros(), "an answer that says no cost has none")
}

func (s *PerplexitySuite) TestASonarAnswerIsPricedOnItsTokensAndSearches() {
	// Probed on 3b7a53f97: requests 1 whatever the answer read, which priced a long answer
	// like a short one.
	s.web.respond = `{
		"output":[{"type":"message","content":[{"type":"output_text","text":"The tunnel is open."}]}],
		"usage":{"input_tokens":1200,"input_tokens_details":{"cached_tokens":200,"cache_read_input_tokens":200},
			"output_tokens":90,"tool_calls_details":{"search_web":{"invocation":2}},
			"cost":{"input_cost":0.00025,"cache_read_cost":0.0000125,"output_cost":0.000225,
				"tool_calls_cost":0.005,"total_cost":0.0054875,"currency":"USD"}}
	}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "is the tunnel open"})
	s.Require().NoError(err)

	s.EqualValues(5_488, found.VendorCostMicros())
	s.Equal(pricing.Quantities{
		pricing.InputTokens:       1_000,
		pricing.CachedInputTokens: 200,
		pricing.OutputTokens:      90,
		pricing.Requests:          2,
	}, found.Used(), "fresh input, cached input, output, and one request per web search")
}

func (s *PerplexitySuite) TestASonarAnswerThatSearchedNothingBillsOnlyItsTokens() {
	s.web.respond = `{
		"output":[{"type":"message","content":[{"type":"output_text","text":"Paris."}]}],
		"usage":{"input_tokens":40,"output_tokens":3}
	}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "capital of France"})
	s.Require().NoError(err)

	s.Equal(pricing.Quantities{
		pricing.InputTokens: 40, pricing.CachedInputTokens: 0, pricing.OutputTokens: 3, pricing.Requests: 0,
	}, found.Used())
	s.Empty(found.Documents)
}

func (s *PerplexitySuite) TestASonarAnswerKeepsNoMoreSourcesThanAsked() {
	// Probed live on 3b7a53f97 with a limit of 2: 15 sources, from two searches.
	s.web.respond = `{"output":[
		{"type":"search_results","results":[{"url":"https://a.example"},{"url":"https://b.example"}]},
		{"type":"search_results","results":[{"url":"https://c.example"}]},
		{"type":"message","content":[{"type":"output_text","text":"Open."}]}
	]}`

	found, err := s.sonar().Search(s.ctx, search.Query{Text: "is it open", Limit: 2})
	s.Require().NoError(err)

	s.Equal([]any{map[string]any{"type": "web_search", "max_results": float64(2)}}, s.web.body["tools"])
	s.Len(found.Documents, 2)
}

func (s *PerplexitySuite) TestTheIndexIsStillPricedAsOneRequest() {
	found, err := s.provider.Search(s.ctx, search.Query{Text: "traffic"})
	s.Require().NoError(err)

	s.Equal(pricing.Quantities{pricing.Requests: 1}, found.Used())
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
