package gemma

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type GemmaSuite struct {
	suite.Suite
}

func TestGemmaSuite(t *testing.T) {
	suite.Run(t, new(GemmaSuite))
}

func (s *GemmaSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.T().Setenv(basetenAPIKeyEnvVar, "")
	s.T().Setenv(baseURLEnvVar, "")
}

// bearerSent builds the provider against a local deployment, asks it one question and
// returns the Authorization header the deployment received.
func (s *GemmaSuite) bearerSent(options Options) string {
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"id\":\"t\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
	}))
	s.T().Cleanup(server.Close)
	s.T().Setenv(baseURLEnvVar, server.URL+"/v1")

	provider, err := New(options)
	s.Require().NoError(err)
	stream, err := provider.Create(s.T().Context(), llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "hi"}},
	})
	s.Require().NoError(err)
	_, err = llm.Collect(stream)
	s.Require().NoError(err)
	return authorization
}

func (s *GemmaSuite) TestAnUndeployedGemmaFailsToBuild() {
	// Failing here is what makes routing move to the next candidate, so a capability
	// shortcut still resolves while nobody has pushed the deployment.
	s.T().Setenv(apiKeyEnvVar, "k")

	_, err := New(Options{})
	s.ErrorContains(err, baseURLEnvVar+" is required")
	s.ErrorContains(err, "deploy/gemma-4", "the error says where the recipe is")
}

func (s *GemmaSuite) TestWithNoKeyTheErrorNamesBothVariables() {
	s.T().Setenv(baseURLEnvVar, "https://model-abc.api.baseten.co/environments/production/sync/v1")

	_, err := New(Options{})
	s.ErrorContains(err, "GEMMA_API_KEY or BASETEN_API_KEY is required")
}

func (s *GemmaSuite) TestTheGemmaKeyIsTheBearerToken() {
	s.T().Setenv(apiKeyEnvVar, "gemma-key")
	s.T().Setenv(basetenAPIKeyEnvVar, "baseten-key")

	s.Equal("Bearer gemma-key", s.bearerSent(Options{}))
}

func (s *GemmaSuite) TestWithoutAGemmaKeyTheBasetenKeyIsTheBearerToken() {
	// Unchanged from before GEMMA_API_KEY existed: the base sent "Bearer baseten-key" here.
	s.T().Setenv(basetenAPIKeyEnvVar, "baseten-key")

	s.Equal("Bearer baseten-key", s.bearerSent(Options{}))
}

func (s *GemmaSuite) TestAGivenKeyBeatsBothVariables() {
	s.T().Setenv(apiKeyEnvVar, "gemma-key")
	s.T().Setenv(basetenAPIKeyEnvVar, "baseten-key")

	s.Equal("Bearer given-key", s.bearerSent(Options{APIKey: "given-key"}))
}

func (s *GemmaSuite) TestStatsUseTheRoutingNameNotTheQualifiedModelID() {
	provider, err := New(Options{APIKey: "k", BaseURL: "http://x/v1", Model: "gemma-4-26B-A4B-it"})
	s.Require().NoError(err)

	s.Equal("gemma-4-26B-A4B-it", provider.Model())
}

func (s *GemmaSuite) TestModelIsQualifiedWithItsOwnerForVLLM() {
	s.Equal("google/gemma-4-26B-A4B-it", upstreamModel("gemma-4-26B-A4B-it"))
	s.Equal("google/gemma-4-31B-it", upstreamModel("google/gemma-4-31B-it"),
		"an already-qualified id is left alone")
}
