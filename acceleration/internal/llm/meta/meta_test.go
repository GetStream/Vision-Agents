package meta

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type MetaSuite struct{ suite.Suite }

func TestMetaSuite(t *testing.T) { suite.Run(t, new(MetaSuite)) }

func (s *MetaSuite) SetupTest() { s.T().Setenv(apiKeyEnvVar, "") }

func (s *MetaSuite) TestRejectsMissingCredentialsAndUnverifiedModels() {
	_, err := New(Options{})
	s.ErrorContains(err, "META_API_KEY is required")
	s.T().Setenv(apiKeyEnvVar, "test-key")
	provider, err := New(Options{})
	s.Require().NoError(err)
	defer provider.Close()
	s.Equal(DefaultModel, provider.Model())
	s.Equal(ProviderName, provider.Provider())
	s.False(provider.Capabilities().StreamsReasoning)
	_, err = New(Options{Model: "muse-spark-1.3-contributor"})
	s.ErrorContains(err, "unsupported model")
	_, err = New(Options{ReasoningEffort: "none"})
	s.ErrorContains(err, "reasoning effort")
}

func (s *MetaSuite) TestUnsupportedToolChoicesAndEffortsFailBeforeNetworkAccess() {
	provider, err := New(Options{APIKey: "test", BaseURL: "http://127.0.0.1:1"})
	s.Require().NoError(err)
	defer provider.Close()
	for _, choice := range []string{"none", "required", "some_function"} {
		_, err := provider.Create(context.Background(), llm.ResponseParams{ToolChoice: choice})
		s.ErrorContains(err, "tool choice")
	}
	_, err = provider.Create(context.Background(), llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "hello"}},
		Reasoning: llm.ReasoningParams{Effort: "ultra"},
	})
	s.ErrorContains(err, "reasoning effort")
}

func (s *MetaSuite) TestExplicitAndDefaultEffortUseTheMetaWireProtocol() {
	for _, effort := range []string{"", "minimal", "high"} {
		s.Run(effort, func() {
			wanted := effort
			if wanted == "" {
				wanted = "low"
			}
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				var request struct {
					Model           string `json:"model"`
					ReasoningEffort string `json:"reasoning_effort"`
					Stream          bool   `json:"stream"`
				}
				if json.NewDecoder(r.Body).Decode(&request) != nil || request.Model != DefaultModel || request.ReasoningEffort != wanted || !request.Stream || r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "Bearer test" {
					http.Error(w, "invalid Meta request", http.StatusBadRequest)
					return
				}
				w.Header().Set("Content-Type", "text/event-stream")
				fmt.Fprint(w, "data: {\"id\":\"test\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello Athena\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n")
			}))
			defer server.Close()
			provider, err := New(Options{APIKey: "test", BaseURL: server.URL + "/v1"})
			s.Require().NoError(err)
			defer provider.Close()
			stream, err := provider.Create(s.T().Context(), llm.ResponseParams{
				Input:     []llm.Message{{Role: llm.User, Content: "hello"}},
				Reasoning: llm.ReasoningParams{Effort: effort},
			})
			s.Require().NoError(err)
			response, err := llm.Collect(stream)
			s.Require().NoError(err)
			s.Equal("Hello Athena", response.OutputText)
		})
	}
}
