//go:build integration

package llmrouter

import (
	"context"
	"encoding/json"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/meta"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type MetaRouterIntegrationSuite struct {
	suite.Suite
	router  *Router
	session *Session
}

func TestMetaRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(MetaRouterIntegrationSuite))
}

func (s *MetaRouterIntegrationSuite) SetupSuite() {
	if os.Getenv("META_API_KEY") == "" {
		s.T().Skip("META_API_KEY not set")
	}
}

func (s *MetaRouterIntegrationSuite) SetupTest() {
	config := routing.ModalityConfig{Providers: []routing.ProviderConfig{{
		Provider: meta.ProviderName, Model: meta.DefaultModel,
		Languages: []string{"en"}, Realtime: true, Tier: routing.HighQuality,
	}}}
	var err error
	s.router, err = New(Options{Config: config, Registry: DefaultRegistry()})
	s.Require().NoError(err)
	s.session, err = s.router.Start(s.T().Context(), Request{
		CustomerID: "athena-integration", Target: "meta/" + meta.DefaultModel,
	})
	s.Require().NoError(err)
	s.Equal("meta", s.session.Provider())
	s.Equal("muse-spark-1.3", s.session.Model())
}

func (s *MetaRouterIntegrationSuite) TearDownTest() {
	if s.session != nil {
		s.session.Close()
	}
	if s.router != nil {
		s.router.Close()
	}
}

func (s *MetaRouterIntegrationSuite) ask(params llm.ResponseParams) (llm.Response, int) {
	ctx, cancel := context.WithTimeout(s.T().Context(), 90*time.Second)
	defer cancel()
	params.MaxOutputTokens = 1024
	params.Reasoning.Effort = "minimal"
	stream, err := s.session.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()
	deltas := 0
	for stream.Next() {
		if _, ok := stream.Current().(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Require().NoError(stream.Err())
	response := stream.Response()
	s.T().Logf("model=%s first_token_ms=%.1f duration_ms=%.1f input_tokens=%d output_tokens=%d", s.session.Model(), response.TimeToFirstTokenMs, response.DurationMs, response.Usage.InputTokens, response.Usage.OutputTokens)
	return response, deltas
}

func (s *MetaRouterIntegrationSuite) TestStreamsAnAnswerAndUsageThroughTheRouter() {
	response, deltas := s.ask(llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "What is the capital of France? Answer with one word."}},
	})
	s.Contains(strings.ToLower(response.OutputText), "paris")
	s.Positive(deltas)
	s.Positive(response.Usage.InputTokens)
	s.Positive(response.Usage.OutputTokens)
	s.Equal(llm.StatusCompleted, response.Status)
}

func (s *MetaRouterIntegrationSuite) TestToolCallAndResultRoundTrip() {
	input := []llm.Message{{Role: llm.User, Content: "Use lookup_project to get the status of project athena. Do not guess its status."}}
	tools := []llm.Tool{{Name: "lookup_project", Description: "Read the current status of an internal project.", Parameters: map[string]any{
		"type": "object", "properties": map[string]any{"project": map[string]any{"type": "string"}}, "required": []string{"project"},
	}}}
	response, _ := s.ask(llm.ResponseParams{Input: input, Tools: tools, ToolChoice: "auto"})
	s.Require().Len(response.ToolCalls, 1)
	call := response.ToolCalls[0]
	s.Equal("lookup_project", call.Name)
	var arguments struct {
		Project string `json:"project"`
	}
	s.Require().NoError(json.Unmarshal([]byte(call.Arguments), &arguments))
	s.Equal("athena", arguments.Project)
	s.NotEmpty(call.ID)
	input = append(input, llm.Message{Role: llm.Assistant, Content: response.OutputText, ToolCalls: response.ToolCalls}, llm.Message{
		Role: llm.ToolResult, ToolCallID: call.ID, Content: `{"project":"athena","status":"amber-742"}`,
	})
	final, deltas := s.ask(llm.ResponseParams{Input: input, Tools: tools})
	s.Contains(final.OutputText, "amber-742")
	s.Empty(final.ToolCalls)
	s.Positive(deltas)
}

func (s *MetaRouterIntegrationSuite) TestCancellationStopsGeneration() {
	ctx, cancel := context.WithTimeout(s.T().Context(), 90*time.Second)
	defer cancel()
	stream, err := s.session.Create(ctx, llm.ResponseParams{
		Input:     []llm.Message{{Role: llm.User, Content: "Write the numbers from 1 to 1000, one per line."}},
		Reasoning: llm.ReasoningParams{Effort: "minimal"}, MaxOutputTokens: 2048,
	})
	s.Require().NoError(err)
	defer stream.Close()
	var interrupted bool
	for stream.Next() {
		if _, ok := stream.Current().(llm.OutputTextDelta); ok && !interrupted {
			interrupted = true
			s.Require().NoError(stream.Close())
		}
	}
	s.Require().NoError(stream.Err())
	s.True(interrupted)
	s.Equal(llm.StatusCancelled, stream.Response().Status)
}
