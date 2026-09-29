package locateanything

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

type LocateAnythingSuite struct {
	suite.Suite
}

func TestLocateAnythingSuite(t *testing.T) {
	suite.Run(t, new(LocateAnythingSuite))
}

func (s *LocateAnythingSuite) SetupTest() {
	s.T().Setenv(apiKeyEnvVar, "")
	s.T().Setenv(baseURLEnvVar, "")
}

func (s *LocateAnythingSuite) TestAnUndeployedModelFailsToBuild() {
	s.T().Setenv(apiKeyEnvVar, "k")

	_, err := New(Options{})
	s.ErrorContains(err, baseURLEnvVar+" is required")
	s.ErrorContains(err, "deploy/locate-anything", "the error says where the recipe is")
}

func (s *LocateAnythingSuite) TestCredentialsComeFromTheEnvironmentWhenNotGiven() {
	s.T().Setenv(baseURLEnvVar, "https://model-abc.api.baseten.co/environments/production/sync/v1")

	_, err := New(Options{})
	s.ErrorContains(err, apiKeyEnvVar+" is required")

	s.T().Setenv(apiKeyEnvVar, "from-env")
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.Equal(ProviderName, provider.Provider())
	s.Equal(defaultModel, provider.Model())
}

func (s *LocateAnythingSuite) TestTheImageReachesTheDeploymentAndTheBoxesComeBack() {
	const boxes = "<ref>car</ref><box><10><20><110><220></box>"
	var sent struct {
		Model    string `json:"model"`
		Messages []struct {
			Content []struct {
				Type     string `json:"type"`
				Text     string `json:"text"`
				ImageURL struct {
					URL string `json:"url"`
				} `json:"image_url"`
			} `json:"content"`
		} `json:"messages"`
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		s.Equal("/v1/chat/completions", r.URL.Path)
		s.Equal("Bearer k", r.Header.Get("Authorization"))
		s.NoError(json.NewDecoder(r.Body).Decode(&sent))

		chunk, err := json.Marshal(map[string]any{
			"id": "c1", "object": "chat.completion.chunk", "model": defaultModel,
			"choices": []map[string]any{{"index": 0, "delta": map[string]any{"content": boxes}, "finish_reason": "stop"}},
			"usage":   map[string]any{"prompt_tokens": 900, "completion_tokens": 12, "total_tokens": 912},
		})
		s.NoError(err)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = fmt.Fprintf(w, "data: %s\n\ndata: [DONE]\n\n", chunk)
	}))
	s.T().Cleanup(server.Close)

	provider, err := New(Options{APIKey: "k", BaseURL: server.URL + "/v1"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = provider.Close() })

	stream, err := provider.Create(context.Background(), llm.ResponseParams{ID: "r1", Input: []llm.Message{{
		Role: llm.User,
		Parts: []llm.ContentPart{
			{Image: &llm.ImagePart{MIME: "image/jpeg", Data: []byte{0xff, 0xd8}}},
			{Text: "Locate all the instances that matches the following description: car."},
		},
	}}})
	s.Require().NoError(err)
	response, err := llm.Collect(stream)
	s.Require().NoError(err)

	s.Equal(boxes, response.OutputText)
	s.Equal(int64(900), response.Usage.InputTokens, "usage is what the deployment is billed by")
	s.Equal(defaultModel, sent.Model)
	s.Require().Len(sent.Messages, 1)
	s.Require().Len(sent.Messages[0].Content, 2)
	s.Equal("image_url", sent.Messages[0].Content[0].Type)
	s.Equal("data:image/jpeg;base64,/9g=", sent.Messages[0].Content[0].ImageURL.URL)
	s.Equal("text", sent.Messages[0].Content[1].Type)
}
