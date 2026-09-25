//go:build integration

package xai

import (
	"bytes"
	"context"
	"encoding/json"
	"image"
	"image/color"
	"image/draw"
	"image/png"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/openaicompat"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type XAIIntegrationSuite struct {
	suite.Suite
	provider *openaicompat.LLM
}

func TestXAIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(XAIIntegrationSuite))
}

func (s *XAIIntegrationSuite) SetupSuite() {
	if os.Getenv(apiKeyEnvVar) == "" {
		s.T().Skip(apiKeyEnvVar + " not set")
	}
	provider, err := New(Options{})
	s.Require().NoError(err)
	s.provider = provider
}

func (s *XAIIntegrationSuite) TearDownSuite() {
	if s.provider != nil {
		s.provider.Close()
	}
}

func (s *XAIIntegrationSuite) ask(params llm.ResponseParams) (llm.Response, []llm.Event) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	stream, err := s.provider.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	s.Require().NoError(stream.Err())
	return stream.Response(), events
}

func (s *XAIIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	complete, events := s.ask(llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input:        []llm.Message{{Role: llm.User, Content: "What is the capital of France?"}},
	})

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal(llm.StatusCompleted, complete.Status)
	s.Positive(complete.Usage.InputTokens)
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.Usage.OutputTokensDetails.ReasoningTokens, "grok cannot be told not to think")

	var thinking bool
	for _, event := range events {
		if _, ok := event.(llm.ReasoningTextDelta); ok {
			thinking = true
		}
	}
	s.True(thinking, "the reasoning streams ahead of the answer")
}

func (s *XAIIntegrationSuite) TestCallsATool() {
	complete, _ := s.ask(llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Content: "What's the weather in Paris?"}},
		Tools: []llm.Tool{{
			Name:        "get_weather",
			Description: "Look up the weather somewhere",
			Parameters: map[string]any{
				"type":       "object",
				"properties": map[string]any{"city": map[string]any{"type": "string"}},
				"required":   []string{"city"},
			},
		}},
		ToolChoice: "required",
	})

	s.Require().Len(complete.ToolCalls, 1)
	call := complete.ToolCalls[0]
	s.Equal("get_weather", call.Name)
	var args struct{ City string }
	s.Require().NoError(json.Unmarshal([]byte(call.Arguments), &args))
	s.Contains(strings.ToLower(args.City), "paris")
}

func (s *XAIIntegrationSuite) TestReadsAnImage() {
	square := image.NewRGBA(image.Rect(0, 0, 64, 64))
	draw.Draw(square, square.Bounds(), &image.Uniform{C: color.RGBA{R: 255, A: 255}}, image.Point{}, draw.Src)
	var encoded bytes.Buffer
	s.Require().NoError(png.Encode(&encoded, square))

	complete, _ := s.ask(llm.ResponseParams{
		Input: []llm.Message{{Role: llm.User, Parts: []llm.ContentPart{
			{Text: "What colour is this image? One word."},
			{Image: &llm.ImagePart{MIME: "image/png", Data: encoded.Bytes()}},
		}}},
	})
	s.Contains(strings.ToLower(complete.OutputText), "red")
}
