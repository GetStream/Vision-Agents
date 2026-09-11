//go:build integration

package openai

import (
	"bytes"
	"context"
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
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

type OpenAIIntegrationSuite struct {
	suite.Suite
}

func TestOpenAIIntegrationSuite(t *testing.T) {
	suite.Run(t, new(OpenAIIntegrationSuite))
}

func (s *OpenAIIntegrationSuite) SetupSuite() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		s.T().Skip("OPENAI_API_KEY not set")
	}
}

func (s *OpenAIIntegrationSuite) start(options Options) *LLM {
	provider, err := New(options)
	s.Require().NoError(err)
	s.T().Cleanup(func() { provider.Close() })
	return provider
}

// ask runs one request to the end. A provider failure ends the stream, so a rejected request reports what went wrong rather than timing out.
func (s *OpenAIIntegrationSuite) ask(provider *LLM, params llm.ResponseParams) (llm.Response, []llm.Event) {
	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	stream, err := provider.Create(ctx, params)
	s.Require().NoError(err)
	defer stream.Close()

	var events []llm.Event
	for stream.Next() {
		events = append(events, stream.Current())
	}
	s.Require().NoError(stream.Err())
	return stream.Response(), events
}

func (s *OpenAIIntegrationSuite) TestAnswersAndReportsWhatItCost() {
	provider := s.start(Options{})

	complete, events := s.ask(provider, llm.ResponseParams{
		ID:           "c1",
		Instructions: "Answer with a single word and no punctuation.",
		Input: []llm.Message{
			{Role: llm.User, Content: "What is the capital of France?"},
		},
		MaxOutputTokens: 32,
	})

	s.Contains(strings.ToLower(complete.OutputText), "paris")
	s.Equal("c1", complete.ID)
	s.Positive(complete.Usage.InputTokens)
	s.Positive(complete.Usage.OutputTokens)
	s.Positive(complete.TimeToFirstTokenMs)
	s.Equal(llm.StatusCompleted, complete.Status)

	var deltas int
	for _, event := range events {
		if _, ok := event.(llm.OutputTextDelta); ok {
			deltas++
		}
	}
	s.Positive(deltas, "the answer should stream rather than arrive in one lump")
}

func (s *OpenAIIntegrationSuite) TestConversationHistoryIsHonoured() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Instructions: "Answer with a single number and nothing else.",
		Input: []llm.Message{
			{Role: llm.User, Content: "My favourite number is 7. Remember it."},
			{Role: llm.Assistant, Content: "Noted."},
			{Role: llm.User, Content: "What is my favourite number?"},
		},
		MaxOutputTokens: 32,
	})

	s.Contains(complete.OutputText, "7", "the whole conversation travels with the request")
}

func (s *OpenAIIntegrationSuite) TestATruncatedAnswerSaysWhyItStopped() {
	provider := s.start(Options{})

	complete, _ := s.ask(provider, llm.ResponseParams{
		Input:           []llm.Message{{Role: llm.User, Content: "Write a long essay about the sea."}},
		MaxOutputTokens: 16,
	})

	s.Equal(llm.StatusIncomplete, complete.Status)
	s.Equal(llm.ReasonMaxOutputTokens, complete.IncompleteReason)
	s.NotEmpty(complete.OutputText)
}

func (s *OpenAIIntegrationSuite) TestTwoImageInputsAreUnderstoodInOrder() {
	provider := s.start(Options{})
	parts := []llm.ContentPart{{Text: "Name the color of the first image, then the second. Reply only red, blue."}}
	for _, c := range []color.Color{color.RGBA{R: 255, A: 255}, color.RGBA{B: 255, A: 255}} {
		picture := image.NewRGBA(image.Rect(0, 0, 64, 64))
		draw.Draw(picture, picture.Bounds(), image.NewUniform(c), image.Point{}, draw.Src)
		var data bytes.Buffer
		s.Require().NoError(png.Encode(&data, picture))
		parts = append(parts, llm.ContentPart{Image: &llm.ImagePart{MIME: "image/png", Data: data.Bytes()}})
	}
	complete, _ := s.ask(provider, llm.ResponseParams{Input: []llm.Message{{Role: llm.User, Parts: parts}}, MaxOutputTokens: 128})
	answer := strings.ToLower(complete.OutputText)
	s.Contains(answer, "red")
	s.Contains(answer, "blue")
	s.Less(strings.Index(answer, "red"), strings.Index(answer, "blue"))
}
