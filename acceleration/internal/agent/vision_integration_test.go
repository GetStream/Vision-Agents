//go:build integration

package agent

import (
	"context"
	_ "embed"
	"log/slog"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// order7341 is a red circle and a blue square above the words "ORDER 7341", in black.
//
//go:embed testdata/order_7341.png
var order7341 []byte

// visionAnswerWithin bounds one answer about a picture from a real model.
const visionAnswerWithin = time.Minute

// VisionIntegrationSuite sends a picture to an agent in writing on a real model that sees,
// with no vision skill, and reads what it answers.
type VisionIntegrationSuite struct {
	suite.Suite
	ctx context.Context
	llm *llmrouter.Router
}

func TestVisionIntegrationSuite(t *testing.T) {
	suite.Run(t, new(VisionIntegrationSuite))
}

func (s *VisionIntegrationSuite) SetupSuite() {
	if os.Getenv("GOOGLE_API_KEY") == "" {
		s.T().Skip("GOOGLE_API_KEY not set")
	}
	s.ctx = context.Background()
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	s.llm, err = llmrouter.New(llmrouter.Options{
		Config: config[routing.LLM], Registry: llmrouter.DefaultRegistry(), Logger: slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
}

func (s *VisionIntegrationSuite) TestAnAgentWithoutAVisionSkillReadsThePictureItIsSent() {
	reader, err := New(Options{
		Text:         true,
		Instructions: "Answer questions about what you are shown.",
		CustomerID:   "vision-integration",
		LLM:          s.llm,
		LLMTarget:    "gemini/gemini-3.8-flash",
		Logger:       slog.New(slog.DiscardHandler),
	})
	s.Require().NoError(err)
	s.Require().NoError(reader.Join(s.ctx))
	events := collect(reader)
	defer func() { _ = reader.Close(); <-events.done }()

	turnID, err := reader.RespondTo(s.ctx, "What is in this image? Name each shape and its colour, and read any text exactly.",
		[]llm.ImagePart{{MIME: "image/png", Data: order7341}})
	s.Require().NoError(err)

	var answer string
	s.Require().Eventually(func() bool {
		for _, event := range events.seen() {
			if responded, ok := event.(Responded); ok && responded.TurnID == turnID {
				answer = strings.ToLower(responded.Text)
				return true
			}
		}
		return false
	}, visionAnswerWithin, 100*time.Millisecond, "the agent never answered the picture")
	s.Contains(answer, "7341", "the text in the picture was not read")
	s.Contains(answer, "red")
	s.Contains(answer, "blue")
}
