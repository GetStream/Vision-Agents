//go:build integration

package stsrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sts"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// STSRouterIntegrationSuite routes a real conversation through the built-in config, which
// is where the registry, the config and a vendor meet for the first time.
type STSRouterIntegrationSuite struct {
	suite.Suite
	router *Router
}

func TestSTSRouterIntegrationSuite(t *testing.T) {
	suite.Run(t, new(STSRouterIntegrationSuite))
}

func (s *STSRouterIntegrationSuite) SetupSuite() {
	if os.Getenv("OPENAI_API_KEY") == "" {
		s.T().Skip("OPENAI_API_KEY not set")
	}
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)
	s.router, err = New(Options{Config: config[routing.STS], Registry: DefaultRegistry()})
	s.Require().NoError(err)
}

func (s *STSRouterIntegrationSuite) TearDownSuite() {
	if s.router != nil {
		s.router.Close()
	}
}

func (s *STSRouterIntegrationSuite) TestATypedTurnIsAnsweredThroughTheRouter() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Minute)
	defer cancel()
	on := true

	session, err := s.router.Start(ctx, Request{
		CustomerID: "integration",
		Target:     "openai/gpt-realtime-2",
		Options: options.STS{
			Instructions:     "Answer in one short sentence.",
			Text:             &on,
			OutputTranscript: &on,
		},
	})
	s.Require().NoError(err)
	defer session.Close()

	s.Require().NoError(session.SendText("Say hello.", sts.Participant{ID: "caller"}))

	deadline := time.After(time.Minute)
	var spoke bool
	for {
		select {
		case event, open := <-session.Events():
			s.Require().True(open, "the session ended before the model answered")
			switch typed := event.(type) {
			case sts.AudioChunk:
				spoke = true
			case sts.ResponseComplete:
				s.True(spoke, "the model should have spoken before the reply settled")
				s.Greater(typed.AudioDurationMs, 0.0)
				return
			case sts.Error:
				s.Require().False(typed.Fatal, "provider error: %v", typed.Err)
			}
		case <-deadline:
			s.FailNow("timed out waiting for the reply")
		}
	}
}
