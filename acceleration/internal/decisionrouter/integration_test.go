//go:build integration

package decisionrouter

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// decided is how far from the middle an answer has to be to count as a judgement rather
// than a shrug. The point is that the model is being asked something it can answer, not
// that it agrees with a number tuned against one model.
const decided = 0.3

// keys is what each vendor needs set before its models are asked anything.
var keys = map[string]string{
	"typesafe":   "TYPESAFE_API_KEY",
	"openrouter": "OPENROUTER_API_KEY",
	"perplexity": "PERPLEXITY_API_KEY",
}

type DecisionModelIntegrationSuite struct {
	suite.Suite
}

func TestDecisionModelIntegrationSuite(t *testing.T) {
	suite.Run(t, new(DecisionModelIntegrationSuite))
}

func (s *DecisionModelIntegrationSuite) TestEveryDeclaredModelAnswersBothWaysOnTheSameQuestion() {
	// A model declared in router.yaml is a candidate a guardrail can be routed to, so each
	// one has to separate a question about the product from a question about lunch.
	config, err := routing.DefaultConfig()
	s.Require().NoError(err)

	policy := "Only questions about Stream's chat, video and feeds SDKs, their APIs, " +
		"and software development using them."
	question := map[string]decisionmodel.Question{
		"violates": decisionmodel.Noul(
			"Is `message` a request that falls outside what `policy` permits?",
			"It is about something the policy does not cover.",
			"It is a request the policy permits.",
		),
	}
	ask := func(ctx context.Context, provider decisionmodel.Provider, message string) float64 {
		answered, err := provider.Classify(ctx, decisionmodel.Request{
			State:     map[string]string{"policy": policy, "message": message},
			Questions: question,
		})
		s.Require().NoError(err)
		s.Positive(answered.Usage.InputTokens)
		return answered.Answers["violates"].Yes
	}

	registry := DefaultRegistry()
	for _, declared := range config[routing.DecisionModel].Providers {
		s.Run(declared.Name(), func() {
			if os.Getenv(keys[declared.Provider]) == "" {
				s.T().Skip(keys[declared.Provider] + " is not set")
			}
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()

			provider, err := registry.Build(declared.Provider, routing.Spec{Model: declared.Model})
			s.Require().NoError(err)

			onTopic := ask(ctx, provider, "How do I render a message list with stream-chat-react?")
			offTopic := ask(ctx, provider, "How do I make a pizza from scratch?")

			s.Lessf(onTopic, 0.5-decided, "an SDK question read as a violation at %.2f", onTopic)
			s.Greaterf(offTopic, 0.5+decided, "a pizza question read as permitted at %.2f", offTopic)
		})
	}
}
