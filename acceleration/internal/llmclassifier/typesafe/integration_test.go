//go:build integration

package typesafe

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llmclassifier"
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// jevDecided is how far from the middle an answer has to be for this to count as a
// judgement rather than a shrug. The point of these tests is that the model is being asked
// something it can answer, not that it agrees with a number tuned against it.
const jevDecided = 0.3

type TypeSafeIntegrationSuite struct {
	suite.Suite
	ctx    context.Context
	client *Client
}

func TestTypeSafeIntegrationSuite(t *testing.T) {
	suite.Run(t, new(TypeSafeIntegrationSuite))
}

func (s *TypeSafeIntegrationSuite) SetupSuite() {
	if os.Getenv("TYPESAFE_API_KEY") == "" {
		s.T().Skip("TYPESAFE_API_KEY is not set")
	}

	var cancel context.CancelFunc
	s.ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	s.T().Cleanup(cancel)

	client, err := New(Options{})
	s.Require().NoError(err)
	s.client = client
}

func (s *TypeSafeIntegrationSuite) TestANoulAnswersBothWaysOnTheSameQuestion() {
	// The two states are the guardrail's own two cases: a question about the product and
	// a question about lunch. If the model cannot separate those, nothing built on it can.
	policy := "Only questions about Stream's chat, video and feeds SDKs, their APIs, " +
		"and software development using them."

	question := map[string]llmclassifier.Question{
		"violates": llmclassifier.Noul(
			"Is `message` a request that falls outside what `policy` permits?",
			"It is about something the policy does not cover.",
			"It is a request the policy permits.",
		),
	}

	onTopic, err := s.client.Classify(s.ctx, llmclassifier.Request{
		State: map[string]string{
			"policy":  policy,
			"message": "How do I render a message list with stream-chat-react?",
		},
		Questions: question,
	})
	s.Require().NoError(err)

	offTopic, err := s.client.Classify(s.ctx, llmclassifier.Request{
		State: map[string]string{
			"policy":  policy,
			"message": "How do I make a pizza from scratch?",
		},
		Questions: question,
	})
	s.Require().NoError(err)

	s.Lessf(onTopic.Answers["violates"].Yes, 0.5-jevDecided,
		"an SDK question read as a violation at %.2f", onTopic.Answers["violates"].Yes)
	s.Greaterf(offTopic.Answers["violates"].Yes, 0.5+jevDecided,
		"a pizza question read as permitted at %.2f", offTopic.Answers["violates"].Yes)
}

func (s *TypeSafeIntegrationSuite) TestTheAnswerSaysWhichVersionMadeItAndWhatItRead() {
	// An alias moves when a release ships, so a threshold tuned against one version needs
	// the version that answered rather than the name that was asked.
	answered, err := s.client.Classify(s.ctx, llmclassifier.Request{
		State: "the payouts have been failing for three days",
		Questions: map[string]llmclassifier.Question{
			"urgent": llmclassifier.Noul("Does this convey urgency?", "", ""),
		},
	})
	s.Require().NoError(err)

	s.NotEqual(DefaultModel, answered.Model, "an alias should resolve to a version")
	s.Positive(answered.Usage.InputTokens)
}
