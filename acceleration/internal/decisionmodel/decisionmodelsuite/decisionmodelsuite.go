//go:build integration

// Package decisionmodelsuite is what every decision model provider is held to against its real
// API: a question it can answer comes back answered, and the answer says what it read, which
// is what it is billed by.
//
// A provider suite embeds Suite, says how to build its provider, and inherits those tests.
// Anything only one provider does stays in that provider's own file.
package decisionmodelsuite

import (
	"context"
	"os"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
	// The providers need their credentials, which live in the repository's .env rather
	// than in the environment an editor happens to run a test with.
	_ "github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
)

// defaultTimeout bounds one judgement, looser than the providers' own.
const defaultTimeout = 30 * time.Second

// urgent is what every test asks: a question with a clear answer.
var urgent = decisionmodel.Request{
	State: "the payouts have been failing for three days and our customers are leaving",
	Questions: map[string]decisionmodel.Question{
		"urgent": decisionmodel.Noul("Does this convey urgency?", "", ""),
	},
}

// Suite is the shared behaviour. The fields are set where the suite is constructed rather
// than in a SetupSuite of the provider's own, which would shadow this one.
type Suite struct {
	suite.Suite

	// New builds a provider.
	New func() decisionmodel.Provider
	// Requires are the environment variables whose absence skips rather than fails.
	Requires []string
	// Timeout bounds one judgement.
	Timeout time.Duration

	// Provider is built once and never closed: a judgement is one request.
	Provider decisionmodel.Provider

	// answered is the question, asked once for every test, since each answer is billed.
	answered *decisionmodel.Result
}

func (s *Suite) SetupSuite() {
	s.Require().NotNil(s.New, "a provider suite has to say how to build its provider")
	for _, name := range s.Requires {
		if os.Getenv(name) == "" {
			s.T().Skipf("%s not set", name)
		}
	}
	if s.Timeout == 0 {
		s.Timeout = defaultTimeout
	}
	s.Provider = s.New()
}

// Answered is the question, asked on the first call and kept for the rest.
func (s *Suite) Answered() decisionmodel.Result {
	if s.answered == nil {
		ctx, cancel := context.WithTimeout(context.Background(), s.Timeout)
		defer cancel()
		answered, err := s.Provider.Classify(ctx, urgent)
		s.Require().NoError(err)
		s.answered = &answered
	}
	return *s.answered
}

// TestAQuestionWithAClearAnswerIsAnswered is the one that says the provider works at all.
func (s *Suite) TestAQuestionWithAClearAnswerIsAnswered() {
	answer, ok := s.Answered().Answers["urgent"]

	s.Require().True(ok, "the question should be answered")
	s.Greater(answer.Yes, 0.5, "three days of failed payouts is urgent")
}

// TestTheWorkIsBilled: the answer says what it read, which is what the judgement is billed by.
func (s *Suite) TestTheWorkIsBilled() {
	used := s.Answered().Used()

	s.False(used.Zero(), "the answer should say what it read")
}
