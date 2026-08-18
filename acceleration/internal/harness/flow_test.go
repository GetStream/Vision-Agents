package harness

import (
	"context"
	"fmt"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
)

type FlowSuite struct {
	suite.Suite
	ctx context.Context

	model *stubLLM
	flow  *flow

	mu      sync.Mutex
	decided []Decided
}

func TestFlowSuite(t *testing.T) {
	suite.Run(t, new(FlowSuite))
}

func (s *FlowSuite) SetupTest() {
	s.ctx = context.Background()
	s.model = newStubLLM()
	s.decided = nil

	emitter := NewEmitter(16)
	drained := make(chan struct{})
	go func() {
		defer close(drained)
		for event := range emitter.Events() {
			decided, ok := event.(Decided)
			if !ok {
				continue
			}
			s.mu.Lock()
			s.decided = append(s.decided, decided)
			s.mu.Unlock()
		}
	}()

	s.flow = newFlow(stubSession(&s.Suite, s.ctx, s.model), emitter, slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() {
		_ = s.flow.Close()
		emitter.Close()
		<-drained
	})
}

// decisions is what the controller has decided so far.
func (s *FlowSuite) decisions() []Decided {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Decided(nil), s.decided...)
}

func (s *FlowSuite) TestTheConversationIsQuotedRatherThanReplayed() {
	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID:           "candidate-1",
		Instructions: "be brief",
		Participant:  "Alex",
		History: []llm.Message{
			{Role: llm.User, Content: "what is the capital of France"},
			{Role: llm.Assistant, Content: "Paris."},
		},
		Text: "and of Spain",
	}))

	asked := s.model.requests()
	s.Require().Len(asked, 1)
	s.Require().Lenf(asked[0].Messages, 1,
		"a conversation replayed as turns is one the controller answers instead of judging: %v",
		asked[0].Messages)
	question := asked[0].Messages[0].Content
	s.Contains(question, "Caller: what is the capital of France")
	s.Contains(question, "Agent: Paris.")
	s.Contains(question, `"and of Spain"`, "the words being judged have to be in there")
	s.True(asked[0].JSON, "the answer is parsed, so prose is not an option")
}

func (s *FlowSuite) TestOnlyTheRecentConversationIsShown() {
	var history []llm.Message
	for turn := range flowHistory {
		history = append(history,
			llm.Message{Role: llm.User, Content: fmt.Sprintf("question %d", turn)},
			llm.Message{Role: llm.Assistant, Content: fmt.Sprintf("answer %d", turn)},
		)
	}

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "candidate-1", History: history, Text: "and now"}))

	question := s.model.requests()[0].Messages[0].Content
	s.NotContains(question, "question 0", "an old turn does not decide whose the floor is")
	s.Contains(question, fmt.Sprintf("answer %d", flowHistory-1))
}

func (s *FlowSuite) TestAnUnreadableAnswerStillEarnsTheCallerAReply() {
	// The controller answering the caller instead of classifying them is exactly how a
	// question used to go unanswered: nobody was left to notice it had been asked.
	s.model.answers["candidate-1"] = "With three days, I would start in the Alfama district."

	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID: "candidate-1", Participant: "Alex", Text: "what should I not miss",
	}))

	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, settleFor,
		5*time.Millisecond, "the caller's turn was dropped")
	decided := s.decisions()[0]
	s.Equal(Respond, decided.Disposition)
	s.Equal(Continue, decided.Floor, "an unreadable answer is no reason to cut the agent off")
	s.NoError(decided.Error())
}

func (s *FlowSuite) TestEveryConversationalDecisionParses() {
	for _, disposition := range []Disposition{Wait, Ignore, Respond, Clarify} {
		for _, floor := range []Floor{Stop, Shorten, Continue} {
			answer, err := parseFlow(`{"disposition":"` + string(disposition) +
				`","floor":"` + string(floor) + `"}`)

			s.Require().NoError(err)
			s.Equal(disposition, answer.Disposition)
			s.Equal(floor, answer.Floor)
		}
	}
}

func (s *FlowSuite) TestMarkdownFencesDoNotHideAValidDecision() {
	answer, err := parseFlow("```json\n" +
		`{"disposition":"respond","floor":"continue"}` +
		"\n```")

	s.Require().NoError(err)
	s.Equal(Respond, answer.Disposition)
	s.Equal(Continue, answer.Floor)
}

func (s *FlowSuite) TestInventedDecisionsAreRejected() {
	_, err := parseFlow(`{"disposition":"guess","floor":"continue"}`)
	s.ErrorContains(err, "invalid flow disposition")

	_, err = parseFlow(`{"disposition":"respond","floor":"talk-over"}`)
	s.ErrorContains(err, "invalid floor decision")
}

func (s *FlowSuite) TestExtraSpeechIsRejected() {
	_, err := parseFlow(`Sure: {"disposition":"respond","floor":"continue"}`)
	s.ErrorContains(err, "decode flow decision")
}
