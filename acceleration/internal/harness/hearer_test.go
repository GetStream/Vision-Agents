package harness

import (
	"context"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
)

// stubHearer rules as a test says, after a test's delay, or not at all until its turn is
// abandoned.
type stubHearer struct {
	ruling Decided
	err    error
	hang   bool
}

func (h stubHearer) Hear(ctx context.Context, _ FlowTurn) (Decided, error) {
	if h.hang {
		<-ctx.Done()
		return Decided{}, ctx.Err()
	}
	return h.ruling, h.err
}

// respondingJudge says every turn is a finished thought for the agent.
type respondingJudge struct{}

func (respondingJudge) Classify(context.Context, decisionmodel.Request) (decisionmodel.Result, error) {
	answers := map[string]decisionmodel.Answer{}
	for fact := range judgeQuestions() {
		answers[fact] = decisionmodel.Answer{}
	}
	answers["addressed_to_agent"] = decisionmodel.Answer{Yes: 1}
	answers["finished"] = decisionmodel.Answer{Yes: 1}
	return decisionmodel.Result{Answers: answers}, nil
}

func (respondingJudge) Close() error { return nil }

// HearerSuite runs the flow controller with a Hearer before its judge.
type HearerSuite struct {
	suite.Suite

	emitter *Emitter
	drained chan struct{}

	mu      sync.Mutex
	decided []Decided
}

func TestHearerSuite(t *testing.T) {
	suite.Run(t, new(HearerSuite))
}

func (s *HearerSuite) SetupTest() {
	s.decided = nil
	s.emitter = NewEmitter(16)
	s.drained = make(chan struct{})
	go func() {
		defer close(s.drained)
		for event := range s.emitter.Events() {
			if decided, ok := event.(Decided); ok {
				s.mu.Lock()
				s.decided = append(s.decided, decided)
				s.mu.Unlock()
			}
		}
	}()
	s.T().Cleanup(func() {
		s.emitter.Close()
		<-s.drained
	})
}

func (s *HearerSuite) ruling(hearer Hearer, turn FlowTurn) Decided {
	s.T().Helper()
	f := newFlow(nil, s.emitter, slog.New(slog.DiscardHandler))
	f.judge, f.hearer = respondingJudge{}, hearer
	defer func() { s.Require().NoError(f.Close()) }()
	s.Require().NoError(f.Decide(turn))
	var got Decided
	s.Require().Eventually(func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		if len(s.decided) == 0 {
			return false
		}
		got = s.decided[len(s.decided)-1]
		return got.CandidateID == turn.ID
	}, settleFor, 5*time.Millisecond, "nobody ruled")
	return got
}

func (s *HearerSuite) TestTheHearersRulingIsTheFlowControllers() {
	got := s.ruling(stubHearer{ruling: Decided{Disposition: Respond, Floor: Continue, Tool: "book_crossing"}},
		FlowTurn{ID: "t", Text: "book me on the eight o'clock", Ended: true})
	s.Equal(Heard, got.Ruler)
	s.Equal(Respond, got.Disposition)
	s.Equal("book_crossing", got.Tool)
}

func (s *HearerSuite) TestAHearersWaitSaysWhenToAskAgain() {
	got := s.ruling(stubHearer{ruling: Decided{Disposition: Wait, Floor: Continue, Again: 300 * time.Millisecond}},
		FlowTurn{ID: "t", Text: "my number is four one", Ended: true})
	s.Equal(Wait, got.Disposition)
	s.Equal(300*time.Millisecond, got.Again)
}

func (s *HearerSuite) TestATurnTheHearerCannotRuleOnIsTheJudges() {
	got := s.ruling(stubHearer{err: errors.New("refused")}, FlowTurn{ID: "t", Text: "hello", Ended: true})
	s.Equal(Primary, got.Ruler)
	s.Equal(Respond, got.Disposition)
}

// Words over the agent are ruled on within the overlap's deadline, a hung hearer included.
func (s *HearerSuite) TestAHungHearerLeavesTheJudgeTheRestOfTheDeadline() {
	start := time.Now()
	got := s.ruling(stubHearer{hang: true}, FlowTurn{ID: "t", Text: "wait no", Speaking: true, Unfinished: true})
	s.Equal(Primary, got.Ruler)
	s.GreaterOrEqual(time.Since(start), overlapDeadline/3)
	s.Less(time.Since(start), overlapDeadline)
}
