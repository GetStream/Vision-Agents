package harness

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel"
)

// stubHearer rules as a test says, after a test's delay, or not at all until its turn is
// abandoned. overAgent, when set, is how long it has to rule on words over the agent.
type stubHearer struct {
	ruling    Decided
	err       error
	hang      bool
	overAgent time.Duration
}

func (h stubHearer) HearDeadline(turn FlowTurn) time.Duration {
	if h.overAgent > 0 && turn.Unfinished && turn.Speaking {
		return h.overAgent
	}
	return HearDeadline(turn)
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
	// logger is the flow controller's.
	logger *slog.Logger

	mu      sync.Mutex
	decided []Decided
}

func TestHearerSuite(t *testing.T) {
	suite.Run(t, new(HearerSuite))
}

func (s *HearerSuite) SetupTest() {
	s.decided = nil
	s.logger = slog.New(slog.DiscardHandler)
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
	f := newFlow(nil, s.emitter, s.logger)
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
	got := s.ruling(stubHearer{ruling: Decided{Disposition: Respond, Floor: Continue}},
		FlowTurn{ID: "t", Text: "book me on the eight o'clock", Ended: true})
	s.Equal(Heard, got.Ruler)
	s.Equal(Respond, got.Disposition)
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

// A turn the hearer could not rule on is said where a deployment's logs show it, with why:
// said at debug, nothing at info or above told a language model ruling in the hearer's place,
// its round trip in front of the answer, from the hearer ruling.
func (s *HearerSuite) TestATurnTheHearerCannotRuleOnIsLogged() {
	var logs bytes.Buffer
	s.logger = slog.New(slog.NewJSONHandler(&logs, &slog.HandlerOptions{Level: slog.LevelInfo}))
	s.ruling(stubHearer{err: errors.New("the server refused the agent")}, FlowTurn{ID: "t", Text: "hello", Ended: true})

	var said []map[string]any
	for line := range bytes.Lines(logs.Bytes()) {
		var record map[string]any
		s.Require().NoError(json.Unmarshal(line, &record))
		if record["msg"] == "the hearer did not rule, the flow controller will" {
			said = append(said, record)
		}
	}
	s.Require().Len(said, 1, "the turn left to the flow controller was not logged: %s", logs.String())
	s.Equal("WARN", said[0]["level"])
	s.Equal("t", said[0]["candidate"])
	s.Equal("the server refused the agent", said[0]["error"])
}

// Words over the agent are ruled on within the overlap's deadline, a hung hearer included.
func (s *HearerSuite) TestAHungHearerLeavesTheJudgeTheRestOfTheDeadline() {
	start := time.Now()
	got := s.ruling(stubHearer{hang: true}, FlowTurn{ID: "t", Text: "wait no", Speaking: true, Unfinished: true})
	s.Equal(Primary, got.Ruler)
	s.GreaterOrEqual(time.Since(start), overlapDeadline/3)
	s.Less(time.Since(start), overlapDeadline)
}

// A Hearer whose server needs longer over the agent has that long, and the judge what is
// left of the overlap's deadline.
func (s *HearerSuite) TestAHearerSaysHowLongItHasOverTheAgent() {
	start := time.Now()
	got := s.ruling(stubHearer{hang: true, overAgent: 400 * time.Millisecond}, FlowTurn{ID: "t", Text: "wait no", Speaking: true, Unfinished: true})
	s.Equal(Primary, got.Ruler)
	s.GreaterOrEqual(time.Since(start), 400*time.Millisecond)
	s.Less(time.Since(start), overlapDeadline)
}
