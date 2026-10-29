package harness

import (
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/systemone"
	"github.com/GetStream/Vision-Agents/acceleration/internal/decisionmodel/typesafe"
)

// JudgeSuite runs the flow controller on a decision model, served by a stub of the System One
// endpoint that answers whatever the test sets.
type JudgeSuite struct {
	suite.Suite

	server  *httptest.Server
	flow    *flow
	emitter *Emitter
	drained chan struct{}

	mu      sync.Mutex
	answers string
	states  []map[string]any
	decided []Decided
}

func TestJudgeSuite(t *testing.T) {
	suite.Run(t, new(JudgeSuite))
}

func (s *JudgeSuite) SetupTest() {
	s.answers, s.states, s.decided = "", nil, nil
	s.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var asked struct {
			State map[string]any `json:"state"`
		}
		_ = json.Unmarshal(raw, &asked)
		s.mu.Lock()
		s.states = append(s.states, asked.State)
		answers := s.answers
		s.mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		_, _ = fmt.Fprintf(w, `{"model":"jev-test","answers":{%s},"usage":{"input_tokens":10}}`, answers)
	}))
	client, err := typesafe.New(systemone.Options{APIKey: "test", BaseURL: s.server.URL})
	s.Require().NoError(err)

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
	s.flow = newFlow(nil, s.emitter, slog.New(slog.DiscardHandler))
	s.flow.judge = client
	s.T().Cleanup(func() {
		_ = s.flow.Close()
		s.emitter.Close()
		<-s.drained
		s.server.Close()
	})
}

// answer has the stub answer every fact with these probabilities of yes, and no for the rest.
// A fact named in missing is left unanswered.
func (s *JudgeSuite) answer(yes map[string]float64, missing ...string) {
	var parts []string
	for fact := range judgeQuestions() {
		if slices.Contains(missing, fact) {
			continue
		}
		parts = append(parts, fmt.Sprintf(`%q:{"type":"noul","noul":%v}`, fact, yes[fact]))
	}
	s.mu.Lock()
	s.answers = strings.Join(parts, ",")
	s.mu.Unlock()
}

func (s *JudgeSuite) ruling(turn FlowTurn) Decided {
	s.T().Helper()
	s.Require().NoError(s.flow.Decide(turn))
	var got Decided
	s.Require().Eventually(func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		if len(s.decided) == 0 {
			return false
		}
		got = s.decided[len(s.decided)-1]
		return got.CandidateID == turn.ID
	}, time.Second, 5*time.Millisecond, "the judge never ruled")
	return got
}

func (s *JudgeSuite) TestACompleteThoughtForTheAgentIsAnswered() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "A table for four at seven, please.", Ended: true})

	s.Require().NoError(got.Err)
	s.Equal(Respond, got.Disposition)
	s.Equal(Primary, got.Ruler, "the judge has no model to fall back on")
	s.Equal("A table for four at seven, please.", s.states[0]["heard"])
}

func (s *JudgeSuite) TestARequestTheTranscriberHasNotEndedWaitsForTheEnd() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9})
	said := FlowTurn{ID: "t1", Participant: "caller", Text: "A table for three this Friday at six fifteen"}

	first := s.ruling(said)
	said.ID, said.Ended = "t2", true
	ended := s.ruling(said)

	s.Equal(Wait, first.Disposition)
	s.Equal(Respond, ended.Disposition)
}

func (s *JudgeSuite) TestACoughOverTheAgentNeverTakesTheFloor() {
	s.answer(map[string]float64{"addressed_to_agent": 0.9, "non_speech": 0.8, "objects": 0.7})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "[cough]", Speaking: true,
		Reply: "Your table is booked for"})

	s.Equal(Continue, got.Floor)
	s.Equal(Ignore, got.Disposition)
}

func (s *JudgeSuite) TestWordsForSomebodyElseAreIgnored() {
	s.answer(map[string]float64{"addressed_to_agent": 0.1, "finished": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "Mom, can we go to the park after?"})

	s.Equal(Ignore, got.Disposition)
}

func (s *JudgeSuite) TestANumberStillBeingReadOutIsWaitedOn() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.6, "still_growing": 0.8})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "My PIN is one zero four"})

	s.Equal(Wait, got.Disposition)
}

func (s *JudgeSuite) TestAnUnfinishedInterruptionDecidesOnlyTheFloor() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "objects": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "wait, make it", Speaking: true,
		Unfinished: true, Reply: "Booked for four"})

	s.Equal(Wait, got.Disposition)
	s.Equal(Stop, got.Floor)
}

func (s *JudgeSuite) TestAnAdditionShortensTheReply() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9, "adds_to_request": 0.8, "objects": 0.2})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "and Saturday as well", Speaking: true,
		Reply: "Friday at seven is free"})

	s.Equal(Respond, got.Disposition)
	s.Equal(Shorten, got.Floor)
}

func (s *JudgeSuite) TestARecordedMenuIsWaitedOnThoughItIsNotForTheAgent() {
	s.answer(map[string]float64{"recorded_menu": 0.96, "addressed_to_agent": 0.37, "finished": 0.7})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "For balances, press one. For payments,"})

	s.Equal(Wait, got.Disposition)
}

func (s *JudgeSuite) TestACallerStillGivingWhatTheyPromisedIsWaitedOn() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9, "listing": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "My name is Ines Moreau."})

	s.Equal(Wait, got.Disposition)
}

func (s *JudgeSuite) TestAPlainRequestJevIsUnsureOfIsAnsweredRatherThanQuestioned() {
	s.answer(map[string]float64{"addressed_to_agent": 0.97, "finished": 0.95, "ambiguous": 0.56})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller",
		Text: "I'd like to move my Monday visit to Wednesday afternoon please.", Ended: true})

	s.Equal(Respond, got.Disposition)
}

func (s *JudgeSuite) TestAPauseJevIsUnsureEndsTheThoughtIsWaitedOn() {
	s.answer(map[string]float64{"addressed_to_agent": 0.96, "finished": 0.62})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "Ines Moreau, born November twenty third 1979"})

	s.Equal(Wait, got.Disposition)
}

func (s *JudgeSuite) TestAGreetingOnItsOwnIsWaitedOn() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9, "greeting_only": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "Hi."})

	s.Equal(Wait, got.Disposition)
}

func (s *JudgeSuite) TestAHalfSaidCorrectionStopsTheAgentButIsNotAnswered() {
	s.answer(map[string]float64{"addressed_to_agent": 0.92, "finished": 0.33, "objects": 0.9})

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Speaking: true,
		Reply: "A party of six at seven this Saturday.", Text: "A table for six at seven. Actually, no"})

	s.Equal(Wait, got.Disposition)
	s.Equal(Stop, got.Floor)
}

func (s *JudgeSuite) TestAnUnansweredFactIsAFailure() {
	s.answer(map[string]float64{"addressed_to_agent": 0.95, "finished": 0.9}, "objects")

	got := s.ruling(FlowTurn{ID: "t1", Participant: "caller", Text: "Hello?"})

	s.Error(got.Err)
}
