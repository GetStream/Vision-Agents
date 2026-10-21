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

	model   *stubLLM
	flow    *flow
	emitter *Emitter
	drained chan struct{}

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

	s.emitter = NewEmitter(16)
	s.drained = make(chan struct{})
	go func() {
		defer close(s.drained)
		for event := range s.emitter.Events() {
			decided, ok := event.(Decided)
			if !ok {
				continue
			}
			s.mu.Lock()
			s.decided = append(s.decided, decided)
			s.mu.Unlock()
		}
	}()

	s.flow = newFlow(stubSession(&s.Suite, s.ctx, s.model), s.emitter, slog.New(slog.DiscardHandler))
	s.T().Cleanup(func() {
		_ = s.flow.Close()
		s.emitter.Close()
		<-s.drained
	})
}

// decisions is what the controller has decided so far.
func (s *FlowSuite) decisions() []Decided {
	s.mu.Lock()
	defer s.mu.Unlock()
	return append([]Decided(nil), s.decided...)
}

// waitAsked waits until the model has been asked n times. Create runs on a goroutine, so
// asserting on requests immediately after Decide would race it.
func (s *FlowSuite) waitAsked(n int) []llm.ResponseParams {
	s.T().Helper()
	s.Require().Eventually(func() bool { return len(s.model.requests()) >= n }, settleFor,
		5*time.Millisecond, "the controller was never asked")
	return s.model.requests()
}

func (s *FlowSuite) TestWordsThatCallSomebodyByAFamilyNameAreForThem() {
	// Voicebench's child said "Mom, can I get the chicken nuggets there?" at the caller's end,
	// Gemma ruled it respond, and the agent answered the child.
	s.Contains(flowInstructions, `Words that open by calling somebody "Mom", "Dad", "honey" or "sweetie"`)
	s.Contains(flowInstructions, "are addressed to them, not the agent: choose ignore")
}

func (s *FlowSuite) TestACallerWhoSaidTheyWouldGiveEverythingIsWaitedOn() {
	// "I'll give you everything up front so we don't go back and forth" was ruled respond,
	// and the agent started talking ten times inside the caller's monologue.
	s.Contains(flowInstructions, "Once the caller has said they will give everything at once, choose wait at each pause")
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

	asked := s.waitAsked(1)
	s.Require().Lenf(asked[0].Input, 1,
		"a conversation replayed as turns is one the controller answers instead of judging: %v",
		asked[0].Input)
	question := asked[0].Input[0].Content
	s.Contains(question, "Caller: what is the capital of France")
	s.Contains(question, "Agent: Paris.")
	s.Contains(question, `"and of Spain"`, "the words being judged have to be in there")
	s.Equal(llm.FormatJSONObject, asked[0].Text.Format,
		"the answer is parsed, so prose is not an option")
	s.Equal(512, asked[0].MaxOutputTokens,
		"thinking models spend the budget before the JSON, and 32 was only enough for a fragment")
}

func (s *FlowSuite) TestAnOverlappingReplyIsQuotedSoTheControllerCanHold() {
	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID:          "candidate-1",
		Participant: "Alex",
		Text:        "cough",
		Speaking:    true,
		Reply:       "party of four at 7:30",
	}))

	question := s.waitAsked(1)[0].Input[0].Content
	s.Contains(question, `is speaking right now and has so far said "party of four at 7:30"`)
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

	question := s.waitAsked(1)[0].Input[0].Content
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

func (s *FlowSuite) TestAnUnreadableOverlapStopsTheAgent() {
	s.model.answers["over-1"] = `{"dis`

	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID: "over-1", Participant: "Alex", Text: "oh my god stop", Unfinished: true, Speaking: true,
	}))

	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, settleFor,
		5*time.Millisecond, "the overlap was never ruled")
	decided := s.decisions()[0]
	s.Equal(Wait, decided.Disposition)
	s.Equal(Stop, decided.Floor, "truncated JSON while the caller talks over the agent is a stop")
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

func (s *FlowSuite) TestCancelAbandonsACreateThatHasNotReturned() {
	hold := make(chan struct{})
	s.model.holdCreate = hold

	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID: "old", Participant: "Alex", Text: "okay",
	}))
	s.waitAsked(1)

	s.Require().NoError(s.flow.Cancel("old"))
	s.Require().Eventually(func() bool {
		return len(s.model.requests()) == 1 && len(s.decisions()) == 0
	}, settleFor, 5*time.Millisecond, "the cancelled create still produced a ruling")
	s.Empty(s.decisions(), "a ruling about words that have moved on must not be acted on")
}

func (s *FlowSuite) TestCancelStartsTheWaitingTurnBeforeCreateReturns() {
	hold := make(chan struct{})
	s.model.holdCreate = hold

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "old", Participant: "Alex", Text: "okay"}))
	s.waitAsked(1)

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "mid", Participant: "Alex", Text: "okay wait"}))
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "new", Participant: "Alex", Text: "okay wait stop"}))
	time.Sleep(20 * time.Millisecond)
	s.Len(s.model.requests(), 1, "Cerebras must not see a pile of superseded asks")

	s.Require().NoError(s.flow.Cancel("old"))
	asked := s.waitAsked(2)
	s.Equal("new", asked[1].ID, "the follow-up must not wait for the cancelled create to return")
	for _, d := range s.decisions() {
		s.NotContains([]string{"old", "mid"}, d.CandidateID,
			"superseded words must not be ruled on")
	}
}

func (s *FlowSuite) TestCancelOfAHungCreateLetsTheNextDecideStart() {
	hold := make(chan struct{})
	s.model.holdCreate = hold

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "old", Participant: "Alex", Text: "okay"}))
	s.waitAsked(1)
	s.Require().NoError(s.flow.Cancel("old"))

	s.Require().NoError(s.flow.Decide(FlowTurn{
		ID: "new", Participant: "Alex", Text: "tell me something else",
	}))
	asked := s.waitAsked(2)
	s.Equal("new", asked[1].ID, "a hung create that was cancelled must not keep the mailbox busy")
}

func (s *FlowSuite) TestCancelCanOverlapTheReturningStream() {
	// Release each blocked Create concurrently with Cancel and repeat to exercise the stream
	// handoff under the race detector.
	for round := range 100 {
		id := fmt.Sprintf("round-%d", round)
		hold := make(chan struct{})
		s.model.mu.Lock()
		s.model.holdCreate = hold
		s.model.mu.Unlock()
		s.Require().NoError(s.flow.Decide(FlowTurn{ID: id, Participant: "Alex", Text: "okay"}))
		s.waitAsked(round + 1)

		cancelled := make(chan error, 1)
		go func() { cancelled <- s.flow.Cancel(id) }()
		close(hold)
		s.Require().NoError(<-cancelled)
	}
}

func (s *FlowSuite) TestNothingIsAskedOnceTheFlowIsClosed() {
	s.Require().NoError(s.flow.Close())

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "late", Participant: "Alex", Text: "okay"}))
	s.flow.running.Wait()
	s.emitter.Close()
	<-s.drained

	s.Empty(s.model.requests(), "a decision started after Close would outlive the sessions it was given")
	s.Empty(s.decisions())
}

func (s *FlowSuite) TestSlowOverlapYieldsBeforeTheOriginalReplyFinishes() {
	s.model.holdCreate = make(chan struct{})
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "over-slow", Participant: "Alex", Text: "stop, give me one sentence", Unfinished: true, Speaking: true}))
	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, 2*time.Second, 5*time.Millisecond)
	decision := s.decisions()[0]
	s.Equal("over-slow", decision.CandidateID)
	s.Equal(Wait, decision.Disposition)
	s.Equal(Stop, decision.Floor)
	s.NoError(decision.Error())
}

func (s *FlowSuite) TestSupersededSlowOverlapCannotInterruptLater() {
	s.model.holdCreate = make(chan struct{})
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "over-old", Participant: "Alex", Text: "stop", Unfinished: true, Speaking: true}))
	s.Require().NoError(s.flow.Cancel("over-old"))
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "over-new", Participant: "Alex", Text: "wait, make it shorter", Unfinished: true, Speaking: true}))
	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, 2*time.Second, 5*time.Millisecond)
	s.Equal("over-new", s.decisions()[0].CandidateID)
	s.Equal(Stop, s.decisions()[0].Floor)
}

func (s *FlowSuite) TestSlowOverlapStreamAlsoYieldsTheFloor() {
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "over-stream", Participant: "Alex", Text: "stop, give me one sentence", Unfinished: true, Speaking: true}))
	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, 2*time.Second, 5*time.Millisecond)
	s.Equal(Stop, s.decisions()[0].Floor)
	s.Equal(Wait, s.decisions()[0].Disposition)
	s.NoError(s.decisions()[0].Error())
}

func (s *FlowSuite) TestADecisionThatOutlivesTheDeadlineStillRulesOnTheTurn() {
	// Staging, 2026-10-19 22:01Z: the decision stream opened, no token came within 3 s, the
	// deadline closed it, and the caller's question was never ruled on, so never answered.
	s.flow.deadline = 50 * time.Millisecond

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "slow", Participant: "Alex", Text: "what time do you open"}))

	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, settleFor,
		5*time.Millisecond, "a decision the deadline closed left the caller's turn unruled")
	decided := s.decisions()[0]
	s.Equal("slow", decided.CandidateID)
	s.ErrorIs(decided.Error(), context.DeadlineExceeded,
		"the agent answers a finished turn whose ruling failed, so the timeout has to reach it")
	s.model.mu.Lock()
	script := s.model.scripts["slow"]
	s.model.mu.Unlock()
	s.Require().NotNil(script, "the stream had started, so the deadline is what closed it")
	s.True(script.Abandoned())
}

func (s *FlowSuite) TestAFastDecisionIsRuledAsAnswered() {
	s.model.answers["fast"] = `{"disposition":"respond","floor":"continue"}`

	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "fast", Participant: "Alex", Text: "what time do you open"}))

	s.Require().Eventually(func() bool { return len(s.decisions()) == 1 }, settleFor,
		5*time.Millisecond, "the decision was never ruled")
	decided := s.decisions()[0]
	s.Equal("fast", decided.CandidateID)
	s.Equal(Respond, decided.Disposition)
	s.Equal(Continue, decided.Floor)
	s.NoError(decided.Error())
}

func (s *FlowSuite) TestAStreamClosedBeforeTheDeadlineIsNotRuledOn() {
	s.Require().NoError(s.flow.Decide(FlowTurn{ID: "closed", Participant: "Alex", Text: "what time do you open"}))
	var stream *llm.Stream
	s.Require().Eventually(func() bool {
		s.flow.mu.Lock()
		defer s.flow.mu.Unlock()
		if asked := s.flow.pending["closed"]; asked != nil {
			stream = asked.stream
		}
		return stream != nil
	}, settleFor, 5*time.Millisecond, "the stream never started")

	s.Require().NoError(stream.Close())
	s.flow.running.Wait()
	s.emitter.Close()
	<-s.drained

	s.Empty(s.decisions(), "only the deadline turns a closed stream into a ruling")
}

func (s *FlowSuite) TestARulingSaysWhoseAnswerItIs() {
	ruled := func(id, answer string, writtenBy [2]string) Ruler {
		s.model.mu.Lock()
		s.model.answers[id], s.model.writtenBy = answer, writtenBy
		s.model.mu.Unlock()
		s.Require().NoError(s.flow.Decide(FlowTurn{ID: id, Participant: "Alex", Text: "two"}))
		s.Require().Eventually(func() bool { return len(s.decisions()) > 0 && s.decisions()[len(s.decisions())-1].CandidateID == id },
			settleFor, 5*time.Millisecond, "no ruling came back")
		return s.decisions()[len(s.decisions())-1].Ruler
	}
	respond := `{"disposition":"respond","floor":"continue"}`

	s.Equal(Primary, ruled("own", respond, [2]string{}))
	s.Empty(ruled("unusable", "not json", [2]string{}), "a default the controller fell back on is nobody's answer")
	s.Equal(Fallback, ruled("hedged", respond, [2]string{"", "another-model"}))
	s.Equal(Fallback, ruled("elsewhere", respond, [2]string{"another-provider", ""}))
}
