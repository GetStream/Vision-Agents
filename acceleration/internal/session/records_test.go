package session

import (
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
)

// heldRecorder keeps what a session tried to write, so a test can read the conversation the
// way somebody loading it back would.
type heldRecorder struct {
	mu        sync.Mutex
	responses []store.AgentResponse
	finished  []finishedTurn
	items     []store.AgentResponseItem
}

type finishedTurn struct {
	id      string
	status  string
	failure string
}

func (r *heldRecorder) Responding(row store.AgentResponse) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.responses = append(r.responses, row)
}

func (r *heldRecorder) Responded(id, status, failure string, _ time.Time) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.finished = append(r.finished, finishedTurn{id: id, status: status, failure: failure})
}

func (r *heldRecorder) Item(item store.AgentResponseItem) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.items = append(r.items, item)
}

// kinds is the conversation as a reader would see it: what happened, in order.
func (r *heldRecorder) kinds() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	kinds := make([]string, 0, len(r.items))
	for _, item := range r.items {
		kinds = append(kinds, item.Kind)
	}
	return kinds
}

type RecordSuite struct {
	suite.Suite
	held    *heldRecorder
	session *Session
}

func TestRecordSuite(t *testing.T) {
	suite.Run(t, new(RecordSuite))
}

func (s *RecordSuite) SetupTest() {
	s.held = &heldRecorder{}
	s.session = &Session{
		id:      "session-1",
		spec:    Spec{CustomerID: "acme"},
		records: s.held,
	}
}

// took plays a whole turn through the mapping.
func (s *RecordSuite) took(events ...Event) {
	for _, event := range events {
		s.session.record(event)
	}
}

func (s *RecordSuite) TestAPlainTurnIsAQuestionAndAnAnswer() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "Is Stream better than Sendbird?"},
		agent.ResponseDelta{TurnID: "turn-1", Text: "It "},
		agent.ResponseDelta{TurnID: "turn-1", Text: "is."},
		agent.Responded{TurnID: "turn-1", Text: "It is."},
	)

	s.Require().Len(s.held.responses, 1)
	s.Equal("session-1", s.held.responses[0].SessionID)
	s.Equal("acme", s.held.responses[0].CustomerID)
	s.Equal("Is Stream better than Sendbird?", s.held.responses[0].Said)

	// The deltas are not items. A hundred fragments of one sentence are the sentence.
	s.Equal([]string{store.ItemSaid, store.ItemAnswer}, s.held.kinds())
	s.Require().Len(s.held.finished, 1)
	s.Equal(store.ResponseCompleted, s.held.finished[0].status)
}

func (s *RecordSuite) TestItemsKeepTheOrderTheyHappenedIn() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "What is in my cart?"},
		agent.ToolStarted{TurnID: "turn-1", ID: "call-1", Tool: "cart"},
		agent.ToolRan{TurnID: "turn-1", ID: "call-1", Tool: "cart", Result: "two things"},
		agent.Responded{TurnID: "turn-1", Text: "Two things."},
	)

	s.Equal([]string{store.ItemSaid, store.ItemToolCall, store.ItemToolResult, store.ItemAnswer},
		s.held.kinds())
	for i, item := range s.held.items {
		s.Equal(i, item.Ordinal, "the writer assigns the position, not the database")
	}
}

func (s *RecordSuite) TestATurnThatDelegatesStaysOpenUntilItComesBack() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "Work out the refund"},
		agent.Delegated{TurnID: "turn-1", TaskID: "task-1", Skill: "think", Prompt: "compute it"},
		// PendingWork means the agent will speak again about the same question.
		agent.Responded{TurnID: "turn-1", Text: "Let me work that out.", PendingWork: true},
	)
	s.Empty(s.held.finished, "a turn with work outstanding is not over")

	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "resumption"},
		agent.Responded{TurnID: "turn-1", Text: "It is fourteen pounds."},
	)

	s.Require().Len(s.held.responses, 1, "coming back is the same turn, not a second one")
	s.Require().Len(s.held.finished, 1)
	s.Equal(store.ResponseCompleted, s.held.finished[0].status)
	// The resumption is not recorded as something the person said: they asked once.
	s.Equal([]string{store.ItemSaid, store.ItemThought, store.ItemAnswer, store.ItemAnswer},
		s.held.kinds())
}

func (s *RecordSuite) TestTwoTurnsAtOnceDoNotMixTheirItems() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "first"},
		agent.Responding{TurnID: "turn-2", Prompt: "second"},
		agent.ToolRan{TurnID: "turn-2", Tool: "cart", Result: "for the second"},
		agent.Responded{TurnID: "turn-1", Text: "answer to the first"},
		agent.Responded{TurnID: "turn-2", Text: "answer to the second"},
	)

	s.Require().Len(s.held.responses, 2)
	first, second := s.held.responses[0].ID, s.held.responses[1].ID
	s.NotEqual(first, second)

	byResponse := map[string][]string{}
	for _, item := range s.held.items {
		byResponse[item.ResponseID] = append(byResponse[item.ResponseID], item.Kind)
	}
	s.Equal([]string{store.ItemSaid, store.ItemAnswer}, byResponse[first])
	s.Equal([]string{store.ItemSaid, store.ItemToolResult, store.ItemAnswer}, byResponse[second])
}

func (s *RecordSuite) TestAnInterruptedTurnIsCancelledRatherThanFailed() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "a long question"},
		agent.Interrupted{TurnID: "turn-1"},
	)

	s.Require().Len(s.held.finished, 1)
	// The caller stopped it. Nothing went wrong, and what was already said still counts.
	s.Equal(store.ResponseCancelled, s.held.finished[0].status)
	s.Empty(s.held.finished[0].failure)
}

func (s *RecordSuite) TestARefusedTurnIsRecordedAsTheRefusalItIs() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "something against the policy"},
		agent.Blocked{TurnID: "turn-1", Reason: "off topic", Probability: 0.9},
		agent.Responded{TurnID: "turn-1", Text: "I cannot help with that."},
	)

	// The reply the caller heard was the policy's refusal, so it is not filed as the
	// agent's answer: a conversation read back should not show the agent agreeing.
	s.Equal([]string{store.ItemSaid, store.ItemBlocked, store.ItemBlocked}, s.held.kinds())
	s.Equal("off topic", s.held.items[1].Payload["reason"])
}

func (s *RecordSuite) TestAFailureFailsWhateverWasInFlight() {
	s.took(
		agent.Responding{TurnID: "turn-1", Prompt: "first"},
		agent.Responding{TurnID: "turn-2", Prompt: "second"},
		agent.Error{Context: "llm", Err: errors.New("upstream is down")},
	)

	// An error names which part failed rather than which turn, so both open turns fail:
	// one left open would read back forever as a question nobody answered.
	s.Len(s.held.finished, 2)
	for _, turn := range s.held.finished {
		s.Equal(store.ResponseFailed, turn.status)
		s.Equal("llm: upstream is down", turn.failure)
	}
}

func (s *RecordSuite) TestATurnAbandonedWhenTheSessionEndedIsClosed() {
	s.took(agent.Responding{TurnID: "turn-1", Prompt: "mid-sentence"})
	s.session.abandonTurns()

	s.Require().Len(s.held.finished, 1)
	s.Equal(store.ResponseCancelled, s.held.finished[0].status)
}

func (s *RecordSuite) TestAnItemForATurnNobodyAnnouncedIsDropped() {
	// Inventing a response for it would leave a row with no question on it, which reads
	// back as a turn nobody asked for.
	s.took(agent.ToolRan{TurnID: "unknown", Tool: "cart", Result: "nothing"})

	s.Empty(s.held.responses)
	s.Empty(s.held.items)
}

func (s *RecordSuite) TestASessionWithNoRecorderWritesNothing() {
	// This is what incognito is: the manager hands over no recorder, so there is nowhere to
	// write rather than a flag every writer has to remember to check.
	quiet := &Session{id: "session-1", spec: Spec{CustomerID: "acme"}}

	quiet.record(agent.Responding{TurnID: "turn-1", Prompt: "off the record"})
	quiet.record(agent.Responded{TurnID: "turn-1", Text: "understood"})
	quiet.abandonTurns()

	s.Empty(s.held.responses)
}

func (s *RecordSuite) TestTheRowSaysWhoseSessionItIsRatherThanWhoTheAgentJoinedAs() {
	created := &Session{
		id:      "session-1",
		created: time.Now(),
		spec: Spec{
			CustomerID: "acme",
			ConfigID:   "cfg-docs",
			AgentName:  "docs",
			// UserID is who the agent joins the call as, which is not whose session it is.
			UserID:     "agent",
			Caller:     routing.Caller{UserID: "jlahey"},
			CallerKind: auth.KindAuthenticated,
			Project:    "Health",
			Title:      "A refund",
			Custom:     map[string]any{"tenant": "acme"},
		},
	}

	row := sessionRow(created)

	s.Equal("jlahey", row.UserID, "recording the agent's own id would make every session the agent's")
	s.Equal(string(auth.KindAuthenticated), row.CallerKind)
	s.Equal("docs", row.AgentName)
	s.Equal("Health", row.Project)
	s.Equal("A refund", row.Title)
	s.Equal(store.SessionRunning, row.State)
}

func (s *RecordSuite) TestMatchesLiveAnswersTheSameFiltersAsTheStore() {
	live := &Session{
		created: time.Now(),
		spec: Spec{
			Caller:    routing.Caller{UserID: "jlahey"},
			ConfigID:  "cfg-docs",
			AgentName: "docs",
			Project:   "Health",
			Custom:    map[string]any{"tenant": "acme", "seats": 4},
		},
	}

	s.True(matchesLive(live, store.SessionFilter{}))
	s.True(matchesLive(live, store.SessionFilter{UserID: "jlahey", AgentName: "docs"}))
	s.True(matchesLive(live, store.SessionFilter{Custom: map[string]string{"tenant": "acme"}}))
	// Numbers survive the round trip through the query string they arrived as.
	s.True(matchesLive(live, store.SessionFilter{Custom: map[string]string{"seats": "4"}}))

	s.False(matchesLive(live, store.SessionFilter{UserID: "randy"}))
	s.False(matchesLive(live, store.SessionFilter{Project: "Docs"}))
	s.False(matchesLive(live, store.SessionFilter{Custom: map[string]string{"tenant": "other"}}))
	// Every session this process holds is running, so asking for the closed ones excludes
	// all of them rather than none.
	s.False(matchesLive(live, store.SessionFilter{State: store.SessionClosed}))
	s.True(matchesLive(live, store.SessionFilter{State: store.SessionRunning}))
}
