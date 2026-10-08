package session

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/auth"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
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
	videos    []string
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

func (r *heldRecorder) Described(string, string, string, string, map[string]any) {}

func (r *heldRecorder) Chose(string, string, string, string) {}

func (r *heldRecorder) SawVideo(id string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.videos = append(r.videos, id)
}

func (r *heldRecorder) Flush(context.Context) error { return nil }

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

func (s *RecordSuite) TestAReplyDeliveringAToolResultFinishesTheTurnThatAskedForIt() {
	s.took(
		agent.Responding{TurnID: "text-1", Prompt: "Where is my order?"},
		agent.ToolStarted{TurnID: "text-1", ID: "call-1", Tool: "lookup_order"},
		agent.Responded{TurnID: "text-1", PendingWork: true},
		agent.ToolRan{TurnID: "text-1", ID: "call-1", Tool: "lookup_order", Result: "ships tomorrow"},
		agent.Responding{TurnID: "tool-1", Continues: "text-1"},
		agent.Responded{TurnID: "tool-1", Text: "It ships tomorrow."},
	)

	s.Require().Len(s.held.responses, 1, "the reply after the tool answers the same question")
	s.Require().Len(s.held.finished, 1)
	s.Equal(s.held.responses[0].ID, s.held.finished[0].id)
	s.Equal(store.ResponseCompleted, s.held.finished[0].status)
	s.Equal([]string{store.ItemSaid, store.ItemToolCall, store.ItemAnswer, store.ItemToolResult, store.ItemAnswer},
		s.held.kinds())

	s.session.abandonTurns()
	s.Len(s.held.finished, 1, "a finished turn is not closed again when the session ends")
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
	created := time.Now()
	live := &Session{
		id:       "b",
		created:  created,
		modality: store.ModalityVoice,
		spec: Spec{
			Caller:    routing.Caller{UserID: "jlahey"},
			AgentName: "docs",
			Project:   "Health",
		},
	}

	s.True(matchesLive(live, store.SessionFilter{}))
	s.True(matchesLive(live, store.SessionFilter{UserID: "jlahey", AgentName: "docs", Project: "Health", Modality: store.ModalityVoice}))

	s.False(matchesLive(live, store.SessionFilter{UserID: "randy"}))
	s.False(matchesLive(live, store.SessionFilter{AgentName: "sales"}))
	s.False(matchesLive(live, store.SessionFilter{Project: "Docs"}))
	s.False(matchesLive(live, store.SessionFilter{Modality: store.ModalityVideo}))

	// A session nothing records sorts by when it began, the same as its cursor.
	s.True(matchesLive(live, store.SessionFilter{Cursor: &store.SessionPosition{UpdatedAt: created, ID: "c"}}))
	s.False(matchesLive(live, store.SessionFilter{Cursor: &store.SessionPosition{UpdatedAt: created, ID: "a"}}))
}

func (s *RecordSuite) TestFramesOfTheUsersVideoMakeTheSessionAVideoOneOnce() {
	held := &heldRecorder{}
	live := &Session{id: "b", modality: store.ModalityVoice, records: held}
	frames := framesRunner{parts: []llm.ContentPart{{Image: &llm.ImagePart{URL: "data:image/jpeg;base64,AA=="}}}}
	runner := &videoRunner{next: frames, session: live}

	_, err := runner.Run(s.T().Context(), llm.ToolCall{Name: "lookup_order"})
	s.Require().NoError(err)
	s.Equal(store.ModalityVoice, live.Modality(), "another tool's images are not the user's video")

	for range 2 {
		_, err = runner.Run(s.T().Context(), llm.ToolCall{Name: agent.VideoFramesTool})
		s.Require().NoError(err)
	}
	s.Equal(store.ModalityVideo, live.Modality())
	s.Equal([]string{"b"}, held.videos, "recorded once, however many frames follow")
}

func (s *RecordSuite) TestACaptureThatReturnsNoFramesLeavesTheModalityAlone() {
	live := &Session{id: "b", modality: store.ModalityText}
	runner := &videoRunner{next: framesRunner{parts: llm.TextParts("no video")}, session: live}

	_, err := runner.Run(s.T().Context(), llm.ToolCall{Name: agent.VideoFramesTool})
	s.Require().NoError(err)
	s.Equal(store.ModalityText, live.Modality())
}

// framesRunner is a caller whose every tool answers with the same parts.
type framesRunner struct {
	parts []llm.ContentPart
}

func (r framesRunner) Run(context.Context, llm.ToolCall) ([]llm.ContentPart, error) {
	return r.parts, nil
}
