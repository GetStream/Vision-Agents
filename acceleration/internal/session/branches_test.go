//go:build integration

package session

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"testing"
	"time"

	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	persistent "github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llm/llmtest"
	"github.com/GetStream/Vision-Agents/acceleration/internal/llmrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/metered"
	"github.com/GetStream/Vision-Agents/acceleration/internal/options"
	"github.com/GetStream/Vision-Agents/acceleration/internal/routing"
	"github.com/GetStream/Vision-Agents/acceleration/internal/store"
	"github.com/GetStream/Vision-Agents/acceleration/internal/streamapp"
	"github.com/GetStream/Vision-Agents/acceleration/internal/stt"
	"github.com/GetStream/Vision-Agents/acceleration/internal/sttrouter"
	"github.com/GetStream/Vision-Agents/acceleration/internal/testenv"
	"github.com/GetStream/Vision-Agents/acceleration/internal/tts"
	"github.com/GetStream/Vision-Agents/acceleration/internal/ttsrouter"
)

// answeringLLM answers every question with a numbered answer to it, so a question answered
// again reads differently from its first answer.
type answeringLLM struct {
	mu       sync.Mutex
	answered int
}

func (a *answeringLLM) Start(context.Context) error { return nil }

func (a *answeringLLM) Create(_ context.Context, params llm.ResponseParams) (*llm.Stream, error) {
	asked := ""
	for _, message := range params.Input {
		if message.Role == llm.User {
			asked = message.Content
		}
	}
	a.mu.Lock()
	a.answered++
	reply := fmt.Sprintf("Answer %d to %s", a.answered, asked)
	a.mu.Unlock()
	script := llmtest.New(llm.StreamOptions{ResponseID: params.ID, Provider: a.Provider(), Model: a.Model()})
	script.OutputText(reply)
	script.Done()
	return script.Stream(), nil
}

func (a *answeringLLM) Provider() string               { return "stub" }
func (a *answeringLLM) Model() string                  { return "stub-llm" }
func (a *answeringLLM) Capabilities() llm.Capabilities { return llm.Capabilities{} }
func (a *answeringLLM) Close() error                   { return nil }

// BranchesSuite regenerates, edits and brings back the turns of conversations kept in Chat,
// in memory, with their responses in Postgres.
type BranchesSuite struct {
	suite.Suite
	ctx      context.Context
	store    *store.Store
	customer string
	chat     *chattest.Server
	service  *persistent.Service
	manager  *Manager
	memory   *stubMemory
}

func TestBranchesSuite(t *testing.T) {
	suite.Run(t, new(BranchesSuite))
}

func (s *BranchesSuite) SetupSuite() {
	dsn := os.Getenv("ROUTER_POSTGRES_DSN")
	if dsn == "" {
		s.T().Skip("ROUTER_POSTGRES_DSN is not set")
	}
	s.ctx = context.Background()
	db, err := store.Open(testenv.Database(dsn, "session_branches"))
	s.Require().NoError(err)
	s.Require().NoError(db.Migrate(s.ctx))
	s.store = db
	s.T().Cleanup(func() { s.Require().NoError(db.Close()) })
}

func (s *BranchesSuite) SetupTest() {
	s.customer = "branches-" + uuid.NewString()
	s.chat = chattest.NewServer(s.T())
	s.service = persistent.NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.memory = &stubMemory{}

	logger := slog.New(slog.DiscardHandler)
	transcription := sttrouter.NewRegistry()
	transcription.Register("stub", func(routing.Spec) (stt.STT, error) { return &stubSTT{emitter: stt.NewEmitter(64)}, nil })
	transcriber, err := sttrouter.New(sttrouter.Options{Config: stubConfig(), Registry: transcription, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(transcriber.Close)
	// The first model a session opens holds its conversation; the others are its flow
	// controller and the titler.
	var opened sync.Mutex
	conversing := map[string]bool{}
	reasoning := llmrouter.NewRegistry()
	reasoning.Register("stub", func(routing.Spec) (llmrouter.Provider, error) {
		opened.Lock()
		defer opened.Unlock()
		if !conversing["first"] {
			conversing["first"] = true
			return &answeringLLM{}, nil
		}
		return &stubLLM{reply: "Sure."}, nil
	})
	reasoner, err := llmrouter.New(llmrouter.Options{Config: stubConfig(), Registry: reasoning, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(reasoner.Close)
	speech := ttsrouter.NewRegistry()
	speech.Register("stub", func(routing.Spec) (tts.TTS, error) { return &stubTTS{emitter: tts.NewEmitter(64)}, nil })
	speaker, err := ttsrouter.New(ttsrouter.Options{Config: stubConfig(), Registry: speech, Logger: logger})
	s.Require().NoError(err)
	s.T().Cleanup(speaker.Close)

	m, err := NewManager(ManagerOptions{
		LLM: reasoner, STT: transcriber, TTS: speaker, Store: s.store, Logger: logger,
		Conversations: s.service, Memory: metered.NewMemory(s.memory, nil),
		Edge: func(context.Context, Spec, streamapp.Bound, *slog.Logger) (agent.Edge, error) {
			return attendedEdge{newQuietEdge()}, nil
		},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { s.Require().NoError(m.Shutdown()) })
	s.manager = m
}

// opens a text session kept in Chat for employee-1.
func (s *BranchesSuite) opens() *Session {
	created, err := s.manager.Create(s.ctx, Spec{
		CustomerID: s.customer, Text: true, PersistConversation: true,
		LLMTarget: "en-low-latency", Caller: routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = created.Close() })
	return created
}

// asks text of the session under a new request id, waits for the answer and returns the
// response it was recorded as.
func (s *BranchesSuite) asks(session *Session, text string) string {
	_, responseID, err := session.RespondCommand(s.ctx, uuid.NewString(), text, "", nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().NotEmpty(responseID)
	s.settled(session)
	return responseID
}

// settled waits until the session's reply is answered, in Chat and in the store.
func (s *BranchesSuite) settled(session *Session) {
	s.Require().Eventually(func() bool {
		last := s.visible(session)
		return len(last) > 0 && last[len(last)-1].text != "" && last[len(last)-1].state == "completed" && !s.service.Writing()
	}, 5*time.Second, 5*time.Millisecond, "the reply was never written")
	s.Require().NoError(s.manager.records.Flush(s.ctx))
}

// onScreen is a message the channel shows.
type onScreen struct {
	id, text, role, state string
	custom                map[string]any
}

// visible are the messages a read of the session's channel returns, oldest first.
func (s *BranchesSuite) visible(session *Session) []onScreen {
	ids := map[string]bool{}
	for _, id := range s.chat.Visible(session.ID()) {
		ids[id] = true
	}
	var out []onScreen
	for _, stored := range s.chat.Stored(session.ID()) {
		if !ids[stored["id"].(string)] {
			continue
		}
		custom, _ := stored["custom"].(map[string]any)
		runtime, _ := custom["support_runtime"].(map[string]any)
		role, _ := runtime["role"].(string)
		state, _ := runtime["state"].(string)
		text, _ := stored["text"].(string)
		out = append(out, onScreen{id: stored["id"].(string), text: text, role: role, state: state, custom: custom})
	}
	return out
}

// texts are what the session's channel shows, oldest first.
func (s *BranchesSuite) texts(session *Session) []string {
	var out []string
	for _, m := range s.visible(session) {
		out = append(out, m.text)
	}
	return out
}

// history is what the session's model is handed of the conversation: its questions and
// answers, oldest first.
func history(session *Session) []string {
	var out []string
	for _, message := range session.current().History() {
		if (message.Role == llm.User || message.Role == llm.Assistant) && message.Content != "" {
			out = append(out, message.Content)
		}
	}
	return out
}

// response is a session's response as the store keeps it.
func (s *BranchesSuite) response(session *Session, id string) store.AgentResponse {
	response, err := s.store.SessionResponse(s.ctx, s.customer, session.ID(), id)
	s.Require().NoError(err)
	return response
}

// mark is the place a onScreen message carries among its alternatives.
func mark(m onScreen) map[string]any {
	b, _ := m.custom[persistent.BranchField].(map[string]any)
	return b
}

func (s *BranchesSuite) TestEveryTurnIsAnsweredAfterTheOneBeforeAndCarriesItsResponse() {
	session := s.opens()

	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")

	s.Empty(s.response(session, first).ParentResponseID)
	s.Equal(first, s.response(session, first).QuestionResponseID)
	s.Equal(first, s.response(session, second).ParentResponseID)
	s.Equal(second, s.response(session, second).QuestionResponseID)
	onScreen := s.visible(session)
	s.Require().Len(onScreen, 4)
	for i, want := range []string{first, first, second, second} {
		s.Equal(want, mark(onScreen[i])["response_id"], "message %d", i)
		s.Nil(mark(onScreen[i])["of"], "a turn with no alternatives is 1 of 1")
	}
	s.Equal(first, mark(onScreen[2])["parent_response_id"])
}

func (s *BranchesSuite) TestARegenerateAnswersTheQuestionAgainFromTheConversationBeforeIt() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")

	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)

	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 3 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session), "the model holds the conversation Chat shows")
	s.Require().NotEmpty(branched.ResponseID)
	regenerated := s.response(session, branched.ResponseID)
	s.Equal(first, regenerated.ParentResponseID)
	s.Equal(second, regenerated.QuestionResponseID, "a regenerate answers the replaced response's question")
	s.NotNil(s.response(session, second).RewoundAt)
	s.Equal(store.SiblingsOfReply, branched.Of)
	s.Equal(2, branched.Index)
	s.Equal(2, branched.Count)
	s.True(branched.Rewound)
	s.False(branched.RewoundTo.Inclusive)
	onScreen := s.visible(session)
	s.Equal(branched.RewoundTo.MessageID, onScreen[2].id, "rewound to the question, which stays")
	s.Equal(map[string]any{"response_id": branched.ResponseID, "parent_response_id": first, "question_response_id": second,
		"of": "reply", "index": float64(2), "count": float64(2)}, mark(onScreen[3]))

	third := s.asks(session, "And Sunday?")
	s.Equal(branched.ResponseID, s.response(session, third).ParentResponseID, "the next turn follows the regenerated one")
}

func (s *BranchesSuite) TestAnEditAsksTheNewQuestionFromTheConversationBeforeIt() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")

	branched, err := session.Edit(s.ctx, uuid.NewString(), second, "And on Sunday?", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)

	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And on Sunday?", "Answer 3 to And on Sunday?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	edited := s.response(session, branched.ResponseID)
	s.Equal(first, edited.ParentResponseID)
	s.Equal(branched.ResponseID, edited.QuestionResponseID, "an edit asks a question of its own")
	s.Equal(store.SiblingsOfQuestion, branched.Of)
	s.Equal(2, branched.Index)
	s.Equal(2, branched.Count)
	s.True(branched.RewoundTo.Inclusive)
	onScreen := s.visible(session)
	s.Equal(map[string]any{"response_id": branched.ResponseID, "parent_response_id": first, "question_response_id": branched.ResponseID,
		"of": "question", "index": float64(2), "count": float64(2)}, mark(onScreen[2]))
	s.Nil(mark(onScreen[3])["of"], "the edited question has one answer")
}

// A regenerated reply brought back is sent again as 1 of 2, not as the 1 of 1 it was hidden
// as, and the model carries on from it.
func (s *BranchesSuite) TestActivatingARegeneratedReplySendsItAgainWithItsPlaceNow() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)

	activated, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)

	s.Equal(second, activated.LeafResponseID)
	s.Equal(1, activated.Resent)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	onScreen := s.visible(session)
	s.Equal(map[string]any{"response_id": second, "parent_response_id": first, "question_response_id": second,
		"of": "reply", "index": float64(1), "count": float64(2)}, mark(onScreen[3]))
	s.NotEmpty(onScreen[3].custom[persistent.ResentFromField])
	s.Nil(s.response(session, second).RewoundAt)
	s.NotNil(s.response(session, branched.ResponseID).RewoundAt)

	again, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.True(again.AlreadyActive, "a branch on the conversation is left as it is")

	third := s.asks(session, "And Sunday?")
	s.Equal(second, s.response(session, third).ParentResponseID)
}

// A branch brought back, hidden again and brought back once more has two copies of its reply,
// the original and the one sent again: it is sent once.
func (s *BranchesSuite) TestABranchBroughtBackTwiceIsSentOnce() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	written := func() {
		s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
	}
	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	written()
	_, err = session.Activate(s.ctx, branched.ResponseID)
	s.Require().NoError(err)
	written()

	activated, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	written()

	s.Equal(1, activated.Resent)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
}

func (s *BranchesSuite) TestActivatingAnEditedQuestionSendsBothMessagesAgainWithTheirPlaceNow() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	_, err := session.Edit(s.ctx, uuid.NewString(), second, "And on Sunday?", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)

	activated, err := session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)

	s.Equal(2, activated.Resent)
	s.True(activated.RewoundTo.Inclusive)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	onScreen := s.visible(session)
	s.Equal(map[string]any{"response_id": second, "parent_response_id": first, "question_response_id": second,
		"of": "question", "index": float64(1), "count": float64(2)}, mark(onScreen[2]))
	s.Equal(map[string]any{"response_id": second, "parent_response_id": first, "question_response_id": second}, mark(onScreen[3]))
}

// Bringing back a response below the first one an edit hid brings back the branch from where
// it left the conversation, not from that response.
func (s *BranchesSuite) TestActivatingAResponseDeepInAHiddenBranchBringsBackTheWholeBranch() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	third := s.asks(session, "And Sunday?")
	_, err := session.Edit(s.ctx, uuid.NewString(), second, "And on Monday?", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)

	activated, err := session.Activate(s.ctx, third)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)

	s.Equal(third, activated.LeafResponseID)
	s.Equal(4, activated.Resent)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 2 to And tomorrow?",
		"And Sunday?", "Answer 3 to And Sunday?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	s.Nil(s.response(session, second).RewoundAt)
	s.Nil(s.response(session, third).RewoundAt)
}

func (s *BranchesSuite) TestARegenerateOnACallIsRefused() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	s.Require().NoError(session.StartVoice(s.ctx, Call{}))

	_, err := session.Regenerate(s.ctx, uuid.NewString(), first, options.LLM{})

	refused, ok := errors.AsType[*persistent.BranchesUnsupportedError](err)
	s.Require().True(ok, "%v", err)
	s.Equal(persistent.UnsupportedCall, refused.Reason)
}

func (s *BranchesSuite) TestARegenerateOfAConversationNotKeptInChatIsRefused() {
	created, err := s.manager.Create(s.ctx, Spec{CustomerID: s.customer, Text: true, LLMTarget: "en-low-latency"})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = created.Close() })
	responseID, err := created.Respond(s.ctx, "Is it raining?", nil, options.LLM{})
	s.Require().NoError(err)

	_, err = created.Regenerate(s.ctx, uuid.NewString(), responseID, options.LLM{})

	refused, ok := errors.AsType[*persistent.BranchesUnsupportedError](err)
	s.Require().True(ok, "%v", err)
	s.Equal(persistent.UnsupportedNotKept, refused.Reason)
}

func (s *BranchesSuite) TestARegenerateChatHasNoRoomForIsRefusedAndTheConversationCarriesOn() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.chat.CapRewinds(1)
	_, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	third := s.asks(session, "And Sunday?")

	// Answering the third question again would leave a second range, apart from the first.
	_, err = session.Regenerate(s.ctx, uuid.NewString(), third, options.LLM{})

	s.Require().ErrorIs(err, persistent.ErrBranchLimit)
	s.asks(session, "And Monday?")
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 3 to And tomorrow?",
		"And Sunday?", "Answer 4 to And Sunday?", "And Monday?", "Answer 5 to And Monday?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
}

func (s *BranchesSuite) TestTruncatingAUsersMemoriesDeletesTheCopiesOfTheirBranches() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	_, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	copies, err := s.store.BranchMessages(s.ctx, s.customer, session.ID(), nil)
	s.Require().NoError(err)
	s.Require().NotEmpty(copies)

	s.Require().NoError(s.manager.TruncateMemories(s.ctx, s.customer, "employee-1"))

	copies, err = s.store.BranchMessages(s.ctx, s.customer, session.ID(), nil)
	s.Require().NoError(err)
	s.Empty(copies)
}

// A reopened conversation carries on from the branch it was left on, not from its newest
// response.
func (s *BranchesSuite) TestAReopenedSessionCarriesOnFromTheBranchItWasLeftOn() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	_, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
	spec := session.Spec()
	s.Require().NoError(session.Close())

	reopened, err := s.manager.Create(s.ctx, Spec{
		ID: spec.ID, CustomerID: s.customer, Text: true, PersistConversation: true,
		ConversationID: spec.ConversationID, Reopened: session.CreatedAt(), LLMTarget: "en-low-latency",
		Caller: routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = reopened.Close() })
	third := s.asks(reopened, "And Sunday?")

	s.Equal(second, s.response(reopened, third).ParentResponseID)
	s.Equal(s.texts(reopened), history(reopened))
}

// asksWithDocument asks text of the session with a file, as the API sends one: pinned for the
// turn, and kept under the response it was sent with.
func (s *BranchesSuite) asksWithDocument(session *Session, text, name string) string {
	document := agent.Document{ID: store.NewID(), Name: name, Pages: 1, Text: "the text of " + name}
	responseID := s.asksIn(agent.WithDocument(s.ctx, document), session, text)
	spec := session.Spec()
	s.Require().NoError(s.store.AddConversationDocument(s.ctx, &store.ConversationDocument{
		ID: document.ID, CustomerID: s.customer, OwnerUserID: spec.Caller.UserID, SessionID: session.ID(),
		ConversationID: spec.ConversationID, ResponseID: responseID, Name: name, Pages: 1, SHA256: "digest", Markdown: document.Text,
	}))
	return responseID
}

// asksIn is asks on ctx.
func (s *BranchesSuite) asksIn(ctx context.Context, session *Session, text string) string {
	_, responseID, err := session.RespondCommand(ctx, uuid.NewString(), text, "", nil, options.LLM{})
	s.Require().NoError(err)
	s.Require().NotEmpty(responseID)
	s.settled(session)
	return responseID
}

// pinned are the names of the files pinned in the session's conversation, oldest first.
func pinned(session *Session) []string {
	var names []string
	for _, document := range session.Documents() {
		names = append(names, document.Name)
	}
	return names
}

// reopens the session's conversation after closing it, as employee-1 carrying on later.
func (s *BranchesSuite) reopens(session *Session) *Session {
	spec := session.Spec()
	s.Require().NoError(session.Close())
	reopened, err := s.manager.Create(s.ctx, Spec{
		ID: spec.ID, CustomerID: s.customer, Text: true, PersistConversation: true,
		ConversationID: spec.ConversationID, Reopened: session.CreatedAt(), LLMTarget: "en-low-latency",
		Caller: routing.Caller{UserID: "employee-1"},
	})
	s.Require().NoError(err)
	s.T().Cleanup(func() { _ = reopened.Close() })
	return reopened
}

// The question a regenerate answers again keeps the file sent with it, now and once the
// conversation is reopened, though the response it was sent with is off the conversation.
func (s *BranchesSuite) TestARegeneratedQuestionKeepsItsDocument() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asksWithDocument(session, "What does the lease say?", "lease.pdf")
	s.Require().Equal([]string{"lease.pdf"}, pinned(session))

	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	s.Require().NotEmpty(branched.ResponseID)

	s.Equal([]string{"lease.pdf"}, pinned(session))
	reopened := s.reopens(session)
	s.Equal([]string{"lease.pdf"}, pinned(reopened), "a reopen reads the regenerated turn's copy")
}

// An edit replaces the question and the file sent with it, which stays with the branch it
// left and comes back with it.
func (s *BranchesSuite) TestAnEditedQuestionsDocumentStaysWithItsBranch() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asksWithDocument(session, "What does the lease say?", "lease.pdf")

	_, err := session.Edit(s.ctx, uuid.NewString(), second, "And the deposit?", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	s.Empty(pinned(session), "the edited question came with no file")

	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
	s.Equal([]string{"lease.pdf"}, pinned(session))
	reopened := s.reopens(session)
	s.Equal([]string{"lease.pdf"}, pinned(reopened))
}

// Activating a branch pins the files its turns were sent with, and activating away from it
// takes them back.
func (s *BranchesSuite) TestActivatingABranchPinsTheDocumentsOfItsTurns() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	edited, err := session.Edit(s.ctx, uuid.NewString(), second, "And on Sunday?", "", options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	third := s.asksWithDocument(session, "What does the forecast say?", "forecast.pdf")
	written := func() {
		s.Require().Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
	}

	_, err = session.Activate(s.ctx, second)
	s.Require().NoError(err)
	written()
	s.Empty(pinned(session), "the branch brought back was sent no file")

	activated, err := session.Activate(s.ctx, edited.ResponseID)
	s.Require().NoError(err)
	written()
	s.Equal(third, activated.LeafResponseID)
	s.Equal([]string{"forecast.pdf"}, pinned(session))
}

// A regenerate whose rewind Chat made but whose answer was lost finds out from Chat that it
// landed and carries on: the model and the next question follow what the person sees.
func (s *BranchesSuite) TestARegenerateWhoseRewindAnswerWasLostCarriesOnFromWhatChatShows() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	s.chat.LoseRewindAnswer()

	branched, err := session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})

	s.Require().NoError(err)
	s.settled(session)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 3 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	third := s.asks(session, "And Sunday?")
	s.Equal(branched.ResponseID, s.response(session, third).ParentResponseID)
}

// A switch that cannot be settled once Chat may have rewound refuses every other turn, which
// would be answered from a conversation nobody knows, until its own request is retried.
func (s *BranchesSuite) TestASwitchThatCannotBeSettledRefusesOtherTurnsUntilItsRequestIsRetried() {
	session := s.opens()
	s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	request := uuid.NewString()
	s.chat.LoseRewindAnswer()
	waiting, release := s.chat.Hold("/rewind")
	failed := make(chan error, 1)
	go func() {
		_, err := session.Regenerate(s.ctx, request, second, options.LLM{})
		failed <- err
	}()
	<-waiting
	// Chat cannot be read once it rewound, so whether it did is not known.
	s.chat.Fail("POST", "/query")
	release()
	s.Require().ErrorIs(<-failed, ErrSwitchUnfinished)

	_, _, err := session.RespondCommand(s.ctx, uuid.NewString(), "And Sunday?", "", nil, options.LLM{})
	s.Require().ErrorIs(err, ErrSwitchUnfinished)
	_, err = session.Regenerate(s.ctx, uuid.NewString(), second, options.LLM{})
	s.Require().ErrorIs(err, ErrSwitchUnfinished)

	s.chat.Heal("POST", "/query")
	branched, err := session.Regenerate(s.ctx, request, second, options.LLM{})
	s.Require().NoError(err)
	s.settled(session)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And tomorrow?", "Answer 3 to And tomorrow?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	third := s.asks(session, "And Sunday?")
	s.Equal(branched.ResponseID, s.response(session, third).ParentResponseID)
}

// An edit retried after Chat hid its question carries on the switch it began, though the
// question it replaces is no longer in the channel to be found.
func (s *BranchesSuite) TestAnEditRetriedAfterChatHidItsQuestionCarriesOnItsSwitch() {
	session := s.opens()
	first := s.asks(session, "Is it raining?")
	second := s.asks(session, "And tomorrow?")
	request := uuid.NewString()
	s.chat.LoseRewindAnswer()
	waiting, release := s.chat.Hold("/rewind")
	failed := make(chan error, 1)
	go func() {
		_, err := session.Edit(s.ctx, request, second, "And on Sunday?", "", options.LLM{})
		failed <- err
	}()
	<-waiting
	s.chat.Fail("POST", "/query")
	release()
	s.Require().ErrorIs(<-failed, ErrSwitchUnfinished)
	s.chat.Heal("POST", "/query")

	branched, err := session.Edit(s.ctx, request, second, "And on Sunday?", "", options.LLM{})

	s.Require().NoError(err)
	s.settled(session)
	s.Equal([]string{"Is it raining?", "Answer 1 to Is it raining?", "And on Sunday?", "Answer 3 to And on Sunday?"}, s.texts(session))
	s.Equal(s.texts(session), history(session))
	s.Equal(first, s.response(session, branched.ResponseID).ParentResponseID)
}
