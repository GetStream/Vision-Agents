package conversation_test

import (
	"context"
	"testing"
	"time"

	getstream "github.com/GetStream/getstream-go/v5"
	"github.com/google/uuid"
	"github.com/stretchr/testify/suite"

	"github.com/GetStream/Vision-Agents/acceleration/internal/agent"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation"
	"github.com/GetStream/Vision-Agents/acceleration/internal/conversation/chattest"
)

// BranchesSuite is a session's own conversation, kept in Chat in memory, whose turns are
// answered again, asked again in other words and brought back.
type BranchesSuite struct {
	suite.Suite
	chat    *chattest.Server
	service *conversation.Service
	ctx     context.Context
	// session is the session's id, which its channel is named after, and cid the channel's.
	session, cid string
	conv         *conversation.Conversation
}

func TestBranchesSuite(t *testing.T) {
	suite.Run(t, new(BranchesSuite))
}

func (s *BranchesSuite) SetupTest() {
	s.chat = chattest.NewServer(s.T())
	s.service = conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(s.service.Close)
	s.ctx = context.Background()
	s.session = uuid.NewString()
	s.cid = "agent:" + s.session
	conv, _, _, err := s.service.CreateInApp(s.ctx, 0, "customer", "agent", s.cid, "", "", nil, "")
	s.Require().NoError(err)
	s.conv = conv
}

// written waits until every write the service holds is in Chat.
func (s *BranchesSuite) written() {
	s.Eventually(func() bool { return !s.service.Writing() }, 5*time.Second, 5*time.Millisecond)
}

// custom is the custom data Chat holds for message id.
func (s *BranchesSuite) custom(id string) map[string]any {
	for _, stored := range s.chat.Stored(s.session) {
		if stored["id"] == id {
			custom, _ := stored["custom"].(map[string]any)
			return custom
		}
	}
	s.FailNow("no such message in Chat", id)
	return nil
}

// answered asks text under the command id in turn, has the agent answer it with reply, and
// waits for both messages to be in Chat.
func (s *BranchesSuite) answered(turn conversation.Turn, id, text, reply string) conversation.CommandReceipt {
	receipt, err := s.conv.BeginCommandWithTurn(turn, id, text, "", nil)
	s.Require().NoError(err)
	s.replied(id, reply)
	return receipt
}

// replied has the agent answer the command id with reply, and waits for it to be in Chat.
func (s *BranchesSuite) replied(id, reply string) {
	turnID := "turn-" + id
	s.conv.BindTurn(id, turnID)
	s.conv.Observe(agent.Responding{TurnID: turnID})
	s.conv.Observe(agent.ResponseDelta{TurnID: turnID, Text: reply})
	s.conv.Observe(agent.Responded{TurnID: turnID})
	s.written()
}

func (s *BranchesSuite) TestBothMessagesOfATurnCarryItsResponseFromTheirCreation() {
	turn := conversation.Turn{ResponseID: "resp-2", ParentResponseID: "resp-1", QuestionResponseID: "resp-2",
		Question: conversation.Place{Index: 2, Count: 2}}

	receipt := s.answered(turn, "ask-2", "what about Tuesday?", "Tuesday works.")

	s.Equal(map[string]any{"response_id": "resp-2", "parent_response_id": "resp-1", "question_response_id": "resp-2",
		"of": "question", "index": float64(2), "count": float64(2)}, s.custom(receipt.UserMessageID)[conversation.BranchField])
	s.Equal(map[string]any{"response_id": "resp-2", "parent_response_id": "resp-1", "question_response_id": "resp-2"},
		s.custom(receipt.AssistantMessageID)[conversation.BranchField], "a reply with no alternatives is 1 of 1, which is not written")
}

func (s *BranchesSuite) TestACommandWithoutATurnIsMarkedAsBefore() {
	receipt, err := s.conv.BeginCommand("ask-1", "hello", "")
	s.Require().NoError(err)
	s.written()

	s.NotContains(s.custom(receipt.UserMessageID), conversation.BranchField)
	s.NotContains(s.custom(receipt.AssistantMessageID), conversation.BranchField)
}

func (s *BranchesSuite) TestAMarkIsReadBack() {
	turn := conversation.Turn{ResponseID: "resp-1", QuestionResponseID: "resp-1", Reply: conversation.Place{Index: 1, Count: 3}}
	receipt := s.answered(turn, "ask-1", "hello", "Hi.")

	page, err := s.service.History(s.ctx, "customer", "agent", s.cid, "")

	s.Require().NoError(err)
	s.Require().Len(page.Messages, 2)
	s.Equal(receipt.AssistantMessageID, page.Messages[1].ID)
	s.Equal(&conversation.Branch{ResponseID: "resp-1", QuestionResponseID: "resp-1", Of: conversation.BranchOfReply, Index: 1, Count: 3},
		page.Messages[1].Branch)
}

// A router of another version may write a mark this one cannot read; the message is read all
// the same, without one.
func (s *BranchesSuite) TestAMarkThisRouterCannotReadLeavesTheMessageReadable() {
	receipt := s.answered(conversation.Turn{}, "ask-1", "hello", "Hi.")
	for _, mark := range []any{
		map[string]any{"response_id": "resp-1", "of": "sideways", "index": 1, "count": 2, "added_later": true},
		"not an object",
		map[string]any{"of": "reply"},
	} {
		user := "agent"
		_, err := s.chat.Client.Chat().UpdateMessagePartial(s.ctx, receipt.AssistantMessageID, &getstream.UpdateMessagePartialRequest{
			UserID: &user, Set: map[string]any{conversation.BranchField: mark}})
		s.Require().NoError(err)

		page, err := s.service.History(s.ctx, "customer", "agent", s.cid, "")

		s.Require().NoError(err)
		s.Require().Len(page.Messages, 2, mark)
		s.Equal("Hi.", page.Messages[1].Text, mark)
	}
	page, err := s.service.History(s.ctx, "customer", "agent", s.cid, "")
	s.Require().NoError(err)
	s.Nil(page.Messages[1].Branch)
}

func (s *BranchesSuite) TestATurnWithAnInvalidResponseIDIsRefused() {
	_, err := s.conv.BeginCommandWithTurn(conversation.Turn{ResponseID: "resp 1"}, "ask-1", "hello", "", nil)

	s.Error(err)
}

func (s *BranchesSuite) TestARegenerateWritesOnlyANewReplyToTheQuestion() {
	asked := s.answered(conversation.Turn{}, "ask-1", "hello", "Hi.")

	receipt, err := s.conv.BeginRegenerate(conversation.Turn{ResponseID: "resp-2", QuestionResponseID: "resp-1",
		Reply: conversation.Place{Index: 2, Count: 2}}, "again-1", asked.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-1", "Hello there.")

	s.Equal(asked.UserMessageID, receipt.UserMessageID)
	s.Equal([]string{"hello", "Hi.", "Hello there."}, s.chat.Messages(s.session))
	runtime, _ := s.custom(receipt.AssistantMessageID)["support_runtime"].(map[string]any)
	s.Equal(asked.UserMessageID, runtime["question_id"])
	s.Equal("again-1", runtime["command_id"])
	s.Equal(map[string]any{"response_id": "resp-2", "question_response_id": "resp-1", "of": "reply", "index": float64(2), "count": float64(2)},
		s.custom(receipt.AssistantMessageID)[conversation.BranchField])
}

func (s *BranchesSuite) TestARegenerateRetriedIsTheSameCommandAndAnotherQuestionUnderItsIDIsAConflict() {
	asked := s.answered(conversation.Turn{}, "ask-1", "hello", "Hi.")
	other := s.answered(conversation.Turn{}, "ask-2", "and you?", "Fine.")
	turn := conversation.Turn{ResponseID: "resp-3", QuestionResponseID: "resp-1"}
	first, err := s.conv.BeginRegenerate(turn, "again-1", asked.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-1", "Hello there.")

	retried, err := s.conv.BeginRegenerate(turn, "again-1", asked.UserMessageID)
	s.Require().NoError(err)
	s.True(retried.Duplicate)
	s.Equal(first.AssistantMessageID, retried.AssistantMessageID)

	_, err = s.conv.BeginRegenerate(turn, "again-1", other.UserMessageID)
	s.ErrorIs(err, conversation.ErrCommandConflict)
}

// The ledger a reopen rebuilds from Chat knows the regenerate as the asker's command.
func (s *BranchesSuite) TestARegenerateIsTheAskersCommandAfterAReopen() {
	asked := s.answered(conversation.Turn{}, "ask-1", "hello", "Hi.")
	receipt, err := s.conv.BeginRegenerate(conversation.Turn{ResponseID: "resp-2", QuestionResponseID: "resp-1"}, "again-1", asked.UserMessageID)
	s.Require().NoError(err)
	s.replied("again-1", "Hello there.")
	s.conv.Release()
	later := conversation.NewForChat(s.chat.Client)
	s.T().Cleanup(later.Close)

	reopened, _, _, err := later.OpenInApp(s.ctx, 0, "customer", "agent", s.cid, "", "", nil, "")
	s.Require().NoError(err)

	found, err := reopened.Command("again-1")
	s.Require().NoError(err)
	s.Equal(receipt.AssistantMessageID, found.AssistantMessageID)
	s.Equal(asked.UserMessageID, found.UserMessageID)
	s.Equal("completed", found.State)
	retried, err := reopened.BeginRegenerate(conversation.Turn{ResponseID: "resp-2", QuestionResponseID: "resp-1"}, "again-1", asked.UserMessageID)
	s.Require().NoError(err)
	s.True(retried.Duplicate)
}

func (s *BranchesSuite) TestARegenerateWaitsForTheReplyBeingWritten() {
	asked, err := s.conv.BeginCommand("ask-1", "hello", "")
	s.Require().NoError(err)

	_, err = s.conv.BeginRegenerate(conversation.Turn{ResponseID: "resp-2"}, "again-1", asked.UserMessageID)

	s.ErrorIs(err, conversation.ErrResponseRunning)
}
